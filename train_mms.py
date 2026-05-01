# ── Patch peft/transformers 5.0 conflict ──────────────────────────────────────
import transformers as _transformers
if not hasattr(_transformers, 'EncoderDecoderCache'):
    _transformers.EncoderDecoderCache = type('EncoderDecoderCache', (), {})

import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import Dict, List, Union

import evaluate
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)
from src.utils import validate_dataset, dataset_summary


with open("configs/mms.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = cfg["model"]["name"]      # "facebook/mms-1b-fl102"
LANGUAGE   = cfg["model"]["language"]  # "amh"
FINAL_DIR  = Path(cfg["output"]["final_model_dir"])
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Load processor the correct way for MMS ────────────────────────────────────
# We download the adapter's own tokenizer files directly
# This guarantees token IDs match exactly what the CTC head was trained with
print(f"Loading MMS processor for '{LANGUAGE}'...")

from huggingface_hub import hf_hub_download
import json

# Download the adapter's tokenizer files
tokenizer_config = hf_hub_download(
    repo_id=MODEL_NAME,
    filename=f"{LANGUAGE}/tokenizer_config.json"
)
vocab_file = hf_hub_download(
    repo_id=MODEL_NAME,
    filename=f"{LANGUAGE}/vocab.json"
)

print(f"Adapter tokenizer config: {tokenizer_config}")
print(f"Adapter vocab file: {vocab_file}")

# Verify vocab content
with open(vocab_file, encoding="utf-8") as f:
    vocab = json.load(f)
print(f"Vocab size: {len(vocab)}")
print(f"First 5 tokens: {list(vocab.items())[:5]}")

# Load feature extractor from base model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

# Load tokenizer from adapter's own files — NOT manually built
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=vocab_file,
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token=" ",   # MMS uses space, not |
    do_lower_case=False,
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

# Save processor to final dir immediately so it's always in sync with model
processor.save_pretrained(str(FINAL_DIR))
print(f"✅ Processor saved to {FINAL_DIR}")
print(f"   Vocab size: {tokenizer.vocab_size}")


# ── Load model + adapter ───────────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME}...")
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    ignore_mismatched_sizes=True,
    ctc_loss_reduction="mean",
    pad_token_id=tokenizer.pad_token_id,
)
model.load_adapter(LANGUAGE)
model.freeze_base_model()
print(f"✅ Adapter loaded | Base frozen | pad_token_id: {tokenizer.pad_token_id}")


# ── Validate data ──────────────────────────────────────────────────────────────
print("\nValidating datasets...")
dataset_summary(cfg["data"]["train_csv"])
dataset_summary(cfg["data"]["val_csv"])

ok  = validate_dataset(cfg["data"]["train_csv"])
ok &= validate_dataset(cfg["data"]["val_csv"])
if not ok:
    raise SystemExit("Fix dataset errors before training.")

train_df = pd.read_csv(cfg["data"]["train_csv"])
val_df   = pd.read_csv(cfg["data"]["val_csv"])

train_dataset = Dataset.from_pandas(train_df).cast_column(
    cfg["data"]["audio_column"], Audio(sampling_rate=cfg["data"]["sample_rate"])
)
val_dataset = Dataset.from_pandas(val_df).cast_column(
    cfg["data"]["audio_column"], Audio(sampling_rate=cfg["data"]["sample_rate"])
)
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")


# ── Preprocessing ──────────────────────────────────────────────────────────────
def prepare_dataset(batch, audio_col, text_col):
    audio = batch[audio_col]

    batch["input_values"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_values[0]

    batch["input_length"] = len(batch["input_values"])

    # Tokenize — MMS uses space as word delimiter so no substitution needed
    batch["labels"] = processor.tokenizer(batch[text_col]).input_ids
    return batch


train_dataset = train_dataset.map(
    partial(prepare_dataset,
            audio_col=cfg["data"]["audio_column"],
            text_col=cfg["data"]["text_column"]),
    remove_columns=train_dataset.column_names,
    num_proc=1
)
val_dataset = val_dataset.map(
    partial(prepare_dataset,
            audio_col=cfg["data"]["audio_column"],
            text_col=cfg["data"]["text_column"]),
    remove_columns=val_dataset.column_names,
    num_proc=1
)


# ── CTC Collator ───────────────────────────────────────────────────────────────
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids":    f["labels"]}       for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)


# ── Metrics ────────────────────────────────────────────────────────────────────
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.tokenizer.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(pred.label_ids, group_tokens=False)

    # Quick sanity print every eval so you can see quality during training
    if pred_str:
        print(f"\n  Sample — actual  : {label_str[0][:50]}")
        print(f"  Sample — predicted: {pred_str[0][:50]}\n")

    return {
        "wer": round(wer_metric.compute(
            predictions=[p.strip() for p in pred_str],
            references=[l.strip() for l in label_str]
        ), 4),
        "cer": round(cer_metric.compute(
            predictions=[p.strip() for p in pred_str],
            references=[l.strip() for l in label_str]
        ), 4),
    }


# ── Training ───────────────────────────────────────────────────────────────────
t = cfg["training"]

training_args = TrainingArguments(
    output_dir=t["output_dir"],
    per_device_train_batch_size=t["batch_size"],
    per_device_eval_batch_size=t["eval_batch_size"],
    max_steps=t["max_steps"],
    warmup_steps=t["warmup_steps"],
    learning_rate=float(t["learning_rate"]),
    fp16=t["fp16"] and torch.cuda.is_available(),
    gradient_checkpointing=t["gradient_checkpointing"],
    eval_strategy="steps",
    eval_steps=t["eval_steps"],
    save_strategy="steps",
    save_steps=t["save_steps"],
    save_total_limit=t["save_total_limit"],
    logging_steps=t["logging_steps"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    dataloader_num_workers=2,
    report_to=["tensorboard"],
    # ✅ These two ensure full model is always saved
    save_safetensors=True,
    save_only_model=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

print("\nStarting MMS training...")
trainer.train()

# Save final model — processor already saved to FINAL_DIR above
# so it's always in sync regardless of training interruptions
trainer.save_model(str(FINAL_DIR))
print(f"✅ MMS model saved → {FINAL_DIR}")