"""
Fine-tunes Wav2Vec2 XLSR-53 on Amharic.

Key difference from MMS: Wav2Vec2 XLSR doesn't natively know Amharic,
so we need to build a custom vocabulary from your training transcripts.

Usage: python train_wav2vec2.py
"""

import yaml
import json
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
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from src.utils import validate_dataset, dataset_summary


with open("configs/wav2vec2.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_DIR   = Path(cfg["data"]["train_csv"]).parent
OUTPUT_DIR = Path(cfg["output"]["final_model_dir"])
VOCAB_PATH = DATA_DIR / "amharic_vocab.json"


# ── Step 1: Build Amharic vocabulary from your transcripts ────────────────────
# Wav2Vec2 needs a character-level vocabulary built from your actual data
# This is the key difference from MMS/Whisper which have built-in vocabularies

def build_vocabulary():
    if VOCAB_PATH.exists():
        print(f"Vocabulary already exists at {VOCAB_PATH}")
        return

    print("Building Amharic character vocabulary from training data...")

    train_df = pd.read_csv(cfg["data"]["train_csv"])
    val_df   = pd.read_csv(cfg["data"]["val_csv"])
    all_text = pd.concat([train_df, val_df])[cfg["data"]["text_column"]]

    # Collect every unique character in your transcripts
    vocab = set()
    for text in all_text.dropna():
        vocab.update(list(str(text)))

    # Remove spaces — handled as word boundary token
    vocab.discard(" ")

    # Build vocab dict with required special tokens
    vocab_dict = {char: i for i, char in enumerate(sorted(vocab))}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    vocab_dict["|"]     = len(vocab_dict)  # word boundary token

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    print(f"✅ Vocabulary: {len(vocab_dict)} characters → {VOCAB_PATH}")
    print(f"   Sample chars: {list(vocab_dict.keys())[:10]}")


build_vocabulary()


# ── Step 2: Load processor with custom Amharic vocab ──────────────────────────
tokenizer = Wav2Vec2CTCTokenizer(
    str(VOCAB_PATH),
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

# Save processor so we can load it later
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
processor.save_pretrained(str(OUTPUT_DIR))


# ── Step 3: Load model with custom vocab size ──────────────────────────────────
print(f"\nLoading {cfg['model']['name']}...")

model = Wav2Vec2ForCTC.from_pretrained(
    cfg["model"]["name"],
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
)

# Freeze feature encoder — only fine-tune transformer layers + CTC head
model.freeze_feature_encoder()
print("Feature encoder frozen")


# ── Step 4: Data loading (same pattern as MMS) ─────────────────────────────────
print("\nValidating datasets...")
dataset_summary(cfg["data"]["train_csv"])
ok  = validate_dataset(cfg["data"]["train_csv"])
ok &= validate_dataset(cfg["data"]["val_csv"])
if not ok:
    raise SystemExit("Fix dataset errors before training.")

def load_csv_as_dataset(csv_path):
    return Dataset.from_pandas(pd.read_csv(csv_path)).cast_column(
        cfg["data"]["audio_column"],
        Audio(sampling_rate=cfg["data"]["sample_rate"])
    )

train_dataset = load_csv_as_dataset(cfg["data"]["train_csv"])
val_dataset   = load_csv_as_dataset(cfg["data"]["val_csv"])


def prepare_dataset(batch, audio_col, text_col):
    audio = batch[audio_col]

    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    # Replace spaces with word boundary token for CTC
    text = batch[text_col].replace(" ", "|")

    with processor.as_target_processor():
        batch["labels"] = processor(text).input_ids

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


# ── Collator + metrics (identical to MMS) ─────────────────────────────────────
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]}           for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)
wer_metric    = evaluate.load("wer")
cer_metric    = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    return {
        "wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4),
        "cer": round(cer_metric.compute(predictions=pred_str, references=label_str), 4),
    }


# ── Train ──────────────────────────────────────────────────────────────────────
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
    report_to=["tensorboard"],
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

print("Starting Wav2Vec2 training...")
trainer.train()
trainer.save_model(cfg["output"]["final_model_dir"])
print(f"✅ Wav2Vec2 model saved → {cfg['output']['final_model_dir']}")