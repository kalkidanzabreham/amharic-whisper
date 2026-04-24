"""
Fine-tunes MMS (Meta Massively Multilingual Speech) on Amharic.
MMS explicitly supports Amharic — it starts from a much better baseline
than Whisper for this language.

Key differences from train.py (Whisper):
- Uses Wav2Vec2CTCTokenizer instead of WhisperProcessor
- Uses CTC loss instead of Seq2Seq
- Uses Trainer instead of Seq2SeqTrainer
- Different data collator (CTC-specific)
- No generate() — uses argmax decoding

Usage: python train_mms.py
"""

import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import evaluate
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)
from src.utils import validate_dataset, dataset_summary


# ── Load config ────────────────────────────────────────────────────────────────
with open("configs/mms.yaml") as f:
    cfg = yaml.safe_load(f)


# ── Load processor ─────────────────────────────────────────────────────────────
print(f"Loading {cfg['model']['name']}...")

processor = Wav2Vec2Processor.from_pretrained(cfg["model"]["name"])

# Tell MMS which language to use
processor.tokenizer.set_target_lang(cfg["model"]["language"])


# ── Load model ─────────────────────────────────────────────────────────────────
model = Wav2Vec2ForCTC.from_pretrained(
    cfg["model"]["name"],
    target_lang=cfg["model"]["language"],
    ignore_mismatched_sizes=True,  # required when switching target language
)

# Freeze the base encoder — only fine-tune the CTC head
# This is critical for small datasets — prevents catastrophic forgetting
model.freeze_base_model()
print("Base model frozen — only fine-tuning CTC head")


# ── Validate + load data ───────────────────────────────────────────────────────
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

    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_values[0]

    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch[text_col]).input_ids

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


# ── CTC Data Collator ──────────────────────────────────────────────────────────
# CTC needs its own collator — different from Whisper's Seq2Seq collator
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]}           for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )

        # Replace padding with -100 so CTC loss ignores it
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
    pred_logits = pred.predictions
    pred_ids    = np.argmax(pred_logits, axis=-1)  # CTC uses argmax, not generate()

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    pred_str  = [p.strip() for p in pred_str]
    label_str = [l.strip() for l in label_str]

    return {
        "wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4),
        "cer": round(cer_metric.compute(predictions=pred_str, references=label_str), 4),
    }


# ── Training arguments ─────────────────────────────────────────────────────────
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
)


# ── Trainer ────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

print("Starting MMS training...")
trainer.train()

trainer.save_model(cfg["output"]["final_model_dir"])
processor.save_pretrained(cfg["output"]["final_model_dir"])
print(f"✅ MMS model saved → {cfg['output']['final_model_dir']}")