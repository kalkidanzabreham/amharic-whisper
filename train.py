"""
Main training script for Amharic Whisper fine-tuning.
All bugs from ChatGPT version are fixed here.

Usage: python train.py
"""

import yaml
import torch
import pandas as pd
from functools import partial

from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from src.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import compute_metrics
from src.utils import validate_dataset, dataset_summary



# ── Load config ────────────────────────────────────────────────────────────────
with open("configs/amharic.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = cfg["model"]["name"]
LANGUAGE   = cfg["model"]["language"]
TASK       = cfg["model"]["task"]


print("Validating datasets before training...")
dataset_summary(cfg["data"]["train_csv"])
dataset_summary(cfg["data"]["val_csv"])

ok = validate_dataset(cfg["data"]["train_csv"])
ok &= validate_dataset(cfg["data"]["val_csv"])

if not ok:
    raise SystemExit("Fix dataset errors above before training.")


# ── Load processor & model ─────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.generation_config.language = LANGUAGE
model.generation_config.task     = TASK
model.generation_config.forced_decoder_ids = None  # let model decide tokens

# Gradient checkpointing — saves VRAM, essential on Colab T4
if cfg["training"]["gradient_checkpointing"]:
    model.config.use_cache = False   # must disable cache when using grad checkpointing
    model.gradient_checkpointing_enable()


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading datasets...")
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

    # Extract log-mel spectrogram features from raw audio
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    # Tokenize Amharic transcript
    batch["labels"] = processor.tokenizer(batch[text_col]).input_ids

    return batch

audio_col = cfg["data"]["audio_column"]
text_col  = cfg["data"]["text_column"]

train_dataset = train_dataset.map(
    partial(prepare_dataset, audio_col=audio_col, text_col=text_col),
    remove_columns=train_dataset.column_names,
    num_proc=1   # keep at 1 for audio — multiprocessing causes issues with audio
)
val_dataset = val_dataset.map(
    partial(prepare_dataset, audio_col=audio_col, text_col=text_col),
    remove_columns=val_dataset.column_names,
    num_proc=1
)


# ── Data Collator ──────────────────────────────────────────────────────────────
# This is THE piece ChatGPT missed entirely
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# ── Training Arguments ─────────────────────────────────────────────────────────
t = cfg["training"]

training_args = Seq2SeqTrainingArguments(
    output_dir=t["output_dir"],

    # Batch & steps
    per_device_train_batch_size=t["batch_size"],
    per_device_eval_batch_size=t["eval_batch_size"],
    max_steps=t["max_steps"],          # steps > epochs for speech tasks
    warmup_steps=t["warmup_steps"],

    # Optimizer
    learning_rate=t["learning_rate"],

    # Precision & memory
    fp16=t["fp16"] and torch.cuda.is_available(),
    gradient_checkpointing=t["gradient_checkpointing"],

    # Evaluation & saving
    eval_strategy="steps",
    eval_steps=t["eval_steps"],
    save_strategy="steps",
    save_steps=t["save_steps"],
    logging_steps=t["logging_steps"],

    # Generation — required for Seq2Seq evaluation
    predict_with_generate=True,
    generation_max_length=225,    # Whisper's max output length

    # Keep best model
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,      # lower WER = better

    # Performance
    dataloader_num_workers=t["dataloader_num_workers"],

    report_to=["tensorboard"],    # optional — run tensorboard --logdir ./checkpoints
)


# ── Trainer ────────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=partial(compute_metrics, processor=processor),
    processing_class=processor,          # ✅ full processor, NOT processor.feature_extractor
)


# ── Train ──────────────────────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

print("Saving final model...")
trainer.save_model(cfg["output"]["final_model_dir"])
processor.save_pretrained(cfg["output"]["final_model_dir"])
print(f"Done. Model saved → {cfg['output']['final_model_dir']}")