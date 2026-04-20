"""
Evaluates the fine-tuned model on the test set.
Run this AFTER training is complete.

Usage: python evaluate.py
       python evaluate.py --model_dir ./whisper-amharic-final --test_csv data/test.csv
"""

import argparse
import yaml
import pandas as pd
from functools import partial
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from src.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import compute_metrics


def run_evaluation(model_dir: str, test_csv: str, config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading model from {model_dir}...")
    processor = WhisperProcessor.from_pretrained(model_dir)
    model     = WhisperForConditionalGeneration.from_pretrained(model_dir)

    # Load test data
    test_df = pd.read_csv(test_csv)
    test_dataset = Dataset.from_pandas(test_df).cast_column(
        cfg["data"]["audio_column"],
        Audio(sampling_rate=cfg["data"]["sample_rate"])
    )

    audio_col = cfg["data"]["audio_column"]
    text_col  = cfg["data"]["text_column"]

    def prepare(batch):
        audio = batch[audio_col]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch[text_col]).input_ids
        return batch

    test_dataset = test_dataset.map(
        prepare,
        remove_columns=test_dataset.column_names,
        num_proc=1
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Minimal trainer just for evaluation
    eval_args = Seq2SeqTrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=cfg["training"]["eval_batch_size"],
        predict_with_generate=True,
        generation_max_length=225,
        fp16=False,   # safer for eval
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, processor=processor),
        tokenizer=processor,
    )

    print("Running evaluation on test set...")
    results = trainer.evaluate(eval_dataset=test_dataset)

    print("\n── Test Set Results ──────────────────")
    print(f"  WER : {results.get('eval_wer', 'N/A'):.4f}  (lower is better, 0.0 = perfect)")
    print(f"  CER : {results.get('eval_cer', 'N/A'):.4f}  (character error rate)")
    print("─────────────────────────────────────")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",   default="./whisper-amharic-final")
    parser.add_argument("--test_csv",    default="data/test.csv")
    parser.add_argument("--config_path", default="configs/amharic.yaml")
    args = parser.parse_args()

    run_evaluation(args.model_dir, args.test_csv, args.config_path)