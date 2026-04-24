"""
Evaluates all trained models on the same test set and prints a comparison.

Usage: python compare_models.py
"""

import torch
import numpy as np
import pandas as pd
import soundfile as sf
import evaluate
from pathlib import Path
from transformers import (
    pipeline,
    AutoProcessor,
    SeamlessM4TModel,
)

TEST_CSV  = "/content/drive/MyDrive/amharic-whisper/data/test.csv"
N_SAMPLES = 100   # increase for more reliable numbers

MODELS = {
    "Whisper (fine-tuned)": {
        "type": "pipeline",
        "path": "/content/drive/MyDrive/amharic-whisper/whisper-amharic-final",
        "task": "automatic-speech-recognition",
    },
    "MMS (fine-tuned)": {
        "type": "pipeline",
        "path": "/content/drive/MyDrive/amharic-whisper/mms-amharic-final",
        "task": "automatic-speech-recognition",
    },
    "Wav2Vec2 XLSR (fine-tuned)": {
        "type": "pipeline",
        "path": "/content/drive/MyDrive/amharic-whisper/wav2vec2-amharic-final",
        "task": "automatic-speech-recognition",
    },
}

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

test_df = pd.read_csv(TEST_CSV).sample(
    min(N_SAMPLES, pd.read_csv(TEST_CSV).shape[0]),
    random_state=42
)

results = []

for model_name, config in MODELS.items():
    model_path = Path(config["path"])
    if not model_path.exists():
        print(f"Skipping {model_name} — not trained yet ({config['path']})")
        continue

    print(f"\nEvaluating: {model_name}...")

    try:
        pipe = pipeline(config["task"], model=config["path"])
        pred_strs, label_strs = [], []

        for _, row in test_df.iterrows():
            try:
                out = pipe(row["audio_path"])
                pred_strs.append(out["text"].strip())
                label_strs.append(str(row["text"]).strip())
            except:
                pass

        wer = wer_metric.compute(predictions=pred_strs, references=label_strs)
        cer = cer_metric.compute(predictions=pred_strs, references=label_strs)

        results.append({
            "Model":   model_name,
            "WER":     round(wer, 4),
            "CER":     round(cer, 4),
            "Samples": len(pred_strs),
        })
        print(f"  WER: {wer:.4f} | CER: {cer:.4f}")

    except Exception as e:
        print(f"  Failed: {e}")


# ── Print comparison table ─────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Model':<28} {'WER':>8} {'CER':>8} {'Samples':>8}")
print("="*55)

for r in sorted(results, key=lambda x: x["WER"]):
    print(f"{r['Model']:<28} {r['WER']:>8.4f} {r['CER']:>8.4f} {r['Samples']:>8}")

print("="*55)
print("Lower WER/CER = better. WER < 0.30 is production-usable.")