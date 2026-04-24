"""
Tests SeamlessM4T on Amharic test set — ZERO training required.
Run this before training anything to know your baseline.

Usage: python -c "exec(open('seamless_test.py').read())"
"""

import torch
import pandas as pd
import soundfile as sf
import evaluate
from transformers import AutoProcessor, SeamlessM4TModel

print("Loading SeamlessM4T...")
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model     = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
model.eval()

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

test_df  = pd.read_csv("/content/drive/MyDrive/amharic-whisper/data/test.csv")
samples  = test_df.sample(min(50, len(test_df)), random_state=42)  # test on 50 samples

pred_strs  = []
label_strs = []

print(f"Running zero-shot transcription on {len(samples)} samples...\n")

for i, (_, row) in enumerate(samples.iterrows()):
    try:
        audio, sr = sf.read(row["audio_path"])
        inputs    = processor(audios=audio, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs, tgt_lang="amh")

        prediction = processor.decode(output[0].tolist(), skip_special_tokens=True)
        pred_strs.append(prediction.strip())
        label_strs.append(str(row["text"]).strip())

        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(samples)}")

    except Exception as e:
        print(f"  Sample {i} failed: {e}")

wer = wer_metric.compute(predictions=pred_strs, references=label_strs)
cer = cer_metric.compute(predictions=pred_strs, references=label_strs)

print(f"\n── SeamlessM4T Zero-Shot Results ──")
print(f"  WER : {wer:.4f}")
print(f"  CER : {cer:.4f}")
print(f"  Samples tested: {len(pred_strs)}")