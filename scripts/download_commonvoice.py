"""
Downloads Amharic Common Voice + FLEURS from HuggingFace and saves as CSV.
This gives you real labeled data to start with immediately.

Usage: python scripts/download_commonvoice.py
"""

from datasets import load_dataset
import pandas as pd
import os
import soundfile as sf
from pathlib import Path


def save_split(dataset, split_name: str, audio_dir: str, csv_path: str):
    Path(audio_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    for i, sample in enumerate(dataset):
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        filename = f"{split_name}_{i:05d}.wav"
        filepath = os.path.join(audio_dir, filename)

        sf.write(filepath, audio_array, sr)
        rows.append({"audio_path": filepath, "text": sample["sentence"]})

        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1} samples")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows → {csv_path}")


def main():
    print("Downloading Common Voice Amharic...")
    # Common Voice is the best starting point — real human speech
    cv = load_dataset("mozilla-foundation/common_voice_13_0", "am", trust_remote_code=True)

    save_split(cv["train"], "cv_train", "data/processed", "data/commonvoice_train.csv")
    save_split(cv["test"],  "cv_test",  "data/processed", "data/commonvoice_test.csv")

    print("\nDownloading FLEURS Amharic...")
    # FLEURS adds more diverse speech — combine with Common Voice
    fleurs = load_dataset("google/fleurs", "am_et", trust_remote_code=True)

    save_split(fleurs["train"], "fl_train", "data/processed", "data/fleurs_train.csv")
    save_split(fleurs["test"],  "fl_test",  "data/processed", "data/fleurs_test.csv")

    # Merge both sources into one training CSV
    cv_train = pd.read_csv("data/commonvoice_train.csv")
    fl_train  = pd.read_csv("data/fleurs_train.csv")
    merged = pd.concat([cv_train, fl_train], ignore_index=True).sample(frac=1, random_state=42)
    merged.to_csv("data/train.csv", index=False)
    print(f"\nMerged training set: {len(merged)} samples → data/train.csv")


if __name__ == "__main__":
    main()