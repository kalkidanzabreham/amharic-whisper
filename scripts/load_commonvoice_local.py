"""
Loads Common Voice Amharic from a locally downloaded tar.gz or extracted folder.
Mozilla moved Common Voice off HuggingFace in October 2025.

Download dataset manually from:
  https://commonvoice.mozilla.org/en/datasets

Then run:
  python scripts/load_commonvoice_local.py --cv_dir data/raw/am

Usage:
  python scripts/load_commonvoice_local.py --cv_dir path/to/extracted/am/folder
"""

import os
import argparse
import tarfile
import pandas as pd
import soundfile as sf
import torchaudio
import torchaudio.transforms as T
from pathlib import Path


AUDIO_OUT = Path("data/processed")
DATA_DIR  = Path("data")


def convert_mp3_to_wav(mp3_path: str, wav_path: str, target_sr: int = 16000):
    """Common Voice audio is mp3 — convert to 16kHz mono WAV for Whisper."""
    waveform, sr = torchaudio.load(mp3_path)

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

    torchaudio.save(wav_path, waveform, target_sr)


def process_tsv(tsv_path: str, clips_dir: str, split_name: str, output_csv: str):
    """
    Reads a Common Voice TSV file, converts mp3s to WAV, saves CSV.
    Common Voice TSV columns: client_id, path, sentence, ...
    """
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"\nProcessing {split_name}: {len(df)} samples from {tsv_path}")

    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = 0

    for i, row in df.iterrows():
        text = str(row.get("sentence", "")).strip()
        mp3_filename = row.get("path", "")

        if not text or not mp3_filename:
            failed += 1
            continue

        mp3_path = os.path.join(clips_dir, mp3_filename)

        if not os.path.exists(mp3_path):
            failed += 1
            continue

        wav_filename = f"cv_{split_name}_{i:05d}.wav"
        wav_path = str(AUDIO_OUT / wav_filename)

        try:
            convert_mp3_to_wav(mp3_path, wav_path)
            rows.append({"audio_path": wav_path, "text": text})
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Warning: failed on {mp3_filename} — {e}")

        if (i + 1) % 200 == 0:
            print(f"  Converted {i + 1}/{len(df)}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"  ✅ {split_name}: {len(result_df)} saved → {output_csv}  ({failed} skipped)")
    return result_df


def extract_if_needed(cv_dir: str) -> str:
    """
    If user passed a .tar.gz path, extract it first.
    Otherwise assume it's already extracted and return as-is.
    """
    if cv_dir.endswith(".tar.gz") or cv_dir.endswith(".tgz"):
        print(f"Extracting {cv_dir}...")
        extract_to = str(Path(cv_dir).parent / "am_extracted")
        with tarfile.open(cv_dir, "r:gz") as tar:
            tar.extractall(extract_to)
        print(f"Extracted to {extract_to}")
        # Find the 'am' subfolder inside
        extracted = Path(extract_to)
        candidates = list(extracted.rglob("train.tsv"))
        if candidates:
            return str(candidates[0].parent)
        return extract_to
    return cv_dir


def main(cv_dir: str):
    cv_dir = extract_if_needed(cv_dir)
    cv_path = Path(cv_dir)

    clips_dir = str(cv_path / "clips")

    if not os.path.exists(clips_dir):
        raise FileNotFoundError(
            f"Could not find 'clips' folder inside {cv_dir}\n"
            f"Make sure you're pointing to the extracted 'am' folder."
        )

    # Process each split
    # Common Voice has: train.tsv, dev.tsv (= val), test.tsv
    splits = {
        "train": DATA_DIR / "commonvoice_train.csv",
        "dev":   DATA_DIR / "commonvoice_val.csv",    # dev = validation
        "test":  DATA_DIR / "commonvoice_test.csv",
    }

    dataframes = {}
    for split_name, output_csv in splits.items():
        tsv_path = str(cv_path / f"{split_name}.tsv")
        if not os.path.exists(tsv_path):
            print(f"  Skipping {split_name}.tsv — not found")
            continue
        dataframes[split_name] = process_tsv(
            tsv_path, clips_dir, split_name, str(output_csv)
        )

    print("\n── Merging with FLEURS (if available) ───────────────")

    # Try to also pull FLEURS — still free, no auth needed
    try:
        from datasets import load_dataset
        print("Downloading FLEURS Amharic (no login required)...")

        fl_train = load_dataset("google/fleurs", "am_et", split="train")
        fl_test  = load_dataset("google/fleurs", "am_et", split="test")

        def save_fleurs(dataset, split_name, csv_path):
            import soundfile as sf
            rows = []
            for i, sample in enumerate(dataset):
                text = str(sample.get("transcription", "")).strip()
                if not text:
                    continue
                filename = f"fl_{split_name}_{i:05d}.wav"
                filepath = str(AUDIO_OUT / filename)
                sf.write(filepath, sample["audio"]["array"], sample["audio"]["sampling_rate"])
                rows.append({"audio_path": filepath, "text": text})
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            print(f"  ✅ FLEURS {split_name}: {len(df)} samples → {csv_path}")
            return df

        fl_train_df = save_fleurs(fl_train, "train", str(DATA_DIR / "fleurs_train.csv"))
        fl_test_df  = save_fleurs(fl_test,  "test",  str(DATA_DIR / "fleurs_test.csv"))

        # Merge Common Voice train + FLEURS train
        to_merge = [fl_train_df]
        if "train" in dataframes and not dataframes["train"].empty:
            to_merge.append(dataframes["train"])

        merged_train = pd.concat(to_merge, ignore_index=True).sample(frac=1, random_state=42)
        merged_train.to_csv(str(DATA_DIR / "train.csv"), index=False)
        print(f"\n✅ Merged train.csv: {len(merged_train)} total samples")

        # Test set
        to_merge_test = [fl_test_df]
        if "test" in dataframes and not dataframes["test"].empty:
            to_merge_test.append(dataframes["test"])
        merged_test = pd.concat(to_merge_test, ignore_index=True)
        merged_test.to_csv(str(DATA_DIR / "test.csv"), index=False)
        print(f"✅ Merged test.csv:  {len(merged_test)} total samples")

    except Exception as e:
        print(f"  FLEURS failed ({e}) — using Common Voice only")

        # Fallback — just use Common Voice splits directly
        if "train" in dataframes:
            dataframes["train"].to_csv(str(DATA_DIR / "train.csv"), index=False)
        if "test" in dataframes:
            dataframes["test"].to_csv(str(DATA_DIR / "test.csv"), index=False)

    # val.csv — prefer Common Voice dev split over auto-splitting
    if "dev" in dataframes and not dataframes["dev"].empty:
        dataframes["dev"].to_csv(str(DATA_DIR / "val.csv"), index=False)
        print(f"✅ val.csv created from Common Voice dev split")
        print("   (No need to run split_dataset.py — val.csv already exists)")
    else:
        print("\n⚠️  No dev.tsv found. Run split_dataset.py to create val.csv:")
        print("   python scripts/split_dataset.py")

    print("\n── Next step ─────────────────────────────────────────")
    print("  python train.py")
    print("──────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv_dir",
        required=True,
        help="Path to extracted Common Voice 'am' folder, or path to .tar.gz file"
    )
    args = parser.parse_args()
    main(args.cv_dir)