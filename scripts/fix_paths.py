"""
Fixes the path mismatch between CSV records and actual file locations.

The download script saved relative paths like:
    data/processed/cv_train_00000.wav

But files are actually at:
    /content/drive/MyDrive/amharic-whisper/data/processed/cv_train_00000.wav

This script rewrites all CSVs with correct absolute paths.

Usage: python scripts/fix_paths.py
"""

import os
import pandas as pd
from pathlib import Path


# ── Config: set these to match your actual setup ───────────────────────────────

# Where your project actually lives (the folder containing data/, train.py, etc.)
PROJECT_ROOT = Path("/content/drive/MyDrive/amharic-whisper")

# All CSVs to fix
CSV_FILES = [
    PROJECT_ROOT / "data" / "train.csv",
    PROJECT_ROOT / "data" / "val.csv",
    PROJECT_ROOT / "data" / "test.csv",
]

AUDIO_COLUMN = "audio_path"


# ── Fix ────────────────────────────────────────────────────────────────────────

def fix_csv(csv_path: Path, project_root: Path):
    if not csv_path.exists():
        print(f"Skipping (not found): {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if AUDIO_COLUMN not in df.columns:
        print(f"Skipping (no '{AUDIO_COLUMN}' column): {csv_path}")
        return

    fixed = 0
    missing = 0

    def fix_path(p: str) -> str:
        nonlocal fixed, missing
        p = str(p).strip()

        # Already absolute and exists — nothing to do
        if os.path.isabs(p) and os.path.exists(p):
            return p

        # Strip any leading path components and rebuild from project root
        # Handles cases like "data/processed/x.wav" or "./data/processed/x.wav"
        filename = Path(p).name                          # just "cv_train_00000.wav"
        absolute = project_root / "data" / "processed" / filename

        if absolute.exists():
            fixed += 1
            return str(absolute)
        else:
            missing += 1
            return str(absolute)  # write correct path anyway, flag below

    df[AUDIO_COLUMN] = df[AUDIO_COLUMN].apply(fix_path)
    df.to_csv(csv_path, index=False)

    print(f"✅ {csv_path.name}: {fixed} paths fixed, {missing} files not found on disk")


def main():
    print(f"Project root: {PROJECT_ROOT}\n")

    for csv_file in CSV_FILES:
        fix_csv(csv_file, PROJECT_ROOT)

    print("\nDone. Re-run: python train.py")
    print("\nIf you still see 'file not found' errors, your PROJECT_ROOT is wrong.")
    print(f"Current PROJECT_ROOT = {PROJECT_ROOT}")
    print("Update it at the top of this script to match your actual path.")


if __name__ == "__main__":
    main()