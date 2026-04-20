"""
Splits train.csv into train + val sets.
Run this AFTER download_commonvoice.py and BEFORE train.py.

Why: The download script creates train.csv and test.csv but not val.csv.
     Whisper fine-tuning needs a validation set during training to track WER
     and save the best checkpoint. Without it, train.py crashes.

Usage: python scripts/split_dataset.py
       python scripts/split_dataset.py --val_size 0.15  # custom split
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split(train_csv: str, val_size: float, random_state: int):
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} total training samples from {train_csv}")

    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} samples found. Need at least 20 to split meaningfully. "
            "Run download_commonvoice.py first or add more audio to data/processed/"
        )

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        shuffle=True
    )

    # Overwrite train.csv with the reduced training set
    train_df.to_csv(train_csv, index=False)

    # Write val.csv next to it
    val_path = Path(train_csv).parent / "val.csv"
    val_df.to_csv(val_path, index=False)

    print(f"Split complete:")
    print(f"  Train → {len(train_df)} samples  ({train_csv})")
    print(f"  Val   → {len(val_df)} samples  ({val_path})")
    print(f"\nYou can now run: python train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        default="data/train.csv",
        help="Path to the training CSV to split"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )
    args = parser.parse_args()
    split(args.train_csv, args.val_size, args.random_state)