"""
Utility helpers used across the project.
Main job: validate your audio files BEFORE training starts
so you don't waste GPU hours on a dataset with broken files.
"""

import os
import torch
import torchaudio
import pandas as pd
from pathlib import Path


def validate_dataset(csv_path: str, audio_col: str = "audio_path", text_col: str = "text") -> bool:
    """
    Reads a CSV and checks every audio file before training.
    Catches problems like:
    - Audio file doesn't exist
    - Audio is not 16kHz
    - Audio is stereo (needs to be mono)
    - Transcript is empty or NaN
    - Audio is shorter than 0.1s or longer than 30s (Whisper's max)

    Returns True if all samples pass, False if any fail.
    Prints a report of every broken sample so you can fix them.
    """
    df = pd.read_csv(csv_path)
    errors = []

    print(f"Validating {len(df)} samples from {csv_path}...")

    for i, row in df.iterrows():
        audio_path = row[audio_col]
        text = row[text_col]

        # Check transcript
        if pd.isna(text) or str(text).strip() == "":
            errors.append(f"Row {i}: Empty transcript — {audio_path}")
            continue

        # Check file exists
        if not os.path.exists(audio_path):
            errors.append(f"Row {i}: File not found — {audio_path}")
            continue

        # Check audio properties
        try:
            info = torchaudio.info(audio_path)

            if info.sample_rate != 16000:
                errors.append(
                    f"Row {i}: Wrong sample rate {info.sample_rate}Hz (need 16000) — {audio_path}"
                )

            if info.num_channels > 1:
                errors.append(
                    f"Row {i}: Stereo audio (need mono) — {audio_path}"
                )

            duration = info.num_frames / info.sample_rate
            if duration < 0.1:
                errors.append(f"Row {i}: Audio too short ({duration:.2f}s) — {audio_path}")
            if duration > 30.0:
                errors.append(
                    f"Row {i}: Audio too long ({duration:.1f}s > 30s Whisper max) — {audio_path}. "
                    "Split this file or it will be truncated."
                )

        except Exception as e:
            errors.append(f"Row {i}: Could not read audio — {audio_path} ({e})")

    if errors:
        print(f"\n❌ Found {len(errors)} problems:\n")
        for err in errors:
            print(f"   {err}")
        print(f"\nFix these before running train.py\n")
        return False
    else:
        print(f"✅ All {len(df)} samples passed validation\n")
        return True


def get_audio_duration(audio_path: str) -> float:
    """Returns duration of an audio file in seconds."""
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def dataset_summary(csv_path: str, audio_col: str = "audio_path"):
    """
    Prints a summary of a dataset CSV — total duration, sample count, avg duration.
    Useful to know if you have enough data before starting training.
    """
    df = pd.read_csv(csv_path)
    durations = []

    for path in df[audio_col]:
        try:
            durations.append(get_audio_duration(path))
        except:
            pass

    total_hours = sum(durations) / 3600
    avg_seconds = sum(durations) / len(durations) if durations else 0

    print(f"Dataset: {csv_path}")
    print(f"  Samples      : {len(df)}")
    print(f"  Total audio  : {total_hours:.2f} hours")
    print(f"  Avg duration : {avg_seconds:.1f}s per sample")
    print(f"  Readable     : {len(durations)}/{len(df)} files\n")