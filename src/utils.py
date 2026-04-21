"""
Utility helpers — uses soundfile instead of torchaudio.info
which has inconsistent API across versions.
"""

import os
import pandas as pd
import soundfile as sf


def get_audio_duration(audio_path: str) -> float:
    """Returns duration of an audio file in seconds using soundfile."""
    info = sf.info(audio_path)
    return info.duration


def dataset_summary(csv_path: str, audio_col: str = "audio_path"):
    df = pd.read_csv(csv_path)
    durations = []
    failed = 0

    for path in df[audio_col]:
        try:
            durations.append(get_audio_duration(path))
        except Exception as e:
            failed += 1

    if not durations:
        print(f"Dataset: {csv_path}")
        print(f"  ❌ Could not read any files ({failed} failed)\n")
        return

    total_hours = sum(durations) / 3600
    avg_seconds = sum(durations) / len(durations)

    print(f"Dataset: {csv_path}")
    print(f"  Samples      : {len(df)}")
    print(f"  Total audio  : {total_hours:.2f} hours")
    print(f"  Avg duration : {avg_seconds:.1f}s per sample")
    print(f"  Readable     : {len(durations)}/{len(df)} files\n")


def validate_dataset(csv_path: str, audio_col: str = "audio_path", text_col: str = "text") -> bool:
    df = pd.read_csv(csv_path)
    errors = []

    print(f"Validating {len(df)} samples from {csv_path}...")

    for i, row in df.iterrows():
        audio_path = row[audio_col]
        text = row[text_col]

        if pd.isna(text) or str(text).strip() == "":
            errors.append(f"Row {i}: Empty transcript — {audio_path}")
            continue

        if not os.path.exists(audio_path):
            errors.append(f"Row {i}: File not found — {audio_path}")
            continue

        try:
            info = sf.info(audio_path)

            if info.samplerate != 16000:
                errors.append(
                    f"Row {i}: Wrong sample rate {info.samplerate}Hz (need 16000) — {audio_path}"
                )

            if info.channels > 1:
                errors.append(
                    f"Row {i}: Stereo audio (need mono) — {audio_path}"
                )

            if info.duration < 0.1:
                errors.append(f"Row {i}: Too short ({info.duration:.2f}s) — {audio_path}")

            if info.duration > 30.0:
                errors.append(
                    f"Row {i}: Too long ({info.duration:.1f}s > 30s Whisper max) — {audio_path}"
                )

        except Exception as e:
            errors.append(f"Row {i}: Cannot read file — {audio_path} ({e})")

    if errors:
        print(f"\n❌ Found {len(errors)} problems:\n")
        for err in errors[:20]:   # cap at 20 so terminal doesn't flood
            print(f"   {err}")
        if len(errors) > 20:
            print(f"   ... and {len(errors) - 20} more")
        print()
        return False

    print(f"✅ All {len(df)} samples passed\n")
    return True