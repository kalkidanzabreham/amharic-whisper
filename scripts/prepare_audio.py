"""
Converts any audio format to 16kHz mono WAV — required by Whisper.
Run this BEFORE building your CSV.

Usage: python scripts/prepare_audio.py --input data/raw --output data/processed
"""

import os
import argparse
import torchaudio
import torchaudio.transforms as T
from pathlib import Path


def convert_audio(input_path: str, output_path: str, target_sr: int = 16000):
    waveform, sample_rate = torchaudio.load(input_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sr:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    torchaudio.save(output_path, waveform, target_sr)


def process_directory(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [f for f in input_path.rglob("*") if f.suffix.lower() in supported]

    print(f"Found {len(files)} audio files")

    for i, audio_file in enumerate(files):
        out_file = output_path / (audio_file.stem + ".wav")
        try:
            convert_audio(str(audio_file), str(out_file))
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(files)}")
        except Exception as e:
            print(f"Failed: {audio_file.name} — {e}")

    print("Done. All audio converted to 16kHz mono WAV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    process_directory(args.input, args.output)