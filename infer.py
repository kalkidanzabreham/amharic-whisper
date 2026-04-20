"""
Transcribe a new Amharic audio file using your fine-tuned model.

Usage: python infer.py --audio path/to/audio.wav
"""

import argparse
from transformers import pipeline


def transcribe(audio_path: str, model_dir: str) -> str:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        chunk_length_s=30,      # handles long audio by chunking it
        stride_length_s=5,      # overlap between chunks for continuity
    )
    result = pipe(audio_path, return_timestamps=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",     required=True,  help="Path to audio file")
    parser.add_argument("--model_dir", default="./whisper-amharic-final")
    args = parser.parse_args()

    output = transcribe(args.audio, args.model_dir)
    print("\nTranscript:")
    print(output["text"])

    print("\nWith timestamps:")
    for chunk in output.get("chunks", []):
        start, end = chunk["timestamp"]
        print(f"  [{start:.1f}s → {end:.1f}s]  {chunk['text']}")