"""
DataCollator for Whisper fine-tuning.
This is THE most critical missing piece from ChatGPT's version.
Without this, training either crashes or trains on wrong data.

What it does:
- Pads audio features to same length in a batch
- Pads label token sequences to same length
- Replaces label padding with -100 so loss ignores padding
- Removes BOS token from labels (Whisper adds it at generation time)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # --- Pad audio input features ---
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # --- Pad label sequences ---
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present — Whisper adds it during generation
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch