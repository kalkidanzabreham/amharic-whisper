"""
WER (Word Error Rate) and CER (Character Error Rate) computation.
CER matters more for Amharic because it's written in Ge'ez script
where character-level accuracy is a meaningful signal.
"""

import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # ✅ Fix: replace -100 with pad token before decoding — ChatGPT missed this
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Strip whitespace — common source of artificially bad WER
    pred_str  = [p.strip() for p in pred_str]
    label_str = [l.strip() for l in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4)   # track both — CER is more informative for Amharic
    }