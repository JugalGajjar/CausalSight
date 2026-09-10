"""R_out: final-answer correctness. Unchanged from standard GRPO."""

from __future__ import annotations


def score(prediction: str, gold: str) -> float:
    return float(prediction.strip().lower() == gold.strip().lower())
