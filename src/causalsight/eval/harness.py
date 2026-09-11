"""Evaluation harness entry point (notes/PLAN.md Phase B).

Usage: cs-eval --model <path-or-hf-id> --bench clevrer [--blind] [--out results/run.jsonl]

Each benchmark loader yields dicts with: id, question_type, video_path, question, options, gold.
Writes one JSONL line per item: id, question_type, prediction, gold, correct.
"""

from __future__ import annotations

import argparse
from pathlib import Path

BENCHMARKS = ["clevrer", "causalvqa", "mmvu", "videomme", "tempcompass", "videoholmes"]


def load_benchmark(name: str, split: str):
    raise NotImplementedError(f"loader for {name} not written yet")


def main() -> None:
    p = argparse.ArgumentParser(description="CausalSight evaluation harness")
    p.add_argument("--model", required=True)
    p.add_argument("--bench", choices=BENCHMARKS, required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--blind", action="store_true", help="replace the video with a blank clip")
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    raise NotImplementedError(f"harness body not written yet (model={args.model}, bench={args.bench}); see notes/PLAN.md Phase B")


if __name__ == "__main__":
    main()
