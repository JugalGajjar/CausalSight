"""Evaluation harness (notes/PLAN.md Phase B).

  cs-eval --model Qwen/Qwen2.5-VL-3B-Instruct --bench clevrer --split validation --limit 200 \
          --out results/zeroshot_qwen3b_clevrer.jsonl [--blind] [--max-frames 16]

Writes one JSONL row per item (id, question_type, prompt, raw, pred, gold, correct, ...) and a
summary JSON next to it. `--blind` replaces every frame with black (Blind Gap control).
`--model dummy` runs the whole pipeline without weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from causalsight.eval.benchmarks import load_benchmark
from causalsight.eval.video import blank_like, sample_frames
from causalsight.models import load_backend


def run(model: str, bench: str, split: str, root: Path, out: Path | None, limit: int | None, blind: bool, max_frames: int, **model_kw) -> dict:
    backend = load_backend(model, **model_kw)
    benchmark = load_benchmark(bench, root)
    rows: list[dict] = []
    frame_cache: dict[Path, list] = {}
    t0 = time.time()
    fout = out.open("w") if out else None
    for item in benchmark.items(split, limit):
        if backend.name == "dummy":
            frames = []
        else:
            if item.video_path not in frame_cache:
                frame_cache.clear()
                frame_cache[item.video_path] = sample_frames(item.video_path, max_frames)
            frames = frame_cache[item.video_path]
            if blind:
                frames = blank_like(frames)
        raw = backend.generate(frames, item.prompt)
        sc = benchmark.score(item, raw)
        row = {"id": item.id, "question_type": item.question_type, "prompt": item.prompt, "raw": raw, "gold": item.gold, **sc, **item.meta}
        rows.append(row)
        if fout:
            fout.write(json.dumps(row) + "\n")
            fout.flush()
    if fout:
        fout.close()
    summary = {
        "model": model,
        "bench": bench,
        "split": split,
        "blind": blind,
        "max_frames": max_frames,
        "n": len(rows),
        "seconds": round(time.time() - t0, 1),
        "overall": (sum(r["correct"] for r in rows) / len(rows)) if rows else None,
    }
    if hasattr(benchmark, "summarize"):
        summary["by_type"] = benchmark.summarize(rows)
    if out:
        out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="CausalSight evaluation harness")
    p.add_argument("--model", required=True, help="HF model id, or 'dummy'")
    p.add_argument("--bench", required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--root", type=Path, default=None, help="benchmark root (default data/raw/<bench>)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--blind", action="store_true", help="replace the video with black frames")
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    root = args.root or Path("data/raw") / args.bench
    summary = run(args.model, args.bench, args.split, root, args.out, args.limit, args.blind, args.max_frames, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
