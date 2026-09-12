"""Evaluation harness (notes/PLAN.md Phase B).

  cs-eval --model Qwen/Qwen2.5-VL-3B-Instruct --bench clevrer --split validation --limit 200 \
          --out results/zeroshot_qwen3b_clevrer.jsonl [--blind] [--max-frames 16]

Writes one JSONL row per item (id, question_type, prompt, raw, pred, gold, correct, ...) and a
summary JSON next to it. `--blind` replaces every frame with black (Blind Gap control). `--mask evidence --chains <jsonl>`
blacks out the ground-truth evidence regions (Evidence Sensitivity); `--mask random` is its control.
Passing `--chains` at all restricts the run to items that have evidence, so plain, blind, and masked
runs with the same `--chains --per-type --seed` evaluate exactly the same items.
`--model dummy` runs the whole pipeline without weights.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from causalsight.eval.benchmarks import load_benchmark
from causalsight.eval.masking import (
    EvidenceIndex,
    TrackMasker,
    mask_frames,
    masked_fraction,
    random_regions,
)
from causalsight.eval.video import blank_like, sample_frames
from causalsight.models import load_backend
from causalsight.train.format import ANSWER_RE, SYSTEM_PROMPT
from causalsight.train.grpo import PLAIN_SYSTEM

INSTR = {"none": "", "plain": PLAIN_SYSTEM, "chain": SYSTEM_PROMPT}

N_TOTAL_FRAMES = {"clevrer": 128}


def stratified_sample(items: list, per_type: int, seed: int) -> list:
    """Up to `per_type` items of each question type, chosen deterministically; original order kept."""
    rng = random.Random(seed)
    by_type: dict[str, list] = {}
    for it in items:
        by_type.setdefault(it.question_type, []).append(it)
    chosen = set()
    for its in by_type.values():
        for it in rng.sample(its, min(per_type, len(its))):
            chosen.add(it.id)
    return [it for it in items if it.id in chosen]


def run(
    model: str,
    bench: str,
    split: str,
    root: Path,
    out: Path | None,
    limit: int | None,
    blind: bool,
    max_frames: int,
    mask: str = "none",
    chains: Path | None = None,
    seed: int = 0,
    per_type: int | None = None,
    proposals_root: Path | None = None,
    instr: str = "none",
    max_new_tokens: int = 64,
    **model_kw,
) -> dict:
    backend = load_backend(model, **model_kw)
    benchmark = load_benchmark(bench, root)
    evidence = EvidenceIndex.from_chains(chains) if chains else None
    if mask != "none" and evidence is None:
        raise ValueError("--mask needs --chains <generated chains jsonl>")
    rng = random.Random(seed)
    n_total = N_TOTAL_FRAMES.get(bench, 128)
    tracker = TrackMasker(proposals_root or root / "derender_proposals") if mask in ("track", "track_random") else None
    track_evidence = EvidenceIndex.from_chains(chains, last_only=True) if (tracker is not None and chains) else None
    rows: list[dict] = []
    frame_cache: dict[Path, list] = {}
    t0 = time.time()
    fout = out.open("w") if out else None
    items = list(benchmark.items(split, None if per_type else limit))
    if evidence is not None:
        # keep the item set identical across plain / blind / masked runs on the same chains file
        items = [it for it in items if evidence.get(it.id)]
    if per_type:
        items = stratified_sample(items, per_type, seed)
    for item in items:
        regions = evidence.get(item.id) if evidence else []
        if mask == "random":
            regions = random_regions(regions, rng)
        elif tracker is not None:
            scene_index = int(item.id.split("_")[0])
            decisive = track_evidence.get(item.id) if track_evidence else []
            control = mask == "track_random"
            regions = tracker.regions_for(scene_index, decisive, n_total, max_frames, control=control, rng=rng)
            if not regions:
                continue  # decisive evidence matched no detection; skipped identically in both track modes
        if backend.name == "dummy":
            frames = []
        else:
            if item.video_path not in frame_cache:
                frame_cache.clear()
                frame_cache[item.video_path] = sample_frames(item.video_path, max_frames)
            frames = frame_cache[item.video_path]
            if blind:
                frames = blank_like(frames)
            elif mask != "none":
                frames = mask_frames(frames, regions, n_total)
        prompt = item.prompt + ("\n" + INSTR[instr] if INSTR[instr] else "")
        raw = backend.generate(frames, prompt, max_new_tokens=max_new_tokens)
        m = ANSWER_RE.search(raw)
        sc = benchmark.score(item, m.group(1).strip() if m else raw)  # trained models answer inside <answer> tags
        row = {"id": item.id, "question_type": item.question_type, "prompt": item.prompt, "raw": raw, "gold": item.gold, **sc, **item.meta}
        if regions:
            row["masked_fraction"] = round(masked_fraction(regions, n_total), 4)
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
        "instr": instr,
        "max_new_tokens": max_new_tokens,
        "mask": mask,
        "chains": str(chains) if chains else None,
        "per_type": per_type,
        "seed": seed,
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
    p.add_argument(
        "--mask",
        choices=["none", "evidence", "random", "track", "track_random"],
        default="none",
        help="evidence: black out GT evidence boxes at their moments; random: same boxes at random positions; "
        "track: remove the decisive evidence objects for the whole video; track_random: same-size masks elsewhere",
    )
    p.add_argument("--proposals-root", type=Path, default=None, help="CLEVRER derender proposals dir (track modes)")
    p.add_argument("--chains", type=Path, default=None, help="generated chains jsonl providing the evidence regions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-type", type=int, default=None, help="stratified subset: this many items per question type")
    p.add_argument("--instr", choices=list(INSTR), default="none", help="append the training-time format instruction (plain: <think>/<answer>; chain: triplet steps)")
    p.add_argument("--max-new-tokens", type=int, default=64, help="raise to ~384 for --instr plain/chain")
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    root = args.root or Path("data/raw") / args.bench
    summary = run(
        args.model, args.bench, args.split, root, args.out, args.limit, args.blind, args.max_frames,
        mask=args.mask, chains=args.chains, seed=args.seed, per_type=args.per_type, proposals_root=args.proposals_root,
        instr=args.instr, max_new_tokens=args.max_new_tokens, device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
