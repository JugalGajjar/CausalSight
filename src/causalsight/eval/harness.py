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

from causalsight.data.schema import Evidence
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
from causalsight.train.data import load_frames
from causalsight.train.format import ANSWER_RE, SYSTEM_PROMPT, parse_chain
from causalsight.train.grpo import PLAIN_SYSTEM

INSTR = {"none": "", "plain": PLAIN_SYSTEM, "chain": SYSTEM_PROMPT}

N_TOTAL_FRAMES = {"clevrer": 128}


def chain_metrics(raw: str, gt_regions: list, iou_thr: float = 0.5, t_tol: int = 8) -> dict:
    """Chain-level metrics for a generated response: format validity, step count, and grounding of the
    emitted evidence against the ground-truth evidence regions of the question.
    ground_st:      mean over emitted steps of the best strict spatiotemporal IoU with any GT region
    ground_spatial: mean over emitted steps of the best spatial IoU among GT regions within t_tol frames
    ground_recall:  fraction of GT regions hit by some step with spatial IoU >= iou_thr within t_tol frames
    t_offset:       mean frames between each step's span and the nearest GT span"""
    p = parse_chain(raw)
    out = {"chain_ok": int(p.chain is not None), "n_steps": p.n_steps}
    if p.chain is None or not gt_regions:
        return out
    gt = [Evidence(t0, t1, tuple(b)) for (t0, t1, b) in gt_regions]
    steps = [t.evidence for t in p.chain.triplets]

    def near(a: Evidence, b: Evidence) -> bool:
        return a.temporal_offset(b) <= t_tol

    out["ground_st"] = sum(max(e.st_iou(g) for g in gt) for e in steps) / len(steps)
    out["ground_spatial"] = sum(max((e.spatial_iou(g) for g in gt if near(e, g)), default=0.0) for e in steps) / len(steps)
    out["ground_recall"] = sum(1 for g in gt if any(near(e, g) and e.spatial_iou(g) >= iou_thr for e in steps)) / len(gt)
    out["t_offset"] = sum(min(e.temporal_offset(g) for g in gt) for e in steps) / len(steps)
    return out


def stratified_sample(items: list, per_type: int, seed: int, key=None) -> list:
    """Up to `per_type` *questions* of each type, chosen deterministically; original order kept.
    `key` maps an item to its question id so per-option items of one question stay together, and the
    same seed selects the same questions in multi and per-option modes."""
    key = key or (lambda it: it.id)
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = {}
    seen: set[str] = set()
    for it in items:
        q = key(it)
        if q not in seen:
            seen.add(q)
            by_type.setdefault(it.question_type, []).append(q)
    chosen = set()
    for qs in by_type.values():
        chosen.update(rng.sample(qs, min(per_type, len(qs))))
    return [it for it in items if key(it) in chosen]


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
    mc: str = "multi",
    frames_dir: Path | None = None,
    batch_size: int = 1,
    max_items: int | None = None,
    **model_kw,
) -> dict:
    backend = load_backend(model, **model_kw)
    benchmark = load_benchmark(bench, root)
    if hasattr(benchmark, "mc_mode"):
        benchmark.mc_mode = mc
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
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    fout = out.open("w") if out else None
    items = list(benchmark.items(split, None if per_type else limit))
    qkey = lambda it: it.meta.get("qid", it.id)  # per-option items share the question's evidence
    if evidence is not None:
        # keep the item set identical across plain / blind / masked runs on the same chains file
        items = [it for it in items if evidence.get(qkey(it))]
    if per_type:
        items = stratified_sample(items, per_type, seed, key=qkey)
    if max_items:
        items = items[:max_items]  # pilot: first items of the same stratified subset
    pending: list[tuple] = []

    def flush() -> None:
        if not pending:
            return
        raws = backend.generate_batch([p[1] for p in pending], [p[3] for p in pending], max_new_tokens)
        for (item, _frames, regs, prompt), raw in zip(pending, raws):
            m = ANSWER_RE.search(raw)
            sc = benchmark.score(item, m.group(1).strip() if m else raw)  # trained models answer inside <answer> tags
            if instr == "chain":
                sc.update(chain_metrics(raw, evidence.get_for_item(item.id, qkey(item)) if evidence else []))
            row = {"id": item.id, "question_type": item.question_type, "prompt": prompt, "raw": raw, "gold": item.gold, **sc, **item.meta}
            if regs:
                row["masked_fraction"] = round(masked_fraction(regs, n_total), 4)
            rows.append(row)
            if fout:
                fout.write(json.dumps(row) + "\n")
                fout.flush()
        pending.clear()

    for item in items:
        regions = evidence.get(qkey(item)) if evidence else []
        if mask != "none" and not regions:
            continue  # no localizable evidence for this item; skip so masked/unmasked sets match
        if mask == "random":
            regions = random_regions(regions, rng)
        elif tracker is not None:
            scene_index = int(item.id.split("_")[0])
            decisive = track_evidence.get(qkey(item)) if track_evidence else []
            regions = tracker.regions_for(scene_index, decisive, n_total, max_frames, control=(mask == "track_random"), rng=rng)
            if not regions:
                continue  # decisive evidence matched no detection; skipped identically in both track modes
        if backend.name == "dummy":
            frames = []
        else:
            if item.video_path not in frame_cache:
                if len(frame_cache) > 2 * batch_size:
                    frame_cache.clear()
                if frames_dir:  # pre-extracted frames (eval pack); no video decoding
                    frame_cache[item.video_path] = load_frames(frames_dir, {"frames": f"frames/{item.video_path.stem}"}, max_frames)
                else:
                    frame_cache[item.video_path] = sample_frames(item.video_path, max_frames)
            frames = frame_cache[item.video_path]
            if blind:
                frames = blank_like(frames)
            elif mask != "none":
                frames = mask_frames(frames, regions, n_total)
        prompt = item.prompt + ("\n" + INSTR[instr] if INSTR[instr] else "")
        pending.append((item, frames, regions, prompt))
        if len(pending) >= batch_size:
            flush()
    flush()
    if fout:
        fout.close()
    summary = {
        "model": model,
        "bench": bench,
        "split": split,
        "blind": blind,
        "instr": instr,
        "max_new_tokens": max_new_tokens,
        "mc": mc,
        "batch_size": batch_size,
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
    if instr == "chain" and rows:
        ch: dict[str, dict] = {}
        for t in sorted({r["question_type"] for r in rows}) + ["ALL"]:
            rs = [r for r in rows if t == "ALL" or r["question_type"] == t]
            ok = [r for r in rs if r.get("chain_ok")]
            d = {"n": len(rs), "format_rate": sum(r.get("chain_ok", 0) for r in rs) / len(rs), "mean_steps": sum(r.get("n_steps", 0) for r in rs) / len(rs)}
            for k in ("ground_st", "ground_spatial", "ground_recall", "t_offset"):
                vals = [r[k] for r in ok if k in r]
                if vals:
                    d[k] = sum(vals) / len(vals)
            ch[t] = d
        summary["chain"] = ch
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
    p.add_argument("--mc", choices=["multi", "option"], default="multi", help="multiple choice as one letters prompt, or one yes/no item per option (chain models)")
    p.add_argument("--frames-dir", type=Path, default=None, help="eval pack dir with frames/<video>/fNN.jpg (skips video decoding)")
    p.add_argument("--batch-size", type=int, default=1, help="items per generate call; 16 on an 80 GB GPU, 1 on the Mac")
    p.add_argument("--max-items", type=int, default=None, help="pilot: stop after this many items of the stratified subset")
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--init-adapter", default=None, help="adapter to merge before --model (normally found via the run's config.yaml)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    root = args.root or Path("data/raw") / args.bench
    summary = run(
        args.model, args.bench, args.split, root, args.out, args.limit, args.blind, args.max_frames,
        mask=args.mask, chains=args.chains, seed=args.seed, per_type=args.per_type, proposals_root=args.proposals_root,
        instr=args.instr, max_new_tokens=args.max_new_tokens, mc=args.mc, frames_dir=args.frames_dir, batch_size=args.batch_size, max_items=args.max_items, device=args.device, init_adapter=args.init_adapter,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
