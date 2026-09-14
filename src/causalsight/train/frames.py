"""Pre-extract training frames so the GPU box never decodes video.

  cs-frames --rl data/rl/clevrer_train_rl.jsonl --per-type 375 \
            --sft data/sft/clevrer_train_sft.jsonl --sft-per-type 2000 \
            --out data/frames/train --n-frames 16

Selects a stratified RL subset (per question type) and a stratified SFT subset (per question type and
final answer), extracts `n_frames` uniformly sampled JPEG frames for every video they reference into
`<out>/frames/<video_stem>/f00.jpg ...`, and writes `rl_subset.jsonl`, `sft_subset.jsonl` (each record
gains a `frames` field, relative to <out>) and `manifest.json`. Frame sampling reuses the evaluation
sampler so training and evaluation see the same frames. Zip <out> and upload it to Drive.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from causalsight.eval.video import sample_frames


def stratified(records: list[dict], key, per_group: int, rng: random.Random) -> list[dict]:
    groups: dict = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    out = []
    for g in sorted(groups, key=str):
        rs = groups[g]
        out.extend(rng.sample(rs, min(per_group, len(rs))))
    return out


def _extract(args: tuple[str, str, int, int]) -> tuple[str, int]:
    video_path, out_dir, n_frames, max_side = args
    out = Path(out_dir)
    if out.exists() and len(list(out.glob("f*.jpg"))) == n_frames:
        return video_path, n_frames
    out.mkdir(parents=True, exist_ok=True)
    frames = sample_frames(video_path, n_frames)
    for i, f in enumerate(frames):
        if max(f.size) > max_side:
            scale = max_side / max(f.size)
            f = f.resize((round(f.width * scale), round(f.height * scale)))
        f.save(out / f"f{i:02d}.jpg", quality=90)
    return video_path, len(frames)


def run(rl: Path | None, per_type: int, sft: Path | None, sft_per_type: int, out: Path, n_frames: int, max_side: int, seed: int, workers: int) -> dict:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    rl_sub: list[dict] = []
    sft_sub: list[dict] = []
    if rl:
        rl_sub = stratified([json.loads(line) for line in rl.open()], lambda r: r["question_type"], per_type, rng)
    if sft:
        # descriptive: balance subtypes; MC: balance yes/no per type (per-option chains are ~50/50 anyway)
        key = lambda r: (r["question_type"], r["subtype"] if r["question_type"] == "descriptive" else r["gold"])
        sft_sub = stratified([json.loads(line) for line in sft.open()], key, sft_per_type, rng)
    videos = sorted({r["path"] for r in rl_sub + sft_sub})
    jobs = [(v, str(out / "frames" / Path(v).stem), n_frames, max_side) for v in videos]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _fut in as_completed([ex.submit(_extract, j) for j in jobs]):
            done += 1
            if done % 200 == 0:
                print(f"  extracted {done}/{len(jobs)} videos")
    for name, recs in (("rl_subset.jsonl", rl_sub), ("sft_subset.jsonl", sft_sub)):
        with (out / name).open("w") as f:
            for r in recs:
                r = dict(r)
                r["frames"] = f"frames/{Path(r['path']).stem}"
                f.write(json.dumps(r) + "\n")
    manifest = {
        "n_frames": n_frames,
        "max_side": max_side,
        "seed": seed,
        "videos": len(videos),
        "rl": dict(Counter(r["question_type"] for r in rl_sub)),
        "sft": dict(Counter(r["question_type"] for r in sft_sub)),
        "sft_answers": dict(Counter((r["question_type"], r["gold"]) for r in sft_sub).most_common(12)) if sft_sub else {},
    }
    manifest["sft_answers"] = {f"{k[0]}:{k[1]}": v for k, v in manifest["sft_answers"].items()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_eval_pack(root: Path, split: str, chains: Path, per_type: int, seed: int, out: Path, n_frames: int, max_side: int, workers: int) -> dict:
    """Everything the harness needs to evaluate on a GPU box without videos: the questions file, frames for
    the stratified evaluation subset's videos, their derender proposals (track masking), and the chains
    file (evidence regions). Selection reproduces `cs-eval --chains ... --per-type --seed`."""
    import shutil

    from causalsight.eval.benchmarks.clevrer import ClevrerBenchmark
    from causalsight.eval.harness import stratified_sample
    from causalsight.eval.masking import EvidenceIndex

    idx = EvidenceIndex.from_chains(chains)
    items = [it for it in ClevrerBenchmark(root).items(split) if idx.get(it.id)]
    items = stratified_sample(items, per_type, seed)
    videos = sorted({str(it.video_path) for it in items})
    out.mkdir(parents=True, exist_ok=True)
    (out / "questions").mkdir(exist_ok=True)
    shutil.copy(root / "questions" / f"{split}.json", out / "questions" / f"{split}.json")
    shutil.copy(chains, out / "chains.jsonl")
    (out / "derender_proposals").mkdir(exist_ok=True)
    for v in videos:
        scene = int(Path(v).stem.split("_")[1])
        shutil.copy(root / "derender_proposals" / f"proposal_{scene:05d}.json", out / "derender_proposals" / f"proposal_{scene:05d}.json")
    jobs = [(v, str(out / "frames" / Path(v).stem), n_frames, max_side) for v in videos]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_extract, jobs))
    manifest = {"split": split, "per_type": per_type, "seed": seed, "items": len(items), "videos": len(videos), "n_frames": n_frames}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-pack", action="store_true", help="build an evaluation pack instead of training frames")
    ap.add_argument("--root", type=Path, default=Path("data/raw/clevrer"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--chains", type=Path, default=None)
    ap.add_argument("--rl", type=Path, default=None)
    ap.add_argument("--per-type", type=int, default=375)
    ap.add_argument("--sft", type=Path, default=None)
    ap.add_argument("--sft-per-type", type=int, default=800, help="per (descriptive subtype) or (MC type, yes/no) group")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.eval_pack:
        m = build_eval_pack(a.root, a.split, a.chains, a.per_type, a.seed, a.out, a.n_frames, a.max_side, a.workers)
        print(json.dumps(m, indent=2))
        print(f"now: cd {a.out.parent} && zip -qr {a.out.name}.zip {a.out.name}   # then upload to Drive")
        return
    m = run(a.rl, a.per_type, a.sft, a.sft_per_type, a.out, a.n_frames, a.max_side, a.seed, a.workers)
    print(json.dumps(m, indent=2))
    print(f"now: cd {a.out.parent} && zip -qr {a.out.name}.zip {a.out.name}   # then upload to Drive")


if __name__ == "__main__":
    main()
