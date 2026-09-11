"""Validate the traced executor against CLEVRER ground-truth answers.

  cs-clevrer-validate descriptive --n-videos 500 --thresholds 0.02,0.05,0.1,0.2,0.3,0.5
      agreement of executor answers with GT answers for descriptive questions, per moving threshold
      and per subtype; picks the best threshold.
  cs-clevrer-validate explanatory --n-videos 500
      explanatory choices are fully computable from observed events; agreement with choice labels.

Both should be near 100%. Anything else means the scene model or a module is wrong.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from causalsight.data.clevrer_executor import TracedExecutor
from causalsight.data.clevrer_sim import Scene


def find_annotation(root: Path, split: str, scene_index: int) -> Path:
    lo = (scene_index // 1000) * 1000
    return root / "annotations" / split / f"annotation_{split}" / f"annotation_{lo:05d}-{lo + 1000:05d}" / f"annotation_{scene_index:05d}.json"


def load_questions(root: Path, split: str, n_videos: int | None) -> list[dict]:
    qs = json.loads((root / "questions" / f"{split}.json").read_text())
    return qs[:n_videos] if n_videos else qs


def run_descriptive(root: Path, split: str, videos: list[dict], thresholds: list[float]) -> None:
    results: dict[float, Counter] = {t: Counter() for t in thresholds}
    totals: Counter = Counter()
    motion_totals: Counter = Counter()
    motion_hits: dict[float, Counter] = {t: Counter() for t in thresholds}
    mismatch_examples: dict[float, list] = defaultdict(list)
    for v in videos:
        ann = find_annotation(root, split, v["scene_index"])
        scenes = {t: Scene.from_annotation(ann, moving_th=t) for t in thresholds}
        for q in v["questions"]:
            if q["question_type"] != "descriptive":
                continue
            st = q["question_subtype"]
            uses_motion = any(m in q["program"] for m in ("filter_moving", "filter_stationary"))
            totals[st] += 1
            if uses_motion:
                motion_totals[st] += 1
            for t in thresholds:
                out = TracedExecutor(scenes[t]).run(q["program"])
                ok = TracedExecutor.answer_str(out) == str(q["answer"])
                if ok:
                    results[t][st] += 1
                    if uses_motion:
                        motion_hits[t][st] += 1
                elif len(mismatch_examples[t]) < 5:
                    mismatch_examples[t].append((v["scene_index"], q["question"], q["program"], q["answer"], TracedExecutor.answer_str(out)))
    n = sum(totals.values())
    print(f"descriptive questions: {n} over {len(videos)} videos\n")
    print(f"{'threshold':>10s} {'overall':>8s} " + " ".join(f"{st:>15s}" for st in sorted(totals)) + "   motion-only")
    best = None
    for t in thresholds:
        acc = sum(results[t].values()) / n
        mo = sum(motion_hits[t].values()) / max(1, sum(motion_totals.values()))
        row = f"{t:10.3f} {acc:8.4f} " + " ".join(f"{results[t][st] / totals[st]:15.4f}" for st in sorted(totals)) + f"   {mo:.4f}"
        print(row)
        if best is None or acc > best[1]:
            best = (t, acc)
    print(f"\nbest threshold: {best[0]} (overall agreement {best[1]:.4f})")
    print("\nmismatches at best threshold:")
    for ex in mismatch_examples[best[0]]:
        print(" ", ex)


def run_explanatory(root: Path, split: str, videos: list[dict]) -> None:
    tot = Counter()
    examples = []
    for v in videos:
        scene = Scene.from_annotation(find_annotation(root, split, v["scene_index"]))
        for q in v["questions"]:
            if q["question_type"] != "explanatory":
                continue
            for c in q["choices"]:
                ex = TracedExecutor(scene)
                out = ex.run(c["program"] + q["program"])
                pred = {"yes": "correct", "no": "wrong"}.get(TracedExecutor.answer_str(out), TracedExecutor.answer_str(out))
                tot[pred == c["answer"]] += 1
                if pred != c["answer"] and len(examples) < 5:
                    examples.append((v["scene_index"], q["question"], c["choice"], c["answer"], pred))
    n = sum(tot.values())
    print(f"explanatory choices: {n}; agreement {tot[True] / n:.4f}")
    for e in examples:
        print(" ", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["descriptive", "explanatory"])
    ap.add_argument("--root", type=Path, default=Path("data/raw/clevrer"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-videos", type=int, default=300)
    ap.add_argument("--thresholds", default="0.02,0.05,0.1,0.15,0.2,0.3,0.5")
    args = ap.parse_args()
    videos = load_questions(args.root, args.split, args.n_videos)
    if args.mode == "descriptive":
        run_descriptive(args.root, args.split, videos, [float(x) for x in args.thresholds.split(",")])
    else:
        run_explanatory(args.root, args.split, videos)


if __name__ == "__main__":
    main()
