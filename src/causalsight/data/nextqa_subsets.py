"""Stratified NExT-QA subsets: a training pool for pseudo-labeled chains and a validation test set.

  cs-nextqa-subsets --root data/raw/nextqa --out data/subsets --train-videos 1500 --val-questions 1000

Training: all questions of `train_videos` randomly chosen training videos (bounded frame extraction),
then capped per question family so causal/temporal/descriptive are balanced. Validation: `val_questions`
questions stratified by family. Writes nextqa_train_pool.jsonl and nextqa_val_ids.json.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from causalsight.eval.benchmarks.nextqa import FAMILY, load_rows, load_vidmap, video_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/nextqa"))
    ap.add_argument("--out", type=Path, default=Path("data/subsets"))
    ap.add_argument("--train-videos", type=int, default=1500)
    ap.add_argument("--per-family", type=int, default=2200, help="cap per family in the training pool")
    ap.add_argument("--val-questions", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    vidmap = load_vidmap(a.root)
    train = load_rows(a.root, "train")
    vids = sorted({r["video"] for r in train})
    chosen = set(rng.sample(vids, a.train_videos))
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        if r["video"] in chosen:
            by_fam[FAMILY[r["type"]]].append(r)
    pool = []
    for fam, rs in sorted(by_fam.items()):
        rng.shuffle(rs)
        pool.extend(rs[: a.per_family])
    rng.shuffle(pool)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "nextqa_train_pool.jsonl").open("w") as f:
        for r in pool:
            rec = {"problem_id": f"nq_{r['video']}_{r['qid']}", "video": r["video"], "path": str(video_path(a.root, r["video"], vidmap)), "question": r["question"], "options": [r[f"a{i}"] for i in range(5)], "answer_idx": int(r["answer"]), "type_code": r["type"], "family": FAMILY[r["type"]]}
            f.write(json.dumps(rec) + "\n")
    val = load_rows(a.root, "val")
    vf: dict[str, list[dict]] = defaultdict(list)
    for r in val:
        vf[FAMILY[r["type"]]].append(r)
    k = a.val_questions // 3
    ids = []
    for fam, rs in sorted(vf.items()):
        rng.shuffle(rs)
        ids.extend(f"{r['video']}_{r['qid']}" for r in rs[:k])
    (a.out / "nextqa_val_ids.json").write_text(json.dumps(sorted(ids)))
    print("train pool:", len(pool), dict(Counter(r["family"] for r in [{"family": FAMILY[r["type"]]} for r in pool])), "| videos:", len({r["video"] for r in pool}))
    print("val ids:", len(ids), "| videos:", len({i.rsplit('_', 1)[0] for i in ids}))


if __name__ == "__main__":
    main()
