"""Faithfulness summary from the four Stage 0 runs (plain / blind / mask_evidence / mask_random).

  cs-faithfulness results/stage0 --tag qwen3b

Blind Gap            = acc(plain) - acc(blind)
Evidence Sensitivity = flip rate among items correct in plain: masked-evidence vs masked-random;
                       ES gap = ES(evidence) - ES(random)
Track gap            = same with the evidence objects removed for the whole video vs other objects
                       removed (mask_track / mask_track_random), the stronger CLEVRER intervention
All numbers per question type and overall, computed on the intersection of item ids, with paired
bootstrap 95% CIs (2000 resamples over items) for the two gaps.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

N_BOOT = 2000


def _ci(values_fn, tids: list[str], rng: random.Random, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Paired bootstrap over item ids: resample items, recompute the statistic, take the 2.5/97.5 percentiles."""
    stats = []
    n = len(tids)
    for _ in range(n_boot):
        sample = [tids[rng.randrange(n)] for _ in range(n)]
        v = values_fn(sample)
        if v is not None:
            stats.append(v)
    if not stats:
        return (float("nan"), float("nan"))
    stats.sort()
    return (stats[int(0.025 * len(stats))], stats[min(len(stats) - 1, int(0.975 * len(stats)))])


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(line) for line in path.open())}


def report(dir_: Path, tag: str) -> dict:
    runs = {k: load(dir_ / f"{tag}_{k}.jsonl") for k in ("plain", "blind", "mask_evidence", "mask_random", "mask_track", "mask_track_random") if (dir_ / f"{tag}_{k}.jsonl").exists()}
    if "plain" not in runs:
        raise SystemExit(f"missing {tag}_plain.jsonl in {dir_}")
    # track runs skip items without a same-size control; main stats use the non-track intersection,
    # track stats use the sub-intersection that includes the track runs
    main_keys = [k for k in runs if not k.startswith("mask_track")]
    ids = set.intersection(*(set(runs[k]) for k in main_keys))
    track_ids = set.intersection(ids, *(set(runs[k]) for k in runs if k.startswith("mask_track"))) if any(k.startswith("mask_track") for k in runs) else set()
    by_type: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_type[runs["plain"][i]["question_type"]].append(i)
    by_type["ALL"] = list(ids)
    out: dict[str, dict] = {}
    rng = random.Random(0)

    def blind_gap(ids_: list[str]) -> float:
        return sum(runs["plain"][i]["correct"] - runs["blind"][i]["correct"] for i in ids_) / len(ids_)

    def flip(k: str, ids_: list[str]) -> float | None:
        cp = [i for i in ids_ if runs["plain"][i]["correct"]]
        return None if not cp else sum(1 for i in cp if not runs[k][i]["correct"]) / len(cp)

    def es_gap(ids_: list[str]) -> float | None:
        a, b = flip("mask_evidence", ids_), flip("mask_random", ids_)
        return None if a is None else a - b

    def track_gap(ids_: list[str]) -> float | None:
        a, b = flip("mask_track", ids_), flip("mask_track_random", ids_)
        return None if a is None else a - b

    for t, tids in sorted(by_type.items()):
        acc = {k: sum(runs[k][i]["correct"] for i in tids) / len(tids) for k in runs}
        row = {"n": len(tids), "acc_plain": acc["plain"]}
        if "blind" in acc:
            row["acc_blind"] = acc["blind"]
            row["blind_gap"] = blind_gap(tids)
            row["blind_gap_ci"] = _ci(blind_gap, tids, rng)
        for k in ("mask_evidence", "mask_random"):
            if k in runs:
                v = flip(k, tids)
                if v is not None:
                    row[f"flip_{k[5:]}"] = v
        if "flip_evidence" in row and "flip_random" in row:
            row["es_gap"] = row["flip_evidence"] - row["flip_random"]
            row["es_gap_ci"] = _ci(es_gap, tids, rng)
        ttids = [i for i in tids if i in track_ids]
        if ttids and "mask_track" in runs and "mask_track_random" in runs:
            row["n_track"] = len(ttids)
            a, b = flip("mask_track", ttids), flip("mask_track_random", ttids)
            if a is not None:
                row["flip_track"], row["flip_track_random"] = a, b
                row["track_gap"] = a - b
                row["track_gap_ci"] = _ci(track_gap, ttids, rng)
        out[t] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    rep = report(args.dir, args.tag)
    cols = ["n", "acc_plain", "acc_blind", "blind_gap", "blind_gap_ci", "flip_evidence", "flip_random", "es_gap", "es_gap_ci"]
    if any("track_gap" in r for r in rep.values()):
        cols += ["n_track", "flip_track", "flip_track_random", "track_gap", "track_gap_ci"]

    def fmt(v) -> str:
        if isinstance(v, tuple):
            return f"[{v[0]:.2f},{v[1]:.2f}]"
        return f"{v:.3f}" if isinstance(v, float) else str(v if v is not None else "")

    print(f"{'type':16s}" + "".join(f"{c:>14s}" for c in cols))
    for t, row in rep.items():
        print(f"{t:16s}" + "".join(f"{fmt(row.get(c)):>14s}" for c in cols))
    (args.dir / f"{args.tag}_faithfulness.json").write_text(json.dumps(rep, indent=2))
    (args.dir / f"{args.tag}_faithfulness.md").write_text(
        "| type | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
        + "\n".join("| " + t + " | " + " | ".join(fmt(row.get(c)) for c in cols) + " |" for t, row in rep.items())
        + "\n"
    )


if __name__ == "__main__":
    main()
