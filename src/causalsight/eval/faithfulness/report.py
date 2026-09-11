"""Faithfulness summary from the four Stage 0 runs (plain / blind / mask_evidence / mask_random).

  cs-faithfulness results/stage0 --tag qwen3b

Blind Gap            = acc(plain) - acc(blind)
Evidence Sensitivity = flip rate among items correct in plain: masked-evidence vs masked-random;
                       ES gap = ES(evidence) - ES(random)
All numbers per question type and overall, computed on the intersection of item ids.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(line) for line in path.open())}


def report(dir_: Path, tag: str) -> dict:
    runs = {k: load(dir_ / f"{tag}_{k}.jsonl") for k in ("plain", "blind", "mask_evidence", "mask_random") if (dir_ / f"{tag}_{k}.jsonl").exists()}
    if "plain" not in runs:
        raise SystemExit(f"missing {tag}_plain.jsonl in {dir_}")
    ids = set.intersection(*(set(r) for r in runs.values()))
    by_type: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_type[runs["plain"][i]["question_type"]].append(i)
    by_type["ALL"] = list(ids)
    out: dict[str, dict] = {}
    for t, tids in sorted(by_type.items()):
        acc = {k: sum(runs[k][i]["correct"] for i in tids) / len(tids) for k in runs}
        row = {"n": len(tids), "acc_plain": acc["plain"]}
        if "blind" in acc:
            row["acc_blind"] = acc["blind"]
            row["blind_gap"] = acc["plain"] - acc["blind"]
        correct_plain = [i for i in tids if runs["plain"][i]["correct"]]
        for k in ("mask_evidence", "mask_random"):
            if k in runs and correct_plain:
                flips = sum(1 for i in correct_plain if not runs[k][i]["correct"]) / len(correct_plain)
                row[f"flip_{k.split('_')[1]}"] = flips
        if "flip_evidence" in row and "flip_random" in row:
            row["es_gap"] = row["flip_evidence"] - row["flip_random"]
        out[t] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    rep = report(args.dir, args.tag)
    cols = ["n", "acc_plain", "acc_blind", "blind_gap", "flip_evidence", "flip_random", "es_gap"]
    print(f"{'type':16s}" + "".join(f"{c:>14s}" for c in cols))
    for t, row in rep.items():
        print(f"{t:16s}" + "".join(f"{row[c]:14.3f}" if isinstance(row.get(c), float) else f"{row.get(c, '')!s:>14s}" for c in cols))
    (args.dir / f"{args.tag}_faithfulness.json").write_text(json.dumps(rep, indent=2))
    (args.dir / f"{args.tag}_faithfulness.md").write_text(
        "| type | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
        + "\n".join("| " + t + " | " + " | ".join(f"{row[c]:.3f}" if isinstance(row.get(c), float) else str(row.get(c, "")) for c in cols) + " |" for t, row in rep.items())
        + "\n"
    )


if __name__ == "__main__":
    main()
