"""Assemble the real/synthetic training arms (proposal Section 5, mixed-data experiment).

  cs-mix --clevrer-sft <train pack>/sft_subset.jsonl --clevrer-rl <repo>/data/subsets/rl_balanced_1500_options.jsonl \
         --nextqa-sft <pseudo>/nextqa_sft.jsonl --nextqa-rl <pseudo>/nextqa_rl.jsonl \
         --clevrer-frames /content/data/train --nextqa-frames /content/data/nextqa_pack --out <dir>

Writes, with absolute `frames` paths so one trainer run can read both packs:
  mixed_chains_sft.jsonl   4,400 CLEVRER chains + 8,000 NExT-QA chains (yes/no balanced)
  mixed_plain_sft.jsonl    the same questions with bare <answer> responses (real data without chains)
  real_chains_sft.jsonl    the 8,000 NExT-QA chains only
  mixed_rl.jsonl           750 CLEVRER + 750 NExT-QA per-option RL records (with gt_chains)
  real_rl.jsonl            1,500 NExT-QA per-option RL records
NExT-QA evidence spans (0-15 sampled-frame grid) are rescaled to the 0-127 position scale CLEVRER uses,
so "t" means the same thing in every record; spans get +-4 positions.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from causalsight.data.schema import TripletChain
from causalsight.train.format import format_chain


def _rescale_chain(chain_d: dict) -> dict:
    for t in chain_d["triplets"]:
        e = t["evidence"]
        c = round(e["t_start"] * 127 / 15)
        e["t_start"], e["t_end"] = max(0, c - 4), min(127, c + 4)
    return chain_d


def _abs_frames(rec: dict, root: Path) -> dict:
    rec = dict(rec)
    rec["frames"] = str(root / rec["frames"])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clevrer-sft", type=Path, required=True)
    ap.add_argument("--clevrer-rl", type=Path, required=True)
    ap.add_argument("--nextqa-sft", type=Path, required=True)
    ap.add_argument("--nextqa-rl", type=Path, required=True)
    ap.add_argument("--clevrer-frames", type=Path, required=True)
    ap.add_argument("--nextqa-frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clevrer-n", type=int, default=4400)
    ap.add_argument("--nextqa-n", type=int, default=8000)
    ap.add_argument("--rl-per-source", type=int, default=750)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    csft = [json.loads(line) for line in a.clevrer_sft.open()]
    rng.shuffle(csft)
    csft = [_abs_frames(r, a.clevrer_frames) for r in csft[: a.clevrer_n]]

    nsft_all = [json.loads(line) for line in a.nextqa_sft.open()]
    by_gold = {"yes": [r for r in nsft_all if r["gold"] == "yes"], "no": [r for r in nsft_all if r["gold"] == "no"]}
    nsft = []
    for g in ("yes", "no"):
        rng.shuffle(by_gold[g])
        nsft.extend(by_gold[g][: a.nextqa_n // 2])
    rng.shuffle(nsft)
    fixed = []
    for r in nsft:
        r = _abs_frames(r, a.nextqa_frames)
        r["chain"] = _rescale_chain(r["chain"])
        r["response"] = format_chain(TripletChain.from_json(json.dumps(r["chain"])))
        fixed.append(r)
    nsft = fixed

    def write(name: str, rows: list[dict]) -> None:
        with (a.out / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)}", dict(Counter(r.get("question_type") for r in rows)))

    mixed = csft + nsft
    rng.shuffle(mixed)
    write("mixed_chains_sft.jsonl", mixed)
    plain = []
    for r in mixed:
        r = dict(r)
        r["response"] = f"<answer>{r['gold']}</answer>"
        r.pop("chain", None)
        plain.append(r)
    write("mixed_plain_sft.jsonl", plain)
    write("real_chains_sft.jsonl", nsft)

    crl = [json.loads(line) for line in a.clevrer_rl.open()]
    rng.shuffle(crl)
    crl = [_abs_frames(r, a.clevrer_frames) for r in crl[: a.rl_per_source]]
    nrl_all = [json.loads(line) for line in a.nextqa_rl.open()]
    rng.shuffle(nrl_all)
    nrl = []
    for r in nrl_all[: 2 * a.rl_per_source]:
        r = _abs_frames(r, a.nextqa_frames)
        for c in r["gt_chains"]:
            _rescale_chain(c)
        nrl.append(r)
    mixed_rl = crl + nrl[: a.rl_per_source]
    rng.shuffle(mixed_rl)
    write("mixed_rl.jsonl", mixed_rl)
    write("real_rl.jsonl", nrl)


if __name__ == "__main__":
    main()
