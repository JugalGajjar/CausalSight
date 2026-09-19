"""Recompute chain-level metrics from saved per-item outputs (no model needed).

  cs-rescore-chains <results_dir> --tag sft3b --chains <chains.jsonl> [--condition plain]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalsight.eval.harness import chain_metrics
from causalsight.eval.masking import EvidenceIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chains", type=Path, required=True)
    ap.add_argument("--condition", default="plain")
    a = ap.parse_args()
    idx = EvidenceIndex.from_chains(a.chains)
    rows = [json.loads(line) for line in (a.dir / f"{a.tag}_{a.condition}.jsonl").open()]
    out: dict[str, dict] = {}
    for t in sorted({r["question_type"] for r in rows}) + ["ALL"]:
        rs = [r for r in rows if t == "ALL" or r["question_type"] == t]
        ms = [chain_metrics(r["raw"], idx.get_for_item(r["id"], r.get("qid", r["id"]))) for r in rs]
        d = {"n": len(rs), "format_rate": sum(m["chain_ok"] for m in ms) / len(ms)}
        for k in ("ground_st", "ground_spatial", "ground_recall", "t_offset"):
            vals = [m[k] for m in ms if k in m]
            if vals:
                d[k] = round(sum(vals) / len(vals), 4)
        out[t] = d
    print(json.dumps(out, indent=1))
    (a.dir / f"{a.tag}_{a.condition}.chain_rescored.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
