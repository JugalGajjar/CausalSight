"""Summarize a cs-grpo log: reward, format, KL, completion length and skip rate per window.

  cs-trainlog runs/stage0/log.jsonl [--window 100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path, window: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    out = []
    for s in range(0, len(rows), window):
        w = rows[s : s + window]
        upd = [r for r in w if not r.get("skipped")]
        rk = [k for k in w[0] if k.startswith("r_")]
        d = {
            "steps": f"{w[0]['step']}-{w[-1]['step']}",
            "n": len(w),
            "skip_rate": round(1 - len(upd) / len(w), 3),
            "reward": round(sum(r["reward_mean"] for r in w) / len(w), 3),
            **{k: round(sum(r[k] for r in w) / len(w), 3) for k in rk},
            "comp_len": round(sum(r["comp_len"] for r in w) / len(w), 1),
            "kl": round(sum(r["kl"] for r in upd) / len(upd), 4) if upd else 0.0,
            "grad_norm": round(sum(r.get("grad_norm", 0.0) for r in upd) / len(upd), 2) if upd else 0.0,
            "sec_per_step": round((w[-1]["sec"] - w[0]["sec"]) / max(1, len(w) - 1), 1),
        }
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--window", type=int, default=100)
    a = ap.parse_args()
    rows = summarize(a.log, a.window)
    cols = list(rows[0].keys())
    print("".join(f"{c:>14s}" for c in cols))
    for r in rows:
        print("".join(f"{r[c]!s:>14s}" for c in cols))


if __name__ == "__main__":
    main()
