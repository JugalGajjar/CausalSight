"""NExT-QA (Xiao et al., CVPR 2021): real-video multiple-choice QA, 5 options, one correct.

Layout: <root>/repo/dataset/nextqa/{train,val,test}.csv, <root>/repo/dataset/nextqa/map_vid_vidorID.json,
videos at <root>/NExTVideo/<vidor-folder>/<video>.mp4. Question types: CW/CH causal (why/how), TN/TC/TP
temporal, DC/DL/DO descriptive; we report the three families. mc_mode 'option' yields one yes/no item per
option (chain models), per-question = all five options judged correctly.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

from causalsight.eval.benchmarks import Benchmark, Item
from causalsight.eval.benchmarks.answers import LETTERS, normalize_short, parse_letters

FAMILY = {"CW": "causal", "CH": "causal", "TN": "temporal", "TC": "temporal", "TP": "temporal", "DC": "descriptive", "DL": "descriptive", "DO": "descriptive"}
MC_PROMPT = "{q}?\n{opts}\nExactly one option is correct. Answer with the letter of the correct option."
OPTION_PROMPT = "{q}?\nOption {letter}: {choice}\nIs this option correct? Answer yes or no."


def load_rows(root: Path, split: str) -> list[dict]:
    with (root / "repo" / "dataset" / "nextqa" / f"{split}.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def video_path(root: Path, vid: str, vidmap: dict) -> Path:
    return root / "NExTVideo" / f"{vidmap[vid]}.mp4"


def load_vidmap(root: Path) -> dict:
    return json.loads((root / "repo" / "dataset" / "nextqa" / "map_vid_vidorID.json").read_text())


class NextQABenchmark(Benchmark):
    name = "nextqa"

    def __init__(self, root: Path, mc_mode: str = "multi", ids: set[str] | None = None) -> None:
        self.root = Path(root)
        self.mc_mode = mc_mode
        self.ids = ids  # optional subset of "<video>_<qid>" keys

    def items(self, split: str = "val", limit: int | None = None) -> Iterator[Item]:
        vidmap = load_vidmap(self.root)
        n = 0
        for row in load_rows(self.root, split):
            key = f"{row['video']}_{row['qid']}"
            if self.ids is not None and key not in self.ids:
                continue
            opts = [row[f"a{i}"] for i in range(5)]
            gold = LETTERS[int(row["answer"])]
            vp = video_path(self.root, row["video"], vidmap)
            fam = FAMILY.get(row["type"], row["type"].lower())
            meta = {"qid": key, "type_code": row["type"], "frames_key": row["video"]}
            if self.mc_mode == "option":
                for i, o in enumerate(opts):
                    yield Item(f"{key}_{i}", fam, vp, OPTION_PROMPT.format(q=row["question"], letter=LETTERS[i], choice=o), "yes" if LETTERS[i] == gold else "no", meta={**meta, "qset": key, "option": i, "per_option": True})
            else:
                opts_txt = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
                yield Item(key, fam, vp, MC_PROMPT.format(q=row["question"], opts=opts_txt), gold, options=opts, meta=meta)
            n += 1
            if limit and n >= limit:
                return

    def score(self, item: Item, prediction: str) -> dict:
        if item.meta.get("per_option"):
            pred = normalize_short(prediction)
            if pred.isdigit():
                pred = "no" if pred == "0" else "yes"
            return {"pred": pred, "correct": int(pred == item.gold)}
        pred = parse_letters(prediction, len(item.options))
        return {"pred": pred, "correct": int(pred == item.gold)}

    @staticmethod
    def summarize(rows: list[dict]) -> dict:
        out: dict[str, dict] = {}
        per_q: dict[str, int] = {}
        fam: dict[str, str] = {}
        for r in rows:
            key = r.get("qset") or r["id"]
            per_q[key] = min(per_q.get(key, 1), r["correct"])
            fam[key] = r["question_type"]
        for t in sorted(set(fam.values())) + ["ALL"]:
            qs = [q for q in per_q if t == "ALL" or fam[q] == t]
            d = {"n": len(qs), "acc": sum(per_q[q] for q in qs) / max(1, len(qs))}
            if rows and rows[0].get("per_option"):
                rs = [r for r in rows if t == "ALL" or r["question_type"] == t]
                d["per_option"] = sum(r["correct"] for r in rs) / len(rs)
            out[t] = d
        return out
