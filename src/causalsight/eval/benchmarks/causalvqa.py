"""CausalVQA (Foss et al., NeurIPS 2025 D&B), real-video causal QA from Ego-Exo4D clips.

Layout: <root>/CausalVQA/{debug,test}/{<split>_metadata.csv, videos/<renamed_video>}. Only the debug
split carries answers (99 questions), so that is the scored split. Each question has two choice sets
(choices1/correct1 and choices2/correct2, a perturbed re-ordering); each set is one item, and a question
counts as paired-correct only if both are answered correctly (the benchmark's own metric). Five options,
exactly one correct. Types: Anticipation, Counterfactual, Hypothetical, Planning, Descriptive.

mc_mode 'multi': one item per choice set, answer = letter. mc_mode 'option': one yes/no item per option
(chain models), per-question = all five options judged correctly.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path

from causalsight.eval.benchmarks import Benchmark, Item
from causalsight.eval.benchmarks.answers import LETTERS, normalize_short, parse_letters

MC_PROMPT = "{q}\n{opts}\nExactly one option is correct. Answer with the letter of the correct option."
OPTION_PROMPT = "{q}\nOption {letter}: {choice}\nIs this option correct? Answer yes or no."


def _split_choices(s: str) -> list[str]:
    parts = [p.strip() for p in s.split("|")]
    return [re.sub(r"^\d+\.\s*", "", p) for p in parts if p]


class CausalVQABenchmark(Benchmark):
    name = "causalvqa"

    def __init__(self, root: Path, mc_mode: str = "multi") -> None:
        self.root = Path(root)
        self.mc_mode = mc_mode

    def items(self, split: str = "debug", limit: int | None = None) -> Iterator[Item]:
        d = self.root / "CausalVQA" / split
        n = 0
        with (d / f"{split}_metadata.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                q = row["question"].strip().strip('"')
                vp = d / "videos" / row["renamed_video"]
                for k in ("1", "2"):
                    opts = _split_choices(row[f"choices{k}"])
                    correct = row.get(f"correct{k}", "").strip()
                    if not correct:
                        continue  # test split: answers removed
                    gold = LETTERS[int(correct) - 1]
                    qid = f"{row['qid']}_{k}"
                    meta = {"qid": row["qid"], "set": k, "difficulty": row.get("difficulty")}
                    if self.mc_mode == "option":
                        for i, o in enumerate(opts):
                            yield Item(f"{qid}_{i}", row["type"].lower(), vp, OPTION_PROMPT.format(q=q, letter=LETTERS[i], choice=o), "yes" if LETTERS[i] == gold else "no", meta={**meta, "qset": qid, "option": i, "per_option": True})
                    else:
                        opts_txt = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
                        yield Item(qid, row["type"].lower(), vp, MC_PROMPT.format(q=q, opts=opts_txt), gold, options=opts, meta=meta)
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
        per_set: dict[str, int] = {}  # choice-set id -> correct (all options in option mode)
        for r in rows:
            key = r.get("qset") or r["id"]
            per_set[key] = min(per_set.get(key, 1), r["correct"])
        by_q: dict[str, list[int]] = {}
        types: dict[str, str] = {}
        for r in rows:
            key = r.get("qset") or r["id"]
            by_q.setdefault(r["qid"], []).append(per_set[key])
            types[r["qid"]] = r["question_type"]
        # de-duplicate sets per question
        by_q = {q: [per_set[f"{q}_1"] if f"{q}_1" in per_set else 0, per_set[f"{q}_2"] if f"{q}_2" in per_set else 0] for q in by_q}
        for t in sorted(set(types.values())) + ["ALL"]:
            qs = [q for q in by_q if t == "ALL" or types[q] == t]
            sets = [v for q in qs for v in by_q[q]]
            d = {"n_questions": len(qs), "n_sets": len(sets), "set_acc": sum(sets) / max(1, len(sets)), "paired_acc": sum(all(by_q[q]) for q in qs) / max(1, len(qs))}
            if rows and rows[0].get("per_option"):
                rs = [r for r in rows if t == "ALL" or r["question_type"] == t]
                d["per_option"] = sum(r["correct"] for r in rs) / len(rs)
            out[t] = d
        return out
