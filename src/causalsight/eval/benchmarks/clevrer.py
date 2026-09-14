"""CLEVRER loader. Reports accuracy per question type; multiple-choice types report both
per-option and per-question accuracy, following the official protocol.

Videos: data/raw/clevrer/videos/<split>/video_<split>/video_XXXXX-YYYYY/video_XXXXX.mp4
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from causalsight.eval.benchmarks import Benchmark, Item
from causalsight.eval.benchmarks.answers import LETTERS, normalize_short, parse_letters

DESCRIPTIVE_PROMPT = {
    "exist": "{q}\nAnswer with yes or no.",
    "count": "{q}\nAnswer with a number.",
    "query_color": "{q}\nAnswer with a single word.",
    "query_material": "{q}\nAnswer with a single word.",
    "query_shape": "{q}\nAnswer with a single word.",
}
MC_PROMPT = (
    "{q}\n{opts}\nOne or more options may be correct. Answer with the letters of all correct options, separated by commas. "
    "If none are correct, answer 'none'."
)
MC_TYPES = ("explanatory", "predictive", "counterfactual")
OPTION_PROMPT = "{q}\nOption {letter}: {choice}\nIs this option correct? Answer yes or no."


def find_video(root: Path, split: str, scene_index: int) -> Path:
    lo = (scene_index // 1000) * 1000
    return root / "videos" / split / f"video_{split}" / f"video_{lo:05d}-{lo + 1000:05d}" / f"video_{scene_index:05d}.mp4"


class ClevrerBenchmark(Benchmark):
    name = "clevrer"

    def __init__(self, root: Path, mc_mode: str = "multi") -> None:
        """mc_mode: 'multi' lists all options and asks for letters; 'option' yields one yes/no item per
        option (the SFT/RL prompt for chain models; official CLEVRER per-option scoring)."""
        self.root = root
        self.mc_mode = mc_mode

    def items(self, split: str = "validation", limit: int | None = None) -> Iterator[Item]:
        videos = json.loads((self.root / "questions" / f"{split}.json").read_text())
        n = 0
        for v in videos:
            vp = find_video(self.root, split, v["scene_index"])
            for q in v["questions"]:
                t = q["question_type"]
                qid = f"{v['scene_index']}_{q['question_id']}"
                if t == "descriptive":
                    st = q.get("question_subtype", "query_color")
                    yield Item(qid, t, vp, DESCRIPTIVE_PROMPT.get(st, DESCRIPTIVE_PROMPT["query_color"]).format(q=q["question"]), normalize_short(str(q["answer"])), meta={"subtype": st})
                elif self.mc_mode == "option":
                    for i, c in enumerate(q["choices"]):
                        gold = "yes" if c["answer"] == "correct" else "no"
                        yield Item(f"{qid}_{i}", t, vp, OPTION_PROMPT.format(q=q["question"], letter=LETTERS[i], choice=c["choice"]), gold, meta={"qid": qid, "option": i, "per_option": True})
                else:
                    opts = [c["choice"] for c in q["choices"]]
                    gold = "".join(LETTERS[i] for i, c in enumerate(q["choices"]) if c["answer"] == "correct")
                    opts_txt = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
                    yield Item(qid, t, vp, MC_PROMPT.format(q=q["question"], opts=opts_txt), gold, options=opts)
                n += 1
                if limit and n >= limit:
                    return

    def score(self, item: Item, prediction: str) -> dict:
        if item.question_type == "descriptive" or item.meta.get("per_option"):
            pred = normalize_short(prediction)
            if (item.meta.get("subtype") == "exist" or item.meta.get("per_option")) and pred.isdigit():
                pred = "no" if pred == "0" else "yes"
            return {"pred": pred, "correct": int(pred == item.gold)}
        pred = parse_letters(prediction, len(item.options))
        per_option = [int((L in pred) == (L in item.gold)) for L in LETTERS[: len(item.options)]]
        return {"pred": pred, "correct": int(pred == item.gold), "option_correct": sum(per_option), "option_total": len(per_option)}

    @staticmethod
    def summarize(rows: list[dict]) -> dict:
        out: dict[str, dict] = {}
        for t in ("descriptive",) + MC_TYPES:
            rs = [r for r in rows if r["question_type"] == t]
            if not rs:
                continue
            if rs[0].get("per_option"):
                # per-option rows: per_option = row accuracy; per_question = all options of a question correct
                by_q: dict[str, list[int]] = {}
                for r in rs:
                    by_q.setdefault(r["qid"], []).append(r["correct"])
                d = {"n": len(by_q), "n_options": len(rs), "per_option": sum(r["correct"] for r in rs) / len(rs), "per_question": sum(all(v) for v in by_q.values()) / len(by_q)}
            else:
                d = {"n": len(rs), "per_question": sum(r["correct"] for r in rs) / len(rs)}
                if t in MC_TYPES:
                    d["per_option"] = sum(r["option_correct"] for r in rs) / sum(r["option_total"] for r in rs)
            out[t] = d
        inf = [t for t in MC_TYPES if t in out]
        if inf:
            tot = sum(out[t]["n"] for t in inf)
            out["inferential(per_question)"] = {"n": tot, "per_question": sum(out[t]["per_question"] * out[t]["n"] for t in inf) / tot}
        return out
