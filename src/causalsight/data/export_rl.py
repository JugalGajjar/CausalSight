"""Export CLEVRER into training files.

  cs-export rl  --split train --out data/rl/clevrer_train_rl.jsonl [--n-videos N]
      One record per question, Video-R1-compatible fields (problem, problem_type, data_type, path, options,
      solution=<answer>..</answer>) plus our own (question_type, subtype, gold). MC solutions hold the
      letters of ALL correct options (CLEVRER is multi-select), e.g. <answer>AC</answer>.

  cs-export sft --chains data/triplets/clevrer_train.jsonl --split train --out data/sft/clevrer_train_sft.jsonl
      One record per chain: prompt (same wording as the RL problem), response in the chain text format,
      video path, and the chain itself for the bridge-loss corruption step.

Video paths are relative to the repo root; the trainer resolves them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from causalsight.data.schema import TripletChain
from causalsight.eval.benchmarks.answers import LETTERS, normalize_short
from causalsight.eval.benchmarks.clevrer import DESCRIPTIVE_PROMPT, MC_PROMPT, find_video
from causalsight.train.format import format_chain

PROBLEM_TYPE = {"count": "numerical", "exist": "free-form", "query_color": "free-form", "query_material": "free-form", "query_shape": "free-form"}


def question_record(root: Path, split: str, scene_index: int, q: dict) -> dict:
    t = q["question_type"]
    vp = find_video(root, split, scene_index)
    rec = {
        "problem_id": f"{scene_index}_{q['question_id']}",
        "data_type": "video",
        "path": str(vp),
        "question_type": t,
        "subtype": q.get("question_subtype"),
        "data_source": "CLEVRER",
    }
    if t == "descriptive":
        st = q.get("question_subtype", "query_color")
        gold = normalize_short(str(q["answer"]))
        rec.update(problem=DESCRIPTIVE_PROMPT[st].format(q=q["question"]), problem_type=PROBLEM_TYPE[st], options=[], gold=gold, solution=f"<answer>{gold}</answer>")
    else:
        opts = [c["choice"] for c in q["choices"]]
        gold = "".join(LETTERS[i] for i, c in enumerate(q["choices"]) if c["answer"] == "correct")
        opts_txt = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        rec.update(problem=MC_PROMPT.format(q=q["question"], opts=opts_txt), problem_type="multiple choice", options=[f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)], gold=gold, solution=f"<answer>{gold}</answer>")
    return rec


def export_rl(root: Path, split: str, out: Path, n_videos: int | None) -> Counter:
    videos = json.loads((root / "questions" / f"{split}.json").read_text())
    if n_videos:
        videos = videos[:n_videos]
    stats = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for v in videos:
            for q in v["questions"]:
                rec = question_record(root, split, v["scene_index"], q)
                stats[rec["question_type"]] += 1
                f.write(json.dumps(rec) + "\n")
    return stats


def export_sft(root: Path, split: str, chains: Path, out: Path, limit: int | None) -> Counter:
    """Descriptive chains map 1:1 to questions. MC chains are per option; the SFT record asks about that
    single option ("Is option X correct?") so the chain's yes/no final answer matches the prompt."""
    videos = {v["scene_index"]: v for v in json.loads((root / "questions" / f"{split}.json").read_text())}
    stats = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with chains.open() as fin, out.open("w") as f:
        for line in fin:
            c = TripletChain.from_json(line)
            si, qid = c.meta["scene_index"], c.meta["question_id"]
            q = next(x for x in videos[si]["questions"] if x["question_id"] == qid)
            base = question_record(root, split, si, q)
            if c.question_type == "descriptive":
                prompt = base["problem"]
            else:
                letter = LETTERS[c.meta["choice_id"]]
                prompt = f"{q['question']}\nOption {letter}: {c.meta['choice']}\nIs this option correct? Answer yes or no."
            rec = {
                "problem_id": base["problem_id"] + (f"_{c.meta['choice_id']}" if "choice_id" in c.meta else ""),
                "data_type": "video",
                "path": base["path"],
                "question_type": c.question_type,
                "subtype": base["subtype"],
                "problem": prompt,
                "response": format_chain(c),
                "gold": c.final_answer,
                "chain": json.loads(c.to_json()),
            }
            stats[c.question_type] += 1
            f.write(json.dumps(rec) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rl", "sft"])
    ap.add_argument("--root", type=Path, default=Path("data/raw/clevrer"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--chains", type=Path, default=None)
    ap.add_argument("--n-videos", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "rl":
        stats = export_rl(args.root, args.split, args.out, args.n_videos)
    else:
        if not args.chains:
            raise SystemExit("--chains required for sft export")
        stats = export_sft(args.root, args.split, args.chains, args.out, args.limit)
    print(dict(stats), "->", args.out)


if __name__ == "__main__":
    main()
