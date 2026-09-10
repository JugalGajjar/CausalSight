"""Programmatic CLEVRER triplet generator (proposal Section 5.1, source 1).

Each CLEVRER question ships with a functional program and per-frame object annotations.
Program steps become sub-questions; annotation lookups give intermediate answers, evidence
boxes, and frame spans; program data flow gives dependency links.

Status: skeleton. Fill in `program_to_chain` once the annotation format has been verified
(notes/PLAN.md, Phase A step 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalsight.data.schema import TripletChain


def program_to_chain(question: dict, annotation: dict) -> TripletChain:
    """Convert one CLEVRER question (with program) plus its video annotation into a chain."""
    raise NotImplementedError("verify CLEVRER annotation schema first; see notes/PLAN.md Phase C")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate CLEVRER triplet chains")
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    questions = json.loads(args.questions.read_text())
    n = 0
    with args.out.open("w") as f:
        for video in questions:
            ann_path = args.annotations / f"annotation_{video['scene_index']:05d}.json"
            annotation = json.loads(ann_path.read_text())
            for q in video["questions"]:
                chain = program_to_chain({**q, "video": video}, annotation)
                chain.validate()
                f.write(chain.to_json() + "\n")
                n += 1
                if args.limit and n >= args.limit:
                    return


if __name__ == "__main__":
    main()
