"""Stage 2 reward terms that compare a generated chain with the question's ground-truth chains
(record field `gt_chains`, attached by `cs-export attach-chains`). All return floats in [0, 1].

grounding : mean over emitted steps of the best spatiotemporal IoU with any GT evidence region of the
            question (R_ground, proposal Section 5.2). 0 if the chain does not parse.
process   : order-aware matching (R_proc). Steps are matched to the best-scoring GT chain by
            similarity = answer-token Jaccard x (0.5 + 0.5 x evidence IoU) via a greedy assignment
            (Hungarian-equivalent for these small sets in practice), then a penalty is subtracted for
            every GT dependency edge whose matched predictions appear in the wrong order or lack the link.
"""

from __future__ import annotations

import re

from causalsight.data.schema import Evidence, TripletChain
from causalsight.train.format import parse_chain
from causalsight.train.rewards.process import order_violations

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(s.lower())) - {"the", "a", "an", "at", "of", "and", "frame"}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0


def _gt_chain(d: dict) -> TripletChain:
    return TripletChain.from_json(__import__("json").dumps({"video_id": "", "question": "", "final_answer": d["final_answer"], "question_type": d.get("question_type"), "triplets": d["triplets"]}))


def gt_regions(rec: dict) -> list[Evidence]:
    out = []
    for c in rec.get("gt_chains", []):
        for t in c["triplets"]:
            e = t["evidence"]
            if tuple(e["box"]) != (0.0, 0.0, 1.0, 1.0):
                out.append(Evidence(e["t_start"], e["t_end"], tuple(e["box"])))
    return out


def grounding(text: str, rec: dict) -> float:
    p = parse_chain(text)
    regs = gt_regions(rec)
    if p.chain is None or not regs:
        return 0.0
    return sum(max(t.evidence.st_iou(g) for g in regs) for t in p.chain.triplets) / len(p.chain.triplets)


def match(pred: TripletChain, ref: TripletChain) -> tuple[dict[int, int], float]:
    """Greedy one-to-one matching ref index -> pred index by descending similarity. Returns the
    matching and the mean similarity over ref steps (unmatched ref steps count 0)."""
    sims = []
    for j, r in enumerate(ref.triplets):
        for i, t in enumerate(pred.triplets):
            sim = _jaccard(t.answer, r.answer) * (0.5 + 0.5 * t.evidence.st_iou(r.evidence))
            sims.append((sim, j, i))
    sims.sort(reverse=True)
    used_p: set[int] = set()
    m: dict[int, int] = {}
    total = 0.0
    for sim, j, i in sims:
        if j in m or i in used_p or sim <= 0:
            continue
        m[j] = i
        used_p.add(i)
        total += sim
    return m, total / max(1, len(ref.triplets))


def process(text: str, rec: dict, beta: float = 0.5) -> float:
    p = parse_chain(text)
    refs = rec.get("gt_chains", [])
    if p.chain is None or not refs:
        return 0.0
    best = 0.0
    for d in refs:
        if d.get("choice_id") is not None and rec.get("problem_type") == "multiple choice":
            pass  # every option chain is a valid reference for the question; take the best
        ref = _gt_chain(d)
        m, score = match(p.chain, ref)
        v, tot = order_violations(p.chain, ref, m)
        pen = beta * (v / tot) if tot else 0.0
        best = max(best, max(0.0, score - pen))
    return best
