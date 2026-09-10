"""R_proc: order-aware process reward.

MatchScore uses the H-GRPO Hungarian matching (Sentence-BERT similarity x spatiotemporal IoU),
then subtracts beta * (order violations over reference dependency edges) / (reference edges).
Invariance is allowed only among steps that are independent in the reference DAG.
"""

from __future__ import annotations

from causalsight.data.schema import TripletChain


def order_violations(pred: TripletChain, ref: TripletChain, matching: dict[int, int]) -> tuple[int, int]:
    """Count reference edges (j -> k) whose matched predictions are reversed or missing a dep link.

    `matching` maps reference index -> predicted index. Returns (violations, total_edges).
    """
    violations = 0
    total = 0
    for k, t in enumerate(ref.triplets):
        for j in t.depends_on:
            total += 1
            if j not in matching or k not in matching:
                violations += 1
                continue
            pj, pk = matching[j], matching[k]
            reversed_order = pj >= pk
            missing_link = pj not in pred.triplets[pk].depends_on
            if reversed_order or missing_link:
                violations += 1
    return violations, total


def score(pred: TripletChain, ref: TripletChain, *, beta: float) -> float:
    raise NotImplementedError("needs Sentence-BERT matching; see H-GRPO cost definition")
