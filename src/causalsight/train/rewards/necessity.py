"""R_nec: step necessity under a frozen verifier p_phi (the Stage 1 checkpoint).

Delta_i = p_phi(a* | v, q, c) - p_phi(a* | v, q, c_{-i})
R_nec   = mean_i 1[Delta_i > delta] - gamma * |{i : Delta_i <= eps}| / N

`TripletChain.without_step(i)` builds c_{-i} including the transitive dependency closure.
"""

from __future__ import annotations

from causalsight.data.schema import TripletChain


def score_from_deltas(deltas: list[float], delta: float, eps: float, gamma: float) -> float:
    n = len(deltas)
    if n == 0:
        return 0.0
    necessary = sum(1 for d in deltas if d > delta) / n
    padding = sum(1 for d in deltas if d <= eps) / n
    return necessary - gamma * padding


def intervened_chains(chain: TripletChain) -> list[TripletChain]:
    return [chain.without_step(i) for i in range(len(chain.triplets))]


def score(chain: TripletChain, verifier, video, question, gold, *, delta, eps, gamma) -> float:
    """`verifier(video, question, chain) -> log p(gold | ...)`. Left abstract until the model code exists."""
    raise NotImplementedError
