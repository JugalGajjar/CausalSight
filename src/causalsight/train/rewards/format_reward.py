"""R_fmt: parseable triplets, boxes inside the frame, spans inside the clip, deps point backward."""

from __future__ import annotations

from causalsight.data.schema import TripletChain


def score(chain: TripletChain | None, num_frames: int) -> float:
    if chain is None:
        return 0.0
    try:
        chain.validate()
    except ValueError:
        return 0.0
    if any(t.evidence.t_end >= num_frames for t in chain.triplets):
        return 0.0
    return 1.0
