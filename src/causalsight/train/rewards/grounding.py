"""R_ground: spatiotemporal IoU against matched reference steps (CLEVRER), or a detector check
that the evidence region contains the entity named in a_i (real video). Size-binned per Ground-R1."""

from __future__ import annotations


def score(*args, **kwargs) -> float:
    raise NotImplementedError
