"""Frame-consistent grounding: at the frames a step cites, does its box cover the object(s) it names?

The GT chains cite one arbitrary frame per object-resolution step (the largest-box frame), so IoU against
that region penalizes a model for citing a different, equally valid frame. Here each emitted step is
checked against the detector's boxes for the objects named in its answer, at the step's own frames.
"""

from __future__ import annotations

import re

from causalsight.data.clevrer_evidence import ProposalIndex, union_box
from causalsight.data.clevrer_sim import COLORS, MATERIALS, SHAPES
from causalsight.data.schema import Evidence, TripletChain

NAME_RE = re.compile(rf"\b({'|'.join(COLORS)}) ({'|'.join(MATERIALS)}) ({'|'.join(SHAPES)})\b")


def named_objects(answer: str) -> list[dict[str, str]]:
    seen, out = set(), []
    for c, m, s in NAME_RE.findall(answer.lower()):
        if (c, m, s) not in seen:
            seen.add((c, m, s))
            out.append({"color": c, "material": m, "shape": s})
    return out


def step_scores(chain: TripletChain, idx: ProposalIndex, tol: int = 3) -> list[dict]:
    """One dict per step that names at least one fully specified object:
    iou (emitted box vs union of the named objects' detected boxes at the cited frames), visible (all named
    objects detected within `tol` frames of the cited span)."""
    out = []
    for k, t in enumerate(chain.triplets):
        objs = named_objects(t.answer)
        if not objs:
            continue
        e = t.evidence
        mid = (e.t_start + e.t_end) // 2
        half = (e.t_end - e.t_start) // 2 + tol
        boxes = []
        for a in objs:
            hit = idx.box(a, mid, half)
            if hit is not None:
                boxes.append(hit[0])
        visible = len(boxes) == len(objs)
        iou = 0.0
        if boxes:
            ref = Evidence(e.t_start, e.t_end, union_box(boxes))
            iou = e.spatial_iou(ref)
        out.append({"index": k, "iou": iou, "visible": visible, "n_objects": len(objs)})
    return out
