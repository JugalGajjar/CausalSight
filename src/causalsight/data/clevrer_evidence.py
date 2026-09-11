"""Evidence boxes for CLEVRER objects from the derender proposals.

Proposals (derender_proposals/proposal_XXXXX.json) hold per-frame detections with an RLE mask and
predicted (color, material, shape). Attribute triples are unique per video in CLEVRER, so a detection
is matched to a ground-truth object by its triple. Detections are noisy: a frame can miss an object
or carry a spurious triple, so lookups fall back to the nearest frame within a tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from causalsight.data.rle import mask_to_box, rle_to_mask
from causalsight.data.schema import Evidence

Box = tuple[float, float, float, float]
FULL_FRAME: Box = (0.0, 0.0, 1.0, 1.0)


def union_box(boxes: list[Box]) -> Box | None:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


@dataclass
class ProposalIndex:
    """frame -> attr triple -> (box, score), decoded lazily per frame."""

    frames: list[dict]
    _cache: dict[int, dict[tuple[str, str, str], tuple[Box, float]]]

    @classmethod
    def load(cls, path: Path | str) -> ProposalIndex:
        d = json.loads(Path(path).read_text())
        frames = sorted(d["frames"], key=lambda f: f["frame_index"])
        return cls(frames=frames, _cache={})

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def _frame(self, f: int) -> dict[tuple[str, str, str], tuple[Box, float]]:
        if f in self._cache:
            return self._cache[f]
        out: dict[tuple[str, str, str], tuple[Box, float]] = {}
        if 0 <= f < len(self.frames):
            for o in self.frames[f].get("objects", []):
                key = (o.get("color"), o.get("material"), o.get("shape"))
                score = float(o.get("score", 0.0))
                if key in out and out[key][1] >= score:
                    continue
                box = mask_to_box(rle_to_mask(o["mask"]))
                if box is not None:
                    out[key] = (box, score)
        self._cache[f] = out
        return out

    def box(self, attrs: dict[str, str], frame: int, tol: int = 3) -> tuple[Box, int] | None:
        """Box for the object with `attrs` at `frame`, or at the nearest frame within +-tol.

        Returns (box, frame_used) or None if the object is not detected in the window.
        """
        key = (attrs["color"], attrs["material"], attrs["shape"])
        for d in range(tol + 1):
            for f in ((frame - d, frame + d) if d else (frame,)):
                hit = self._frame(f).get(key)
                if hit:
                    return hit[0], f
        return None

    def detected_frames(self, attrs: dict[str, str]) -> list[int]:
        key = (attrs["color"], attrs["material"], attrs["shape"])
        return [f for f in range(len(self.frames)) if key in self._frame(f)]


def evidence_for_objects(
    idx: ProposalIndex, attrs_list: list[dict[str, str]], frame: int, span: tuple[int, int] | None = None, tol: int = 3
) -> Evidence | None:
    """Union box of the given objects at `frame` (nearest detections), with span defaulting to [frame, frame]."""
    boxes = []
    for a in attrs_list:
        hit = idx.box(a, frame, tol)
        if hit is None:
            return None
        boxes.append(hit[0])
    u = union_box(boxes)
    if u is None:
        return None
    t0, t1 = span if span else (frame, frame)
    t0 = max(0, min(t0, idx.n_frames - 1))
    t1 = max(t0, min(t1, idx.n_frames - 1))
    return Evidence(t_start=t0, t_end=t1, box=u)


def whole_video_evidence(idx: ProposalIndex) -> Evidence:
    return Evidence(t_start=0, t_end=idx.n_frames - 1, box=FULL_FRAME)
