"""Triplet chain schema (proposal Section 5.1).

A chain is a list of triplets tau_i = <q_i, a_i, e_i, d_i> followed by a final answer.
Evidence e_i is spatiotemporal: a frame span plus a box in normalized [0, 1] coordinates.
d_i lists the indices of earlier triplets this step depends on, making the chain a DAG.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Evidence:
    t_start: int
    t_end: int
    box: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max, normalized

    def validate(self) -> None:
        if self.t_start < 0 or self.t_end < self.t_start:
            raise ValueError(f"bad frame span ({self.t_start}, {self.t_end})")
        if len(self.box) != 4:
            raise ValueError("box must have 4 coordinates")
        x0, y0, x1, y1 = self.box
        if not all(0.0 <= v <= 1.0 for v in self.box):
            raise ValueError(f"box not normalized: {self.box}")
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"box has non-positive area: {self.box}")

    def spatial_iou(self, other: "Evidence") -> float:
        ax0, ay0, ax1, ay1 = self.box
        bx0, by0, bx1, by1 = other.box
        iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        ih = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return inter / union if union > 0 else 0.0

    def temporal_iou(self, other: "Evidence") -> float:
        inter = max(0, min(self.t_end, other.t_end) - max(self.t_start, other.t_start) + 1)
        union = (self.t_end - self.t_start + 1) + (other.t_end - other.t_start + 1) - inter
        return inter / union if union > 0 else 0.0

    def st_iou(self, other: "Evidence") -> float:
        """Spatiotemporal IoU used by R_ground and R_proc: temporal-IoU x spatial-IoU."""
        return self.temporal_iou(other) * self.spatial_iou(other)


@dataclass(frozen=True)
class Triplet:
    question: str
    answer: str
    evidence: Evidence
    depends_on: tuple[int, ...] = ()


@dataclass
class TripletChain:
    video_id: str
    question: str
    triplets: list[Triplet]
    final_answer: str
    question_type: str | None = None  # CLEVRER: descriptive / explanatory / predictive / counterfactual
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.triplets:
            raise ValueError("chain has no triplets")
        for i, t in enumerate(self.triplets):
            if not t.question.strip() or not t.answer.strip():
                raise ValueError(f"triplet {i} has empty question or answer")
            t.evidence.validate()
            for d in t.depends_on:
                if not 0 <= d < i:
                    raise ValueError(f"triplet {i} depends on {d}, which is not an earlier step")

    def dependency_closure(self, i: int) -> set[int]:
        """Indices of all steps that transitively depend on step i (used to build c_{-i})."""
        out: set[int] = set()
        frontier = [i]
        while frontier:
            cur = frontier.pop()
            for j, t in enumerate(self.triplets):
                if cur in t.depends_on and j not in out:
                    out.add(j)
                    frontier.append(j)
        return out

    def without_step(self, i: int) -> "TripletChain":
        """Intervened chain c_{-i}: drop step i and everything downstream of it."""
        drop = self.dependency_closure(i) | {i}
        keep = [j for j in range(len(self.triplets)) if j not in drop]
        remap = {old: new for new, old in enumerate(keep)}
        new_triplets = [
            Triplet(
                question=self.triplets[j].question,
                answer=self.triplets[j].answer,
                evidence=self.triplets[j].evidence,
                depends_on=tuple(remap[d] for d in self.triplets[j].depends_on if d in remap),
            )
            for j in keep
        ]
        return TripletChain(
            video_id=self.video_id,
            question=self.question,
            triplets=new_triplets,
            final_answer=self.final_answer,
            question_type=self.question_type,
            meta={**self.meta, "removed_step": i},
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "TripletChain":
        d = json.loads(s)
        triplets = [
            Triplet(
                question=t["question"],
                answer=t["answer"],
                evidence=Evidence(
                    t_start=t["evidence"]["t_start"],
                    t_end=t["evidence"]["t_end"],
                    box=tuple(t["evidence"]["box"]),
                ),
                depends_on=tuple(t.get("depends_on", ())),
            )
            for t in d["triplets"]
        ]
        return cls(
            video_id=d["video_id"],
            question=d["question"],
            triplets=triplets,
            final_answer=d["final_answer"],
            question_type=d.get("question_type"),
            meta=d.get("meta", {}),
        )
