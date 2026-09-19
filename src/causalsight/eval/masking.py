"""Evidence masking for the Evidence Sensitivity metric (proposal Section 6.3).

Given sampled frames and a set of spatiotemporal evidence regions, black out each region on the
frames whose original index falls inside its span. The random control places boxes of the same
sizes and spans at random positions, so any accuracy difference is about *where* was masked.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from causalsight.data.schema import Evidence

Region = tuple[int, int, tuple[float, float, float, float]]  # t_start, t_end, normalized box


def sampled_frame_indices(n_total: int, n_sampled: int) -> list[int]:
    if n_total <= n_sampled:
        return list(range(n_total))
    return [round(i * (n_total - 1) / (n_sampled - 1)) for i in range(n_sampled)]


def span_tolerance(n_total: int, n_sampled: int) -> int:
    """Half the sampling stride: a region is applied to the sampled frame(s) nearest its span, so a
    single-frame span (the typical collision moment) is never skipped just because that exact frame
    was not sampled."""
    if n_sampled <= 1 or n_total <= n_sampled:
        return 0
    return max(0, round((n_total - 1) / (n_sampled - 1) / 2))


def mask_frames(frames: list[Image.Image], regions: list[Region], n_total: int) -> list[Image.Image]:
    idx = sampled_frame_indices(n_total, len(frames))
    tol = span_tolerance(n_total, len(frames))
    out = []
    for f, orig in zip(frames, idx):
        hits = [b for (t0, t1, b) in regions if t0 - tol <= orig <= t1 + tol]
        if not hits:
            out.append(f)
            continue
        im = f.copy()
        d = ImageDraw.Draw(im)
        w, h = im.size
        for x0, y0, x1, y1 in hits:
            d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], fill=(0, 0, 0))
        out.append(im)
    return out


def random_regions(regions: list[Region], rng: random.Random) -> list[Region]:
    """Same spans and box sizes, random positions."""
    out = []
    for t0, t1, (x0, y0, x1, y1) in regions:
        bw, bh = x1 - x0, y1 - y0
        nx = rng.uniform(0.0, max(0.0, 1.0 - bw))
        ny = rng.uniform(0.0, max(0.0, 1.0 - bh))
        out.append((t0, t1, (nx, ny, nx + bw, ny + bh)))
    return out


def masked_fraction(regions: list[Region], n_total: int) -> float:
    """Mean over frames of the masked area fraction (rough box union ignoring overlaps, capped at 1)."""
    tot = 0.0
    for f in range(n_total):
        a = sum((b[2] - b[0]) * (b[3] - b[1]) for (t0, t1, b) in regions if t0 <= f <= t1)
        tot += min(1.0, a)
    return tot / n_total


def decisive_triplets(chain: dict) -> list[dict]:
    ts = chain["triplets"]
    if chain.get("question_type") in ("predictive", "counterfactual"):
        obs = [t for t in ts if t.get("role") == "observe" or (t.get("role") is None and t["question"].startswith("Do "))]
        if obs:
            return obs
    return ts[-1:]


class EvidenceIndex:
    """Evidence regions per CLEVRER question, from generated chain files. Key: '<scene>_<qid>'."""

    def __init__(self, regions: dict[str, list[Region]], option_regions: dict[str, list[Region]] | None = None) -> None:
        self.regions = regions
        self.option_regions = option_regions or {}  # key '<scene>_<qid>_<choice_id>': that option's own chain

    @classmethod
    def from_chains(cls, path: Path | str, last_only: bool = False) -> EvidenceIndex:
        """`last_only`: keep only the *decisive* triplet(s) of each chain (for MC questions the union over
        the per-option chains): the observed-fact step ("do X and Y collide?", role=observe) for predictive
        and counterfactual chains, otherwise the final triplet. Older chain files without roles fall back
        to the question prefix."""
        regions: dict[str, set[Region]] = defaultdict(set)
        opt: dict[str, set[Region]] = defaultdict(set)
        with Path(path).open() as f:
            for line in f:
                c = json.loads(line)
                key = f"{c['meta']['scene_index']}_{c['meta']['question_id']}"
                cid = c["meta"].get("choice_id")
                for t in (decisive_triplets(c) if last_only else c["triplets"]):
                    e = t["evidence"]
                    box = tuple(e["box"])
                    if box == (0.0, 0.0, 1.0, 1.0):
                        continue  # "none found" whole-video evidence carries no localizable region
                    regions[key].add((e["t_start"], e["t_end"], box))
                    if cid is not None:
                        opt[f"{key}_{cid}"].add((e["t_start"], e["t_end"], box))
        return cls({k: sorted(v) for k, v in regions.items()}, {k: sorted(v) for k, v in opt.items()})

    def get(self, key: str) -> list[Region]:
        return self.regions.get(key, [])

    def get_for_item(self, item_id: str, question_id: str) -> list[Region]:
        """Grounding reference for an item: a per-option item is scored against its own option's chain,
        not the evidence pooled over all options of the question (which dilutes recall)."""
        return self.option_regions.get(item_id) or self.regions.get(question_id, [])


def evidence_to_region(e: Evidence) -> Region:
    return (e.t_start, e.t_end, tuple(e.box))


# ---------------------------------------------------------------- object-track masking (CLEVRER)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class TrackMasker:
    """Remove evidence *objects* for the whole video instead of a box at one moment.

    Evidence regions are matched to proposal detections at their frames (IoU >= `min_iou`, or the
    region containing the detection); the matched objects' boxes are then blacked out on every
    sampled frame where they are detected. The control masks copies of the same tracks at random
    positions that avoid the evidence boxes: identical masked area and timing, different location.
    """

    def __init__(self, proposals_root: Path | str, min_iou: float = 0.3) -> None:
        self.root = Path(proposals_root)
        self.min_iou = min_iou
        self._cache: dict[int, object] = {}

    def _index(self, scene_index: int):
        from causalsight.data.clevrer_evidence import ProposalIndex

        if scene_index not in self._cache:
            self._cache.clear()
            self._cache[scene_index] = ProposalIndex.load(self.root / f"proposal_{scene_index:05d}.json")
        return self._cache[scene_index]

    def evidence_objects(self, scene_index: int, regions: list[Region]) -> set[tuple[str, str, str]]:
        idx = self._index(scene_index)
        found: set[tuple[str, str, str]] = set()
        for t0, t1, box in regions:
            for f in range(t0, t1 + 1):
                for key, (b, _score) in idx._frame(f).items():
                    contained = b[0] >= box[0] - 0.01 and b[1] >= box[1] - 0.01 and b[2] <= box[2] + 0.01 and b[3] <= box[3] + 0.01
                    if contained or _iou(b, box) >= self.min_iou:
                        found.add(key)
        return found

    def all_objects(self, scene_index: int) -> set[tuple[str, str, str]]:
        idx = self._index(scene_index)
        keys: set[tuple[str, str, str]] = set()
        for f in range(0, idx.n_frames, 8):
            keys |= set(idx._frame(f).keys())
        return keys

    def track_regions(self, scene_index: int, objects: set[tuple[str, str, str]], sampled: list[int]) -> list[Region]:
        """One single-frame region per (object, sampled frame) where the object is detected."""
        idx = self._index(scene_index)
        regs: list[Region] = []
        for f in sampled:
            fr = idx._frame(f)
            for key in objects:
                if key in fr:
                    regs.append((f, f, fr[key][0]))
        return regs

    def regions_for(self, scene_index: int, regions: list[Region], n_total: int, n_sampled: int, control: bool, rng: random.Random) -> list[Region]:
        """Evidence-object tracks, or (control) copies of those tracks at random positions that do not
        overlap the evidence boxes on the same frame: same masked area and timing, different place."""
        sampled = sampled_frame_indices(n_total, n_sampled)
        ev = self.evidence_objects(scene_index, regions)
        if not ev:
            return []
        tracks = self.track_regions(scene_index, ev, sampled)
        if not control:
            return tracks
        by_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        for t0, _t1, b in tracks:
            by_frame[t0].append(b)
        out: list[Region] = []
        for t0, t1, box in tracks:
            bw, bh = box[2] - box[0], box[3] - box[1]
            best = None
            for _ in range(30):
                nx = rng.uniform(0.0, max(0.0, 1.0 - bw))
                ny = rng.uniform(0.0, max(0.0, 1.0 - bh))
                cand = (nx, ny, nx + bw, ny + bh)
                overlap = max((_iou(cand, b) for b in by_frame[t0]), default=0.0)
                if best is None or overlap < best[0]:
                    best = (overlap, cand)
                if overlap == 0.0:
                    break
            out.append((t0, t1, best[1]))
        return out
