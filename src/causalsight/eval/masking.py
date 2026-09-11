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


class EvidenceIndex:
    """Evidence regions per CLEVRER question, from generated chain files. Key: '<scene>_<qid>'."""

    def __init__(self, regions: dict[str, list[Region]]) -> None:
        self.regions = regions

    @classmethod
    def from_chains(cls, path: Path | str) -> EvidenceIndex:
        regions: dict[str, set[Region]] = defaultdict(set)
        with Path(path).open() as f:
            for line in f:
                c = json.loads(line)
                key = f"{c['meta']['scene_index']}_{c['meta']['question_id']}"
                for t in c["triplets"]:
                    e = t["evidence"]
                    box = tuple(e["box"])
                    if box == (0.0, 0.0, 1.0, 1.0):
                        continue  # "none found" whole-video evidence carries no localizable region
                    regions[key].add((e["t_start"], e["t_end"], box))
        return cls({k: sorted(v) for k, v in regions.items()})

    def get(self, key: str) -> list[Region]:
        return self.regions.get(key, [])


def evidence_to_region(e: Evidence) -> Region:
    return (e.t_start, e.t_end, tuple(e.box))
