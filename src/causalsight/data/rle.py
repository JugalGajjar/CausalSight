"""COCO run-length-encoded mask decoding, dependency-free (numpy only).

CLEVRER derender proposals store per-object masks as COCO RLE: {"size": [h, w], "counts": <str|list>}.
Counts alternate 0-runs and 1-runs starting with a 0-run, in column-major (Fortran) order.
"""

from __future__ import annotations

import numpy as np


def decode_counts(counts: str | list[int]) -> list[int]:
    """Decode COCO's compressed LEB128-style string into a list of run lengths."""
    if isinstance(counts, list):
        return [int(c) for c in counts]
    out: list[int] = []
    i = 0
    n = len(counts)
    while i < n:
        x = 0
        k = 0
        more = True
        while more:
            c = ord(counts[i]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            i += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(out) > 2:
            x += out[-2]
        out.append(x)
    return out


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    runs = decode_counts(rle["counts"])
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for r in runs:
        if val:
            flat[pos : pos + r] = 1
        pos += r
        val ^= 1
    return flat.reshape((w, h)).T  # Fortran order -> (h, w)


def mask_to_box(mask: np.ndarray, normalized: bool = True) -> tuple[float, float, float, float] | None:
    """Tight box around nonzero pixels as (x_min, y_min, x_max, y_max). None if the mask is empty.

    x_max / y_max are exclusive pixel edges so that a 1-pixel mask has positive area.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if normalized:
        return (x0 / w, y0 / h, x1 / w, y1 / h)
    return (float(x0), float(y0), float(x1), float(y1))
