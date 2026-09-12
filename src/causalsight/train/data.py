"""Training data access: subset records with pre-extracted frames (see causalsight.train.frames)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def load_records(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open()]


def load_frames(root: Path | str, rec: dict, n_frames: int | None = None) -> list[Image.Image]:
    d = Path(root) / rec["frames"]
    files = sorted(d.glob("f*.jpg"))
    if n_frames and len(files) > n_frames:
        idx = [round(i * (len(files) - 1) / (n_frames - 1)) for i in range(n_frames)]
        files = [files[i] for i in idx]
    return [Image.open(f).convert("RGB") for f in files]
