"""Frame sampling and the blind-input control."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def sample_frames(video_path: Path | str, n: int = 16) -> list[Image.Image]:
    """Uniformly sample `n` RGB frames across the whole video."""
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        frames = [f.to_image() for f in container.decode(stream)]
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    if len(frames) <= n:
        return frames
    idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in idx]


def blank_like(frames: list[Image.Image]) -> list[Image.Image]:
    """Same count and size, all black. Used for the Blind Gap metric."""
    w, h = frames[0].size
    return [Image.new("RGB", (w, h), (0, 0, 0)) for _ in frames]
