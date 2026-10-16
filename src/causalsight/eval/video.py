"""Frame sampling and the blind-input control."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def sample_frames(video_path: Path | str, n: int = 16) -> list[Image.Image]:
    """Uniformly sample `n` RGB frames across the whole video, decoding in a stream and keeping only
    the selected frames (a whole clip is never held in memory)."""
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total = stream.frames or 0
        if total <= 0:  # unknown frame count: count in a cheap pass without converting to images
            total = sum(1 for _ in container.decode(stream))
            container.seek(0)
        if total <= 0:
            raise ValueError(f"no frames decoded from {video_path}")
        if total <= n:
            wanted = set(range(total))
        else:
            wanted = {round(i * (total - 1) / (n - 1)) for i in range(n)}
        out = []
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                out.append(frame.to_image())
            if i >= max(wanted):
                break
    if not out:
        raise ValueError(f"no frames decoded from {video_path}")
    return out


def blank_like(frames: list[Image.Image]) -> list[Image.Image]:
    """Same count and size, all black. Used for the Blind Gap metric."""
    w, h = frames[0].size
    return [Image.new("RGB", (w, h), (0, 0, 0)) for _ in frames]
