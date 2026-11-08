"""Entity tracks on real video, for the entity-track test (mask modes cite_track / cite_track_random).

The self-citation test masks one box on a few frames. On real video the cited entity is usually visible on many
frames, so a zero cite gap could mean "the citation is decorative" or "the entity is redundant". The entity-track
test removes the entity *wherever it is detected*: every entity a chain names ("Where is <entity>?") is re-detected
with Grounding DINO on all 16 sampled frames, and the intervention blacks out every detection plus the cited boxes;
the control blacks out boxes of the same sizes on the same frames at random positions that avoid the entity boxes.

Step 1 (GPU, once per set of chain models; ~3k detector calls of 16 frames each for 999 NExT-QA questions):

  cs-entity-tracks results/colab/mixsft3b_nq_plain.jsonl results/colab/realrl3b_nq_plain.jsonl \
      --pack data/frames/nextqa_pack --out results/colab/nextqa_entity_tracks.json

writes {video: {phrase: {"<frame>": [x0, y0, x1, y1]}}} (top detection per sampled frame, normalised) and a
summary of how many of the 16 frames each entity is detected on (the redundancy the test is about).

Step 2: cs-eval --mask cite_track / cite_track_random --chains <tag>_nq_plain.jsonl --tracks <tracks.json>
(data/scripts/cite_track_eval.sh), then cs-faithfulness reports the cite-track gap next to the cite gap.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from causalsight.eval.cite_grounding import N_FRAMES, entity_of, steps_of
from causalsight.eval.masking import CHAIN_SCALE, EvidenceIndex, Region, _iou, sampled_frame_indices, span_tolerance

Box = tuple[float, float, float, float]


def item_entities(raw: str) -> list[str]:
    """Lower-cased entity phrases named by the chain's 'Where is <entity>?' steps (deduplicated, in order)."""
    seen: list[str] = []
    for e, _f, _b in steps_of(raw):
        e = e.lower()
        if e not in seen:
            seen.append(e)
    return seen


def phrases_by_video(output_files: Iterable[Path]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for path in output_files:
        with Path(path).open() as f:
            for line in f:
                r = json.loads(line)
                video = r.get("frames_key") or r["id"].split("_")[0]
                out[video].update(item_entities(r.get("raw", "")))
    return out


def build_tracks(phrases: dict[str, set[str]], detect_all: Callable[[str, str], list[Box | None]]) -> dict[str, dict[str, dict[str, list[float]]]]:
    """detect_all(video, phrase) -> top detection (or None) on each of the N_FRAMES sampled frames."""
    tracks: dict[str, dict[str, dict[str, list[float]]]] = {}
    for video in sorted(phrases):
        tracks[video] = {}
        for phrase in sorted(phrases[video]):
            dets = detect_all(video, phrase)
            tracks[video][phrase] = {str(f): [round(v, 4) for v in b] for f, b in enumerate(dets) if b is not None}
    return tracks


def track_summary(tracks: dict) -> dict:
    counts = [len(v) for ph in tracks.values() for v in ph.values()]
    hist = defaultdict(int)
    for c in counts:
        hist[c] += 1
    return {
        "videos": len(tracks),
        "entities": len(counts),
        "mean_frames_detected": round(sum(counts) / len(counts), 2) if counts else None,
        "share_detected_on_8_or_more_frames": round(sum(c >= 8 for c in counts) / len(counts), 3) if counts else None,
        "share_never_detected": round(sum(c == 0 for c in counts) / len(counts), 3) if counts else None,
        "frames_detected_histogram": {str(k): hist[k] for k in sorted(hist)},
    }


def dino_detect_all(pack: Path, detector: str, threshold: float) -> Callable[[str, str], list[Box | None]]:
    """One batched detector call per (video, phrase) over the pack's 16 frames."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    from causalsight.train.data import load_frames

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(detector)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(device).eval()
    cache: dict[str, list] = {}

    def frames_of(video: str) -> list:
        if video not in cache:
            cache.clear()  # videos are processed in order; keep one in memory
            cache[video] = load_frames(pack, {"frames": f"frames/{video}"}, N_FRAMES)
        return cache[video]

    def detect_all(video: str, phrase: str) -> list[Box | None]:
        frames = frames_of(video)
        text = phrase.lower() + " ."
        inputs = proc(images=frames, text=[text] * len(frames), return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            o = det(**inputs)
        res = proc.post_process_grounded_object_detection(o, inputs.input_ids, threshold=threshold, text_threshold=threshold, target_sizes=[im.size[::-1] for im in frames])
        out: list[Box | None] = []
        for im, r in zip(frames, res):
            if not len(r["scores"]):
                out.append(None)
                continue
            i = int(r["scores"].argmax())
            x0, y0, x1, y1 = [float(v) for v in r["boxes"][i]]
            W, H = im.size
            out.append((max(0.0, x0 / W), max(0.0, y0 / H), min(1.0, x1 / W), min(1.0, y1 / H)))
        return out

    return detect_all


# ---------------------------------------------------------------- masking side


def grid_position(frame: int, n_sampled: int = N_FRAMES, n_total: int = CHAIN_SCALE) -> int:
    """Chain-grid position of sampled frame `frame`, so a (g, g, box) region hits exactly that frame in mask_frames."""
    return sampled_frame_indices(n_total, n_sampled)[frame]


def frames_hit(region: Region, n_total: int, n_sampled: int) -> list[int]:
    idx = sampled_frame_indices(n_total, n_sampled)
    tol = span_tolerance(n_total, n_sampled)
    t0, t1, _ = region
    return [f for f, orig in enumerate(idx) if t0 - tol <= orig <= t1 + tol]


def avoiding_random_regions(regions: list[Region], n_total: int, n_sampled: int, rng: random.Random, tries: int = 30) -> list[Region]:
    """Same spans and box sizes as `regions`, at random positions that avoid (lowest IoU with, ideally zero) every
    intervention box on the frames each region hits: identical masked area and timing, different place."""
    by_frame: dict[int, list[Box]] = defaultdict(list)
    for reg in regions:
        for f in frames_hit(reg, n_total, n_sampled):
            by_frame[f].append(reg[2])
    out: list[Region] = []
    for t0, t1, box in regions:
        bw, bh = box[2] - box[0], box[3] - box[1]
        avoid = [b for f in frames_hit((t0, t1, box), n_total, n_sampled) for b in by_frame[f]]
        best: tuple[float, Box] | None = None
        for _ in range(tries):
            nx = rng.uniform(0.0, max(0.0, 1.0 - bw))
            ny = rng.uniform(0.0, max(0.0, 1.0 - bh))
            cand = (nx, ny, nx + bw, ny + bh)
            overlap = max((_iou(cand, b) for b in avoid), default=0.0)
            if best is None or overlap < best[0]:
                best = (overlap, cand)
            if overlap == 0.0:
                break
        out.append((t0, t1, best[1]))
    return out


class EntityTrackIndex:
    """Per-item regions for the entity-track test: every detection, on every sampled frame, of every entity the
    item's own chain names, plus the regions the chain cited. Single-frame regions on the chain grid."""

    def __init__(self, tracks: dict, entities: dict[str, list[str]], videos: dict[str, str], cited: EvidenceIndex, n_sampled: int = N_FRAMES) -> None:
        self.tracks = tracks
        self.entities = entities
        self.videos = videos
        self.cited = cited
        self.n_sampled = n_sampled

    @classmethod
    def load(cls, tracks_path: Path | str, outputs_path: Path | str) -> EntityTrackIndex:
        tracks = json.loads(Path(tracks_path).read_text())
        tracks.pop("_summary", None)
        entities: dict[str, list[str]] = {}
        videos: dict[str, str] = {}
        with Path(outputs_path).open() as f:
            for line in f:
                r = json.loads(line)
                entities[r["id"]] = item_entities(r.get("raw", ""))
                videos[r["id"]] = r.get("frames_key") or r["id"].split("_")[0]
        return cls(tracks, entities, videos, EvidenceIndex.from_outputs(outputs_path))

    def track_regions(self, item_id: str) -> list[Region]:
        video = self.videos.get(item_id)
        if video is None or video not in self.tracks:
            return []
        regs: set[Region] = set()
        for e in self.entities.get(item_id, []):
            for f, box in self.tracks[video].get(e, {}).items():
                g = grid_position(int(f), self.n_sampled)
                regs.add((g, g, tuple(box)))
        return sorted(regs)

    def get(self, item_id: str) -> list[Region]:
        """Track detections plus the item's own citations; empty when the chain names nothing that was detected
        (such items are skipped under both track modes)."""
        tr = self.track_regions(item_id)
        if not tr:
            return []
        return sorted(set(tr) | set(self.cited.get_cited(item_id)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs", nargs="+", type=Path, help="harness per-item jsonl files of chain models (plain runs); entities are pooled per video")
    ap.add_argument("--pack", type=Path, required=True, help="NExT-QA pack with frames/<video>/fNN.jpg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detector", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=None, help="first N videos (pilot)")
    a = ap.parse_args()
    phrases = phrases_by_video(a.outputs)
    if a.limit:
        phrases = {v: phrases[v] for v in sorted(phrases)[: a.limit]}
    n_calls = sum(len(v) for v in phrases.values())
    print(f"{len(phrases)} videos, {n_calls} (video, phrase) pairs")
    tracks = build_tracks(phrases, dino_detect_all(a.pack, a.detector, a.threshold))
    summ = track_summary(tracks)
    summ.update({"detector": a.detector, "threshold": a.threshold, "sources": [str(p) for p in a.outputs]})
    tracks["_summary"] = summ
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(tracks))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
