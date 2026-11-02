"""Citation grounding on real video: does a cited box land on the entity the step names?

  cs-cite-grounding results/colab/mixrl3b_nq_plain.jsonl --pack data/frames/nextqa_pack \
      --out results/colab/mixrl3b_nq_cite_grounding.json

For every parsed step of every chain in a harness per-item output file: the entity is the step's sub-question
("Where is <entity>?"), the frame is the sampled frame nearest the centre of the cited span (chain grid 0-127 onto
the pack's 16 frames), and Grounding DINO localizes the entity phrase on that frame. We report IoU between the cited
box and the top detection and the hit rate at IoU >= 0.5, against a control box of the same size at a random
position on the same frame. Detector calls are cached per (video, frame, phrase), so the per-option repeats of one
question cost one call. The detector is the one that produced the pseudo-label boxes, so this measures whether the
model reproduces detector-grade localization of what it names, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

from causalsight.eval.masking import CHAIN_SCALE, _iou
from causalsight.train.format import parse_chain

Box = tuple[float, float, float, float]
N_FRAMES = 16
ENTITY_RE = re.compile(r"^\s*where\s+is\s+(?P<e>.+?)\s*\??\s*$", re.IGNORECASE)
WHOLE = 0.98  # boxes covering >= this fraction of the frame are not localizations


def entity_of(question: str) -> str | None:
    """'Where is 2: white dog: the white dog?' -> 'the white dog'; 'Where is the man in red?' -> 'the man in red'."""
    m = ENTITY_RE.match(question)
    if not m:
        return None
    e = m.group("e").strip()
    e = re.sub(r"^\d+\s*:\s*", "", e)  # pseudo-label entity index
    if ": " in e:
        e = e.rsplit(": ", 1)[1]
    e = e.strip(" .")
    return e if 1 <= len(e) <= 80 else None


def frame_of(t0: int, t1: int, n_frames: int = N_FRAMES) -> int:
    c = (t0 + t1) / 2
    return min(n_frames - 1, max(0, round(c * (n_frames - 1) / (CHAIN_SCALE - 1))))


def random_box(box: Box, rng: random.Random) -> Box:
    bw, bh = box[2] - box[0], box[3] - box[1]
    x = rng.uniform(0.0, max(0.0, 1.0 - bw))
    y = rng.uniform(0.0, max(0.0, 1.0 - bh))
    return (x, y, x + bw, y + bh)


def steps_of(raw: str) -> list[tuple[str, int, Box]]:
    """(entity, frame, cited box) for every parsed step with a nameable entity and a non-whole-frame box."""
    p = parse_chain(raw)
    out = []
    if not p.chain:
        return out
    for t in p.chain.triplets:
        e = entity_of(t.question)
        b = tuple(float(v) for v in t.evidence.box)
        if e is None or (b[2] - b[0]) * (b[3] - b[1]) >= WHOLE:
            continue
        out.append((e, frame_of(t.evidence.t_start, t.evidence.t_end), b))
    return out


def score(rows: list[dict], detect: Callable[[str, int, str], Box | None], seed: int = 0) -> dict:
    """detect(video, frame, phrase) -> top detection box (normalized) or None."""
    rng = random.Random(seed)
    agg: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    cache: dict[tuple, Box | None] = {}
    seen_unique = set()
    for r in rows:
        video = r.get("frames_key") or r["id"].split("_")[0]
        types = (r["question_type"], "ALL")
        for k in types:
            agg[k]["n_items"] += 1
        steps = steps_of(r["raw"])
        for e, f, b in steps:
            key = (video, f, e.lower())
            if key not in cache:
                cache[key] = detect(video, f, e)
            det = cache[key]
            seen_unique.add(key)
            ctrl = random_box(b, rng)
            for k in types:
                a = agg[k]
                a["n_steps"] += 1
                if det is None:
                    continue
                a["n_detected"] += 1
                iou, riou = _iou(b, det), _iou(ctrl, det)
                a["iou_sum"] += iou
                a["hit50"] += iou >= 0.5
                a["riou_sum"] += riou
                a["rhit50"] += riou >= 0.5
    out = {}
    for k, a in agg.items():
        nd = a["n_detected"]
        out[k] = {
            "n_items": int(a["n_items"]), "n_steps": int(a["n_steps"]),
            "detected_rate": round(nd / a["n_steps"], 3) if a["n_steps"] else None,
            "mean_iou": round(a["iou_sum"] / nd, 3) if nd else None,
            "hit50": round(a["hit50"] / nd, 3) if nd else None,
            "random_mean_iou": round(a["riou_sum"] / nd, 3) if nd else None,
            "random_hit50": round(a["rhit50"] / nd, 3) if nd else None,
        }
    out["ALL"]["unique_detector_calls"] = len(seen_unique)
    return out


def dino_detector(pack: Path, detector: str, threshold: float) -> Callable[[str, int, str], Box | None]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    from causalsight.train.data import load_frames

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(detector)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(device).eval()
    frames: dict[str, list] = {}

    def detect(video: str, frame: int, phrase: str) -> Box | None:
        if video not in frames:
            frames.clear()  # rows are grouped by video; keep one video's frames in memory
            frames[video] = load_frames(pack, {"frames": f"frames/{video}"}, N_FRAMES)
        img = frames[video][frame]
        inputs = proc(images=img, text=phrase.lower() + " .", return_tensors="pt").to(device)
        with torch.no_grad():
            o = det(**inputs)
        res = proc.post_process_grounded_object_detection(o, inputs.input_ids, threshold=threshold, text_threshold=threshold, target_sizes=[img.size[::-1]])[0]
        if not len(res["scores"]):
            return None
        i = int(res["scores"].argmax())
        x0, y0, x1, y1 = [float(v) for v in res["boxes"][i]]
        W, H = img.size
        return (max(0.0, x0 / W), max(0.0, y0 / H), min(1.0, x1 / W), min(1.0, y1 / H))

    return detect


def to_markdown(rep: dict, tag: str) -> str:
    cols = ["n_items", "n_steps", "detected_rate", "mean_iou", "hit50", "random_mean_iou", "random_hit50"]
    lines = [f"Citation grounding: {tag}", "", "| type | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k in ["ALL"] + sorted(x for x in rep if x != "ALL"):
        lines.append(f"| {k} | " + " | ".join("" if rep[k].get(c) is None else str(rep[k][c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs", type=Path, help="harness per-item jsonl of a chain model (plain run)")
    ap.add_argument("--pack", type=Path, required=True, help="NExT-QA pack with frames/<video>/fNN.jpg")
    ap.add_argument("--out", type=Path, required=True, help="json report (a .md next to it is written too)")
    ap.add_argument("--detector", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=None, help="first N items (pilot)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.outputs.open()]
    rows.sort(key=lambda r: (r.get("frames_key") or r["id"].split("_")[0], r["id"]))  # one video's frames at a time
    if a.limit:
        rows = rows[: a.limit]
    rep = score(rows, dino_detector(a.pack, a.detector, a.threshold), a.seed)
    rep["ALL"]["detector"] = a.detector
    rep["ALL"]["threshold"] = a.threshold
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2))
    md = to_markdown(rep, a.outputs.stem)
    a.out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
