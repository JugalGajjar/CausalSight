"""Entity-track test: every detection of the cited entities on every sampled frame, plus the citations, vs
same-size boxes elsewhere on the same frames. No model, no detector: synthetic chains and tracks."""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from causalsight.eval.entity_tracks import EntityTrackIndex, avoiding_random_regions, frames_hit, grid_position, item_entities, phrases_by_video, track_summary
from causalsight.eval.faithfulness.report import report
from causalsight.eval.masking import CHAIN_SCALE, mask_frames

RAW = (
    "<think>\n[1] Q: Where is 2: white dog: the white dog? | A: the white dog | E: t=4-12 box=(0.100,0.200,0.400,0.600) | deps=\n"
    "[2] Q: Where is the man? | A: the man | E: t=60-68 box=(0.500,0.100,0.700,0.500) | deps=\n"
    "[3] Q: What happens? | A: he waves | E: t=0-127 box=(0.000,0.000,1.000,1.000) | deps=1,2\n</think>\n<answer>yes</answer>"
)
TRACKS = {
    "v1": {
        "the white dog": {"0": [0.1, 0.2, 0.4, 0.6], "5": [0.2, 0.2, 0.5, 0.6], "15": [0.3, 0.3, 0.6, 0.7]},
        "the man": {"7": [0.5, 0.1, 0.7, 0.5]},
        "the ball": {"3": [0.0, 0.0, 0.1, 0.1]},  # named by another item: not this item's entity
    }
}


def write_outputs(path: Path):
    rows = [
        {"id": "v1_3_0", "qid": "v1_3", "frames_key": "v1", "raw": RAW, "correct": 1},
        {"id": "v1_3_1", "qid": "v1_3", "frames_key": "v1", "raw": "<think>\n[1] Q: Where is the cat? | A: cat | E: t=0-8 box=(0.1,0.1,0.2,0.2) | deps=\n</think>\n<answer>no</answer>", "correct": 1},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_entities_and_phrases(tmp_path):
    assert item_entities(RAW) == ["the white dog", "the man"]  # pseudo-label index stripped, whole-frame step ignored
    out = tmp_path / "m_nq_plain.jsonl"
    write_outputs(out)
    assert phrases_by_video([out]) == {"v1": {"the white dog", "the man", "the cat"}}


def test_index_unions_tracks_and_citations(tmp_path):
    out = tmp_path / "m_nq_plain.jsonl"
    write_outputs(out)
    (tmp_path / "tracks.json").write_text(json.dumps({**TRACKS, "_summary": {"videos": 1}}))
    idx = EntityTrackIndex.load(tmp_path / "tracks.json", out)
    regs = idx.get("v1_3_0")
    g = grid_position
    assert (g(0), g(0), (0.1, 0.2, 0.4, 0.6)) in regs and (g(5), g(5), (0.2, 0.2, 0.5, 0.6)) in regs and (g(15), g(15), (0.3, 0.3, 0.6, 0.7)) in regs
    assert (g(7), g(7), (0.5, 0.1, 0.7, 0.5)) in regs
    assert (4, 12, (0.1, 0.2, 0.4, 0.6)) in regs and (60, 68, (0.5, 0.1, 0.7, 0.5)) in regs  # own citations kept
    assert not any(b == (0.0, 0.0, 0.1, 0.1) for _, _, b in regs)  # the ball belongs to another item
    assert idx.get("v1_3_1") == []  # 'the cat' was never detected: skipped under both track modes


def test_grid_regions_hit_exactly_their_frame():
    frames = [Image.new("RGB", (64, 48), (255, 255, 255)) for _ in range(16)]
    regs = [(grid_position(f), grid_position(f), (0.1, 0.2, 0.4, 0.6)) for f in (0, 5, 15)]
    masked = mask_frames(frames, regs, CHAIN_SCALE)
    dark = [int(np.asarray(f)[..., 0].min() < 50) for f in masked]
    assert dark == [1 if f in (0, 5, 15) else 0 for f in range(16)]
    assert frames_hit(regs[1], CHAIN_SCALE, 16) == [5]


def test_control_keeps_size_and_timing_and_avoids_entity_boxes():
    regs = [(grid_position(5), grid_position(5), (0.1, 0.2, 0.4, 0.6)), (grid_position(5), grid_position(5), (0.5, 0.1, 0.7, 0.5)), (4, 12, (0.1, 0.2, 0.4, 0.6))]
    ctrl = avoiding_random_regions(regs, CHAIN_SCALE, 16, random.Random(0))
    assert [(t0, t1) for t0, t1, _ in ctrl] == [(t0, t1) for t0, t1, _ in regs]
    for (_, _, a), (_, _, b) in zip(regs, ctrl):
        assert abs((a[2] - a[0]) - (b[2] - b[0])) < 1e-9 and abs((a[3] - a[1]) - (b[3] - b[1])) < 1e-9
    from causalsight.eval.masking import _iou

    boxes_f5 = [b for t0, t1, b in regs if 5 in frames_hit((t0, t1, b), CHAIN_SCALE, 16)]
    for _, _, c in ctrl[:2]:
        assert max(_iou(c, b) for b in boxes_f5) < 0.05


def test_track_summary_counts_frames():
    s = track_summary(TRACKS)
    assert s["entities"] == 3 and s["frames_detected_histogram"] == {"1": 2, "3": 1}


def test_report_cite_track_gap_on_correct_subset(tmp_path):
    ids = [f"q{i}" for i in range(4)]
    mk = lambda correct, sub=None: [{"id": i, "question_type": "causal", "correct": c} for i, c in zip(ids, correct) if sub is None or i in sub]
    w = lambda name, rows: (tmp_path / f"m_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    w("plain", mk([1, 1, 1, 0]))
    w("blind", mk([1, 0, 0, 0]))
    w("mask_cite", mk([1, 1, 1, 0]))
    w("mask_cite_random", mk([1, 1, 1, 0]))
    w("mask_cite_track", mk([0, 0, 1, 0], sub={"q0", "q1", "q2"}))  # run only on the items correct in plain
    w("mask_cite_track_random", mk([1, 1, 1, 0], sub={"q0", "q1", "q2"}))
    r = report(tmp_path, "m")["ALL"]
    assert r["cite_gap"] == 0.0 and r["n_cite_track"] == 3
    assert abs(r["cite_track_gap"] - 2 / 3) < 1e-9 and r["flip_cite_track_random"] == 0.0
