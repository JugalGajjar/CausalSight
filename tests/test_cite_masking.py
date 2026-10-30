"""Cite masking: regions a model wrote in its own chains, read from harness outputs, masked on the chain grid."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from causalsight.eval.faithfulness.report import report
from causalsight.eval.masking import CHAIN_SCALE, EvidenceIndex, mask_frames, random_regions, sampled_frame_indices, span_tolerance

RAW = (
    "<think>\n[1] Q: Where is the man? | A: left | E: t=4-12 box=(0.100,0.200,0.400,0.600) | deps=\n"
    "[2] Q: What happens? | A: he waves | E: t=0-127 box=(0.000,0.000,1.000,1.000) | deps=1\n</think>\n<answer>yes</answer>"
)


def write_outputs(path: Path):
    rows = [
        {"id": "v1_3_0", "qid": "v1_3", "raw": RAW, "pred": "yes", "gold": "yes", "correct": 1},
        {"id": "v1_3_1", "qid": "v1_3", "raw": "<answer>no</answer>", "pred": "no", "gold": "no", "correct": 1},  # no chain
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_from_outputs_keys_per_item_and_drops_whole_frame(tmp_path):
    out = tmp_path / "m_nq_plain.jsonl"
    write_outputs(out)
    ev = EvidenceIndex.from_outputs(out)
    assert ev.get_cited("v1_3_0") == [(4, 12, (0.1, 0.2, 0.4, 0.6))]  # whole-frame step dropped
    assert ev.get_cited("v1_3_1") == []  # bare answer: nothing to mask, item is skipped in cite modes
    assert ev.get("v1_3") == [(4, 12, (0.1, 0.2, 0.4, 0.6))]


def test_cited_span_masks_the_sampled_frames_on_the_chain_grid():
    frames = [Image.new("RGB", (64, 48), (255, 255, 255)) for _ in range(16)]
    regions = [(4, 12, (0.1, 0.2, 0.4, 0.6))]
    masked = mask_frames(frames, regions, CHAIN_SCALE)
    idx = sampled_frame_indices(CHAIN_SCALE, 16)
    tol = span_tolerance(CHAIN_SCALE, 16)
    dark = [int(np.asarray(f)[..., 0].min() < 50) for f in masked]
    expected = [int(4 - tol <= o <= 12 + tol) for o in idx]  # sampled positions 0, 8, 17, ...: frames 0 and 8 are hit
    assert dark == expected and sum(dark) == 2
    ctrl = random_regions(regions, __import__("random").Random(0))
    assert ctrl[0][:2] == (4, 12) and abs((ctrl[0][2][2] - ctrl[0][2][0]) - 0.3) < 1e-9


def test_report_cite_gap(tmp_path):
    ids = [f"q{i}" for i in range(4)]
    mk = lambda correct, sub=None: [{"id": i, "question_type": "causal", "correct": c} for i, c in zip(ids, correct) if sub is None or i in sub]
    w = lambda name, rows: (tmp_path / f"m_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    w("plain", mk([1, 1, 1, 0]))
    w("blind", mk([1, 0, 0, 0]))
    w("mask_cite", mk([0, 0, 1, 0], sub={"q0", "q1", "q2"}))  # q3 cited nothing: skipped
    w("mask_cite_random", mk([1, 1, 1, 0], sub={"q0", "q1", "q2"}))
    r = report(tmp_path, "m")["ALL"]
    assert r["n"] == 4 and r["n_cite"] == 3
    assert abs(r["flip_cite"] - 2 / 3) < 1e-9 and r["flip_cite_random"] == 0.0 and abs(r["cite_gap"] - 2 / 3) < 1e-9
