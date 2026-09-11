"""Regression test: masking must apply only in --mask modes, and plain runs with --chains must not mask."""

import json
from pathlib import Path

import numpy as np

from causalsight.eval.harness import run
from causalsight.eval.masking import span_tolerance
from causalsight.models.base import VLMBackend


class RecordingBackend(VLMBackend):
    name = "recording"

    def __init__(self, **_):
        self.frames = []

    def generate(self, frames, prompt, max_new_tokens=64):
        self.frames.append(frames)
        return "0"


def write_video(path: Path, n_frames=128, size=(64, 48)):
    import av

    with av.open(str(path), "w") as c:
        st = c.add_stream("mpeg4", rate=25)
        st.width, st.height = size
        st.pix_fmt = "yuv420p"
        for _ in range(n_frames):
            arr = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
            for pkt in st.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def setup(tmp_path: Path):
    root = tmp_path / "clevrer"
    vdir = root / "videos" / "validation" / "video_validation" / "video_10000-11000"
    vdir.mkdir(parents=True)
    write_video(vdir / "video_10000.mp4")
    (root / "questions").mkdir()
    qs = [{"scene_index": 10000, "video_filename": "video_10000.mp4", "questions": [
        {"question_id": 0, "question": "How many spheres?", "question_type": "descriptive", "question_subtype": "count", "program": [], "answer": "0"}]}]
    (root / "questions" / "validation.json").write_text(json.dumps(qs))
    chains = tmp_path / "chains.jsonl"
    chain = {"video_id": "video_10000.mp4", "question": "q", "final_answer": "0", "question_type": "descriptive",
             "meta": {"scene_index": 10000, "question_id": 0},
             "triplets": [{"question": "w", "answer": "a", "depends_on": [], "evidence": {"t_start": 45, "t_end": 45, "box": [0.0, 0.0, 0.5, 0.5]}}]}
    chains.write_text(json.dumps(chain) + "\n")
    return root, chains


def dark_pixels(frames):
    return sum(int(np.asarray(f)[..., 0].min() < 50) for f in frames)


def test_plain_with_chains_does_not_mask_but_evidence_mode_does(tmp_path, monkeypatch):
    root, chains = setup(tmp_path)
    from causalsight import models

    backends = {}

    def fake_load(model, **kw):
        backends[model] = RecordingBackend()
        return backends[model]

    monkeypatch.setattr(models, "load_backend", fake_load)
    import causalsight.eval.harness as H

    monkeypatch.setattr(H, "load_backend", fake_load)

    run("p", "clevrer", "validation", root, None, None, False, 16, mask="none", chains=chains)
    run("e", "clevrer", "validation", root, None, None, False, 16, mask="evidence", chains=chains)
    run("r", "clevrer", "validation", root, None, None, False, 16, mask="random", chains=chains, seed=3)
    assert dark_pixels(backends["p"].frames[0]) == 0
    # span [45,45] with tolerance 4 touches sampled frames 42 only (42..51 -> |45-42|=3, |51-45|=6)
    assert span_tolerance(128, 16) == 4
    assert dark_pixels(backends["e"].frames[0]) == 1
    assert dark_pixels(backends["r"].frames[0]) == 1
    e_frame = np.asarray(backends["e"].frames[0][5])  # sampled index 5 -> original frame 42
    assert e_frame[0, 0, 0] < 50 and e_frame[-1, -1, 0] > 200  # top-left box masked, bottom-right untouched
