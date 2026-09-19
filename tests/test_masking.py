import random

from PIL import Image

from causalsight.eval.masking import (
    mask_frames,
    masked_fraction,
    random_regions,
    sampled_frame_indices,
)


def test_sampled_indices():
    assert sampled_frame_indices(128, 16)[0] == 0 and sampled_frame_indices(128, 16)[-1] == 127
    assert sampled_frame_indices(10, 16) == list(range(10))


def test_mask_frames_only_inside_span():
    frames = [Image.new("RGB", (48, 32), (255, 255, 255)) for _ in range(16)]
    regions = [(0, 10, (0.0, 0.0, 0.5, 0.5))]  # only sampled frames 0 (orig 0) and 1 (orig 8) fall inside
    out = mask_frames(frames, regions, 128)
    assert out[0].getpixel((5, 5)) == (0, 0, 0) and out[0].getpixel((40, 25)) == (255, 255, 255)
    assert out[1].getpixel((5, 5)) == (0, 0, 0)
    assert out[2].getpixel((5, 5)) == (255, 255, 255)
    assert frames[0].getpixel((5, 5)) == (255, 255, 255)  # input untouched


def test_random_regions_preserve_size_and_span():
    regs = [(3, 9, (0.2, 0.3, 0.6, 0.5))]
    r = random_regions(regs, random.Random(1))[0]
    assert (r[0], r[1]) == (3, 9)
    assert abs((r[2][2] - r[2][0]) - 0.4) < 1e-9 and abs((r[2][3] - r[2][1]) - 0.2) < 1e-9
    assert 0 <= r[2][0] and r[2][2] <= 1 and 0 <= r[2][1] and r[2][3] <= 1


def test_masked_fraction():
    assert abs(masked_fraction([(0, 63, (0.0, 0.0, 0.5, 0.5))], 128) - 0.125) < 1e-9


def test_evidence_index_per_option_lookup(tmp_path):
    import json

    from causalsight.eval.masking import EvidenceIndex

    def chain(cid, box):
        return {"question_type": "predictive", "meta": {"scene_index": 1, "question_id": 2, "choice_id": cid},
                "triplets": [{"question": "q", "role": "observe", "evidence": {"t_start": 3, "t_end": 3, "box": box}}]}

    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in (chain(0, [0.1, 0.1, 0.2, 0.2]), chain(1, [0.5, 0.5, 0.6, 0.6]))) + "\n")
    idx = EvidenceIndex.from_chains(p)
    assert len(idx.get("1_2")) == 2  # pooled over options
    assert idx.get_for_item("1_2_1", "1_2") == [(3, 3, (0.5, 0.5, 0.6, 0.6))]  # that option only
    assert len(idx.get_for_item("1_2", "1_2")) == 2  # multi-mode item falls back to pooled
