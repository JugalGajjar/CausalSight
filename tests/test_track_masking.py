import json
import random

from causalsight.eval.masking import TrackMasker


def rle_rect(h, w, x0, y0, x1, y1):
    runs = []
    val = 0
    run = 0
    for c in range(w):
        for r in range(h):
            v = 1 if (x0 <= c < x1 and y0 <= r < y1) else 0
            if v == val:
                run += 1
            else:
                runs.append(run)
                val = v
                run = 1
    runs.append(run)
    return {"size": [h, w], "counts": runs}


def write_proposals(tmp_path, scene=7, n_frames=16):
    d = tmp_path / "props"
    d.mkdir()
    frames = []
    for f in range(n_frames):
        objs = [
            {"mask": rle_rect(20, 40, 2, 2, 10, 10), "color": "red", "material": "metal", "shape": "cube", "score": 0.9},
            {"mask": rle_rect(20, 40, 25, 8, 35, 18), "color": "blue", "material": "rubber", "shape": "sphere", "score": 0.9},
        ]
        frames.append({"frame_index": f, "objects": objs})
    (d / f"proposal_{scene:05d}.json").write_text(json.dumps({"frames": frames}))
    return d


def test_track_masker_matches_object_and_control_picks_other(tmp_path):
    root = write_proposals(tmp_path)
    tm = TrackMasker(root)
    # evidence: the red cube at frame 3 (box in normalized coords of a 40x20 frame)
    regions = [(3, 3, (2 / 40, 2 / 20, 10 / 40, 10 / 20))]
    assert tm.evidence_objects(7, regions) == {("red", "metal", "cube")}
    ev = tm.regions_for(7, regions, n_total=16, n_sampled=16, control=False, rng=random.Random(0))
    assert len(ev) == 16 and all(abs(r[2][0] - 2 / 40) < 1e-9 for r in ev)  # red cube on every sampled frame
    ctrl = tm.regions_for(7, regions, n_total=16, n_sampled=16, control=True, rng=random.Random(0))
    assert len(ctrl) == 16 and all(abs(r[2][0] - 25 / 40) < 1e-9 for r in ctrl)  # the blue sphere instead
    # both objects as evidence -> no same-size control -> empty (caller skips the item)
    both = regions + [(3, 3, (25 / 40, 8 / 20, 35 / 40, 18 / 20))]
    assert tm.regions_for(7, both, n_total=16, n_sampled=16, control=True, rng=random.Random(0)) == []
