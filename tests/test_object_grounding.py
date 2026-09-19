from causalsight.data.schema import Evidence, Triplet, TripletChain
from causalsight.eval.object_grounding import named_objects, step_scores


class FakeIdx:
    """Red metal cube sits at (0.1,0.1,0.3,0.3) on frames 0-9 and at (0.6,0.6,0.8,0.8) on frames 10+."""

    def box(self, attrs, frame, tol=3):
        if (attrs["color"], attrs["material"], attrs["shape"]) != ("red", "metal", "cube"):
            return None
        return ((0.1, 0.1, 0.3, 0.3), frame) if frame < 10 else ((0.6, 0.6, 0.8, 0.8), frame)


def test_named_objects():
    assert named_objects("the collision between the red metal cube and the blue rubber sphere at frame 4") == [
        {"color": "red", "material": "metal", "shape": "cube"}, {"color": "blue", "material": "rubber", "shape": "sphere"}]
    assert named_objects("none") == [] and named_objects("yes, at frame 3") == []


def test_step_scores_use_the_cited_frame():
    ch = TripletChain("v", "q", [
        Triplet("Which object is the cube?", "the red metal cube", Evidence(50, 50, (0.6, 0.6, 0.8, 0.8))),  # right box for frame 50
        Triplet("Which object is the cube?", "the red metal cube", Evidence(50, 50, (0.1, 0.1, 0.3, 0.3))),  # frame-0 box cited at frame 50
        Triplet("How many?", "2", Evidence(0, 0, (0.0, 0.0, 0.5, 0.5))),  # names no object: not scored
        Triplet("Which?", "the green rubber sphere", Evidence(5, 5, (0.0, 0.0, 0.5, 0.5))),  # never detected
    ], "x")
    s = step_scores(ch, FakeIdx())
    assert len(s) == 3
    assert abs(s[0]["iou"] - 1.0) < 1e-9 and s[0]["visible"]
    assert s[1]["iou"] == 0.0 and s[1]["visible"]
    assert s[2]["iou"] == 0.0 and not s[2]["visible"]
