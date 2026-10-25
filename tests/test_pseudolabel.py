from causalsight.data.pseudolabel import parse_teacher


def test_parse_teacher_formats():
    strict = "[1] entity: the man in the blue sweater | observation: he is smiling | frame: 3\n[2] entity: the shirtless men | observation: play-fighting | frame: 7\n[3] entity: a dog | observation: runs | frame: 19\n"
    assert [s["frame"] for s in parse_teacher(strict)] == [3, 7]  # frame 19 out of range
    loose = "2. dog: the black dog | observation: the black dog is running towards the camera. | frame: 2\n3. dog: the brown dog | observation: running too | frame: 2\n4. stick: the black stick | observation: lying on the snow | frame: 4\n"
    steps = parse_teacher(loose)
    assert [s["entity"] for s in steps] == ["the black dog", "the brown dog", "the black stick"]
    dash = "- the red cup | observation: on the table | frame: 0"
    assert parse_teacher(dash)[0]["entity"] == "the red cup"
    assert parse_teacher("nothing useful") == []
