from causalsight.data.pseudolabel import parse_teacher


def test_parse_teacher_steps():
    txt = ("[1] entity: the man in the blue sweater | observation: he is looking at the shirtless men and smiling | frame: 3\n"
           "[2] entity: the shirtless men | observation: they are play-fighting on the grass | frame: 7\n"
           "[3] entity: a dog | observation: runs past | frame: 19\n")  # frame out of range -> dropped
    steps = parse_teacher(txt)
    assert len(steps) == 2 and steps[0]["frame"] == 3 and steps[0]["entity"] == "the man in the blue sweater"
    assert parse_teacher("no steps here") == []
