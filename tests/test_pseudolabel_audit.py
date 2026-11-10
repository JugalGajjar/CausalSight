from causalsight.data.pseudolabel_audit import summarize


def test_summarize_rates_and_model_blocks(tmp_path):
    rows = [("c0_s0_visible", "yes"), ("c0_s0_onbox", "yes"), ("c0_s0_relevant", "yes"), ("c0_s1_visible", "no"), ("c0_s1_onbox", "no"), ("c0_s1_relevant", "yes"),
            ("c0_sufficient", "yes"), ("c0_causal", "no"), ("c0_consistent", "yes"), ("c0_note", '"a, b"'),
            ("c1_s0_visible", "yes"), ("c1_s0_onbox", "no"), ("c1_s0_relevant", "unsure"), ("c1_sufficient", "no"), ("c1_causal", "no"), ("c1_consistent", "unsure")]
    p = tmp_path / "a.csv"
    p.write_text("name,value\n" + "\n".join(f"{k},{v}" for k, v in rows) + "\n")
    s = summarize(p)
    assert s["visible"] == {"n": 3, "yes": 0.667, "no": 0.333, "unsure": 0.0}
    assert s["onbox_given_visible"]["n"] == 2 and s["onbox_given_visible"]["yes"] == 0.5
    assert s["consistent"]["n"] == 2
    g = summarize(p, n_each=1, names=["m0", "m1"])
    assert g["all"] == s and g["m0"]["visible"]["n"] == 2 and g["m1"]["onbox"]["no"] == 1.0
    # a sheet without the model-only question has no "consistent" entry
    q = tmp_path / "b.csv"
    q.write_text("name,value\nc0_s0_visible,yes\nc0_s0_onbox,yes\nc0_s0_relevant,yes\nc0_sufficient,yes\nc0_causal,no\n")
    assert "consistent" not in summarize(q)
