"""Finer cuts of the self-citation results, from saved per-item outputs (no model, no detector).

  cs-faithfulness-splits results/colab --tag mixsft3b_nq [--tracks results/colab/nextqa_entity_tracks.json]

writes <tag>_splits.json with

  by_gold    every gap separately for the items whose option is correct (gold "yes") and wrong (gold "no").
             A correct rejection rarely flips under any mask, and 80% of NExT-QA per-option items are
             rejections, so the pooled gaps are diluted; the "yes" rows are the affirmations.
  by_hit     (with --tracks) cite gap and entity-track gap split by whether every cited "Where is <entity>?"
             box overlaps the detector's box for that entity on the cited frame at IoU >= 0.5.
  area       mean share of pixel-frames blacked out under each masked condition.
  by_text    cite gap and entity-track gap split by whether every entity the chain names already appears in
             the question or option text (the entity is the question's subject) or not.
  control    how much of the cited area the random control also blacks out. The control boxes are not
             stored, but they are a seeded draw in output order, so they are replayed exactly; the cite gap
             is then recomputed on the items whose control does not touch the cited boxes.
  frames     mean number of sampled frames a cited region is applied to.
  by_blind   every gap on the video-dependent items (correct under the plain input, wrong with all frames
             black) and on the rest; the flip rate there divided by 1 is the share of the video dependence
             that the intervention removes.
  symmetric  wrong-to-correct flips and the net change in accuracy under each mask and its control.
  selection  (per-option benchmarks) question-level flips: questions whose selected option is right under the
             plain input and no longer right under the mask or its control.
  dose       (with --tracks) cite gap by how many of the 16 frames show the named entities and by the share
             of those frames the citations cover: the redundancy account predicts a larger gap where the
             entity is rarer or the citation covers more of it.
  clevrer    (with --proposals) cite gap by whether every cited box is on the object it names, and by whether
             the random control lands on an object or only on background.
  recite     what the chain written under the mask cites: the blacked-out box again, another place on a
             masked frame, or a frame the mask did not touch; and the same for the random-mask run, where
             "the box cited before" is still visible (a baseline for how stable citations are).

Gaps are flip-rate differences over the items correct under the plain input, with the question-clustered
paired bootstrap of the main report; `flip_ci` is the interval of the intervention's flip rate itself, an
upper bound on any gap the intervention could show.

  cs-faithfulness-splits results/colab --pool mixsft3b_nq mixsft3b_s1_nq ... --name nq_real

pools the items of several models (clusters stay questions) and writes <name>_pooled.json.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from causalsight.eval.cite_grounding import N_FRAMES, entity_of, frame_of
from causalsight.eval.faithfulness.report import _ci, load
from causalsight.eval.masking import CHAIN_SCALE, WHOLE_FRAME, EvidenceIndex, _iou, random_regions, sampled_frame_indices, span_tolerance
from causalsight.train.format import parse_chain

PAIRS = {"cite": ("mask_cite", "mask_cite_random"), "entity_track": ("mask_cite_track", "mask_cite_track_random"), "gold_evidence": ("mask_evidence", "mask_random"), "object_track": ("mask_track", "mask_track_random")}


def cited_steps(raw: str) -> list[tuple[str | None, int, int, tuple]]:
    """(entity or None, t_start, t_end, box) for every parsed step with a non-whole-frame box."""
    p = parse_chain(raw)
    out = []
    if not p.chain:
        return out
    for t in p.chain.triplets:
        b = tuple(float(v) for v in t.evidence.box)
        if b == WHOLE_FRAME:
            continue
        e = entity_of(t.question)
        out.append((e.lower() if e else None, int(t.evidence.t_start), int(t.evidence.t_end), b))
    return out


def frames_of_span(t0: int, t1: int, n_sampled: int = N_FRAMES, n_total: int = CHAIN_SCALE) -> set[int]:
    idx = sampled_frame_indices(n_total, n_sampled)
    tol = span_tolerance(n_total, n_sampled)
    return {f for f, orig in enumerate(idx) if t0 - tol <= orig <= t1 + tol}


def gap_stats(runs: dict, a: str, b: str, ids: list[str], cluster: dict[str, str], rng: random.Random) -> dict | None:
    """Flip rates of conditions a and b over the items of `ids` correct under plain, their difference and its CI."""
    ids = sorted(i for i in ids if i in runs[a] and i in runs[b])
    cp = [i for i in ids if runs["plain"][i]["correct"]]
    if not cp:
        return None

    def flip(k: str, ids_: list[str]) -> float | None:
        c = [i for i in ids_ if runs["plain"][i]["correct"]]
        return None if not c else sum(1 for i in c if not runs[k][i]["correct"]) / len(c)

    def gap(ids_: list[str]) -> float | None:
        x, y = flip(a, ids_), flip(b, ids_)
        return None if x is None else x - y

    lo, hi = _ci(gap, ids, rng, cluster=cluster)
    flo, fhi = _ci(lambda ids_: flip(a, ids_), ids, rng, cluster=cluster)
    return {"n_correct": len(cp), "flip": flip(a, ids), "flip_ci": [flo, fhi], "flip_control": flip(b, ids), "gap": gap(ids), "gap_ci": [lo, hi]}


def hit_status(raw: str, video_tracks: dict, thr: float = 0.5) -> str | None:
    """'all' if every "Where is <entity>?" step's box overlaps the detection of that entity on the cited frame
    at IoU >= thr, 'not_all' otherwise (a step with no detection on its frame is a miss), None if no such step."""
    hits = []
    for e, t0, t1, b in cited_steps(raw):
        if e is None:
            continue
        det = video_tracks.get(e, {}).get(str(frame_of(t0, t1)))
        hits.append(det is not None and _iou(b, tuple(det)) >= thr)
    if not hits:
        return None
    return "all" if all(hits) else "not_all"


STOP = {"the", "a", "an", "of", "in", "on", "at", "with", "and", "to", "is", "his", "her", "their", "its", "s"}


def _tokens(text: str) -> list[str]:
    return [w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if w not in STOP]


def entities_in_text(raw: str, prompt: str, mode: str = "all") -> bool | None:
    """mode 'all': True if every entity the chain names has all its content words in the question/option text.
    mode 'none': True if no named entity shares a content word with that text."""
    ents = {e for e, *_ in cited_steps(raw) if e}
    if not ents:
        return None
    words = set(_tokens(prompt.split("Is this option correct?")[0]))
    if mode == "none":
        return not any(set(_tokens(e)) & words for e in ents)
    return all(set(_tokens(e)) <= words for e in ents)


def _overlap_share(box: tuple, other: tuple) -> float:
    """Share of `box` covered by `other`."""
    w = min(box[2], other[2]) - max(box[0], other[0])
    h = min(box[3], other[3]) - max(box[1], other[1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    return max(0.0, w) * max(0.0, h) / area if area > 0 else 0.0


def replay_controls(plain_path: Path, control_rows: dict, seed: int = 0) -> dict[str, tuple[list, list]]:
    """Replays the seeded random control of a cite_random run (items in output order, as the harness drew
    them): item id -> (cited regions, control regions)."""
    ev = EvidenceIndex.from_outputs(plain_path)
    rng = random.Random(seed)
    out = {}
    for i in control_rows:  # file order
        regs = ev.get_cited(i)
        out[i] = (regs, random_regions(regs, rng))
    return out


def control_coverage(plain_path: Path, control_rows: dict, seed: int = 0) -> dict[str, float]:
    """Per item, the mean share of each cited box that a control box blacks out on the frames where both apply."""
    out: dict[str, float] = {}
    for i, (regs, ctrl) in replay_controls(plain_path, control_rows, seed).items():
        shares = []
        for t0, t1, b in regs:
            fs = frames_of_span(t0, t1)
            best = 0.0
            for c0, c1, cb in ctrl:
                if fs & frames_of_span(c0, c1):
                    best = max(best, _overlap_share(tuple(b), tuple(cb)))
            shares.append(best)
        if shares:
            out[i] = sum(shares) / len(shares)
    return out


def pooled(dir_: Path, tags: list[str]) -> dict:
    """Gaps over the pooled items of several models; a question's items from every model form one cluster."""
    out: dict = {"tags": tags}
    rng = random.Random(0)
    for name, (a, b) in PAIRS.items():
        runs: dict = {"plain": {}, a: {}, b: {}}
        cluster: dict[str, str] = {}
        blind: dict = {}
        for t in tags:
            if not all((dir_ / f"{t}_{k}.jsonl").exists() for k in ("plain", a, b)):
                continue
            pl = load(dir_ / f"{t}_plain.jsonl")
            if (dir_ / f"{t}_blind.jsonl").exists():
                for i, r in load(dir_ / f"{t}_blind.jsonl").items():
                    blind[f"{t}|{i}"] = r
            for k in (a, b):
                for i, r in load(dir_ / f"{t}_{k}.jsonl").items():
                    runs[k][f"{t}|{i}"] = r
            for i, r in pl.items():
                runs["plain"][f"{t}|{i}"] = r
                cluster[f"{t}|{i}"] = str(r.get("qid") or r.get("qset") or i)
        if not runs[a]:
            continue
        ids = sorted(runs["plain"])
        row = {"all": gap_stats(runs, a, b, ids, cluster, rng)}
        for g in ("yes", "no"):
            row[g] = gap_stats(runs, a, b, [i for i in ids if str(runs["plain"][i].get("gold")) == g], cluster, rng)
        vd = [i for i in ids if blind.get(i) is not None and runs["plain"][i]["correct"] and not blind[i]["correct"]]
        row["video_dependent"] = gap_stats(runs, a, b, vd, cluster, rng)
        row["video_dependent_yes"] = gap_stats(runs, a, b, [i for i in vd if str(runs["plain"][i].get("gold")) == "yes"], cluster, rng)
        cp = [i for i in ids if runs["plain"][i]["correct"] and i in runs[a] and i in runs[b] and blind.get(i) is not None]
        row["blind_flip"] = sum(1 for i in cp if not blind[i]["correct"]) / len(cp) if cp else None
        out[name] = row
    return out


def recite(plain: dict, masked: dict, ids: list[str], thr: float = 0.5) -> dict:
    """Where the chain written under a mask cites, relative to the boxes the plain run cited."""
    c: Counter = Counter()
    same_entities = n_items = 0
    for i in ids:
        if i not in masked:
            continue
        old = cited_steps(plain[i].get("raw", ""))
        new = cited_steps(masked[i].get("raw", ""))
        if not old or not new:
            c["items_without_citation_under_mask"] += bool(old and not new)
            continue
        n_items += 1
        old_by_frame: dict[int, list] = defaultdict(list)
        for _e, t0, t1, b in old:
            for f in frames_of_span(t0, t1):
                old_by_frame[f].append(b)
        eo, en = {e for e, *_ in old if e}, {e for e, *_ in new if e}
        if eo or en:
            c["items_with_entities"] += 1
            same_entities += eo == en
        for _e, t0, t1, b in new:
            f = frame_of(t0, t1)
            c["steps"] += 1
            if f not in old_by_frame:
                c["other_frame"] += 1
            elif max(_iou(b, ob) for ob in old_by_frame[f]) >= thr:
                c["same_box"] += 1
            else:
                c["same_frame_other_place"] += 1
    s = c["steps"] or 1
    return {
        "items": n_items,
        "steps": c["steps"],
        "same_box": c["same_box"] / s,
        "same_frame_other_place": c["same_frame_other_place"] / s,
        "other_frame": c["other_frame"] / s,
        "same_entities": same_entities / c["items_with_entities"] if c["items_with_entities"] else None,
        "items_without_citation_under_mask": c["items_without_citation_under_mask"],
    }


def splits(dir_: Path, tag: str, tracks_path: Path | None = None, proposals: Path | None = None) -> dict:
    names = ["plain", "blind"] + [k for pair in PAIRS.values() for k in pair]
    runs = {k: load(dir_ / f"{tag}_{k}.jsonl") for k in names if (dir_ / f"{tag}_{k}.jsonl").exists()}
    if "plain" not in runs:
        raise SystemExit(f"missing {tag}_plain.jsonl in {dir_}")
    plain = runs["plain"]
    ids = sorted(plain)
    cluster = {i: str(plain[i].get("qid") or plain[i].get("qset") or i) for i in ids}
    rng = random.Random(0)
    out: dict = {"tag": tag}

    by_gold: dict = {}
    for g in ("yes", "no"):
        gids = [i for i in ids if str(plain[i].get("gold")) == g]
        row: dict = {"n": len(gids)}
        if "blind" in runs:
            cp = [i for i in gids if plain[i]["correct"] and i in runs["blind"]]
            row["blind_flip"] = sum(1 for i in cp if not runs["blind"][i]["correct"]) / len(cp) if cp else None
        for name, (a, b) in PAIRS.items():
            if a in runs and b in runs:
                row[name] = gap_stats(runs, a, b, gids, cluster, rng)
        by_gold[g] = row
    out["by_gold"] = by_gold

    out["area"] = {}
    for k, rows in runs.items():
        if not k.startswith("mask_"):
            continue
        mf = [r["masked_fraction"] for r in rows.values() if "masked_fraction" in r]
        if mf:
            out["area"][k] = sum(mf) / len(mf)

    if "blind" in runs:
        bl = runs["blind"]
        vd = [i for i in ids if i in bl and plain[i]["correct"] and not bl[i]["correct"]]
        nvd = [i for i in ids if i in bl and plain[i]["correct"] and bl[i]["correct"]]
        by_blind: dict = {"n_video_dependent": len(vd), "n_not": len(nvd)}
        for name, (a, b) in PAIRS.items():
            if a in runs and b in runs:
                by_blind[name] = {"video_dependent": gap_stats(runs, a, b, vd, cluster, rng), "not": gap_stats(runs, a, b, nvd, cluster, rng),
                                  "video_dependent_yes": gap_stats(runs, a, b, [i for i in vd if str(plain[i].get("gold")) == "yes"], cluster, rng)}
        out["by_blind"] = by_blind

    sym: dict = {}
    for name, (a, b) in PAIRS.items():
        if a in runs and b in runs:
            both = [i for i in ids if i in runs[a] and i in runs[b]]
            wrong = [i for i in both if not plain[i]["correct"]]
            acc = lambda k: sum(runs[k][i]["correct"] for i in both) / len(both)
            sym[name] = {"n": len(both), "n_wrong": len(wrong),
                         "wrong_to_right": sum(runs[a][i]["correct"] for i in wrong) / len(wrong) if wrong else None,
                         "wrong_to_right_control": sum(runs[b][i]["correct"] for i in wrong) / len(wrong) if wrong else None,
                         "net": acc(a) - sum(plain[i]["correct"] for i in both) / len(both), "net_control": acc(b) - sum(plain[i]["correct"] for i in both) / len(both)}
    out["symmetric"] = sym

    if all("qset" in r and "pred" in r for r in plain.values()):
        qs: dict[str, list[str]] = defaultdict(list)
        for i in ids:
            qs[str(plain[i]["qset"])].append(i)

        def credit(rows: dict, items: list[str]) -> float:
            yes = [i for i in items if rows[i]["pred"] == "yes"]
            return 1 / len(items) if not yes else (1 / len(yes) if any(str(plain[i]["gold"]) == "yes" for i in yes) else 0.0)

        sel: dict = {}
        for name, (a, b) in PAIRS.items():
            if a in runs and b in runs:
                full = [q for q, items in qs.items() if all(i in runs[a] and i in runs[b] for i in items) and credit(plain, items) == 1.0]
                if full:
                    sel[name] = {"n_questions": len(full), "flip": sum(credit(runs[a], qs[q]) < 1.0 for q in full) / len(full), "flip_control": sum(credit(runs[b], qs[q]) < 1.0 for q in full) / len(full)}
        out["selection"] = sel

    spans = [len(frames_of_span(t0, t1)) for i in ids for _e, t0, t1, _b in cited_steps(plain[i].get("raw", ""))]
    out["frames"] = sum(spans) / len(spans) if spans else None

    if "mask_cite" in runs and "mask_cite_random" in runs:
        cov = control_coverage(dir_ / f"{tag}_plain.jsonl", runs["mask_cite_random"])
        vals = list(cov.values())
        if vals:
            ctl = {"mean_covered": sum(vals) / len(vals), "share_untouched": sum(v == 0 for v in vals) / len(vals), "share_half_covered": sum(v >= 0.5 for v in vals) / len(vals)}
            ctl["untouched"] = gap_stats(runs, "mask_cite", "mask_cite_random", [i for i in ids if cov.get(i) == 0], cluster, rng)
            ctl["untouched_yes"] = gap_stats(runs, "mask_cite", "mask_cite_random", [i for i in ids if cov.get(i) == 0 and str(plain[i].get("gold")) == "yes"], cluster, rng)
            out["control"] = ctl

    in_text = {i: entities_in_text(plain[i].get("raw", ""), plain[i].get("prompt", "")) for i in ids}
    if any(v is not None for v in in_text.values()):
        by_text: dict = {}
        for key, val in (("in_text", True), ("not_in_text", False)):
            tids = [i for i in ids if in_text[i] is val]
            row = {"n": len(tids)}
            for name in ("cite", "entity_track"):
                a, b = PAIRS[name]
                if a in runs and b in runs:
                    row[name] = gap_stats(runs, a, b, tids, cluster, rng)
                    row[name + "_yes"] = gap_stats(runs, a, b, [i for i in tids if str(plain[i].get("gold")) == "yes"], cluster, rng)
            by_text[key] = row
        tids = [i for i in ids if entities_in_text(plain[i].get("raw", ""), plain[i].get("prompt", ""), "none") is True]
        row = {"n": len(tids)}
        for name in ("cite", "entity_track"):
            a, b = PAIRS[name]
            if a in runs and b in runs:
                row[name] = gap_stats(runs, a, b, tids, cluster, rng)
                row[name + "_yes"] = gap_stats(runs, a, b, [i for i in tids if str(plain[i].get("gold")) == "yes"], cluster, rng)
        by_text["no_overlap"] = row
        out["by_text"] = by_text

    if "mask_cite" in runs:
        cids = [i for i in ids if i in runs["mask_cite"]]
        out["recite"] = {"cite": recite(plain, runs["mask_cite"], cids)}
        if "mask_cite_random" in runs:
            out["recite"]["cite_random"] = recite(plain, runs["mask_cite_random"], cids)
            # stability of the citation under an irrelevant mask, only where the answer itself is unchanged
            # (prediction-preserving filter): otherwise a moved box could reflect a changed answer
            key = "pred" if all("pred" in plain[i] for i in cids[:1]) else "correct"
            same = [i for i in cids if i in runs["mask_cite_random"] and runs["mask_cite_random"][i].get(key) == plain[i].get(key)]
            out["recite"]["cite_random_same_answer"] = {**recite(plain, runs["mask_cite_random"], same), "share_items": len(same) / len(cids) if cids else None}

    if tracks_path and Path(tracks_path).exists():
        tracks = json.loads(Path(tracks_path).read_text())
        tracks.pop("_summary", None)
        dose: dict = {"by_frames": {}, "by_cover": {}}
        nfr: dict[str, float] = {}
        cover: dict[str, float] = {}
        for i in ids:
            vt = tracks.get(plain[i].get("frames_key") or i.split("_")[0], {})
            steps = [st for st in cited_steps(plain[i].get("raw", "")) if st[0]]
            dets = [set(int(f) for f in vt.get(e, {})) for e in {st[0] for st in steps}]
            dets = [d for d in dets if d]
            if not dets:
                continue
            allf = set().union(*dets)
            cited_f = set().union(*(frames_of_span(t0, t1) for _e, t0, t1, _b in steps))
            nfr[i] = sum(len(d) for d in dets) / len(dets)
            cover[i] = len(cited_f & allf) / len(allf)
        for label, lo, hi in (("1-8", 0, 8.5), ("9-13", 8.5, 13.5), ("14-15", 13.5, 15.5), ("16", 15.5, 99)):
            b_ids = [i for i, v in nfr.items() if lo <= v < hi]
            dose["by_frames"][label] = {"n": len(b_ids), "cite": gap_stats(runs, "mask_cite", "mask_cite_random", b_ids, cluster, rng) if "mask_cite" in runs else None,
                                        "entity_track": gap_stats(runs, "mask_cite_track", "mask_cite_track_random", b_ids, cluster, rng) if "mask_cite_track" in runs else None}
        for label, lo, hi in (("<0.15", 0, 0.15), ("0.15-0.3", 0.15, 0.3), (">=0.3", 0.3, 9)):
            b_ids = [i for i, v in cover.items() if lo <= v < hi]
            dose["by_cover"][label] = {"n": len(b_ids), "cite": gap_stats(runs, "mask_cite", "mask_cite_random", b_ids, cluster, rng) if "mask_cite" in runs else None}
        out["dose"] = dose
        status = {i: hit_status(plain[i].get("raw", ""), tracks.get(plain[i].get("frames_key") or i.split("_")[0], {})) for i in ids}
        by_hit: dict = {}
        for s in ("all", "not_all"):
            sids = [i for i in ids if status[i] == s]
            row = {"n": len(sids)}
            for name in ("cite", "entity_track"):
                a, b = PAIRS[name]
                if a in runs and b in runs:
                    row[name] = gap_stats(runs, a, b, sids, cluster, rng)
            by_hit[s] = row
        out["by_hit"] = by_hit
    if proposals and Path(proposals).exists() and "mask_cite" in runs and "mask_cite_random" in runs:
        from causalsight.data.clevrer_evidence import ProposalIndex
        from causalsight.eval.object_grounding import step_scores

        idx_cache: dict[int, object] = {}

        def pidx(i: str):
            scene = int(i.split("_")[0])
            if scene not in idx_cache:
                if len(idx_cache) > 64:
                    idx_cache.clear()
                idx_cache[scene] = ProposalIndex.load(Path(proposals) / f"proposal_{scene:05d}.json")
            return idx_cache[scene]

        grid = sampled_frame_indices(CHAIN_SCALE, N_FRAMES)
        good, bad, on_obj, on_bg = [], [], [], []
        ctrl_of = replay_controls(dir_ / f"{tag}_plain.jsonl", runs["mask_cite_random"])
        for i in sorted(ids, key=lambda x: int(x.split("_")[0])):
            if i not in runs["mask_cite"] or i not in runs["mask_cite_random"]:
                continue
            ch = parse_chain(plain[i].get("raw", "")).chain
            sc = step_scores(ch, pidx(i)) if ch else []
            if sc:
                (good if all(x["iou"] >= 0.5 for x in sc) else bad).append(i)
            hit = False
            for t0, t1, cb in ctrl_of.get(i, ([], []))[1]:
                for f in frames_of_span(t0, t1):
                    for (ob, _score) in pidx(i)._frame(grid[f]).values():
                        if _overlap_share(tuple(ob), tuple(cb)) >= 0.3:  # the control covers at least 30% of an object
                            hit = True
            (on_obj if hit else on_bg).append(i)
        g = lambda sub_: gap_stats(runs, "mask_cite", "mask_cite_random", sub_, cluster, rng)
        out["clevrer"] = {"hit_all": {"n": len(good), "cite": g(good)}, "hit_not_all": {"n": len(bad), "cite": g(bad)},
                          "control_on_object": {"n": len(on_obj), "cite": g(on_obj)}, "control_on_background": {"n": len(on_bg), "cite": g(on_bg)}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--proposals", type=Path, default=None, help="CLEVRER derender proposals dir: adds the citation-correctness and control-strata splits")
    ap.add_argument("--pool", nargs="+", default=None, help="tags to pool (writes <name>_pooled.json)")
    ap.add_argument("--name", default="pooled")
    ap.add_argument("--tracks", type=Path, default=None, help="entity tracks json (cs-entity-tracks): adds the split by citation hit")
    a = ap.parse_args()
    if a.pool:
        res = pooled(a.dir, a.pool)
        (a.dir / f"{a.name}_pooled.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    if not a.tag:
        raise SystemExit("--tag or --pool is required")
    res = splits(a.dir, a.tag, a.tracks, a.proposals)
    (a.dir / f"{a.tag}_splits.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
