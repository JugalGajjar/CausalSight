"""Finer cuts of the self-citation results, from saved per-item outputs (no model, no detector).

  cs-faithfulness-splits results/colab --tag mixsft3b_nq [--tracks results/colab/nextqa_entity_tracks.json]

writes <tag>_splits.json with

  by_gold    every gap separately for the items whose option is correct (gold "yes") and wrong (gold "no").
             A correct rejection rarely flips under any mask, and 80% of NExT-QA per-option items are
             rejections, so the pooled gaps are diluted; the "yes" rows are the affirmations.
  by_hit     (with --tracks) cite gap and entity-track gap split by whether every cited "Where is <entity>?"
             box overlaps the detector's box for that entity on the cited frame at IoU >= 0.5.
  area       mean share of pixel-frames blacked out under each masked condition.
  recite     what the chain written under the mask cites: the blacked-out box again, another place on a
             masked frame, or a frame the mask did not touch; and the same for the random-mask run, where
             "the box cited before" is still visible (a baseline for how stable citations are).

Gaps are flip-rate differences over the items correct under the plain input, with the question-clustered
paired bootstrap of the main report.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from causalsight.eval.cite_grounding import N_FRAMES, entity_of, frame_of
from causalsight.eval.faithfulness.report import _ci, load
from causalsight.eval.masking import CHAIN_SCALE, WHOLE_FRAME, _iou, sampled_frame_indices, span_tolerance
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
    return {"n_correct": len(cp), "flip": flip(a, ids), "flip_control": flip(b, ids), "gap": gap(ids), "gap_ci": [lo, hi]}


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


def splits(dir_: Path, tag: str, tracks_path: Path | None = None) -> dict:
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

    if "mask_cite" in runs:
        cids = [i for i in ids if i in runs["mask_cite"]]
        out["recite"] = {"cite": recite(plain, runs["mask_cite"], cids)}
        if "mask_cite_random" in runs:
            out["recite"]["cite_random"] = recite(plain, runs["mask_cite_random"], cids)

    if tracks_path and Path(tracks_path).exists():
        tracks = json.loads(Path(tracks_path).read_text())
        tracks.pop("_summary", None)
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
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tracks", type=Path, default=None, help="entity tracks json (cs-entity-tracks): adds the split by citation hit")
    a = ap.parse_args()
    res = splits(a.dir, a.tag, a.tracks)
    (a.dir / f"{a.tag}_splits.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
