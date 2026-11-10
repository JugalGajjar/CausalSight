"""Manual audit sheet for pseudo-labelled NExT-QA chains.

  cs-pseudolabel-audit <real_sft.jsonl> --frames data/frames/nextqa_pack --out results/mixed/pseudolabel_audit.html [--n 100]

Samples n chains (one per question, the "yes" item, seed 0) and writes one self-contained HTML page: for every
step the cited frame (the sampled frame nearest the span's centre, as in the grounding check) with the detector box drawn, the entity the teacher named and its observation, and three
yes/no judgements per step (entity visible on this frame; box on the entity; step relevant to the question),
plus one per chain (the steps together are enough to answer). A button at the top downloads the judgements as CSV
(`cs-pseudolabel-audit summarize <csv>` turns it into the numbers for the paper). Images are embedded at 320 px
wide, so the page is ~5 MB and needs nothing but a browser.

  cs-pseudolabel-audit build-model <tag>_plain.jsonl ... --frames data/frames/nextqa_pack --out results/mixed/model_citation_audit.html

does the same for trained models' own chains on validation items (n-each per file, one per question), with one more
chain-level judgement (the steps support the answer the model gave); `summarize <csv> --n-each 50 --names a b`
reports each model separately.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from causalsight.eval.cite_grounding import frame_of

STEP_Q = [("visible", "entity visible on this frame"), ("onbox", "box is on the named entity"), ("relevant", "step is relevant to the question")]
CHAIN_Q = [("sufficient", "the steps together are enough to answer"), ("causal", "at least one step is more than a 'where is' observation")]
MODEL_CHAIN_Q = CHAIN_Q + [("consistent", "the steps support the answer the model gave")]  # only on sheets of a model's own chains


def sample_chains(path: Path, n: int, seed: int) -> list[dict]:
    by_problem: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("gold") != "yes":
                continue
            pid = r["problem_id"].rsplit("_", 1)[0]
            by_problem.setdefault(pid, r)
    keys = sorted(by_problem)
    random.Random(seed).shuffle(keys)
    return [by_problem[k] for k in keys[:n]]


def sample_model_chains(paths: list[Path], n_each: int, seed: int) -> list[dict]:
    """n_each chains per harness output file (a model's own chains on validation items), one per question, drawn
    uniformly from the items whose chain cites at least one region; returned in the record shape `render` expects."""
    from causalsight.train.format import parse_chain

    out: list[dict] = []
    for path in paths:
        by_q: dict[str, list[dict]] = {}
        with Path(path).open() as f:
            for line in f:
                r = json.loads(line)
                ch = parse_chain(r.get("raw", "")).chain
                if ch is None or not ch.triplets:
                    continue
                r["_triplets"] = [{"question": t.question, "answer": t.answer, "evidence": {"t_start": t.evidence.t_start, "t_end": t.evidence.t_end, "box": list(t.evidence.box)}} for t in ch.triplets]
                by_q.setdefault(str(r.get("qid") or r["id"]), []).append(r)
        rng = random.Random(seed)
        qs = sorted(by_q)
        rng.shuffle(qs)
        tag = Path(path).name.replace("_plain.jsonl", "")
        for q in qs[:n_each]:
            r = rng.choice(sorted(by_q[q], key=lambda x: x["id"]))
            lines = r["prompt"].split("\n")
            out.append({"problem": lines[0] + "\n" + (lines[1] if len(lines) > 1 else ""), "problem_id": f"{tag}:{r['id']}", "question_type": r.get("question_type", ""),
                        "frames": "frames/" + str(r.get("frames_key") or r["id"].split("_")[0]), "chain": {"triplets": r["_triplets"]},
                        "label": f"gold {r.get('gold')}, model says {r.get('pred')}", "group": tag})
    return out


def frame_with_box(frames_root: Path, frames_key: str, frame: int, box, width: int = 320) -> str:
    img = Image.open(frames_root / frames_key / f"f{frame:02d}.jpg").convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    d.rectangle([box[0] * W, box[1] * H, box[2] * W, box[3] * H], outline=(255, 120, 0), width=max(2, W // 160))
    img = img.resize((width, round(H * width / W)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def load_prefill(path: Path) -> dict[str, str]:
    """name -> value from an earlier pass. Notes are plain text in a sheet's download and JSON strings in files
    written by older sheets or by hand; both are accepted."""
    out: dict[str, str] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            v = r["value"]
            if r["name"].endswith("_note") and v.startswith('"') and v.endswith('"'):
                try:
                    v = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    pass
            out[r["name"]] = str(v)
    return out


def radio(name: str, prefill: dict[str, str] | None = None) -> str:
    pre = (prefill or {}).get(name)
    return "".join(f'<label><input type="radio" name="{name}" value="{v}"{" checked" if v == pre else ""}> {v}</label> ' for v in ("yes", "no", "unsure"))


def render(chains: list[dict], frames_root: Path, prefill: dict[str, str] | None = None, title: str = "Pseudo-labelled NExT-QA chains: manual audit", csv_name: str = "pseudolabel_audit.csv",
           chain_q: list[tuple[str, str]] = CHAIN_Q, intro: str = "Boxes are the detector's; entities and observations are the teacher's.") -> str:
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Pseudo-label audit</title>",
        "<style>body{font:14px system-ui;margin:20px;max-width:1100px}.chain{border:1px solid #ccc;padding:10px;margin:14px 0}"
        ".step{display:flex;gap:14px;margin:8px 0;align-items:flex-start}.step img{width:320px;border:1px solid #999}"
        ".q{color:#444}label{margin-right:8px}textarea{width:100%}.opt{color:#060}"
        ".grp{background:#234;color:#fff;padding:8px 12px;margin:28px 0 6px;position:sticky;top:0}.tag{background:#234;color:#fff;padding:1px 6px;border-radius:3px}</style>",
        f"<h2>{html.escape(title)}</h2>",
        f"<p>{len(chains)} chains, one per question (seed 0). Judge every step, then click <b>Download CSV</b>. {html.escape(intro)}</p>",
        "<button onclick='dl()'>Download CSV</button>",
    ]
    if prefill:
        parts.append("<p><b>Pre-filled</b> from an earlier pass: every radio is already set; change the ones you disagree with, then download.</p>")
    groups = [r.get("group") for r in chains]
    for ci, r in enumerate(chains):
        ch = r["chain"]
        g = groups[ci]
        if g and (ci == 0 or groups[ci - 1] != g):
            last = max(i for i, x in enumerate(groups) if x == g)
            parts.append(f"<h3 class='grp'>Model {sorted(set(groups), key=groups.index).index(g) + 1} of {len(set(groups))}: {html.escape(g)} (chains {ci + 1}-{last + 1}). "
                         "Same questions as the other model, but this model's own steps, frames and boxes.</h3>")
        frames_key = r["frames"].rstrip("/").rsplit("/", 1)[-1]  # "/content/data/nextqa_pack/frames/<video>" or "frames/<video>"
        q = html.escape(r["problem"].split("\n")[0])
        opt = html.escape(r["problem"].split("\n")[1] if "\n" in r["problem"] else "")
        parts.append(f"<div class='chain' id='c{ci}'><b>#{ci + 1}</b> " + (f"<span class='tag'>{html.escape(g)}</span> " if g else "") + f"<span class='q'>{q}</span> <span class='opt'>{opt} ({html.escape(r.get('label', 'correct'))})</span> "
                     f"<small>{html.escape(r['problem_id'])}, {html.escape(r['question_type'])}</small>")
        for si, t in enumerate(ch["triplets"]):
            e = t["evidence"]
            fr = frame_of(int(e["t_start"]), int(e["t_end"]))  # chain span (0-127) -> nearest of the 16 sampled frames
            b64 = frame_with_box(frames_root, frames_key, fr, e["box"])
            parts.append(f"<div class='step'><img src='data:image/jpeg;base64,{b64}'><div>"
                         f"<div><b>step {si + 1}</b>, span {int(e['t_start'])}-{int(e['t_end'])} of 128, sampled frame {fr}</div><div>{html.escape(t['question'])}</div><div><i>{html.escape(t['answer'])}</i></div>")
            for key, label in STEP_Q:
                parts.append(f"<div>{label}: {radio(f'c{ci}_s{si}_{key}', prefill)}</div>")
            parts.append("</div></div>")
        for key, label in chain_q:
            parts.append(f"<div>{label}: {radio(f'c{ci}_{key}', prefill)}</div>")
        note = (prefill or {}).get(f'c{ci}_note', '')
        parts.append(f"<textarea name='c{ci}_note' rows='1' placeholder='notes'>{html.escape(note)}</textarea></div>")
    parts.append("""<script>
function dl(){const rows=[["name","value"]];document.querySelectorAll("input[type=radio]:checked").forEach(i=>rows.push([i.name,i.value]));
document.querySelectorAll("textarea").forEach(t=>{if(t.value)rows.push([t.name,t.value])});
const csv=rows.map(r=>r.map(v=>/[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v).join(",")).join("\\n");const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download="__CSV_NAME__";a.click();}
// autosave: every change is kept in this browser, so a reload or a closed tab does not lose progress
const KEY="audit:__CSV_NAME__";
function save(){try{const o={};document.querySelectorAll("input[type=radio]:checked").forEach(i=>o[i.name]=i.value);document.querySelectorAll("textarea").forEach(t=>o[t.name]=t.value);localStorage.setItem(KEY,JSON.stringify(o));}catch(e){}}
try{const o=JSON.parse(localStorage.getItem(KEY)||"null");if(o){for(const[k,v]of Object.entries(o)){if(k.endsWith("_note")){const t=document.querySelector('textarea[name="'+k+'"]');if(t)t.value=v;}else{const i=document.querySelector('input[name="'+k+'"][value="'+v+'"]');if(i)i.checked=true;}}}}catch(e){}
document.addEventListener("change",save);document.addEventListener("input",save);
</script>""")
    return "\n".join(parts).replace("__CSV_NAME__", csv_name)


def _rates(values: list[str]) -> dict:
    c = Counter(values)
    n = sum(c.values())
    return {"n": n, "yes": round(c["yes"] / n, 3), "no": round(c["no"] / n, 3), "unsure": round(c["unsure"] / n, 3)}


def _summary(vals: dict[str, str]) -> dict:
    out: dict[str, dict] = {}
    for key, _ in STEP_Q + MODEL_CHAIN_Q:
        got = [v for k, v in vals.items() if k.endswith("_" + key)]
        if got:  # "consistent" is only asked on model sheets
            out[key] = _rates(got)
    seen = [k for k, v in vals.items() if k.endswith("_visible") and v == "yes"]
    if seen:
        out["onbox_given_visible"] = _rates([vals[k[: -len("visible")] + "onbox"] for k in seen if k[: -len("visible")] + "onbox" in vals])
    return out


def summarize(csv_path: Path, n_each: int | None = None, names: list[str] | None = None) -> dict:
    """Rates per judgement. With n_each, also one block per run of n_each consecutive chains (a `build-model` sheet
    lists n_each chains per output file, in the order the files were given)."""
    vals: dict[str, str] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if not row["name"].endswith("_note"):
                vals[row["name"]] = row["value"]
    out = _summary(vals)
    if n_each:
        groups: dict[int, dict[str, str]] = {}
        for k, v in vals.items():
            groups.setdefault(int(k[1:].split("_", 1)[0]) // n_each, {})[k] = v
        out = {"all": out} | {(names[g] if names and g < len(names) else f"group{g}"): _summary(groups[g]) for g in sorted(groups)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("sft", type=Path, help="pseudo-labelled SFT jsonl (real_sft.jsonl)")
    b.add_argument("--frames", type=Path, required=True, help="pack root containing frames/<video>/fNN.jpg")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--n", type=int, default=100)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--prefill", type=Path, default=None, help="CSV from an earlier pass (name,value): pre-check the radios so a verifier only flips disagreements")
    bm = sub.add_parser("build-model", help="audit sheet for a model's own chains (harness per-item outputs)")
    bm.add_argument("outputs", nargs="+", type=Path, help="<tag>_plain.jsonl files")
    bm.add_argument("--frames", type=Path, required=True)
    bm.add_argument("--out", type=Path, required=True)
    bm.add_argument("--n-each", type=int, default=50)
    bm.add_argument("--seed", type=int, default=0)
    bm.add_argument("--prefill", type=Path, default=None)
    s = sub.add_parser("summarize")
    s.add_argument("csv", type=Path)
    s.add_argument("--n-each", type=int, default=None, help="model sheets: also report each block of this many chains (one block per output file)")
    s.add_argument("--names", nargs="*", default=None, help="names for the blocks, in order")
    a = ap.parse_args()
    if a.cmd == "build":
        chains = sample_chains(a.sft, a.n, a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        prefill = load_prefill(a.prefill) if a.prefill else None
        a.out.write_text(render(chains, a.frames / "frames", prefill))
        print(f"{len(chains)} chains, {sum(len(c['chain']['triplets']) for c in chains)} steps -> {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
    elif a.cmd == "build-model":
        chains = sample_model_chains(a.outputs, a.n_each, a.seed)
        prefill = load_prefill(a.prefill) if a.prefill else None
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(render(chains, a.frames / "frames", prefill, title="Model citations on NExT-QA validation items: manual audit", csv_name=a.out.stem + ".csv", chain_q=MODEL_CHAIN_Q,
                                intro=f"Chains, entities and boxes are the models' own ({a.n_each} chains per model, in the order " + ", ".join(p.name.replace("_plain.jsonl", "") for p in a.outputs) + ")."))
        print(f"{len(chains)} chains, {sum(len(c['chain']['triplets']) for c in chains)} steps -> {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
    elif a.cmd == "summarize":
        print(json.dumps(summarize(a.csv, a.n_each, a.names), indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
