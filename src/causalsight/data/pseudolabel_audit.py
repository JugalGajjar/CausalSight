"""Manual audit sheet for pseudo-labelled NExT-QA chains.

  cs-pseudolabel-audit <real_sft.jsonl> --frames data/frames/nextqa_pack --out results/mixed/pseudolabel_audit.html [--n 100]

Samples n chains (one per question, the "yes" item, seed 0) and writes one self-contained HTML page: for every
step the cited frame (the sampled frame nearest the span's centre, as in the grounding check) with the detector box drawn, the entity the teacher named and its observation, and three
yes/no judgements per step (entity visible on this frame; box on the entity; step relevant to the question),
plus one per chain (the steps together are enough to answer). A button at the top downloads the judgements as CSV
(`cs-pseudolabel-audit summarize <csv>` turns it into the numbers for the paper). Images are embedded at 320 px
wide, so the page is ~5 MB and needs nothing but a browser.
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


def frame_with_box(frames_root: Path, frames_key: str, frame: int, box, width: int = 320) -> str:
    img = Image.open(frames_root / frames_key / f"f{frame:02d}.jpg").convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    d.rectangle([box[0] * W, box[1] * H, box[2] * W, box[3] * H], outline=(255, 120, 0), width=max(2, W // 160))
    img = img.resize((width, round(H * width / W)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def radio(name: str, prefill: dict[str, str] | None = None) -> str:
    pre = (prefill or {}).get(name)
    return "".join(f'<label><input type="radio" name="{name}" value="{v}"{" checked" if v == pre else ""}> {v}</label> ' for v in ("yes", "no", "unsure"))


def render(chains: list[dict], frames_root: Path, prefill: dict[str, str] | None = None) -> str:
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Pseudo-label audit</title>",
        "<style>body{font:14px system-ui;margin:20px;max-width:1100px}.chain{border:1px solid #ccc;padding:10px;margin:14px 0}"
        ".step{display:flex;gap:14px;margin:8px 0;align-items:flex-start}.step img{width:320px;border:1px solid #999}"
        ".q{color:#444}label{margin-right:8px}textarea{width:100%}.opt{color:#060}</style>",
        "<h2>Pseudo-labelled NExT-QA chains: manual audit</h2>",
        f"<p>{len(chains)} chains, one per question (seed 0). Judge every step, then click <b>Download CSV</b>. "
        "Boxes are the detector's; entities and observations are the teacher's.</p>",
        "<button onclick='dl()'>Download CSV</button>",
    ]
    if prefill:
        parts.append("<p><b>Pre-filled</b> from an earlier pass: every radio is already set; change the ones you disagree with, then download.</p>")
    for ci, r in enumerate(chains):
        ch = r["chain"]
        frames_key = r["frames"].rstrip("/").rsplit("/", 1)[-1]  # "/content/data/nextqa_pack/frames/<video>" or "frames/<video>"
        q = html.escape(r["problem"].split("\n")[0])
        opt = html.escape(r["problem"].split("\n")[1] if "\n" in r["problem"] else "")
        parts.append(f"<div class='chain' id='c{ci}'><b>#{ci + 1}</b> <span class='q'>{q}</span> <span class='opt'>{opt} (correct)</span> "
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
        for key, label in CHAIN_Q:
            parts.append(f"<div>{label}: {radio(f'c{ci}_{key}', prefill)}</div>")
        note = (prefill or {}).get(f'c{ci}_note', '')
        parts.append(f"<textarea name='c{ci}_note' rows='1' placeholder='notes'>{html.escape(note)}</textarea></div>")
    parts.append("""<script>
function dl(){const rows=[["name","value"]];document.querySelectorAll("input[type=radio]:checked").forEach(i=>rows.push([i.name,i.value]));
document.querySelectorAll("textarea").forEach(t=>{if(t.value)rows.push([t.name,JSON.stringify(t.value)])});
const csv=rows.map(r=>r.map(v=>/[",\\n]/.test(v)&&!v.startsWith('"')?JSON.stringify(v):v).join(",")).join("\\n");const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download="pseudolabel_audit.csv";a.click();}
</script>""")
    return "\n".join(parts)


def summarize(csv_path: Path) -> dict:
    vals: dict[str, str] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            vals[row["name"]] = row["value"]
    out: dict[str, dict] = {}
    for key, _ in STEP_Q + CHAIN_Q:
        c = Counter(v for k, v in vals.items() if k.endswith("_" + key) and not k.endswith("_note"))
        n = sum(c.values())
        out[key] = {"n": n, "yes": round(c["yes"] / n, 3) if n else None, "no": round(c["no"] / n, 3) if n else None, "unsure": round(c["unsure"] / n, 3) if n else None}
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
    s = sub.add_parser("summarize")
    s.add_argument("csv", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        chains = sample_chains(a.sft, a.n, a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        prefill = None
        if a.prefill:
            with a.prefill.open() as f:
                prefill = {r["name"]: (json.loads(r["value"]) if r["name"].endswith("_note") else r["value"]) for r in csv.DictReader(f)}
        a.out.write_text(render(chains, a.frames / "frames", prefill))
        print(f"{len(chains)} chains, {sum(len(c['chain']['triplets']) for c in chains)} steps -> {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
    elif a.cmd == "summarize":
        print(json.dumps(summarize(a.csv), indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
