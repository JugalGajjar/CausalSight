#!/usr/bin/env bash
# NExT-QA validation subset (999 questions, 5 options): plain and blind.
#   PACK=/content/data/nextqa_pack RESULTS_DIR=<drive>/results_v2 BATCH=8 MC=multi bash data/scripts/nextqa_eval.sh <model> <tag> <instr> <max_new_tokens>
set -euo pipefail
MODEL="$1"; TAG="$2"; INSTR="${3:-none}"; MAXTOK="${4:-64}"
MC="${MC:-multi}"; BATCH="${BATCH:-8}"; PACK="${PACK:?set PACK}"; OUT="${RESULTS_DIR:-results/nextqa}"
mkdir -p "$OUT"
COMMON=(--bench nextqa --split val --root "$PACK" --frames-dir "$PACK" --ids-file "$PACK/nextqa_val_ids.json" --model "$MODEL" --instr "$INSTR" --max-new-tokens "$MAXTOK" --mc "$MC" --batch-size "$BATCH")
[ -f "$OUT/${TAG}_nq_plain.summary.json" ] || cs-eval "${COMMON[@]}" --out "$OUT/${TAG}_nq_plain.jsonl"
[ -f "$OUT/${TAG}_nq_blind.summary.json" ] || cs-eval "${COMMON[@]}" --blind --out "$OUT/${TAG}_nq_blind.jsonl"
python - "$OUT" "$TAG" <<'PY'
import json, sys
d, tag = sys.argv[1], sys.argv[2]
p = json.load(open(f"{d}/{tag}_nq_plain.summary.json"))["by_type"]; b = json.load(open(f"{d}/{tag}_nq_blind.summary.json"))["by_type"]
print(f"{'type':14s}{'acc':>8s}{'blind':>8s}{'gap':>8s}")
for t in p: print(f"{t:14s}{p[t]['acc']:8.3f}{b[t]['acc']:8.3f}{p[t]['acc']-b[t]['acc']:8.3f}")
PY
