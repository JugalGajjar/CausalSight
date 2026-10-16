#!/usr/bin/env bash
# Real-video evaluation on the CausalVQA debug split: plain and blind conditions only (no evidence boxes).
#   PACK=/content/data/causalvqa_pack RESULTS_DIR=<drive>/results_v2 BATCH=8 MC=multi \
#     bash data/scripts/causalvqa_eval.sh <model> <tag> <instr> <max_new_tokens>
set -euo pipefail
MODEL="$1"; TAG="$2"; INSTR="${3:-none}"; MAXTOK="${4:-64}"
MC="${MC:-multi}"; BATCH="${BATCH:-8}"; PACK="${PACK:?set PACK}"; OUT="${RESULTS_DIR:-results/causalvqa}"
mkdir -p "$OUT"
COMMON=(--bench causalvqa --split debug --root "$PACK" --frames-dir "$PACK" --model "$MODEL" --instr "$INSTR" --max-new-tokens "$MAXTOK" --mc "$MC" --batch-size "$BATCH")
[ -f "$OUT/${TAG}_cvqa_plain.summary.json" ] || cs-eval "${COMMON[@]}" --out "$OUT/${TAG}_cvqa_plain.jsonl"
[ -f "$OUT/${TAG}_cvqa_blind.summary.json" ] || cs-eval "${COMMON[@]}" --blind --out "$OUT/${TAG}_cvqa_blind.jsonl"
python - "$OUT" "$TAG" <<'PY'
import json, sys
d, tag = sys.argv[1], sys.argv[2]
p = json.load(open(f"{d}/{tag}_cvqa_plain.summary.json"))["by_type"]; b = json.load(open(f"{d}/{tag}_cvqa_blind.summary.json"))["by_type"]
print(f"{'type':16s}{'set_acc':>9s}{'paired':>9s}{'blind_set':>11s}{'blind_gap':>11s}")
for t in p:
    print(f"{t:16s}{p[t]['set_acc']:9.3f}{p[t]['paired_acc']:9.3f}{b[t]['set_acc']:11.3f}{p[t]['set_acc']-b[t]['set_acc']:11.3f}")
PY
