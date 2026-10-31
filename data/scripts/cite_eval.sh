#!/usr/bin/env bash
# Self-cited evidence test: mask the regions the model cited in its own plain-run chains (mask_cite) vs the same
# boxes at random positions (mask_cite_random); report the cite gap next to the blind gap.
#   PACK=/content/data/nextqa_pack RESULTS_DIR=<drive>/results_v2 BATCH=8 bash data/scripts/cite_eval.sh <model> <tag> [cite_tag]
#   BENCH=clevrer PACK=/content/data/evalpack RESULTS_DIR=... BATCH=8 bash data/scripts/cite_eval.sh <model> <tag> [cite_tag]
# The plain run must already exist in RESULTS_DIR: <tag>_nq_plain.jsonl (nextqa, from nextqa_eval.sh) or
# <tag>_plain.jsonl (clevrer, from stage0_baseline.sh). Its items define the evaluated set, so no --per-type here.
# cite_tag (default: tag) names the model whose citations are masked; set it to a chain model's tag to mask a plain
# model with foreign citations.
set -euo pipefail
MODEL="$1"; TAG="$2"; CITE="${3:-$2}"
BENCH="${BENCH:-nextqa}"; INSTR="${INSTR:-chain}"; MAXTOK="${MAXTOK:-640}"; MC="${MC:-option}"; BATCH="${BATCH:-8}"
PACK="${PACK:?set PACK}"; OUT="${RESULTS_DIR:-results/$BENCH}"
case "$BENCH" in
  nextqa)  SUF="_nq"; BARGS=(--bench nextqa --split val --root "$PACK" --frames-dir "$PACK" --ids-file "$PACK/nextqa_val_ids.json");;
  clevrer) SUF="";    BARGS=(--bench clevrer --split validation --root "$PACK" --frames-dir "$PACK");;
  *) echo "BENCH must be nextqa or clevrer"; exit 1;;
esac
CHAINS="$OUT/${CITE}${SUF}_plain.jsonl"
[ -f "$CHAINS" ] || { echo "missing $CHAINS"; exit 1; }
COMMON=("${BARGS[@]}" --model "$MODEL" --instr "$INSTR" --max-new-tokens "$MAXTOK" --mc "$MC" --batch-size "$BATCH" --chains "$CHAINS")
[ -f "$OUT/${TAG}${SUF}_mask_cite.summary.json" ] || cs-eval "${COMMON[@]}" --mask cite --out "$OUT/${TAG}${SUF}_mask_cite.jsonl"
[ -f "$OUT/${TAG}${SUF}_mask_cite_random.summary.json" ] || cs-eval "${COMMON[@]}" --mask cite_random --out "$OUT/${TAG}${SUF}_mask_cite_random.jsonl"
cs-faithfulness "$OUT" --tag "${TAG}${SUF}"
