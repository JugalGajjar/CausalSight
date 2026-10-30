#!/usr/bin/env bash
# Self-cited evidence test on NExT-QA: mask the regions the model cited in its own plain-run chains
# (mask_cite) vs the same boxes at random positions (mask_cite_random); report the cite gap next to the blind gap.
#   PACK=/content/data/nextqa_pack RESULTS_DIR=<drive>/results_v2 BATCH=8 bash data/scripts/cite_eval.sh <model> <tag> [cite_tag]
# <tag>_nq_plain.jsonl must already exist in RESULTS_DIR (from nextqa_eval.sh). cite_tag (default: tag) names the
# model whose citations are masked; set it to a chain model's tag to mask a plain model with foreign citations.
set -euo pipefail
MODEL="$1"; TAG="$2"; CITE="${3:-$2}"
INSTR="${INSTR:-chain}"; MAXTOK="${MAXTOK:-640}"; MC="${MC:-option}"; BATCH="${BATCH:-8}"; PACK="${PACK:?set PACK}"; OUT="${RESULTS_DIR:-results/nextqa}"
CHAINS="$OUT/${CITE}_nq_plain.jsonl"
[ -f "$CHAINS" ] || { echo "missing $CHAINS"; exit 1; }
COMMON=(--bench nextqa --split val --root "$PACK" --frames-dir "$PACK" --ids-file "$PACK/nextqa_val_ids.json" --model "$MODEL" --instr "$INSTR" --max-new-tokens "$MAXTOK" --mc "$MC" --batch-size "$BATCH" --chains "$CHAINS")
[ -f "$OUT/${TAG}_nq_mask_cite.summary.json" ] || cs-eval "${COMMON[@]}" --mask cite --out "$OUT/${TAG}_nq_mask_cite.jsonl"
[ -f "$OUT/${TAG}_nq_mask_cite_random.summary.json" ] || cs-eval "${COMMON[@]}" --mask cite_random --out "$OUT/${TAG}_nq_mask_cite_random.jsonl"
cs-faithfulness "$OUT" --tag "${TAG}_nq"
