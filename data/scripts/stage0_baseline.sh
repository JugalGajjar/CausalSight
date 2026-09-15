#!/usr/bin/env bash
# Stage 0 zero-shot baseline on a stratified CLEVRER validation subset, four conditions on the
# SAME items: plain, blind (black frames), evidence-masked, random-masked.
# Usage: bash data/scripts/stage0_baseline.sh [model] [per_type] [tag] [instr] [max_new_tokens]
#   model may be an adapter dir (runs/stage0/adapter); use instr=plain and max_new_tokens=384 for GRPO checkpoints,
#   instr=chain and max_new_tokens=512 for SFT / CSR checkpoints (adds chain-level metrics to the summaries)
#   per_type=250 -> 1000 items per condition; ~45 min per condition on an M4 Pro, minutes on a GPU.
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-VL-3B-Instruct}"
PER_TYPE="${2:-250}"
TAG="${3:-qwen3b}"
INSTR="${4:-none}"
MAXTOK="${5:-64}"
CHAINS=data/triplets/clevrer_validation.jsonl
# Optional env: MC=option (per-option yes/no items, chain models), EVALPACK=<dir> (frames-dir + root + chains from an eval pack)
MC="${MC:-multi}"
BATCH="${BATCH:-1}"
if [ -n "${EVALPACK:-}" ]; then
  CHAINS="$EVALPACK/chains.jsonl"
  EXTRA=(--root "$EVALPACK" --frames-dir "$EVALPACK" --proposals-root "$EVALPACK/derender_proposals")
else
  EXTRA=()
fi
COMMON=(--bench clevrer --split validation --chains "$CHAINS" --per-type "$PER_TYPE" --seed 0 --model "$MODEL" --instr "$INSTR" --max-new-tokens "$MAXTOK" --mc "$MC" --batch-size "$BATCH" "${EXTRA[@]}")
OUT="${RESULTS_DIR:-results/stage0}"   # on Colab set RESULTS_DIR to a Drive path so results survive disconnects
mkdir -p "$OUT"
run() {  # run <condition-name> [extra cs-eval args]: skip conditions that already have a finished summary
  local name="$1"; shift
  if [ -f "$OUT/${TAG}_${name}.summary.json" ]; then echo "have ${TAG}_${name}, skipping"; return; fi
  cs-eval "${COMMON[@]}" "$@" --out "$OUT/${TAG}_${name}.jsonl"
}
run plain
run blind --blind
run mask_evidence --mask evidence
run mask_random --mask random
cs-faithfulness "$OUT" --tag "$TAG"

# Object-track conditions (stronger intervention; see results/stage0/README.md)
run mask_track --mask track
run mask_track_random --mask track_random
cs-faithfulness "$OUT" --tag "$TAG"
