#!/usr/bin/env bash
# Stage 0 zero-shot baseline on a stratified CLEVRER validation subset, four conditions on the
# SAME items: plain, blind (black frames), evidence-masked, random-masked.
# Usage: bash data/scripts/stage0_baseline.sh [model] [per_type] [tag]
#   per_type=250 -> 1000 items per condition; ~45 min per condition on an M4 Pro, minutes on a GPU.
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-VL-3B-Instruct}"
PER_TYPE="${2:-250}"
TAG="${3:-qwen3b}"
CHAINS=data/triplets/clevrer_validation.jsonl
COMMON=(--bench clevrer --split validation --chains "$CHAINS" --per-type "$PER_TYPE" --seed 0 --model "$MODEL")
mkdir -p results/stage0
cs-eval "${COMMON[@]}"                 --out "results/stage0/${TAG}_plain.jsonl"
cs-eval "${COMMON[@]}" --blind         --out "results/stage0/${TAG}_blind.jsonl"
cs-eval "${COMMON[@]}" --mask evidence --out "results/stage0/${TAG}_mask_evidence.jsonl"
cs-eval "${COMMON[@]}" --mask random   --out "results/stage0/${TAG}_mask_random.jsonl"
cs-faithfulness results/stage0 --tag "$TAG"
