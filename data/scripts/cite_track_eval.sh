#!/usr/bin/env bash
# Entity-track test on real video: mask every detection of the entities a model's own chains name, on every sampled
# frame, plus the cited boxes (mask_cite_track) vs same-size boxes elsewhere on the same frames (mask_cite_track_random).
#   PACK=/content/data/nextqa_pack RESULTS_DIR=<drive>/results BATCH=8 bash data/scripts/cite_track_eval.sh <model> <tag> [item_ids_json]
# Needs in RESULTS_DIR: <tag>_nq_plain.jsonl (the citing run) and nextqa_entity_tracks.json (from cs-entity-tracks, or
# set TRACKS=...). item_ids_json (optional) restricts the run to a list of item ids, e.g. the items correct under the
# plain input (data/subsets/<tag>_nq_plain_correct_ids.json): only those enter the flip rates, so the gap is unchanged
# and the run is ~15% shorter. Re-running skips conditions whose summary exists.
set -euo pipefail
MODEL="$1"; TAG="$2"; IDS="${3:-}"
INSTR="${INSTR:-chain}"; MAXTOK="${MAXTOK:-640}"; MC="${MC:-option}"; BATCH="${BATCH:-8}"
PACK="${PACK:?set PACK}"; OUT="${RESULTS_DIR:-results/nextqa}"; TRACKS="${TRACKS:-$OUT/nextqa_entity_tracks.json}"
CHAINS="$OUT/${TAG}_nq_plain.jsonl"
[ -f "$CHAINS" ] || { echo "missing $CHAINS"; exit 1; }
[ -f "$TRACKS" ] || { echo "missing $TRACKS (run cs-entity-tracks first)"; exit 1; }
COMMON=(--bench nextqa --split val --root "$PACK" --frames-dir "$PACK" --ids-file "$PACK/nextqa_val_ids.json"
        --model "$MODEL" --instr "$INSTR" --max-new-tokens "$MAXTOK" --mc "$MC" --batch-size "$BATCH" --chains "$CHAINS" --tracks "$TRACKS")
[ -n "$IDS" ] && COMMON+=(--item-ids "$IDS")
[ -f "$OUT/${TAG}_nq_mask_cite_track.summary.json" ] || cs-eval "${COMMON[@]}" --mask cite_track --out "$OUT/${TAG}_nq_mask_cite_track.jsonl"
[ -f "$OUT/${TAG}_nq_mask_cite_track_random.summary.json" ] || cs-eval "${COMMON[@]}" --mask cite_track_random --out "$OUT/${TAG}_nq_mask_cite_track_random.jsonl"
cs-faithfulness "$OUT" --tag "${TAG}_nq"
