#!/usr/bin/env bash
# Published small-model video-RL checkpoints through our six-condition suite on the same 1,000 questions.
# All are Qwen2.5-VL checkpoints that answer with <think>/<answer>; evaluated like our GRPO rows
# (multi-option letters prompt, instr=plain, 384 tokens). Needs the eval pack; results go to RESULTS_DIR.
#   EVALPACK=/content/data/evalpack RESULTS_DIR=<drive>/results_v2 BATCH=24 bash data/scripts/baselines_published.sh
set -euo pipefail
export EVALPACK RESULTS_DIR BATCH MC=multi
run() { bash data/scripts/stage0_baseline.sh "$1" 250 "$2" plain 384; }
run QiWang98/VideoRFT-3B           videorft3b
run Falconss1/VideoThinker-R1-3B   videothinker3b
run Video-R1/Video-R1-7B           videor1_7b
