# CausalSight

Step-level counterfactual rewards for faithful video reasoning in small vision-language models.

* `notes/` (untracked) — proposal, execution plan, and dated decision files.
* `src/causalsight/` — Python package: data construction, evaluation harness, faithfulness metrics, rewards, training.
* `configs/` — one YAML per run; commit the config before launching a run.
* `results/` — one JSONL per evaluation run plus a markdown table per milestone.

## Setup

```bash
conda env create -f env/environment.yml
conda activate causalsight
pip install -e ".[dev]"
pytest
```

Training requires CUDA GPUs; see `notes/PLAN.md` for the compute assumptions. The local Mac is used for data construction, harness development, and tests.

## Data and evaluation commands

```bash
bash data/scripts/download_clevrer.sh --no-videos          # annotations, questions, proposals
bash data/scripts/download_clevrer.sh --videos validation  # validation videos (needed for evaluation)
cs-clevrer-check                                           # verify on-disk format
cs-clevrer-validate descriptive --n-videos 1000            # executor vs ground truth
cs-triplets --split train --out data/triplets/clevrer_train.jsonl
cs-eval --model Qwen/Qwen2.5-VL-3B-Instruct --bench clevrer --limit 200 --out results/zeroshot.jsonl
cs-eval --model dummy --bench clevrer --limit 20           # pipeline check without weights
```

## Layout

```
src/causalsight/
  data/            CLEVRER programmatic triplet generator, pseudo-label pipeline
  eval/            benchmark harness (CLEVRER by type, CausalVQA, MMVU, Video-MME, TempCompass, Video-Holmes)
  eval/faithfulness/  blind gap, evidence masking, chain-answer consistency, necessity rate
  train/rewards/   R_out, R_fmt, R_ground, R_nec, R_proc as separate modules
  train/sft/       triplet SFT with bridge loss
  train/rl/        GRPO fork with the CSR reward
```
