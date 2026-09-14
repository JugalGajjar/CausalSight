# Running on Colab (single GPU)

Budget plan (A100 80GB, 44 h): Stage 0 GRPO 1,500 steps ~7 h, Stage 1 SFT ~4 h, Stage 2 CSR GRPO 1,500 steps ~9 h, one ablation (Stage 2 without R_nec) ~9 h, reserve ~15 h. Re-size step counts from the pilot's seconds/step.
Evaluation is never run on Colab; checkpoints are downloaded and scored on the Mac.

## Once, on the Mac
```bash
bash data/scripts/download_clevrer.sh --videos train        # ~12 GB, needed for training frames
cs-frames --rl data/rl/clevrer_train_rl.jsonl --per-type 375 --sft data/sft/clevrer_train_sft.jsonl --sft-per-type 800 --out data/frames/train
cd data/frames && zip -qr train.zip train                    # ~600 MB
```
Upload `train.zip` to Drive under `MyDrive/causalsight/`.

## Each Colab session
```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/JugalGajjar/CausalSight.git /content/CausalSight
!bash /content/CausalSight/colab/setup.sh /content/drive/MyDrive/causalsight/train.zip
```
Pilot first (20 steps, prints seconds per step; multiply by 1500 to size the run):
```python
!cd /content/CausalSight && cs-grpo --config configs/stage0_grpo.yaml --data /content/data/train --out /content/drive/MyDrive/causalsight/runs/stage0 --steps 20
```
Full run, resumable after any disconnect (checkpoints every 50 steps on Drive):
```python
!cd /content/CausalSight && cs-grpo --config configs/stage0_grpo.yaml --data /content/data/train --out /content/drive/MyDrive/causalsight/runs/stage0 --resume
```
Watch `runs/stage0/log.jsonl` on Drive: `r_format` should reach ~1.0 within the first hundred steps and
`r_outcome_lenient` should rise; `kl` should stay small (< 0.1 with beta 0.04); `grad_norm` should be
nonzero on non-skipped steps (`loss` itself is ~0 by construction). `skipped` steps are groups with no
reward variance; expect many early on for easy descriptive questions. If seconds/step exceeds ~16 on the
A100, lower `micro_batch`, `max_new_tokens`, or `n_frames` in the config before launching the full run.

## Stage 1: triplet-chain SFT with the bridge loss (~4 h)
```python
!mkdir -p /content/drive/MyDrive/causalsight/runs/stage1 && cd /content/CausalSight && git pull && nohup cs-sft --config configs/stage1_sft.yaml --data /content/data/train --out /content/drive/MyDrive/causalsight/runs/stage1 > /content/drive/MyDrive/causalsight/runs/stage1/console.txt 2>&1 &
```
Watch `nll` fall (~1.4 -> well under 0.5) and `p_ans_corrupted` fall from ~0.9: the answer should stop being
recoverable from a broken chain. `--resume` continues after a disconnect. Evaluate with `--instr chain --max-new-tokens 512`.

## Stage 2: CSR policy optimization from the SFT checkpoint (pilot, then ~1,000 steps)
The config points at the Stage 1 adapter on Drive and the chain-annotated subset in the repo.
```python
!mkdir -p /content/drive/MyDrive/causalsight/runs/stage2 && cd /content/CausalSight && git pull && cs-grpo --config configs/stage2_csr.yaml --data /content/data/train --out /content/drive/MyDrive/causalsight/runs/stage2 --steps 20
```
Pilot: check seconds/step and GPU memory (`micro_batch` 4 with 640-token chains; raise to 8 if memory allows).
Then the full run with `--resume` and nohup as for Stage 0. Ablation without the necessity reward:
`configs/stage2_no_nec.yaml` into `runs/stage2_no_nec`. Evaluate with `--instr chain --max-new-tokens 640`.

## Evaluating on the GPU (any checkpoint, ~1 h for six conditions)
Upload `data/frames/evalpack.zip` (built once with `cs-frames --eval-pack ...`, ~220 MB) to Drive, then:
```python
!cd /content/data && unzip -q /content/drive/MyDrive/causalsight/evalpack.zip
!cd /content/CausalSight && EVALPACK=/content/data/evalpack MC=option bash data/scripts/stage0_baseline.sh /content/drive/MyDrive/causalsight/runs/stage1/adapter 250 sft3b chain 640
```
Results land in `/content/CausalSight/results/stage0/`; copy the `sft3b_*` files to Drive or download them.
Chain models (SFT, CSR) use `MC=option chain 640`; bare-answer models (base, Stage 0 GRPO) use `MC=multi plain 384`.

## After training, on the Mac
Download `runs/stage0/adapter/` from Drive, then evaluate the merged model with the six Stage 0 conditions
(`cs-eval --model <adapter dir>` support is in the backend: pass the adapter directory and the base model
is loaded and merged automatically).
