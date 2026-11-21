# CausalSight

Code and results for **"Does the Model Use What It Cites? Counterfactual Tests of Grounded Reasoning Chains in
Small Video-Language Models"** by Jugal Gajjar and Kamalasankari Subramaniakuppusamy (equal contribution).

A grounded reasoning chain cites a frame span and a box for every step, so a reader can check where the model
looked. Whether the answer *depends* on the cited region is a different question. This repository tests it by
intervention: the regions a model cited in its own chain are blacked out, and the share of correct answers that
flip is compared with blacking out boxes of the same size elsewhere on the same frames. The paired difference is
the **cite gap**. It needs no annotation, so it applies to any video.

## Results in brief

2-3B video-language models (Qwen2.5-VL-3B, InternVL3-2B) trained to answer through grounded chains, on exact
CLEVRER chains and on pseudo-labelled NExT-QA chains. Flip rates are over the answers that are correct on the
unmodified input; intervals are 95% question-clustered bootstrap.

| Test | Benchmark | Flip, intervention | Flip, control | Gap | Ratio |
|---|---|---|---|---|---|
| Self-citation (cited boxes) | CLEVRER, five chain models | 9.7% | 4.5% | 0.053 [0.045, 0.060] | 2.2 |
| Self-citation (cited boxes) | NExT-QA, five chain models | 2.3% | 2.1% | 0.003 [0.001, 0.005] | 1.1 |
| Entity track (named entity, all frames) | NExT-QA, two chain models | 10.2% | 5.5% | 0.047 [0.039, 0.056] | 1.9 |

With the same weights, the cite gap is 0.089 on CLEVRER and 0.003 on NExT-QA, while the cited boxes pass a
grounding check on both. The NExT-QA chains were supervised to cite two frames of entities that stay visible on
most frames, so little depends on any one cited box and more depends on the cited entity. A citation can be
verifiable and relevant without the answer depending on it, and the usual grounding check does not show the
difference.

## What is in this repository

* The test suite: masking interventions, size-matched controls, paired gaps with intervals, and the diagnostics
  needed to read a gap.
* The pipeline behind the models: the CLEVRER chain generator, the NExT-QA pseudo-labelling, supervised
  fine-tuning with a bridge loss, and GRPO on chains.
* Result summaries for every run reported in the paper (`results/`), with the analysis notes written as the
  experiments ran.
* The manual audit sheets' judgements for the pseudo-labels and for the models' own citations (`results/audits/`).

Not here yet: model weights, per-item outputs and the frame packs. Benchmark videos are not redistributed; the
scripts below download or rebuild what they need.

## Install

```bash
conda env create -f env/environment.yml
conda activate causalsight
pip install -e ".[dev]"
pytest
```

The tests and all statistics run on CPU. Inference and training need a CUDA GPU; every run in the paper used a
single A100 80GB. `pip install -e ".[ml,dev]"` adds the model stack.

## The tests

Every test has the same form: an intervention on the input frames, a size-matched control, and the paired
difference in flip rates over the items that are correct on the unmodified input.

| Gap | Intervention (`cs-eval --mask`) | Control | Needs |
|---|---|---|---|
| Blind gap | all frames black (`--blind`) | none | nothing |
| Gold-evidence gap | `evidence`: annotated evidence boxes | `random`: same boxes, random positions | CLEVRER annotations |
| Object-track gap | `track`: decisive objects on every frame | `track_random`: same-size copies elsewhere | CLEVRER annotations |
| **Cite gap** | `cite`: the boxes the model cited in its own chain | `cite_random`: same boxes, random positions | the model's own output |
| **Entity-track gap** | `cite_track`: every detection of the entities the chain names, on all frames, plus the cited boxes | `cite_track_random`: same-size boxes that avoid the entity | the model's own output and a detector |

`cs-faithfulness` turns the saved runs of one model into the gaps with intervals. `cs-faithfulness-splits` adds
the diagnostics: gaps by gold answer, by whether the cited box hits the named entity, on the items whose control
leaves the cited area untouched, on the video-dependent answers, reverse flips and net accuracy, coverage of the
entity's frames, and what the chain cites under the mask.

The harness reads citations in the chain format below. A model with another citation format needs a parser
that returns the same triplets (`src/causalsight/train/format.py`).

```
<think>
[1] Q: Which object is the gray sphere? | A: the gray rubber sphere | E: t=0-0 box=(0.360,0.620,0.470,0.790) | deps=
[2] Q: Which collision involving the gray sphere happens in the video? | A: ... | E: t=17-21 box=(...) | deps=1
</think>
<answer>blue</answer>
```

`t` indexes a 128-position time grid over the clip and the box is in normalised image coordinates.

## Running the suite

A pipeline check without weights or videos:

```bash
cs-eval --model dummy --bench clevrer --limit 20
```

### CLEVRER

```bash
bash data/scripts/download_clevrer.sh --videos validation
cs-triplets --split validation --out data/triplets/clevrer_validation.jsonl
cs-frames --eval-pack --chains data/triplets/clevrer_validation.jsonl --per-type 250 --out data/frames/evalpack

# plain, blind, gold-evidence and object-track conditions on the same 1,000 stratified questions
EVALPACK=data/frames/evalpack MC=option BATCH=8 bash data/scripts/stage0_baseline.sh <model or adapter dir> 250 <tag> chain 640

# self-citation test on the items of the plain run
BENCH=clevrer PACK=data/frames/evalpack bash data/scripts/cite_eval.sh <model> <tag>
```

Chain models answer one yes/no item per option (`MC=option`, instruction `chain`, 640 new tokens). Models that
answer with a bare letter use `MC=multi` with instruction `plain` or `none`.

### NExT-QA

```bash
# plain and blind runs on the 999-question validation subset (data/subsets/nextqa_val_ids.json)
PACK=<nextqa pack> MC=option bash data/scripts/nextqa_eval.sh <model> <tag> chain 640

# self-citation test
PACK=<nextqa pack> bash data/scripts/cite_eval.sh <model> <tag>

# entity-track test: detect the named entities on all 16 frames once, then mask them
cs-entity-tracks results/runs/<tag>_nq_plain.jsonl --pack <nextqa pack> --out results/runs/nextqa_entity_tracks.json
PACK=<nextqa pack> bash data/scripts/cite_track_eval.sh <model> <tag>

# does a cited box land on the entity the step names?
cs-cite-grounding results/runs/<tag>_nq_plain.jsonl --pack <nextqa pack> --out results/runs/<tag>_nq_cite_grounding.json

# diagnostics from the saved outputs (CPU)
cs-faithfulness-splits results/runs --tag <tag>_nq --tracks results/runs/nextqa_entity_tracks.json
```

A pack is a directory of pre-extracted frames (`frames/<video>/fNN.jpg`) built once with `cs-frames`
(`--eval-pack`, `--nextqa-pack`, `--causalvqa-pack`), so that the GPU machine never decodes video.

The random controls are seeded draws in output order. They are not stored; the analysis replays them exactly.

The scripts write to `results/runs/`, which is gitignored, unless `RESULTS_DIR` is set, so new runs never
overwrite the paper's summaries in `results/evals/`.

## Training

```bash
cs-triplets --split train --out data/triplets/clevrer_train.jsonl          # exact CLEVRER chains from the programs
cs-clevrer-validate descriptive --n-videos 1000                            # executor against the ground-truth answers
cs-pseudolabel teach ... && cs-pseudolabel ground ...                      # NExT-QA chains: teacher steps, detector boxes
cs-mix ...                                                                 # the CLEVRER, mixed and NExT-QA arms
cs-sft  --config configs/stage1_sft.yaml  --data <frames> --out runs/stage1
cs-grpo --config configs/stage2_outcome_only.yaml --data <frames> --out runs/stage2   # chain GRPO: outcome + format rewards
```

`configs/` holds one YAML per run in the paper. They were launched on Colab, so their data and adapter paths
point at a mounted Drive; replace them with your own. `configs/README.md` maps each config to its run and result
tag, and `colab/README.md` is the run sheet that was used.

## Results

`results/README.md` indexes the directories and lists the names that changed between the working notes and the
paper. `results/evals/` has the per-condition summaries and gap tables of
every run. File tags:

| Tag | Model |
|---|---|
| `qwen3b`, `ivl_base2b` | zero-shot bases |
| `grpo3b_bal`, `grpo3b_nat` | outcome-only GRPO on CLEVRER |
| `sft3b`, `chainrl3b` | CLEVRER chain SFT, CLEVRER chain GRPO |
| `csr3b*` | CLEVRER chain GRPO with process rewards (`_nonec` without the necessity term, `_sc` with step-level credit) |
| `mixsft3b`, `mixrl3b` | mixed chain SFT, mixed chain GRPO |
| `realsft3b`, `realrl3b` | NExT-QA chain SFT, NExT-QA chain GRPO |
| `mixplainsft3b`, `mixplainrl3b` | plain (chain-free) counterparts of the mixed arm |
| `ivl_*` | InternVL3-2B runs (`_lr` at the lower RL learning rate) |
| `videorft3b`, `videothinker3b`, `videor1_7b` | released checkpoints (`results/baselines/`) |

Suffixes: `_nq` NExT-QA, `_cvqa` CausalVQA, `_opt` an untrained model in the per-option format, `_s1` and `_s2`
further seeds. `mixplainsft3b_nq_mask_cite*` is the control in which a chain-free model is masked with the mixed
chain SFT model's citations.

## Layout

```
src/causalsight/
  data/               CLEVRER chain generator and validator, NExT-QA pseudo-labels, training mixes, audit sheets
  eval/               harness, masking interventions, entity tracks, citation grounding, benchmarks
  eval/faithfulness/  gaps with intervals (report) and their diagnostics (splits)
  models/             Qwen2.5-VL and InternVL3 backends, and a dummy backend for pipeline checks
  train/              SFT with the bridge loss, GRPO on chains, reward terms
configs/              one YAML per training run
data/scripts/         download and evaluation scripts
data/subsets/         fixed question subsets used for training and evaluation
results/              result summaries and analysis notes
tests/                CPU tests of the generator, the masking, the statistics and the rewards
```

## Data and models

The benchmarks are [CLEVRER](http://clevrer.csail.mit.edu/), [NExT-QA](https://github.com/doc-doc/NExT-QA) and
[CausalVQA](https://github.com/facebookresearch/CausalVQA). The bases are Qwen2.5-VL-3B-Instruct and InternVL3-2B;
the pseudo-label teacher is Qwen2.5-VL-7B-Instruct and the detector is Grounding DINO (tiny). Each comes with
its own licence. `data/subsets/` contains question identifiers and question text derived from CLEVRER and
NExT-QA for the fixed subsets used here.

## Citation

Preprint: [arXiv:2611.21940](https://arxiv.org/abs/2611.21940)

```bibtex
@article{gajjar2026does,
  title={Does the Model Use What It Cites? Counterfactual Tests of Grounded Reasoning Chains in Small Video-Language Models},
  author={Gajjar, Jugal and Subramaniakuppusamy, Kamalasankari},
  journal={arXiv preprint arXiv:2611.21940},
  year={2026}
}
```

## License

MIT. See `LICENSE`.
