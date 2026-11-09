# Mixed real/synthetic chain experiment (Qwen2.5-VL-3B; one seed per arm, a second SFT seed for the two mixed arms)

Arms: `mix` = 4,400 CLEVRER chains + 8,000 NExT-QA pseudo-labelled chains (teacher Qwen2.5-VL-7B steps, Grounding DINO boxes);
`real` = the 8,000 NExT-QA chains only; `plain` = the mix with bare `<answer>` responses (same prompts, no chains).
Each SFT arm is followed by 1,000 steps of GRPO (outcome + format) on 1,500 per-option items (750 CLEVRER + 750 NExT-QA)
(`real` RL: NExT-QA only). All chain-style models answer one yes/no item per option. The plain GRPO arm never produced
`<think>` text (its SFT taught bare answers), so it is an outcome-only RL control on bare answers.
The plain SFT arm was trained with the chain instruction in the prompt and evaluated the same way.
`seed 1` rows are a second SFT run of the same recipe and data (seed changes data order and bridge corruption;
`configs/sft_mix_*_s1.yaml`). The RL arms and the real-only arms have one seed.

## NExT-QA val subset (999 questions; per-option models)

`acc` = all five options judged correctly; `selection` picks the option judged "yes" (credit split on ties), comparable
to the letters prompt; always-"no" per-option rate = 0.80. The first two rows are the untrained references evaluated
in the same per-option format (their letters-prompt accuracy: base 0.730, GRPO balanced 0.750).

| model | acc | causal | temporal | descr | per-option | "yes" rate (20% gold) | selection | blind selection | blind acc |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B base (per option) | 0.461 | 0.426 | 0.411 | 0.547 | 0.860 | 0.19 | 0.595 | 0.270 | 0.099 |
| GRPO balanced (per option) | 0.355 | 0.381 | 0.336 | 0.348 | 0.804 | 0.15 | 0.522 | 0.260 | 0.083 |
| chain SFT (CLEVRER only) | 0.122 | 0.144 | 0.111 | 0.111 | 0.622 | 0.38 | 0.451 | 0.259 | 0.031 |
| + chain GRPO (CLEVRER only) | 0.194 | 0.201 | 0.201 | 0.180 | 0.682 | 0.41 | 0.470 | 0.272 | 0.048 |
| mix plain SFT | 0.445 | 0.378 | 0.408 | 0.550 | 0.826 | 0.32 | 0.637 | 0.396 | 0.198 |
| mix plain SFT, seed 1 | 0.487 | 0.414 | 0.468 | 0.580 | 0.849 | 0.29 | 0.661 | 0.386 | 0.194 |
| + plain GRPO | 0.436 | 0.363 | 0.402 | 0.544 | 0.822 | 0.33 | 0.631 | 0.383 | 0.179 |
| mix chain SFT | 0.476 | 0.468 | 0.393 | 0.568 | 0.845 | 0.27 | 0.637 | 0.357 | 0.141 |
| mix chain SFT, seed 1 | 0.483 | 0.459 | 0.408 | 0.583 | 0.846 | 0.27 | 0.642 | 0.377 | 0.189 |
| + chain GRPO | 0.463 | 0.441 | 0.420 | 0.529 | 0.837 | 0.29 | 0.640 | 0.358 | 0.130 |
| real chain SFT | 0.458 | 0.435 | 0.402 | 0.538 | 0.837 | 0.30 | 0.641 | 0.368 | 0.161 |
| + chain GRPO | 0.481 | 0.438 | 0.444 | 0.562 | 0.852 | 0.28 | 0.653 | 0.379 | 0.187 |

## CLEVRER (1,000 stratified validation questions; per-option accuracy for inferential types)

| model | descr | expl | pred | cf | ALL | blind gap | track gap | box IoU (descr/expl/pred/cf) |
|---|---|---|---|---|---|---|---|---|
| chain SFT (CLEVRER only) | 0.888 | 0.902 | 0.820 | 0.805 | 0.851 | 0.39 | 0.51 | 0.62 / 0.66 / 0.69 / 0.72 |
| + chain GRPO (CLEVRER only) | 0.924 | 0.947 | 0.832 | 0.801 | 0.871 | 0.37 | 0.50 | 0.57 / 0.64 / 0.67 / 0.70 |
| mix plain SFT | 0.892 | 0.970 | 0.900 | 0.848 | 0.906 | 0.39 | 0.48 | n/a (no chains) |
| mix plain SFT, seed 1 | 0.864 | 0.973 | 0.880 | 0.856 | 0.903 | 0.42 | 0.45 | n/a (no chains) |
| + plain GRPO | 0.904 | 0.971 | 0.892 | 0.843 | 0.904 | 0.39 | 0.49 | n/a (no chains) |
| mix chain SFT | 0.860 | 0.648 | 0.798 | 0.763 | 0.739 | 0.27 | 0.49 | 0.56 / 0.63 / 0.70 / 0.72 |
| mix chain SFT, seed 1 | 0.856 | 0.697 | 0.800 | 0.770 | 0.758 | 0.27 | 0.53 | 0.57 / 0.62 / 0.69 / 0.71 |
| + chain GRPO | 0.880 | 0.756 | 0.764 | 0.759 | 0.771 | 0.29 | 0.48 | 0.54 / 0.62 / 0.70 / 0.70 |
| real chain SFT | 0.476 | 0.512 | 0.622 | 0.521 | 0.533 | 0.06 | 0.46 | 0.27 / 0.22 / 0.15 / 0.19 |
| + chain GRPO | 0.464 | 0.485 | 0.674 | 0.529 | 0.535 | 0.08 | 0.44 | 0.24 / 0.22 / 0.18 / 0.18 |

## CausalVQA debug split

Per-option format for every row (always-"no" per-option rate 0.80). Letters-prompt references: base set acc 0.580 /
paired 0.430, GRPO balanced 0.515 / 0.350. Blind, per option: base 0.070 / 0.030, GRPO balanced 0.080 / 0.040.

| model | set acc | paired | per-option | "yes" rate |
|---|---|---|---|---|
| Qwen2.5-VL-3B base (per option) | 0.115 | 0.050 | 0.616 | 0.45 |
| GRPO balanced (per option) | 0.125 | 0.070 | 0.615 | 0.34 |
| chain SFT (CLEVRER only) | 0.020 | 0.010 | 0.461 | 0.54 |
| + chain GRPO (CLEVRER only) | 0.050 | 0.020 | 0.479 | 0.59 |
| mix plain SFT | 0.025 | 0.010 | 0.353 | 0.83 |
| + plain GRPO | 0.020 | 0.010 | 0.342 | 0.84 |
| mix chain SFT | 0.090 | 0.070 | 0.481 | 0.67 |
| + chain GRPO | 0.040 | 0.030 | 0.382 | 0.79 |
| real chain SFT | 0.030 | 0.020 | 0.378 | 0.80 |
| + chain GRPO | 0.030 | 0.020 | 0.383 | 0.79 |

## Reading

1. **Real-video chains repair the real-video collapse.** On NExT-QA the chain models go from per-question 0.12-0.19
   (CLEVRER-only) to 0.46-0.48, per-option from below the always-"no" rate to 0.84-0.85 above it, and selection accuracy
   from 0.45-0.47 to 0.64-0.65. The "yes" rate falls from 0.38-0.41 to 0.27-0.30 against 0.20 gold.
2. **But the repair comes from the real data, not from the chains.** The plain arms trained on the same questions with
   bare answers reach the same selection accuracy (0.63-0.64) and per-question accuracy within 2-3 points (0.44-0.45 vs
   0.46-0.48). Chain supervision neither helps nor hurts real-video accuracy at this scale and seed count.
   The second SFT seed confirms it: plain seed 1 reaches selection 0.661 and per-question 0.487, chain seed 1
   0.642 and 0.483; the seed-to-seed spread of the plain arm (0.637 to 0.661) is larger than any chain-vs-plain
   difference, so the two conditions are indistinguishable on NExT-QA accuracy.
3. **Mixing chain styles costs synthetic accuracy.** The mixed chain SFT drops CLEVRER explanatory from 0.902 to 0.648
   (ALL 0.851 to 0.739), while the plain mix on the same 4,400 CLEVRER questions reaches 0.970 explanatory (ALL 0.906),
   the best CLEVRER numbers of any model. Chain GRPO recovers part of the loss (explanatory 0.756). Object grounding of
   the mixed chains on CLEVRER stays at the CLEVRER-only level (box IoU 0.63-0.72).
   The second chain seed reproduces the cost (explanatory 0.697, ALL 0.758, blind gap 0.27, track gap
   0.53, box IoU 0.65), so it is a property of the recipe, not of one run.
4. **Real-only chains do not transfer to CLEVRER** (ALL 0.53, blind gap 0.06, box IoU 0.15-0.27): the synthetic chains
   are needed for synthetic grounding, so the mix is the only arm that is grounded on both.
5. **CausalVQA does not move.** Every arm stays at set accuracy <= 0.09 and now says "yes" to 67-84% of options.
   NExT-QA chains do not transfer to egocentric anticipation/planning questions. The untrained base in the same
   per-option format reaches only 0.115 set / 0.050 paired (0.580 / 0.430 with letters), so on this benchmark the
   per-option format alone removes most of the accuracy and every arm sits at or below that floor.
6. **The per-option format, not the training, explains the gap to the letters prompt.** The untrained base evaluated
   per option drops from 0.730 (letters) to 0.595 selection accuracy (0.461 per question), and GRPO balanced from
   0.750 to 0.522. Every real-data arm is above the base in the same format (selection 0.63-0.65, +4 to +6 points).
   The base has the highest raw per-option accuracy (0.860) because it is conservative ("yes" rate 0.19 against 0.20
   gold, 0.05 blind), which leaves 27% of questions with no "yes" at all (42% for GRPO balanced); the trained arms say
   "yes" more often (0.27-0.33; the mixed chain GRPO model leaves 9% of questions without a "yes" and gives two or more
   to 40%) and select the right option more often.
7. **On synthetic video the answers depend on the cited evidence; on real video they do not** (tables below). On
   CLEVRER every chain model has a positive cite gap with a CI above zero (0.030-0.089): blacking out what the model
   itself cited flips 2-3x more correct answers than blacking out boxes of the same size on the same frames, and the
   gap is as large as the gold evidence-span gap (mixed chain SFT 0.089 vs 0.094, mixed chain GRPO 0.042 vs 0.067).
   On NExT-QA the same models have cite gaps of 0.001-0.004 with CIs that include zero (mixed chain GRPO 0.042 on
   CLEVRER vs 0.004 on NExT-QA), while blacking out the whole video still costs them 15-25 points. The real-video
   citations are well formed and point at real regions, but the cited box is not necessary for the answer (see
   item 8 and the splits below for what is: the cited entity, partly). The
   CLEVRER-only chain GRPO model is the one model with a small real-video gap (0.015 [0.00,0.02]) on a 7%
   random-flip floor, and the cross-model control (-0.002) confirms the masking operation itself is neutral.
   Both halves replicate: the second mixed chain SFT seed has the same NExT-QA cite gap (0.003 [-0.00,0.01]),
   and an InternVL3-2B chain GRPO model has a CLEVRER cite gap of 0.047 [0.03,0.06], so the synthetic-video
   result is not specific to the Qwen base.
8. **The chain models do use the video more than the plain one, and removing the cited entities from every frame
   recovers only a fifth of that.** Per-option blind gaps on NExT-QA are 0.15-0.25 for the chain arms against 0.125
   for plain SFT, so chain supervision shifts weight from language priors to the frames even though it does not make
   the cited region decisive. The CLEVRER result rules out "the model cannot cite decisively": the same mechanism is
   answer-relevant where evidence is local and unique. The entity-track test (table below) asks whether NExT-QA
   evidence is just spatially redundant: every entity a chain names is re-detected on all 16 frames and every
   detection plus the cited boxes is blacked out, against same-size boxes elsewhere on the same frames. For the mixed
   chain SFT and the real chain GRPO models the gap rises from 0.001-0.003 (one box on a few frames) to 0.046-0.048
   [0.04,0.06] (the entity gone from every frame), against blind gaps of 0.154-0.223 and CLEVRER object-track gaps
   of 0.44-0.49 for the same models. Pooled, nine in ten correct answers stand (94-95% under the control), but the pooled
   number is diluted by the 80% of items that are correct rejections: on the items whose option is correct the
   entity-track gap is 0.098 [0.069,0.127] and 0.109 [0.081,0.137] (flips 19.4% vs 9.6% and 16.7% vs 5.8%), against
   blind flips of 21% and 34% on the same items. So the cited box is unnecessary, the cited entity is moderately
   necessary, and "decorative" is too strong a word for the real-video citations. The remainder points at the supervision: pseudo-labelled chains (teacher steps plus detector boxes)
   taught the model to cite objects that are not the decisive evidence.
9. **The real-video citations are not garbage: they land on the named object.** Re-detecting the entity each step
   names ("Where is the white dog?") with Grounding DINO on the cited frame (table below) finds it 87-93% of the
   time, and the cited box overlaps the detection with mean IoU 0.53-0.57 (hit@0.5 0.58-0.63), against 0.22-0.25
   (0.16-0.21) for a same-size box at a random position. So the chain cites the right object at the right place,
   and that object is still not what decides the answer. The detector is the same family that produced the
   training boxes, so this measures agreement with the pseudo-label source, not ground truth; and the entities are
   the ones the model chose to name, which are not necessarily the question-relevant ones. One pseudo-label
   artifact is visible in the outputs: 20-30% of the mixed models' chains (14% for the real-only arms) cite one
   identical box for every step, and in the blind condition 92-99% do, so a repeated box is the model's default
   when it has nothing to look at.

## Splits of the self-citation results (`cs-faithfulness-splits`)

`cs-faithfulness-splits results/colab --tag <tag> [--tracks results/colab/nextqa_entity_tracks.json]` writes
`<tag>_splits.json`: every gap separately for correct options (gold "yes") and wrong options (gold "no"), split by
whether every cited box hits the detector's box for the named entity (IoU >= 0.5), the masked area, and what the
chain written under the mask cites. Same question-clustered bootstrap as the main report.

NExT-QA. "all-hit" = items whose every cited box hits the detection (29-35% of items); "re-cite" = share of steps
written under the mask that cite the blacked-out box again (cited mask / random mask):

| model | cite gap, correct option | cite gap, wrong option | entity-track, correct option | entity-track, wrong option | cite gap, all-hit items | masked % | re-cite |
|---|---|---|---|---|---|---|---|
| chain GRPO (CLEVRER only) | 0.000 [-0.019,0.019] | 0.019 [0.007,0.032] | -- | -- | -- | 1.4 | 0.49 / 0.57 |
| mix chain SFT | 0.012 [-0.001,0.026] | 0.001 [-0.004,0.005] | 0.098 [0.069,0.127] | 0.037 [0.026,0.048] | 0.011 [0.004,0.019] | 3.9 | 0.14 / 0.52 |
| mix chain SFT, seed 1 | 0.003 [-0.010,0.015] | 0.003 [-0.001,0.007] | -- | -- | 0.005 [-0.003,0.013] | 3.5 | 0.16 / 0.53 |
| mix chain GRPO | 0.010 [0.001,0.018] | 0.002 [-0.003,0.007] | -- | -- | -0.001 [-0.009,0.007] | 3.6 | 0.15 / 0.52 |
| real chain SFT | 0.006 [-0.005,0.017] | 0.003 [-0.003,0.008] | -- | -- | -0.001 [-0.010,0.008] | 4.0 | 0.15 / 0.54 |
| real chain GRPO | 0.005 [-0.004,0.014] | -0.000 [-0.004,0.003] | 0.109 [0.081,0.137] | 0.031 [0.022,0.041] | 0.000 [-0.006,0.006] | 4.1 | 0.17 / 0.55 |
| control: mix plain SFT | 0.002 [-0.005,0.010] | -0.003 [-0.007,0.001] | -- | -- | -- | 3.9 | -- |

CLEVRER (descriptive questions excluded from the gold split):

| model | cite gap, correct option | cite gap, wrong option | gold-evidence gap, correct | gold-evidence gap, wrong | masked % | re-cite |
|---|---|---|---|---|---|---|
| chain GRPO (CLEVRER only) | 0.054 [0.035,0.074] | 0.051 [0.032,0.070] | 0.077 [0.051,0.104] | 0.073 [0.050,0.096] | 1.9 | 0.44 / 0.75 |
| mix chain SFT | 0.088 [0.061,0.116] | 0.106 [0.078,0.134] | 0.087 [0.061,0.115] | 0.108 [0.074,0.143] | 2.4 | 0.48 / 0.81 |
| mix chain GRPO | 0.029 [0.016,0.044] | 0.074 [0.047,0.101] | 0.050 [0.033,0.067] | 0.100 [0.070,0.134] | 1.4 | 0.53 / 0.83 |
| real chain GRPO | 0.021 [0.004,0.037] | 0.027 [0.009,0.045] | 0.078 [0.051,0.105] | 0.050 [0.023,0.077] | 0.4 | 0.45 / 0.77 |
| InternVL3-2B chain GRPO | 0.055 [0.039,0.073] | 0.034 [0.018,0.052] | 0.084 [0.062,0.107] | 0.068 [0.048,0.087] | 0.5 | 0.58 / 0.83 |

Reading: (a) the NExT-QA cite-gap null is not dilution by rejections (<= 0.012 on correct options) and not misplaced
boxes (<= 0.011 where every box hits); (b) the entity-track gap is concentrated on correct options (0.10-0.11); (c) on
CLEVRER the cite gap is symmetric over correct and wrong options; (d) chain-free models need the gold evidence as much
as chain models (gold-evidence gap: zero-shot base 0.063, outcome-only GRPO 0.097-0.119, plain mixed arms
0.067-0.071), so a positive cite gap shows the chain cites pixels the model needs, not that the chain makes them
needed; (e) under the cited-box mask the real-video models re-cite the same box in 14-17% of steps vs 52-55% under
the random mask: the citation follows the pixels, the answer does not. Box IoU over seeds (rescored 2026-11-09):
chain GRPO 0.654 / 0.665 / 0.660, CSR 0.668 / 0.666, CSR without necessity 0.637, step-level credit 0.3: 0.658.

## Manual audit of the pseudo-labelled chains

100 chains sampled uniformly from `mix/real_chains_sft.jsonl` (one per question, seed 0; 299 steps), judged on the
cited frame with the detector box drawn (`cs-pseudolabel-audit`; judgements in `pseudolabel_audit.csv`, rates in
`pseudolabel_audit_summary.json`).

| judgement | n | yes | no | unsure |
|---|---|---|---|---|
| step: named entity visible on the cited frame | 299 | 0.823 | 0.094 | 0.084 |
| step: box is on the named entity | 299 | 0.699 | 0.184 | 0.117 |
| step: relevant to the question | 299 | 0.953 | 0.037 | 0.010 |
| chain: steps suffice to answer | 100 | 0.89 | 0.08 | 0.03 |
| chain: a step beyond locating an entity | 100 | 0.08 | 0.92 | 0.00 |

The box is on the entity in 84% of the steps where the entity is visible (206/246); 51 chains have every step
grounded, 14 have none. Failure modes: box on the wrong one of two people, entity not on the cited frame, stock
phrases for entities not in the video ("the man in the blue sweater"), predicates as entities ("walking on the
path"), whole-frame boxes. The steps are relevant and would suffice if believed, but 92 of 100 chains contain only
"where is X" observations that restate the answer the teacher was shown.

## Self-cited evidence masking (per-option items)

For every item the regions cited in the model's own unmasked chain (`E: t=a-b box=...`, mapped from the 0-127 chain
grid onto the 16 sampled frames; whole-frame boxes dropped) are blacked out (`mask_cite`); the control blacks out boxes
of the same size on the same frames at random positions (`mask_cite_random`). `flip` = share of items correct in the
plain run that become wrong; the cite gap is their paired difference with a paired bootstrap 95% CI (resampled over questions, so the per-option items of one question move together; seeded). Items whose chain cites
no localizable region are skipped (`n_cite` of 4,995). The last row masks the plain-SFT model, which writes no chains,
with the mixed chain SFT model's citations: a check that the masking operation itself is neutral. The same test on
CLEVRER (2,537 stratified per-option items; the gold evidence-span gap from the CLEVRER suite alongside) follows.

NExT-QA (4,995 items):

| model | per-option acc | blind gap | n_cite | flip_cite | flip_cite_random | cite gap [95% CI] | causal cite gap |
|---|---|---|---|---|---|---|---|
| chain GRPO (CLEVRER only) | 0.682 | 0.124 | 4770 | 0.085 | 0.070 | 0.015 [0.00,0.02] | 0.017 |
| mix chain SFT | 0.845 | 0.223 | 4906 | 0.028 | 0.025 | 0.003 [-0.00,0.01] | 0.006 |
| mix chain SFT, seed 1 | 0.846 | 0.158 | 4914 | 0.023 | 0.020 | 0.003 [-0.00,0.01] | 0.002 |
| + chain GRPO | 0.837 | 0.249 | 4912 | 0.022 | 0.018 | 0.004 [-0.00,0.01] | 0.008 |
| real chain SFT | 0.837 | 0.194 | 4891 | 0.029 | 0.025 | 0.003 [-0.00,0.01] | 0.007 |
| + chain GRPO | 0.852 | 0.154 | 4889 | 0.016 | 0.015 | 0.001 [-0.00,0.00] | 0.000 |
| control: mix plain SFT, mix chain SFT's citations | 0.826 | 0.125 | 4906 | 0.014 | 0.016 | -0.002 [-0.01,0.00] | 0.000 |

Entity-track test on NExT-QA (`mask_cite_track` / `mask_cite_track_random`; items correct under the plain input
whose named entities were detected). Entities named by the chains (2,746 phrases over 485 videos) are re-detected
with Grounding DINO tiny (threshold 0.35) on all 16 sampled frames: detected on 14.0 of 16 frames on average, on
8 or more frames for 94% of phrases, never for 1.5%. The intervention covers 8.7% of pixel-frames (cited boxes
alone: 3.9%); the control places the same boxes at random positions that avoid the entity boxes on the same frame.

| model | n | flip (tracks) | flip (random) | entity-track gap [95% CI] | causal | temporal | descriptive | cite gap | blind gap |
|---|---|---|---|---|---|---|---|---|---|
| mix chain SFT | 4087 | 0.110 | 0.062 | 0.048 [0.04,0.06] | 0.046 | 0.036 | 0.061 | 0.003 | 0.223 |
| real chain GRPO | 4140 | 0.095 | 0.049 | 0.046 [0.04,0.06] | 0.043 | 0.031 | 0.065 | 0.001 | 0.154 |

Split by detector coverage: items whose entities were detected on >=14 frames on average have gaps of 0.053 (mix
chain SFT, n=3,088) and 0.050 (real chain GRPO, n=3,287); the rest (n=999 / 853) 0.032 and 0.033. Missed detections
do not hide a larger effect.

CLEVRER (2,537 items):

| model | per-option acc | blind gap | n_cite | flip_cite | flip_cite_random | cite gap [95% CI] | explanatory cite gap | gold evidence-span gap |
|---|---|---|---|---|---|---|---|---|
| chain GRPO (CLEVRER only) | 0.871 | 0.367 | 2527 | 0.088 | 0.036 | 0.051 [0.04,0.06] | 0.042 | 0.076 |
| mix chain SFT | 0.739 | 0.269 | 2512 | 0.165 | 0.077 | 0.089 [0.07,0.11] | 0.156 | 0.094 |
| + chain GRPO | 0.771 | 0.291 | 2520 | 0.078 | 0.036 | 0.042 [0.03,0.05] | 0.050 | 0.067 |
| real chain GRPO | 0.535 | 0.076 | 2533 | 0.069 | 0.038 | 0.030 [0.02,0.04] | 0.011 | 0.062 |
| InternVL3-2B chain GRPO (CLEVRER only) | 0.803 | 0.334 | 2532 | 0.084 | 0.037 | 0.047 [0.03,0.06] | 0.025 | 0.077 |

The real-chain GRPO model, trained on no CLEVRER chains and barely above chance there, still shows a positive CLEVRER
cite gap; its one large per-type gap is descriptive (0.129). Real chain SFT was not run on CLEVRER.
The InternVL row is the chain GRPO run at lr 1e-5 (`runs/internvl_stage2_outcome_only`, `ivl_chainrl2b_*`), whose
RL the results index marks as diverged and does not report; its CLEVRER accuracy (0.803) and track gap (0.549) are
intact, so it is a valid chain model for this test, but it is not the lr 3e-6 model in the InternVL table
(`ivl_chainrl2b_lr_*`, 0.783). Its largest per-type cite gap is predictive (0.108 [0.07,0.15]).
Mask size does not explain the contrast: the cited boxes cover a larger share of the frames on NExT-QA (mean masked
fraction 0.036-0.042 of pixel-frames for the real-data models) than on CLEVRER (0.004-0.024), yet only the CLEVRER
masks move the answers.

## Citation grounding on real video (`cs-cite-grounding`, NExT-QA plain runs)

For every parsed step of the form "Q: Where is <entity>?" the cited span is mapped to its centre frame and Grounding
DINO (tiny, threshold 0.35, the same detector family that produced the training boxes) is run on that frame with the
entity as the phrase. `detected_rate` = share of steps where the detector returns a box; IoU and hit@0.5 compare the
cited box with the top detection on detected steps; `random` is a same-size box at a random position on the same frame.
Steps whose box covers the whole frame are dropped. The CLEVRER-only chain GRPO model writes CLEVRER-style steps, not
"Where is" steps, so only 18 of its steps parse and its row is not informative.

| model | n_steps | detected_rate | mean IoU | hit@0.5 | random IoU | random hit@0.5 |
|---|---|---|---|---|---|---|
| chain GRPO (CLEVRER only) | 18 | 0.889 | 0.254 | 0.125 | 0.100 | 0.000 |
| mix chain SFT | 14274 | 0.900 | 0.544 | 0.603 | 0.230 | 0.180 |
| + chain GRPO | 14395 | 0.916 | 0.558 | 0.624 | 0.238 | 0.194 |
| real chain SFT | 14410 | 0.893 | 0.535 | 0.598 | 0.224 | 0.172 |
| + chain GRPO | 14448 | 0.908 | 0.545 | 0.620 | 0.234 | 0.182 |

Per question type the numbers are flat (causal IoU 0.53-0.55, descriptive 0.55-0.57, temporal 0.53-0.56). Chain GRPO
raises detection and IoU by 1-2 points over its SFT on both arms. Together with the cite-gap table this gives the
real-video picture: the model names a plausible entity, puts the box on it, and answers from elsewhere.
