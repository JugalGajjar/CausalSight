# Mixed real/synthetic chain experiment (Qwen2.5-VL-3B, single seed)

Arms: `mix` = 4,400 CLEVRER chains + 8,000 NExT-QA pseudo-labelled chains (teacher Qwen2.5-VL-7B steps, Grounding DINO boxes);
`real` = the 8,000 NExT-QA chains only; `plain` = the mix with bare `<answer>` responses (same prompts, no chains).
Each SFT arm is followed by 1,000 steps of GRPO (outcome + format) on 1,500 CLEVRER + 1,500 NExT-QA per-option items
(`real` RL: NExT-QA only). All chain-style models answer one yes/no item per option. The plain GRPO arm never produced
`<think>` text (its SFT taught bare answers), so it is an outcome-only RL control on bare answers.
The plain SFT arm was trained with the chain instruction in the prompt and evaluated the same way.

## NExT-QA val subset (999 questions; per-option models)

`acc` = all five options judged correctly; `selection` picks the option judged "yes" (credit split on ties), comparable
to the letters prompt; always-"no" per-option rate = 0.80; letters-prompt references: base 0.730, GRPO balanced 0.750.

| model | acc | causal | temporal | descr | per-option | "yes" rate (20% gold) | selection | blind selection | blind acc |
|---|---|---|---|---|---|---|---|---|---|
| chain SFT (CLEVRER only) | 0.122 | 0.144 | 0.111 | 0.111 | 0.622 | 0.38 | 0.451 | 0.259 | 0.031 |
| + chain GRPO (CLEVRER only) | 0.194 | 0.201 | 0.201 | 0.180 | 0.682 | 0.41 | 0.470 | 0.272 | 0.048 |
| mix plain SFT | 0.445 | 0.378 | 0.408 | 0.550 | 0.826 | 0.32 | 0.637 | 0.396 | 0.198 |
| + plain GRPO | 0.436 | 0.363 | 0.402 | 0.544 | 0.822 | 0.33 | 0.631 | 0.383 | 0.179 |
| mix chain SFT | 0.476 | 0.468 | 0.393 | 0.568 | 0.845 | 0.27 | 0.637 | 0.357 | 0.141 |
| + chain GRPO | 0.463 | 0.441 | 0.420 | 0.529 | 0.837 | 0.29 | 0.640 | 0.358 | 0.130 |
| real chain SFT | 0.458 | 0.435 | 0.402 | 0.538 | 0.837 | 0.30 | 0.641 | 0.368 | 0.161 |
| + chain GRPO | 0.481 | 0.438 | 0.444 | 0.562 | 0.852 | 0.28 | 0.653 | 0.379 | 0.187 |

## CLEVRER (1,000 stratified validation questions; per-option accuracy for inferential types)

| model | descr | expl | pred | cf | ALL | blind gap | track gap | box IoU (descr/expl/pred/cf) |
|---|---|---|---|---|---|---|---|---|
| chain SFT (CLEVRER only) | 0.888 | 0.902 | 0.820 | 0.805 | 0.851 | 0.39 | 0.51 | 0.62 / 0.66 / 0.69 / 0.72 |
| + chain GRPO (CLEVRER only) | 0.924 | 0.947 | 0.832 | 0.801 | 0.871 | 0.37 | 0.50 | 0.57 / 0.64 / 0.67 / 0.70 |
| mix plain SFT | 0.892 | 0.970 | 0.900 | 0.848 | 0.906 | 0.39 | 0.48 | n/a (no chains) |
| + plain GRPO | 0.904 | 0.971 | 0.892 | 0.843 | 0.904 | 0.39 | 0.49 | n/a (no chains) |
| mix chain SFT | 0.860 | 0.648 | 0.798 | 0.763 | 0.739 | 0.27 | 0.49 | 0.56 / 0.63 / 0.70 / 0.72 |
| + chain GRPO | 0.880 | 0.756 | 0.764 | 0.759 | 0.771 | 0.29 | 0.48 | 0.54 / 0.62 / 0.70 / 0.70 |
| real chain SFT | 0.476 | 0.512 | 0.622 | 0.521 | 0.533 | 0.06 | 0.46 | 0.27 / 0.22 / 0.15 / 0.19 |
| + chain GRPO | 0.464 | 0.485 | 0.674 | 0.529 | 0.535 | 0.08 | 0.44 | 0.24 / 0.22 / 0.18 / 0.18 |

## CausalVQA debug split

| model | set acc | paired | per-option | "yes" rate |
|---|---|---|---|---|
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
3. **Mixing chain styles costs synthetic accuracy.** The mixed chain SFT drops CLEVRER explanatory from 0.902 to 0.648
   (ALL 0.851 to 0.739), while the plain mix on the same 4,400 CLEVRER questions reaches 0.970 explanatory (ALL 0.906),
   the best CLEVRER numbers of any model. Chain GRPO recovers part of the loss (explanatory 0.756). Object grounding of
   the mixed chains on CLEVRER stays at the CLEVRER-only level (box IoU 0.63-0.72).
4. **Real-only chains do not transfer to CLEVRER** (ALL 0.53, blind gap 0.06, box IoU 0.15-0.27): the synthetic chains
   are needed for synthetic grounding, so the mix is the only arm that is grounded on both.
5. **CausalVQA does not move.** Every arm stays at set accuracy <= 0.09 and now says "yes" to 67-84% of options.
   NExT-QA chains do not transfer to egocentric anticipation/planning questions.
6. All per-option models remain below the letters-prompt base (selection 0.64-0.65 vs 0.73). Whether this is the
   per-option format or a real deficit needs the base model evaluated per option (pending).
