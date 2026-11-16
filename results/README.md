# Results

All runs use matched preprocessing: 16 uniformly sampled frames and greedy decoding. CLEVRER is 1,000 stratified
validation questions (2,537 per-option items), NExT-QA a 999-question validation subset (4,995 per-option items),
and CausalVQA the debug split.

| Folder | Contents |
|---|---|
| `evals/` | Per-condition summaries, gap tables and diagnostics of every trained and zero-shot model. The paper's tables are generated from these files. One flat folder, files named `<tag>_<condition>`; the tag key is in the top-level README. |
| `baselines/` | The released checkpoints (VideoRFT-3B, VideoThinker-R1-3B, Video-R1-7B) on the same CLEVRER items. |
| `audits/` | Judgements and rates of the three manual audits: the pseudo-labelled chains, the models' own citations, and the models' verdicts. |
| `notes/` | The notes written as the experiments ran, one file per stage. |

Per-item outputs (`*.jsonl`) and the audit sheets (`*.html`) are not tracked.

## Notes

| File | Covers |
|---|---|
| `notes/stage0.md` | zero-shot base and outcome-only GRPO on CLEVRER; how the object-track test was developed |
| `notes/stage1.md` | CLEVRER chain SFT |
| `notes/stage2.md` | GRPO on chains, process rewards, step-level credit, seeds |
| `notes/internvl.md` | the second base, InternVL3-2B |
| `notes/nextqa.md`, `notes/causalvqa.md` | CLEVRER-trained models on NExT-QA and CausalVQA |
| `notes/mixed.md` | the mixed-data experiment, the self-citation and entity-track tests, their diagnostics, and the manual audits |
| `notes/clevrer_format_report.md` | output of `cs-clevrer-check` on the CLEVRER files |

The notes are chronological and record the project as it developed, so early entries use earlier names and a
framing that later results revised; the paper is the final reading. Names that changed:

| In the notes | In the paper |
|---|---|
| ES gap, evidence-span gap | gold-evidence gap |
| track gap | object-track gap |
| cite-track gap | entity-track gap |
| CSR | chain GRPO with grounding, necessity and process rewards |
| obj_iou | box IoU against the named object |
| Stage 0 / 1 / 2 | outcome-only GRPO / chain SFT / GRPO on chains |
| mix, real, plain arms | mixed, NExT-QA chain and plain (chain-free) arms |
