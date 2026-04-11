# Results

This file is the top-level results summary for the FHE privacy experiment implementation in this repo.

It is intended to be a living document. As later phases complete, new result sections should be appended here rather than scattered across separate notes.

## Status

Completed:

- Phase 0: spec freeze
- Phase 1: deterministic data layer
- Phase 2 Stage 1: Qwen2-1.5B-Instruct training
- Phase 2 Stage 1: membership inference evaluation

Remaining:

- Phase 3: Stage 2 replay and leakage scoring
- Phase 4: Stage 3 plaintext filter training and evaluation
- Phase 5: Stage 3 FHE filter evaluation and latency
- Phase 6: integrated reruns and packaging

## Stage 1 Setup

Primary model:

- `Qwen/Qwen2-1.5B-Instruct`

Frozen exposure conditions:

- `1x`
- `10x`
- `50x`

Frozen comparison rule:

- same optimizer-step budget across all three exposure conditions
- full-sequence causal LM loss, not assistant-only masking

Key artifacts:

- training summary: `experiment_runtime/runs/stage1/official_runs_summary.json`
- MIA summary: `experiment_runtime/runs/stage1/mia/mia_summary.json`

## Stage 1 Training Results

Official runs:

| Exposure | Run name | Train examples | Max steps | Final global step | Final train loss | Train runtime (s) | Steps/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1x` | `official-1x-20260411-r1` | 8,000 | 1,000 | 1,000 | 0.3637 | 249.0624 | 4.015 |
| `10x` | `official-10x-20260411-r1` | 8,900 | 1,000 | 1,000 | 0.3644 | 253.4912 | 3.945 |
| `50x` | `official-50x-20260411-r1` | 12,900 | 1,000 | 1,000 | 0.3221 | 253.4363 | 3.946 |

Training interpretation:

- All three runs completed cleanly to `global_step = 1000`.
- Throughput and total compute were stable across exposure conditions, which is what the fixed-step protocol requires.
- The `50x` run ended with a noticeably lower training loss than `1x` and `10x`, but this by itself is not the final experiment result. The MIA results below are the relevant signal.

## Stage 1 MIA Results

Full-population MIA metrics:

| Exposure | AUC-ROC | 95% CI | TPR@1%FPR | 95% CI | TPR@10%FPR | 95% CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| `1x` | 0.5087 | [0.4961, 0.5229] | 0.0163 | [0.0081, 0.0233] | 0.1068 | [0.0925, 0.1196] |
| `10x` | 0.5122 | [0.4999, 0.5263] | 0.0150 | [0.0100, 0.0234] | 0.1119 | [0.0982, 0.1274] |
| `50x` | 0.5546 | [0.5415, 0.5684] | 0.0391 | [0.0305, 0.0504] | 0.1610 | [0.1450, 0.1780] |

Canary-only MIA metrics:

| Exposure | Canary AUC-ROC | Canary TPR@1%FPR | Canary TPR@10%FPR |
| --- | ---: | ---: | ---: |
| `1x` | 0.5241 | 0.02 | 0.12 |
| `10x` | 0.6477 | 0.09 | 0.25 |
| `50x` | 1.0000 | 1.00 | 1.00 |

## Current Interpretation

Current judgment:

- the Stage 1 results look plausible and in line with the experiment design
- they do not currently suggest an obvious implementation bug

Why this looks like the expected pattern:

- whole-population MIA is weak at `1x` and `10x`, which is reasonable because only `100` of the `8,000` member records are overexposed canaries
- whole-population MIA becomes meaningfully stronger at `50x`, which is the intended direction of the exposure manipulation
- the strongest signal is canary-specific: canary MIA rises sharply with exposure count

Most important takeaway so far:

- increasing canary exposure from `1x` and `10x` to `50x` materially increases MIA signal, especially on the canary subset
- at `50x`, canary-only MIA is perfectly separated in the produced evaluation artifacts: canary AUC-ROC is `1.0`, with `TPR@1%FPR = 1.0` and `TPR@10%FPR = 1.0`

Why this does not currently look suspicious:

- the strongest effect is concentrated on the canary subset rather than appearing as a bizarre across-the-board jump at every exposure level
- if the implementation were badly wrong, a more suspicious pattern would be either:
  - strong leakage already at `1x`, or
  - no clear increase as exposure rises
- that is not what we see

Current caution:

- the `50x` canary-only result is extremely strong, so it should be treated as a real signal to validate with Stage 2 rather than as proof on its own
- the next major check is whether Stage 2 targeted extraction results are also clearly stronger for `50x`

This means the project has moved past pure implementation validation and into actual experimental output.

## Pending Result Sections

The following sections should be added as later tasks complete:

- Stage 2 replay and leakage scoring
- Stage 3 plaintext filter training and held-out metrics
- Stage 3 FHE filter metrics and latency
- Integrated final attack reruns with filter active
- Final cross-stage interpretation

## Detailed Artifacts

Primary detailed artifacts for the completed stages:

- `experiment_runtime/runs/stage1/official_runs_summary.json`
- `experiment_runtime/runs/stage1/mia/mia_summary.json`
- `experiment_runtime/runs/stage1/mia/1x/stage1_metrics.json`
- `experiment_runtime/runs/stage1/mia/10x/stage1_metrics.json`
- `experiment_runtime/runs/stage1/mia/50x/stage1_metrics.json`
- `experiment_runtime/runs/stage1/mia/1x/canary_metrics.json`
- `experiment_runtime/runs/stage1/mia/10x/canary_metrics.json`
- `experiment_runtime/runs/stage1/mia/50x/canary_metrics.json`
