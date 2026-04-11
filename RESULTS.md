# Results

This file is the top-level results summary for the FHE privacy experiment implementation in this repo.

It is intended to be a living document. Repo-root results should be summarized here rather than scattered across run folders.

## Status

Completed:

- Phase 0: spec freeze
- Phase 1: deterministic data layer
- Phase 2 Stage 1: Qwen2-1.5B-Instruct training
- Phase 2 Stage 1: membership inference evaluation
- Phase 3: Stage 2 baseline replay and leakage scoring
- Phase 4: Stage 3 plaintext filter training and evaluation
- Phase 5: Stage 3 CKKS/OpenFHE filter parity and latency evaluation
- Phase 6: integrated Stage 2 reruns with plaintext and FHE filters

Remaining:

- final packaging and artifact handoff
- optional Qwen2-7B-Instruct repeat

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
- Stage 2 baseline summary: `experiment_runtime/runs/stage2/baseline/stage2_summary.json`
- Stage 3 plaintext summary: `experiment_runtime/runs/stage3/plaintext/stage3_plaintext_metrics.json`
- Stage 3 FHE summary: `experiment_runtime/runs/stage3/fhe/stage3_fhe_metrics.json`
- Stage 2 filtered summary: `experiment_runtime/runs/stage2/filtered/stage2_filtered_summary.json`

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
- The `50x` run ended with a lower training loss than `1x` and `10x`, but the meaningful privacy signal comes from the MIA and extraction results below.

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

## Stage 2 Baseline Replay Results

Headline Stage 2 baseline metrics:

| Exposure | No-system any leak | No-system full leak | System-prompt any leak | System-prompt full leak | System-prompt refusal rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1x` | 0.964 | 0.400 | 0.036 | 0.000 | 0.000 |
| `10x` | 0.944 | 0.404 | 0.052 | 0.004 | 0.000 |
| `50x` | 0.944 | 0.372 | 0.232 | 0.052 | 0.000 |

Baseline interpretation:

- Without the system prompt, the model is extremely leaky across all three exposure conditions.
- With the system prompt active, leakage drops sharply at `1x` and `10x`.
- With the system prompt active, leakage rises substantially at `50x`, which is the key Stage 2 baseline result.
- `other_canary_leak_rate` is `0.0` in all six baseline runs.
- `refusal_rate` is also `0.0` in all six baseline runs, so the system prompt is not protecting mainly by emitting the exact refusal string.

## Stage 3 Plaintext Filter Results

Held-out plaintext filter metrics:

| Metric | Value |
| --- | ---: |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Selected `C` | 10.0 |
| Selected threshold | 0.4200 |
| `ALLOW` precision | 0.9868 |
| `ALLOW` recall | 0.9933 |
| `ALLOW` F1 | 0.9900 |
| `BLOCK` precision | 0.9933 |
| `BLOCK` recall | 0.9867 |
| `BLOCK` F1 | 0.9900 |
| Macro F1 | 0.9900 |
| Accuracy | 0.9900 |
| Adversarial block rate | 0.9867 |
| Benign false positive rate | 0.0067 |

Plaintext filter interpretation:

- The plaintext sentence-embedding plus logistic-regression filter performs extremely well on the held-out Stage 3 test split.
- The exact linear weights, intercept, and threshold were materialized for later FHE wrapping.
- These numbers are strong, but they are still Stage 3 dataset metrics rather than end-to-end attack-path results.

## Stage 3 FHE Filter Results

CKKS/OpenFHE parity and latency summary:

| Metric | Value |
| --- | ---: |
| Backend | `openfhe` |
| Scheme | `CKKS` |
| Ring dimension | 16384 |
| Embedding dimension | 384 |
| Compiled bundle reused | `true` |
| Plaintext accuracy | 0.9900 |
| FHE accuracy | 0.9900 |
| Plaintext macro F1 | 0.9900 |
| FHE macro F1 | 0.9900 |
| Prediction match rate | 1.0000 |
| Plaintext vs FHE accuracy delta | 0.0000 |
| Mean abs probability delta | 6.4979e-10 |
| Max abs probability delta | 8.1261e-09 |
| End-to-end latency mean (ms) | 34.5460 |
| End-to-end latency p50 (ms) | 34.4224 |
| End-to-end latency p95 (ms) | 35.7685 |

FHE interpretation:

- The CKKS/OpenFHE wrapper preserved the plaintext classifier behavior exactly on the held-out Stage 3 test set.
- Probability drift stayed negligible, at roughly `1e-9` mean and `1e-8` max.
- A reusable compiled-equivalent OpenFHE bundle was serialized and preserved through Git LFS, with a lightweight manifest committed alongside it for fast review.
- The current Stage 3 FHE path keeps sentence encoding plaintext and applies the threshold locally after decrypting the CKKS score.

## Integrated Stage 2 Filtered Rerun Results

System-prompt-active baseline vs filtered reruns:

| Exposure | Baseline any leak | Plaintext-filter any leak | FHE-filter any leak | Plaintext block rate | FHE block rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1x` | 0.036 | 0.000 | 0.000 | 0.988 | 0.988 |
| `10x` | 0.052 | 0.000 | 0.000 | 0.988 | 0.988 |
| `50x` | 0.232 | 0.000 | 0.000 | 0.988 | 0.988 |

Integrated filtered rerun notes:

- `full_canary_record_leak_rate` is `0.0` in all six filtered runs.
- `leak_rate_given_allowed` is `0.0` in all six filtered runs.
- Each filtered run blocks `247 / 250` attack prompts and allows `3 / 250`.
- Plaintext-vs-FHE filter decision parity on the real Stage 2 attack set is exact:
  - decision match rate `1.0`
  - mismatched decisions `0`
  - mean abs filter probability delta `6.6644e-10`
  - max abs filter probability delta `2.3366e-08`

Integrated rerun interpretation:

- On the frozen Stage 2 attack set, the filter layer removes all measured leakage for all three exposure conditions.
- This is the strongest result in the repo: the earlier `50x` memorization and extraction signal is real, but the added filter layer is strong enough to suppress it completely on the tested attack distribution.
- The FHE-wrapped filter preserves the same decisions and same leak outcomes as the plaintext filter on the real attack path, not just on the held-out Stage 3 split.

## Final 1.5B Interpretation

Current judgment:

- the completed 1.5B experiment now looks coherent end to end
- the results do not suggest an obvious implementation bug
- the defense story is now measured, not speculative

Why this pattern is credible:

- whole-population MIA is weak at `1x` and `10x`, then meaningfully stronger at `50x`
- the strongest Stage 1 effect is concentrated on the overexposed canary subset
- Stage 2 baseline replay tells the same story: system-prompt-active leakage stays low at `1x` and `10x`, then rises materially at `50x`
- Stage 3 plaintext evaluation shows a simple linear filter can separate the curated `ALLOW` and `BLOCK` distributions with very high held-out accuracy
- the integrated rerun confirms that this same filter actually suppresses the real Stage 2 leakage path
- the FHE wrapper preserves that behavior with effectively zero decision drift

Most important takeaway:

- increasing canary exposure from `1x` and `10x` to `50x` materially increases memorization and prompt-driven leakage risk
- on this experiment setup, adding a lightweight classifier filter in front of the fraud scorer removes the measured leakage on the frozen attack set
- the CKKS/OpenFHE version preserves the same filter decisions and outcomes, with roughly `35 ms` end-to-end scoring latency per example in the recorded benchmark

Current cautions:

- the integrated rerun uses the frozen Stage 2 attack set, not mixed benign production traffic
- the integrated filtered runs are attack-only, so benign false positive behavior should still be read from the Stage 3 held-out classifier evaluation
- the no-system baseline remains a stress baseline rather than a realistic deployment setting
- the optional `Qwen2-7B-Instruct` repeat has not been run yet

## Remaining Work

For the main `Qwen2-1.5B-Instruct` flow, the scientific part of the experiment is complete.

Remaining repo work is operational:

- final packaging and artifact handoff
- final check that the heavy-artifact archive includes Stage 1 adapters/checkpoints and the Stage 3 FHE compiled bundle
- optional `Qwen2-7B-Instruct` repeat if the designer still wants the scale comparison

## Detailed Artifacts

Primary detailed artifacts for the completed 1.5B flow:

- `experiment_runtime/runs/stage1/official_runs_summary.json`
- `experiment_runtime/runs/stage1/mia/mia_summary.json`
- `experiment_runtime/runs/stage1/mia/1x/stage1_metrics.json`
- `experiment_runtime/runs/stage1/mia/10x/stage1_metrics.json`
- `experiment_runtime/runs/stage1/mia/50x/stage1_metrics.json`
- `experiment_runtime/runs/stage1/mia/1x/canary_metrics.json`
- `experiment_runtime/runs/stage1/mia/10x/canary_metrics.json`
- `experiment_runtime/runs/stage1/mia/50x/canary_metrics.json`
- `experiment_runtime/runs/stage2/baseline/stage2_summary.json`
- `experiment_runtime/runs/stage2/baseline/1x/system_prompt_active/stage2_metrics.json`
- `experiment_runtime/runs/stage2/baseline/10x/system_prompt_active/stage2_metrics.json`
- `experiment_runtime/runs/stage2/baseline/50x/system_prompt_active/stage2_metrics.json`
- `experiment_runtime/runs/stage3/plaintext/stage3_plaintext_metrics.json`
- `experiment_runtime/runs/stage3/plaintext/model_selection.csv`
- `experiment_runtime/runs/stage3/plaintext/validation_threshold_sweep.csv`
- `experiment_runtime/runs/stage3/plaintext/model/model_parameters.json`
- `experiment_runtime/runs/stage3/fhe/stage3_fhe_metrics.json`
- `experiment_runtime/runs/stage3/fhe/latency_summary.json`
- `experiment_runtime/runs/stage3/fhe/plaintext_vs_fhe_comparison.csv`
- `experiment_runtime/runs/stage3/fhe/compiled_bundle_manifest.json`
- `experiment_runtime/runs/stage2/filtered/stage2_filtered_summary.json`
- `experiment_runtime/runs/stage2/filtered/filter_parity_summary.json`
- `experiment_runtime/runs/stage2/filtered/1x/plaintext_filter/stage2_filtered_metrics.json`
- `experiment_runtime/runs/stage2/filtered/10x/plaintext_filter/stage2_filtered_metrics.json`
- `experiment_runtime/runs/stage2/filtered/50x/plaintext_filter/stage2_filtered_metrics.json`
- `experiment_runtime/runs/stage2/filtered/1x/fhe_filter/stage2_filtered_metrics.json`
- `experiment_runtime/runs/stage2/filtered/10x/fhe_filter/stage2_filtered_metrics.json`
- `experiment_runtime/runs/stage2/filtered/50x/fhe_filter/stage2_filtered_metrics.json`
