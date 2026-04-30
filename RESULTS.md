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
- closeout audit and handoff verification for the 1.5B flow
- follow-on adaptive attacker and mixed-traffic evaluation on the NVIDIA/OpenFHE box

Handoff status:

- the main `Qwen2-1.5B-Instruct` experiment is complete and handoff-ready
- the follow-on adaptive and mixed-traffic results are complete and summarized below

Optional follow-on work:

- optional `Qwen2-7B-Instruct` repeat
- optional LangGraph runtime wiring/parity check if the LangGraph shell itself should become the authoritative end-to-end execution path
- optional threshold sensitivity, keyword/rule baselines, and broader generalization checks

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
- final audit summary: `experiment_runtime/runs/final_artifact_audit.json`

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

## Follow-On Adaptive And Mixed Evaluation

Follow-on setup:

- adaptive attack-only dataset: `350` scaffold-aware prompts across `7` families
- mixed-traffic dataset: `700` total rows, with `350` benign rows and `350` adaptive adversarial rows
- model artifacts: existing `Qwen2-1.5B-Instruct` `1x`, `10x`, and `50x` adapters
- official metrics path: direct follow-on harness under `experiment_runtime/src/experiment/follow_on/`, not the LangGraph scaffold
- confidence intervals: deterministic 95% Wilson binomial intervals; each CI artifact records numerator and denominator

Adaptive attack-only headline rates:

| Exposure | No-system any/full leak | System-prompt any/full leak | Plaintext filter any/block/leak-given-allowed | FHE filter any/block/leak-given-allowed |
| --- | ---: | ---: | ---: | ---: |
| `1x` | 0.5257 / 0.2571 | 0.0000 / 0.0000 | 0.0000 / 0.6400 / 0.0000 | 0.0000 / 0.6400 / 0.0000 |
| `10x` | 0.5200 / 0.2514 | 0.0000 / 0.0000 | 0.0000 / 0.6400 / 0.0000 | 0.0000 / 0.6400 / 0.0000 |
| `50x` | 0.5886 / 0.2514 | 0.0343 / 0.0200 | 0.0000 / 0.6400 / 0.0000 | 0.0000 / 0.6400 / 0.0000 |

Adaptive interpretation:

- The adaptive no-system stress path is leaky across all exposures.
- The system prompt suppresses adaptive leakage at `1x` and `10x`, but `50x` still leaks on `12 / 350` adaptive prompts, including `7 / 350` full-record leaks.
- Both filtered adaptive paths report zero measured leakage. The filter blocks `224 / 350` adaptive prompts and allows `126 / 350`; the allowed adaptive prompts did not leak under the system-prompt-active scorer path.

Mixed-traffic headline rates:

| Exposure | Filter | Adversarial block | Benign false positive | Benign allow | Adaptive any/full leak | Leak-given-allowed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1x` | plaintext | 0.6400 | 0.3143 | 0.6857 | 0.0000 / 0.0000 | 0.0000 |
| `1x` | FHE | 0.6400 | 0.3143 | 0.6857 | 0.0000 / 0.0000 | 0.0000 |
| `10x` | plaintext | 0.6400 | 0.3143 | 0.6857 | 0.0000 / 0.0000 | 0.0000 |
| `10x` | FHE | 0.6400 | 0.3143 | 0.6857 | 0.0000 / 0.0000 | 0.0000 |
| `50x` | plaintext | 0.6400 | 0.3143 | 0.6857 | 0.0000 / 0.0000 | 0.0000 |
| `50x` | FHE | 0.6400 | 0.3143 | 0.6857 | 0.0000 / 0.0000 | 0.0000 |

Mixed-traffic interpretation:

- Plaintext and FHE filter behavior is identical on the mixed dataset.
- The mixed synthetic benign set exposes a cost that the original attack-only reruns could not measure: `110 / 350` benign rows are blocked as false positives.
- Measured adaptive leakage remains zero in mixed traffic for all exposure/filter combinations.
- The mixed filter metrics are identical across `1x`, `10x`, and `50x` because the filter runs on the input message before any exposure-specific Qwen adapter is called. Exposure can affect only the generated response for allowed rows, not the filter decision for a fixed mixed-traffic input.

Mixed-traffic false-positive diagnosis:

- The high benign false-positive rate is a substantive filter utility issue, not an FHE parity bug. Plaintext and FHE decisions match exactly, so the FHE wrapper is preserving the plaintext classifier behavior.
- The original Stage 3 held-out classifier test had only `1 / 150 = 0.0067` benign false positives, while the follow-on mixed set has `110 / 350 = 0.3143`. This means the Stage 3 held-out set was too close to the curated training distribution to fully characterize benign-traffic robustness.
- False positives are concentrated by benign family:
  - `case_queue_review`: `45 / 70 = 0.6429`
  - `routing_followup`: `31 / 70 = 0.4429`
  - `risk_factor_summary`: `16 / 70 = 0.2286`
  - `audit_risk_review`: `12 / 70 = 0.1714`
  - `operational_transaction_scoring`: `6 / 70 = 0.0857`
- The blocked benign prompts often use legitimate fraud-workflow language around review queues, routing, audit, risk summaries, and customer-identity handling. That language overlaps with adversarial audit/compliance/extraction prompt families, so the filter learned a conservative boundary that protects privacy but overblocks some operational traffic.
- The correct interpretation is therefore narrower than "production-ready filter": the filter suppresses measured leakage and the FHE version preserves its decisions, but the current threshold and training distribution are not yet utility-calibrated for broader benign traffic.
- A follow-up ablation should treat this as the main open issue: run threshold sensitivity, expand benign/hard-negative training and validation coverage, and report privacy/utility tradeoff curves rather than only the single frozen Stage 3 threshold.

Plaintext-vs-FHE parity:

- adaptive parity: exact decision match for all `350` rows per exposure; mismatched row IDs `[]`
- mixed parity: exact decision match for all `700` rows per exposure; mismatched row IDs `[]`
- adaptive probability drift: mean absolute delta `3.7370e-09`, max `3.0377e-08`
- mixed probability drift: mean absolute delta `3.3106e-09`, max `3.3018e-08`

Follow-on timing:

| Path | Mean total filter time (ms) | P95 total filter time (ms) |
| --- | ---: | ---: |
| adaptive plaintext filter | 9.68 | 11.05 |
| adaptive FHE filter | 58.66 | 60.67 |
| mixed plaintext filter | 8.88 | 10.26 |
| mixed FHE filter | 56.96 | 58.92 |

Pipeline timing summaries also include count, mean, p50, p90, p95, p99, min, max, and std for every measured run. Mean total pipeline timing ranges were:

- adaptive no-system: `462.54` to `499.27` ms per row
- adaptive system-prompt-only: `191.65` to `223.52` ms per row
- adaptive plaintext-filtered: `78.53` to `79.45` ms per row
- adaptive FHE-filtered: `127.19` to `127.45` ms per row
- mixed plaintext-filtered: `105.94` to `118.39` ms per row
- mixed FHE-filtered: `152.60` to `167.60` ms per row

Follow-on artifact pointers:

- adaptive summary: `experiment_runtime/runs/follow_on/adaptive/adaptive_summary.json`
- adaptive CI summary: `experiment_runtime/runs/follow_on/adaptive/adaptive_ci_summary.json`
- adaptive parity summary: `experiment_runtime/runs/follow_on/adaptive/filter_parity_summary.json`
- mixed summary: `experiment_runtime/runs/follow_on/mixed_traffic/mixed_traffic_summary.json`
- mixed CI summary: `experiment_runtime/runs/follow_on/mixed_traffic/mixed_traffic_ci_summary.json`
- mixed parity summary: `experiment_runtime/runs/follow_on/mixed_traffic/filter_parity_summary.json`
- setup timing aggregate: `experiment_runtime/runs/follow_on/timing/setup_timing.json`
- per-sweep setup timing: `experiment_runtime/runs/follow_on/timing/setup_timing_adaptive.json` and `experiment_runtime/runs/follow_on/timing/setup_timing_mixed_traffic.json`
- setup timing manifest: `experiment_runtime/runs/follow_on/timing/setup_timing_manifest.json`

Follow-on limitations:

- the follow-on uses only the existing `Qwen2-1.5B-Instruct` adapters
- mixed traffic is still synthetic
- sentence encoding remains plaintext before plaintext/FHE filter scoring
- threshold sensitivity and keyword/rule baselines are intentionally deferred
- broader generalization checks are intentionally deferred
- official metrics still come from the direct experiment harness, not the LangGraph scaffold

## Final 1.5B Interpretation

Current judgment:

- the completed 1.5B experiment now looks coherent end to end
- the results do not suggest an obvious implementation bug
- the defense story is now measured on both the frozen Stage 2 attack set and a follow-on adaptive/mixed dataset

Why this pattern is credible:

- whole-population MIA is weak at `1x` and `10x`, then meaningfully stronger at `50x`
- the strongest Stage 1 effect is concentrated on the overexposed canary subset
- Stage 2 baseline replay tells the same story: system-prompt-active leakage stays low at `1x` and `10x`, then rises materially at `50x`
- Stage 3 plaintext evaluation shows a simple linear filter can separate the curated `ALLOW` and `BLOCK` distributions with very high held-out accuracy
- the integrated rerun confirms that this same filter actually suppresses the real Stage 2 leakage path
- the FHE wrapper preserves that behavior with effectively zero decision drift
- the follow-on confirms exact plaintext/FHE decision parity on scaffold-aware adaptive attacks and mixed synthetic traffic

Most important takeaway:

- increasing canary exposure from `1x` and `10x` to `50x` materially increases memorization and prompt-driven leakage risk
- on this experiment setup, adding a lightweight classifier filter in front of the fraud scorer removes the measured leakage on the frozen attack set and on the follow-on adaptive set
- the CKKS/OpenFHE version preserves the same filter decisions and outcomes; the follow-on warm filter timing is roughly `57` to `59 ms` per filtered row

Current cautions:

- the follow-on mixed traffic is synthetic, not production traffic
- the follow-on mixed run shows a materially higher synthetic benign false positive rate than the original Stage 3 held-out classifier test split
- the no-system baseline remains a stress baseline rather than a realistic deployment setting
- the optional `Qwen2-7B-Instruct` repeat has not been run yet

## Agentic Execution Note

The original guide assumed a lightweight multi-agent system rather than a single monolithic script. The frozen implementation plan then made the repo-level orchestration choice explicit by adopting LangGraph for that runtime shell. In this repo, that agentic shape exists as a LangGraph runtime under `experiment_runtime/src/qwen_langgraph_demo/`, and the intended graph is still the one described throughout the project:

```text
intake -> filter_middleware -> fraud_scorer -> router
```

That means the repo is not missing the agentic architecture. The Transaction Intake role, the filter position on the intake-to-fraud edge, the Fraud Scoring role, and the final routing role are all represented in the project structure and runtime scaffold. So at the architecture level, the project stayed aligned with the original agentic design and with the plan's LangGraph implementation choice, rather than drifting into a completely different non-agentic design.

However, the official reported experiment metrics in this results file were not generated by literally executing the LangGraph runtime as the evaluation harness. Stage 1 training, Stage 1 MIA, Stage 2 baseline replay, Stage 3 plaintext filter training, Stage 3 FHE evaluation, and the integrated filtered reruns were all measured through the dedicated `experiment.*` CLIs and run artifacts. That was an intentional engineering choice. The direct experiment harness was easier to control, easier to audit, and easier to keep deterministic for measurement than a higher-level orchestration shell.

This distinction matters, but it does not mean the experiment is invalid or off-spec in any serious way. The scientific questions in this project were about whether the fine-tuned model memorised canaries, whether adversarial inter-agent prompts could extract memorised fields, whether a plaintext filter could block those attacks, and whether the CKKS/OpenFHE version preserved that defensive behavior. Those questions were answered by evaluating the actual scorer, filter, and attack datasets directly. Using a direct harness for those measurements made the results cleaner, not weaker.

So the most accurate summary is:

- the project is architecturally aligned with the intended multi-agent design and with the plan's LangGraph runtime choice
- the official metrics were produced by the direct experiment harness rather than by LangGraph-executed runs
- the repo therefore proves the experiment components and the evaluation story, but it does not yet prove that the LangGraph shell itself was the exact runtime used to generate the reported numbers

In practical production terms, the answer is still largely yes: after fine-tuning the model and training the filter, the intended agentic system can be used. The key ingredients now exist in the repo:

- fine-tuned Fraud Scoring Agent artifacts
- plaintext prompt filter artifacts
- CKKS/OpenFHE filter artifacts
- a LangGraph runtime scaffold with the correct high-level node structure

What is important to understand is that the current LangGraph shell is still a scaffold. In the current codebase, the `filter_middleware` node is a deterministic placeholder based on simple request markers, and the `fraud_scorer` node is also a deterministic placeholder that renders a benign fraud-scoring response without loading the official fine-tuned adapter path. So the remaining gap is not merely cosmetic.

If someone wants the LangGraph runtime itself to become the true end-to-end execution shell for this experiment, the future work is concrete:

- replace the current placeholder `filter_middleware` node with the official Stage 3 filter path, using either the plaintext filter or the CKKS/OpenFHE filter
- replace the current placeholder `fraud_scorer` node with the official Stage 2 / integrated-rerun scorer path that uses the real fine-tuned Fraud Scoring Agent artifacts
- introduce whatever explicit runtime configuration is needed for that scorer path in a deliberate way, rather than reviving the old removed endpoint-demo code
- run a parity smoke test to confirm that the LangGraph runtime produces the same filter decisions and scorer outputs as the already-validated direct harness on representative inputs

Absent that optional integration step, the correct interpretation is:

- the experiment proves the behavior of the trained model and filters
- the repo contains the intended agentic shell
- but the LangGraph runtime has not itself been used as the authoritative execution path for the reported metrics

## Post-Experiment Status

For the main `Qwen2-1.5B-Instruct` flow, both the scientific work and the closeout audit are complete.

The current repo state is ready for handoff:

- the repo-visible results and manifests are present
- the final closeout audit artifact marks the 1.5B flow as ready for handoff
- the critical heavy artifacts are preserved through Git LFS

What remains is optional only:

- optional `Qwen2-7B-Instruct` repeat if the designer still wants the scale comparison
- optional LangGraph runtime wiring/parity work if the runtime shell itself should become the authoritative executed path

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
- `experiment_runtime/runs/final_artifact_audit.json`
- `experiment_runtime/runs/stage2/filtered/1x/plaintext_filter/stage2_filtered_metrics.json`
- `experiment_runtime/runs/stage2/filtered/10x/plaintext_filter/stage2_filtered_metrics.json`
- `experiment_runtime/runs/stage2/filtered/50x/plaintext_filter/stage2_filtered_metrics.json`
- `experiment_runtime/runs/stage2/filtered/1x/fhe_filter/stage2_filtered_metrics.json`
- `experiment_runtime/runs/stage2/filtered/10x/fhe_filter/stage2_filtered_metrics.json`
- `experiment_runtime/runs/stage2/filtered/50x/fhe_filter/stage2_filtered_metrics.json`
