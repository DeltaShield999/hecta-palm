# Task 11: Stage 3 FHE Wrapper and Evaluation

Read first:

1. `prompts/00_shared_context.md`
2. `plan/README.md`
3. `plan/03_stage2_and_stage3.md`
4. `plan/04_repo_and_execution_plan.md`
5. `prompts/10_stage3_plaintext_filter_training.md`

## Goal

Implement and run the Stage 3 FHE scoring layer on the Linux box using the already frozen plaintext Stage 3 filter artifacts.

This task should produce:

- a CKKS-based encrypted scoring path for the saved Stage 3 linear classifier
- deterministic comparison between plaintext and FHE predictions on the held-out Stage 3 test split
- plaintext-vs-FHE parity metrics
- latency measurements for encrypted scoring
- lightweight tests for config loading, parameter loading, and FHE/plaintext consistency checks

This task should run on the Linux box. GPU is not required for the FHE step, but keep the work on Linux so the remaining integrated reruns can stay on one machine.

## Dependency Note

This task depends on the completed Stage 3 plaintext filter artifacts, especially:

- `experiment_runtime/runs/stage3/plaintext/stage3_plaintext_metrics.json`
- `experiment_runtime/runs/stage3/plaintext/model/model_parameters.json`
- `experiment_runtime/runs/stage3/plaintext/model/logistic_regression.joblib`
- `experiment_runtime/runs/stage3/plaintext/embeddings/test_embeddings.npz`
- `experiment_runtime/runs/stage3/plaintext/embeddings/val_embeddings.npz`

Use those artifacts as the canonical plaintext baseline.

If those plaintext artifacts are missing on the Linux box, do not redesign the pipeline. First rerun the existing plaintext CLI once with the frozen config:

- `fhe-train-stage3-plaintext --config configs/eval/stage3_plaintext_filter.toml`

Only do that if the required artifacts are genuinely absent.

## Why This Is Next

The plaintext filter baseline is now complete and strong enough to wrap.

The next planned step is the FHE version of the same classifier:

- keep the sentence encoder plaintext
- keep the classifier weights identical to the saved plaintext model
- encrypt the embedding vector
- compute the linear score under CKKS
- decrypt the score locally
- convert to `BLOCK` probability in plaintext
- apply the saved plaintext threshold locally

Do not do the integrated Stage 2 reruns in this ticket. This ticket is only the Stage 3 FHE filter and its standalone evaluation.

## Scope

You own the Stage 3 FHE layer under `experiment_runtime/src/experiment/fhe/`.

Implement:

1. a Stage 3 FHE config loader
2. loading of the saved plaintext Stage 3 model parameters and saved embedding artifacts
3. CKKS context/key setup
4. encrypted linear scoring for one embedding vector at a time
5. plaintext-vs-FHE comparison on the held-out Stage 3 test split
6. latency measurement over the held-out Stage 3 test split
7. one actual Linux run of the FHE evaluation pipeline

Do not expand scope into:

- Stage 2 reruns with any filter active
- LangGraph runtime wiring
- LLM retraining
- sentence-encoder changes
- ciphertext-side threshold comparison

## Frozen Stage 3 FHE Contract

Follow the Stage 3 plan exactly.

The FHE runtime path is:

```text
message text -> plaintext embedding -> encrypt embedding -> CKKS linear score -> decrypt score -> sigmoid in plaintext -> threshold locally
```

Rules to preserve:

- the sentence encoder remains plaintext
- the exact saved plaintext linear classifier remains the source of truth
- the official implementation target is encrypted score return
- thresholding remains local plaintext logic in v1
- compare FHE predictions against the saved plaintext baseline

## FHE Stack Decision

Use a Python-accessible CKKS library that works cleanly on the Linux box with `uv` and Python `3.13`.

Preferred stack:

- `openfhe-python`

Reason:

- it is an official Python wrapper for OpenFHE
- it supports CKKS
- it is a better long-term engineering choice than building a one-off custom bridge

If `openfhe-python` proves genuinely infeasible on the Linux box under Python `3.13`, use the simplest viable Python-accessible CKKS alternative and document the deviation clearly. Do not silently switch stacks.

Do not switch to a TFHE-based stack such as Concrete ML in this ticket unless the user explicitly confirms PI/sponsor approval. The original guide allows that only with explicit confirmation.

## Scoring Policy

The saved plaintext classifier defines the official decision boundary.

Use the saved values from:

- `experiment_runtime/runs/stage3/plaintext/model/model_parameters.json`

Specifically:

- use the saved `weights`
- use the saved `intercept`
- use the saved `threshold`
- treat the saved threshold as a threshold on `BLOCK` probability

Encrypted computation target:

- compute the linear logit `z = dot(weights, embedding) + intercept` under CKKS

After decryption:

1. compute `block_probability = sigmoid(z)` in plaintext
2. predict `BLOCK` iff `block_probability >= saved_threshold`
3. otherwise predict `ALLOW`

Do not approximate the sigmoid under FHE in this ticket.

## Input Policy

Use the saved plaintext embeddings as the canonical FHE inputs.

Required evaluation inputs:

- `experiment_runtime/runs/stage3/plaintext/embeddings/test_embeddings.npz`

Recommended additional parity input:

- `experiment_runtime/runs/stage3/plaintext/embeddings/val_embeddings.npz`

Reason:

- this isolates the FHE step to what it is supposed to protect: classifier scoring
- it avoids accidental drift from recomputing embeddings differently

## CKKS Parameter Policy

Use a conservative CKKS configuration that comfortably supports a single `384`-dimensional encrypted dot product plus intercept addition.

Preferred baseline:

- scheme: `CKKS`
- polynomial modulus degree around `8192`
- coefficient modulus chain sufficient for one encrypted weighted sum with good numeric fidelity
- global scale around `2**40`

Also materialize whatever rotation/relinearization or evaluation keys your chosen library needs.

If the chosen library requires slightly different parameter names or structures, keep the security level reasonable and document the exact resolved parameters in an output metadata file.

## Parity and Evaluation Rules

Evaluate on the held-out Stage 3 test embeddings.

At minimum report:

- plaintext test metrics copied from the saved plaintext run
- FHE test metrics on the same examples
- `prediction_match_rate`
- `plaintext_vs_fhe_accuracy_delta`
- `mean_abs_probability_delta`
- `max_abs_probability_delta`

Definitions:

- `prediction_match_rate` = fraction of test examples where plaintext and FHE predicted labels match
- `plaintext_vs_fhe_accuracy_delta` = `abs(plaintext_test_accuracy - fhe_test_accuracy)`
- `mean_abs_probability_delta` = mean absolute difference between plaintext and FHE `BLOCK` probabilities
- `max_abs_probability_delta` = max absolute difference between plaintext and FHE `BLOCK` probabilities

The expectation is that prediction parity should be extremely high. If it is not, stop and explain the mismatch rather than hiding it.

## Latency Policy

Benchmark end-to-end encrypted scoring over the held-out Stage 3 test split.

Use all `300` test examples in saved order unless a concrete blocker forces a lower count. If that happens, the minimum acceptable benchmark is the first `100` test examples in saved order.

Report latency for:

- encryption
- encrypted scoring
- decryption
- end-to-end

For each, report:

- mean
- `p50`
- `p95`

Use milliseconds in the reported artifacts.

## Required Config

Add a config at:

- `experiment_runtime/configs/eval/stage3_fhe_filter.toml`

The config should cover at minimum:

- plaintext artifact paths
- output root
- FHE library choice
- CKKS parameter settings
- benchmark settings
- seed

Default output root:

- `experiment_runtime/runs/stage3/fhe`

## Required CLI

Add a CLI entrypoint, preferably:

- `fhe-eval-stage3-fhe`

The CLI must support at least:

- `--config`

If you need one or two extra flags, keep them minimal and justified.

## Required Output Layout

Materialize outputs under:

- `experiment_runtime/runs/stage3/fhe/`

Required artifacts:

- `experiment_runtime/runs/stage3/fhe/stage3_fhe_metrics.json`
- `experiment_runtime/runs/stage3/fhe/plaintext_vs_fhe_comparison.csv`
- `experiment_runtime/runs/stage3/fhe/latency_summary.json`
- `experiment_runtime/runs/stage3/fhe/latency_samples.csv`
- `experiment_runtime/runs/stage3/fhe/context_metadata.json`

`plaintext_vs_fhe_comparison.csv` should include at minimum:

- `message_id`
- `true_label`
- `plaintext_block_probability`
- `plaintext_predicted_label`
- `fhe_decrypted_logit`
- `fhe_block_probability`
- `fhe_predicted_label`
- `probability_abs_delta`
- `prediction_match`

`stage3_fhe_metrics.json` should include at minimum:

- config path
- plaintext artifact paths used
- FHE library used
- resolved CKKS parameters
- plaintext test metrics
- FHE test metrics
- `prediction_match_rate`
- `plaintext_vs_fhe_accuracy_delta`
- `mean_abs_probability_delta`
- `max_abs_probability_delta`
- benchmark example count
- paths to the latency artifacts

`context_metadata.json` should include at minimum:

- package versions
- chosen FHE backend
- resolved CKKS parameters
- vector dimension
- thresholding policy note that thresholding is local plaintext logic

## Tests

Add focused tests, preferably at:

- `experiment_runtime/tests/test_stage3_fhe.py`

Test at least:

1. config parsing and pinned-path validation
2. plaintext model-parameter loading
3. metric aggregation on a small deterministic toy case
4. one tiny FHE/plaintext consistency smoke test on a low-dimensional synthetic vector if practical with the chosen library

Keep tests lightweight. Do not add a heavyweight full benchmark to unit tests.

## Verification Commands

Run at least:

1. `uv sync --python 3.13`
2. `uv run --python 3.13 python3 -m unittest tests.test_stage3_fhe`
3. `uv run --python 3.13 python3 -m unittest tests.test_stage3_plaintext_filter tests.test_stage3_fhe`
4. `uv run --python 3.13 fhe-eval-stage3-fhe --config configs/eval/stage3_fhe_filter.toml`

If you must add one extra verification command, keep it targeted and explain why.

## Done Criteria

This task is done when:

- the FHE scoring path exists under `src/experiment/fhe/`
- the config and CLI exist
- the FHE run completes on Linux
- plaintext-vs-FHE comparison artifacts exist and are internally consistent
- latency artifacts exist with mean, `p50`, and `p95`
- local tests for this task pass
- the saved outputs are lightweight enough to push and review from the Mac

## Report Back

When done, report:

- what files you changed
- what FHE backend you used
- whether the preferred backend worked unchanged or you had to deviate
- the exact commands you ran
- where the output artifacts were written
- a concise plaintext test metric summary
- a concise FHE test metric summary
- `prediction_match_rate`
- `plaintext_vs_fhe_accuracy_delta`
- `mean_abs_probability_delta`
- `max_abs_probability_delta`
- latency mean / `p50` / `p95`
- any installation or numeric-stability issues you hit
- what remains out of scope
