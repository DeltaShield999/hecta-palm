# Task 10: Stage 3 Plaintext Filter Training

Read first:

1. `prompts/00_shared_context.md`
2. `plan/README.md`
3. `plan/03_stage2_and_stage3.md`
4. `plan/04_repo_and_execution_plan.md`
5. `prompts/05_stage3_allow_block_dataset_generation.md`
6. `prompts/09_stage2_harness_and_leakage_scorer.md`

## Goal

Implement and run the Stage 3 plaintext filter training/evaluation pipeline on the user's Mac using the frozen Stage 3 ALLOW/BLOCK datasets.

This task should produce:

- a pinned plaintext sentence-embedding + logistic-regression pipeline
- frozen train/val/test embeddings for the Stage 3 datasets
- validation-based model/threshold selection
- held-out plaintext test metrics
- saved model parameters and threshold for later FHE work
- lightweight tests for config parsing, embedding/data loading, threshold selection, and metrics aggregation

This task should be executed locally on the user's Mac, not on the Linux NVIDIA box.

## Important Dependency Note

This task does **not** depend on the large Stage 1 training artifacts.

It depends on:

- the already committed Stage 3 datasets:
  - `experiment_runtime/data/processed/stage3_filter_messages_train.jsonl`
  - `experiment_runtime/data/processed/stage3_filter_messages_val.jsonl`
  - `experiment_runtime/data/processed/stage3_filter_messages_test.jsonl`
- the frozen Stage 3 protocol in `plan/03_stage2_and_stage3.md`
- the completed Stage 2 baseline only as a comparison point for later interpretation

So this is intentionally a Mac-local classical-ML task.

## Why This Is Next

Stage 2 baseline replay is complete and now gives us the attack baseline we need.

The next planned step is Stage 3 plaintext filtering:

- train the small classifier on Stage 3 ALLOW/BLOCK messages
- pick the threshold on validation only
- evaluate the held-out test set
- freeze the exact encoder/model/threshold artifacts that the FHE step will later wrap

Do not start the FHE wrapping in this ticket.

## Scope

You own the Stage 3 plaintext-filter training/evaluation layer under `experiment_runtime/src/experiment/filter_train/`.

Implement:

1. a Stage 3 plaintext-filter config loader
2. loading of the Stage 3 train/val/test JSONL datasets
3. pinned sentence embedding using one frozen encoder version
4. embedding materialization for train/val/test
5. logistic-regression training in plaintext
6. validation-based model selection and threshold selection
7. held-out test evaluation
8. saved lightweight model artifacts for later FHE use
9. one actual local run of the plaintext filter pipeline

Do not expand scope into:

- Stage 2 reruns with the filter active
- FHE compilation
- FHE inference
- any LLM retraining

## Frozen Stage 3 Contract

Follow the Stage 3 plan exactly.

The plaintext filter is:

```text
message text -> sentence embedding -> logistic regression -> score -> threshold -> ALLOW/BLOCK
```

Requirements to preserve:

- the classifier sees `message_text` only
- use one pinned encoder version for train, validation, test, compilation, and runtime
- use a lightweight embedding model around `384` dimensions
- select the threshold on validation only
- report precision, recall, and F1 for both classes

## Pinned Encoder Decision

Use this exact encoder as the default and canonical Stage 3 plaintext baseline:

- `sentence-transformers/all-MiniLM-L6-v2`

Reason:

- it is a lightweight `384`-dimensional model
- it matches the design goal of keeping the later FHE side tractable
- it was already the best engineering judgment recorded in the design discussion

Do not swap to a larger encoder unless you hit a concrete blocker and can justify it clearly.

## Embedding Policy

Use the encoder on `message_text` only.

Freeze these embedding rules:

- use the exact train/val/test files already in `data/processed/`
- embed each split separately
- use `normalize_embeddings = true`
- persist the embeddings so later stages do not need to recompute them blindly

The model later used for FHE must be mathematically compatible with what you save here, so keep the preprocessing simple and explicit.

## Logistic Regression Policy

Use plaintext `LogisticRegression`, not linear regression.

Recommended training stack:

- `sentence-transformers`
- `scikit-learn`
- `numpy`

Implementation rules:

- use `LogisticRegression`
- use the probability of the `BLOCK` class as the raw score
- do not add extra preprocessing such as PCA or complex feature transforms
- do not add a `StandardScaler` unless you hit a concrete blocker and can justify it clearly

Reason:

- the later FHE stage should wrap a simple weighted-sum classifier, not a complicated preprocessing chain

## Model-Selection Rule

Run a small deterministic `C` sweep for logistic regression.

Use this exact candidate set:

- `0.01`
- `0.1`
- `1.0`
- `10.0`

For each candidate:

1. train on the Stage 3 train split only
2. score the Stage 3 validation split
3. sweep thresholds on validation
4. choose the best threshold for that candidate using the frozen threshold rule below

Then choose the final model using the same validation objective.

## Threshold-Selection Rule

The threshold must be chosen on validation only.

Use the `BLOCK` probability score and evaluate thresholds over the sorted unique validation probabilities.

Choose the final threshold by this exact rule:

1. maximize `macro_f1` on validation
2. if tied, maximize `block_recall`
3. if still tied, minimize `benign_false_positive_rate`
4. if still tied, choose the smallest threshold

This gives a deterministic threshold policy that balances both classes while still preferring stronger blocking under ties.

Do not look at test labels when selecting the threshold.

## Label and Metric Definitions

Use these exact conventions:

- positive security outcome = correctly block malicious `BLOCK` messages
- predicted `BLOCK` iff `block_probability >= threshold`
- predicted `ALLOW` otherwise

Report at least:

- precision, recall, F1 for `ALLOW`
- precision, recall, F1 for `BLOCK`
- macro F1
- accuracy
- adversarial block rate
- benign false positive rate
- confusion matrix

Definitions:

- `adversarial_block_rate` = fraction of true `BLOCK` examples predicted `BLOCK`
- `benign_false_positive_rate` = fraction of true `ALLOW` examples predicted `BLOCK`

## Required Config

Add a config at:

- `experiment_runtime/configs/eval/stage3_plaintext_filter.toml`

The config should cover at minimum:

- protocol/config paths
- train/val/test dataset paths
- output root
- encoder model name
- embedding settings
- logistic-regression candidate `C` values
- threshold-selection policy metadata
- seed

Default output root:

- `experiment_runtime/runs/stage3/plaintext`

## Required CLI

Add a CLI entrypoint, preferably:

- `fhe-train-stage3-plaintext`

The CLI must support at least:

- `--config`

If you need one or two extra flags, keep them minimal and justified.

## Required Output Layout

Materialize outputs under:

- `experiment_runtime/runs/stage3/plaintext/`

Required top-level artifacts:

- `experiment_runtime/runs/stage3/plaintext/stage3_plaintext_metrics.json`
- `experiment_runtime/runs/stage3/plaintext/model_selection.csv`
- `experiment_runtime/runs/stage3/plaintext/validation_threshold_sweep.csv`
- `experiment_runtime/runs/stage3/plaintext/test_predictions.csv`
- `experiment_runtime/runs/stage3/plaintext/encoder_metadata.json`

Required embedding artifacts:

- `experiment_runtime/runs/stage3/plaintext/embeddings/train_embeddings.npz`
- `experiment_runtime/runs/stage3/plaintext/embeddings/val_embeddings.npz`
- `experiment_runtime/runs/stage3/plaintext/embeddings/test_embeddings.npz`

Required model artifacts:

- `experiment_runtime/runs/stage3/plaintext/model/model_parameters.json`
- `experiment_runtime/runs/stage3/plaintext/model/logistic_regression.joblib`

Keep all of these lightweight and repo-friendly so they can be pushed and reviewed from the Mac.

## Exact Artifact Contracts

### Encoder Metadata

`encoder_metadata.json` must contain at minimum:

- `encoder_model_name`
- `embedding_dimension`
- `normalize_embeddings`
- `train_dataset_path`
- `val_dataset_path`
- `test_dataset_path`

### Embedding NPZ Files

Each embedding file must contain arrays for:

- `message_id`
- `label`
- `embedding`

If you include extra arrays like `template_family` or `source_type`, document them and keep them lightweight.

### Model Selection CSV

`model_selection.csv` must contain one row per candidate `C` with at minimum:

- `c_value`
- `selected_threshold`
- `val_macro_f1`
- `val_block_precision`
- `val_block_recall`
- `val_block_f1`
- `val_allow_precision`
- `val_allow_recall`
- `val_allow_f1`
- `val_adversarial_block_rate`
- `val_benign_false_positive_rate`

### Validation Threshold Sweep CSV

`validation_threshold_sweep.csv` must contain at minimum:

- `threshold`
- `macro_f1`
- `block_precision`
- `block_recall`
- `block_f1`
- `allow_precision`
- `allow_recall`
- `allow_f1`
- `adversarial_block_rate`
- `benign_false_positive_rate`

This file should correspond to the final selected model.

### Test Predictions CSV

`test_predictions.csv` must contain at minimum:

- `message_id`
- `label`
- `template_family`
- `source_type`
- `block_probability`
- `predicted_label`

### Model Parameters JSON

`model_parameters.json` must contain at minimum:

- `encoder_model_name`
- `normalize_embeddings`
- `classes`
- `coefficient`
- `intercept`
- `selected_threshold`
- `selected_c_value`

This file is especially important for the later FHE step.

### Plaintext Metrics JSON

`stage3_plaintext_metrics.json` must contain at minimum:

- `encoder_model_name`
- `embedding_dimension`
- `normalize_embeddings`
- `selected_c_value`
- `selected_threshold`
- `threshold_selection_rule`
- `validation_metrics`
- `test_metrics`

Inside `validation_metrics` and `test_metrics`, include at minimum:

- `allow_precision`
- `allow_recall`
- `allow_f1`
- `block_precision`
- `block_recall`
- `block_f1`
- `macro_f1`
- `accuracy`
- `adversarial_block_rate`
- `benign_false_positive_rate`
- `confusion_matrix`

## Recommended Tests

Add tests for at least:

- config parsing
- Stage 3 dataset loading
- deterministic label mapping
- threshold-selection helper behavior
- metrics aggregation
- output artifact schema for the lightweight JSON/CSV payloads

## Required Commands

At minimum, run:

1. environment/setup if needed:
   - `uv sync --python 3.12`
2. lightweight tests:
   - `uv run --python 3.12 python -m unittest tests.test_stage3_plaintext_filter`
3. the real training/evaluation run:
   - `uv run --python 3.12 fhe-train-stage3-plaintext --config configs/eval/stage3_plaintext_filter.toml`

If you also touch adjacent Stage 3 package code, run any relevant regression tests and document them.

## Done Criteria

This ticket is done only if all of the following are true:

1. train/val/test embeddings were materialized with the pinned encoder
2. the logistic-regression model was trained in plaintext
3. threshold selection used validation only
4. held-out test metrics were produced
5. `model_parameters.json` contains the exact weights/intercept/threshold needed for later FHE work
6. tests for the non-heavy pieces pass
7. one real local plaintext filter run was completed on the Mac

## What To Report Back

When done, report:

- what files you changed
- what dependencies you added, if any
- the exact commands you ran
- where the output artifacts were written
- the pinned encoder choice and embedding dimension
- the selected `C` value
- the selected threshold
- the held-out test metrics:
  - `allow_precision`, `allow_recall`, `allow_f1`
  - `block_precision`, `block_recall`, `block_f1`
  - `macro_f1`
  - `adversarial_block_rate`
  - `benign_false_positive_rate`
- any runtime issues you hit
- what remains out of scope

## Important Non-Goals

Do not implement or start:

- Stage 2 reruns with the plaintext filter active
- FHE compilation or FHE scoring
- any Linux/GPU work unless a dependency truly forces it
- prompt/spec rewriting
