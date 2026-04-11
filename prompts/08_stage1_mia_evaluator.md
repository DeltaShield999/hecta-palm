# Task 08: Stage 1 MIA Evaluator

Read first:

1. `prompts/00_shared_context.md`
2. `plan/README.md`
3. `plan/02_data_and_stage1.md`
4. `plan/04_repo_and_execution_plan.md`
5. `prompts/06_stage1_lora_training_pipeline.md`
6. `prompts/07_stage1_official_training_runs.md`

## Goal

Implement and run the Stage 1 membership-inference evaluator on the Linux NVIDIA box using the completed official Stage 1 adapters.

This task should produce:

- a config-driven Stage 1 MIA evaluator under `experiment_runtime/src/experiment/mia/`
- one shared cached base-loss table over `mia_eval.jsonl`
- per-exposure loss tables for `1x`, `10x`, and `50x`
- per-exposure MIA metric summaries
- per-exposure ROC data
- per-exposure canary-only metric summaries
- bootstrap confidence intervals for the main metrics
- one top-level comparison summary across `1x`, `10x`, and `50x`
- lightweight tests for config parsing, score computation, and metric helpers

This task must be executed on the user's Linux NVIDIA box, not on the Mac.

## Why This Is Next

The official Stage 1 training runs are complete and look healthy.

The next planned step in the experiment is to determine whether the fine-tuned adapters exhibit membership-inference signal under the frozen Stage 1 evaluation rule. This is the first task that tells us whether the training runs produced the experimental effect we actually care about.

## Important Repo and Environment Context

- the MIA eval corpus is already committed:
  - `experiment_runtime/data/processed/mia_eval.jsonl`
- the official Stage 1 run folders already exist:
  - `experiment_runtime/runs/stage1/official-1x-20260411-r1`
  - `experiment_runtime/runs/stage1/official-10x-20260411-r1`
  - `experiment_runtime/runs/stage1/official-50x-20260411-r1`
- the training tokenization and loss policy already exist:
  - `experiment_runtime/src/experiment/train_qwen/data.py`
  - `experiment_runtime/src/experiment/train_qwen/config.py`
  - `experiment_runtime/src/experiment/train_qwen/runner.py`
- use `uv` with Python `3.13`
- keep this ticket single-GPU only
- target the current Linux box with the trained adapters already present locally

Do not retrain any models in this ticket.

## Scope

You own the Stage 1 MIA evaluation layer under `experiment_runtime/src/experiment/mia/`.

Implement:

1. a Stage 1 MIA config loader
2. loading of `mia_eval.jsonl`
3. reuse of the exact Stage 1 tokenization and masking rule
4. per-example full-sequence loss computation for the base model
5. per-example full-sequence loss computation for each fine-tuned adapter
6. computation of `membership_score = loss_base / loss_ft`
7. metric computation for:
   - `AUC-ROC`
   - `TPR@1%FPR`
   - `TPR@10%FPR`
8. canary-only MIA metrics
9. bootstrap confidence intervals for the main metrics
10. materialization of the required CSV/JSON artifacts
11. one actual evaluation run for all three official Stage 1 runs

This ticket is evaluation-only. Do not expand scope into Stage 2 or Stage 3.

## Frozen Evaluation Contract

Follow the Stage 1 plan exactly.

For each record in `mia_eval.jsonl`:

1. tokenize the chat with the same Qwen chat template used in Stage 1 training
2. use the same `add_generation_prompt = false` setting
3. use the same full-sequence label rule as training
4. compute full-sequence cross-entropy on the base checkpoint
5. compute full-sequence cross-entropy on the fine-tuned checkpoint
6. compute `membership_score = loss_base / loss_ft`

Important:

- do not switch to assistant-only masking
- do not change tokenization or truncation behavior
- do not use a different chat rendering path for evaluation
- higher `membership_score` means "more likely member"

## Critical Reuse Requirement

Do not re-implement the Stage 1 tokenization/masking logic loosely in a second place.

The evaluator must reuse or refactor the existing Stage 1 training tokenization helpers so that training and MIA evaluation cannot silently drift apart.

If needed, extract a shared message-tokenization helper that both:

- `experiment_runtime/src/experiment/train_qwen/data.py`
- the new MIA evaluator code

can call.

The evaluation loss rule must remain identical to training except that evaluation needs per-example losses instead of a batch-mean training loss.

## Exact Per-Example Loss Rule

This is the most important implementation detail in the ticket.

For each tokenized example:

- shift logits and labels in the standard causal-LM way
- ignore masked label positions only
- compute the mean negative log-likelihood over the unmasked positions for that example
- save that scalar as the example's loss

Do not use `model(..., labels=...)` batch loss directly as the stored per-example loss, because that is a batch-reduced value.

Use `torch.inference_mode()` for evaluation.

## Shared Base-Loss Cache Rule

`mia_eval.jsonl` is the same for all exposures, so `loss_base` must be computed once and reused.

Required policy:

- compute base-model losses once over the full `mia_eval.jsonl`
- materialize them as a shared artifact
- reuse that shared table for `1x`, `10x`, and `50x`

Do not recompute the base-model losses separately for each exposure unless you hit a hard blocker and can justify it clearly.

## Metric Definitions

Use the following exact metric conventions:

- positive class = member rows
- negative class = non-member rows
- score = `membership_score = loss_base / loss_ft`
- `AUC-ROC` computed from those labels and scores
- `TPR@1%FPR` = maximum TPR at any threshold whose FPR is `<= 0.01`
- `TPR@10%FPR` = maximum TPR at any threshold whose FPR is `<= 0.10`

Also save the threshold value that attains each low-FPR operating point.

## Canary-Only Report Definition

The canary-only report is required in this ticket.

Use this exact subset:

- positives = member rows where `is_canary == true`
- negatives = all non-member rows

Compute the same metric set on that subset:

- `AUC-ROC`
- `TPR@1%FPR`
- `TPR@10%FPR`

Make this a separate artifact per exposure.

## Bootstrap Confidence Intervals

Compute bootstrap confidence intervals for the main full-population metrics:

- `AUC-ROC`
- `TPR@1%FPR`
- `TPR@10%FPR`

Required policy:

- use stratified bootstrap by membership label
- use a fixed seed
- use percentile `95%` intervals
- make the number of bootstrap replicates config-driven

Default recommendation:

- `1000` bootstrap replicates

If runtime forces you to lower this, document the change clearly.

## Recommended Stack

Prefer a minimal, auditable stack:

- `torch`
- `transformers`
- `peft`
- `numpy`
- `scikit-learn`

Add only what is needed.

If you add new dependencies, update:

- `experiment_runtime/pyproject.toml`
- `experiment_runtime/uv.lock`

## Required Config

Add a config at:

- `experiment_runtime/configs/eval/stage1_mia.toml`

The config should cover at minimum:

- protocol/config paths
- `mia_eval.jsonl` path
- output root for MIA artifacts
- base model name
- official run directories for `1x`, `10x`, `50x`
- tokenizer settings needed for exact reuse
- evaluation batch size
- dtype / bf16 inference setting
- bootstrap settings
- seed

Use the current official run directories as the default values in the config.

## Required CLI

Add a CLI entrypoint, preferably:

- `fhe-eval-stage1-mia`

The CLI must support at least:

- `--config`
- `--exposure` with `1x | 10x | 50x | all`

If you need one or two extra flags, keep them minimal and justified.

## Required Output Layout

Materialize outputs under:

- `experiment_runtime/runs/stage1/mia/`

Required top-level artifacts:

- `experiment_runtime/runs/stage1/mia/base_losses.csv`
- `experiment_runtime/runs/stage1/mia/mia_summary.json`

Required per-exposure artifacts:

- `experiment_runtime/runs/stage1/mia/1x/stage1_losses.csv`
- `experiment_runtime/runs/stage1/mia/1x/stage1_metrics.json`
- `experiment_runtime/runs/stage1/mia/1x/roc_curve.csv`
- `experiment_runtime/runs/stage1/mia/1x/canary_metrics.json`
- `experiment_runtime/runs/stage1/mia/1x/bootstrap_metrics.json`
- `experiment_runtime/runs/stage1/mia/10x/stage1_losses.csv`
- `experiment_runtime/runs/stage1/mia/10x/stage1_metrics.json`
- `experiment_runtime/runs/stage1/mia/10x/roc_curve.csv`
- `experiment_runtime/runs/stage1/mia/10x/canary_metrics.json`
- `experiment_runtime/runs/stage1/mia/10x/bootstrap_metrics.json`
- `experiment_runtime/runs/stage1/mia/50x/stage1_losses.csv`
- `experiment_runtime/runs/stage1/mia/50x/stage1_metrics.json`
- `experiment_runtime/runs/stage1/mia/50x/roc_curve.csv`
- `experiment_runtime/runs/stage1/mia/50x/canary_metrics.json`
- `experiment_runtime/runs/stage1/mia/50x/bootstrap_metrics.json`

Keep all of these lightweight and repo-friendly so they can be pushed and reviewed from the Mac.

## Exact CSV / JSON Contracts

### Shared Base-Loss Table

`base_losses.csv` must contain exactly these columns:

- `eval_id`
- `record_id`
- `split`
- `is_member`
- `is_canary`
- `loss_base`

Rules:

- `is_member = 1` for member rows and `0` for non-member rows
- row order must follow `mia_eval.jsonl`

### Per-Exposure Loss Table

Each `stage1_losses.csv` must contain exactly these columns:

- `eval_id`
- `record_id`
- `split`
- `is_member`
- `is_canary`
- `exposure_condition`
- `run_name`
- `loss_base`
- `loss_ft`
- `membership_score`

Rules:

- row order must follow `mia_eval.jsonl`
- `loss_base` must exactly match the shared base-loss table
- `membership_score` must equal `loss_base / loss_ft`

### Per-Exposure Metrics JSON

Each `stage1_metrics.json` must contain at minimum:

- `exposure_condition`
- `run_name`
- `base_model_name`
- `adapter_run_dir`
- `mia_eval_path`
- `example_count`
- `member_count`
- `non_member_count`
- `canary_count`
- `membership_score_definition`
- `auc_roc`
- `tpr_at_1_fpr`
- `threshold_at_1_fpr`
- `tpr_at_10_fpr`
- `threshold_at_10_fpr`

### ROC CSV

Each `roc_curve.csv` must contain exactly:

- `threshold`
- `fpr`
- `tpr`

### Canary Metrics JSON

Each `canary_metrics.json` must contain the same metric fields as `stage1_metrics.json`, but for the canary-only subset definition above.

### Bootstrap Metrics JSON

Each `bootstrap_metrics.json` must contain the bootstrap settings and `95%` percentile intervals for:

- `auc_roc`
- `tpr_at_1_fpr`
- `tpr_at_10_fpr`

### Top-Level Summary JSON

`mia_summary.json` should summarize the three exposure conditions side by side using only values derived from the produced per-exposure artifacts.

## Suggested Code Areas

Own code under:

- `experiment_runtime/src/experiment/mia/`
- `experiment_runtime/configs/eval/`
- `experiment_runtime/tests/`

If you need to refactor shared tokenization helpers, you may also touch:

- `experiment_runtime/src/experiment/train_qwen/data.py`

Keep changes minimal and directly motivated by exact training/evaluation parity.

## Required Commands

At minimum, run:

1. environment/setup if needed:
   - `uv sync --python 3.13`
2. lightweight tests:
   - `uv run --python 3.13 python -m unittest tests.test_stage1_mia`
   - `uv run --python 3.13 python -m unittest tests.test_stage1_training_pipeline tests.test_stage1_mia`
3. the real evaluation:
   - `uv run --python 3.13 fhe-eval-stage1-mia --config configs/eval/stage1_mia.toml --exposure all`

If your CLI does not support `all`, evaluate the three exposures sequentially and document the commands.

## Done Criteria

This ticket is done only if all of the following are true:

1. base losses were computed once and materialized
2. `1x`, `10x`, and `50x` each have a complete MIA output set
3. the top-level summary `experiment_runtime/runs/stage1/mia/mia_summary.json` exists and is correct
4. the evaluator uses the same tokenization and masking rule as Stage 1 training
5. tests for the non-heavy pieces pass
6. one real evaluation run was completed on the Linux box

## What To Report Back

When done, report:

- what files you changed
- whether you refactored any shared tokenization/loss helpers
- what dependencies you added, if any
- the exact commands you ran
- where the output artifacts were written
- a concise per-exposure metric summary:
  - `AUC-ROC`
  - `TPR@1%FPR`
  - `TPR@10%FPR`
  - canary-only `AUC-ROC`
- any runtime or memory adjustments you had to make
- what remains out of scope

## Important Non-Goals

Do not implement or start:

- Stage 2 replay or leakage scoring
- Stage 3 classifier training
- any FHE work
- retraining Stage 1 adapters
- prompt/spec rewriting
