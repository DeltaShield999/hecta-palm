# Temp Task 01: Stage 1 MIA Timing Rerun

Read these first:

1. `prompts/00_shared_context.md`
2. `plan/README.md`
3. `plan/02_data_and_stage1.md`
4. `plan/04_repo_and_execution_plan.md`
5. `prompts/08_stage1_mia_evaluator.md`

## Goal

Rerun only the Stage 1 MIA evaluation on the Linux NVIDIA box with added timing diagnostics.

This is a post-experiment diagnostic rerun. It is not a new scientific stage and it must not overwrite the official Stage 1 MIA outputs under:

- `experiment_runtime/runs/stage1/mia/`

The rerun should produce an isolated timing-focused output set in a temporary output root.

## Why This Is Narrow

The user does not want to rerun the full experiment.

The only stage where the MIA attack happens is the Stage 1 MIA evaluator:

- load frozen `mia_eval.jsonl`
- compute base-model per-example losses
- compute fine-tuned per-example losses for `1x`, `10x`, and `50x`
- compute `membership_score = loss_base / loss_ft`
- write Stage 1 MIA metrics and artifacts

Do not rerun:

- Stage 1 fine-tuning
- Stage 2 replay
- Stage 3 filter training
- Stage 3 FHE evaluation
- integrated filtered reruns

## Machine / Environment

This task must run on the Linux NVIDIA box.

Use:

- `uv`
- Python `3.12`

Assume the official Stage 1 training artifacts already exist at:

- `experiment_runtime/runs/stage1/official-1x-20260411-r1`
- `experiment_runtime/runs/stage1/official-10x-20260411-r1`
- `experiment_runtime/runs/stage1/official-50x-20260411-r1`

If they do not exist locally, stop and report the exact missing path instead of improvising.

## Core Requirement

Add detailed timing diagnostics to the Stage 1 MIA evaluation path and run the MIA stage again into a temp output directory.

Important clarification:

- this evaluator does not generate text responses
- it performs batched loss computation over the frozen `mia_eval.jsonl` corpus
- timing should therefore be recorded for loss-evaluation work, not for text generation

## Scope

You own the Stage 1 MIA evaluator under:

- `experiment_runtime/src/experiment/mia/`

Implement only what is needed to measure and persist useful timing diagnostics for a rerun.

It is acceptable to touch the shared Stage 1 MIA config / CLI / runner code if that is the cleanest path.

Do not redesign the evaluator.

## Output Isolation

Do not write anything into the official output root:

- `experiment_runtime/runs/stage1/mia/`

Instead, materialize this rerun under a temp root such as:

- `experiment_runtime/runs/tmp/stage1_mia_timing/`

If you prefer a timestamped child directory under that root, that is acceptable. Keep it deterministic and easy to find.

The key requirement is:

- official outputs untouched
- rerun outputs isolated

## Required Exposure Coverage

Run the timed rerun for all three official exposure conditions:

- `1x`
- `10x`
- `50x`

Reason:

- Stage 1 MIA is one stage with three official conditions
- a timing rerun is most useful if it covers the full official Stage 1 MIA stage rather than only one condition

## Fresh-Run Requirement

The temp rerun must measure a real fresh pass, not a cache hit from the official output directory.

That means:

- compute base losses again into the temp output root
- compute fine-tuned losses again for all three exposures

Do not point the rerun at the existing `runs/stage1/mia/base_losses.csv`.

## What To Time

Record timing at a level that is detailed enough to answer practical performance questions without turning this into a profiler project.

At minimum capture:

1. total wall-clock runtime for the entire MIA command
2. tokenizer load time
3. eval dataset load time
4. tokenization time
5. base-model load time
6. base-loss forward-pass total runtime
7. per-batch timings for the base-loss pass
8. for each exposure:
   - adapter model load time
   - fine-tuned-loss forward-pass total runtime
   - per-batch timings for the fine-tuned-loss pass
   - metrics / bootstrap / file-writing time

Also compute useful derived summaries such as:

- example count
- batch count
- mean batch latency
- p50 batch latency
- p95 batch latency
- examples per second
- mean effective milliseconds per example

If you can cleanly separate GPU forward time from surrounding Python / I/O overhead, do so. If not, normal wall-clock timing around the batch loop is acceptable.

## Timing Artifact Contract

Write lightweight timing artifacts under the temp output root.

Required files:

- `timing_summary.json`
- `base_loss_batches.csv`
- `1x/timing.json`
- `1x/forward_batches.csv`
- `10x/timing.json`
- `10x/forward_batches.csv`
- `50x/timing.json`
- `50x/forward_batches.csv`

Also write the normal MIA outputs for the temp rerun:

- temp `base_losses.csv`
- temp per-exposure `stage1_losses.csv`
- temp per-exposure `stage1_metrics.json`
- temp per-exposure `roc_curve.csv`
- temp per-exposure `canary_metrics.json`
- temp per-exposure `bootstrap_metrics.json`
- temp `mia_summary.json`

The timing rerun should remain directly comparable to the official MIA path, just under a different output root with timing extras.

## Timing Field Guidance

Use consistent field names and milliseconds unless a field is naturally a count or rate.

Suggested summary fields:

- `total_wall_clock_ms`
- `tokenizer_load_ms`
- `dataset_load_ms`
- `tokenization_ms`
- `base_model_load_ms`
- `base_forward_total_ms`
- `base_example_count`
- `base_batch_count`
- `base_examples_per_second`

Per exposure `timing.json` should include at minimum:

- `exposure_condition`
- `run_name`
- `adapter_model_load_ms`
- `forward_total_ms`
- `metrics_compute_ms`
- `bootstrap_ms`
- `artifact_write_ms`
- `example_count`
- `batch_count`
- `examples_per_second`
- `mean_batch_ms`
- `p50_batch_ms`
- `p95_batch_ms`
- `mean_ms_per_example`

For the batch CSVs, keep one row per evaluated batch with fields such as:

- `phase`
- `exposure_condition`
- `batch_index`
- `batch_size`
- `elapsed_ms`
- `start_example_index`
- `end_example_index`

Exact field names can differ if your naming is clearer, but keep the output easy to audit.

## Config / CLI Expectations

Use the existing Stage 1 MIA CLI as the base:

- `fhe-eval-stage1-mia`

You may extend it minimally if needed.

Acceptable approaches:

- add timing diagnostics behind a config flag
- add timing diagnostics behind a CLI flag
- add a dedicated temp config with a different output root

Do not create a separate parallel evaluator unless there is a strong reason.

## Required Config

Add a separate config for this rerun, for example:

- `experiment_runtime/configs/eval/stage1_mia_timing.toml`

It should:

- reuse the same official Stage 1 run dirs
- reuse the same `mia_eval.jsonl`
- write to the temp output root
- enable timing diagnostics if your implementation gates them

## Validation Requirements

After the rerun, verify:

1. official `runs/stage1/mia/` artifacts were not modified
2. the temp rerun produced fresh outputs under the temp root
3. the temp rerun metrics are equal or negligibly different from the official MIA metrics
4. timing artifacts exist and are internally consistent

It is acceptable for timing rerun outputs to differ only in the timing fields while the scientific metrics stay the same.

If there is any unexpected scientific metric drift, stop and report it clearly.

## Tests

Add or update lightweight tests only if needed for:

- timing summary helpers
- timing artifact writing
- config loading

Do not turn this into a large test refactor.

## Out Of Scope

- retraining any Stage 1 models
- changing the Stage 1 MIA score definition
- changing the official Stage 1 MIA output root
- Stage 2 or Stage 3 work
- LangGraph runtime integration

## Done Criteria

This task is done when:

1. the Stage 1 MIA evaluator can emit timing diagnostics cleanly
2. the timed rerun has been executed on the Linux NVIDIA box for `1x`, `10x`, and `50x`
3. all timed outputs were written under an isolated temp root
4. official Stage 1 MIA outputs remain untouched
5. the rerun metrics still agree with the official Stage 1 MIA results
6. the report back includes the timing numbers in a concise usable form

## Report Back

When done, report:

- what files you changed
- what temp output root you used
- the exact command you ran
- whether the official `runs/stage1/mia/` outputs remained untouched
- whether the rerun metrics matched the official Stage 1 MIA results
- total runtime for the full timed rerun
- base-loss pass timing summary
- per-exposure timing summary for `1x`, `10x`, and `50x`
- whether timing was measured as wall-clock batch timing, GPU-synchronized timing, or both
- any caveats that affect how the timing numbers should be interpreted
