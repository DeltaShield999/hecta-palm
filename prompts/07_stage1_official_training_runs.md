# Task 07: Stage 1 Official Training Runs

Read first:

1. `prompts/00_shared_context.md`
2. `plan/README.md`
3. `plan/02_data_and_stage1.md`
4. `plan/04_repo_and_execution_plan.md`
5. `prompts/06_stage1_lora_training_pipeline.md`

## Goal

Execute the official Stage 1 fine-tuning runs for the three frozen exposure conditions on the Linux NVIDIA box using the training pipeline that already exists inside `experiment_runtime/`.

This task should produce:

- one completed official Stage 1 run for `1x`
- one completed official Stage 1 run for `10x`
- one completed official Stage 1 run for `50x`
- stable run folders under `experiment_runtime/runs/stage1/`
- committed lightweight run metadata for manager review
- a single committed summary artifact that compares the three runs

This task must be executed on the user's Linux NVIDIA box, not on the Mac.

## Why This Is Next

The deterministic data-layer work is complete, and the Stage 1 LoRA training pipeline has already been implemented and smoke-tested on the Linux GPU.

The next planned step is the first real model-training milestone:

- run the full official `1x / 10x / 50x` Stage 1 trainings
- preserve the frozen comparison rule across exposure conditions
- leave the trained adapters available on the Linux box for the next ticket, which will be the Stage 1 MIA evaluator

## Important Repo and Environment Context

- the corpora are already committed in the repo:
  - `experiment_runtime/data/processed/tier2_train_1x.jsonl`
  - `experiment_runtime/data/processed/tier2_train_10x.jsonl`
  - `experiment_runtime/data/processed/tier2_train_50x.jsonl`
- the training pipeline already exists:
  - `experiment_runtime/src/experiment/train_qwen/`
- the current config already exists:
  - `experiment_runtime/configs/train/stage1_lora.toml`
- use `uv` with Python `3.12`
- keep this ticket single-GPU only
- target the current Linux box with the `NVIDIA RTX PRO 6000 Blackwell`

Do not regenerate datasets in this ticket.

## Scope

You own the official Stage 1 training execution under `experiment_runtime/`.

Your job is to:

1. verify the existing training pipeline is still healthy on the Linux box
2. make only minimal pipeline fixes if a real blocker appears
3. run the official `1x`, `10x`, and `50x` trainings sequentially
4. preserve isolated run folders and local heavyweight artifacts for later tickets
5. commit lightweight metadata artifacts so the manager can review the outcome after pull
6. write one small summary artifact that compares the three completed runs

This ticket is primarily an execution ticket, not a refactor ticket.

Do not expand scope into:

- Stage 1 MIA evaluation
- Stage 2 replay
- Stage 3 filter training
- FHE integration
- distributed or multi-GPU training

## Frozen Contract You Must Preserve

Do not change these experiment rules:

- base model: `Qwen/Qwen2-1.5B-Instruct`
- standard LoRA, not QLoRA
- no 4-bit quantization
- single GPU only
- full-sequence causal LM loss
- no assistant-only masking
- same optimizer-step budget across `1x`, `10x`, and `50x`
- same tokenizer max sequence length across all three runs
- same learning-rate schedule across all three runs

The point of the comparison is canary exposure, not changing the training recipe per condition.

## Allowed Adjustments

Assume the current config is the starting point.

You may make a small training-pipeline adjustment only if you hit a concrete blocker during the real runs. Examples of acceptable changes:

- lowering `per_device_train_batch_size` to fit memory
- increasing `gradient_accumulation_steps` to preserve effective batch behavior
- enabling `gradient_checkpointing` if memory requires it
- a small bug fix in the training runner or config handling

If you change the training recipe:

- keep the change shared across all three exposure conditions
- keep the optimizer-step budget identical across all three exposure conditions
- keep the loss rule unchanged
- document exactly why the change was necessary
- rerun any relevant lightweight tests before launching the official runs

Do not change:

- model family
- exposure definitions
- corpus contents
- loss masking policy
- dataset ordering

## Run Naming

Use explicit official run names under `experiment_runtime/runs/stage1/`.

Recommended names:

- `official-1x-20260410-r1`
- `official-10x-20260410-r1`
- `official-50x-20260410-r1`

If you must rerun one condition after a genuine failure, increment the trailing revision number and keep only the final successful run folder in your final report.

Do not overwrite an existing run directory.

## Execution Policy

Run the three trainings sequentially on the Linux GPU:

1. `1x`
2. `10x`
3. `50x`

Do not run them in parallel on one GPU.

Use the existing CLI:

- `fhe-train-stage1`

Use the non-smoke path. Do not pass `--smoke`.

## Required Commands

At minimum, run:

1. environment/setup command if needed:
   - `uv sync --python 3.12`
2. quick regression checks before long runs:
   - `uv run --python 3.12 python -m unittest tests.test_stage1_training_pipeline`
   - `uv run --python 3.12 python -m unittest tests.test_stage1_corpora tests.test_stage1_training_pipeline`
3. the three official trainings:
   - `uv run --python 3.12 fhe-train-stage1 --config configs/train/stage1_lora.toml --exposure 1x --run-name <run_name>`
   - `uv run --python 3.12 fhe-train-stage1 --config configs/train/stage1_lora.toml --exposure 10x --run-name <run_name>`
   - `uv run --python 3.12 fhe-train-stage1 --config configs/train/stage1_lora.toml --exposure 50x --run-name <run_name>`

If you add any helper commands for inspection or summarization, keep them minimal and document them.

## Required Run Artifacts

Each official run folder must contain the same lightweight metadata structure already used by the smoke run:

- `resolved_config.toml`
- `run_metadata.json`
- `environment.json`
- `train_metrics.json`
- `trainer_state.json`
- `adapter_model/adapter_config.json`
- tokenizer metadata if the runner writes it

Heavy local artifacts should remain on the Linux box for later work:

- adapter weights
- trainer checkpoints
- optimizer state
- full tokenizer files if they are already gitignored

Do not remove needed local heavy artifacts after the run. The next Linux ticket will need the trained adapters.

## Committed Summary Artifact

Add one committed summary file at:

- `experiment_runtime/runs/stage1/official_runs_summary.json`

It should summarize the three final successful runs with, at minimum:

- run name
- run directory
- exposure condition
- corpus path
- train example count
- configured max steps
- final global step
- final training loss if available
- train runtime if available
- train steps per second if available
- sequence length summary
- base model name
- whether smoke was disabled

Use only information derived from the produced run artifacts. Do not hand-edit unverifiable numbers.

## Git Hygiene

The repo already ignores heavyweight run binaries under `experiment_runtime/runs/.gitignore`.

In this ticket:

- keep lightweight metadata visible and commit it
- do not try to commit large adapter `.safetensors` files or checkpoint directories
- do not weaken the run-artifact ignore policy unless there is a concrete reason

## Done Criteria

This ticket is done only if all of the following are true:

1. three official runs completed successfully on Linux:
   - `1x`
   - `10x`
   - `50x`
2. each run has a complete lightweight metadata folder under `experiment_runtime/runs/stage1/`
3. the summary file `experiment_runtime/runs/stage1/official_runs_summary.json` exists and is correct
4. any code/config changes needed for successful execution are committed
5. the final report clearly distinguishes:
   - committed lightweight artifacts
   - local-only heavyweight artifacts that remain on the Linux box

## What To Report Back

When done, report:

- what files you changed
- whether the existing pipeline ran unchanged or required fixes
- the exact final run names
- the exact commands you ran
- a concise per-run summary:
  - exposure
  - train examples
  - global step
  - final train loss
  - train runtime
  - train steps per second
- where the local heavy adapter artifacts now live on the Linux box
- what remains out of scope

## Important Non-Goals

Do not implement or start:

- Stage 1 MIA scoring
- Stage 2 inference or leakage scoring
- Stage 3 classifier training
- any FHE work
- any additional prompt/spec rewriting
