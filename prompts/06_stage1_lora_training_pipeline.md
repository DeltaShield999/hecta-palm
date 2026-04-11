# Task 06: Stage 1 LoRA Training Pipeline

Read first:

1. `prompts/00_shared_context.md`
2. `plan/README.md`
3. `plan/02_data_and_stage1.md`
4. `plan/04_repo_and_execution_plan.md`

## Goal

Implement the Stage 1 Qwen2-1.5B-Instruct LoRA training pipeline inside `experiment_runtime/` and verify it with a smoke run on the Linux NVIDIA box.

This task should produce:

- a config-driven single-GPU training pipeline for `Qwen/Qwen2-1.5B-Instruct`
- support for the three frozen exposure conditions:
  - `1x`
  - `10x`
  - `50x`
- full-sequence causal LM loss over the chat-formatted examples
- isolated run-folder outputs under `experiment_runtime/runs/stage1/`
- one completed smoke run on the Linux GPU
- tests for the non-heavy pieces of the pipeline

This task must be executed on the user's Linux NVIDIA box, not on the Mac.

## Why This Is Next

All Phase 1 deterministic data-layer work is complete:

- Tier 1 records
- Stage 1 corpora
- Stage 2 attack prompts
- Stage 3 ALLOW/BLOCK datasets

The next planned step is Phase 2 Stage 1 training. This ticket should implement the training pipeline and prove it works with a smoke run before anyone launches the full `1x / 10x / 50x` experiment runs.

## Important Repo and Environment Context

- the datasets are already committed in the repo on the Linux box
- do not regenerate the deterministic datasets in this ticket
- use `uv` with Python `3.12`
- assume a single powerful NVIDIA GPU is available
- target the current Linux box with an `NVIDIA RTX PRO 6000 Blackwell`
- keep this ticket single-GPU only
- do not implement distributed training, multi-node training, or cluster orchestration

## Scope

You own the Stage 1 training pipeline under `experiment_runtime/src/experiment/train_qwen/`.

Implement:

1. a Stage 1 training config loader
2. dataset loading for `tier2_train_1x.jsonl`, `tier2_train_10x.jsonl`, and `tier2_train_50x.jsonl`
3. Qwen chat tokenization for the frozen `messages` arrays
4. full-sequence loss labeling
5. LoRA model setup for `Qwen/Qwen2-1.5B-Instruct`
6. a single-GPU training runner
7. isolated run-folder output writing
8. a CLI entrypoint for training
9. lightweight tests for config/dataset/tokenization behavior
10. one actual Linux GPU smoke run

## Read The Frozen Contract Carefully

Follow the Stage 1 plan exactly:

- primary model: `Qwen/Qwen2-1.5B-Instruct`
- the training inputs are:
  - `experiment_runtime/data/processed/tier2_train_1x.jsonl`
  - `experiment_runtime/data/processed/tier2_train_10x.jsonl`
  - `experiment_runtime/data/processed/tier2_train_50x.jsonl`
- each row already contains the frozen `messages` array
- use full-sequence causal LM loss over the chat example
- do not use assistant-only SFT masking
- padding should be masked out
- the user-message tokens carrying canary content must remain in the loss
- the official comparison rule across `1x / 10x / 50x` is fixed optimizer steps, not fixed epochs

This ticket does not run the official full trainings yet. It only builds the pipeline and proves it works with a smoke run.

## Model and Training Policy

Use standard LoRA on top of the base model. Do not use QLoRA or 4-bit quantization for the default pipeline in this project unless you hit a hard blocker and can justify it clearly.

Reason:

- this experiment is about memorization and extraction behavior
- introducing quantized training as the default adds an unnecessary confound
- the target GPU is strong enough that standard LoRA is the cleaner baseline

Use `bf16` training by default on this box.

## Recommended Default LoRA Settings

Set these as the initial config defaults unless you discover a concrete incompatibility:

- `r = 32`
- `lora_alpha = 64`
- `lora_dropout = 0.05`
- target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

These should be config values, not hardcoded constants.

## Tokenization and Loss Rule

This is the most important implementation rule in the ticket.

Use the tokenizer's Qwen chat template on the frozen `messages` arrays.

Implementation rule:

1. read the `messages` array from the JSONL row
2. render/tokenize it with the Qwen tokenizer chat template
3. set `labels = input_ids`
4. mask only padding positions with `-100`
5. do not create assistant-only labels
6. do not mask the user-message tokens

Use:

- `add_generation_prompt = False`

Do not use `trl.SFTTrainer` defaults or any assistant-only masking helpers that would contradict the frozen loss rule.

## Recommended Stack

Prefer a minimal, auditable stack:

- `torch`
- `transformers`
- `peft`
- `accelerate`
- `safetensors`

Add only what is needed for this ticket.

Do not pull in extra training frameworks unless there is a concrete reason.

## Required Config

Add a training config under:

- `experiment_runtime/configs/train/stage1_lora.toml`

The config should cover at minimum:

- protocol/config paths
- input corpus paths for `1x`, `10x`, `50x`
- output root
- base model name
- tokenizer settings
- max sequence length
- LoRA parameters
- training parameters
- smoke-run parameters
- seed

Keep it easy to override from the CLI where useful.

## Required CLI

Add a CLI entrypoint, preferably:

- `fhe-train-stage1`

The CLI must support at least:

- `--config`
- `--exposure` with `1x | 10x | 50x`
- `--run-name`
- `--smoke`

If you need one or two extra flags, keep them minimal and justified.

## Run Output Requirements

Every run must write to an isolated folder under:

- `experiment_runtime/runs/stage1/<run_name>/`

Each run folder should contain at minimum:

- resolved config snapshot
- run metadata
- environment summary
- training metrics JSON
- trainer state if available
- final adapter weights
- tokenizer artifacts if needed for later reuse

Suggested files:

- `resolved_config.toml`
- `run_metadata.json`
- `environment.json`
- `train_metrics.json`
- `trainer_state.json`
- `adapter_model/`

Keep the output structure simple and stable.

## Smoke-Run Contract

This ticket must execute one real smoke run on the Linux GPU.

Use these requirements:

- exposure: `1x`
- use the real `tier2_train_1x.jsonl` file
- cap the smoke run so it is fast and cheap
- save a real adapter artifact at the end

Recommended smoke defaults:

- `max_train_examples = 256`
- `max_steps = 5`
- `logging_steps = 1`
- `save_steps = 5`

These values can live in the config under a smoke section.

The smoke run should prove:

- model loads
- tokenizer path works
- dataset path works
- labels are constructed correctly
- training runs on GPU
- a final adapter can be saved

## Single-GPU Assumptions

Keep the first implementation single-GPU.

That means:

- one process
- one GPU
- no DDP
- no DeepSpeed
- no FSDP

If you use `accelerate`, keep it in a single-GPU configuration.

## Suggested Code Areas

Own code under:

- `experiment_runtime/src/experiment/train_qwen/`
- `experiment_runtime/configs/train/`
- `experiment_runtime/tests/`

You will likely want modules like:

- config loading
- dataset loading / tokenization
- collator or padding helper
- LoRA model setup
- training runner

You may also update:

- `experiment_runtime/pyproject.toml`

for dependencies and the CLI entrypoint.

## Validation Requirements

Implement validation or tests that check at minimum:

1. the training config loads correctly
2. all three exposure corpora can be selected by config/CLI
3. the dataset reader preserves the frozen `messages` structure
4. tokenization uses the Qwen chat template
5. labels are full-sequence labels, not assistant-only labels
6. padding is masked with `-100`
7. the CLI builds the expected run folder path
8. the smoke run writes the required artifacts

Tests do not need to launch a real full training job on CPU. Keep them lightweight where possible.

## Practical Constraints

- use `uv` with Python `3.12`
- keep the implementation easy to audit
- do not redesign the Stage 1 protocol
- do not regenerate the datasets
- do not run the full `1x / 10x / 50x` experiment in this ticket
- do not implement the MIA evaluator yet
- do not start Stage 2 replay, Stage 3 filter training, or FHE work

## Non-Goals

Do not implement yet:

- official full `1x / 10x / 50x` training runs
- Stage 1 MIA scoring
- Qwen2-7B training
- vLLM serving
- Stage 2 inference
- Stage 3 plaintext filter training
- FHE integration

This ticket ends at a working Stage 1 training pipeline plus one successful Linux GPU smoke run.

## Deliverables

When done, the repo should have:

- a config-driven Stage 1 training pipeline
- a Stage 1 training config file
- a CLI entrypoint
- lightweight tests
- one completed smoke run folder under `runs/stage1/`

## Done Criteria

This task is done when:

1. the pipeline can launch from the CLI on Linux
2. the smoke run completes on GPU
3. a final adapter artifact is saved
4. the run folder contains the required metadata and metrics
5. lightweight tests pass
6. the final report clearly states:
   - what files changed
   - what training stack was used
   - how full-sequence labels are constructed
   - what the smoke-run settings were
   - what commands were run
   - what artifacts were produced
   - what remains out of scope
