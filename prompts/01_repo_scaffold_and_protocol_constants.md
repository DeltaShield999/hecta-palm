# Task 01: Repo Scaffold and Protocol Constants

Read first:

1. `prompts/00_shared_context.md`
2. `plan/README.md`
3. `plan/01_overview_and_architecture.md`
4. `plan/04_repo_and_execution_plan.md`
5. `plan/02_data_and_stage1.md`
6. `plan/03_stage2_and_stage3.md`

## Goal

Start implementation by turning `experiment_runtime/` from a translation demo into the experiment's initial runtime and package scaffold.

This task is local-first and should be done on the user's Mac. Do not assume NVIDIA Linux access for this ticket.

## Why This Is First

We have already frozen the protocol in `plan/`.

The next bottleneck is not training or data generation. It is giving the project a clean implementation skeleton and turning the frozen protocol into machine-readable config files that every later ticket can rely on.

This task should create that foundation.

## Scope

You own the initial scaffold inside `experiment_runtime/`.

Implement:

1. repo/package structure for the experiment
2. machine-readable protocol constants derived from the frozen plan
3. a minimal LangGraph experiment skeleton for the pipeline
4. a minimal CLI or entrypoint for the experiment harness
5. basic tests or smoke checks for config loading and graph construction
6. docs updates inside `experiment_runtime/` so the project no longer reads as only a translation demo

## Required Structure

Use `experiment_runtime/` as the home for the runtime implementation.

Create or evolve the code toward this shape:

```text
experiment_runtime/
  configs/
  data/
    raw/
    processed/
  runs/
  scripts/
  src/
    qwen_langgraph_demo/
      graph/
      nodes/
      tools/
      runtime/
    experiment/
      schemas/
      data_gen/
      chat_render/
      train_qwen/
      mia/
      attacks/
      filter_train/
      fhe/
      eval/
  tests/
```

You do not need to fully implement every module. Create the structure and the minimal code needed to make the scaffold coherent.

## Protocol Constants To Materialize

Create machine-readable config files for the frozen experiment protocol. These should not be buried only in markdown.

At minimum, materialize configs for:

- model family and model variants
- fixed exposure conditions
- system prompt text
- canonical intake message wrapper
- canonical benign fraud-scoring response template
- Stage 2 attack families
- Stage 2 decoding settings
- Stage 2 leakage-scoring rules
- Stage 3 ALLOW/BLOCK labeling rule
- Stage 3 dataset defaults
- Stage 3 filter runtime shape

Use a readable format like YAML or JSON.

## LangGraph Skeleton

Build a minimal graph that reflects the experiment architecture:

- `intake`
- `filter_middleware`
- `fraud_scorer`
- `router`

Requirements:

- use deterministic placeholder logic for `intake`, `filter_middleware`, and `router`
- the `fraud_scorer` can be a stub or wrapper placeholder for now
- graph construction must be explicit and readable
- do not implement the full experiment in this ticket

## Non-Goals

Do not implement yet:

- Tier 1 data generation
- Tier 2 chat rendering logic
- LoRA training
- MIA scoring
- Stage 2 attack execution
- Stage 3 classifier training
- FHE integration

This is a scaffold ticket only.

## Practical Constraints

- use `uv` with Python `3.13`
- do not keep the project positioned as a translation demo
- avoid giant speculative abstractions
- keep the skeleton simple and easy for later Codex sessions to extend
- if you preserve any old demo code, make sure it does not remain the main documented workflow

## Deliverables

When done, the repo should have:

- a coherent experiment-oriented package layout
- frozen protocol config files checked into `experiment_runtime/configs/`
- a buildable/importable LangGraph scaffold for the experiment pipeline
- a basic CLI or entrypoint for the scaffold
- smoke tests for config loading and graph creation
- updated `README.md` and, if needed, `SETUP.md`

## Done Criteria

This task is done when:

1. a future Codex session can start from the scaffold without reading the old translation-demo assumptions
2. frozen protocol constants exist as config files rather than only markdown
3. the LangGraph scaffold can be imported and instantiated
4. basic tests or smoke checks pass
5. the final report clearly says what was changed, what was verified, and what remains stubbed
