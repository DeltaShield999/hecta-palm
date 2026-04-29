# Follow-On Shared Context

Read this before every follow-on implementation task.

## Project State

The original `Qwen2-1.5B-Instruct` FHE privacy experiment is complete and handoff-ready.

Original result artifacts live under:

- `experiment_runtime/runs/stage1/`
- `experiment_runtime/runs/stage2/`
- `experiment_runtime/runs/stage3/`

Do not replace, mutate, or reinterpret the frozen original experiment artifacts unless the specific task explicitly asks for a narrow documentation update.

## Follow-On Scope

The authoritative follow-on scope is:

1. `plan2.md`
2. `plan/05_follow_on_adaptive_evaluation.md`

The designer explicitly settled the scope:

```text
Use plan2.md as the authoritative scope. Do not implement threshold sensitivity, keyword/rule baselines, or the broader generalization check in this pass. The priority is adaptive attacker evaluation, mixed-traffic evaluation, confidence intervals, expanded timing, and updated result artifacts/docs. We can add ablations later after the adaptive evaluation is stable.
```

This means `plan1.md` is background only.

## Required Read Order

For every follow-on task, read:

1. `prompts/00_shared_context.md`
2. `prompts/follow_on/00_follow_on_shared_context.md`
3. `plan/README.md`
4. `plan/05_follow_on_adaptive_evaluation.md`
5. the task-specific prompt

Read original plan files only when the task needs the existing Stage 1, Stage 2, Stage 3, or repo-layout details.

## Repo Organization

The follow-on is not a separate project.

Use the existing runtime package:

- configs under `experiment_runtime/configs/follow_on/`
- data under `experiment_runtime/data/processed/follow_on/`
- runs under `experiment_runtime/runs/follow_on/`
- code under `experiment_runtime/src/experiment/follow_on/`
- tests under `experiment_runtime/tests/`

Reuse existing modules where practical:

- `experiment.eval` for Stage 2 model replay and leakage scoring
- `experiment.filter_train` for plaintext filter artifacts and embedding helpers
- `experiment.fhe` for OpenFHE scorer reuse
- `qwen_langgraph_demo.runtime.protocol` for the frozen prompt/template protocol

Do not route official follow-on metrics through the placeholder LangGraph scaffold.

## Environment

This repo uses Python `3.12`.

Local Mac is suitable for:

- data generation
- validators
- configs
- confidence interval utilities
- timing helpers
- unit tests that do not load Qwen or OpenFHE

The Linux NVIDIA/OpenFHE box is required for:

- official Qwen adapter generation runs
- full adaptive attack replay
- full mixed-traffic replay
- FHE-filter runs

Use `uv` with Python `3.12`.

For OpenFHE tasks, use:

```bash
uv sync --python 3.12 --extra fhe
```

## Non-Goals

Do not implement in this pass:

- threshold sensitivity
- keyword/rule baselines
- broader generalization checks
- new Stage 1 training
- 7B repeat
- full LLM encryption
- encrypted sentence embedding
- production LangGraph integration

## Done Criteria Pattern

Each follow-on task should finish with:

- files implemented in the requested areas
- tests or smoke checks run when practical
- generated artifacts if the task owns materialization or official runs
- clear notes about any skipped GPU/OpenFHE verification
- no unrelated rewrites
