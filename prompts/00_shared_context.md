# Shared Context

Read this before starting any implementation task for this project.

## Project Context

This repo contains an experiment with three stages:

1. fine-tune a Qwen2 fraud-scoring model on synthetic financial records with repeated canaries and run a membership inference attack
2. place that model in a simple agentic pipeline and test whether a compromised upstream agent can extract memorized canary PII
3. add a message filter on the intake-to-fraud edge, train it in plaintext, then run it with CKKS-based FHE scoring and re-run the attack

The canonical implementation plan is in `plan/`.

Read order:

1. `plan/README.md`
2. the task-specific plan files referenced by your prompt

Primary source hierarchy:

1. `docs/md/FHE_Experiment_Engineering_Guide.md`
2. the `plan/` documents
3. `chats/FHE_Experiment_Engineering_Guide_chat.md`
4. papers and notebooks only as optional background

## Architecture

The experiment has an agentic component, but it is intentionally lightweight.

Use `experiment_runtime/` as the LangGraph harness.

Target runtime flow:

`intake -> filter_middleware -> fraud_scorer -> router`

Rules:

- only the `fraud_scorer` is LLM-backed
- `intake` and `router` are deterministic nodes
- the filter is explicit middleware on the intake-to-fraud edge
- the intake agent is the attacker-controlled component during Stage 2 and Stage 3

## Current Stage

Protocol and planning are frozen. Implementation is about to begin.

That means:

- do not reopen core protocol questions unless you find a real contradiction
- do not redesign the experiment
- build from the plan

## Environment Notes

The user is working on a Mac locally.

Some later tasks must run on an NVIDIA Linux box. Assume:

- local Mac is fine for scaffolding, configs, schemas, data generation, validation, prompt files, and general Python work
- NVIDIA Linux is needed later for Qwen training, vLLM serving, and likely other GPU-heavy tasks

If your task is local-only, do not introduce Linux-only assumptions.

## Repo Rules

- `experiment_runtime/` is relevant and is the intended LangGraph base for this experiment
- use `uv` with Python `3.12`
- do not care about backward compatibility unless explicitly told otherwise
- do not run full builds after trivial changes
- keep outputs deterministic where possible
- prefer explicit modules over notebook-only logic

## Working Style

When you work on a task:

- stay within the task boundary
- read only the plan files relevant to your task
- implement the requested code, docs, tests, and configs directly
- do not start adjacent future work unless the prompt explicitly asks for it

## What Counts As Done

Unless the prompt says otherwise, done means:

- the requested files and modules exist
- the code structure matches the plan
- tests or smoke checks for that task pass if practical
- you report what changed, what you verified, and what remains out of scope
