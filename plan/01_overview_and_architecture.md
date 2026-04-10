# Experiment Overview and Architecture

## 1. Purpose

This plan set is the implementation-facing source of truth for the experiment described in [FHE_Experiment_Engineering_Guide.md](../docs/md/FHE_Experiment_Engineering_Guide.md).

It consolidates:

- fixed requirements from the engineering guide
- implementation clarifications extracted from [FHE_Experiment_Engineering_Guide_chat.md](../chats/FHE_Experiment_Engineering_Guide_chat.md)
- explicit defaults for underspecified areas so future Codex sessions can build against one concrete plan

The papers in `docs/md` and the notebooks in `notebooks/md` are background only. They are not normative sources for this experiment.

## 2. Source Hierarchy

Use sources in this order:

1. `docs/md/FHE_Experiment_Engineering_Guide.md`
2. the `plan/` documents
3. `chats/FHE_Experiment_Engineering_Guide_chat.md`
4. background papers and notebooks, only if needed for intuition

If the plan documents conflict with the guide on a fixed constraint, the guide wins. If the plan documents resolve an area the guide left open, the plan documents win unless the experiment designer provides a newer signed-off replacement.

## 3. Experiment in One Sentence

Train a Qwen2 fraud-scoring model on synthetic records with repeated canaries, show that a compromised upstream agent can sometimes extract memorized canary PII despite a system prompt, then show that a message filter on the intake-to-fraud edge blocks most of those attacks and can be run with CKKS-based FHE.

## 4. Fixed Constraints From the Guide

| Area | Fixed requirement |
| --- | --- |
| Model family | Qwen2 |
| Primary model | `Qwen2-1.5B-Instruct` |
| Secondary model | `Qwen2-7B-Instruct` |
| Fine-tuning method | LoRA on attention projection layers |
| Exposure conditions | Exactly `1x`, `10x`, `50x` |
| MIA metrics | `AUC-ROC`, `TPR@1%FPR`, `TPR@10%FPR` |
| Member split | `8,000` member records |
| Non-member split | `2,000` held-out non-members |
| Canary content | Tier 1 records with synthetic `customer_name`, `date_of_birth`, `account_number` |
| Pipeline shape | deterministic intake -> LLM fraud scorer -> deterministic router |
| Filter model | sentence embeddings + logistic regression |
| FHE scheme | CKKS via OpenFHE, or sponsor-approved compatible alternative |

## 4.1 Repo-Level Engineering Decision

The guide leaves the orchestration framework open. In this repo, use `LangGraph` and evolve the existing `experiment_runtime/` project into the experiment harness.

This is now a local implementation decision, not a research constraint.

Implications:

- the experiment does have an agentic component, but it is lightweight
- the LangGraph graph should model the intake, filter, fraud scorer, and router flow
- only the fraud-scoring node is an LLM-backed agent
- the intake and router should remain deterministic nodes even if they are represented inside a LangGraph graph
- the filter should be explicit middleware on the intake-to-fraud edge, not hidden inside the fraud node

## 5. Final Architecture

### 5.1 Components

| Component | Role | Notes |
| --- | --- | --- |
| Transaction Intake Agent | deterministic message formatter | treated as compromised during attack runs |
| Filter Middleware | embedding, encryption, decryption, thresholding, logging | sits between intake and fraud scorer |
| Fraud Scoring Agent | LoRA fine-tuned Qwen2 | only LLM in the system |
| Case Routing Agent | deterministic post-processor | reads fraud score and routes case |
| FHE Inference Service | CKKS logistic-regression scoring | never sees plaintext embeddings |

### 5.1.1 LangGraph Mapping

Implement the experimental pipeline as a small LangGraph app with explicit nodes such as:

- `intake`
- `filter_middleware`
- `fraud_scorer`
- `router`

Recommended rule:

- use LangGraph for orchestration and message passing
- keep the actual business logic of data generation, MIA, filter training, FHE scoring, and evaluation in regular Python modules outside the graph definition

### 5.2 Trust Boundary

The intake agent is the attacker-controlled component in Stage 2 and Stage 3. The filter middleware is not part of the compromised intake agent.

The only valid message path to the fraud model is:

```text
intake -> filter middleware -> fraud scorer
```

The intake agent must not be able to call the fraud model directly.

### 5.3 Stage Order

Build and evaluate in this order:

1. freeze data and protocol spec
2. generate data and validators
3. fine-tune `Qwen2-1.5B-Instruct`
4. run Stage 1 MIA
5. build Stage 2 attack harness
6. train plaintext filter
7. add FHE wrapper around the filter
8. rerun attack with filter active
9. package results
10. repeat for `Qwen2-7B-Instruct`

Do not start with FHE. Do not start with 7B.
