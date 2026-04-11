# Task 03: Tier 2 Chat Rendering and Stage 1 Corpora

Read first:

1. `prompts/00_shared_context.md`
2. `plan/README.md`
3. `plan/02_data_and_stage1.md`
4. `plan/04_repo_and_execution_plan.md`

## Goal

Implement the deterministic Tier 2 chat-rendering layer and materialize the frozen Stage 1 corpora inside `experiment_runtime/`.

This task should produce:

- deterministic chat rendering from Tier 1 records into the frozen `system` / `user` / `assistant` format
- the three Stage 1 training corpora:
  - `tier2_train_1x.jsonl`
  - `tier2_train_10x.jsonl`
  - `tier2_train_50x.jsonl`
- the frozen MIA evaluation corpus:
  - `mia_eval.jsonl`
- validators and tests that enforce the corpus contract

This task is local-first and should be done on the user's Mac. Do not assume NVIDIA Linux access.

## Why This Is Next

Task 02 produced the source factual dataset:

- `experiment_runtime/data/processed/tier1_records.parquet`
- `experiment_runtime/data/processed/canary_registry.csv`

The next hard dependency is the chat-formatted corpora that Stage 1 fine-tuning and Stage 1 MIA will consume.

Without these artifacts, the Linux/NVIDIA fine-tuning work cannot start.

## Scope

You own the Tier 2 rendering and Stage 1 corpus materialization layer inside `experiment_runtime/`.

Implement:

1. deterministic rendering of Tier 1 records into the frozen chat structure
2. deterministic benign assistant-response generation for those chats
3. materialization of:
   - `tier2_train_1x.jsonl`
   - `tier2_train_10x.jsonl`
   - `tier2_train_50x.jsonl`
   - `mia_eval.jsonl`
4. validators for schema, message shape, exposure counts, and MIA composition
5. tests for determinism and validator behavior
6. a CLI or script entrypoint that materializes the Stage 1 corpora

## Read The Frozen Contract Carefully

Follow the plan exactly for Tier 2 and Stage 1:

- use the exact system prompt already frozen in `configs/protocol/stage1.toml`
- use the exact user-message outer format already frozen in `configs/protocol/stage1.toml`
- use the exact benign request text already frozen in `configs/protocol/stage1.toml`
- use the exact assistant response outer format already frozen in `configs/protocol/stage1.toml`
- training corpora use only member records
- MIA evaluation corpus uses all Tier 1 records once
- use deterministic templates only for this task
- do not use an LLM for Tier 2 rendering in this ticket

Optional LLM paraphrasing is allowed by the plan in principle, but it is explicitly out of scope for this task.

## Required Artifacts

Materialize at least:

- `experiment_runtime/data/processed/tier2_train_1x.jsonl`
- `experiment_runtime/data/processed/tier2_train_10x.jsonl`
- `experiment_runtime/data/processed/tier2_train_50x.jsonl`
- `experiment_runtime/data/processed/mia_eval.jsonl`

If you need intermediate files, keep them under `experiment_runtime/data/processed/` and document them.

## Exact Message Structure

Each rendered example must contain:

- a `system` message with the exact frozen system prompt
- a `user` message rendered with the exact intake template and the exact benign request text
- an `assistant` message rendered with the exact benign response template

The `messages` field in JSONL must be an ordered JSON array of objects:

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

Do not serialize `messages` as a single string.

Reuse the existing protocol helpers where appropriate:

- `qwen_langgraph_demo.runtime.protocol.load_protocol_bundle`
- `ProtocolBundle.render_intake_message(...)`
- `ProtocolBundle.format_assistant_response(...)`

Do not duplicate the protocol strings in a second place if you can avoid it.

## Exact Training-Corpus Schema

Each JSONL row in the three training corpora must contain:

- `example_id`
- `record_id`
- `canary_id`
- `messages`
- `split`
- `exposure_condition`

Rules:

- `split` must be `member` for all training-corpus rows
- `canary_id` must be `null` for non-canaries and the actual canary ID string for canaries
- `exposure_condition` must be one of `1x`, `10x`, `50x`

Freeze this `example_id` convention:

- `train_{exposure_condition}_{record_id}_r{repeat_index:02d}`

Examples:

- `train_1x_REC-000123_r01`
- `train_10x_REC-000456_r07`

## Exact MIA-Eval Schema

Each JSONL row in `mia_eval.jsonl` must contain:

- `eval_id`
- `record_id`
- `split`
- `is_canary`
- `messages`

Freeze this `eval_id` convention:

- `mia_{record_id}`

Rules:

- include every Tier 1 record exactly once
- that means all `10,000` rows:
  - `8,000` member
  - `2,000` non-member
- `messages` must use the same exact chat structure as training
- do not add `exposure_condition` to `mia_eval.jsonl`

## Exposure Rule

Materialize the three training corpora exactly like this:

- `1x`: all `8,000` member records appear once
- `10x`: all `8,000` member records appear once, then add `9` extra appearances for each of the `100` canaries
- `50x`: all `8,000` member records appear once, then add `49` extra appearances for each of the `100` canaries

This yields exact corpus sizes:

- `1x`: `8,000`
- `10x`: `8,900`
- `50x`: `12,900`

The corpus contents should be deterministic and auditable.

Freeze this row-order policy:

1. base pass: all member records sorted by `record_id` ascending, each with `repeat_index = 1`
2. extra canary repetitions: canary records sorted by `canary_id` ascending, appended in repetition order:
   - `repeat_index = 2..10` for `10x`
   - `repeat_index = 2..50` for `50x`

Do not randomize ordering in this task.

## Benign Assistant-Response Generation

You must generate deterministic benign assistant responses for the training and MIA corpora.

Requirements:

- do not use an LLM
- produce exactly the frozen outer format:
  - `Decision: ...`
  - `Fraud score: ...`
  - `Rationale: ...`
- format the score with exactly two decimals
- rationale must be one or two sentences
- rationale must be grounded in transaction features
- rationale must not quote or reveal customer PII

Important implementation rule:

- training-time rendering and runtime-time rendering should not drift

So, do this:

- create a shared deterministic fraud-response renderer under `experiment_runtime/src/experiment/chat_render/`
- update the current scaffold fraud scorer in `qwen_langgraph_demo/nodes/fraud_scorer.py` to reuse that shared renderer if needed

The goal is that the current runtime harness and the materialized corpora follow the same benign fraud-scoring behavior rather than maintaining two unrelated implementations.

Use only transaction fields that are visible in the frozen user message.

Do not use hidden labels or hidden metadata that the model would not see at inference time.

## Determinism Requirements

This task must be deterministic.

If the materialization command is run twice against the same Tier 1 artifacts and config, it should produce byte-for-byte stable JSONL ordering and content.

Use deterministic Python only.

## Suggested Code Areas

Own code under:

- `experiment_runtime/src/experiment/chat_render/`
- `experiment_runtime/src/experiment/data_gen/` if shared loaders/helpers are needed
- `experiment_runtime/src/experiment/schemas/` if you need corpus schemas
- `experiment_runtime/configs/` if you add a Stage 1 corpus config
- `experiment_runtime/tests/`

You may also update:

- `experiment_runtime/src/qwen_langgraph_demo/nodes/fraud_scorer.py`

if that is needed to keep runtime rendering aligned with the corpora.

## Validation Requirements

Implement validators that check at minimum:

1. exact row counts for each corpus:
   - `8,000`
   - `8,900`
   - `12,900`
   - `10,000`
2. training corpora contain only member records
3. MIA eval contains every Tier 1 record exactly once
4. `messages` is a three-item ordered array with roles:
   - `system`
   - `user`
   - `assistant`
5. the `system` content matches the frozen protocol exactly
6. the `user` content matches the frozen intake wrapper exactly
7. the `assistant` content matches the frozen response outer format exactly
8. exposure counts for each canary are exact:
   - `1x`: `1`
   - `10x`: `10`
   - `50x`: `50`
9. non-canary members appear exactly once in every training corpus
10. assistant messages do not contain the source record's:
   - `customer_name`
   - `date_of_birth`
   - `account_number`

If helpful, implement reusable validation functions plus test coverage.

## Practical Constraints

- use `uv` with Python `3.12`
- keep the implementation easy to audit
- do not redesign the corpus schema
- do not start tokenization, training, LoRA, or MIA scoring
- do not introduce random shuffling in this ticket
- do not add speculative abstractions for later stages

## Non-Goals

Do not implement yet:

- Qwen fine-tuning
- training launch scripts for Linux
- tokenization pipelines
- loss computation
- MIA scoring
- Stage 2 attack prompt generation
- Stage 3 ALLOW/BLOCK dataset generation
- FHE integration

This ticket ends at deterministic chat rendering plus Stage 1 corpus materialization and validation.

## Deliverables

When done, the repo should have:

- deterministic chat-rendering code
- a shared benign response renderer
- materialization code or a CLI for Stage 1 corpora
- generated JSONL artifacts in `data/processed/`
- validators
- tests for determinism and exposure correctness
- runtime fraud-scoring alignment if needed

## Done Criteria

This task is done when:

1. all four JSONL artifacts can be generated locally
2. the corpora satisfy the frozen contract exactly
3. validators catch contract violations
4. tests or smoke checks pass
5. the final report clearly states:
   - what files changed
   - how assistant responses are generated
   - how exposure counts are enforced
   - what commands were run
   - what was verified
   - what remains out of scope
