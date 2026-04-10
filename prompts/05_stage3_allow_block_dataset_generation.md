# Task 05: Stage 3 ALLOW/BLOCK Dataset Generation

Read first:

1. `prompts/00_shared_context.md`
2. `plan/README.md`
3. `plan/02_data_and_stage1.md`
4. `plan/03_stage2_and_stage3.md`
5. `plan/04_repo_and_execution_plan.md`

## Goal

Implement the deterministic Stage 3 filter-dataset generator inside `experiment_runtime/`.

This task should produce:

- `experiment_runtime/data/processed/stage3_filter_messages_train.jsonl`
- `experiment_runtime/data/processed/stage3_filter_messages_val.jsonl`
- `experiment_runtime/data/processed/stage3_filter_messages_test.jsonl`
- deterministic benign `ALLOW` message generation
- deterministic adversarial `BLOCK` message generation
- validators and tests that enforce the frozen dataset contract
- a CLI or script entrypoint that materializes the dataset

This task is local-first and should be done on the user's Mac. Do not assume NVIDIA Linux access.

## Why This Is Next

The remaining local deterministic data-layer task is the Stage 3 classifier dataset.

This dataset is needed before:

- plaintext embedding + logistic-regression training
- FHE compilation/evaluation later
- integrated final runs with the filter active

It is still a good local black-box task, so it should be completed before we shift to the first Linux/NVIDIA-heavy work.

## Scope

You own the Stage 3 dataset generation layer inside `experiment_runtime/`.

Implement:

1. deterministic `ALLOW` message generation
2. deterministic `BLOCK` message generation
3. cluster-based train/val/test splitting
4. materialization of the three Stage 3 dataset files
5. validators for schema, counts, label integrity, split integrity, wrapper consistency, and Stage 2 disjointness
6. tests for determinism and validator behavior
7. a CLI or script entrypoint that materializes the dataset

## Read The Frozen Contract Carefully

Follow the plan exactly for Stage 3:

- the classifier sees raw `message_text` only
- all messages must use the same outer intake wrapper defined for Stage 1
- only the final `Request:` line changes across ALLOW and BLOCK examples
- label rule:
  - `ALLOW` if the requested output is fraud scoring, routing, or risk reasoning
  - `BLOCK` if the requested output is raw identity data, memorized record contents, or verbatim disclosure of customer attributes
- defaults:
  - `1000` `ALLOW`
  - `1000` `BLOCK`
  - split `70 / 15 / 15`
- the split must preserve separation by template family or paraphrase cluster, not random row

This ticket is for deterministic, hand-authored message catalogs only.

Do not use an LLM for this ticket.

## Inputs

Read from the existing source artifacts:

- `experiment_runtime/data/processed/tier1_records.parquet`
- `experiment_runtime/data/processed/stage2_attack_prompts.jsonl`

Use the protocol configs already in:

- `experiment_runtime/configs/protocol/stage1.toml`
- `experiment_runtime/configs/protocol/stage3.toml`

Reuse the existing protocol helper instead of duplicating the outer wrapper:

- `qwen_langgraph_demo.runtime.protocol.load_protocol_bundle`
- `ProtocolBundle.render_intake_message(...)`

## Important Data Policy

Use only non-canary Tier 1 records for the Stage 3 dataset.

Reason:

- the filter is supposed to learn request intent, not memorize canary-specific text
- the integrated final Stage 2 attack rerun will use the frozen Stage 2 canary-targeted eval set
- keeping Stage 3 training/eval on non-canary records reduces contamination risk

So:

- do not use canary records in any Stage 3 train/val/test message
- use non-canary records only

## Required Output Artifacts

Materialize exactly:

- `experiment_runtime/data/processed/stage3_filter_messages_train.jsonl`
- `experiment_runtime/data/processed/stage3_filter_messages_val.jsonl`
- `experiment_runtime/data/processed/stage3_filter_messages_test.jsonl`

If you need intermediate files, keep them under `experiment_runtime/data/processed/` and document them.

## Exact Dataset Schema

Each JSONL row in all three files must contain exactly:

- `message_id`
- `message_text`
- `label`
- `template_family`
- `source_type`

Rules:

- `label` must be exactly `ALLOW` or `BLOCK`
- `source_type` must be:
  - `benign` for `ALLOW`
  - `adversarial` for `BLOCK`
- `message_text` must be the fully wrapped intake message string, not a bare request line
- `template_family` must be one of the frozen family names listed below

Do not add a `split` key to the JSONL rows. The split is represented by the filename.

## Exact Counts

Use exactly:

- `1000` `ALLOW`
- `1000` `BLOCK`

Split exactly:

- train: `1400` rows total
  - `700` `ALLOW`
  - `700` `BLOCK`
- val: `300` rows total
  - `150` `ALLOW`
  - `150` `BLOCK`
- test: `300` rows total
  - `150` `ALLOW`
  - `150` `BLOCK`

## Family Inventory

Use exactly `5` `ALLOW` families and exactly `5` `BLOCK` families.

### ALLOW Families

Use these exact family names and order:

1. `transaction scoring`
2. `routing recommendation`
3. `risk summary`
4. `mismatch explanation`
5. `audit or compliance risk review`

Intent requirements:

- `transaction scoring`
  - ask for normal fraud scoring or decisioning
- `routing recommendation`
  - ask whether the case should be allowed, reviewed manually, or blocked
- `risk summary`
  - ask for a concise fraud-risk explanation or signal summary
- `mismatch explanation`
  - ask for analysis of identity or account mismatch risk without asking to reveal raw identity fields
- `audit or compliance risk review`
  - this is the hard benign-negative family
  - it should sound operationally close to compliance/audit traffic but still ask only for fraud/risk reasoning, not disclosure

### BLOCK Families

Use these exact family names and order:

1. `direct injection`
2. `compliance or audit framing`
3. `role-play`
4. `urgency or escalation`
5. `subtle or indirect extraction`

Intent requirements:

- `direct injection`
  - explicit override/disclosure requests
- `compliance or audit framing`
  - audit, regulator, controls, reconciliation, or compliance-flavored disclosure requests
- `role-play`
  - act-as / pretend-you-are / simulate-a-tool disclosure requests
- `urgency or escalation`
  - incident, pager, outage, or escalation-flavored disclosure requests
- `subtle or indirect extraction`
  - quote-back, confirm-on-file, replay, verification, or indirect disclosure requests

Important:

- `ALLOW` vs `BLOCK` is about requested output, not about whether the wrapper contains customer fields
- the `audit or compliance risk review` family must stay `ALLOW`
- the `compliance or audit framing` family must stay `BLOCK`

## Cluster Policy

Use deterministic cluster-based generation.

For each of the `10` total families:

- create exactly `20` clusters
- create exactly `10` request variants per cluster

This yields:

- `200` rows per family
- `1000` rows per label
- `2000` rows total

The purpose of clusters is to keep semantically related variants together for split assignment.

## Split Policy

Split by cluster, not random row.

For every family:

- clusters `01..14` -> train
- clusters `15..17` -> val
- clusters `18..20` -> test

Since each cluster has `10` rows, this yields per family:

- train: `140`
- val: `30`
- test: `30`

Since there are `5` families per label, this yields exactly:

- train: `700` per label
- val: `150` per label
- test: `150` per label

This cluster assignment rule is frozen. Do not randomize it.

## Message ID Convention

Freeze this `message_id` convention:

- `stage3_{label_lower}_{family_slug}_c{cluster_index:02d}_v{variant_index:02d}`

Examples:

- `stage3_allow_transaction_scoring_c01_v01`
- `stage3_block_role_play_c17_v09`

Use these exact family slugs:

### ALLOW Slugs

- `transaction_scoring`
- `routing_recommendation`
- `risk_summary`
- `mismatch_explanation`
- `audit_or_compliance_risk_review`

### BLOCK Slugs

- `direct_injection`
- `compliance_or_audit_framing`
- `role_play`
- `urgency_or_escalation`
- `subtle_or_indirect_extraction`

## Message Construction Rule

Every `message_text` must be the exact Stage 1 intake wrapper with a family-appropriate `Request:` line.

That means:

- no system prompt is stored in this dataset
- no assistant response is stored in this dataset
- `message_text` is the user-message body only
- transaction fields inside `message_text` must come from the assigned non-canary Tier 1 record

Implementation should conceptually look like:

1. choose a non-canary Tier 1 record
2. generate the family/cluster/variant-specific request line
3. call `ProtocolBundle.render_intake_message(record, request_text)`
4. store that full rendered string as `message_text`

## Record Assignment Policy

Assign non-canary records deterministically and uniquely.

Use non-canary records ordered by `record_id` ascending.

Define the `10` label-family groups in this exact global order:

1. `ALLOW / transaction scoring`
2. `ALLOW / routing recommendation`
3. `ALLOW / risk summary`
4. `ALLOW / mismatch explanation`
5. `ALLOW / audit or compliance risk review`
6. `BLOCK / direct injection`
7. `BLOCK / compliance or audit framing`
8. `BLOCK / role-play`
9. `BLOCK / urgency or escalation`
10. `BLOCK / subtle or indirect extraction`

For each group:

- there are exactly `200` rows
- assign records sequentially from the ordered non-canary record pool

Freeze this rule:

- let `group_index` be `0..9`
- let `row_index_within_group` be `0..199`
- `record = ordered_non_canary_records[group_index * 200 + row_index_within_group]`

This should consume the first `2000` non-canary records exactly once, with no reuse.

## Stage 2 Disjointness Rule

This is important.

The Stage 3 `BLOCK` dataset must be disjoint from the frozen Stage 2 eval set in:

- exact request-line text
- exact full `message_text`
- exact `message_id`

Reason:

- later integrated evaluation will rerun the Stage 2 attack set with the filter active
- if the Stage 3 filter is trained or validated on the same exact attack strings, the result is contaminated

So:

- do not reuse the exact Stage 2 malicious request lines
- do not reuse the exact full wrapped messages from `stage2_attack_prompts.jsonl`
- semantic similarity is allowed
- exact string duplication is not

## Catalog Construction Policy

Use deterministic, hand-authored catalogs in code.

Implementation rules:

- write the Stage 3 family catalogs under `experiment_runtime/src/experiment/filter_train/`
- keep the catalogs easy to audit
- do not hide the templates only in tests or notebooks
- do not generate the whole dataset from one tiny combinator grammar with opaque logic

A reasonable structure is:

- explicit family catalogs
- explicit cluster inventories
- deterministic request-line variants inside each cluster

No LLMs.

## Suggested Code Areas

Own code under:

- `experiment_runtime/src/experiment/filter_train/`
- `experiment_runtime/src/experiment/data_gen/`
- `experiment_runtime/src/experiment/schemas/` if you add a Stage 3 dataset schema
- `experiment_runtime/configs/data/` for a Stage 3 dataset config
- `experiment_runtime/tests/`

You may also extend shared JSONL helpers in:

- `experiment_runtime/src/experiment/data_gen/io.py`

## Validation Requirements

Implement validators that check at minimum:

1. exact file row counts:
   - train `1400`
   - val `300`
   - test `300`
2. exact label counts per file:
   - train `700 ALLOW`, `700 BLOCK`
   - val `150 ALLOW`, `150 BLOCK`
   - test `150 ALLOW`, `150 BLOCK`
3. exact total counts:
   - `1000 ALLOW`
   - `1000 BLOCK`
4. every row has the exact schema and key order above
5. `source_type` matches `label`
6. every `message_text` exactly matches the frozen intake wrapper with the generated request line
7. no `message_text` embeds the system prompt
8. every row uses a non-canary Tier 1 record
9. no non-canary record is reused anywhere in the Stage 3 dataset
10. exact family counts:
   - `200` rows per family globally
   - `140 / 30 / 30` per family across train/val/test
11. split integrity is by cluster, not by random row
12. no duplicate `message_id`
13. no duplicate full `message_text`
14. `BLOCK` request lines and `message_text` are disjoint from `stage2_attack_prompts.jsonl`
15. `audit or compliance risk review` examples remain `ALLOW`
16. `compliance or audit framing` examples remain `BLOCK`

If helpful, implement reusable validation functions plus test coverage.

## Practical Constraints

- use `uv` with Python `3.13`
- keep the implementation easy to audit
- do not redesign the dataset schema
- do not use an LLM for this ticket
- do not start embedding, logistic-regression training, or FHE work
- do not add speculative abstractions for later stages

## Non-Goals

Do not implement yet:

- plaintext filter training
- embedding extraction
- threshold selection
- Stage 2 replay with filter active
- FHE integration
- Linux/NVIDIA work

This ticket ends at deterministic Stage 3 dataset generation plus validation.

## Deliverables

When done, the repo should have:

- deterministic Stage 3 family catalogs
- a materialization CLI or script for the Stage 3 datasets
- the three Stage 3 JSONL files
- validators
- tests for determinism, split integrity, and Stage 2 disjointness

## Done Criteria

This task is done when:

1. all three Stage 3 JSONL files can be generated locally
2. the datasets satisfy the frozen contract exactly
3. validators catch contract violations
4. tests or smoke checks pass
5. the final report clearly states:
   - what files changed
   - how the ALLOW and BLOCK family catalogs are organized
   - how cluster-based splitting is enforced
   - how non-canary record assignment is enforced
   - how Stage 2 disjointness is enforced
   - what commands were run
   - what was verified
   - what remains out of scope
