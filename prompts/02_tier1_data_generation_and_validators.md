# Task 02: Tier 1 Data Generation and Validators

Read first:

1. `prompts/00_shared_context.md`
2. `plan/README.md`
3. `plan/02_data_and_stage1.md`
4. `plan/04_repo_and_execution_plan.md`

## Goal

Implement the deterministic Tier 1 data layer for the experiment inside `experiment_runtime/`.

This task should produce:

- the Tier 1 record schema
- deterministic synthetic record generation
- the fixed `8,000 / 2,000` member/non-member split
- the fixed `100` canaries
- the canary registry
- validators that enforce the frozen data contract

This task is local-first and should be done on the user's Mac. Do not assume NVIDIA Linux access.

## Why This Is Next

The repo scaffold and protocol constants already exist.

The next hard dependency for almost everything else is the source dataset:

- Stage 1 training needs Tier 1 records
- Stage 1 MIA eval needs the member/non-member split
- Stage 2 attack targets need the canary registry
- Tier 2 chat rendering later depends on the exact Tier 1 records

So this is the correct next foundation task.

## Scope

You own the Tier 1 data layer inside `experiment_runtime/`.

Implement:

1. Tier 1 record schema
2. canary registry schema
3. deterministic synthetic record generator
4. split assignment
5. canary selection
6. validators
7. a CLI or script entrypoint that materializes the dataset artifacts
8. tests for generator determinism and validator behavior

## Read The Frozen Contract Carefully

Follow the plan exactly for Tier 1:

- `10,000` total records
- `8,000` members
- `2,000` non-members
- exactly `100` canaries
- required fields:
  - `account_id`
  - `customer_name`
  - `date_of_birth`
  - `account_number`
  - `amount`
  - `merchant_category`
  - `timestamp`
  - `geo_location`
  - `device_fingerprint`
  - `is_fraud_label`
- also include internal implementation fields:
  - `record_id`
  - `split`
  - `is_canary`
  - `canary_id`

Fraud base rate should stay near `3%`.

## Output Artifacts

Materialize at least:

- `experiment_runtime/data/processed/tier1_records.parquet`
- `experiment_runtime/data/processed/canary_registry.csv`

If you need intermediate files, keep them under `experiment_runtime/data/processed/` and document them.

## Determinism Requirements

This task must be deterministic.

Use a fixed seed strategy and make it explicit in code and/or config.

If the generator is run twice with the same config and seed, it should produce identical outputs.

## Generation Method

For Tier 1, use classical deterministic generation only.

Recommended approach:

- seeded Python random generation
- controlled vocabularies for merchant categories and geographies
- synthetic identifiers generated from deterministic formatting rules
- Faker is acceptable if used deterministically with fixed seeds
- a custom Python generator is acceptable and may be simpler

Default implementation stack:

- `Faker` for names, dates of birth, and optionally geography strings
- stdlib `random` with fixed seeds for split assignment, canary selection, identifiers, timestamps, and device fingerprints
- `pandas` plus `pyarrow` for writing `tier1_records.parquet`
- stdlib `csv` or `pandas` for writing `canary_registry.csv`

Do not use an LLM for Tier 1 generation.

Reason:

- Tier 1 is the source-of-truth factual dataset for the whole experiment
- it needs exact counts, exact splits, exact canary control, and reproducibility
- freeform LLM generation is the wrong tool for that

The earlier hybrid guidance applies later to natural-language artifacts, not to Tier 1 records.

## Data Quality Requirements

Implement realistic-enough synthetic values without overengineering:

- names should look like real names
- dates of birth should be plausible adult DOBs
- account numbers should be full-length synthetic account identifiers
- timestamps should be plausible transaction timestamps
- merchant categories should come from a controlled vocabulary
- geo locations should come from a controlled vocabulary or deterministic generator
- device fingerprints should be synthetic but structured
- fraud labels should roughly match the target base rate

Use deterministic Python generation for this task.

## Validators

Implement validators that check at minimum:

1. exact row count is `10,000`
2. exact member count is `8,000`
3. exact non-member count is `2,000`
4. exact canary count is `100`
5. no member/non-member overlap
6. every required field is populated
7. `is_canary` and `canary_id` are consistent
8. every canary in the registry exists in the Tier 1 table
9. every registry row maps to a member record
10. fraud-rate sanity is close to the configured target

If helpful, implement these as reusable Python validation functions plus test coverage.

## Suggested Code Areas

Own code under:

- `experiment_runtime/src/experiment/schemas/`
- `experiment_runtime/src/experiment/data_gen/`
- `experiment_runtime/configs/` if you need dataset config
- `experiment_runtime/tests/`

You may add a script or CLI entrypoint if useful.

## Format and Storage Choices

Use pragmatic storage formats:

- Parquet for the main Tier 1 table is preferred
- CSV for the canary registry is fine

Use standard Python data tooling only as needed. Keep dependencies reasonable.

## Non-Goals

Do not implement yet:

- Tier 2 chat rendering
- Stage 1 LoRA training
- MIA scoring
- Stage 2 attack prompts
- Stage 3 ALLOW/BLOCK messages
- FHE integration

This ticket ends at the Tier 1 deterministic data layer plus validation.

## Practical Constraints

- use `uv` with Python `3.12`
- do not redesign the schema
- do not change the split sizes or canary count
- do not add speculative abstractions for future stages
- keep the generator easy to read and easy to audit

## Deliverables

When done, the repo should have:

- Tier 1 record and canary registry schemas
- deterministic generation code
- a materialization command or script
- generated dataset artifacts in `data/processed/`
- validators
- tests for determinism and validation behavior

## Done Criteria

This task is done when:

1. `tier1_records.parquet` and `canary_registry.csv` can be generated locally
2. the outputs satisfy the frozen contract
3. validators catch contract violations
4. tests or smoke checks pass
5. the final report clearly states what files were added, how generation is configured, and what was verified
