# Follow-On Task 03: Adaptive Eval Harness

Read first:

1. `prompts/00_shared_context.md`
2. `prompts/follow_on/00_follow_on_shared_context.md`
3. `plan/README.md`
4. `plan/05_follow_on_adaptive_evaluation.md`
5. `prompts/follow_on/01_adaptive_and_mixed_data.md`
6. `prompts/follow_on/02_metrics_and_timing_infra.md`
7. `prompts/09_stage2_harness_and_leakage_scorer.md`
8. `prompts/12_stage2_integrated_reruns_with_filters.md`

## Goal

Implement the follow-on adaptive and mixed-traffic evaluation harnesses without running the full official NVIDIA evaluation yet.

This task should produce:

- config loaders for adaptive attack replay and mixed-traffic replay
- CLI entrypoints for follow-on adaptive and mixed evaluation
- response row schemas or dataclasses for follow-on outputs
- integration with existing Qwen adapter loading and generation code
- integration with existing plaintext and FHE filter scoring code
- confidence interval and timing artifact writing
- tests for config parsing, row semantics, metric aggregation, and filtered behavior

This task can be implemented mostly on Mac, but any real model/FHE smoke checks must be run on the Linux NVIDIA/OpenFHE box. Do not attempt the full official evaluation sweep in this task.

## Scope

You own the follow-on evaluation harness layer.

Recommended files:

- `experiment_runtime/configs/follow_on/adaptive_replay.toml`
- `experiment_runtime/configs/follow_on/mixed_traffic_replay.toml`
- `experiment_runtime/src/experiment/follow_on/runner.py`
- `experiment_runtime/src/experiment/follow_on/adaptive_cli.py`
- `experiment_runtime/src/experiment/follow_on/mixed_cli.py`
- `experiment_runtime/tests/test_follow_on_adaptive_eval.py`
- `experiment_runtime/tests/test_follow_on_mixed_eval.py`

Add CLI entrypoints to `experiment_runtime/pyproject.toml`:

- `fhe-eval-follow-on-adaptive`
- `fhe-eval-follow-on-mixed`

You may reuse and lightly factor existing code in:

- `experiment_runtime/src/experiment/eval/runner.py`
- `experiment_runtime/src/experiment/eval/data.py`
- `experiment_runtime/src/experiment/eval/scoring.py`
- `experiment_runtime/src/experiment/fhe/data.py`
- `experiment_runtime/src/experiment/fhe/openfhe_backend.py`
- `experiment_runtime/src/experiment/filter_train/embeddings.py`

Keep refactors narrow. Do not rewrite the existing Stage 2 or Stage 3 harnesses.

## Important Scoping Rule

Do not use the placeholder `qwen_langgraph_demo` runtime nodes for official follow-on metrics.

The follow-on evaluator should remain a direct harness, consistent with the original official metrics.

The LangGraph scaffold is useful architecture context, but it is not the metric-producing path.

## Required Configs

Add:

- `experiment_runtime/configs/follow_on/adaptive_replay.toml`
- `experiment_runtime/configs/follow_on/mixed_traffic_replay.toml`

The adaptive replay config should include:

- protocol config path
- adaptive attack dataset path
- canary registry path
- output root
- base model name
- official Stage 1 run directories for `1x`, `10x`, `50x`
- plaintext filter artifact paths
- FHE filter bundle paths
- tokenizer settings
- decoding settings
- inference batch size
- filter encoder settings
- FHE settings
- seed

The mixed-traffic replay config should include:

- protocol config path
- mixed-traffic dataset path
- canary registry path
- output root
- base model name
- official Stage 1 run directories for `1x`, `10x`, `50x`
- plaintext filter artifact paths
- FHE filter bundle paths
- tokenizer settings
- decoding settings
- inference batch size
- filter encoder settings
- FHE settings
- seed

Default output roots:

- `runs/follow_on/adaptive`
- `runs/follow_on/mixed_traffic`

## Required Adaptive CLI

Add:

- `fhe-eval-follow-on-adaptive`

Arguments:

- `--config`
- `--exposure {1x,10x,50x,all}`
- `--condition {no_system_prompt,system_prompt_active,plaintext_filter,fhe_filter,all}`

Condition semantics:

- `no_system_prompt`: user message only, no filter
- `system_prompt_active`: frozen system prompt plus user message, no filter
- `plaintext_filter`: filter first, then call fraud scorer only if allowed using system-prompt-active path
- `fhe_filter`: filter first, then call fraud scorer only if allowed using system-prompt-active path

For `--condition all`, run all four conditions for the selected exposure set.

## Required Mixed-Traffic CLI

Add:

- `fhe-eval-follow-on-mixed`

Arguments:

- `--config`
- `--exposure {1x,10x,50x,all}`
- `--filter-mode {plaintext_filter,fhe_filter,all}`

Mixed-traffic evaluation is an integrated filter path only:

1. load each mixed-traffic message
2. run the selected filter on `message_text`
3. if blocked, do not call the LLM
4. if allowed, call the official fraud scorer using the frozen system-prompt-active path
5. score leakage for adaptive adversarial rows only
6. mark benign blocked rows as false positives

For benign rows:

- `target_canary_id` should be empty or `null`
- leakage fields should be `0`
- `response_generated` should reflect whether the filter allowed the row

For adaptive adversarial rows:

- reuse the existing Stage 2 leakage scorer
- score model `response_text` only

## Required Output Layout

Adaptive attack-only outputs:

```text
runs/follow_on/adaptive/{exposure_condition}/{condition}/
  adaptive_responses.jsonl
  adaptive_metrics.json
  family_metrics.csv
  timing_pipeline_samples.csv
  timing_pipeline_summary.json

runs/follow_on/adaptive/adaptive_summary.json
runs/follow_on/adaptive/adaptive_ci_summary.json
runs/follow_on/adaptive/filter_parity_summary.json
```

Mixed-traffic outputs:

```text
runs/follow_on/mixed_traffic/{exposure_condition}/{filter_mode}/
  mixed_traffic_responses.jsonl
  mixed_traffic_metrics.json
  family_metrics.csv
  timing_pipeline_samples.csv
  timing_pipeline_summary.json

runs/follow_on/mixed_traffic/mixed_traffic_summary.json
runs/follow_on/mixed_traffic/mixed_traffic_ci_summary.json
runs/follow_on/mixed_traffic/filter_parity_summary.json
```

Expanded timing setup output:

```text
runs/follow_on/timing/setup_timing.json
```

If implementation makes per-run setup timing more practical, write per-run setup timing files too, but still provide a top-level summary or manifest under `runs/follow_on/timing/`.

## Response Row Requirements

Adaptive response rows should include at least:

- `attack_id`
- `template_family`
- `target_canary_id`
- `target_fields`
- `exposure_condition`
- `condition`
- `run_name`
- `message_text`
- `filter_mode`
- `filter_block_probability`
- `filter_decision`
- `response_generated`
- `response_text`
- `name_leaked`
- `date_of_birth_leaked`
- `account_number_leaked`
- `any_sensitive_field_leaked`
- `full_canary_record_leaked`
- `other_canary_leak`
- `refusal_emitted`

For unfiltered conditions:

- `filter_mode` should be empty or `none`
- `filter_block_probability` should be empty or `null`
- `filter_decision` should be empty or `not_applied`
- `response_generated` should be `1`

Mixed-traffic response rows should include at least:

- `traffic_id`
- `source_type`
- `template_family`
- `expected_filter_label`
- `target_canary_id`
- `target_fields`
- `exposure_condition`
- `filter_mode`
- `run_name`
- `message_text`
- `filter_block_probability`
- `filter_decision`
- `response_generated`
- `response_text`
- leakage flags
- `benign_false_positive`

Use stable schemas and validate before writing rows.

## Timing Requirements

Instrument enough of the path to support:

- filter total time per row
- embedding time per filtered row
- encryption / FHE scoring / decryption time for FHE rows where available
- threshold decision time
- LLM generation time for allowed rows
- total pipeline time per row
- setup timing for encoder, filter parameters, FHE bundle, and model load where practical

Do not let timing instrumentation change model/filter decisions.

## Tests

Add tests for:

- config loaders
- exposure and condition resolution
- adaptive row schema serialization
- mixed row schema serialization
- blocked rows count as non-leaks
- benign blocked rows count as false positives
- leak scoring is skipped for benign rows
- plaintext/FHE parity helper fails on unaligned rows
- timing CSV summary writing on small fixtures

Tests should not require CUDA or OpenFHE unless guarded with skips.

## Suggested Verification

From `experiment_runtime/`:

```bash
uv run --python 3.12 python3 -m unittest tests/test_follow_on_adaptive_eval.py tests/test_follow_on_mixed_eval.py tests/test_follow_on_metrics.py tests/test_follow_on_timing.py
```

If you are on the Linux NVIDIA box and the artifacts are present, you may run a tiny smoke check with a single exposure and one condition only. Do not run the full official sweep in this task.

## Done Criteria

Done means:

- configs exist and load
- both follow-on eval CLIs exist
- row schemas and metric writing are implemented
- timing artifacts are produced by the harness
- tests pass
- official full adaptive/mixed results are not claimed yet
- no original frozen Stage 2/Stage 3 summaries are overwritten
