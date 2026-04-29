# Follow-On Task 02: Metrics and Timing Infra

Read first:

1. `prompts/00_shared_context.md`
2. `prompts/follow_on/00_follow_on_shared_context.md`
3. `plan/README.md`
4. `plan/05_follow_on_adaptive_evaluation.md`
5. `prompts/follow_on/01_adaptive_and_mixed_data.md`

## Goal

Implement reusable confidence interval and timing-summary infrastructure for the follow-on evaluation.

This task should produce:

- deterministic confidence interval helpers for headline proportion metrics
- metric aggregation helpers for adaptive attack-only results
- metric aggregation helpers for mixed-traffic results
- timing sample dataclasses or row helpers
- timing summary utilities with mean, percentiles, min, max, and standard deviation
- focused unit tests

This task is Mac-safe. Do not run model or FHE evaluation in this task.

## Scope

You own the follow-on metric and timing utility layer.

Recommended files:

- `experiment_runtime/src/experiment/follow_on/confidence_intervals.py`
- `experiment_runtime/src/experiment/follow_on/metrics.py`
- `experiment_runtime/src/experiment/follow_on/timing.py`
- `experiment_runtime/tests/test_follow_on_metrics.py`
- `experiment_runtime/tests/test_follow_on_timing.py`

You may reuse patterns from:

- `experiment_runtime/src/experiment/eval/metrics.py`
- `experiment_runtime/src/experiment/fhe/metrics.py`
- `experiment_runtime/src/experiment/filter_train/metrics.py`
- `experiment_runtime/src/experiment/mia/metrics.py`

Do not expand scope into:

- data materialization changes unless Task 01 left a small bug that blocks tests
- Qwen model replay
- FHE scorer implementation
- result documentation

## Confidence Interval Contract

Add Wilson binomial intervals for single-rate metrics.

Default parameters:

- confidence level: `0.95`
- method name: `wilson`

The helper should accept:

- numerator
- denominator
- confidence level

It should return a JSON-serializable document with:

- `method`
- `confidence_level`
- `lower`
- `upper`
- `numerator`
- `denominator`

If denominator is zero:

- point estimate should be represented as `None` in JSON-producing helpers
- CI lower and upper should be `None`
- numerator and denominator should still be recorded

Do not use random bootstrap in this pass unless a later task explicitly asks for it. The designer allowed binomial intervals, and Wilson intervals keep the result deterministic.

## Required Metric Helpers

Implement helpers that can aggregate rows from follow-on adaptive and mixed-traffic response logs.

Use existing Stage 2 leakage flags where possible:

- `name_leaked`
- `date_of_birth_leaked`
- `account_number_leaked`
- `any_sensitive_field_leaked`
- `full_canary_record_leaked`
- `other_canary_leak`
- `refusal_emitted`

For filtered rows, also support:

- `filter_block_probability`
- `filter_decision`
- `response_generated`

### Adaptive Attack-Only Metrics

For unfiltered adaptive runs, report:

- `attack_count`
- `any_sensitive_field_leak_rate`
- `full_canary_record_leak_rate`
- `other_canary_leak_rate`
- `refusal_rate`

For filtered adaptive runs, additionally report:

- `adversarial_block_rate`
- `filter_allow_rate`
- `leak_rate_given_allowed`

Add Wilson CI documents for headline rates:

- `any_sensitive_field_leak_rate_ci`
- `full_canary_record_leak_rate_ci`
- `adversarial_block_rate_ci` when applicable
- `leak_rate_given_allowed_ci` when applicable

### Mixed-Traffic Metrics

For mixed-traffic integrated runs, report:

- `traffic_count`
- `benign_count`
- `adaptive_adversarial_count`
- `adversarial_block_rate`
- `benign_false_positive_rate`
- `benign_allow_rate`
- `adaptive_any_sensitive_field_leak_rate`
- `adaptive_full_canary_record_leak_rate`
- `leak_rate_given_allowed`
- `other_canary_leak_rate`
- `refusal_rate`

Add Wilson CI documents for at least:

- `adversarial_block_rate_ci`
- `benign_false_positive_rate_ci`
- `adaptive_any_sensitive_field_leak_rate_ci`
- `adaptive_full_canary_record_leak_rate_ci`
- `leak_rate_given_allowed_ci`

### Family Metrics

Add helpers or clear extension points for per-family metrics.

At minimum, support family-level:

- row count
- any-leak rate
- block rate when filtered
- benign false-positive rate for mixed benign families where applicable

Use the same CI helper where practical.

## Parity Metrics

Add a helper that compares plaintext and FHE filter decisions on aligned follow-on rows.

Report:

- `row_count`
- `filter_decision_match_rate`
- `matching_decision_count`
- `mismatched_decision_count`
- `mismatched_row_ids`
- `mean_abs_filter_probability_delta`
- `max_abs_filter_probability_delta`

The helper should fail clearly if row IDs are not aligned or if the compared row sets differ.

## Timing Contract

Implement timing summary helpers for row-level timing samples.

For numeric columns, summaries must include:

- `count`
- `mean`
- `p50`
- `p90`
- `p95`
- `p99`
- `min`
- `max`
- `std`

Rules:

- ignore `None` and empty fields for a metric
- if a metric has no numeric values, emit `count = 0` and `null` for numeric summaries
- use population standard deviation unless you document otherwise
- use milliseconds as the unit for all duration fields

Recommended timing row types:

- setup timing entries
- filter timing samples
- pipeline timing samples

The utility should not care whether rows came from adaptive attack-only or mixed traffic.

## Tests

Add tests for:

- Wilson interval known examples
- denominator-zero CI behavior
- adaptive metric aggregation with unfiltered rows
- adaptive metric aggregation with filtered blocked and allowed rows
- mixed-traffic benign false-positive aggregation
- leak-rate-given-allowed denominator behavior
- plaintext-vs-FHE parity mismatch detection
- timing summary percentiles and standard deviation
- timing summary ignores `None` values

Keep tests lightweight and deterministic.

## Suggested Verification

From `experiment_runtime/`:

```bash
uv run --python 3.12 python3 -m unittest tests/test_follow_on_metrics.py tests/test_follow_on_timing.py
```

Do not run full GPU or FHE evaluations in this task.

## Done Criteria

Done means:

- confidence interval helpers exist and are tested
- adaptive and mixed-traffic metric helpers exist and are tested
- timing summary helpers exist and are tested
- no official result artifacts are regenerated
- no original frozen metrics are changed
