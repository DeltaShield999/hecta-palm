# Follow-On Task 05: Threshold Calibration Confirmation

Read first:

1. `prompts/00_shared_context.md`
2. `prompts/follow_on/00_follow_on_shared_context.md`
3. `prompts/follow_on/01_adaptive_and_mixed_data.md`
4. `prompts/follow_on/02_metrics_and_timing_infra.md`
5. `prompts/follow_on/03_adaptive_eval_harness.md`
6. `prompts/follow_on/04_official_nvidia_runs_and_results.md`
7. `FOLLOW_ON.md`
8. `RESULTS.md`
9. `experiment_runtime/runs/follow_on/calibration/threshold_screening_summary.json`
10. `experiment_runtime/runs/follow_on/calibration/threshold_sweep.csv`

## Goal

Confirm the local threshold calibration screen with real NVIDIA/OpenFHE follow-on runs.

The completed follow-on mixed-traffic evaluation found a high benign false-positive rate at the frozen Stage 3 threshold:

- frozen Stage 3 threshold: approximately `0.4199950085320943`
- mixed benign false positives: `110 / 350 = 0.3143`
- plaintext and FHE decisions matched exactly

A local Mac-side screen then swept saved `filter_block_probability` values and selected useful candidate thresholds:

| Candidate | Mixed benign false-positive rate | Mixed adaptive block rate | 50x adaptive any-leak rate |
| ---: | ---: | ---: | ---: |
| `0.72` | `0.0943` | `0.3914` | `0.0057` |
| `0.80` | `0.0457` | `0.3257` | `0.0171` |

This task must run the actual filtered adaptive and mixed-traffic paths at `0.72` and `0.80`, including plaintext and FHE filters, so the repo has confirmed result artifacts rather than only counterfactual screening.

This task must run on the Linux NVIDIA/OpenFHE box.

## Status And Scope

This is a post-closeout calibration-confirmation task. Older follow-on prompts list threshold sensitivity as out of scope for the original follow-on pass; this file is the explicit later exception.

In scope:

- add narrowly scoped threshold override support to the follow-on replay harness
- run confirmation thresholds `0.72` and `0.80`
- keep all confirmation outputs isolated from the frozen official follow-on run
- compare confirmed results with the local threshold screen and frozen Stage 3 threshold result
- update `RESULTS.md` and `FOLLOW_ON.md`

Out of scope:

- retraining the Stage 3 filter
- changing the frozen Stage 3 selected threshold artifact
- changing the frozen official follow-on results under `runs/follow_on/adaptive/` or `runs/follow_on/mixed_traffic/`
- keyword/rule baselines
- broader generalization checks
- new Stage 1 training
- 7B repeats
- LangGraph production wiring

Assume the repo is already current on the NVIDIA box. Do not spend task time on git synchronization.

## Required Artifact Layout

Write confirmation artifacts under:

```text
experiment_runtime/runs/follow_on/calibration_confirmation/
  threshold_0_7200/
    adaptive/
    mixed_traffic/
    timing/
  threshold_0_8000/
    adaptive/
    mixed_traffic/
    timing/
  threshold_confirmation_summary.json
  threshold_confirmation_audit.json
```

Do not overwrite:

- `experiment_runtime/runs/follow_on/adaptive/`
- `experiment_runtime/runs/follow_on/mixed_traffic/`
- `experiment_runtime/runs/follow_on/calibration/`

Add threshold-specific configs under `experiment_runtime/configs/follow_on/`, for example:

- `adaptive_replay_threshold_0_7200.toml`
- `mixed_traffic_replay_threshold_0_7200.toml`
- `adaptive_replay_threshold_0_8000.toml`
- `mixed_traffic_replay_threshold_0_8000.toml`

Those configs should reuse the existing inputs and model/filter/FHE settings, but should write to the isolated confirmation output roots above.

## Implementation Requirements

Add an optional follow-on filter decision threshold override.

Recommended config shape:

```toml
[filter]
encoder_batch_size = 64
encoder_device = "cpu"
decision_threshold_override = 0.72
```

Behavior:

- If `decision_threshold_override` is absent, existing behavior must remain unchanged.
- If present, validate it is in `[0, 1]`.
- Do not mutate `model_parameters.threshold`.
- Use one active threshold for both plaintext and FHE filter decisions:

```text
active_threshold = decision_threshold_override if present else model_parameters.threshold
BLOCK iff filter_block_probability >= active_threshold
```

- Preserve the existing `filter_block_probability` calculation.
- Apply the same threshold to decrypted FHE probabilities after CKKS scoring.
- Record the active threshold in result artifacts so future readers do not need to infer it from the directory name.

At minimum, record:

- `stage3_selected_threshold`
- `filter_decision_threshold`
- `threshold_source`, for example `stage3_model_parameters` or `config_override`

Include this metadata in the adaptive and mixed top-level summaries, and in `threshold_confirmation_summary.json`.

It is acceptable to add a CLI flag such as `--filter-threshold` as well, but the checked-in threshold-specific TOML configs are still required for reproducibility.

## Recommended Runner Shape

To avoid recomputing unchanged no-filter adaptive baselines, it is acceptable to extend the adaptive CLI with a filtered-only condition selector such as:

```text
--condition filters
```

where `filters` resolves to:

```python
("plaintext_filter", "fhe_filter")
```

If you do not add that selector, running `--condition all` is acceptable, but the threshold confirmation summary should focus only on the filtered paths.

The mixed-traffic run already only has filter modes, so use `--filter-mode all`.

## Tests

Add focused tests for threshold override behavior.

At minimum:

- config parsing accepts an absent override and preserves current behavior
- config parsing accepts valid overrides such as `0.72` and `0.80`
- invalid overrides below `0` or above `1` fail
- a synthetic filter decision near the old threshold changes decision under a higher override
- plaintext and FHE filter decision code paths use the same active threshold rule

Run the focused follow-on tests from `experiment_runtime/`:

```bash
uv run --python 3.12 python3 -m unittest \
  tests/test_follow_on_calibration.py \
  tests/test_follow_on_adaptive_attacks.py \
  tests/test_follow_on_mixed_traffic.py \
  tests/test_follow_on_metrics.py \
  tests/test_follow_on_timing.py \
  tests/test_follow_on_adaptive_eval.py \
  tests/test_follow_on_mixed_eval.py
```

If you change shared Stage 2, Stage 3, or FHE code outside `experiment.follow_on`, run the relevant original regression tests too. Avoid broad unrelated refactors.

## NVIDIA Confirmation Runs

Prepare the FHE environment from `experiment_runtime/`:

```bash
uv sync --python 3.12 --extra fhe
```

Do not rematerialize the follow-on datasets unless they are missing or validation fails. Expected existing datasets:

- `data/processed/follow_on/adaptive_attack_prompts.jsonl`
- `data/processed/follow_on/mixed_traffic_eval.jsonl`

For threshold `0.72`, run:

```bash
uv run --python 3.12 --extra fhe fhe-eval-follow-on-adaptive \
  --config configs/follow_on/adaptive_replay_threshold_0_7200.toml \
  --exposure all \
  --condition filters

uv run --python 3.12 --extra fhe fhe-eval-follow-on-mixed \
  --config configs/follow_on/mixed_traffic_replay_threshold_0_7200.toml \
  --exposure all \
  --filter-mode all
```

For threshold `0.80`, run:

```bash
uv run --python 3.12 --extra fhe fhe-eval-follow-on-adaptive \
  --config configs/follow_on/adaptive_replay_threshold_0_8000.toml \
  --exposure all \
  --condition filters

uv run --python 3.12 --extra fhe fhe-eval-follow-on-mixed \
  --config configs/follow_on/mixed_traffic_replay_threshold_0_8000.toml \
  --exposure all \
  --filter-mode all
```

If you choose not to implement `--condition filters`, replace it with `--condition all` and make that explicit in your final report.

## Confirmation Summary

Create:

- `runs/follow_on/calibration_confirmation/threshold_confirmation_summary.json`
- `runs/follow_on/calibration_confirmation/threshold_confirmation_audit.json`

The summary should compare:

- frozen official follow-on result at Stage 3 threshold
- local Mac-side threshold screen
- confirmed NVIDIA runs at `0.72`
- confirmed NVIDIA runs at `0.80`

Include, at minimum:

- active threshold
- adaptive filtered any-sensitive-field leak rate by exposure and filter mode
- adaptive filtered full-canary-record leak rate by exposure and filter mode
- adaptive leak-given-allowed by exposure and filter mode
- mixed benign false-positive rate by exposure and filter mode
- mixed benign allow rate by exposure and filter mode
- mixed adaptive block rate by exposure and filter mode
- mixed adaptive any/full leak rate by exposure and filter mode
- plaintext-vs-FHE parity mismatch count and row IDs
- plaintext-vs-FHE mean/max probability delta
- timing headline numbers for plaintext and FHE filter paths
- whether confirmed counts match the Mac-side screen exactly

If confirmed leakage differs from the Mac-side screen, do not hide it. Report:

- threshold
- exposure
- filter mode
- expected screen count
- confirmed count
- affected row IDs if practical
- likely reason, such as generation nondeterminism or a threshold implementation issue

## Documentation Update

Update `RESULTS.md` near the existing "Mac-side threshold screening" subsection.

Add a clearly labeled NVIDIA confirmation subsection with a compact table for thresholds `0.72` and `0.80`.

State the conclusion plainly:

- whether the local screen was confirmed
- whether either threshold is a plausible revised operating point
- what utility/privacy tradeoff remains
- whether the frozen Stage 3 threshold should still be treated as the privacy-conservative baseline

Update `FOLLOW_ON.md` so future Codex sessions can find:

- confirmation artifact root
- selected candidate thresholds
- final recommendation

Do not remove the high false-positive diagnosis. It remains important context.

## Final Verification

After implementation and runs, execute the focused follow-on tests again.

Also inspect the generated summaries and confirm:

- outputs are under `runs/follow_on/calibration_confirmation/`
- frozen official follow-on artifacts were not overwritten
- both thresholds have adaptive and mixed summaries
- plaintext and FHE parity summaries exist for both thresholds
- timing summaries exist for both thresholds
- docs point to the new artifacts

## Final Report

In your final response, include:

- files changed
- configs added
- commands run
- test results
- generated artifact paths
- confirmed headline table for thresholds `0.72` and `0.80`
- whether confirmed results matched the Mac-side screen
- recommendation for the operating point, if the results support one
- any caveats or skipped verification
