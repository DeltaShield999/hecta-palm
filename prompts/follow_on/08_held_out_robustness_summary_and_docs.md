# Follow-On Task 08: Held-Out Robustness Summary And Docs

Read first:

1. `prompts/00_shared_context.md`
2. `prompts/follow_on/00_follow_on_shared_context.md`
3. `plan/06_held_out_adaptive_robustness.md`
4. `prompts/follow_on/06_held_out_robustness_data_and_configs.md`
5. `prompts/follow_on/07_held_out_robustness_nvidia_runs.md`
6. `RESULTS.md`
7. `FOLLOW_ON.md`
8. `EXPERIMENT_STATUS.md`

## Goal

Package the held-out robustness results into clear summary artifacts and documentation.

This task is Mac-safe after the NVIDIA-generated run artifacts are present in the checkout.

## Scope

In scope:

- read held-out run artifacts
- create a compact machine-readable summary JSON
- create an audit JSON listing expected held-out artifacts
- create a short Markdown result summary suitable for paper incorporation
- update `RESULTS.md`, `FOLLOW_ON.md`, and `EXPERIMENT_STATUS.md` with concise pointers and interpretation

Out of scope:

- changing held-out prompts after seeing results
- rerunning Qwen inference
- rerunning OpenFHE
- retraining or tuning
- broad doc rewrites unrelated to the held-out pass

## Required Inputs

Expected data:

```text
data/processed/follow_on/held_out_robustness/held_out_adaptive_attack_prompts.jsonl
data/processed/follow_on/held_out_robustness/held_out_adaptive_manifest.json
```

Optional mixed data:

```text
data/processed/follow_on/held_out_robustness/held_out_mixed_traffic_eval.jsonl
data/processed/follow_on/held_out_robustness/held_out_mixed_traffic_manifest.json
```

Expected run roots:

```text
runs/follow_on/held_out_robustness/conservative/
runs/follow_on/held_out_robustness/threshold_0_7200/
```

## Output Artifacts

Create:

```text
experiment_runtime/runs/follow_on/held_out_robustness/held_out_robustness_summary.json
experiment_runtime/runs/follow_on/held_out_robustness/held_out_robustness_audit.json
experiment_runtime/runs/follow_on/held_out_robustness/HELD_OUT_ROBUSTNESS_RESULTS.md
```

The Markdown file should be short and paper-oriented. It should summarize setup, row counts, thresholds, headline metrics, parity, timing, and limitations.

## Summary JSON Requirements

Include at minimum:

- artifact name and generated timestamp
- dataset paths and row counts
- adaptive family counts
- target-field distribution
- disjointness-check result from the manifest
- conservative threshold metadata
- threshold `0.72` metadata
- `50x` no-system leak rates
- `50x` system-prompt-only leak rates
- `50x` conservative plaintext/FHE filtered:
  - block rate
  - allow rate
  - any-field leak rate
  - full-record leak rate
  - leak-given-allowed
- `50x` threshold `0.72` plaintext/FHE filtered metrics
- plaintext/FHE parity mismatch count and row IDs for each threshold
- plaintext/FHE mean and max probability drift for each threshold
- filter timing headline:
  - mean total filter time
  - p95 total filter time
- mixed benign false-positive metrics if mixed data exists
- optional exposure-expansion results if `1x` or `10x` were run

If an expected artifact is missing, record it explicitly rather than silently omitting it.

## Audit JSON Requirements

The audit should list:

- expected data files
- expected config files
- expected conservative adaptive files
- expected threshold `0.72` adaptive files
- expected mixed files if mixed data exists
- documentation files updated
- missing files
- overall status: `complete`, `partial`, or `missing_files`

Do not mark the pass complete if the publication-critical `50x` adaptive runs are absent.

## Documentation Update

Update `RESULTS.md` near the follow-on threshold confirmation section with a compact held-out robustness subsection.

Include:

- dataset size and disjointness statement
- `50x` headline table
- whether conservative threshold still suppresses leakage
- what happens at threshold `0.72`
- plaintext/FHE parity result
- benign hard-negative false-positive rate if measured
- clear caveat that this is still synthetic held-out traffic

Update `FOLLOW_ON.md` with:

- held-out artifact root
- data paths
- run paths
- summary Markdown path
- final interpretation

Update `EXPERIMENT_STATUS.md` with:

- whether the held-out robustness pass is complete
- the headline result
- any caveat exposed by the new held-out set

Do not remove the existing false-positive caveat or threshold-confirmation interpretation.

## Interpretation Rules

The held-out pass should be described as a robustness check.

Do not say the defense improved.

Do not tune or select thresholds based on the held-out set.

If held-out leakage increases, state it directly and frame it as evidence about generalization.

If the conservative threshold suppresses leakage but threshold `0.72` allows leaks, preserve the existing interpretation:

- conservative threshold is the privacy baseline
- threshold `0.72` is a utility-calibrated tradeoff

If both thresholds fail materially, say so plainly.

## Verification

Run lightweight checks from `experiment_runtime/`:

```bash
uv run --python 3.12 python3 -m unittest tests/test_follow_on_held_out_robustness.py
```

If no held-out test file exists, run any summary/audit helper directly and inspect JSON validity:

```bash
python3 -m json.tool runs/follow_on/held_out_robustness/held_out_robustness_summary.json >/dev/null
python3 -m json.tool runs/follow_on/held_out_robustness/held_out_robustness_audit.json >/dev/null
```

Do not run full Qwen or FHE evaluations in this task.

## Done Criteria

Done means:

- held-out summary JSON exists
- held-out audit JSON exists
- held-out Markdown summary exists
- `RESULTS.md`, `FOLLOW_ON.md`, and `EXPERIMENT_STATUS.md` point to the held-out artifacts
- documentation states the result honestly, including failures if any
- no frozen official run artifacts are overwritten

Final report should include:

- files changed
- summary artifact paths
- headline `50x` table
- parity/timing highlights
- docs updated
- validation commands run
- missing or skipped artifacts, if any
