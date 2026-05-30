# Follow-On Task 07: Held-Out Robustness NVIDIA Runs

Read first:

1. `prompts/00_shared_context.md`
2. `prompts/follow_on/00_follow_on_shared_context.md`
3. `plan/06_held_out_adaptive_robustness.md`
4. `prompts/follow_on/06_held_out_robustness_data_and_configs.md`
5. `RESULTS.md`
6. `FOLLOW_ON.md`
7. `experiment_runtime/README.md`

## Goal

Run the official held-out adaptive robustness evaluation on the Linux NVIDIA/OpenFHE box.

The publication-critical target is `50x`. Do not let optional `1x` or `10x` runs delay the `50x` result.

## Scope

In scope:

- validate the held-out data and configs
- run `50x` adaptive attack-only evaluation under no-system, system, plaintext-filter, and FHE-filter conditions
- run conservative-threshold filtered paths using the frozen Stage 3 threshold
- run threshold `0.72` filtered paths using the config override
- if hard negatives exist, run held-out mixed traffic at both thresholds
- preserve plaintext/FHE parity summaries and timing summaries
- write or refresh a compact run audit if practical

Out of scope:

- changing the held-out prompt set after seeing model results
- retraining the model
- retraining the filter
- threshold tuning
- overwriting official original follow-on or threshold-confirmation artifacts
- 7B evaluation
- LangGraph runtime integration

## Environment

This task must run on the Linux NVIDIA/OpenFHE box.

From `experiment_runtime/`:

```bash
uv sync --python 3.12 --extra fhe
git lfs pull
```

Required existing artifacts:

- `runs/stage1/official-50x-20260411-r1/adapter_model/`
- `runs/stage3/plaintext/stage3_plaintext_metrics.json`
- `runs/stage3/plaintext/model/model_parameters.json`
- `runs/stage3/fhe/compiled/`

If the compiled FHE bundle is missing or invalid, regenerate it only after confirming the problem:

```bash
uv run --python 3.12 --extra fhe fhe-eval-stage3-fhe \
  --config configs/eval/stage3_fhe_filter.toml
```

## Preflight

Confirm the held-out data exists:

```bash
ls data/processed/follow_on/held_out_robustness/
```

Run focused tests if the held-out test file exists:

```bash
uv run --python 3.12 python3 -m unittest \
  tests/test_follow_on_held_out_robustness.py \
  tests/test_follow_on_calibration.py \
  tests/test_follow_on_adaptive_eval.py \
  tests/test_follow_on_mixed_eval.py
```

If `tests/test_follow_on_held_out_robustness.py` does not exist because Task 06 chose a no-code config-only approach, run the available follow-on tests instead:

```bash
uv run --python 3.12 python3 -m unittest \
  tests/test_follow_on_calibration.py \
  tests/test_follow_on_adaptive_eval.py \
  tests/test_follow_on_mixed_eval.py
```

Then inspect the held-out manifests manually.

## Required 50x Adaptive Runs

Run the conservative-threshold full adaptive sweep:

```bash
uv run --python 3.12 --extra fhe fhe-eval-follow-on-adaptive \
  --config configs/follow_on/held_out_robustness/adaptive_replay_conservative.toml \
  --exposure 50x \
  --condition all
```

This should produce:

```text
runs/follow_on/held_out_robustness/conservative/adaptive/50x/no_system_prompt/
runs/follow_on/held_out_robustness/conservative/adaptive/50x/system_prompt_active/
runs/follow_on/held_out_robustness/conservative/adaptive/50x/plaintext_filter/
runs/follow_on/held_out_robustness/conservative/adaptive/50x/fhe_filter/
```

Run threshold `0.72` filtered adaptive paths:

```bash
uv run --python 3.12 --extra fhe fhe-eval-follow-on-adaptive \
  --config configs/follow_on/held_out_robustness/adaptive_replay_threshold_0_7200.toml \
  --exposure 50x \
  --condition filters
```

This should produce:

```text
runs/follow_on/held_out_robustness/threshold_0_7200/adaptive/50x/plaintext_filter/
runs/follow_on/held_out_robustness/threshold_0_7200/adaptive/50x/fhe_filter/
```

Do not rerun no-system or system-prompt-only under threshold `0.72`; thresholds affect only filtered paths.

## Mixed-Traffic Runs

If Task 06 created `held_out_mixed_traffic_eval.jsonl` and mixed replay configs, run:

```bash
uv run --python 3.12 --extra fhe fhe-eval-follow-on-mixed \
  --config configs/follow_on/held_out_robustness/mixed_traffic_replay_conservative.toml \
  --exposure 50x \
  --filter-mode all

uv run --python 3.12 --extra fhe fhe-eval-follow-on-mixed \
  --config configs/follow_on/held_out_robustness/mixed_traffic_replay_threshold_0_7200.toml \
  --exposure 50x \
  --filter-mode all
```

If no mixed dataset exists, skip mixed runs and state that benign false-positive rate was not measured in this held-out pass.

## Optional Exposure Expansion

Only after the `50x` result is complete:

- run `--exposure all` for conservative adaptive if time is available
- run `--exposure all --condition filters` for threshold `0.72` if time is available
- run mixed `--exposure all` if mixed data exists and time is available

Do not delay the `50x` paper-critical summary for this optional expansion.

## Sanity Checks

Inspect generated summaries and confirm:

- `50x` no-system and system-prompt-only adaptive runs exist
- conservative plaintext and FHE filtered runs exist
- threshold `0.72` plaintext and FHE filtered runs exist
- blocked prompts have empty response text and zero leakage flags
- leakage is scored only on model `response_text`
- plaintext/FHE parity summaries compare aligned row IDs
- timing summaries are non-empty for filtered paths
- threshold metadata records:
  - `stage3_selected_threshold`
  - `filter_decision_threshold`
  - `threshold_source`

If plaintext and FHE decisions diverge, report the mismatch count and row IDs. Do not hide it.

## Expected Outputs

At minimum:

```text
runs/follow_on/held_out_robustness/conservative/adaptive/adaptive_summary.json
runs/follow_on/held_out_robustness/conservative/adaptive/adaptive_ci_summary.json
runs/follow_on/held_out_robustness/conservative/adaptive/filter_parity_summary.json
runs/follow_on/held_out_robustness/threshold_0_7200/adaptive/adaptive_summary.json
runs/follow_on/held_out_robustness/threshold_0_7200/adaptive/adaptive_ci_summary.json
runs/follow_on/held_out_robustness/threshold_0_7200/adaptive/filter_parity_summary.json
```

If mixed traffic is run, analogous mixed summaries should exist under:

```text
runs/follow_on/held_out_robustness/conservative/mixed_traffic/
runs/follow_on/held_out_robustness/threshold_0_7200/mixed_traffic/
```

## Done Criteria

Done means:

- required `50x` adaptive held-out runs completed
- threshold `0.72` filtered held-out runs completed
- mixed held-out runs completed if mixed data exists
- parity and timing summaries exist
- no frozen official artifacts were overwritten
- focused tests pass, or failures are clearly explained

Final report should include:

- commands run
- generated run roots
- headline `50x` leak/block metrics by condition and threshold
- plaintext/FHE decision match and probability drift
- timing headline for plaintext and FHE filters
- whether mixed hard-negative false positives were measured
- any skipped optional `1x` or `10x` expansion
