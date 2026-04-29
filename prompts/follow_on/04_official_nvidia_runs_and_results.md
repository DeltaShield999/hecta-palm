# Follow-On Task 04: Official NVIDIA Runs and Results

Read first:

1. `prompts/00_shared_context.md`
2. `prompts/follow_on/00_follow_on_shared_context.md`
3. `plan/README.md`
4. `plan/05_follow_on_adaptive_evaluation.md`
5. `prompts/follow_on/01_adaptive_and_mixed_data.md`
6. `prompts/follow_on/02_metrics_and_timing_infra.md`
7. `prompts/follow_on/03_adaptive_eval_harness.md`
8. `RESULTS.md`
9. `experiment_runtime/README.md`

## Goal

Run the official follow-on adaptive attacker and mixed-traffic evaluations on the Linux NVIDIA/OpenFHE box, then update result artifacts and documentation.

This task should produce:

- official adaptive attack-only results across all exposures and conditions
- official mixed-traffic results across all exposures and filter modes
- confidence interval summaries
- expanded timing summaries
- plaintext-vs-FHE parity summaries for follow-on filter decisions
- updated `RESULTS.md`
- optional `FOLLOW_ON_RESULTS.md` if detailed tables become too large for `RESULTS.md`

This task must run on the Linux NVIDIA/OpenFHE box.

## Scope

You own official execution and result packaging for the follow-on.

You may make narrow bug fixes to the follow-on data, metric, timing, or harness code if official execution reveals a blocker.

Do not expand scope into:

- threshold sensitivity
- keyword/rule baselines
- broader generalization checks
- new Stage 1 training
- 7B repeat
- LangGraph production wiring
- rewriting original Stage 1/2/3 summaries

## Environment

From `experiment_runtime/`, prepare the environment:

```bash
uv sync --python 3.12 --extra fhe
```

Make sure Git LFS payloads are present:

```bash
git lfs pull
```

Required artifacts:

- `runs/stage1/official-1x-20260411-r1/adapter_model/`
- `runs/stage1/official-10x-20260411-r1/adapter_model/`
- `runs/stage1/official-50x-20260411-r1/adapter_model/`
- `runs/stage3/plaintext/stage3_plaintext_metrics.json`
- `runs/stage3/plaintext/model/model_parameters.json`
- `runs/stage3/fhe/compiled/`

If the FHE compiled bundle is missing, regenerate it with:

```bash
uv run --python 3.12 --extra fhe fhe-eval-stage3-fhe --config configs/eval/stage3_fhe_filter.toml
```

Only do that if the bundle is genuinely absent or invalid.

## Preflight Checks

Before running official evaluation, check:

```bash
uv run --python 3.12 python3 -m unittest \
  tests/test_follow_on_adaptive_attacks.py \
  tests/test_follow_on_mixed_traffic.py \
  tests/test_follow_on_metrics.py \
  tests/test_follow_on_timing.py \
  tests/test_follow_on_adaptive_eval.py \
  tests/test_follow_on_mixed_eval.py
```

If tests fail, fix the follow-on implementation before running official evaluations.

Do not run the full original project test sweep unless you changed shared Stage 2/Stage 3 code in a way that warrants it.

## Materialize Follow-On Data

From `experiment_runtime/`:

```bash
uv run --python 3.12 fhe-materialize-follow-on-adaptive --config configs/follow_on/adaptive_attacks.toml
uv run --python 3.12 fhe-materialize-follow-on-mixed --config configs/follow_on/mixed_traffic.toml
```

Expected outputs:

- `data/processed/follow_on/adaptive_attack_prompts.jsonl`
- `data/processed/follow_on/adaptive_attack_manifest.json`
- `data/processed/follow_on/mixed_traffic_eval.jsonl`
- `data/processed/follow_on/mixed_traffic_manifest.json`

Inspect manifests and confirm:

- `350` adaptive attack rows
- `700` mixed-traffic rows
- `350` benign mixed rows
- `350` adaptive adversarial mixed rows
- adaptive and mixed text disjointness checks passed

## Official Adaptive Attack-Only Evaluation

Run the full adaptive sweep:

```bash
uv run --python 3.12 --extra fhe fhe-eval-follow-on-adaptive \
  --config configs/follow_on/adaptive_replay.toml \
  --exposure all \
  --condition all
```

This should produce all `3 x 4 = 12` official adaptive attack-only runs:

- `1x / no_system_prompt`
- `1x / system_prompt_active`
- `1x / plaintext_filter`
- `1x / fhe_filter`
- `10x / no_system_prompt`
- `10x / system_prompt_active`
- `10x / plaintext_filter`
- `10x / fhe_filter`
- `50x / no_system_prompt`
- `50x / system_prompt_active`
- `50x / plaintext_filter`
- `50x / fhe_filter`

Required top-level outputs:

- `runs/follow_on/adaptive/adaptive_summary.json`
- `runs/follow_on/adaptive/adaptive_ci_summary.json`
- `runs/follow_on/adaptive/filter_parity_summary.json`

Required per-run outputs:

- `adaptive_responses.jsonl`
- `adaptive_metrics.json`
- `family_metrics.csv`
- `timing_pipeline_samples.csv`
- `timing_pipeline_summary.json`

## Official Mixed-Traffic Evaluation

Run the full mixed-traffic sweep:

```bash
uv run --python 3.12 --extra fhe fhe-eval-follow-on-mixed \
  --config configs/follow_on/mixed_traffic_replay.toml \
  --exposure all \
  --filter-mode all
```

This should produce all `3 x 2 = 6` official mixed-traffic runs:

- `1x / plaintext_filter`
- `1x / fhe_filter`
- `10x / plaintext_filter`
- `10x / fhe_filter`
- `50x / plaintext_filter`
- `50x / fhe_filter`

Required top-level outputs:

- `runs/follow_on/mixed_traffic/mixed_traffic_summary.json`
- `runs/follow_on/mixed_traffic/mixed_traffic_ci_summary.json`
- `runs/follow_on/mixed_traffic/filter_parity_summary.json`

Required per-run outputs:

- `mixed_traffic_responses.jsonl`
- `mixed_traffic_metrics.json`
- `family_metrics.csv`
- `timing_pipeline_samples.csv`
- `timing_pipeline_summary.json`

## Timing Artifacts

Confirm setup and warm-path timing artifacts exist.

Expected top-level path:

- `runs/follow_on/timing/setup_timing.json`

If the harness writes per-run setup timing files, also write or update a top-level manifest so future readers can discover them.

For every timing summary, confirm it includes:

- `count`
- `mean`
- `p50`
- `p90`
- `p95`
- `p99`
- `min`
- `max`
- `std`

## Result Sanity Checks

Before documentation updates, inspect the summaries and confirm:

- all expected exposure/condition runs are present
- adaptive attack-only filtered rows treat blocked prompts as non-leaks
- benign mixed rows blocked by the filter are counted as false positives
- leakage is scored only on model `response_text`
- plaintext and FHE parity summaries compare aligned row IDs
- confidence intervals include numerator and denominator
- timing summaries have non-empty counts for measured paths

If plaintext and FHE filter decisions diverge, do not hide it. Report the observed mismatch count and affected row IDs in the summaries and docs.

## Documentation Update

Update `RESULTS.md` with a concise follow-on section.

Include:

- adaptive attacker setup
- adaptive attack-only headline table
- mixed-traffic headline table
- confidence interval method
- plaintext-vs-FHE parity result
- timing headline numbers
- limitations

Keep the original Stage 1/2/3 result sections intact.

If detailed tables make `RESULTS.md` too large, create:

- `FOLLOW_ON_RESULTS.md`

Then add a concise summary plus link from `RESULTS.md`.

Limitations to mention if still true:

- follow-on uses the existing `Qwen2-1.5B-Instruct` adapters only
- mixed traffic is still synthetic
- sentence encoding remains plaintext
- threshold sensitivity and keyword/rule baselines are intentionally deferred
- official metrics still come from the direct experiment harness, not the LangGraph scaffold

## Optional Artifact Audit

If time permits, write or update a lightweight follow-on audit JSON under:

- `runs/follow_on/follow_on_artifact_audit.json`

It should list:

- expected data artifacts
- expected adaptive run artifacts
- expected mixed-traffic run artifacts
- timing artifacts
- documentation files updated
- any missing files

Do not spend time building a large audit framework if the summaries and docs are already clear.

## Suggested Final Verification

From `experiment_runtime/`:

```bash
uv run --python 3.12 python3 -m unittest \
  tests/test_follow_on_adaptive_attacks.py \
  tests/test_follow_on_mixed_traffic.py \
  tests/test_follow_on_metrics.py \
  tests/test_follow_on_timing.py \
  tests/test_follow_on_adaptive_eval.py \
  tests/test_follow_on_mixed_eval.py
```

If you changed shared existing Stage 2, Stage 3, or FHE modules, also run the affected original tests, for example:

```bash
uv run --python 3.12 python3 -m unittest \
  tests/test_stage2_eval.py \
  tests/test_stage2_filtered_eval.py \
  tests/test_stage3_fhe.py
```

## Done Criteria

Done means:

- official adaptive attack-only sweep completed
- official mixed-traffic sweep completed
- all required follow-on summary JSON/CSV artifacts exist
- confidence intervals are included
- timing artifacts are included
- `RESULTS.md` is updated, with `FOLLOW_ON_RESULTS.md` if useful
- tests pass or any skipped tests are clearly justified
- no original frozen result artifact is overwritten except documentation that intentionally summarizes the follow-on
