# Experiment Status

This is the quickest orientation file for future Codex sessions. It is a current-state map, not the canonical metric source; use [RESULTS.md](./RESULTS.md) for exact numbers and detailed interpretation.

The repo contains a completed three-stage `Qwen2-1.5B-Instruct` FHE privacy experiment, a completed adaptive-attacker/mixed-traffic follow-on, a completed threshold calibration confirmation, and a completed final held-out adaptive robustness check. The original experiment and follow-on artifacts are preserved as frozen result sets; future work should add new artifacts rather than rewriting them.

## Read First

Canonical docs:

1. [RESULTS.md](./RESULTS.md) - canonical results, interpretation, limitations, and artifact pointers.
2. [FOLLOW_ON.md](./FOLLOW_ON.md) - follow-on-specific orientation and artifact layout.
3. [plan/README.md](./plan/README.md) - implementation plan index.
4. [prompts/README.md](./prompts/README.md) - task handoff prompt index.
5. [experiment_runtime/README.md](./experiment_runtime/README.md) - runtime commands and package layout.

Doc roles:

- `README.md` is the short repo entry point.
- `EXPERIMENT_STATUS.md` is the cross-project current-state map.
- `RESULTS.md` is the canonical result and interpretation record.
- `FOLLOW_ON.md` is the follow-on-specific artifact and history map.

For a research-abstract-style snapshot of the follow-on, read the "Follow-on executive summary" in [RESULTS.md](./RESULTS.md#follow-on-adaptive-and-mixed-evaluation).

## Current State

Completed:

- Stage 1: `Qwen2-1.5B-Instruct` LoRA training across `1x`, `10x`, and `50x` canary exposure conditions.
- Stage 2: attack replay and leakage scoring.
- Stage 3: plaintext ALLOW/BLOCK filter training, CKKS/OpenFHE wrapper, and filter evaluation.
- Integrated Stage 2 reruns with plaintext and FHE filters.
- Follow-on adaptive attacker evaluation.
- Follow-on mixed benign/adversarial traffic evaluation.
- Follow-on confidence intervals, timing, and plaintext-vs-FHE parity summaries.
- Threshold calibration screening and NVIDIA/OpenFHE confirmation at thresholds `0.72` and `0.80`.
- Final held-out adaptive robustness check at the conservative Stage 3 threshold and threshold `0.72`.

Not complete / future ablations:

- Stage 3 v2 retraining with broader benign and hard-negative coverage.
- Keyword/rule baselines.
- Additional broader generalization checks.
- `Qwen2-7B-Instruct` repeat.
- Production LangGraph wiring/parity integration.

## Artifact Map

Frozen original experiment artifacts:

- `experiment_runtime/runs/stage1/`
- `experiment_runtime/runs/stage2/`
- `experiment_runtime/runs/stage3/`

Completed follow-on artifacts:

- `experiment_runtime/data/processed/follow_on/`
- `experiment_runtime/runs/follow_on/adaptive/`
- `experiment_runtime/runs/follow_on/mixed_traffic/`
- `experiment_runtime/runs/follow_on/timing/`

Threshold calibration artifacts:

- Mac-side threshold screen: `experiment_runtime/runs/follow_on/calibration/`
- NVIDIA/OpenFHE confirmation: `experiment_runtime/runs/follow_on/calibration_confirmation/`
- Combined confirmation summary: `experiment_runtime/runs/follow_on/calibration_confirmation/threshold_confirmation_summary.json`

Final held-out robustness artifacts:

- held-out data: `experiment_runtime/data/processed/follow_on/held_out_robustness/`
- held-out configs: `experiment_runtime/configs/follow_on/held_out_robustness/`
- held-out runs and summaries: `experiment_runtime/runs/follow_on/held_out_robustness/`
- paper-oriented summary: `experiment_runtime/runs/follow_on/held_out_robustness/HELD_OUT_ROBUSTNESS_RESULTS.md`

Do not mutate frozen official artifacts unless a task explicitly asks for a new experiment that intentionally replaces them. Prefer new output roots for any new ablation.

## Supported Claims

The full experiment is successful and serious for these claims:

- Canary overexposure can produce measurable memorization/leakage behavior under the tested setup.
- The direct replay harness and leakage scorer quantify leakage across exposure conditions.
- The Stage 3 filter suppresses measured leakage under the tested attack paths at the conservative frozen threshold.
- The CKKS/OpenFHE filter preserves plaintext filter decisions with exact observed decision parity on the tested rows.
- The follow-on adaptive, mixed-traffic, threshold-confirmation, and held-out evaluations make the defense story stronger by testing beyond the original frozen Stage 2 attack set.

## Important Caveat

The current Stage 3 filter is not production-ready or broadly utility-calibrated.

The follow-on mixed-traffic run exposed a high benign false-positive rate at the frozen Stage 3 threshold:

- frozen Stage 3 threshold: approximately `0.4199950085320943`
- mixed benign false positives: `110 / 350 = 31.43%`
- plaintext and FHE decisions matched exactly
- measured filtered adaptive leakage remained suppressed

The final held-out robustness check adds a second caveat: zero measured leakage on the original adaptive set did not fully generalize to fresh held-out adaptive wording.

Held-out `50x` results:

| Threshold | Any-field leak | Full-record leak | Adaptive block | Benign hard-negative FP | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| `~0.4200` | `2 / 343 = 0.58%` | `0 / 343 = 0.00%` | `270 / 343 = 78.72%` | `74 / 200 = 37.00%` | privacy-conservative, not zero-leak on held-out wording |
| `0.72` | `4 / 343 = 1.17%` | `0 / 343 = 0.00%` | `185 / 343 = 53.94%` | `32 / 200 = 16.00%` | better utility, more held-out any-field leakage |

Plaintext and FHE decisions matched exactly on every held-out filtered row. The held-out traffic is still synthetic, and exact message/request disjointness is not the same as production realism or semantic disjointness.

Interpretation:

- This is not an FHE bug.
- This is not an evaluation-harness bug.
- This is not random noise.
- It shows that the Stage 3 benign train/validation distribution was too narrow or too easy relative to the follow-on benign traffic.

The blocked benign prompts often use legitimate fraud-operations language around case queues, routing, risk review, audit, and identity handling. That language overlaps semantically with adversarial extraction prompts, so the classifier learned a conservative boundary.

## Threshold Calibration

The tuned threshold is the filter's block-probability cutoff:

```text
BLOCK if filter_block_probability >= threshold
ALLOW otherwise
```

The calibration pass kept the same trained classifier and moved only this decision cutoff. It did not retrain the classifier and did not mutate the frozen Stage 3 model parameters.

Confirmed threshold results:

| Threshold | Mixed benign FP | Mixed adaptive block | 50x adaptive any leak | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| `~0.4200` | `110 / 350 = 31.43%` | `224 / 350 = 64.00%` | `0 / 350 = 0.00%` | privacy-conservative frozen baseline |
| `0.72` | `33 / 350 = 9.43%` | `137 / 350 = 39.14%` | `2 / 350 = 0.57%` | most plausible utility-calibrated candidate tested |
| `0.80` | `16 / 350 = 4.57%` | `114 / 350 = 32.57%` | `6 / 350 = 1.71%` | better utility, sharper privacy tradeoff |

Current recommendation:

- Keep the frozen Stage 3 threshold as the privacy-conservative baseline.
- Treat threshold `0.72` as the most plausible utility-calibrated operating point among the confirmed candidates, if small `50x` adaptive leakage is acceptable.
- Treat threshold tuning as an operating-point adjustment, not a real data-distribution fix.
- Treat Stage 3 v2 retraining with broader benign/hard-negative coverage as the next substantive ablation.

The final held-out pass keeps this recommendation intact, but it narrows the privacy claim: the conservative threshold substantially suppresses held-out leakage and keeps full-record leakage at zero, while allowing `2 / 343` any-field leaks.

## Execution Notes

This repo uses Python `3.12` for the experiment runtime.

Use `uv` from `experiment_runtime/`:

```bash
uv run --python 3.12 python3 -m unittest tests/test_follow_on_calibration.py
```

Mac-safe work:

- docs
- data materialization
- validators
- metrics/timing helpers
- threshold screening from saved artifacts
- unit tests that do not load Qwen or OpenFHE

NVIDIA/OpenFHE box required:

- Qwen adapter generation/replay runs
- full follow-on adaptive or mixed replay
- FHE filter scoring
- official threshold confirmation reruns

## Future Work

Most valuable next experiment:

- Stage 3 v2 filter retraining with expanded benign hard negatives, especially case-queue review, routing follow-up, risk-review, audit, and identity-handling language.

Other useful work:

- Report full privacy/utility operating curves, not only selected thresholds.
- Add keyword/rule baselines.
- Run additional broader generalization checks.
- Repeat key flows on `Qwen2-7B-Instruct`.
- Wire the result-producing path into the LangGraph scaffold if production parity becomes a goal.
