# Follow-On Adaptive Evaluation

For the cross-project current-state map, read [EXPERIMENT_STATUS.md](./EXPERIMENT_STATUS.md). This file is the follow-on-specific artifact and history map; exact metric interpretation lives in [RESULTS.md](./RESULTS.md).

The original `Qwen2-1.5B-Instruct` FHE privacy experiment is complete. Its canonical summary is still [RESULTS.md](./RESULTS.md), and its original artifacts remain under:

- `experiment_runtime/runs/stage1/`
- `experiment_runtime/runs/stage2/`
- `experiment_runtime/runs/stage3/`

This repo also contains the completed follow-on adaptive attacker and mixed-traffic experiment requested after the original closeout. The canonical follow-on result summary is the [Follow-On Adaptive And Mixed Evaluation](./RESULTS.md#follow-on-adaptive-and-mixed-evaluation) section in `RESULTS.md`.

Planning and handoff records:

1. [plan/follow_on_designer_scope.md](./plan/follow_on_designer_scope.md)
2. [plan/05_follow_on_adaptive_evaluation.md](./plan/05_follow_on_adaptive_evaluation.md)
3. [prompts/follow_on/](./prompts/follow_on/)

Completed follow-on artifact layout:

- configs: `experiment_runtime/configs/follow_on/`
- data: `experiment_runtime/data/processed/follow_on/`
- official runs and summaries: `experiment_runtime/runs/follow_on/`
- calibration screening artifacts: `experiment_runtime/runs/follow_on/calibration/`
- NVIDIA threshold confirmation artifacts: `experiment_runtime/runs/follow_on/calibration_confirmation/`
- final held-out robustness artifacts: `experiment_runtime/runs/follow_on/held_out_robustness/`
- code: `experiment_runtime/src/experiment/follow_on/`
- tests: `experiment_runtime/tests/test_follow_on_*.py`

The follow-on reused the existing runtime package and direct evaluation harness style. It did not become a separate `follow_on_experiment/` project, and it did not replace the frozen original experiment results.

Important final interpretation:

- The follow-on confirms that the plaintext and CKKS/OpenFHE filters suppress measured adaptive leakage on the tested adaptive and mixed datasets.
- Plaintext and FHE filter decisions match exactly on the follow-on rows.
- The mixed-traffic run exposed a serious utility limitation: `110 / 350 = 0.3143` benign mixed rows were blocked as false positives, much higher than the original Stage 3 held-out false-positive rate.
- The tuned threshold is the filter decision cutoff: `BLOCK` when `filter_block_probability >= threshold`, otherwise `ALLOW`.
- A Mac-side threshold screen under `experiment_runtime/runs/follow_on/calibration/` kept the same trained classifier and swept alternate cutoffs; it showed that higher thresholds reduce false positives but reintroduce some `50x` adaptive leakage.
- NVIDIA/OpenFHE confirmation runs under `experiment_runtime/runs/follow_on/calibration_confirmation/` confirmed the Mac-side count screen exactly for thresholds `0.72` and `0.80`.
- At threshold `0.72`, mixed benign false positives drop to `33 / 350 = 0.0943`, mixed adaptive block rate drops to `137 / 350 = 0.3914`, and `50x` adaptive leakage is `2 / 350 = 0.0057`.
- At threshold `0.80`, mixed benign false positives drop to `16 / 350 = 0.0457`, mixed adaptive block rate drops to `114 / 350 = 0.3257`, and `50x` adaptive leakage is `6 / 350 = 0.0171`.
- Plaintext and FHE filter decisions match exactly in the threshold confirmation runs.
- Recommended interpretation: keep the frozen Stage 3 threshold as the privacy-conservative baseline; treat `0.72` as the more plausible utility-calibrated operating point among the confirmed candidates if a small amount of `50x` adaptive leakage is acceptable.
- The final held-out robustness pass reused the frozen model, filter, and thresholds on a fresh `343` row scaffold-aware adaptive set plus `200` benign hard negatives. At the conservative threshold, filtered `50x` any-field leakage was `2 / 343 = 0.0058`, full-record leakage was `0 / 343`, and benign hard-negative false positives were `74 / 200 = 0.3700`. At threshold `0.72`, any-field leakage was `4 / 343 = 0.0117`, full-record leakage was `0 / 343`, and benign hard-negative false positives were `32 / 200 = 0.1600`.
- The held-out pass confirms exact plaintext/FHE decision parity, but it also shows that the zero-leakage result on the original adaptive set does not fully generalize to fresh held-out adaptive wording.
- Full-experiment judgment: successful and serious for the privacy/leakage-suppression/FHE-parity claims, but not evidence of a production-ready, broadly utility-calibrated filter. The false positives point to narrow Stage 3 benign coverage and motivate a future benign/hard-negative retraining ablation.
- Keyword/rule baselines, additional broader generalization checks, and benign/hard-negative filter expansion remain future ablation work.

Threshold confirmation pointers:

- combined confirmation summary: `experiment_runtime/runs/follow_on/calibration_confirmation/threshold_confirmation_summary.json`
- confirmation audit: `experiment_runtime/runs/follow_on/calibration_confirmation/threshold_confirmation_audit.json`
- threshold `0.72` run root: `experiment_runtime/runs/follow_on/calibration_confirmation/threshold_0_7200/`
- threshold `0.80` run root: `experiment_runtime/runs/follow_on/calibration_confirmation/threshold_0_8000/`
- reproducible configs:
  - `experiment_runtime/configs/follow_on/adaptive_replay_threshold_0_7200.toml`
  - `experiment_runtime/configs/follow_on/mixed_traffic_replay_threshold_0_7200.toml`
  - `experiment_runtime/configs/follow_on/adaptive_replay_threshold_0_8000.toml`
  - `experiment_runtime/configs/follow_on/mixed_traffic_replay_threshold_0_8000.toml`

Held-out robustness pointers:

- held-out artifact root: `experiment_runtime/runs/follow_on/held_out_robustness/`
- held-out data root: `experiment_runtime/data/processed/follow_on/held_out_robustness/`
- held-out configs: `experiment_runtime/configs/follow_on/held_out_robustness/`
- adaptive data: `experiment_runtime/data/processed/follow_on/held_out_robustness/held_out_adaptive_attack_prompts.jsonl`
- benign hard negatives: `experiment_runtime/data/processed/follow_on/held_out_robustness/held_out_benign_hard_negatives.jsonl`
- mixed held-out data: `experiment_runtime/data/processed/follow_on/held_out_robustness/held_out_mixed_traffic_eval.jsonl`
- conservative run root: `experiment_runtime/runs/follow_on/held_out_robustness/conservative/`
- threshold `0.72` run root: `experiment_runtime/runs/follow_on/held_out_robustness/threshold_0_7200/`
- machine-readable summary: `experiment_runtime/runs/follow_on/held_out_robustness/held_out_robustness_summary.json`
- audit: `experiment_runtime/runs/follow_on/held_out_robustness/held_out_robustness_audit.json`
- paper-oriented summary: `experiment_runtime/runs/follow_on/held_out_robustness/HELD_OUT_ROBUSTNESS_RESULTS.md`
