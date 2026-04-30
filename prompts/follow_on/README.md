# Follow-On Prompts

This folder contains handoff prompts for the adaptive-attacker follow-on experiment.

Authoritative scope:

1. `plan/follow_on_designer_scope.md`
2. `plan/05_follow_on_adaptive_evaluation.md`
3. `prompts/follow_on/00_follow_on_shared_context.md`

Use the prompts in order:

1. [00 Follow-On Shared Context](./00_follow_on_shared_context.md)
2. [01 Adaptive and Mixed Data](./01_adaptive_and_mixed_data.md)
3. [02 Metrics and Timing Infra](./02_metrics_and_timing_infra.md)
4. [03 Adaptive Eval Harness](./03_adaptive_eval_harness.md)
5. [04 Official NVIDIA Runs and Results](./04_official_nvidia_runs_and_results.md)

Post-closeout calibration confirmation:

6. [05 Threshold Calibration Confirmation](./05_threshold_calibration_confirmation.md)

Execution policy:

- run tasks sequentially
- vet each task before starting the next
- update later prompts only if a completed task materially changes paths or interfaces
- preserve all original Stage 1, Stage 2, and Stage 3 artifacts

Out of scope for this follow-on pass:

- threshold sensitivity
- keyword/rule baselines
- broader generalization checks
- `Qwen2-7B-Instruct` repeat
- LangGraph parity integration

Task 05 is a later exception to the original threshold-sensitivity non-goal. It exists because the completed mixed-traffic run exposed a high benign false-positive rate and the local calibration screen selected candidate thresholds for NVIDIA confirmation.
