# Follow-On Adaptive Evaluation

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
- code: `experiment_runtime/src/experiment/follow_on/`
- tests: `experiment_runtime/tests/test_follow_on_*.py`

The follow-on reused the existing runtime package and direct evaluation harness style. It did not become a separate `follow_on_experiment/` project, and it did not replace the frozen original experiment results.

Important final interpretation:

- The follow-on confirms that the plaintext and CKKS/OpenFHE filters suppress measured adaptive leakage on the tested adaptive and mixed datasets.
- Plaintext and FHE filter decisions match exactly on the follow-on rows.
- The mixed-traffic run exposed a serious utility limitation: `110 / 350 = 0.3143` benign mixed rows were blocked as false positives, much higher than the original Stage 3 held-out false-positive rate.
- Threshold sensitivity, keyword/rule baselines, broader generalization checks, and benign/hard-negative filter expansion remain future ablation work.
