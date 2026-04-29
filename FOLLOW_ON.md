# Follow-On Adaptive Evaluation

The original `Qwen2-1.5B-Instruct` FHE privacy experiment is complete. Its canonical summary is still [RESULTS.md](./RESULTS.md), and its original artifacts remain under:

- `experiment_runtime/runs/stage1/`
- `experiment_runtime/runs/stage2/`
- `experiment_runtime/runs/stage3/`

This repo now also contains planning and handoff material for a follow-on experiment requested after that completion.

Authoritative follow-on scope:

1. [plan2.md](./plan2.md)
2. [plan/05_follow_on_adaptive_evaluation.md](./plan/05_follow_on_adaptive_evaluation.md)
3. [prompts/follow_on/](./prompts/follow_on/)

`plan1.md` is background only. For this pass, do not implement threshold sensitivity, keyword/rule baselines, or the broader generalization check.

Planned follow-on artifact layout:

- configs: `experiment_runtime/configs/follow_on/`
- data: `experiment_runtime/data/processed/follow_on/`
- runs: `experiment_runtime/runs/follow_on/`
- code: `experiment_runtime/src/experiment/follow_on/`

The follow-on should reuse the existing runtime package and direct evaluation harness style. It should not become a separate `follow_on_experiment/` project, and it should not replace the frozen original experiment results.
