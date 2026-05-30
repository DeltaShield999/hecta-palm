# Plan

This folder contains the final implementation plan for the FHE experiment in a small set of focused Markdown files.

Start here, then read the files in order:

1. [Overview and Architecture](./01_overview_and_architecture.md)
2. [Data Contracts and Stage 1](./02_data_and_stage1.md)
3. [Stage 2 and Stage 3 Protocol](./03_stage2_and_stage3.md)
4. [Repo Structure and Execution Plan](./04_repo_and_execution_plan.md)
5. [Follow-On Adaptive Evaluation](./05_follow_on_adaptive_evaluation.md)
6. [Held-Out Adaptive Robustness Evaluation](./06_held_out_adaptive_robustness.md)

Follow-on designer scope:

- [Follow-On Designer Scope](./follow_on_designer_scope.md)

Structure rule:

- `01` is the conceptual and architectural entry point
- `02` freezes the data layer and Stage 1 protocol
- `03` freezes the attack, filter, and FHE protocol
- `04` tells future Codex sessions where code should live and in what order to build it
- `05` freezes the post-completion adaptive attacker and mixed-traffic follow-on scope
- `06` freezes the final held-out adaptive robustness check requested after threshold confirmation

This `plan/` folder is now the canonical implementation plan for the experiment.
