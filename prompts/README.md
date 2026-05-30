# Prompts

This folder contains manager-written prompts for separate Codex sessions working on the FHE experiment.

Use prompts in this order:

1. [00_shared_context.md](./00_shared_context.md)
2. task-specific prompt for the current ticket

Prompt policy:

- Every Codex session should read `00_shared_context.md` first.
- Each task prompt should also point to only the specific `plan/` files needed for that task.
- Keep each task tightly scoped, with explicit done criteria and owned files.
- Prefer sequential execution unless there is a clearly independent parallel task.

Current implementation stage:

- spec freeze is complete
- Task 01 scaffold work is complete
- Task 02 Tier 1 data generation is complete
- Task 03 Tier 2 chat rendering and Stage 1 corpora are complete
- Task 04 Stage 2 attack prompt generation is complete
- Task 05 Stage 3 ALLOW/BLOCK dataset generation is complete
- Task 06 Stage 1 LoRA training pipeline is complete
- Task 07 Stage 1 official training runs are complete
- Task 08 Stage 1 MIA evaluator is complete
- Task 09 Stage 2 harness and leakage scorer is complete
- Task 10 Stage 3 plaintext filter training is complete
- Task 11 Stage 3 FHE wrapper and evaluation is complete
- Task 12 integrated Stage 2 reruns with filters is complete
- Task 13 final closeout verification is complete
- the main `Qwen2-1.5B-Instruct` flow is complete and handoff-ready
- the adaptive-attacker and mixed-traffic follow-on is complete
- the threshold calibration screen and NVIDIA/OpenFHE confirmation are complete
- final held-out adaptive robustness prompts have been added for a later publication robustness run
- future optional work remains, such as Stage 3 v2 retraining, keyword/rule baselines, broader generalization checks, the `Qwen2-7B-Instruct` repeat, or LangGraph runtime parity integration

Ad hoc post-experiment prompts live under:

- [temp/](./temp/README.md)

Follow-on adaptive attacker prompts live under:

- [follow_on/](./follow_on/README.md)

Current status:

- [EXPERIMENT_STATUS.md](../EXPERIMENT_STATUS.md) is the high-level read-first map for the completed experiment, follow-on, threshold confirmation, caveats, and future work.

Follow-on scope note:

- `plan/follow_on_designer_scope.md` is the authoritative designer scope for the adaptive-attacker follow-on
- `plan/05_follow_on_adaptive_evaluation.md` freezes the repo organization and task sequence
- `plan/06_held_out_adaptive_robustness.md` freezes the later held-out adaptive robustness check
- threshold sensitivity was later handled for the confirmed `0.72` and `0.80` operating points; keyword/rule baselines and broader generalization checks remain deferred
- the held-out robustness prompts are a scoped publication check, not a filter tuning or retraining pass
