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
- the next task is `10_stage3_plaintext_filter_training.md`
