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
- implementation has not started yet
- the next task is `01_repo_scaffold_and_protocol_constants.md`
