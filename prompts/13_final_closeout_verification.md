# Task 13: Final Closeout Verification

Read these first:

1. `prompts/00_shared_context.md`
2. `RESULTS.md`
3. `plan/04_repo_and_execution_plan.md`
4. `docs/md/FHE_Experiment_Engineering_Guide.md` Sections 6.1 through 6.3

## Goal

Do a final closeout verification pass for the completed `Qwen2-1.5B-Instruct` experiment.

This is not a new experiment task. It is an artifact and handoff audit.

The main thing to verify is that the repo and Git LFS together contain the critical deliverables needed for handoff and reproduction.

## Scope

Verify:

1. the critical heavy artifacts exist locally on the Linux machine
2. the critical heavy artifacts are tracked through Git LFS
3. the critical lightweight results artifacts are committed normally and present
4. the current repo-visible documentation points to the right results and artifact paths

If something critical is missing or mis-tracked, fix it if it is straightforward and low risk. Otherwise stop and report the exact gap.

Do not expand this into new experiments, plots, or code refactors.

## Important Constraints

- Work from the Linux machine that has the full working copy and the large artifacts
- Use the current Python `3.12` baseline
- Do not rerun training or evaluation jobs
- Do not create new experimental outputs
- Do not modify scientific results
- Keep any repo changes minimal and limited to closeout verification artifacts or straightforward tracking fixes

## What To Verify

### A. Critical heavy artifacts

At minimum, verify these exist locally and are Git LFS-tracked:

- Stage 1 official adapter artifacts for:
  - `experiment_runtime/runs/stage1/official-1x-20260411-r1/`
  - `experiment_runtime/runs/stage1/official-10x-20260411-r1/`
  - `experiment_runtime/runs/stage1/official-50x-20260411-r1/`
- Stage 1 heavy checkpoint/state artifacts that are needed for reproducibility
- Stage 3 compiled OpenFHE bundle under:
  - `experiment_runtime/runs/stage3/fhe/compiled/`

Also verify the committed processed datasets needed to reproduce the reported flow are present and appropriately tracked if they are large.

### B. Critical lightweight artifacts

Verify these repo-visible results artifacts exist and are committed:

- `RESULTS.md`
- `experiment_runtime/README.md`
- `experiment_runtime/runs/stage1/official_runs_summary.json`
- `experiment_runtime/runs/stage1/mia/mia_summary.json`
- `experiment_runtime/runs/stage2/baseline/stage2_summary.json`
- `experiment_runtime/runs/stage3/plaintext/stage3_plaintext_metrics.json`
- `experiment_runtime/runs/stage3/fhe/stage3_fhe_metrics.json`
- `experiment_runtime/runs/stage3/fhe/compiled_bundle_manifest.json`
- `experiment_runtime/runs/stage2/filtered/stage2_filtered_summary.json`
- `experiment_runtime/runs/stage2/filtered/filter_parity_summary.json`

### C. Final audit artifact

Write one lightweight closeout artifact to the repo:

- `experiment_runtime/runs/final_artifact_audit.json`

It should summarize:

- repo commit audited
- whether the critical heavy artifacts were found locally
- whether the critical heavy artifacts were found in Git LFS
- whether the critical lightweight artifacts were found
- any gaps detected
- whether the 1.5B experiment is ready for handoff

## Suggested Checks

Use normal non-interactive git and shell commands such as:

- `git lfs ls-files`
- `git status --short`
- `git rev-parse HEAD`
- `find` / `rg` to confirm presence of required files

You can use a small Python script if that is the cleanest way to produce the audit JSON.

## Done Criteria

This task is done when:

1. the heavy-artifact situation is verified clearly
2. the lightweight handoff artifacts are verified clearly
3. `experiment_runtime/runs/final_artifact_audit.json` exists and is accurate
4. any real packaging gap is either fixed or explicitly reported

## Out Of Scope

- new plots
- new training or evaluation runs
- 7B work
- LangGraph parity work
- refactoring unrelated code

## Report Back

When done, report:

- what files you changed
- what commit you audited
- whether all critical heavy artifacts were present locally
- whether all critical heavy artifacts were in Git LFS
- whether all critical lightweight artifacts were present
- whether any fixes were needed
- whether the 1.5B experiment is ready for handoff
- any remaining closeout caveats
