# Hecta Palm

This repo contains the three-stage FHE privacy experiment plus the runtime implementation under [experiment_runtime](./experiment_runtime/).

Useful entry points:

- [experiment_runtime/README.md](./experiment_runtime/README.md): runtime setup and local commands
- [plan/README.md](./plan/README.md): implementation plan and protocol
- [RESULTS.md](./RESULTS.md): project results summary

## Git LFS

This repo now uses Git LFS for heavyweight run artifacts that need to be preserved across machines.

Examples of LFS-tracked artifacts in this repo:

- Stage 1 LoRA adapters and full trainer checkpoints
- large tokenizer snapshots inside checkpoints
- Stage 3 OpenFHE compiled bundle files

Lightweight metadata such as JSON, CSV, TOML, Jinja, and Markdown files remain in normal Git.

## Clone With All LFS Files

If `git-lfs` is installed and enabled, a normal clone is enough:

```bash
git clone git@github.com:DeltaShield999/hecta-palm.git
cd hecta-palm
git lfs pull
```

Notes:

- In many setups, `git clone` will already download the LFS payloads automatically.
- Running `git lfs pull` after cloning is still a safe way to ensure all tracked large files are present.

## Clone Without Downloading LFS Files

If you want the repo contents and LFS pointer files only, without downloading the heavyweight payloads yet:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:DeltaShield999/hecta-palm.git
cd hecta-palm
```

That gives you the Git checkout, but the large LFS-backed files will remain as pointer files until you fetch them explicitly.

## Download LFS Files Later

In an existing clone that was created without the LFS payloads:

```bash
git lfs pull
```

If `git-lfs` was not installed when you cloned, install it first, then run:

```bash
git lfs install
git lfs pull
```

## Current Behavior

- Full clone with LFS payloads: supported
- Clone without LFS payloads: supported
- Fetch LFS payloads later in the same clone: supported

So there is no special alternate clone command required for the full version beyond having `git-lfs` installed. The optional path is the metadata-only clone using `GIT_LFS_SKIP_SMUDGE=1`.
