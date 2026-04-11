# Setup Guide

This package now contains the completed `Qwen2-1.5B-Instruct` experiment runtime and its supporting CLIs.

Use this file for environment setup and machine-role guidance. For the experiment outcome summary, read [../RESULTS.md](../RESULTS.md). For the current runtime surface and artifact layout, read [README.md](./README.md).

## Machine Roles

There are three practical environments in this project:

1. local macOS for repo navigation, documentation work, deterministic data tasks, and Stage 3 plaintext work
2. Linux with NVIDIA GPU for Stage 1 model training and Stage 2 replay/evaluation
3. Linux with the optional `fhe` extra for Stage 3 OpenFHE work and the integrated filtered reruns

In practice, the Linux GPU box can serve as both `2` and `3`.

## Base Environment

From `experiment_runtime/`:

```bash
uv venv --python 3.12 --clear
uv sync --python 3.12
```

This is enough for:

- data materialization
- plaintext Stage 3 filter training
- local test runs
- the LangGraph demo/runtime scaffold

## FHE Environment

For Stage 3 OpenFHE work on Linux:

```bash
uv sync --python 3.12 --extra fhe
```

Notes:

- the optional `fhe` extra installs `openfhe`
- the FHE path was implemented and evaluated on Linux, not on macOS

## Git LFS

This repo uses Git LFS for heavy experiment artifacts.

After cloning on a fresh machine, make sure the payloads are present:

```bash
git lfs pull
```

This matters especially for:

- Stage 1 adapters and checkpoints
- the compiled Stage 3 OpenFHE bundle

## Common Verification Commands

Full local test sweep:

```bash
uv run --python 3.12 python3 -m unittest discover -s tests -p "test_*.py"
```

LangGraph demo smoke run:

```bash
uv run fhe-experiment-run --json
```

Stage 3 plaintext filter training:

```bash
uv run --python 3.12 fhe-train-stage3-plaintext --config configs/eval/stage3_plaintext_filter.toml
```

Stage 3 FHE evaluation on Linux:

```bash
uv run --python 3.12 --extra fhe fhe-eval-stage3-fhe --config configs/eval/stage3_fhe_filter.toml
```

Integrated Stage 2 filtered reruns on Linux:

```bash
uv run --python 3.12 --extra fhe fhe-eval-stage2-filtered --config configs/eval/stage2_filtered_replay.toml --exposure all --filter-mode all
```

## Troubleshooting

### `uv sync --python 3.12` fails

Check:

- `uv` is installed
- Python `3.12` is available or downloadable through `uv`

### `git lfs pull` does not fetch the expected large files

Check:

- `git-lfs` is installed
- `git lfs install` has been run on that machine
- the clone is on the expected branch / commit

### macOS tests pass but FHE tests are skipped

That is expected if `openfhe` is not installed locally. The FHE path is optional on macOS and was validated on Linux.

### The LangGraph demo behavior does not match the official experiment outputs

That can also be expected in the current repo state. The official metrics were produced by the dedicated `experiment.*` harness, not by the LangGraph demo shell. See [../RESULTS.md](../RESULTS.md) for the current `Agentic Execution Note`.
