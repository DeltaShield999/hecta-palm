# FHE Experiment Runtime

This package contains the runtime, data materializers, training CLIs, evaluation CLIs, and artifact layout for the `Qwen2-1.5B-Instruct` FHE privacy experiment.

The main 1.5B experiment flow is implemented end to end:

- deterministic Tier 1, Stage 1, Stage 2, and Stage 3 data materialization
- Stage 1 LoRA training
- Stage 1 membership inference evaluation
- Stage 2 baseline replay and leakage scoring
- Stage 3 plaintext filter training
- Stage 3 CKKS/OpenFHE filter parity and latency evaluation
- integrated Stage 2 reruns with plaintext and FHE filters active

Top-level experiment results are summarized in [../RESULTS.md](../RESULTS.md). The frozen implementation plan lives in [../plan/](../plan/).

## Important Scope Notes

- Official experiment results come from the `experiment.*` CLIs and the artifacts under `runs/`.
- The `qwen_langgraph_demo/` package is still present as a historical LangGraph demo/runtime scaffold, but it is not the official execution path for the reported Stage 1, Stage 2, or Stage 3 results.
- Heavy artifacts are tracked through Git LFS. The repo still keeps lightweight summaries, manifests, and metrics in normal Git, but the large adapters, checkpoints, and compiled OpenFHE bundle now require `git lfs pull` after clone if the payloads are not already present.

## Environment

This package is now standardized on Python `3.12`.

Base environment:

```bash
uv venv --python 3.12 --clear
uv sync --python 3.12
```

Optional Linux FHE environment:

```bash
uv sync --python 3.12 --extra fhe
```

Notes:

- `openfhe` is installed only through the optional `fhe` extra.
- The OpenFHE Python wrapper used here does not support the earlier Python `3.13` setup that this repo started with.
- `.env` is only needed if you want to run the LangGraph demo or the endpoint smoke test.

## CLI Surface

Core experiment CLIs:

| Command | Purpose | Typical machine |
| --- | --- | --- |
| `fhe-materialize-tier1` | Materialize Tier 1 records and canary registry | Mac or Linux |
| `fhe-materialize-stage1` | Materialize Stage 1 corpora and MIA eval set | Mac or Linux |
| `fhe-materialize-stage2` | Materialize Stage 2 attack prompts | Mac or Linux |
| `fhe-materialize-stage3` | Materialize Stage 3 ALLOW/BLOCK datasets | Mac or Linux |
| `fhe-train-stage1` | Train the Qwen2-1.5B-Instruct LoRA adapter | Linux GPU |
| `fhe-eval-stage1-mia` | Run Stage 1 membership inference evaluation | Linux GPU |
| `fhe-eval-stage2` | Run Stage 2 baseline attack replay | Linux GPU |
| `fhe-train-stage3-plaintext` | Train the plaintext Stage 3 filter | Mac or Linux |
| `fhe-eval-stage3-fhe` | Run Stage 3 CKKS/OpenFHE parity and latency eval | Linux with `--extra fhe` |
| `fhe-eval-stage2-filtered` | Run integrated Stage 2 reruns with plaintext/FHE filters | Linux with `--extra fhe` |

Demo and scaffold CLIs:

| Command | Purpose |
| --- | --- |
| `fhe-experiment-run` | Run the local LangGraph scaffold |
| `fhe-endpoint-smoke` | Smoke-test a remote OpenAI-compatible endpoint |

## Typical Commands

Local test sweep:

```bash
uv run --python 3.12 python3 -m unittest discover -s tests -p "test_*.py"
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

## Artifact Layout

The directories that matter most:

| Path | Contents |
| --- | --- |
| `data/processed/` | Materialized Tier 1, Stage 1, Stage 2, and Stage 3 datasets |
| `runs/stage1/` | Official LoRA runs and MIA outputs |
| `runs/stage2/baseline/` | Baseline replay outputs without the Stage 3 filter |
| `runs/stage2/filtered/` | Integrated reruns with plaintext and FHE filters active |
| `runs/stage3/plaintext/` | Plaintext filter model, embeddings, selection sweep, and metrics |
| `runs/stage3/fhe/` | FHE parity, latency, and compiled-bundle manifest artifacts |

Heavy Git LFS artifacts:

- Stage 1 adapter weights and trainer checkpoints
- Stage 3 compiled OpenFHE bundle under `runs/stage3/fhe/compiled/`

Repo-visible lightweight artifacts:

- JSON summaries and metrics
- CSV comparisons and sweeps
- the Stage 3 FHE compiled bundle manifest

## Source Layout

The code is split by experiment role:

| Path | Role |
| --- | --- |
| `src/experiment/data_gen/` | Tier 1, Stage 1, Stage 2, and Stage 3 dataset generation |
| `src/experiment/chat_render/` | Shared benign message/response rendering |
| `src/experiment/train_qwen/` | Stage 1 LoRA training pipeline |
| `src/experiment/mia/` | Stage 1 membership inference evaluation |
| `src/experiment/eval/` | Stage 2 replay, leakage scoring, and integrated filtered reruns |
| `src/experiment/filter_train/` | Stage 3 plaintext embedding + logistic regression filter |
| `src/experiment/fhe/` | Stage 3 CKKS/OpenFHE scoring, parity, latency, and bundle persistence |
| `src/qwen_langgraph_demo/` | Historical LangGraph demo/runtime scaffold |

## LangGraph Demo

The demo graph is still:

```text
intake -> filter_middleware -> fraud_scorer -> router
```

Run it locally with:

```bash
uv run fhe-experiment-run --json
```

That path is useful for local runtime smoke checks and demo purposes. It is not the official source of the reported experiment metrics.

## Current Result Pointers

The most important repo-visible outputs are:

- `runs/stage1/official_runs_summary.json`
- `runs/stage1/mia/mia_summary.json`
- `runs/stage2/baseline/stage2_summary.json`
- `runs/stage3/plaintext/stage3_plaintext_metrics.json`
- `runs/stage3/fhe/stage3_fhe_metrics.json`
- `runs/stage2/filtered/stage2_filtered_summary.json`
- `runs/stage2/filtered/filter_parity_summary.json`

For the concise human-readable summary, start with [../RESULTS.md](../RESULTS.md).
