# Setup Guide

This package now targets the FHE experiment scaffold rather than the old translation demo.

There are two distinct environments to keep in mind:

1. local macOS for scaffold work, config loading, graph construction, and non-GPU development
2. remote NVIDIA Linux for later GPU-heavy stages such as Qwen training and model serving

This ticket only needs the local setup, but the endpoint smoke test can optionally use a remote OpenAI-compatible model server.

## Local Setup

From `experiment_runtime/`:

```bash
uv venv --python 3.12 --clear
uv sync --python 3.12
cp .env.example .env
```

Local verification:

```bash
uv run fhe-experiment-run
uv run --python 3.12 python -m unittest discover -s tests -p "test_*.py"
```

## Environment Variables

Current `.env.example`:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8001/v1
MODEL_NAME=Qwen/Qwen2-1.5B-Instruct
OPENAI_API_KEY=EMPTY
GPU_TARGET=local macOS scaffold; remote NVIDIA Linux required later for GPU stages
```

`GPU_TARGET` is informational only.

## Optional Remote Endpoint Smoke Test

If you already have a remote OpenAI-compatible endpoint available, you can test connectivity with:

```bash
uv run fhe-endpoint-smoke "Summarize the purpose of the fraud scoring agent."
```

This is optional for the current scaffold ticket.

## Remote Endpoint Notes

Later stages will likely use:

- a remote NVIDIA Linux box
- `vLLM` or another OpenAI-compatible serving layer
- Qwen2 family models, especially `Qwen/Qwen2-1.5B-Instruct` early and `Qwen/Qwen2-7B-Instruct` later

Suggested baseline environment variables:

```bash
export SSH_USER="root"
export SSH_HOST="YOUR_REMOTE_HOST"
export SSH_PORT="YOUR_REMOTE_SSH_PORT"
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
export REMOTE_DIR="/workspace/fhe-experiment-runtime"
export REMOTE_VLLM_PORT="8000"
export LOCAL_TUNNEL_PORT="8001"
```

## Optional SSH Tunnel Pattern

If you are exposing a remote OpenAI-compatible server on port `8000`, this local tunnel shape is still a reasonable default:

```bash
ssh -f -N -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" -L "${LOCAL_TUNNEL_PORT}:localhost:${REMOTE_VLLM_PORT}"
```

Then the local `.env` can keep:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8001/v1
```

## Troubleshooting

### `uv sync --python 3.12` fails

Check:

- `uv` is installed locally
- Python `3.12` is available or downloadable through `uv`

### The local scaffold runs but the endpoint smoke test fails

Check:

- the remote API is actually running
- the tunnel is active
- `.env` points to the right `OPENAI_BASE_URL`
- `.env` uses the correct `MODEL_NAME`

### The local scaffold blocks every request

The current filter is a deterministic placeholder, not the final Stage 3 classifier. This is expected to be replaced later.
