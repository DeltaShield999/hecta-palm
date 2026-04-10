# Setup Guide

This document describes how to reproduce the current working setup:

- remote Vast GPU box running `vLLM`
- local LangGraph app running from this repo
- SSH tunnel connecting the local app to the remote model server

At the end, you should be able to run:

```bash
uv run test-vllm-endpoint "What is the meaning of life?"
uv run qwen-langgraph-demo "I would like a coffee, please."
```

## Required Inputs

Set these variables in your local shell first, then the rest of this guide is copy-paste ready:

```bash
export SSH_USER="root"
export SSH_HOST="YOUR_VAST_HOST"
export SSH_PORT="YOUR_VAST_SSH_PORT"
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
export REMOTE_DIR="/workspace/qwen-langgraph-demo"
export REMOTE_VLLM_PORT="8000"
export LOCAL_TUNNEL_PORT="8001"
```

Notes:

- `SSH_HOST` is the public IP or hostname of your Vast instance
- `SSH_PORT` is the SSH port assigned by Vast
- `HF_TOKEN` is strongly recommended for reliable model downloads
- `REMOTE_DIR` can be changed if you want a different remote project path
- `REMOTE_VLLM_PORT=8000` and `LOCAL_TUNNEL_PORT=8001` are the defaults used in this guide

## Current Working Topology

```text
+-------------------------+          SSH tunnel           +----------------------+
| Local LangGraph app     | 127.0.0.1:8001 -> :8000      | Remote Vast GPU box  |
|                         |------------------------------>|                      |
| coordinator             |                               | vLLM                 |
| translate tool wrapper  |<------------------------------| Qwen/Qwen3.5-4B      |
| presenter               |        OpenAI-compatible API  | RTX PRO 6000         |
+-------------------------+                               +----------------------+
```

## Versions And Assumptions

This repo is currently working with:

- local Python via `uv` using `3.11`
- remote `vLLM 0.17.1`
- remote model `Qwen/Qwen3.5-4B`
- remote GPU class example: `RTX PRO 6000 Blackwell 96GB`
- remote API port: `8000`
- local tunnel port: `8001`

This setup is not tied to that exact GPU. A box like `H100` is also fine for this model.

The remote server currently runs Qwen with:

- `--reasoning-parser qwen3`
- `--default-chat-template-kwargs '{"enable_thinking": false}'`
- `--enable-auto-tool-choice`
- `--tool-call-parser qwen3_coder`

That `enable_thinking=false` setting is important. It keeps the model from spending too many tokens on visible reasoning output.

## 1. Provision The Remote Box

Use a Linux box with:

- NVIDIA GPU
- enough VRAM for `Qwen/Qwen3.5-4B`
- internet access for model download
- SSH access

Connect with your own SSH values:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST"
```

If you want the original local port forward that was used early in setup:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" -L 8080:localhost:8080
```

That `8080` tunnel is not used by the LangGraph app. The app uses a separate tunnel on local `8001`.

## 2. Prepare The Remote Environment

SSH into the box and create or use the project directory:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" <<EOF
mkdir -p "$REMOTE_DIR"
cd "$REMOTE_DIR"
pwd
EOF
```

Make sure these remote prerequisites exist:

- `python3`
- `uv`
- `tmux`
- `curl`

Useful checks:

```bash
python3 --version
uv --version
tmux -V
curl --version
```

Then create the remote environment and install `vLLM`:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" <<EOF
cd "$REMOTE_DIR"
python3 --version
uv --version
tmux -V
curl --version
uv venv --python 3.11
source .venv/bin/activate
uv pip install --python .venv/bin/python --upgrade "vllm==0.17.1"
EOF
```

If you want to mirror the remote project structure used in the current working setup, create:

- `scripts/`
- `scripts/serve_qwen.sh`
- a local Hugging Face cache directory at `/workspace/.hf_home`

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" <<EOF
cd "$REMOTE_DIR"
mkdir -p scripts
mkdir -p /workspace/.hf_home
EOF
```

## 3. Configure Hugging Face Access

Large model downloads are more reliable with a Hugging Face token.

On the remote box, export:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" <<EOF
export HF_TOKEN="$HF_TOKEN"
echo "HF_TOKEN set for current shell"
EOF
```

If you want it persisted for future shells:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" <<EOF
grep -q 'export HF_TOKEN=' ~/.bashrc || echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc
EOF
```

The current setup also uses:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" <<EOF
grep -q 'export HF_HOME=' ~/.bashrc || echo 'export HF_HOME=/workspace/.hf_home' >> ~/.bashrc
EOF
```

## 4. Create The Remote Serve Script

Create `"$REMOTE_DIR/scripts/serve_qwen.sh"` with:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" <<EOF
mkdir -p "$REMOTE_DIR/scripts"
cat > "$REMOTE_DIR/scripts/serve_qwen.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$REMOTE_DIR"
source .venv/bin/activate

export HF_HOME=/workspace/.hf_home
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}

vllm serve Qwen/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port $REMOTE_VLLM_PORT \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
SCRIPT
chmod +x "$REMOTE_DIR/scripts/serve_qwen.sh"
EOF
```

## 5. Run The Remote Server In tmux

The clean operator pattern is to keep `vLLM` in its own tmux session.

On the remote box:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "tmux new-session -d -A -s ssh_tmux 'cd \"$REMOTE_DIR\" && ./scripts/serve_qwen.sh'"
```

Wait for startup to complete. A healthy startup ends with lines like:

- `Application startup complete.`
- route registration including `/v1/models`

You can later reattach read-only with:

```bash
ssh -t -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "tmux attach -t ssh_tmux -r"
```

## 6. Verify The Remote API On The Box

From the remote box itself:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "curl http://127.0.0.1:$REMOTE_VLLM_PORT/v1/models"
```

You should see `Qwen/Qwen3.5-4B` in the response.

## 7. Prepare The Local Repo

On your local machine, in this repo:

```bash
uv venv --python 3.11 --clear
uv sync --python 3.11
cp .env.example .env
```

The current `.env.example` points to:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8001/v1
MODEL_NAME=Qwen/Qwen3.5-4B
OPENAI_API_KEY=EMPTY
GPU_TARGET=remote RTX PRO 6000 Blackwell 96GB via SSH tunnel
```

`GPU_TARGET` is optional and only used as descriptive metadata.

## 8. Create The Local Tunnel

The LangGraph app is wired by default to local `8001`, so create this tunnel:

```bash
ssh -f -N -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" -L "${LOCAL_TUNNEL_PORT}:localhost:${REMOTE_VLLM_PORT}"
```

That maps:

- local `127.0.0.1:$LOCAL_TUNNEL_PORT`
- to remote `127.0.0.1:$REMOTE_VLLM_PORT`

If you change the local tunnel port, update `.env` accordingly.

## 9. Verify The Tunnel

From your local machine:

```bash
curl "http://127.0.0.1:$LOCAL_TUNNEL_PORT/v1/models"
```

Expected: a JSON response listing `Qwen/Qwen3.5-4B`.

## 10. Test The Raw Endpoint From This Repo

Use the included smoke test:

```bash
uv run test-vllm-endpoint "What is the meaning of life?"
```

Or:

```bash
uv run test-vllm-endpoint --max-tokens 3000 "Translate this sentence: I would like a coffee, please."
```

## 11. Run The LangGraph Demo Locally

```bash
uv run qwen-langgraph-demo "I would like a coffee, please."
```

This should:

- run the local coordinator node
- have the coordinator call the `translate` tool repeatedly
- send those tool requests to the remote `vLLM` endpoint through the tunnel
- return a final translation block

## What The Current Demo Is

The current demo is:

- one LangGraph `coordinator`
- one general `translate(text, target_language)` tool
- one `presenter` node

The supported languages are hardcoded in code:

- French
- Spanish
- German
- Italian
- Portuguese
- Japanese
- Arabic

The coordinator has moderate autonomy:

- it decides which tool invocations to make
- but the workflow shape and tool schema are still fixed in code

## Troubleshooting

### `curl http://127.0.0.1:$LOCAL_TUNNEL_PORT/v1/models` fails locally

Check:

- remote `vLLM` is still running
- your SSH tunnel is still active
- remote server is listening on `$REMOTE_VLLM_PORT`

Useful checks:

```bash
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "curl http://127.0.0.1:$REMOTE_VLLM_PORT/v1/models"
```

on the remote box, and:

```bash
lsof -nP -iTCP:"$LOCAL_TUNNEL_PORT" -sTCP:LISTEN
```

locally.

### The model outputs too much visible reasoning

Keep this in the remote serve script:

```bash
--default-chat-template-kwargs '{"enable_thinking": false}'
```

### Port `8080` conflicts on Vast

In the current box, remote `8080` is used by Jupyter. Do not use it for `vLLM`.

Use:

- remote `$REMOTE_VLLM_PORT` for `vLLM`
- local `$LOCAL_TUNNEL_PORT` for the SSH tunnel

### Model downloads are slow or flaky

Use a Hugging Face token and set:

```bash
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
export HF_HOME=/workspace/.hf_home
```
