# Qwen Translation Demo

This project contains one LangGraph demo only:

- a `coordinator` agent
- a general `translate(text, target_language)` tool
- a `presenter` step that returns the final formatted output

The model runs remotely on the Vast box through `vLLM`.
The LangGraph app runs locally and talks to the model through an SSH tunnel.

For full infrastructure bring-up instructions, see [SETUP.md](SETUP.md).

## How The Demo Works

The flow is:

1. The `coordinator` node receives the input sentence.
2. The coordinator decides which of the hardcoded supported languages to use.
3. The coordinator invokes the general `translate` tool once per selected language.
4. Each tool invocation makes its own call to the shared remote `vLLM` endpoint.
5. The coordinator collects the tool outputs and formats the final translation block.
6. The `presenter` node returns the final result.

The supported languages are hardcoded in the app:

- French
- Spanish
- German
- Italian
- Portuguese
- Japanese
- Arabic

This version has **more autonomy than the earlier fixed-node demo**, because the coordinator decides which translation tool calls to make. However, it is still only **moderately autonomous**:

- the graph shape is still fixed in code
- the available languages are fixed in code
- the tool name and tool schema are fixed in code
- the coordinator decides which hardcoded languages to use, but it does not invent new tools or restructure the workflow

So the deterministic code still controls the overall workflow, while the model is responsible for:

- deciding which supported languages to invoke
- invoking the `translate` tool with arguments
- formatting the final translation output

## Graph Diagram

```text
                 +------------------+
Input sentence ->|   coordinator    |
                 +------------------+
                           |
                           | issues translation tool calls
                           v
                 +------------------+
                 |  translate tool  |
                 | (called N times) |
                 +------------------+
                           |
                           | each call hits remote vLLM
                           v
                 +------------------+
                 |    presenter      |
                 +------------------+
                           |
                           v
                    Final formatted output
```

## System Diagram

```text
+-------------------------+          SSH tunnel           +----------------------+
| Local LangGraph app     | 127.0.0.1:8001 -> :8000      | Remote Vast GPU box  |
|                         |------------------------------>|                      |
| coordinator             |                               | vLLM                 |
| translate tool wrapper  |<------------------------------| Qwen/Qwen3.5-4B      |
| presenter               |        OpenAI-compatible API  | RTX PRO 6000         |
+-------------------------+                               +----------------------+
```

## Files

- `src/qwen_langgraph_demo/main.py`
  - coordinator + translation tool LangGraph app
- `src/qwen_langgraph_demo/test_endpoint.py`
  - basic endpoint smoke test
- `.env.example`
  - local environment defaults

## Local Usage

Create the local environment:

```bash
uv venv --python 3.11 --clear
uv sync --python 3.11
cp .env.example .env
```

## Basic Endpoint Test

Check the models endpoint:

```bash
curl http://127.0.0.1:8001/v1/models
```

Send a basic prompt:

```bash
uv run test-vllm-endpoint "What is the meaning of life?"
```

You can raise the response budget if needed:

```bash
uv run test-vllm-endpoint --max-tokens 3000 "What is the meaning of life?"
```

## Run The Translation Demo

```bash
uv run qwen-langgraph-demo "I would like a coffee, please."
```

Example output shape:

```text
Original: I would like a coffee, please.
French: ...
Spanish: ...
German: ...
Italian: ...
Portuguese: ...
Japanese: ...
Arabic: ...
```

The commands above assume the remote `vLLM` server is already running and your SSH tunnel is already active. See [SETUP.md](SETUP.md) for that setup.
