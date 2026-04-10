import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a basic prompt to the remote vLLM endpoint.")
    parser.add_argument("prompt", nargs="?", default="Say hello in one sentence and confirm the model name.")
    parser.add_argument("--max-tokens", type=int, default=1600, help="Maximum completion tokens.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    )

    messages = [
        {"role": "system", "content": "Answer directly and completely. Streaming is enabled. Reasoning is allowed if the model emits it."},
        {"role": "user", "content": args.prompt},
    ]

    if args.no_stream:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "Qwen/Qwen3.5-4B"),
            messages=messages,
            max_completion_tokens=args.max_tokens,
        )
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning", "") or ""
        if reasoning.strip():
            print(reasoning)
        if content.strip():
            print(content)
        if not reasoning.strip() and not content.strip():
            print("[empty response content]")
        return

    stream = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "Qwen/Qwen3.5-4B"),
        messages=messages,
        max_completion_tokens=args.max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )

    saw_reasoning = False
    saw_content = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning", None)
        content = getattr(delta, "content", None)
        if reasoning:
            if not saw_reasoning:
                print("[reasoning]")
                saw_reasoning = True
            print(reasoning, end="", flush=True)
        if content:
            if not saw_content:
                if saw_reasoning:
                    print("\n\n[content]")
                saw_content = True
            print(content, end="", flush=True)

    if saw_reasoning or saw_content:
        print()
    else:
        print("[empty streamed response]")
