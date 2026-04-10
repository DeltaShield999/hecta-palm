from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class ModelEndpointSettings:
    model_name: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    max_completion_tokens: int = 1_200

    @classmethod
    def from_env(cls) -> "ModelEndpointSettings":
        load_dotenv()
        return cls(
            model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2-1.5B-Instruct"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        )


def build_chat_model(settings: ModelEndpointSettings | None = None) -> ChatOpenAI:
    resolved = settings or ModelEndpointSettings.from_env()
    return ChatOpenAI(
        model=resolved.model_name,
        base_url=resolved.base_url,
        api_key=resolved.api_key,
        temperature=resolved.temperature,
        max_completion_tokens=resolved.max_completion_tokens,
    )
