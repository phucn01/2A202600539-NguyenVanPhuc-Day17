from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider aliases to the canonical provider name."""

    value = (value or "").strip().lower()
    aliases = {
        "anthorpic": "anthropic",
        "anthropic": "anthropic",
        "gpt": "openai",
        "openai": "openai",
        "gemini": "gemini",
        "google": "gemini",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "custom": "custom",
    }
    return aliases.get(value, value)


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the real chat model for the selected provider.

    The lab uses deterministic offline agents by default, so this helper only
    needs to work when the matching provider packages are installed.
    """

    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
        )
    if provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=config.base_url or os.getenv("CUSTOM_BASE_URL"),
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key or os.getenv("GEMINI_API_KEY"),
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url or os.getenv("OLLAMA_BASE_URL"):
            kwargs["base_url"] = config.base_url or os.getenv("OLLAMA_BASE_URL")
        return ChatOllama(**kwargs)
    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "langchain-openrouter is not installed in this environment."
            ) from exc

        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=config.base_url or os.getenv("OPENROUTER_BASE_URL"),
        )

    raise ValueError(f"Unsupported provider: {config.provider}")
