from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig."""

    try:  # optional dependency; safe to ignore when not installed
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    judge_provider = os.getenv("JUDGE_PROVIDER", provider).strip().lower()
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)

    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "900"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "8"))

    def make_provider_config(prefix: str, provider_name: str, model: str) -> ProviderConfig:
        base_url = os.getenv(f"{prefix}_BASE_URL")
        api_key = os.getenv(f"{prefix}_API_KEY")
        return ProviderConfig(
            provider=provider_name,
            model_name=model,
            temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
            api_key=api_key,
            base_url=base_url,
        )

    model = make_provider_config(provider.upper(), provider, model_name)
    judge_model = make_provider_config("JUDGE", judge_provider, judge_model_name)

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )
