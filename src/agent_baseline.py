from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, normalize_text
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline agent with within-session memory only."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        prompt_context = "\n".join(item["content"] for item in session.messages)
        prompt_tokens = estimate_tokens(prompt_context) + estimate_tokens(message)

        session.messages.append({"role": "user", "content": message})
        response = self._local_response(session.messages, message)
        session.messages.append({"role": "assistant", "content": response})

        agent_tokens = estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _local_response(self, messages: list[dict[str, str]], message: str) -> str:
        lowered = normalize_text(message)

        def search_latest(patterns: list[str]) -> str | None:
            for item in reversed(messages):
                content = normalize_text(item.get("content", ""))
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
            return None

        if "ten" in lowered:
            name = search_latest([r"(?:minh ten la|ten minh la|toi ten la)\s+([^\.\n,;]+)"])
            if name:
                return f"Mình nhớ bạn tên là {name}."

        if "o dau" in lowered or "o" == lowered:
            location = search_latest([r"(?:minh dang o|hien tai minh dang o|minh o)\s+([^\.\n,;]+)"])
            if location:
                return f"Mình nhớ bạn đang ở {location}."

        if any(token in lowered for token in ["nghe", "nghe nghiep", "lam gi", "lam nghe"]):
            profession = search_latest([r"(?:minh dang lam|minh la)\s+([^\.\n,;]+)"])
            if profession:
                return f"Mình nhớ bạn đang làm {profession}."

        if "style" in lowered or "tra loi" in lowered:
            style = search_latest([r"(?:tra loi|style tra loi)[^\n.]*?(ngan gon[^.\n,;]*|bullet[^.\n,;]*|3 bullet[^.\n,;]*)"])
            if style:
                return f"Mình nhớ style bạn thích là {style}."

        summary_items = [
            search_latest([r"(?:minh ten la|ten minh la|toi ten la)\s+([^\.\n,;]+)"]),
            search_latest([r"(?:minh dang o|hien tai minh dang o|minh o)\s+([^\.\n,;]+)"]),
            search_latest([r"(?:minh dang lam|minh la)\s+([^\.\n,;]+)"]),
        ]
        summary = "; ".join(sorted({item for item in summary_items if item}))
        if summary:
            return f"Mình nhớ một số thông tin trong thread này: {summary}."

        return "Mình đã ghi nhận ý của bạn."

    def _maybe_build_langchain_agent(self):
        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())

        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except Exception:
            return self._reply_offline(thread_id, message)

        system_prompt = (
            "You are a concise assistant. Answer in Vietnamese, keep responses short, "
            "and remember only the conversation in this thread."
        )
        chat_messages = [SystemMessage(content=system_prompt)]
        for item in session.messages:
            role = item.get("role", "user")
            content = item.get("content", "")
            if role == "assistant":
                chat_messages.append(AIMessage(content=content))
            else:
                chat_messages.append(HumanMessage(content=content))
        chat_messages.append(HumanMessage(content=message))

        result = self.langchain_agent.invoke(chat_messages)
        response = getattr(result, "content", str(result))

        prompt_tokens = estimate_tokens(system_prompt + "\n" + "\n".join(item["content"] for item in session.messages) + "\n" + message)
        agent_tokens = estimate_tokens(response)

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }
