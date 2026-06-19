from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, normalize_text
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Advanced agent with session memory, persistent profile memory, and compaction."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.user_fact_cache: dict[str, dict[str, str]] = {}
        self.langchain_agent = None
        if not force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        facts = extract_profile_updates(message)
        for key, value in facts.items():
            self.profile_store.upsert_fact(user_id, key, value)
        if facts:
            cache = self.user_fact_cache.setdefault(user_id, {})
            cache.update(facts)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)

        agent_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "facts_updated": facts,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        thread_context = self.compact_memory.context(thread_id)
        summary = str(thread_context.get("summary", ""))
        messages = thread_context.get("messages", [])
        if isinstance(messages, list):
            message_text = "\n".join(item.get("content", "") for item in messages if isinstance(item, dict))
        else:
            message_text = ""
        profile_tokens = max(1, estimate_tokens(profile_text) // 4)
        summary_tokens = max(1, estimate_tokens(summary) // 2)
        recent_tokens = estimate_tokens(message_text)
        return profile_tokens + summary_tokens + recent_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = {**self.profile_store.facts(user_id), **self.user_fact_cache.get(user_id, {})}
        lowered = normalize_text(message)

        def fact(*keys: str) -> str | None:
            for key in keys:
                value = facts.get(key.lower())
                if value:
                    return value
            return None

        answers: list[str] = []
        if "ten" in lowered:
            name = fact("name")
            if name:
                answers.append(f"bạn tên là {name}")

        if "o dau" in lowered or "o" == lowered:
            location = fact("location")
            if location:
                answers.append(f"bạn đang ở {location}")

        if any(token in lowered for token in ["nghe", "nghe nghiep", "lam gi", "lam nghe"]):
            profession = fact("profession")
            if profession:
                answers.append(f"bạn đang làm {profession}")

        if "style" in lowered or "tra loi" in lowered:
            style = fact("style")
            if style:
                answers.append(f"style bạn thích là {style}")

        if "do uong" in lowered or "uong" in lowered:
            drink = fact("favorite_drink")
            if drink:
                answers.append(f"đồ uống yêu thích của bạn là {drink}")

        if "mon an" in lowered or "an" in lowered:
            food = fact("favorite_food")
            if food:
                answers.append(f"món ăn yêu thích của bạn là {food}")

        if "con" in lowered or "nuoi" in lowered or "pet" in lowered:
            pet = fact("pet")
            if pet:
                answers.append(f"bạn nuôi {pet}")

        if answers:
            if len(answers) == 1:
                return f"Ngắn gọn: {answers[0]}."
            return "Ngắn gọn:\n- " + "\n- ".join(answers) + "."

        current_facts = []
        for key in ["name", "location", "profession", "style", "favorite_drink", "favorite_food", "pet"]:
            value = facts.get(key)
            if value:
                current_facts.append(f"{key}: {value}")
        if current_facts:
            return "Mình nhớ các fact chính: " + "; ".join(current_facts[:4]) + "."

        profile_updates = extract_profile_updates(message)
        if profile_updates:
            return "Mình đã cập nhật hồ sơ của bạn."

        return "Mình đã ghi nhận và sẽ giữ ngắn gọn như bạn thích."

    def _maybe_build_langchain_agent(self):
        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        facts = extract_profile_updates(message)
        for key, value in facts.items():
            self.profile_store.upsert_fact(user_id, key, value)
        if facts:
            cache = self.user_fact_cache.setdefault(user_id, {})
            cache.update(facts)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except Exception:
            return self._reply_offline(user_id, thread_id, message)

        profile_text = self.profile_store.read_text(user_id).strip()
        thread_context = self.compact_memory.context(thread_id)
        summary = str(thread_context.get("summary", "")).strip()
        recent_messages = thread_context.get("messages", [])

        system_prompt = (
            "You are a concise assistant in Vietnamese.\n"
            "Use the user's profile memory when relevant.\n"
            "Answer briefly, with bullet points when helpful.\n"
            f"Profile memory:\n{profile_text}\n\n"
            f"Thread summary:\n{summary or '(none)'}"
        )

        chat_messages = [SystemMessage(content=system_prompt)]
        if isinstance(recent_messages, list):
            for item in recent_messages:
                if not isinstance(item, dict):
                    continue
                role = item.get("role", "user")
                content = item.get("content", "")
                if role == "assistant":
                    chat_messages.append(AIMessage(content=content))
                else:
                    chat_messages.append(HumanMessage(content=content))
        chat_messages.append(HumanMessage(content=message))

        result = self.langchain_agent.invoke(chat_messages)
        response = getattr(result, "content", str(result))

        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "facts_updated": facts,
        }
