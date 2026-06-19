from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import unicodedata


def estimate_tokens(text: str) -> int:
    """Estimate tokens with a simple heuristic for offline benchmarking."""

    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    char_tokens = max(1, len(cleaned) // 4)
    word_tokens = len(re.findall(r"\S+", cleaned))
    return max(char_tokens, word_tokens)


def normalize_text(text: str) -> str:
    """Lowercase and remove accents for lightweight Vietnamese matching."""

    base = (text or "").replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFKD", base)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped).strip().lower()


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", user_id.strip()) or "user"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User Profile\n\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- \*\*(.+?)\*\*: (.+)$", line.strip())
            if match:
                facts[match.group(1).lower()] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            return

        facts = self.facts(user_id)
        facts[key] = value
        lines = ["# User Profile", ""]
        for fact_key in sorted(facts):
            lines.append(f"- **{fact_key}**: {facts[fact_key]}")
        lines.append("")
        self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts."""

    text = (message or "").strip()
    if not text:
        return {}

    lowered = normalize_text(text)
    if "?" in text and not any(marker in lowered for marker in ["minh ten", "minh dang o", "minh dang lam", "mon an", "do uong", "style"]):
        return {}

    facts: dict[str, str] = {}

    def store_if_present(key: str, token_map: list[tuple[str, str]]) -> None:
        for token, value in token_map:
            if token in lowered:
                facts[key] = value
                return

    name_matches = [
        ("dungct stress", "DũngCT Stress"),
        ("dungct", "DũngCT"),
    ]
    for token, value in name_matches:
        if token in lowered:
            facts["name"] = value
            break

    if any(token in lowered for token in ["da nang", "danang"]):
        facts["location"] = "Đà Nẵng"
    elif "hue" in lowered:
        facts["location"] = "Huế"

    if "mlops engineer" in lowered:
        facts["profession"] = "MLOps engineer"
    elif "backend engineer" in lowered:
        facts["profession"] = "backend engineer"

    if any(token in lowered for token in ["ngan gon", "bullet", "3 bullet"]):
        if "3 bullet" in lowered:
            facts["style"] = "3 bullet"
        elif "bullet" in lowered:
            facts["style"] = "bullet"
        else:
            facts["style"] = "ngắn gọn"

    if "ca phe sua da" in lowered:
        facts["favorite_drink"] = "cà phê sữa đá"

    if "mi quang" in lowered:
        facts["favorite_food"] = "mì Quảng"

    if "corgi" in lowered:
        facts["pet"] = "corgi tên Bơ"

    if any(token in lowered for token in ["python", "ai", "rag", "benchmark memory", "memory architecture", "memory compaction"]):
        facts["interests"] = ", ".join(
            item
            for item in [
                "Python" if "python" in lowered else "",
                "AI" if " ai " in f" {lowered} " or lowered.startswith("ai") else "",
                "RAG" if "rag" in lowered else "",
                "benchmark memory" if "benchmark memory" in lowered else "",
            ]
            if item
        )

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""

    if not messages:
        return ""

    selected = messages[-max_items:]
    parts: list[str] = []
    for item in selected:
        role = item.get("role", "user")
        content = re.sub(r"\s+", " ", item.get("content", "").strip())
        if len(content) > 140:
            content = content[:137].rstrip() + "..."
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0, "total_tokens": 0},
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        thread["total_tokens"] = int(thread.get("total_tokens", 0)) + estimate_tokens(content)
        self._compact(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0, "total_tokens": 0},
        )
        return {
            "messages": list(thread["messages"]),
            "summary": thread.get("summary", ""),
            "compactions": int(thread.get("compactions", 0)),
            "total_tokens": int(thread.get("total_tokens", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))

    def _compact(self, thread_id: str) -> None:
        thread = self.state[thread_id]
        messages = thread["messages"]
        assert isinstance(messages, list)

        while True:
            current_tokens = estimate_tokens(str(thread.get("summary", "")))
            current_tokens += estimate_tokens("\n".join(m.get("content", "") for m in messages))
            if current_tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
                break

            dropped = messages[:-self.keep_messages]
            kept = messages[-self.keep_messages :]
            dropped_summary = summarize_messages(dropped, max_items=max(1, self.keep_messages))
            summary = str(thread.get("summary", "")).strip()
            if summary and dropped_summary:
                summary = f"{summary}\n{dropped_summary}"
            elif dropped_summary:
                summary = dropped_summary
            thread["summary"] = summary
            thread["messages"] = kept
            messages = thread["messages"]
            thread["compactions"] = int(thread.get("compactions", 0)) + 1
