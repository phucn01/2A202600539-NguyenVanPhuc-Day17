from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import normalize_text


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    answer_norm = normalize_text(answer)
    matched = 0
    for item in expected:
        if normalize_text(item) in answer_norm:
            matched += 1
    if matched == 0:
        return 0.0
    if matched == len(expected):
        return 1.0
    if matched >= (len(expected) + 1) // 2:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    score = recall_points(answer, expected)
    answer_norm = normalize_text(answer)
    if answer_norm.startswith("ngan gon") or answer_norm.startswith("minh nho"):
        score += 0.15
    if "\n-" in answer or answer.count("\n") >= 2:
        score += 0.1
    if len(answer.strip()) < 160:
        score += 0.1
    return round(min(score, 1.0), 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    start_sizes: dict[str, int] = {}
    end_sizes: dict[str, int] = {}
    compactions = 0

    for conversation in conversations:
        user_id = conversation["user_id"]
        conv_id = conversation["id"]
        train_thread = f"{conv_id}:train"
        recall_thread = f"{conv_id}:recall"

        if hasattr(agent, "memory_file_size"):
            start_sizes[user_id] = agent.memory_file_size(user_id)

        for turn in conversation.get("turns", []):
            result = agent.reply(user_id, train_thread, turn)
            total_agent_tokens += int(result.get("agent_tokens", 0))
            total_prompt_tokens += int(result.get("prompt_tokens", 0))

        for recall in conversation.get("recall_questions", []):
            result = agent.reply(user_id, recall_thread, recall["question"])
            answer = result.get("response", "")
            total_agent_tokens += int(result.get("agent_tokens", 0))
            total_prompt_tokens += int(result.get("prompt_tokens", 0))
            recall_scores.append(recall_points(answer, recall.get("expected_contains", [])))
            quality_scores.append(heuristic_quality(answer, recall.get("expected_contains", [])))

        if hasattr(agent, "memory_file_size"):
            end_sizes[user_id] = agent.memory_file_size(user_id)
        if hasattr(agent, "compaction_count"):
            compactions += agent.compaction_count(train_thread) + agent.compaction_count(recall_thread)

    growth = sum(end_sizes.get(user_id, 0) - start_sizes.get(user_id, 0) for user_id in start_sizes)
    recall_score = round(sum(recall_scores) / len(recall_scores), 2) if recall_scores else 0.0
    quality = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=recall_score,
        response_quality=quality,
        memory_growth_bytes=growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [headers]
    for row in rows:
        table.append(
            [
                row.agent_name,
                str(row.agent_tokens_only),
                str(row.prompt_tokens_processed),
                f"{row.recall_score:.2f}",
                f"{row.response_quality:.2f}",
                str(row.memory_growth_bytes),
                str(row.compactions),
            ]
        )

    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    lines = []
    lines.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(table[0])))
    lines.append(" | ".join("-" * widths[i] for i in range(len(headers))))
    for row in table[1:]:
        lines.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _section_report(title: str, rows: list[BenchmarkRow]) -> str:
    best_recall = max(rows, key=lambda row: row.recall_score)
    best_prompt = min(rows, key=lambda row: row.prompt_tokens_processed)
    best_memory = min(rows, key=lambda row: row.memory_growth_bytes)
    lines = [f"## {title}", ""]
    lines.append(format_rows(rows))
    lines.append("")
    lines.append(f"- Best recall: {best_recall.agent_name} ({best_recall.recall_score:.2f})")
    lines.append(f"- Lowest prompt cost: {best_prompt.agent_name} ({best_prompt.prompt_tokens_processed})")
    lines.append(f"- Smallest memory growth: {best_memory.agent_name} ({best_memory.memory_growth_bytes} bytes)")
    return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    root = config.base_dir

    standard = load_conversations(root / "data" / "conversations.json")
    stress = load_conversations(root / "data" / "advanced_long_context.json")

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    standard_rows = [
        run_agent_benchmark("Baseline", baseline, standard, config),
        run_agent_benchmark("Advanced", advanced, standard, config),
    ]

    # Fresh instances for the stress benchmark so the comparison is not polluted.
    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)
    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress, config),
        run_agent_benchmark("Advanced", advanced_stress, stress, config),
    ]

    report = [
        "# RESULT",
        "",
        "## Standard Benchmark",
        "",
        format_rows(standard_rows),
        "",
        "## Long-Context Stress Benchmark",
        "",
        format_rows(stress_rows),
        "",
        "## Analysis",
        "",
        "- Advanced should score higher on cross-session recall because it persists stable facts in `User.md`.",
        "- Baseline should keep memory growth near zero because it does not write persistent profile files.",
        "- Compaction matters most in the long-context stress run because it trims prompt context while preserving durable facts.",
    ]
    result_path = root / "RESULT.md"
    result_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(_section_report("Standard Benchmark", standard_rows))
    print()
    print(_section_report("Long-Context Stress Benchmark", stress_rows))
    print()
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
