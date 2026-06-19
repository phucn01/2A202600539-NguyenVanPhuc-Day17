from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=120,
        compact_keep_messages=3,
        model=ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0),
        judge_model=ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    path = agent.profile_store.write_text("user-1", "# User Profile\n\n- **name**: DungCT\n")
    assert path.exists()
    assert "DungCT" in agent.profile_store.read_text("user-1")

    assert agent.profile_store.edit_text("user-1", "DungCT", "DzungCT")
    content = agent.profile_store.read_text("user-1")
    assert "DzungCT" in content


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    thread_id = "thread-1"
    for idx in range(10):
        agent.reply("user-1", thread_id, f"Message number {idx} with enough text to grow memory.")

    assert agent.compaction_count(thread_id) > 0
    context = agent.compact_memory.context(thread_id)
    assert len(context["messages"]) <= config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    user_id = "user-1"
    train_thread = "train"
    recall_thread = "recall"

    advanced.reply(user_id, train_thread, "Mình tên là DũngCT và mình đang ở Huế.")
    advanced.reply(user_id, train_thread, "Mình đang làm MLOps engineer.")
    baseline.reply(user_id, train_thread, "Mình tên là DũngCT và mình đang ở Huế.")
    baseline.reply(user_id, train_thread, "Mình đang làm MLOps engineer.")

    advanced_answer = advanced.reply(user_id, recall_thread, "Hiện tại mình đang ở đâu và làm nghề gì?")["response"]
    baseline_answer = baseline.reply(user_id, recall_thread, "Hiện tại mình đang ở đâu và làm nghề gì?")["response"]

    assert "Huế" in advanced_answer or "Hue" in advanced_answer
    assert "MLOps" in advanced_answer
    assert "Huế" not in baseline_answer and "Hue" not in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "user-1"
    thread_id = "long-thread"
    for idx in range(15):
        text = f"Turn {idx}: this is a long message to increase context and trigger compaction."
        baseline.reply(user_id, thread_id, text)
        advanced.reply(user_id, thread_id, text)

    assert advanced.prompt_token_usage(thread_id) <= baseline.prompt_token_usage(thread_id)
