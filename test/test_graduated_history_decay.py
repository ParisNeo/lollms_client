from types import SimpleNamespace

from lollms_client.lollms_personality.lollms_personality import (
    LollmsPersonality,
    _DECAY_TAG,
    _HISTORY_HOT_ZONE,
)


def _make_agent() -> LollmsPersonality:
    return LollmsPersonality(name="decay-test-agent")


def _tool_result_message(tool_name: str, line_count: int, status: str = "SUCCESS") -> str:
    body = "\n".join(f"line {i}" for i in range(line_count))
    return (
        f"=== TOOL RESULT: {tool_name} ===\n"
        f'<tool_result name="{tool_name}" status="{status}">\n{body}\n</tool_result>\n\n'
        "[SYSTEM DIRECTIVE: proceed with the task.]"
    )


def _build_history(total_messages: int):
    history = []
    for i in range(total_messages):
        if i % 2 == 0:
            history.append(SimpleNamespace(sender_type="assistant", content=f"Assistant round text {i}"))
        else:
            history.append(SimpleNamespace(sender_type="user", content=_tool_result_message("tool_demo", 80)))
    return history


def test_hot_zone_messages_remain_verbatim():
    agent = _make_agent()
    history = _build_history(30)
    original_tail = [vh.content for vh in history[-_HISTORY_HOT_ZONE:]]
    agent._apply_graduated_history_decay(history)
    assert [vh.content for vh in history[-_HISTORY_HOT_ZONE:]] == original_tail


def test_old_tool_results_are_truncated():
    agent = _make_agent()
    history = _build_history(30)
    agent._apply_graduated_history_decay(history)
    old_user = history[1]
    assert _DECAY_TAG in old_user.content
    assert "line 0" in old_user.content
    assert "line 40" not in old_user.content
    assert "line 79" in old_user.content
    assert len(old_user.content) < len(_tool_result_message("tool_demo", 80))


def test_failed_results_keep_more_context_than_success():
    agent = _make_agent()

    failed_history = [SimpleNamespace(sender_type="user", content=_tool_result_message("tool_fail", 80, status="FAILED"))]
    failed_history.extend(SimpleNamespace(sender_type="user", content="filler") for _ in range(20))
    agent._apply_graduated_history_decay(failed_history)
    assert _DECAY_TAG in failed_history[0].content
    assert "line 25" in failed_history[0].content

    success_history = [SimpleNamespace(sender_type="user", content=_tool_result_message("tool_ok", 80))]
    success_history.extend(SimpleNamespace(sender_type="user", content="filler") for _ in range(20))
    agent._apply_graduated_history_decay(success_history)
    assert _DECAY_TAG in success_history[0].content
    assert "line 25" not in success_history[0].content


def test_decay_is_idempotent():
    agent = _make_agent()
    history = _build_history(30)
    agent._apply_graduated_history_decay(history)
    snapshot = [vh.content for vh in history]
    agent._apply_graduated_history_decay(history)
    assert [vh.content for vh in history] == snapshot


def test_short_history_is_untouched():
    agent = _make_agent()
    history = _build_history(10)
    original = [vh.content for vh in history]
    agent._apply_graduated_history_decay(history)
    assert [vh.content for vh in history] == original


def test_old_artifact_bodies_are_collapsed():
    agent = _make_agent()
    big_artifact = '<artifact name="script.py" type="code">\n' + "print('x')\n" * 100 + "</artifact>"
    history = [SimpleNamespace(sender_type="assistant", content=big_artifact)]
    history.extend(SimpleNamespace(sender_type="user", content="filler") for _ in range(20))
    agent._apply_graduated_history_decay(history)
    assert "print('x')" not in history[0].content
    assert 'name="script.py"' in history[0].content
    assert _DECAY_TAG in history[0].content


def test_chronicle_block_lists_actions():
    agent = _make_agent()
    calls = [
        {"round": 1, "name": "tool_read_file", "parameters": {"file_name": "config.json"}},
        {"round": 2, "name": "tool_grep_files", "parameters": {"pattern": "search_mode"}},
    ]
    results = [{"success": True}, {"success": False}]
    block = agent._build_turn_chronicle_block(calls, results)
    assert "TURN CHRONICLE" in block
    assert "tool_read_file" in block
    assert "file_name=config.json" in block
    assert "OK" in block
    assert "FAILED" in block


def test_chronicle_empty_without_calls():
    agent = _make_agent()
    assert agent._build_turn_chronicle_block([], []) == ""


def test_chronicle_caps_entries():
    agent = _make_agent()
    calls = [
        {"round": i, "name": "tool_x", "parameters": {"file_name": f"f{i}"}}
        for i in range(100)
    ]
    results = [{"success": True}] * 100
    block = agent._build_turn_chronicle_block(calls, results)
    assert "omitted" in block
    assert "file_name=f0)" not in block
    assert "file_name=f99)" in block


def test_texts_are_repetitive_detects_paraphrased_preamble():
    text_a = (
        "I'll investigate this datastore persistence issue. Let me start by exploring "
        "the codebase to understand how datastores are activated and managed in discussions."
    )
    text_b = (
        "I'll investigate the datastore persistence issue. Let me start by exploring "
        "the codebase structure to understand how datastores are managed in discussions."
    )
    assert LollmsPersonality._texts_are_repetitive(text_a, text_b)


def test_texts_are_repetitive_allows_distinct_texts():
    text_a = "The migration plan was executed and every file was moved to its target folder."
    text_b = "Now I will run the test suite to validate the new module imports work."
    assert not LollmsPersonality._texts_are_repetitive(text_a, text_b)


def test_texts_are_repetitive_ignores_empty_inputs():
    assert not LollmsPersonality._texts_are_repetitive("", "some text")
    assert not LollmsPersonality._texts_are_repetitive("some text", "")


def test_last_real_assistant_text_skips_marker_messages():
    history = [
        SimpleNamespace(sender_type="assistant", content="Real preamble about the datastore task."),
        SimpleNamespace(sender_type="user", content="tool report"),
        SimpleNamespace(sender_type="assistant", content="[Assistant repeated its previous preamble; actions were executed.]"),
    ]
    assert LollmsPersonality._last_real_assistant_text(history) == "Real preamble about the datastore task."


def test_last_real_assistant_text_returns_empty_without_real_text():
    history = [
        SimpleNamespace(sender_type="user", content="tool report"),
        SimpleNamespace(sender_type="assistant", content="[Assistant executed batched actions]"),
    ]
    assert LollmsPersonality._last_real_assistant_text(history) == ""
    assert LollmsPersonality._last_real_assistant_text([]) == ""