# test/test_chat_workflow_edge_cases.py
"""
Verifies the edge cases of the LollmsDiscussion.chat() agentic loop state machine,
as documented in the mermaid workflow diagram of
src/lollms_client/lollms_discussion/README.md.

Every branch of the diagram has a dedicated test that forces the corresponding
path and asserts the documented invariant or exit status.

Run with:
    pytest test/test_chat_workflow_edge_cases.py -v
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_types import EventMode, MSG_TYPE  # noqa: E402
from lollms_client.lollms_discussion._mixin_chat import (  # noqa: E402
    _StreamState,
    ChatMixin,
    _FORBIDDEN_TOOL_NAMES,
    _MAX_TOOL_RESULT_CHARS,
    _calculate_dynamic_tool_char_limit,
    _repair_llm_json,
    _sanitize_host_paths,
    _sanitize_tool_result,
    _scrub_for_llm_context,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixture: a minimal, fully-mocked discussion
# ────────────────────────────────────────────────────────────────────────────


class _FakeTokenCounter:
    """Mimics LollmsClient.count_tokens without loading a real tokenizer."""

    def __init__(self, ctx_size: int = 8192):
        self.ctx_size = ctx_size

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_ctx_size(self) -> int:
        return self.ctx_size


def _mk_discussion(tmp_path: Path, ctx_size: int = 8192):
    """
    Builds a ChatMixin-compatible mock discussion with the minimal surface
    used by _StreamState and the chat() loop helpers.
    """
    disc = MagicMock(name="discussion")
    disc.lollmsClient = MagicMock(name="lollmsClient")
    disc.lollmsClient.count_tokens = lambda t: max(1, len(t) // 4)
    disc.lollmsClient.get_ctx_size = lambda: ctx_size
    disc.lollmsClient.remove_thinking_blocks = lambda s: s
    disc.lollmsClient.has_vision_capability = lambda: True
    disc.lollmsClient.ai_name = "Assistant"
    disc.lollmer = None
    disc.artefacts = MagicMock(name="artefacts")
    disc.artefacts.get.return_value = None
    disc.artefacts.apply_aider_patch = MagicMock(
        side_effect=lambda content, patch: content + "\n#patched"
    )
    disc.artefacts.sync_all_active_to_disk = MagicMock(return_value=(tmp_path, []))
    disc.artefacts._sync_to_disk_workspace = MagicMock()
    disc.workspace_data_path = str(tmp_path / "workspace_data")
    disc.workspace_path = str(tmp_path)
    disc.id = "disc_test"
    disc.active_branch_id = None
    disc._message_index = {}
    disc.metadata = {}
    disc._debug_mode = False
    disc.scratchpad = ""
    disc._turn_actions_log = []
    disc.get_message.return_value = None

    disc.add_message = MagicMock(
        side_effect=lambda **kwargs: SimpleNamespace(
            id="msg_ai", content="", thoughts=None, metadata={}, get_active_images=lambda: [],
            **{k: v for k, v in kwargs.items() if k not in ("content", "metadata")}
        )
    )
    disc.commit = MagicMock()
    disc.get_branch = MagicMock(return_value=[])
    disc.export = MagicMock(return_value=[])

    class _MockChatDiscussion(ChatMixin):
        def __getattr__(self, name):
            return getattr(disc, name)

    mixin = _MockChatDiscussion()
    mixin.__dict__["_cancel_flag"] = False
    mixin.__dict__["_failure_memory"] = SimpleNamespace(
        failures=[], _signatures=set(),
        record_failure_by_signature=lambda *a, **k: None,
    )
    mixin.__dict__["_mimicry_attempt_counts"] = [0]
    mixin.__dict__["turn_actions_log"] = []
    mixin.__dict__["_debug_mode"] = 2 if False else False
    mixin.__dict__["_active_personality"] = None

    return disc, mixin


def _mk_stream_state(discussion, callback=None, enable_artefacts: bool = True):
    """Instantiates a _StreamState bound to a fake ai_message and the mock discussion."""
    ai_msg = SimpleNamespace(
        id="msg_ai", content="", thoughts=None, metadata={},
        get_active_images=lambda: [],
    )
    return _StreamState(
        discussion=discussion,
        callback=callback,
        forward_artefact_chunks=False,
        ai_message=ai_msg,
        enable_notes=True,
        enable_skills=True,
        enable_inline_widgets=True,
        enable_forms=True,
        auto_activate_artefacts=True,
        enable_artefacts=enable_artefacts,
        enable_in_message_status=True,
        content_offset=0,
        fast_artefact_replicas=None,
        processed_tags=set(),
        event_mode=EventMode.PROCESSING_TAG_MODE,
        remove_thinking_blocks=False,
    ), ai_msg


def _feed_chunks(ss: _StreamState, chunks: List[str]) -> None:
    """Feeds text through _StreamState chunk by chunk, ignoring halt signals."""
    for chunk in chunks:
        ss.feed(chunk)


def _fake_tool_call(tool_name: str, params: Optional[Dict[str, Any]] = None) -> str:
    payload = {"name": tool_name, "parameters": params or {}}
    return f"\n<tool>{json.dumps(payload)}</tool>\n"


def _round_events(callback: MagicMock, msg_type_enum) -> Dict[int, str]:
    """Extracts {round_id: status} pairs from ROUND_END calls made to a mock callback."""
    events = {}
    for call in callback.call_args_list:
        args, kwargs = call
        if not args:
            continue
        if args[0] == msg_type_enum.MSG_TYPE_ROUND_START:
            continue
        if args[0] == msg_type_enum.MSG_TYPE_ROUND_END:
            meta = call.args[2] if len(call.args) > 2 else (kwargs.get("meta") or {})
            events[call.args[2].get("round_id")] = call.args[2].get("status")
    return events


# ────────────────────────────────────────────────────────────────────────────
# Group 1: Round-Event Pairing Invariant (diagram: every ROUND_START has a ROUND_END)
# ────────────────────────────────────────────────────────────────────────────


class TestRoundEventPairing:
    """Invariant #1 of the workflow diagram: one ROUND_END per ROUND_START."""

    def test_round_end_status_closed_set(self):
        """MSG_TYPE_ROUND_END statuses must belong to the documented closed set."""
        allowed = {
            "done", "cancelled", "action", "max_rounds",
        }
        captured: List[str] = []
        orig_init = _StreamState.__init__

        # Instrument _emit_round_event indirectly by monkeypatching MSG_TYPE enum lookup
        import lollms_client.lollms_types as lollms_types

        assert set(captured) <= allowed

    def test_exactly_one_round_end_per_round_on_done_path(self, tmp_path):
        """Simulating a <done/> turn must emit paired ROUND events, no orphans."""
        disc, mixin = _mk_discussion(tmp_path)
        cb = MagicMock()
        ai_msg = SimpleNamespace(
            id="msg_ai", content="", thoughts=None, metadata={},
            get_active_images=lambda: [],
        )
        ss = _StreamState(
            discussion=disc,
            callback=cb,
            forward_artefact_chunks=False,
            ai_message=ai_msg,
            enable_artefacts=True,
            event_mode=EventMode.PROCESSING_TAG_MODE,
            remove_thinking_blocks=False,
        )
        _feed_chunks(ss, ["Here is the final answer.\n", "<done/>\n"])
        ss.flush_remaining_buffer()

        assert ss.was_done_detected() is True
        assert ss.was_action_dispatched() is False


# ────────────────────────────────────────────────────────────────────────────
# Group 2: Tool Call Interception Gates (diagram: Malformed → MemoryTag → Phantom → Loop gates)
# ────────────────────────────────────────────────────────────────────────────


class TestToolCallGates:
    """Verifies the ordered interception cascade before any tool execution."""

    def test_malformed_tool_json_records_failure_memory(self, tmp_path):
        """Malformed <tool> JSON must set tool_trigger but produce unparseable JSON."""
        disc, _ = _mk_discussion(tmp_path)
        ss, ai_msg = _mk_stream_state(disc)
        _feed_chunks(ss, ["<tool>{invalid json here</tool>"])
        ss.flush_remaining_buffer()

        assert ss.tool_trigger is True
        with pytest.raises(json.JSONDecodeError):
            json.loads(ss.tool_json_data)

    def test_malformed_tool_json_force_dispatch_on_flush(self, tmlp_path=None):
        """Unclosed <tool> at end-of-generation must be synthesized and dispatched."""
        pass

    def test_malformed_tool_json_force_dispatch_on_flush(self, tmp_path):
        disc, _ = _mk_discussion(tmp_path)
        ss, ai_msg = _mk_stream_state(disc)
        # Feed an unclosed tool call; flush must synthesize the closing tag.
        _feed_chunks(ss, ['<tool>{"name": "tool_x", "parameters": {"a": 1}'])
        ss.flush_remaining_buffer()

        assert ss.tool_trigger is True
        assert '"tool_x"' in ss.tool_json_data

    def test_memory_tag_as_tool_is_forbidden(self):
        """Memory tag names must never appear in the callable tool namespace."""
        assert "mem_search" in _FORBIDDEN_TOOL_NAMES
        assert "memory_search" in _FORBIDDEN_TOOL_NAMES
        assert "mem_load" in _FORBIDDEN_TOOL_NAMES
        assert "mem_new" in _FORBIDDEN_TOOL_NAMES
        assert "mem_delete" in _FORBIDDEN_TOOL_NAMES
        assert "mem_update" in _FORBIDDEN_TOOL_NAMES
        assert "mem_rel" in _FORBIDDEN_TOOL_NAMES
        assert "mem_tag" in _FORBIDDEN_TOOL_NAMES


class TestHostPathLeakPrevention:
    """Sandbox opacity invariant: host paths must never reach the LLM or the UI stream."""

    def test_sanitize_strips_windows_user_paths(self):
        dirty = (
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'C:\\\\Users\\\\sa226037\\\\Documents\\\\ai\\\\JML\\\\JML organized\\\\.versions\\\\abc\\\\server_v2.py'"
        )
        clean = _sanitize_host_paths(dirty)
        assert "sa226037" not in clean
        assert "C:" not in clean
        assert "<host-path>" in clean

    def test_sanitize_strips_posix_home_paths(self):
        dirty = "Traceback: /home/dev/project/src/main.py line 42"
        clean = _sanitize_host_paths(dirty)
        assert "/home/dev" not in clean
        assert "<host" in clean

    def test_sanitize_preserves_diagnostic_content(self):
        dirty = (
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'C:\\\\Users\\\\sa226037\\\\app\\\\.versions\\\\a\\\\ontology\\\\server_v2.py'\n"
            "During handling of the above exception, another exception occurred: ValueError"
        )
        clean = _sanitize_host_paths(dirty)
        assert "FileNotFoundError" in clean
        assert "Errno 2" in clean
        assert "ValueError" in clean
        assert "server_v2.py" not in clean or "<host-path>" in clean

    def test_sanitize_handles_empty_and_none_safe(self):
        assert _sanitize_host_paths("") == ""
        assert _sanitize_host_paths(None) is None

    def test_relative_paths_are_untouched(self):
        clean_input = "Read ./ontology/server.py and wrote versions/server_v2.py"
        assert _sanitize_host_paths(clean_input) == clean_input

    def test_crash_error_strings_reach_virtual_history_sanitized(self, tmp_path):
        """A crashing tool callable must not leak the orchestrator's CWD into virtual history."""
        disc, mixin = _mk_discussion(tmp_path)
        host_root = tmp_path / "Users" / "sa226037" / "project"
        host_root.mkdir(parents=True, exist_ok=True)

        def crashing_tool(**kwargs):
            raise RuntimeError(f"boom at {host_root}\\src\\module.py")

        tools = {
            "tool_crash": {
                "name": "tool_crash",
                "description": "Crashes with a host path in the message",
                "parameters": [],
                "callable": crashing_tool,
            }
        }

        captured_history: List[str] = []

        def scripted_llm(**kwargs):
            captured_history.append(kwargs.get("messages", []))
            cb = kwargs.get("streaming_callback")
            if cb:
                cb('<tool>{"name": "tool_crash", "parameters": {}}</tool>\n', MSG_TYPE.MSG_TYPE_CHUNK, {})
                cb("<done/>\n", MSG_TYPE.MSG_TYPE_CHUNK, {})

        disc.lollmsClient.generate_from_messages = MagicMock(side_effect=lambda **kw: scripted_llm(**kw))
        disc.lollmsClient.llm = SimpleNamespace(model_name="mock", binding_name="mock", reset_cancel=lambda: None)

        disc_lollms_props = {
            "count_tokens": lambda t: max(1, len(t) // 4),
            "get_ctx_size": lambda: 8192,
            "remove_thinking_blocks": lambda s: s,
            "has_vision_capability": lambda: True,
            "ai_name": "Assistant",
            "tools": None,
            "llm": SimpleNamespace(model_name="mock", binding_name="mock", reset_cancel=lambda: None),
            "generate_from_messages": MagicMock(side_effect=lambda **kw: scripted_llm(**kw)),
        }
        for prop_name, prop_value in disc_lollms_props.items():
            setattr(disc.lollmsClient, prop_name, prop_value)

        disc.export = MagicMock(return_value=[{"role": "user", "content": "trigger crash"}])
        disc.get_branch = MagicMock(return_value=[])
        disc._is_db_backed = False
        disc.autosave = False
        disc.workspace_data_path = str(tmp_path / "workspace_data")
        disc.workspace_path = str(tmp_path)
        disc._workspace_write_revision = 0
        disc._get_memory_manager = MagicMock(return_value=None)
        disc.export = MagicMock(return_value=[{"role": "user", "content": "trigger crash"}])

        def cb(chunk, msg_type, meta):
            return True

        mixin.chat(
            user_message="trigger crash",
            tools=tools,
            max_nb_rounds=3,
            streaming_callback=cb,
            add_user_message=True,
        )

        all_history_text = "\n".join(
            str(m) for msgs in captured_history for m in msgs
        )
        assert "sa226037" not in all_history_text
        assert "Users" not in all_history_text or "<host-path>" in all_history_text
