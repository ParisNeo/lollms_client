#!/usr/bin/env python3
"""
test_high_grade_agent.py
=============================================================================
Comprehensive test suite for the upgraded high-grade Lollms Agent system.

Covers:
  1. SkillsManager — SKILL.md loading, CRUD operations, mode-based context.
  2. CapabilityFlags — Boolean gating of tools and features.
  3. SubAgentSpawner — Depth/count limits, recursion prevention.
  4. ModelSwitcher — Model listing, switching, restoration.
  5. BindingToolsBuilder — TTI/TTS/STT tool generation from bindings.
  6. _AgentStreamState — <tool> tag parsing, <done/> detection, code fence protection.
  7. Agent.chat() — Multi-round tool execution, loop prevention, workspace sync.
  8. Agent legacy methods — generate(), generate_with_tools() backward compatibility.
=============================================================================
"""

import sys
import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_agent.lollms_agent import (
    Agent,
    AgentRole,
    Skill,
    SkillsManager,
    CapabilityFlags,
    SubAgentSpawner,
    ModelSwitcher,
    BindingToolsBuilder,
    ToolsManager,
    _AgentStreamState,
    _parse_skill_md,
    _DEFAULT_SKILLS_DIR,
)
from lollms_client.lollms_types import MSG_TYPE


# ===========================================================================
# Mock Client — simulates LLM token streaming for Agent.chat()
# ===========================================================================

class ScriptedAgentClient:
    """
    Mock client that streams deterministic token chunks per reasoning round.
    Mimics the LollmsClient interface expected by Agent.chat().
    """

    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock-agent"
        self.binding_name = "mock"
        self.tti = None
        self.tts = None
        self.stt = None
        self.ttm = None
        self.ttv = None
        self.tools = None
        self.round_scripts = []
        self.current_round = 0

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_image_tokens(self, img) -> int:
        return 0

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def reset_cancel(self):
        pass

    def cancel(self):
        pass

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "Mock text generation."

    def generate_structured_content(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {"result": "mock"}

    def generate_from_messages(self, messages: list, **kwargs) -> str:
        callback = kwargs.get("streaming_callback")
        script = (
            self.round_scripts[self.current_round]
            if self.current_round < len(self.round_scripts)
            else ["<done/>"]
        )
        self.current_round += 1
        if callback:
            for chunk in script:
                keep_going = callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK, {})
                if keep_going is False:
                    break
        return ""


# ===========================================================================
# 1. TestSkillsManager
# ===========================================================================

class TestSkillsManager(unittest.TestCase):

    def setUp(self):
        self.tmp_skills_dir = tempfile.mkdtemp(prefix="lollms_skills_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_skills_dir, ignore_errors=True)

    def _create_skill_file(self, dir_name: str, title: str, description: str = "Test skill", category: str = "test", content: str = "Test content", tags: list = None, always_visible: bool = False):
        skill_dir = Path(self.tmp_skills_dir) / dir_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        tags = tags or ["test"]

        fm = "---\n"
        fm += f'title: "{title}"\n'
        fm += f'description: "{description}"\n'
        fm += f'category: "{category}"\n'
        fm += f'tags: [{", ".join(tags)}]\n'
        fm += f'always_visible: {str(always_visible).lower()}\n'
        fm += "---\n\n"
        fm += f"# {title}\n\n{content}\n"

        skill_file.write_text(fm, encoding="utf-8")
        return skill_file

    def test_load_skill_with_frontmatter(self):
        self._create_skill_file("my_skill", "My Test Skill", description="A test skill", content="Do X then Y")
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="loadable", default_skills_dir=self.tmp_skills_dir)
        skills = mgr.list_skills()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["title"], "My Test Skill")
        self.assertEqual(skills[0]["description"], "A test skill")
        self.assertEqual(skills[0]["category"], "test")

    def test_load_skill_plain_markdown(self):
        skill_dir = Path(self.tmp_skills_dir) / "plain_skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("# Plain Skill\n\nThis is a plain markdown skill.\n", encoding="utf-8")

        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="loadable", default_skills_dir=self.tmp_skills_dir)
        skills = mgr.list_skills()
        self.assertTrue(any(s["title"] == "Plain Skill" for s in skills))

    def test_loadable_mode_context(self):
        self._create_skill_file("skill_a", "Skill A", description="Desc A")
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="loadable", default_skills_dir=self.tmp_skills_dir)
        ctx = mgr.build_context()
        self.assertIn("Skill A", ctx)
        self.assertIn("Desc A", ctx)
        self.assertIn("tool_load_skill", ctx)

    def test_always_visible_mode_context(self):
        self._create_skill_file("skill_a", "Skill A", description="Desc A", content="Full content here", always_visible=True)
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="always_visible", default_skills_dir=self.tmp_skills_dir)
        ctx = mgr.build_context()
        self.assertIn("Full content here", ctx)
        self.assertIn("ACTIVE SKILLS", ctx)

    def test_mixed_mode_context(self):
        self._create_skill_file("vis_skill", "Visible Skill", content="Visible content", always_visible=True)
        self._create_skill_file("load_skill", "Loadable Skill", description="Loadable desc")
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="mixed", default_skills_dir=self.tmp_skills_dir)
        ctx = mgr.build_context()
        self.assertIn("Visible content", ctx)
        self.assertIn("Loadable Skill", ctx)
        self.assertIn("Loadable desc", ctx)

    def test_load_skill_by_title(self):
        self._create_skill_file("my_skill", "My Skill", content="The content")
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="loadable", default_skills_dir=self.tmp_skills_dir)
        content = mgr.load_skill("My Skill")
        self.assertIsNotNone(content)
        self.assertIn("The content", content)

    def test_load_skill_fuzzy_search(self):
        self._create_skill_file("python_debug", "Python Debugging", content="Debug tips")
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="loadable", default_skills_dir=self.tmp_skills_dir)
        content = mgr.load_skill("python debug")
        self.assertIsNotNone(content)
        self.assertIn("Debug tips", content)

    def test_search_skills(self):
        self._create_skill_file("py_skill", "Python Best Practices", description="Python tips", tags=["python", "coding"])
        self._create_skill_file("data_skill", "Data Analysis", description="Data tips", tags=["data", "pandas"])
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="loadable", default_skills_dir=self.tmp_skills_dir)
        results = mgr.search_skills("python")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Python Best Practices")

    def test_add_skill_creates_file(self):
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="loadable", default_skills_dir=self.tmp_skills_dir)
        result = mgr.add_skill("New Skill", "A new skill", "general", "New content here", ["new"])
        self.assertTrue(result["success"])
        skills = mgr.list_skills()
        self.assertTrue(any(s["title"] == "New Skill" for s in skills))

    def test_update_skill(self):
        self._create_skill_file("my_skill", "My Skill", content="Original content")
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="loadable", default_skills_dir=self.tmp_skills_dir)
        result = mgr.update_skill("My Skill", "Updated content")
        self.assertTrue(result["success"])
        content = mgr.load_skill("My Skill")
        self.assertIn("Updated content", content)

    def test_delete_skill(self):
        self._create_skill_file("my_skill", "My Skill", content="Content to delete")
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="loadable", default_skills_dir=self.tmp_skills_dir)
        self.assertEqual(len(mgr.list_skills()), 1)
        result = mgr.delete_skill("My Skill")
        self.assertTrue(result["success"])
        self.assertEqual(len(mgr.list_skills()), 0)

    def test_reload_skills(self):
        self._create_skill_file("skill_a", "Skill A")
        mgr = SkillsManager(skills_dirs=[self.tmp_skills_dir], mode="loadable", default_skills_dir=self.tmp_skills_dir)
        self.assertEqual(len(mgr.list_skills()), 1)
        self._create_skill_file("skill_b", "Skill B")
        mgr.reload()
        self.assertEqual(len(mgr.list_skills()), 2)


# ===========================================================================
# 2. TestCapabilityFlags
# ===========================================================================

class TestCapabilityFlags(unittest.TestCase):

    def test_defaults_are_safe(self):
        caps = CapabilityFlags()
        self.assertFalse(caps.enable_code_execution)
        self.assertFalse(caps.enable_external_file_access)
        self.assertFalse(caps.enable_networking)
        self.assertFalse(caps.enable_model_switching)
        self.assertTrue(caps.enable_sub_agents)
        self.assertTrue(caps.enable_skill_creation)
        self.assertTrue(caps.enable_skill_loading)
        self.assertTrue(caps.enable_image_generation)
        self.assertTrue(caps.enable_workspace_tools)

    def test_to_dict_contains_all_flags(self):
        caps = CapabilityFlags(enable_code_execution=True, enable_tts=True)
        d = caps.to_dict()
        self.assertTrue(d["enable_code_execution"])
        self.assertTrue(d["enable_tts"])
        self.assertIn("skills_mode", d)
        self.assertIn("max_sub_agent_depth", d)

    def test_skills_mode_values(self):
        caps = CapabilityFlags(skills_mode="always_visible")
        self.assertEqual(caps.skills_mode, "always_visible")
        caps = CapabilityFlags(skills_mode="mixed")
        self.assertEqual(caps.skills_mode, "mixed")


# ===========================================================================
# 3. TestSubAgentSpawner
# ===========================================================================

class TestSubAgentSpawner(unittest.TestCase):

    def setUp(self):
        self.client = ScriptedAgentClient()
        from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
        self.personality = LollmsPersonality(
            name="TestAgent",
            author="test",
            category="test",
            description="Test agent",
            system_prompt="You are a test agent.",
        )
        self.tmp_ws = tempfile.mkdtemp(prefix="lollms_subagent_ws_")
        self.agent = Agent(
            lc=self.client,
            personality=self.personality,
            workspace_path=self.tmp_ws,
            capabilities=CapabilityFlags(enable_sub_agents=True, max_sub_agent_depth=2, max_sub_agents_per_turn=3),
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_ws, ignore_errors=True)

    def test_can_spawn_within_limits(self):
        spawner = self.agent._sub_agent_spawner
        self.assertTrue(spawner.can_spawn())

    def test_per_turn_limit(self):
        spawner = self.agent._sub_agent_spawner
        spawner.reset_turn()
        for _ in range(3):
            spawner._spawned_this_turn += 1
        self.assertFalse(spawner.can_spawn())

    def test_depth_limit(self):
        spawner = self.agent._sub_agent_spawner
        spawner.set_depth(2)
        self.assertFalse(spawner.can_spawn())

    def test_reset_turn_clears_count(self):
        spawner = self.agent._sub_agent_spawner
        spawner._spawned_this_turn = 5
        spawner.reset_turn()
        self.assertEqual(spawner._spawned_this_turn, 0)


# ===========================================================================
# 4. TestModelSwitcher
# ===========================================================================

class TestModelSwitcher(unittest.TestCase):

    def setUp(self):
        self.client = ScriptedAgentClient()

    def test_list_models_via_list_models_method(self):
        mock_llm = MagicMock()
        mock_llm.list_models.return_value = ["model_a", "model_b"]
        mock_llm.model_name = "model_a"
        self.client.llm = mock_llm

        switcher = ModelSwitcher(self.client)
        models = switcher.list_models()
        self.assertIn("model_a", models)
        self.assertIn("model_b", models)

    def test_list_models_via_available_models_attr(self):
        mock_llm = MagicMock()
        del mock_llm.list_models
        mock_llm.available_models = ["m1", "m2"]
        mock_llm.model_name = "m1"
        self.client.llm = mock_llm

        switcher = ModelSwitcher(self.client)
        models = switcher.list_models()
        self.assertIn("m1", models)

    def test_get_current_model(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "current_model"
        self.client.llm = mock_llm

        switcher = ModelSwitcher(self.client)
        self.assertEqual(switcher.get_current_model(), "current_model")

    def test_switch_model_remote_binding(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "old_model"
        del mock_llm.unload_model
        del mock_llm.load_model
        self.client.llm = mock_llm

        switcher = ModelSwitcher(self.client)
        result = switcher.switch_model("new_model")
        self.assertTrue(result["success"])
        self.assertEqual(mock_llm.model_name, "new_model")
        self.assertEqual(switcher._current_model, "new_model")

    def test_switch_model_local_binding(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "old_model"
        mock_llm.load_model.return_value = True
        self.client.llm = mock_llm

        switcher = ModelSwitcher(self.client)
        result = switcher.switch_model("new_model")
        self.assertTrue(result["success"])
        mock_llm.unload_model.assert_called_once()
        mock_llm.load_model.assert_called_with("new_model")

    def test_switch_model_load_failure_restores_original(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "original_model"
        mock_llm.load_model.side_effect = [False, True]
        self.client.llm = mock_llm

        switcher = ModelSwitcher(self.client)
        result = switcher.switch_model("bad_model")
        self.assertFalse(result["success"])
        mock_llm.load_model.assert_called_with("original_model")

    def test_restore_original_model(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "original_model"
        del mock_llm.unload_model
        del mock_llm.load_model
        self.client.llm = mock_llm

        switcher = ModelSwitcher(self.client)
        switcher.switch_model("temp_model")
        result = switcher.restore_original_model()
        self.assertTrue(result["success"])
        self.assertEqual(mock_llm.model_name, "original_model")


# ===========================================================================
# 5. TestBindingToolsBuilder
# ===========================================================================

class TestBindingToolsBuilder(unittest.TestCase):

    def test_no_bindings_returns_empty(self):
        client = MagicMock()
        client.tti = None
        client.tts = None
        client.stt = None
        client.ttm = None
        client.ttv = None
        caps = CapabilityFlags()
        tools = BindingToolsBuilder.build_tools(client, caps, None)
        self.assertEqual(len(tools), 0)

    def test_tti_generate_tool_built(self):
        client = MagicMock()
        client.tti = MagicMock()
        client.tts = None
        client.stt = None
        client.ttm = None
        client.ttv = None
        caps = CapabilityFlags(enable_image_generation=True, enable_image_editing=False)
        tools = BindingToolsBuilder.build_tools(client, caps, None)
        self.assertIn("tool_generate_image", tools)
        self.assertNotIn("tool_edit_image", tools)

    def test_tti_edit_tool_built(self):
        client = MagicMock()
        client.tti = MagicMock()
        client.tts = None
        client.stt = None
        client.ttm = None
        client.ttv = None
        caps = CapabilityFlags(enable_image_generation=False, enable_image_editing=True)
        tools = BindingToolsBuilder.build_tools(client, caps, None)
        self.assertIn("tool_edit_image", tools)
        self.assertNotIn("tool_generate_image", tools)

    def test_tts_tool_built(self):
        client = MagicMock()
        client.tti = None
        client.tts = MagicMock()
        client.stt = None
        caps = CapabilityFlags(enable_tts=True)
        tools = BindingToolsBuilder.build_tools(client, caps, None)
        self.assertIn("tool_text_to_speech", tools)

    def test_tts_tool_not_built_when_disabled(self):
        client = MagicMock()
        client.tts = MagicMock()
        caps = CapabilityFlags(enable_tts=False)
        tools = BindingToolsBuilder.build_tools(client, caps, None)
        self.assertNotIn("tool_text_to_speech", tools)

    def test_stt_tool_built(self):
        client = MagicMock()
        client.stt = MagicMock()
        client.tti = None
        client.tts = None
        caps = CapabilityFlags(enable_stt=True)
        tools = BindingToolsBuilder.build_tools(client, caps, None)
        self.assertIn("tool_speech_to_text", tools)

    def test_ttm_tool_built(self):
        client = MagicMock()
        client.ttm = MagicMock()
        client.tti = None
        client.tts = None
        caps = CapabilityFlags(enable_ttm=True)
        tools = BindingToolsBuilder.build_tools(client, caps, None)
        self.assertIn("tool_generate_music", tools)

    def test_ttv_tool_built(self):
        client = MagicMock()
        client.ttv = MagicMock()
        client.tti = None
        client.tts = None
        caps = CapabilityFlags(enable_ttv=True)
        tools = BindingToolsBuilder.build_tools(client, caps, None)
        self.assertIn("tool_generate_video", tools)

    def test_tti_generate_tool_callable(self):
        client = MagicMock()
        client.tti = MagicMock()
        client.tti.generate_image.return_value = b"fake_png_data"
        caps = CapabilityFlags(enable_image_generation=True)
        tmp_ws = tempfile.mkdtemp(prefix="lollms_tti_test_")
        try:
            tools = BindingToolsBuilder.build_tools(client, caps, Path(tmp_ws))
            tool_spec = tools["tool_generate_image"]
            result = tool_spec["callable"](prompt="a cat", width=64, height=64)
            self.assertTrue(result["success"])
            self.assertIn("image_filename", result)
            self.assertIn("image_b64", result)
            client.tti.generate_image.assert_called_once_with(prompt="a cat", width=64, height=64)
        finally:
            shutil.rmtree(tmp_ws, ignore_errors=True)


# ===========================================================================
# 6. TestAgentStreamState
# ===========================================================================

class TestAgentStreamState(unittest.TestCase):

    def setUp(self):
        self.chunks = []
        self.callback = lambda chunk, msg_type=None, meta=None: self.chunks.append(chunk)

    def test_plain_text_passes_through(self):
        ss = _AgentStreamState(callback=self.callback)
        ss.feed("Hello world")
        ss.flush_remaining_buffer()
        self.assertEqual(ss.get_clean_text(), "Hello world")
        self.assertFalse(ss.tool_trigger)
        self.assertFalse(ss.was_done_detected())

    def test_done_tag_detected(self):
        ss = _AgentStreamState(callback=self.callback)
        ss.feed("Task complete.\n<done/>")
        self.assertTrue(ss.was_done_detected())
        self.assertNotIn("<done/>", ss.get_clean_text())

    def test_processing_block_mimicry_blocked(self):
        ss = _AgentStreamState(callback=self.callback)
        result = ss.feed("<processing type=\"tool\">")
        self.assertFalse(result)
        self.assertNotIn("<processing", ss.get_clean_text())

    def test_tool_tag_accumulation_and_dispatch(self):
        ss = _AgentStreamState(callback=self.callback)
        ss.feed('<tool>{"name": "test_tool", "parameters": {"x": 1}}</tool>')
        self.assertTrue(ss.tool_trigger)
        json_str = ss.get_tool_call_json()
        self.assertIsNotNone(json_str)
        data = json.loads(json_str)
        self.assertEqual(data["name"], "test_tool")
        self.assertEqual(data["parameters"]["x"], 1)

    def test_tool_tag_split_across_chunks(self):
        ss = _AgentStreamState(callback=self.callback)
        ss.feed('<tool>{"name": "test')
        ss.feed('_tool", "parameters": {}}</tool>')
        self.assertTrue(ss.tool_trigger)
        data = json.loads(ss.get_tool_call_json())
        self.assertEqual(data["name"], "test_tool")

    def test_flat_parameter_normalization(self):
        ss = _AgentStreamState(callback=self.callback)
        ss.feed('<tool>{"name": "my_tool", "file": "test.txt", "content": "hello"}</tool>')
        self.assertTrue(ss.tool_trigger)
        data = json.loads(ss.get_tool_call_json())
        self.assertEqual(data["name"], "my_tool")
        self.assertIn("parameters", data)
        self.assertEqual(data["parameters"]["file"], "test.txt")
        self.assertEqual(data["parameters"]["content"], "hello")

    def test_code_fence_protection(self):
        ss = _AgentStreamState(callback=self.callback)
        ss.feed("Here is code:\n```python\n<tool>{\"name\": \"fake\"}</tool>\n```\nDone.")
        ss.flush_remaining_buffer()
        self.assertFalse(ss.tool_trigger, "Tool tag inside code fence should NOT be intercepted.")
        self.assertIn("<tool>", ss.get_clean_text())

    def test_unclosed_code_fence_recovery(self):
        ss = _AgentStreamState(callback=self.callback)
        ss.feed("Text\n```python\nsome code")
        ss.flush_remaining_buffer()
        self.assertIn("some code", ss.get_clean_text())

    def test_inline_code_protection(self):
        ss = _AgentStreamState(callback=self.callback)
        ss.feed("Use `<tool>` to call tools.")
        ss.flush_remaining_buffer()
        self.assertFalse(ss.tool_trigger)
        self.assertIn("`<tool>`", ss.get_clean_text())

    def test_unclosed_tool_tag_flush_recovery(self):
        ss = _AgentStreamState(callback=self.callback)
        ss.feed('<tool>{"name": "incomplete"')
        ss.flush_remaining_buffer()
        self.assertTrue(ss.tool_trigger)
        data = json.loads(ss.get_tool_call_json())
        self.assertEqual(data["name"], "incomplete")

    def test_action_dispatched_flag(self):
        ss = _AgentStreamState(callback=self.callback)
        ss.feed('<tool>{"name": "test", "parameters": {}}</tool>')
        self.assertTrue(ss.was_action_dispatched())


# ===========================================================================
# 7. TestAgentChat — Full Agentic Loop
# ===========================================================================

class TestAgentChat(unittest.TestCase):

    def setUp(self):
        self.client = ScriptedAgentClient()
        from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
        self.personality = LollmsPersonality(
            name="TestAgent",
            author="test",
            category="test",
            description="A test agent.",
            system_prompt="You are a test agent. Use tools when needed.",
        )
        self.tmp_ws = tempfile.mkdtemp(prefix="lollms_agent_chat_")
        self.agent = Agent(
            lc=self.client,
            personality=self.personality,
            name="TestBot",
            role=AgentRole.IMPLEMENTER,
            workspace_path=self.tmp_ws,
            capabilities=CapabilityFlags(
                enable_code_execution=True,
                enable_workspace_tools=True,
                enable_sub_agents=False,
                enable_skill_creation=True,
                enable_skill_loading=True,
            ),
            max_tokens_per_turn=2048,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_ws, ignore_errors=True)

    def test_conversational_response_no_tools(self):
        self.client.round_scripts = [
            ["Hello! I am doing well, thank you for asking."]
        ]
        result = self.agent.chat(
            prompt="Hello, how are you?",
            max_reasoning_steps=5,
            use_internal_history=False,
        )
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(len(result["tool_calls"]), 0)
        self.assertIn("Hello", result["response"])

    def test_done_tag_termination(self):
        self.client.round_scripts = [
            ["I will write a file.\n", '<tool>{"name": "tool_write_file", "parameters": {"file_name": "test.txt", "content": "hello"}}</tool>'],
            ["File written successfully.\n<done/>"],
        ]
        result = self.agent.chat(
            prompt="Write a file called test.txt with content 'hello'.",
            max_reasoning_steps=10,
            use_internal_history=False,
        )
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "tool_write_file")
        self.assertTrue((Path(self.tmp_ws) / "test.txt").exists())
        self.assertEqual((Path(self.tmp_ws) / "test.txt").read_text(), "hello")

    def test_tool_execution_and_workspace_change_detection(self):
        self.client.round_scripts = [
            ['<tool>{"name": "tool_write_file", "parameters": {"file_name": "output.py", "content": "print(42)"}}</tool>'],
            ["Done.\n<done/>"],
        ]
        result = self.agent.chat(
            prompt="Create output.py",
            max_reasoning_steps=5,
            use_internal_history=False,
        )
        self.assertEqual(len(result["workspace_changes"]), 1)
        self.assertEqual(result["workspace_changes"][0]["action"], "created")
        self.assertEqual(result["workspace_changes"][0]["path"], "output.py")

    def test_phantom_tool_interception(self):
        self.client.round_scripts = [
            ['<tool>{"name": "tool_nonexistent", "parameters": {}}</tool>'],
            ["I cannot do that.\n<done/>"],
        ]
        result = self.agent.chat(
            prompt="Use a nonexistent tool.",
            max_reasoning_steps=5,
            use_internal_history=False,
        )
        self.assertEqual(len(result["tool_calls"]), 0)
        self.assertIn("response", result)

    def test_success_loop_prevention(self):
        self.client.round_scripts = [
            ['<tool>{"name": "tool_list_files", "parameters": {}}</tool>'],
            ['<tool>{"name": "tool_list_files", "parameters": {}}</tool>'],
            ["Finished.\n<done/>"],
        ]
        result = self.agent.chat(
            prompt="List files twice.",
            max_reasoning_steps=10,
            use_internal_history=False,
        )
        successful_calls = [tc for tc in result["tool_calls"] if tc["name"] == "tool_list_files"]
        self.assertLessEqual(len(successful_calls), 1,
                             "Success-loop prevention should block the second identical call.")

    def test_malformed_tool_call_correction(self):
        self.client.round_scripts = [
            ['<tool>{}</tool>'],
            ['<tool>{"name": "tool_list_files", "parameters": {}}</tool>'],
            ["Listed.\n<done/>"],
        ]
        result = self.agent.chat(
            prompt="List files.",
            max_reasoning_steps=10,
            use_internal_history=False,
        )
        tool_names = [tc["name"] for tc in result["tool_calls"]]
        self.assertIn("tool_list_files", tool_names)

    def test_skills_created_tracking(self):
        self.client.round_scripts = [
            ['<tool>{"name": "tool_create_skill", "parameters": {"title": "Test Skill", "description": "A test", "category": "test", "content": "Content here"}}</tool>'],
            ["Skill created.\n<done/>"],
        ]
        result = self.agent.chat(
            prompt="Create a skill called Test Skill.",
            max_reasoning_steps=5,
            use_internal_history=False,
        )
        self.assertEqual(len(result["skills_created"]), 1)
        self.assertEqual(result["skills_created"][0], "Test Skill")

    def test_cancellation(self):
        self.client.round_scripts = [
            ["This is a long response..."],
        ]

        def cancel_callback(chunk, msg_type=None, meta=None):
            self.agent.cancel_generation()
            return False

        result = self.agent.chat(
            prompt="Long task",
            max_reasoning_steps=5,
            streaming_callback=cancel_callback,
            use_internal_history=False,
        )
        self.assertTrue(result["was_cancelled"])

    def test_internal_history_preserved(self):
        self.client.round_scripts = [
            ["Hello!\n<done/>"],
        ]
        self.agent.chat(prompt="Hi", max_reasoning_steps=3, use_internal_history=True)
        self.assertEqual(len(self.agent._conversation), 2)
        self.assertEqual(self.agent._conversation[0]["role"], "user")
        self.assertEqual(self.agent._conversation[1]["role"], "assistant")

    def test_internal_history_cleared(self):
        self.agent._conversation.append({"role": "user", "content": "old"})
        self.agent._conversation.append({"role": "assistant", "content": "old reply"})
        self.agent.clear_conversation()
        self.assertEqual(len(self.agent._conversation), 0)


# ===========================================================================
# 8. TestAgentLegacyMethods — Backward Compatibility
# ===========================================================================

class TestAgentLegacyMethods(unittest.TestCase):

    def setUp(self):
        self.client = ScriptedAgentClient()
        self.client.generate_text = MagicMock(return_value="Direct generation result.")
        from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
        self.personality = LollmsPersonality(
            name="LegacyAgent",
            author="test",
            category="test",
            description="Legacy test agent.",
            system_prompt="You are a legacy test agent.",
        )
        self.tmp_ws = tempfile.mkdtemp(prefix="lollms_legacy_")
        self.agent = Agent(
            lc=self.client,
            personality=self.personality,
            workspace_path=self.tmp_ws,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_ws, ignore_errors=True)

    def test_generate_text(self):
        result = self.agent.generate(prompt="Hello", system_prompt="Be helpful.")
        self.assertEqual(result, "Direct generation result.")
        self.client.generate_text.assert_called_once()

    def test_generate_structured(self):
        self.client.generate_structured_content = MagicMock(return_value={"key": "value"})
        result = self.agent.generate_structured(prompt="Test", schema={"key": "str"})
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "value")

    def test_generate_with_tools_delegates_to_chat(self):
        self.client.round_scripts = [
            ["I will use a tool.\n", '<tool>{"name": "tool_list_files", "parameters": {}}</tool>'],
            ["Done.\n<done/>"],
        ]
        result = self.agent.generate_with_tools(
            prompt="List files",
            tools=[],
            max_tool_rounds=5,
            auto_execute=True,
        )
        self.assertIn("response", result)
        self.assertIn("tool_calls", result)
        self.assertIn("rounds", result)

    def test_generate_with_tools_manual_mode(self):
        self.client.generate_from_messages = MagicMock(return_value='<tool>{"name": "test_tool", "parameters": {"x": 1}}</tool>')
        result = self.agent.generate_with_tools(
            prompt="Use a tool",
            tools=[],
            auto_execute=False,
        )
        self.assertIn("response", result)
        self.assertIn("tool_calls", result)
        self.assertEqual(len(result["tool_calls"]), 1)

    def test_generate_with_tools_sync(self):
        self.client.round_scripts = [
            ["Direct answer without tools.\n<done/>"],
        ]
        result = self.agent.generate_with_tools_sync(
            prompt="Answer me",
            tools=[],
        )
        self.assertIsInstance(result, str)


# ===========================================================================
# 9. TestAgentToolDiscovery — Verify capability gating
# ===========================================================================

class TestAgentToolDiscovery(unittest.TestCase):

    def setUp(self):
        self.client = ScriptedAgentClient()
        from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
        self.personality = LollmsPersonality(
            name="DiscoveryAgent",
            author="test",
            category="test",
            description="Tool discovery test.",
            system_prompt="You are a test agent.",
        )
        self.tmp_ws = tempfile.mkdtemp(prefix="lollms_discovery_")

    def tearDown(self):
        shutil.rmtree(self.tmp_ws, ignore_errors=True)

    def test_workspace_tools_included(self):
        agent = Agent(
            lc=self.client,
            personality=self.personality,
            workspace_path=self.tmp_ws,
            capabilities=CapabilityFlags(enable_workspace_tools=True),
        )
        tools = agent._discover_tools(None, None, False)
        self.assertIn("tool_write_file", tools)
        self.assertIn("tool_read_file", tools)
        self.assertIn("tool_list_files", tools)

    def test_skill_tools_included_when_enabled(self):
        agent = Agent(
            lc=self.client,
            personality=self.personality,
            workspace_path=self.tmp_ws,
            capabilities=CapabilityFlags(enable_skill_loading=True, enable_skill_creation=True),
        )
        tools = agent._discover_tools(None, None, False)
        self.assertIn("tool_load_skill", tools)
        self.assertIn("tool_list_skills", tools)
        self.assertIn("tool_create_skill", tools)
        self.assertIn("tool_update_skill", tools)

    def test_skill_tools_excluded_when_disabled(self):
        agent = Agent(
            lc=self.client,
            personality=self.personality,
            workspace_path=self.tmp_ws,
            capabilities=CapabilityFlags(enable_skill_loading=False, enable_skill_creation=False),
        )
        tools = agent._discover_tools(None, None, False)
        self.assertNotIn("tool_load_skill", tools)
        self.assertNotIn("tool_create_skill", tools)

    def test_sub_agent_tool_included_when_enabled(self):
        agent = Agent(
            lc=self.client,
            personality=self.personality,
            workspace_path=self.tmp_ws,
            capabilities=CapabilityFlags(enable_sub_agents=True, max_sub_agent_depth=2),
        )
        tools = agent._discover_tools(None, None, False)
        self.assertIn("tool_spawn_sub_agent", tools)

    def test_sub_agent_tool_excluded_when_disabled(self):
        agent = Agent(
            lc=self.client,
            personality=self.personality,
            workspace_path=self.tmp_ws,
            capabilities=CapabilityFlags(enable_sub_agents=False),
        )
        tools = agent._discover_tools(None, None, False)
        self.assertNotIn("tool_spawn_sub_agent", tools)

    def test_model_switching_tools_included_when_enabled(self):
        agent = Agent(
            lc=self.client,
            personality=self.personality,
            workspace_path=self.tmp_ws,
            capabilities=CapabilityFlags(enable_model_switching=True),
        )
        tools = agent._discover_tools(None, None, False)
        self.assertIn("tool_switch_model", tools)
        self.assertIn("tool_list_models", tools)

    def test_binding_tools_included(self):
        self.client.tti = MagicMock()
        agent = Agent(
            lc=self.client,
            personality=self.personality,
            workspace_path=self.tmp_ws,
            capabilities=CapabilityFlags(enable_image_generation=True),
        )
        tools = agent._discover_tools(None, None, False)
        self.assertIn("tool_generate_image", tools)


if __name__ == "__main__":
    unittest.main()
