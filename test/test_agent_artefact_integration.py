import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from lollms_client.lollms_personality import (
    LollmsPersonality,
    Agent,
    AgentRole,
    CapabilityFlags,
)
from lollms_client.tools_bindings.lcp.default_tools.grep_files.grep_files import tool_grep_files


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.llm.model_name = "test-model"
    client.llm.binding_name = "test-binding"
    client.count_tokens = lambda text: len(text) // 4
    client.generate_text.return_value = "Test response"
    client.generate_from_messages.return_value = "Test response"
    return client


@pytest.fixture
def mock_personality():
    p = MagicMock()
    p.name = "TestBot"
    p.system_prompt = "You are a test bot."
    p.has_data = False
    return p


@pytest.fixture
def temp_workspace(tmp_path):
    ws = tmp_path / "test_workspace"
    ws.mkdir()
    (ws / "main.py").write_text("def hello():\n    print('world')\n", encoding="utf-8")
    (ws / "config.json").write_text('{"key": "value"}', encoding="utf-8")
    (ws / "data.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    return ws


class TestMinimalAgentToolsetRemoval:
    def test_toolset_file_deleted(self):
        toolset_path = Path("src/lollms_client/tools_bindings/lcp/default_tools/minimal_agent_toolset/minimal_agent_toolset.py")
        assert not toolset_path.exists(), "minimal_agent_toolset.py should be deleted in favor of standard grep_files"

    def test_grep_files_finds_pattern(self, temp_workspace):
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(str(temp_workspace))
            result = tool_grep_files(pattern="hello", file_extension=".py")
            assert result["success"] is True
            assert result["matches_count"] >= 1
            assert "main.py" in result["output"]
        finally:
            os.chdir(old_cwd)

    def test_grep_files_respects_extension_filter(self, temp_workspace):
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(str(temp_workspace))
            result = tool_grep_files(pattern="value", file_extension=".json")
            assert result["success"] is True
            assert result["matches_count"] >= 1
            assert "config.json" in result["output"]
            assert ".csv" not in result["output"]
        finally:
            os.chdir(old_cwd)

    def test_grep_files_no_matches(self, temp_workspace):
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(str(temp_workspace))
            result = tool_grep_files(pattern="zzz_nonexistent_zzz")
            assert result["success"] is True
            assert result["matches_count"] == 0
        finally:
            os.chdir(old_cwd)


class TestAgentArtefactSystemOptIn:
    def test_artefact_system_disabled_by_default(self, mock_client, mock_personality, temp_workspace):
        agent = LollmsPersonality(
            name=mock_personality.name,
            system_prompt=mock_personality.system_prompt,
            lollms_client=mock_client,
            workspace_path=str(temp_workspace),
            enable_artefact_system=False
        )
        assert getattr(agent, "_artefact_manager", None) is None

    def test_artefact_system_enabled(self, mock_client, mock_personality, temp_workspace):
        agent = LollmsPersonality(
            name=mock_personality.name,
            system_prompt=mock_personality.system_prompt,
            lollms_client=mock_client,
            workspace_path=str(temp_workspace),
            enable_artefact_system=True
        )
        assert getattr(agent, "_artefact_manager", None) is not None

    def test_artefact_system_scans_workspace(self, mock_client, mock_personality, temp_workspace):
        agent = LollmsPersonality(
            name=mock_personality.name,
            system_prompt=mock_personality.system_prompt,
            lollms_client=mock_client,
            workspace_path=str(temp_workspace),
            enable_artefact_system=True
        )
        am = agent._artefact_manager
        all_arts = am.list()
        titles = [a["title"] for a in all_arts]
        assert "main.py" in titles
        assert "config.json" in titles
        assert "data.csv" in titles

    def test_artefact_context_zone_renders_tree(self, mock_client, mock_personality, temp_workspace):
        agent = LollmsPersonality(
            name=mock_personality.name,
            system_prompt=mock_personality.system_prompt,
            lollms_client=mock_client,
            workspace_path=str(temp_workspace),
            enable_artefact_system=True
        )
        zone = agent._artefact_manager.build_artefacts_context_zone()
        assert "Workspace Directory Tree Index" in zone


class TestDisableArtefactVersioning:
    def test_versioning_disabled_overwrites_in_place(self, mock_client, mock_personality, temp_workspace):
        agent = LollmsPersonality(
            name=mock_personality.name,
            system_prompt=mock_personality.system_prompt,
            lollms_client=mock_client,
            workspace_path=str(temp_workspace),
            enable_artefact_system=True,
            disable_artefact_versioning=True
        )
        am = agent._artefact_manager

        art1 = am.add(title="test.py", artefact_type="code", content="v1 content", active=True)
        test_arts = [a for a in am.list() if a["title"] == "test.py"]
        assert len(test_arts) == 1
        assert test_arts[0]["content"] == ""
        assert test_arts[0].get("content_source") == "disk"
        assert test_arts[0]["version"] == 1
        assert art1["content"] == "v1 content"
        assert am._read_content_from_disk(test_arts[0]) == "v1 content"

        am.add(title="test.py", artefact_type="code", content="v2 content", active=True)
        test_arts = [a for a in am.list() if a["title"] == "test.py"]
        assert len(test_arts) == 1
        assert test_arts[0]["content"] == ""
        assert test_arts[0]["version"] == 1
        assert am._read_content_from_disk(test_arts[0]) == "v2 content"

    def test_versioning_enabled_creates_new_versions(self, mock_client, mock_personality, temp_workspace):
        agent = LollmsPersonality(
            name=mock_personality.name,
            system_prompt=mock_personality.system_prompt,
            lollms_client=mock_client,
            workspace_path=str(temp_workspace),
            enable_artefact_system=True,
            disable_artefact_versioning=False
        )
        am = agent._artefact_manager

        am.add(title="test.py", artefact_type="code", content="v1 content", active=True)
        am.update(title="test.py", new_content="v2 content")
        test_arts = [a for a in am.list() if a["title"] == "test.py"]
        assert len(test_arts) == 2
        versions = [a["version"] for a in test_arts]
        assert 1 in versions
        assert 2 in versions


class TestAgentFileSearchTool:
    def test_tool_registered_in_discovery(self, mock_client, mock_personality, temp_workspace):
        agent = LollmsPersonality(
            name=mock_personality.name,
            system_prompt=mock_personality.system_prompt,
            lollms_client=mock_client,
            workspace_path=str(temp_workspace)
        )
        active_tools = agent._discover_tools(None, None)
        assert "tool_grep_files" in active_tools or "tool_find_files" in active_tools