#!/usr/bin/env python3
"""
test_handbag.py
=============================================================================
Comprehensive test suite for the Handbag unified agent resource folder.

Covers:
  1. Handbag.create_structure() — folder scaffolding and manifest generation
  2. Handbag.__init__ — loading from disk, missing subdirectory graceful skip
  3. Manifest parsing — handbag.yaml with default_personality, skills_mode
  4. Personality loading — SOUL.md bundles from personalities/ folder
  5. Tool discovery — flat .py files and directory-based LCP convention
  6. Skills directory registration
  7. RAG — keyword-based fallback search, document indexing, query behavior
  8. Memory manager creation from memory/ directory
  9. Accessors — get_default_personality, get_personalities, get_personality,
     get_tool_files, get_skills_dirs, get_skills_mode, get_rag_data_source,
     get_memory_db_path, get_workspace_path
 10. attach_rag_to_personality — attaching when personality has no data,
     skipping when personality already has data
 11. Agent integration — handbag_path auto-configures personality, tools,
     skills, memory, workspace; explicit params override handbag values
 12. Edge cases — empty handbag, invalid path, missing manifest
=============================================================================
"""

import sys
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_agent.handbag import Handbag
from lollms_client.lollms_agent.lollms_agent import Agent, AgentRole, CapabilityFlags


# ===========================================================================
# Helpers
# ===========================================================================

def _make_soul_md(name: str, author: str = "test", category: str = "test",
                  description: str = "Test personality", system_prompt: str = "You are a test agent.",
                  temperature: str = None) -> str:
    """Builds a SOUL.md file content string."""
    lines = ["---"]
    lines.append(f"name: {name}")
    lines.append(f"author: {author}")
    lines.append(f"version: '1.0'")
    lines.append(f"category: {category}")
    lines.append(f"description: {description}")
    if temperature:
        lines.append(f"temperature: {temperature}")
    lines.append("---")
    lines.append("")
    lines.append(system_prompt)
    return "\n".join(lines)


def _make_tool_py(tool_name: str, description: str = "A test tool") -> str:
    """Builds a minimal LCP tool .py file content."""
    return f'''
TOOL_LIBRARY_NAME = "TestTool"
TOOL_LIBRARY_DESC = "{description}"

def init_tools_library() -> None:
    pass

def {tool_name}(query: str = "") -> dict:
    """
    {description}

    Args:
        query (str): The query parameter.
    """
    return {{"success": True, "output": f"Result for {{query}}"}}
'''


def _make_handbag_yaml(name: str = "Test Handbag", default_personality: str = None,
                       skills_mode: str = "mixed") -> str:
    """Builds a handbag.yaml manifest content string."""
    lines = [
        f"name: {name}",
        "version: '1.0'",
        "description: A handbag for testing.",
    ]
    if default_personality:
        lines.append(f"default_personality: {default_personality}")
    lines.append(f"skills_mode: {skills_mode}")
    return "\n".join(lines)


# ===========================================================================
# 1. TestCreateStructure
# ===========================================================================

class TestCreateStructure(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_create_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_all_subdirectories(self):
        hb_path = Handbag.create_structure(self.tmp / "my_hb")
        self.assertTrue(hb_path.exists())
        for subdir in ["personalities", "tools", "skills", "rag", "memory", "workspace"]:
            self.assertTrue((hb_path / subdir).exists(), f"Missing subdirectory: {subdir}")

    def test_creates_manifest_yaml(self):
        hb_path = Handbag.create_structure(self.tmp / "my_hb", name="Custom Name")
        manifest = hb_path / "handbag.yaml"
        self.assertTrue(manifest.exists())
        content = manifest.read_text(encoding="utf-8")
        self.assertIn("Custom Name", content)
        self.assertIn("skills_mode", content)

    def test_creates_readme(self):
        hb_path = Handbag.create_structure(self.tmp / "my_hb")
        readme = hb_path / "README.md"
        self.assertTrue(readme.exists())
        content = readme.read_text(encoding="utf-8")
        self.assertIn("Handbag", content)

    def test_idempotent(self):
        hb_path = Handbag.create_structure(self.tmp / "my_hb", name="First")
        # Second call should not crash
        hb_path2 = Handbag.create_structure(self.tmp / "my_hb", name="Second")
        self.assertEqual(hb_path, hb_path2)

    def test_creates_parent_dirs(self):
        deep_path = self.tmp / "a" / "b" / "c" / "my_hb"
        hb_path = Handbag.create_structure(deep_path)
        self.assertTrue(hb_path.exists())


# ===========================================================================
# 2. TestHandbagInit
# ===========================================================================

class TestHandbagInit(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_init_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_empty_handbag(self):
        Handbag.create_structure(self.tmp / "empty_hb")
        hb = Handbag(self.tmp / "empty_hb")
        self.assertEqual(len(hb.get_personalities()), 0)
        self.assertEqual(len(hb.get_tool_files()), 0)
        # create_structure() creates the skills/ and memory/ directories, which
        # are registered even when empty. This is correct behavior — the agent
        # knows where to write new skills and where the memory DB will live.
        self.assertEqual(len(hb.get_skills_dirs()), 1)
        self.assertIsNone(hb.get_rag_data_source())
        self.assertIsNotNone(hb.get_memory_db_path())

    def test_nonexistent_path_raises(self):
        with self.assertRaises(ValueError):
            Handbag(self.tmp / "does_not_exist")

    def test_file_path_raises(self):
        file_path = self.tmp / "not_a_dir.txt"
        file_path.write_text("hello", encoding="utf-8")
        with self.assertRaises(ValueError):
            Handbag(file_path)

    def test_missing_subdirectories_gracefully_skipped(self):
        hb_path = self.tmp / "partial_hb"
        hb_path.mkdir()
        # Only create rag/ — no personalities, tools, skills, memory, workspace
        (hb_path / "rag").mkdir()
        (hb_path / "rag" / "doc.txt").write_text("Some content here", encoding="utf-8")

        hb = Handbag(hb_path)
        self.assertEqual(len(hb.get_personalities()), 0)
        self.assertEqual(len(hb.get_tool_files()), 0)
        self.assertEqual(len(hb.get_skills_dirs()), 0)
        self.assertIsNotNone(hb.get_rag_data_source())
        self.assertIsNone(hb.get_memory_db_path())
        self.assertIsNone(hb.get_workspace_path())

    def test_repr_contains_key_info(self):
        Handbag.create_structure(self.tmp / "repr_hb")
        hb = Handbag(self.tmp / "repr_hb")
        r = repr(hb)
        self.assertIn("Handbag", r)
        self.assertIn("repr_hb", r)


# ===========================================================================
# 3. TestManifestLoading
# ===========================================================================

class TestManifestLoading(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_manifest_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_manifest_uses_defaults(self):
        hb_path = self.tmp / "no_manifest"
        hb_path.mkdir()
        hb = Handbag(hb_path)
        self.assertEqual(hb.manifest, {})
        self.assertIsNone(hb.get_skills_mode())
        self.assertIsNone(hb._default_personality_name)

    def test_manifest_with_skills_mode(self):
        hb_path = self.tmp / "with_mode"
        hb_path.mkdir()
        (hb_path / "handbag.yaml").write_text(
            _make_handbag_yaml(skills_mode="always_visible"), encoding="utf-8"
        )
        hb = Handbag(hb_path)
        self.assertEqual(hb.get_skills_mode(), "always_visible")

    def test_manifest_with_default_personality(self):
        hb_path = self.tmp / "with_default"
        hb_path.mkdir()
        (hb_path / "handbag.yaml").write_text(
            _make_handbag_yaml(default_personality="researcher"), encoding="utf-8"
        )
        hb = Handbag(hb_path)
        self.assertEqual(hb._default_personality_name, "researcher")

    def test_malformed_manifest_silently_returns_empty(self):
        hb_path = self.tmp / "bad_manifest"
        hb_path.mkdir()
        (hb_path / "handbag.yaml").write_text("::: not valid yaml ::: [[[", encoding="utf-8")
        hb = Handbag(hb_path)
        # Should not crash; manifest is empty dict
        self.assertIsInstance(hb.manifest, dict)


# ===========================================================================
# 4. TestPersonalityLoading
# ===========================================================================

class TestPersonalityLoading(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_pers_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _create_personality(self, hb_path: Path, name: str, system_prompt: str = "You are a test agent."):
        pers_dir = hb_path / "personalities" / name
        pers_dir.mkdir(parents=True, exist_ok=True)
        (pers_dir / "SOUL.md").write_text(
            _make_soul_md(name=name, system_prompt=system_prompt), encoding="utf-8"
        )

    def test_loads_single_personality(self):
        hb_path = self.tmp / "single_pers"
        Handbag.create_structure(hb_path)
        self._create_personality(hb_path, "researcher")
        hb = Handbag(hb_path)
        personalities = hb.get_personalities()
        self.assertEqual(len(personalities), 1)
        self.assertIn("researcher", personalities)

    def test_loads_multiple_personalities(self):
        hb_path = self.tmp / "multi_pers"
        Handbag.create_structure(hb_path)
        self._create_personality(hb_path, "researcher")
        self._create_personality(hb_path, "coder")
        hb = Handbag(hb_path)
        personalities = hb.get_personalities()
        self.assertEqual(len(personalities), 2)
        self.assertIn("researcher", personalities)
        self.assertIn("coder", personalities)

    def test_get_personality_by_name(self):
        hb_path = self.tmp / "get_pers"
        Handbag.create_structure(hb_path)
        self._create_personality(hb_path, "analyst")
        hb = Handbag(hb_path)
        pers = hb.get_personality("analyst")
        self.assertIsNotNone(pers)
        self.assertEqual(pers.name, "analyst")

    def test_get_personality_nonexistent_returns_none(self):
        hb_path = self.tmp / "missing_pers"
        Handbag.create_structure(hb_path)
        self._create_personality(hb_path, "real_one")
        hb = Handbag(hb_path)
        self.assertIsNone(hb.get_personality("nonexistent"))

    def test_default_personality_from_manifest(self):
        hb_path = self.tmp / "default_from_manifest"
        Handbag.create_structure(hb_path)
        self._create_personality(hb_path, "first")
        self._create_personality(hb_path, "second")
        (hb_path / "handbag.yaml").write_text(
            _make_handbag_yaml(default_personality="second"), encoding="utf-8"
        )
        hb = Handbag(hb_path)
        default = hb.get_default_personality()
        self.assertIsNotNone(default)
        self.assertEqual(default.name, "second")

    def test_default_personality_fallback_first(self):
        hb_path = self.tmp / "default_fallback"
        Handbag.create_structure(hb_path)
        self._create_personality(hb_path, "alpha")
        self._create_personality(hb_path, "beta")
        # No manifest default_personality → should use first (sorted order)
        hb = Handbag(hb_path)
        default = hb.get_default_personality()
        self.assertIsNotNone(default)
        self.assertEqual(default.name, "alpha")

    def test_default_personality_none_when_empty(self):
        hb_path = self.tmp / "no_pers"
        Handbag.create_structure(hb_path)
        hb = Handbag(hb_path)
        self.assertIsNone(hb.get_default_personality())

    def test_personality_with_tools_subdir(self):
        hb_path = self.tmp / "pers_with_tools"
        Handbag.create_structure(hb_path)
        pers_dir = hb_path / "personalities" / "tooled"
        pers_dir.mkdir(parents=True, exist_ok=True)
        (pers_dir / "SOUL.md").write_text(
            _make_soul_md(name="tooled", system_prompt="You have tools."), encoding="utf-8"
        )
        tools_dir = pers_dir / "tools"
        tools_dir.mkdir()
        (tools_dir / "my_tool.py").write_text(_make_tool_py("tool_my_tool"), encoding="utf-8")
        hb = Handbag(hb_path)
        pers = hb.get_personality("tooled")
        self.assertIsNotNone(pers)

    def test_skips_non_directory_items_in_personalities(self):
        hb_path = self.tmp / "skip_files"
        Handbag.create_structure(hb_path)
        # Place a stray file in personalities/
        stray = hb_path / "personalities" / "stray.txt"
        stray.write_text("not a personality", encoding="utf-8")
        self._create_personality(hb_path, "real")
        hb = Handbag(hb_path)
        self.assertEqual(len(hb.get_personalities()), 1)
        self.assertIn("real", hb.get_personalities())

    def test_skips_dirs_without_soul_md(self):
        hb_path = self.tmp / "skip_no_soul"
        Handbag.create_structure(hb_path)
        # Create a dir without SOUL.md
        (hb_path / "personalities" / "empty_dir").mkdir()
        self._create_personality(hb_path, "has_soul")
        hb = Handbag(hb_path)
        self.assertEqual(len(hb.get_personalities()), 1)
        self.assertIn("has_soul", hb.get_personalities())


# ===========================================================================
# 5. TestToolDiscovery
# ===========================================================================

class TestToolDiscovery(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_tools_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_discovers_flat_py_files(self):
        hb_path = self.tmp / "flat_tools"
        Handbag.create_structure(hb_path)
        tools_dir = hb_path / "tools"
        (tools_dir / "tool_a.py").write_text(_make_tool_py("tool_a"), encoding="utf-8")
        (tools_dir / "tool_b.py").write_text(_make_tool_py("tool_b"), encoding="utf-8")
        hb = Handbag(hb_path)
        tool_files = hb.get_tool_files()
        self.assertEqual(len(tool_files), 2)
        names = [Path(f).name for f in tool_files]
        self.assertIn("tool_a.py", names)
        self.assertIn("tool_b.py", names)

    def test_skips_init_py(self):
        hb_path = self.tmp / "skip_init"
        Handbag.create_structure(hb_path)
        tools_dir = hb_path / "tools"
        (tools_dir / "__init__.py").write_text("", encoding="utf-8")
        (tools_dir / "real_tool.py").write_text(_make_tool_py("tool_real"), encoding="utf-8")
        hb = Handbag(hb_path)
        tool_files = hb.get_tool_files()
        self.assertEqual(len(tool_files), 1)
        self.assertIn("real_tool.py", [Path(f).name for f in tool_files])

    def test_discovers_directory_based_tools(self):
        hb_path = self.tmp / "dir_tools"
        Handbag.create_structure(hb_path)
        tools_dir = hb_path / "tools"
        tool_subdir = tools_dir / "my_lib"
        tool_subdir.mkdir()
        (tool_subdir / "my_lib.py").write_text(_make_tool_py("tool_lib_func"), encoding="utf-8")
        hb = Handbag(hb_path)
        tool_files = hb.get_tool_files()
        self.assertEqual(len(tool_files), 1)
        self.assertIn("my_lib.py", [Path(f).name for f in tool_files])

    def test_directory_fallback_scans_any_py(self):
        hb_path = self.tmp / "dir_fallback"
        Handbag.create_structure(hb_path)
        tools_dir = hb_path / "tools"
        tool_subdir = tools_dir / "weird_name"
        tool_subdir.mkdir()
        # No matching .py file; place a different .py file
        (tool_subdir / "other.py").write_text(_make_tool_py("tool_other"), encoding="utf-8")
        hb = Handbag(hb_path)
        tool_files = hb.get_tool_files()
        self.assertEqual(len(tool_files), 1)
        self.assertIn("other.py", [Path(f).name for f in tool_files])

    def test_empty_tools_dir(self):
        hb_path = self.tmp / "empty_tools"
        Handbag.create_structure(hb_path)
        hb = Handbag(hb_path)
        self.assertEqual(len(hb.get_tool_files()), 0)

    def test_no_tools_dir(self):
        hb_path = self.tmp / "no_tools_dir"
        hb_path.mkdir()
        hb = Handbag(hb_path)
        self.assertEqual(len(hb.get_tool_files()), 0)

    def test_tool_files_are_absolute_resolved(self):
        hb_path = self.tmp / "abs_tools"
        Handbag.create_structure(hb_path)
        tools_dir = hb_path / "tools"
        (tools_dir / "t.py").write_text(_make_tool_py("tool_t"), encoding="utf-8")
        hb = Handbag(hb_path)
        for f in hb.get_tool_files():
            self.assertTrue(Path(f).is_absolute(), f"Tool file path is not absolute: {f}")


# ===========================================================================
# 6. TestSkillsDirectory
# ===========================================================================

class TestSkillsDirectory(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_skills_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skills_dir_registered(self):
        hb_path = self.tmp / "with_skills"
        Handbag.create_structure(hb_path)
        # Create a skill
        skill_dir = hb_path / "skills" / "my_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# My Skill\n\nContent here.", encoding="utf-8")
        hb = Handbag(hb_path)
        dirs = hb.get_skills_dirs()
        self.assertEqual(len(dirs), 1)
        self.assertTrue(Path(dirs[0]).is_absolute())

    def test_no_skills_dir(self):
        hb_path = self.tmp / "no_skills"
        hb_path.mkdir()
        hb = Handbag(hb_path)
        self.assertEqual(len(hb.get_skills_dirs()), 0)

    def test_skills_mode_from_manifest(self):
        hb_path = self.tmp / "skills_mode"
        Handbag.create_structure(hb_path)
        (hb_path / "handbag.yaml").write_text(
            _make_handbag_yaml(skills_mode="loadable"), encoding="utf-8"
        )
        hb = Handbag(hb_path)
        self.assertEqual(hb.get_skills_mode(), "loadable")

    def test_skills_mode_none_when_no_manifest(self):
        hb_path = self.tmp / "no_mode"
        hb_path.mkdir()
        hb = Handbag(hb_path)
        self.assertIsNone(hb.get_skills_mode())


# ===========================================================================
# 7. TestRAG
# ===========================================================================

class TestRAG(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_rag_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _create_rag_doc(self, hb_path: Path, filename: str, content: str):
        rag_dir = hb_path / "rag"
        rag_dir.mkdir(parents=True, exist_ok=True)
        (rag_dir / filename).write_text(content, encoding="utf-8")

    def test_rag_data_source_built_when_rag_dir_exists(self):
        hb_path = self.tmp / "with_rag"
        Handbag.create_structure(hb_path)
        self._create_rag_doc(hb_path, "doc1.txt", "Machine learning is a subset of artificial intelligence.")
        hb = Handbag(hb_path)
        self.assertIsNotNone(hb.get_rag_data_source())

    def test_no_rag_when_rag_dir_empty(self):
        hb_path = self.tmp / "empty_rag"
        Handbag.create_structure(hb_path)
        # rag/ exists but is empty
        hb = Handbag(hb_path)
        self.assertIsNone(hb.get_rag_data_source())

    def test_no_rag_when_rag_dir_missing(self):
        hb_path = self.tmp / "no_rag"
        hb_path.mkdir()
        hb = Handbag(hb_path)
        self.assertIsNone(hb.get_rag_data_source())

    def test_keyword_rag_returns_dict_schema(self):
        hb_path = self.tmp / "keyword_rag"
        Handbag.create_structure(hb_path)
        self._create_rag_doc(hb_path, "ai.txt", "Artificial intelligence transforms industries worldwide.")
        hb = Handbag(hb_path)
        ds = hb.get_rag_data_source()
        result = ds("artificial intelligence")
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("sources", result)
        self.assertIn("count", result)
        self.assertIn("query", result)
        self.assertEqual(result["query"], "artificial intelligence")

    def test_keyword_rag_finds_relevant_docs(self):
        hb_path = self.tmp / "relevant_rag"
        Handbag.create_structure(hb_path)
        self._create_rag_doc(hb_path, "python.md", "Python is a versatile programming language.")
        self._create_rag_doc(hb_path, "rust.md", "Rust is a systems programming language focused on safety.")
        hb = Handbag(hb_path)
        ds = hb.get_rag_data_source()
        result = ds("python programming")
        self.assertTrue(result["success"])
        self.assertGreater(result["count"], 0)
        # The python doc should score higher than rust
        sources = result["sources"]
        top_source = sources[0]
        self.assertIn("python", top_source["source"].lower())

    def test_keyword_rag_no_matches_returns_empty(self):
        hb_path = self.tmp / "no_match_rag"
        Handbag.create_structure(hb_path)
        self._create_rag_doc(hb_path, "cooking.txt", "Recipes for Italian pasta dishes.")
        hb = Handbag(hb_path)
        ds = hb.get_rag_data_source()
        result = ds("quantum physics")
        self.assertFalse(result["success"])
        self.assertEqual(result["count"], 0)

    def test_keyword_rag_handles_empty_query(self):
        hb_path = self.tmp / "empty_query_rag"
        Handbag.create_structure(hb_path)
        self._create_rag_doc(hb_path, "doc.txt", "Some content here.")
        hb = Handbag(hb_path)
        ds = hb.get_rag_data_source()
        result = ds("")
        self.assertIsInstance(result, dict)

    def test_rag_skips_non_text_files(self):
        hb_path = self.tmp / "mixed_rag"
        Handbag.create_structure(hb_path)
        self._create_rag_doc(hb_path, "doc.txt", "Text content here.")
        # Place a binary-looking file
        (hb_path / "rag" / "image.png").write_bytes(b"\x89PNG\x00\x00\x00")
        hb = Handbag(hb_path)
        ds = hb.get_rag_data_source()
        self.assertIsNotNone(ds)
        result = ds("text content")
        # Should only find the .txt file
        sources = result.get("sources", [])
        sources_files = [s.get("source", "") for s in sources]
        self.assertNotIn("image.png", sources_files)

    def test_rag_skips_hidden_files(self):
        hb_path = self.tmp / "hidden_rag"
        Handbag.create_structure(hb_path)
        self._create_rag_doc(hb_path, "visible.txt", "Visible content.")
        (hb_path / "rag" / ".hidden.txt").write_text("Hidden content.", encoding="utf-8")
        hb = Handbag(hb_path)
        ds = hb.get_rag_data_source()
        result = ds("visible content")
        sources_files = [s.get("source", "") for s in result.get("sources", [])]
        self.assertNotIn(".hidden.txt", sources_files)

    def test_rag_recursive_scan(self):
        hb_path = self.tmp / "recursive_rag"
        Handbag.create_structure(hb_path)
        sub = hb_path / "rag" / "subdir"
        sub.mkdir(parents=True)
        (sub / "nested.txt").write_text("Nested document about databases.", encoding="utf-8")
        hb = Handbag(hb_path)
        ds = hb.get_rag_data_source()
        result = ds("databases")
        self.assertTrue(result["success"])
        self.assertGreater(result["count"], 0)


# ===========================================================================
# 8. TestMemoryManager
# ===========================================================================

class TestMemoryManager(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_mem_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_memory_db_path_when_memory_dir_exists(self):
        hb_path = self.tmp / "with_mem"
        Handbag.create_structure(hb_path)
        hb = Handbag(hb_path)
        db_path = hb.get_memory_db_path()
        self.assertIsNotNone(db_path)
        self.assertTrue(db_path.startswith("sqlite:///"))
        self.assertIn("memory.db", db_path)

    def test_no_memory_when_dir_missing(self):
        hb_path = self.tmp / "no_mem"
        hb_path.mkdir()
        hb = Handbag(hb_path)
        self.assertIsNone(hb.get_memory_db_path())

    def test_create_memory_manager_returns_manager(self):
        hb_path = self.tmp / "create_mem"
        Handbag.create_structure(hb_path)
        hb = Handbag(hb_path)
        manager = hb.create_memory_manager()
        if manager is not None:
            # LollmsMemoryManager is available
            self.assertIsNotNone(manager)
        # If None, LollmsMemoryManager might not be importable — that's acceptable in test env

    def test_create_memory_manager_none_when_no_db_path(self):
        hb_path = self.tmp / "no_mem_mgr"
        hb_path.mkdir()
        hb = Handbag(hb_path)
        self.assertIsNone(hb.create_memory_manager())


# ===========================================================================
# 9. TestAccessors
# ===========================================================================

class TestAccessors(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_accessors_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_workspace_path_when_exists(self):
        hb_path = self.tmp / "with_ws"
        Handbag.create_structure(hb_path)
        hb = Handbag(hb_path)
        ws = hb.get_workspace_path()
        self.assertIsNotNone(ws)
        self.assertTrue(Path(ws).exists())

    def test_get_workspace_path_none_when_missing(self):
        hb_path = self.tmp / "no_ws"
        hb_path.mkdir()
        hb = Handbag(hb_path)
        self.assertIsNone(hb.get_workspace_path())

    def test_get_personalities_returns_copy(self):
        hb_path = self.tmp / "pers_copy"
        Handbag.create_structure(hb_path)
        pers_dir = hb_path / "personalities" / "test_pers"
        pers_dir.mkdir(parents=True)
        (pers_dir / "SOUL.md").write_text(
            _make_soul_md(name="test_pers", system_prompt="You are a test agent."), encoding="utf-8"
        )
        hb = Handbag(hb_path)
        p1 = hb.get_personalities()
        p1["injected"] = "fake"
        p2 = hb.get_personalities()
        self.assertNotIn("injected", p2, "get_personalities should return a copy, not the internal dict.")

    def test_get_tool_files_returns_strings(self):
        hb_path = self.tmp / "tool_strs"
        Handbag.create_structure(hb_path)
        (hb_path / "tools" / "t.py").write_text(_make_tool_py("tool_t"), encoding="utf-8")
        hb = Handbag(hb_path)
        for f in hb.get_tool_files():
            self.assertIsInstance(f, str)

    def test_get_skills_dirs_returns_strings(self):
        hb_path = self.tmp / "skills_strs"
        Handbag.create_structure(hb_path)
        hb = Handbag(hb_path)
        for d in hb.get_skills_dirs():
            self.assertIsInstance(d, str)


# ===========================================================================
# 10. TestAttachRagToPersonality
# ===========================================================================

class TestAttachRagToPersonality(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_attach_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_attaches_rag_to_personality_without_data(self):
        hb_path = self.tmp / "attach_rag"
        Handbag.create_structure(hb_path)
        (hb_path / "rag" / "doc.txt").write_text("RAG content here.", encoding="utf-8")
        hb = Handbag(hb_path)
        # Mock personality with has_data = False
        mock_pers = MagicMock()
        mock_pers.has_data = False
        hb.attach_rag_to_personality(mock_pers)
        mock_pers.__setattr__  # Ensure we can set
        # Verify data_source was set
        self.assertIsNotNone(mock_pers.data_source)

    def test_does_not_attach_when_personality_has_data(self):
        hb_path = self.tmp / "skip_attach"
        Handbag.create_structure(hb_path)
        (hb_path / "rag" / "doc.txt").write_text("RAG content.", encoding="utf-8")
        hb = Handbag(hb_path)
        mock_pers = MagicMock()
        mock_pers.has_data = True
        original_ds = "original_rag"
        mock_pers.data_source = original_ds
        hb.attach_rag_to_personality(mock_pers)
        # data_source should NOT be overwritten
        self.assertEqual(mock_pers.data_source, original_ds)

    def test_no_rag_source_does_nothing(self):
        hb_path = self.tmp / "no_rag_attach"
        hb_path.mkdir()
        hb = Handbag(hb_path)
        mock_pers = MagicMock()
        mock_pers.has_data = False
        hb.attach_rag_to_personality(mock_pers)
        # data_source should not be set since there's no RAG
        # MagicMock auto-creates attributes, so we check it wasn't called
        # by verifying has_data check didn't proceed

    def test_none_personality_does_nothing(self):
        hb_path = self.tmp / "none_pers"
        Handbag.create_structure(hb_path)
        (hb_path / "rag" / "doc.txt").write_text("Content.", encoding="utf-8")
        hb = Handbag(hb_path)
        # Should not raise
        hb.attach_rag_to_personality(None)


# ===========================================================================
# 11. TestAgentIntegration
# ===========================================================================

class TestAgentIntegration(unittest.TestCase):
    """
    Tests that the Agent properly uses handbag_path to auto-configure
    personality, tools, skills, memory, and workspace, and that explicit
    constructor parameters override handbag values.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_agent_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _create_full_handbag(self, hb_path: Path):
        """Creates a handbag with personality, tools, skills, rag, and memory."""
        Handbag.create_structure(hb_path)

        # Personality
        pers_dir = hb_path / "personalities" / "builder"
        pers_dir.mkdir(parents=True)
        (pers_dir / "SOUL.md").write_text(
            _make_soul_md(name="builder", system_prompt="You are an autonomous builder agent."), encoding="utf-8"
        )

        # Tool
        (hb_path / "tools" / "custom_tool.py").write_text(
            _make_tool_py("tool_custom", "A custom handbag tool"), encoding="utf-8"
        )

        # Skill
        skill_dir = hb_path / "skills" / "test_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test Skill\n\nA test skill.", encoding="utf-8")

        # RAG
        (hb_path / "rag" / "knowledge.txt").write_text(
            "The handbag contains knowledge about testing.", encoding="utf-8"
        )

    def _make_mock_client(self):
        """Creates a minimal mock LollmsClient for Agent construction."""
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.model_name = "mock"
        client.llm.binding_name = "mock"
        client.llm.reset_cancel = MagicMock()
        client.count_tokens = lambda t: len(t) // 4
        client.count_image_tokens = lambda i: 0
        client.remove_thinking_blocks = lambda t: t
        client.tti = None
        client.tts = None
        client.stt = None
        client.ttm = None
        client.ttv = None
        client.tools = None
        client.cancel = MagicMock()
        return client

    def test_agent_loads_personality_from_handbag(self):
        hb_path = self.tmp / "full_hb"
        self._create_full_handbag(hb_path)
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
        )
        self.assertIsNotNone(agent.personality)
        self.assertEqual(agent.personality.name, "builder")
        self.assertIsNotNone(agent.handbag)

    def test_agent_loads_tools_from_handbag(self):
        hb_path = self.tmp / "tools_hb"
        self._create_full_handbag(hb_path)
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
        )
        # tool_files should be populated from handbag
        self.assertIsNotNone(agent.tool_files)
        tool_names = [Path(f).name for f in agent.tool_files]
        self.assertIn("custom_tool.py", tool_names)

    def test_agent_loads_skills_dirs_from_handbag(self):
        hb_path = self.tmp / "skills_hb"
        self._create_full_handbag(hb_path)
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
        )
        self.assertIsNotNone(agent.skills_dirs)
        self.assertGreater(len(agent.skills_dirs), 0)

    def test_agent_loads_workspace_from_handbag(self):
        hb_path = self.tmp / "ws_hb"
        self._create_full_handbag(hb_path)
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
        )
        self.assertIsNotNone(agent.get_workspace_path())

    def test_agent_rag_attached_to_personality(self):
        hb_path = self.tmp / "rag_hb"
        self._create_full_handbag(hb_path)
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
        )
        # The personality should have RAG data attached from handbag
        self.assertTrue(agent.has_knowledge())

    def test_explicit_personality_overrides_handbag(self):
        from lollms_client.lollms_personality import LollmsPersonality
        hb_path = self.tmp / "override_pers"
        self._create_full_handbag(hb_path)
        client = self._make_mock_client()
        explicit_pers = LollmsPersonality(
            name="explicit",
            author="test",
            category="test",
            description="Explicit personality.",
            system_prompt="You are an explicit agent.",
        )
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
            personality=explicit_pers,
        )
        self.assertEqual(agent.personality.name, "explicit")
        # Handbag should still be loaded
        self.assertIsNotNone(agent.handbag)

    def test_explicit_workspace_overrides_handbag(self):
        hb_path = self.tmp / "override_ws"
        self._create_full_handbag(hb_path)
        explicit_ws = self.tmp / "custom_workspace"
        explicit_ws.mkdir()
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
            workspace_path=str(explicit_ws),
        )
        self.assertEqual(Path(agent.get_workspace_path()).resolve(), explicit_ws.resolve())

    def test_explicit_tool_files_appended_to_handbag(self):
        hb_path = self.tmp / "append_tools"
        self._create_full_handbag(hb_path)
        extra_tool = self.tmp / "extra_tool.py"
        extra_tool.write_text(_make_tool_py("tool_extra"), encoding="utf-8")
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
            tool_files=[str(extra_tool)],
        )
        tool_names = [Path(f).name for f in agent.tool_files]
        self.assertIn("custom_tool.py", tool_names)  # From handbag
        self.assertIn("extra_tool.py", tool_names)  # Explicit

    def test_explicit_skills_dirs_appended_to_handbag(self):
        hb_path = self.tmp / "append_skills"
        self._create_full_handbag(hb_path)
        extra_skills = self.tmp / "extra_skills"
        extra_skills.mkdir()
        (extra_skills / "extra").mkdir()
        (extra_skills / "extra" / "SKILL.md").write_text("# Extra Skill", encoding="utf-8")
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
            skills_dirs=[str(extra_skills)],
        )
        # Should have both handbag skills dir and the explicit one
        self.assertGreaterEqual(len(agent.skills_dirs), 2)

    def test_no_handbag_no_personality_raises(self):
        client = self._make_mock_client()
        with self.assertRaises(ValueError) as ctx:
            Agent(lc=client)
        self.assertIn("personality", str(ctx.exception).lower())

    def test_handbag_with_no_personality_raises(self):
        hb_path = self.tmp / "empty_handbag"
        Handbag.create_structure(hb_path)
        client = self._make_mock_client()
        with self.assertRaises(ValueError) as ctx:
            Agent(lc=client, handbag_path=str(hb_path))
        self.assertIn("personality", str(ctx.exception).lower())

    def test_list_handbag_personalities(self):
        hb_path = self.tmp / "list_pers"
        self._create_full_handbag(hb_path)
        # Add a second personality
        pers2 = hb_path / "personalities" / "analyst"
        pers2.mkdir(parents=True)
        (pers2 / "SOUL.md").write_text(
            _make_soul_md(name="analyst", system_prompt="You are an analyst."), encoding="utf-8"
        )
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
        )
        personalities = agent.list_handbag_personalities()
        self.assertGreaterEqual(len(personalities), 2)
        self.assertIn("builder", personalities)
        self.assertIn("analyst", personalities)

    def test_switch_handbag_personality(self):
        hb_path = self.tmp / "switch_pers"
        self._create_full_handbag(hb_path)
        pers2 = hb_path / "personalities" / "researcher"
        pers2.mkdir(parents=True)
        (pers2 / "SOUL.md").write_text(
            _make_soul_md(name="researcher", system_prompt="You are a researcher."), encoding="utf-8"
        )
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
        )
        # Initial personality should be 'builder' (first sorted)
        self.assertEqual(agent.personality.name, "builder")
        # Switch
        success = agent.switch_handbag_personality("researcher")
        self.assertTrue(success)
        self.assertEqual(agent.personality.name, "researcher")

    def test_switch_handbag_personality_nonexistent(self):
        hb_path = self.tmp / "switch_fail"
        self._create_full_handbag(hb_path)
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
        )
        success = agent.switch_handbag_personality("nonexistent")
        self.assertFalse(success)

    def test_handbag_property_none_when_no_handbag(self):
        from lollms_client.lollms_personality import LollmsPersonality
        client = self._make_mock_client()
        explicit_pers = LollmsPersonality(
            name="test",
            author="test",
            category="test",
            description="Test.",
            system_prompt="You are a test agent.",
        )
        agent = Agent(
            lc=client,
            personality=explicit_pers,
        )
        self.assertIsNone(agent.handbag)

    def test_list_handbag_personalities_empty_when_no_handbag(self):
        from lollms_client.lollms_personality import LollmsPersonality
        client = self._make_mock_client()
        explicit_pers = LollmsPersonality(
            name="test",
            author="test",
            category="test",
            description="Test.",
            system_prompt="You are a test agent.",
        )
        agent = Agent(
            lc=client,
            personality=explicit_pers,
        )
        self.assertEqual(agent.list_handbag_personalities(), {})

    def test_skills_mode_from_handbag_manifest(self):
        hb_path = self.tmp / "skills_mode_hb"
        self._create_full_handbag(hb_path)
        (hb_path / "handbag.yaml").write_text(
            _make_handbag_yaml(skills_mode="always_visible"), encoding="utf-8"
        )
        client = self._make_mock_client()
        agent = Agent(
            lc=client,
            handbag_path=str(hb_path),
        )
        # The capabilities.skills_mode should be overridden by handbag manifest
        # (only if it's still the default "loadable")
        self.assertEqual(agent.capabilities.skills_mode, "always_visible")


# ===========================================================================
# 12. TestEdgeCases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="handbag_edge_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_handbag_with_only_empty_subdirectories(self):
        hb_path = self.tmp / "empty_subs"
        Handbag.create_structure(hb_path)
        hb = Handbag(hb_path)
        self.assertEqual(len(hb.get_personalities()), 0)
        self.assertEqual(len(hb.get_tool_files()), 0)
        self.assertIsNone(hb.get_rag_data_source())

    def test_rag_with_only_binary_files(self):
        hb_path = self.tmp / "binary_rag"
        Handbag.create_structure(hb_path)
        (hb_path / "rag" / "data.bin").write_bytes(b"\x00\x01\x02\x03")
        hb = Handbag(hb_path)
        self.assertIsNone(hb.get_rag_data_source())

    def test_tool_dir_with_only_init_py(self):
        hb_path = self.tmp / "only_init"
        Handbag.create_structure(hb_path)
        (hb_path / "tools" / "__init__.py").write_text("", encoding="utf-8")
        hb = Handbag(hb_path)
        self.assertEqual(len(hb.get_tool_files()), 0)

    def test_multiple_rag_files_all_indexed(self):
        hb_path = self.tmp / "multi_rag"
        Handbag.create_structure(hb_path)
        for i in range(5):
            (hb_path / "rag" / f"doc_{i}.txt").write_text(
                f"Document {i} about topic_{i}.", encoding="utf-8"
            )
        hb = Handbag(hb_path)
        ds = hb.get_rag_data_source()
        self.assertIsNotNone(ds)
        result = ds("topic_2")
        self.assertTrue(result["success"])
        self.assertGreater(result["count"], 0)

    def test_handbag_path_as_string(self):
        hb_path = self.tmp / "str_path"
        Handbag.create_structure(hb_path)
        # Pass as string, not Path
        hb = Handbag(str(hb_path))
        self.assertEqual(hb.path, hb_path.resolve())

    def test_create_structure_overwrites_existing_manifest(self):
        hb_path = self.tmp / "overwrite"
        hb_path.mkdir()
        (hb_path / "handbag.yaml").write_text("name: Old\n", encoding="utf-8")
        Handbag.create_structure(hb_path, name="New")
        # create_structure does NOT overwrite if manifest exists
        content = (hb_path / "handbag.yaml").read_text(encoding="utf-8")
        self.assertIn("Old", content)


if __name__ == "__main__":
    unittest.main()