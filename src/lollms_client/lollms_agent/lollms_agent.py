# lollms_agent.py
# High-Grade Agentic System with Skills, Sub-Agents, Model Switching, and Binding Integration.
#
# Key Subsystems:
#   - SkillsManager: Loadable/always-visible SKILL.md files (separate from workspace)
#   - CapabilityFlags: Boolean gates for code execution, networking, sub-agents, etc.
#   - SubAgentSpawner: Delegation to focused child agents with depth/count limits
#   - ModelSwitcher: On-the-fly model switching via lollms_client's mount capabilities
#   - BindingToolsBuilder: Exposes TTI/TTS/STT/TTM/TTV bindings as callable tools
#   - ToolsManager: File-based LCP tool discovery and execution
#   - _AgentStreamState: Transactional stream parser for <tool> and <done/> tags

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import sys
import uuid
import traceback
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from ascii_colors import ASCIIColors, trace_exception

if TYPE_CHECKING:
    from lollms_client.lollms_core import LollmsClient
    from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

try:
    from lollms_client.lollms_types import MSG_TYPE
except ImportError:
    class MSG_TYPE:
        MSG_TYPE_CHUNK = "chunk"
        MSG_TYPE_INFO = "info"
        MSG_TYPE_NEW_MESSAGE = "new_message"
        MSG_TYPE_THOUGHT_CHUNK = "thought"

try:
    from lollms_client.lollms_memory import FailureMemory
except ImportError:
    class FailureMemory:
        def __init__(self):
            self.failures = []
            self._signatures = set()
        def record_failure_by_signature(self, sig, error):
            self.failures.append({"signature": sig, "error": error})
            self._signatures.add(sig)


# ===========================================================================
# AgentRole — semantic label for an agent's function
# ===========================================================================

class AgentRole:
    PROPOSER        = "proposer"
    CRITIC          = "critic"
    DEVIL_ADVOCATE  = "devil_advocate"
    DOMAIN_EXPERT   = "domain_expert"
    SYNTHESIZER     = "synthesizer"
    MODERATOR       = "moderator"
    IMPLEMENTER     = "implementer"
    TESTER          = "tester"
    NARRATOR        = "narrator"
    PLAYER          = "player"
    FREEFORM        = "freeform"


# ===========================================================================
# Skill — A single skill data structure
# ===========================================================================

@dataclass
class Skill:
    """Represents a single skill loaded from a SKILL.md file."""
    title: str
    description: str
    category: str
    tags: List[str]
    content: str
    file_path: Optional[Path] = None
    always_visible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "always_visible": self.always_visible,
            "file_path": str(self.file_path) if self.file_path else None,
        }


# ===========================================================================
# SkillsManager — Loads and manages SKILL.md files from external directories
# ===========================================================================

_DEFAULT_SKILLS_DIR = Path.home() / ".lollms_hub" / "agent_skills"


def _parse_skill_md(file_path: Path, default_always_visible: bool = False) -> Optional[Skill]:
    """Parses a SKILL.md file into a Skill object. Supports YAML frontmatter and plain markdown."""
    try:
        raw_content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    title = file_path.stem
    description = ""
    category = ""
    tags: List[str] = []
    body = raw_content
    fm_always_visible = None  # Track frontmatter always_visible override

    # Try YAML frontmatter
    if raw_content.startswith("---"):
        fm_match = re.match(r'^---\n(.*?)\n---\n(.*)', raw_content, re.DOTALL)
        if fm_match:
            fm_text = fm_match.group(1)
            body = fm_match.group(2)
            for line in fm_text.splitlines():
                line = line.strip()
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("category:"):
                    category = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("tags:"):
                    tags_str = line.split(":", 1)[1].strip()
                    if tags_str.startswith("[") and tags_str.endswith("]"):
                        tags_str = tags_str[1:-1]
                    tags = [t.strip().strip('"\'') for t in tags_str.split(",") if t.strip()]
                elif line.startswith("always_visible:"):
                    val = line.split(":", 1)[1].strip().lower()
                    if val in ("true", "1", "yes"):
                        fm_always_visible = True
                    elif val in ("false", "0", "no"):
                        fm_always_visible = False
        else:
            # Malformed frontmatter, treat as plain markdown
            pass
    else:
        # Plain markdown: first H1 as title, first paragraph as description
        h1_match = re.match(r'^#\s+(.+)', raw_content)
        if h1_match:
            title = h1_match.group(1).strip()
            rest = raw_content[h1_match.end():].strip()
            desc_match = re.match(r'^([^\n#]+)', rest)
            if desc_match:
                description = desc_match.group(1).strip()

    return Skill(
        title=title,
        description=description,
        category=category,
        tags=tags,
        content=body.strip(),
        file_path=file_path,
        always_visible=fm_always_visible if fm_always_visible is not None else default_always_visible,
    )


class SkillsManager:
    """
    Manages SKILL.md files from external directories (not the workspace).

    Modes:
        - "always_visible": All skill contents are injected into the system prompt.
        - "loadable": Only skill titles/descriptions are listed; the agent loads content on-demand.
        - "mixed": Always-visible skills are fully injected; loadable skills are listed by title only.
    """

    def __init__(
        self,
        skills_dirs: Optional[List[Union[str, Path]]] = None,
        skills_files: Optional[List[Union[str, Path]]] = None,
        mode: str = "loadable",
        default_skills_dir: Optional[Union[str, Path]] = None,
    ):
        self.mode = mode
        self._default_dir = Path(default_skills_dir) if default_skills_dir else _DEFAULT_SKILLS_DIR
        self._skills_dirs: List[Path] = []
        self._skills_files: List[Path] = []
        self.skills: Dict[str, Skill] = {}

        # Collect directories
        if skills_dirs:
            for d in skills_dirs:
                p = Path(d)
                if p.exists() and p.is_dir():
                    self._skills_dirs.append(p.resolve())

        # Always include default directory
        self._default_dir.mkdir(parents=True, exist_ok=True)
        if self._default_dir not in self._skills_dirs:
            self._skills_dirs.append(self._default_dir.resolve())

        # Collect explicit files
        if skills_files:
            for f in skills_files:
                p = Path(f)
                if p.exists() and p.is_file() and p.suffix.lower() == ".md":
                    self._skills_files.append(p.resolve())

        self.reload()

    def reload(self):
        """Reloads all skills from configured directories and files."""
        self.skills.clear()
        seen_paths = set()

        # Load from explicit files first
        for fp in self._skills_files:
            if fp in seen_paths:
                continue
            seen_paths.add(fp)
            skill = _parse_skill_md(fp, default_always_visible=(self.mode == "always_visible"))
            if skill:
                self.skills[skill.title.lower()] = skill

        # Load from directories
        for d in self._skills_dirs:
            self._scan_directory(d, seen_paths)

    def _scan_directory(self, directory: Path, seen_paths: set):
        """Scans a directory for SKILL.md files (either directly or in subdirectories)."""
        if not directory.exists() or not directory.is_dir():
            return

        # Check for SKILL.md directly in this directory
        direct_skill = directory / "SKILL.md"
        if direct_skill.exists() and direct_skill.resolve() not in seen_paths:
            seen_paths.add(direct_skill.resolve())
            skill = _parse_skill_md(direct_skill, default_always_visible=(self.mode == "always_visible"))
            if skill:
                self.skills[skill.title.lower()] = skill
            return  # This directory IS a skill, don't scan deeper

        # Scan subdirectories for SKILL.md files
        for item in sorted(directory.iterdir()):
            if item.is_dir():
                skill_file = item / "SKILL.md"
                if skill_file.exists() and skill_file.resolve() not in seen_paths:
                    seen_paths.add(skill_file.resolve())
                    skill = _parse_skill_md(skill_file, default_always_visible=(self.mode == "always_visible"))
                    if skill:
                        self.skills[skill.title.lower()] = skill
            elif item.is_file() and item.suffix.lower() == ".md" and item.name != "README.md":
                if item.resolve() not in seen_paths:
                    seen_paths.add(item.resolve())
                    skill = _parse_skill_md(item, default_always_visible=(self.mode == "always_visible"))
                    if skill:
                        self.skills[skill.title.lower()] = skill

    def get_visible_skills_context(self) -> str:
        """Builds the system prompt context for always-visible skills."""
        visible = [s for s in self.skills.values() if s.always_visible]
        if not visible:
            return ""
        lines = ["=== ACTIVE SKILLS (Always Visible) ==="]
        for skill in visible:
            lines.append(f"\n--- Skill: {skill.title} ---")
            if skill.description:
                lines.append(f"Description: {skill.description}")
            if skill.category:
                lines.append(f"Category: {skill.category}")
            lines.append(f"\n{skill.content}")
            lines.append(f"--- End Skill: {skill.title} ---")
        lines.append("=== END ACTIVE SKILLS ===")
        return "\n".join(lines)

    def get_loadable_skills_index(self) -> str:
        """Builds a compact index of loadable skills for the system prompt."""
        loadable = [s for s in self.skills.values() if not s.always_visible]
        if not loadable:
            return ""
        lines = ["=== AVAILABLE SKILLS (Loadable on Demand) ==="]
        lines.append("Use the `tool_load_skill` tool to load the full content of any skill listed below.")
        lines.append("")
        for skill in loadable:
            desc = skill.description or "No description"
            cat = f" [{skill.category}]" if skill.category else ""
            lines.append(f"- **{skill.title}**{cat}: {desc}")
        lines.append("=== END AVAILABLE SKILLS ===")
        return "\n".join(lines)

    def get_mixed_context(self) -> str:
        """Builds context for mixed mode: visible skills full + loadable skills index."""
        parts = []
        visible_ctx = self.get_visible_skills_context()
        if visible_ctx:
            parts.append(visible_ctx)
        loadable_ctx = self.get_loadable_skills_index()
        if loadable_ctx:
            parts.append(loadable_ctx)
        return "\n\n".join(parts) if parts else ""

    def build_context(self) -> str:
        """Builds the skills context for the system prompt based on the current mode."""
        if self.mode == "always_visible":
            return self.get_visible_skills_context()
        elif self.mode == "loadable":
            return self.get_loadable_skills_index()
        else:  # mixed
            return self.get_mixed_context()

    def search_skills(self, query: str) -> List[Skill]:
        """Searches skills by title, description, tags, or content."""
        query_lower = query.lower()
        results = []
        for skill in self.skills.values():
            score = 0
            if query_lower in skill.title.lower():
                score += 3
            if query_lower in skill.description.lower():
                score += 2
            if any(query_lower in tag.lower() for tag in skill.tags):
                score += 2
            if query_lower in skill.content.lower():
                score += 1
            if score > 0:
                results.append((score, skill))
        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results]

    def load_skill(self, title: str) -> Optional[str]:
        """Loads the full content of a skill by title."""
        skill = self.skills.get(title.lower())
        if skill:
            return f"--- Skill: {skill.title} ---\n{skill.content}\n--- End Skill: {skill.title} ---"
        # Fuzzy search
        matches = self.search_skills(title)
        if matches:
            skill = matches[0]
            return f"--- Skill: {skill.title} ---\n{skill.content}\n--- End Skill: {skill.title} ---"
        return None

    def list_skills(self) -> List[Dict[str, Any]]:
        """Returns a list of all skills with metadata."""
        return [s.to_dict() for s in self.skills.values()]

    def add_skill(
        self,
        title: str,
        description: str,
        category: str,
        content: str,
        tags: Optional[List[str]] = None,
        always_visible: bool = False,
    ) -> Dict[str, Any]:
        """Creates a new SKILL.md file in the default skills directory."""
        tags = tags or []
        safe_title = re.sub(r'[^\w\-_]', '_', title)
        skill_dir = self._default_dir / safe_title
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        fm_lines = ["---"]
        fm_lines.append(f'title: "{title}"')
        fm_lines.append(f'description: "{description}"')
        fm_lines.append(f'category: "{category}"')
        fm_lines.append(f'tags: [{", ".join(tags)}]')
        fm_lines.append(f'always_visible: {str(always_visible).lower()}')
        fm_lines.append("---")
        fm_lines.append("")
        fm_lines.append(content)

        skill_file.write_text("\n".join(fm_lines), encoding="utf-8")

        # Reload to register the new skill
        self.reload()

        return {
            "success": True,
            "output": f"Skill '{title}' created successfully at {skill_file}.",
            "file_path": str(skill_file),
        }

    def update_skill(self, title: str, new_content: str) -> Dict[str, Any]:
        """Updates the content of an existing skill."""
        skill = self.skills.get(title.lower())
        if not skill:
            matches = self.search_skills(title)
            if matches:
                skill = matches[0]
            else:
                return {"success": False, "error": f"Skill '{title}' not found."}

        if not skill.file_path or not skill.file_path.exists():
            return {"success": False, "error": f"Skill file for '{title}' not found on disk."}

        # Read existing file to preserve frontmatter
        raw = skill.file_path.read_text(encoding="utf-8")
        if raw.startswith("---"):
            fm_match = re.match(r'^---\n(.*?)\n---\n(.*)', raw, re.DOTALL)
            if fm_match:
                updated = f"---\n{fm_match.group(1)}\n---\n\n{new_content}"
            else:
                updated = new_content
        else:
            updated = new_content

        skill.file_path.write_text(updated, encoding="utf-8")
        self.reload()

        return {"success": True, "output": f"Skill '{title}' updated successfully."}

    def delete_skill(self, title: str) -> Dict[str, Any]:
        """Deletes a skill and its file."""
        skill = self.skills.get(title.lower())
        if not skill:
            return {"success": False, "error": f"Skill '{title}' not found."}

        if skill.file_path and skill.file_path.exists():
            skill.file_path.unlink()
            # Also remove the parent directory if it's empty (except default dir)
            parent = skill.file_path.parent
            if parent != self._default_dir and parent.exists():
                try:
                    parent.rmdir()  # Only works if empty
                except OSError:
                    pass

        self.reload()
        return {"success": True, "output": f"Skill '{title}' deleted."}


# ===========================================================================
# CapabilityFlags — Boolean gates for agent capabilities
# ===========================================================================

@dataclass
class CapabilityFlags:
    """
    Controls what the agent is allowed to do.
    All dangerous capabilities default to False for safety.
    """
    # Code execution
    enable_code_execution: bool = False

    # File access
    enable_external_file_access: bool = False  # Access files outside workspace

    # Networking
    enable_networking: bool = False  # Internet/network tools

    # Multimodal bindings
    enable_image_generation: bool = True
    enable_image_editing: bool = True
    enable_tts: bool = False
    enable_stt: bool = False
    enable_ttm: bool = False  # Text-to-music
    enable_ttv: bool = False  # Text-to-video

    # Agentic features
    enable_sub_agents: bool = True
    enable_model_switching: bool = False
    enable_skill_creation: bool = True
    enable_skill_loading: bool = True

    # Skills display mode: "always_visible", "loadable", "mixed"
    skills_mode: str = "loadable"

    # Sub-agent limits
    max_sub_agent_depth: int = 3
    max_sub_agents_per_turn: int = 5

    # Workspace file tools (always enabled if workspace is configured)
    # These are not toggleable for security reasons — workspace tools are always safe
    enable_workspace_tools: bool = True  # tool_write_file, tool_read_file, tool_list_files

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_code_execution": self.enable_code_execution,
            "enable_external_file_access": self.enable_external_file_access,
            "enable_networking": self.enable_networking,
            "enable_image_generation": self.enable_image_generation,
            "enable_image_editing": self.enable_image_editing,
            "enable_tts": self.enable_tts,
            "enable_stt": self.enable_stt,
            "enable_ttm": self.enable_ttm,
            "enable_ttv": self.enable_ttv,
            "enable_sub_agents": self.enable_sub_agents,
            "enable_model_switching": self.enable_model_switching,
            "enable_skill_creation": self.enable_skill_creation,
            "enable_skill_loading": self.enable_skill_loading,
            "skills_mode": self.skills_mode,
            "max_sub_agent_depth": self.max_sub_agent_depth,
            "max_sub_agents_per_turn": self.max_sub_agents_per_turn,
        }


# ===========================================================================
# SubAgentSpawner — Delegation to focused child agents
# ===========================================================================

class SubAgentSpawner:
    """
    Spawns child agents for sub-task delegation.
    Enforces recursion depth and per-turn spawn count limits.
    """

    def __init__(self, parent_agent: 'Agent', max_depth: int = 3, max_per_turn: int = 5):
        self.parent = parent_agent
        self.max_depth = max_depth
        self.max_per_turn = max_per_turn
        self._current_depth = 0
        self._spawned_this_turn = 0

    def reset_turn(self):
        self._spawned_this_turn = 0

    def set_depth(self, depth: int):
        self._current_depth = depth

    def can_spawn(self) -> bool:
        return (
            self._current_depth < self.max_depth and
            self._spawned_this_turn < self.max_per_turn
        )

    def spawn(
        self,
        instruction: str,
        personality_conditioning: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        max_steps: int = 5,
    ) -> Dict[str, Any]:
        """
        Spawns a child agent to perform a sub-task.
        The child shares the parent's workspace but has NO sub-agent capability
        (to prevent infinite recursion).

        Args:
            instruction: The specific task for the child agent.
            personality_conditioning: Custom system prompt for the child.
            model_name: Specific model to use (None = parent's model).
            temperature: Low temperature for focused work (default 0.3).
            max_steps: Maximum reasoning steps for the child (default 5).
        """
        if not self.can_spawn():
            return {
                "success": False,
                "error": f"Sub-agent spawn limit reached (depth: {self._current_depth}/{self.max_depth}, spawned: {self._spawned_this_turn}/{self.max_per_turn})."
            }

        self._spawned_this_turn += 1

        try:
            from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

            # Create a focused personality for the child
            child_personality = LollmsPersonality(
                name=f"SubAgent_{uuid.uuid4().hex[:6]}",
                author="lollms_agent",
                category="sub_agent",
                description="A focused sub-agent spawned for a specific task.",
                system_prompt=personality_conditioning or (
                    "You are a focused sub-agent. Execute the given task precisely and return the result. "
                    "Do not engage in conversational pleasantries. Focus solely on the task."
                ),
            )

            # Create child agent with disabled sub-agents and model switching
            child_caps = CapabilityFlags(
                enable_code_execution=self.parent.capabilities.enable_code_execution,
                enable_image_generation=False,  # Children don't need image gen
                enable_image_editing=False,
                enable_sub_agents=False,  # CRITICAL: Prevent infinite recursion
                enable_model_switching=False,
                enable_skill_loading=self.parent.capabilities.enable_skill_loading,
                enable_skill_creation=False,  # Children can't create skills
                skills_mode="loadable",
                max_sub_agent_depth=0,
            )

            child_agent = Agent(
                lc=self.parent.lc,
                personality=child_personality,
                name=f"SubAgent_{self._spawned_this_turn}",
                role=AgentRole.IMPLEMENTER,
                workspace_path=self.parent.get_workspace_path(),
                capabilities=child_caps,
                skills_manager=self.parent.skills_manager,  # Share skills
                model_params=self.parent.model_params,
                max_tokens_per_turn=self.parent.max_tokens_per_turn,
                memory_manager=None,  # Children don't write to memory
                _parent_depth=self._current_depth + 1,
            )

            # If model_name specified, temporarily switch
            original_model = None
            if model_name and hasattr(self.parent.lc, 'llm'):
                try:
                    original_model = getattr(self.parent.lc.llm, 'model_name', None)
                except Exception:
                    pass

            # Execute child chat (non-streaming, no internal history)
            result = child_agent.chat(
                prompt=instruction,
                streaming_callback=None,
                max_reasoning_steps=max_steps,
                temperature=temperature,
                use_internal_history=False,
            )

            child_response = result.get("response", "")
            child_tool_calls = result.get("tool_calls", [])

            return {
                "success": True,
                "output": child_response,
                "child_tool_calls": child_tool_calls,
                "child_rounds": result.get("rounds", 0),
                "prompt_injection": f"\n\n=== 🧠 SUB-AGENT REPORT ===\nThe sub-agent completed: '{instruction[:100]}...'\n\n{child_response}\n=== END SUB-AGENT REPORT ===",
            }

        except Exception as e:
            trace_exception(e)
            return {
                "success": False,
                "error": f"Sub-agent spawn failed: {e}",
                "traceback": traceback.format_exc(),
            }


# ===========================================================================
# ModelSwitcher — On-the-fly model switching
# ===========================================================================

class ModelSwitcher:
    """
    Allows the agent to switch between models during a session.
    Uses the LLM binding's mount/load capabilities.
    """

    def __init__(self, client: 'LollmsClient'):
        self.client = client
        self._original_model: Optional[str] = None
        self._current_model: Optional[str] = None
        self._available_models: List[str] = []

    def _get_llm(self):
        return getattr(self.client, 'llm', None)

    def list_models(self) -> List[str]:
        """Lists available models from the binding."""
        llm = self._get_llm()
        if not llm:
            return []

        # Try different methods based on binding type
        if hasattr(llm, 'list_models'):
            try:
                return llm.list_models()
            except Exception:
                pass

        if hasattr(llm, 'available_models'):
            try:
                return llm.available_models
            except Exception:
                pass

        # For local bindings with a models directory
        if hasattr(llm, 'models_path'):
            try:
                models_dir = Path(llm.models_path)
                if models_dir.exists():
                    exts = {'.gguf', '.bin', '.onnx', '.pt', '.safetensors'}
                    return [f.name for f in models_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
            except Exception:
                pass

        return self._available_models

    def get_current_model(self) -> str:
        llm = self._get_llm()
        if llm:
            return getattr(llm, 'model_name', 'unknown')
        return 'unknown'

    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """
        Switches to a different model.
        For local bindings: unloads current model and loads the new one.
        For remote bindings: updates the model_name parameter.
        """
        llm = self._get_llm()
        if not llm:
            return {"success": False, "error": "No LLM binding available."}

        # Store original model for restoration
        if self._original_model is None:
            self._original_model = getattr(llm, 'model_name', None)

        try:
            # For local bindings with load_model/unload_model
            if hasattr(llm, 'unload_model') and hasattr(llm, 'load_model'):
                try:
                    llm.unload_model()
                except Exception:
                    pass
                success = llm.load_model(model_name)
                if not success:
                    # Try to restore original
                    if self._original_model:
                        try:
                            llm.load_model(self._original_model)
                        except Exception:
                            pass
                    return {"success": False, "error": f"Failed to load model '{model_name}'."}
                self._current_model = model_name
                return {
                    "success": True,
                    "output": f"Switched to model '{model_name}'.",
                    "current_model": model_name,
                }

            # For remote bindings, just set model_name
            elif hasattr(llm, 'model_name'):
                old_model = llm.model_name
                llm.model_name = model_name
                self._current_model = model_name
                return {
                    "success": True,
                    "output": f"Switched from '{old_model}' to '{model_name}'.",
                    "current_model": model_name,
                }

            else:
                return {"success": False, "error": "Binding does not support model switching."}

        except Exception as e:
            trace_exception(e)
            return {"success": False, "error": f"Model switch failed: {e}"}

    def restore_original_model(self) -> Dict[str, Any]:
        """Restores the original model if it was switched."""
        if self._original_model and self._current_model != self._original_model:
            return self.switch_model(self._original_model)
        return {"success": True, "output": "No restoration needed."}


# ===========================================================================
# BindingToolsBuilder — Exposes lollms_client bindings as callable tools
# ===========================================================================

class BindingToolsBuilder:
    """
    Builds callable tools from lollms_client's multimodal bindings (TTI, TTS, STT, etc.).
    Each tool is only registered if the corresponding binding is available and the
    capability flag is enabled.
    """

    @staticmethod
    def build_tools(client: 'LollmsClient', caps: CapabilityFlags, workspace_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
        """Builds all binding-based tools based on available bindings and capability flags."""
        tools: Dict[str, Dict[str, Any]] = {}

        # TTI (Text-to-Image)
        tti = getattr(client, 'tti', None)
        if tti is not None:
            if caps.enable_image_generation:
                tools["tool_generate_image"] = BindingToolsBuilder._make_tti_generate_tool(tti, workspace_path)
            if caps.enable_image_editing:
                tools["tool_edit_image"] = BindingToolsBuilder._make_tti_edit_tool(tti, workspace_path)

        # TTS (Text-to-Speech)
        tts = getattr(client, 'tts', None)
        if tts is not None and caps.enable_tts:
            tools["tool_text_to_speech"] = BindingToolsBuilder._make_tts_tool(tts, workspace_path)

        # STT (Speech-to-Text)
        stt = getattr(client, 'stt', None)
        if stt is not None and caps.enable_stt:
            tools["tool_speech_to_text"] = BindingToolsBuilder._make_stt_tool(stt, workspace_path)

        # TTM (Text-to-Music)
        ttm = getattr(client, 'ttm', None)
        if ttm is not None and caps.enable_ttm:
            tools["tool_generate_music"] = BindingToolsBuilder._make_ttm_tool(ttm, workspace_path)

        # TTV (Text-to-Video)
        ttv = getattr(client, 'ttv', None)
        if ttv is not None and caps.enable_ttv:
            tools["tool_generate_video"] = BindingToolsBuilder._make_ttv_tool(ttv, workspace_path)

        return tools

    @staticmethod
    def _make_tti_generate_tool(tti_binding, workspace_path: Optional[Path]) -> Dict[str, Any]:
        def tool_generate_image(prompt: str, width: int = 1024, height: int = 1024, file_name: str = "") -> dict:
            """
            Generate an image from a text prompt using the Text-to-Image binding.

            Args:
                prompt (str): Detailed English prompt describing the image to generate.
                width (int, optional): Image width in pixels. Defaults to 1024.
                height (int, optional): Image height in pixels. Defaults to 1024.
                file_name (str, optional): Output filename (without extension). Auto-generated if empty.
            """
            try:
                img_bytes = tti_binding.generate_image(prompt=prompt, width=width, height=height)
                if not img_bytes:
                    return {"success": False, "error": "Image generation returned no data."}

                fname = file_name or f"generated_image_{uuid.uuid4().hex[:6]}"
                if not fname.endswith(".png"):
                    fname += ".png"

                save_path = Path(fname)
                if workspace_path:
                    save_path = workspace_path / fname
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(img_bytes)

                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                return {
                    "success": True,
                    "output": f"Image generated and saved as '{fname}'.",
                    "image_filename": fname,
                    "image_b64": img_b64,
                    "prompt_injection": f"\n\n✅ **Image Generated:** `{fname}`\nReference it in your response."
                }
            except Exception as e:
                return {"success": False, "error": f"Image generation failed: {e}"}

        return {
            "name": "tool_generate_image",
            "description": "Generate an image from a text prompt using the Text-to-Image (TTI) binding. The image is saved to the workspace.",
            "parameters": [
                {"name": "prompt", "type": "str", "description": "Detailed English prompt describing the image."},
                {"name": "width", "type": "int", "description": "Image width in pixels (default 1024).", "optional": True},
                {"name": "height", "type": "int", "description": "Image height in pixels (default 1024).", "optional": True},
                {"name": "file_name", "type": "str", "description": "Output filename without extension (auto-generated if empty).", "optional": True},
            ],
            "callable": tool_generate_image,
        }

    @staticmethod
    def _make_tti_edit_tool(tti_binding, workspace_path: Optional[Path]) -> Dict[str, Any]:
        def tool_edit_image(prompt: str, image_file_name: str = "") -> dict:
            """
            Edit an existing image in the workspace using a text prompt.

            Args:
                prompt (str): Detailed English prompt describing the edits to apply.
                image_file_name (str): Filename of the image to edit (in the workspace).
            """
            try:
                # Load source image
                source_b64 = None
                if image_file_name:
                    img_path = Path(image_file_name)
                    if not img_path.exists() and workspace_path:
                        img_path = workspace_path / image_file_name
                    if img_path.exists():
                        raw = img_path.read_bytes()
                        source_b64 = base64.b64encode(raw).decode('utf-8')

                if not source_b64:
                    return {"success": False, "error": f"Source image '{image_file_name}' not found in workspace."}

                img_bytes = tti_binding.edit_image(image=source_b64, prompt=prompt)
                if not img_bytes:
                    return {"success": False, "error": "Image edit returned no data."}

                fname = f"edited_image_{uuid.uuid4().hex[:6]}.png"
                save_path = Path(fname)
                if workspace_path:
                    save_path = workspace_path / fname
                save_path.write_bytes(img_bytes)

                return {
                    "success": True,
                    "output": f"Image edited and saved as '{fname}'.",
                    "image_filename": fname,
                }
            except Exception as e:
                return {"success": False, "error": f"Image edit failed: {e}"}

        return {
            "name": "tool_edit_image",
            "description": "Edit an existing image in the workspace using a text prompt via the TTI binding.",
            "parameters": [
                {"name": "prompt", "type": "str", "description": "Detailed prompt describing the edits."},
                {"name": "image_file_name", "type": "str", "description": "Filename of the source image in the workspace."},
            ],
            "callable": tool_edit_image,
        }

    @staticmethod
    def _make_tts_tool(tts_binding, workspace_path: Optional[Path]) -> Dict[str, Any]:
        def tool_text_to_speech(text: str, voice: str = "", language: str = "en", file_name: str = "") -> dict:
            """
            Convert text to speech audio using the TTS binding.

            Args:
                text (str): The text to synthesize into speech.
                voice (str, optional): Voice name to use (binding-specific).
                language (str, optional): Language code (e.g., 'en', 'fr'). Defaults to 'en'.
                file_name (str, optional): Output filename (without extension). Auto-generated if empty.
            """
            try:
                audio_bytes = tts_binding.generate_audio(text=text, voice=voice or None, language=language)
                if not audio_bytes:
                    return {"success": False, "error": "TTS returned no audio data."}

                fname = file_name or f"speech_{uuid.uuid4().hex[:6]}"
                if not fname.endswith(".wav"):
                    fname += ".wav"

                save_path = Path(fname)
                if workspace_path:
                    save_path = workspace_path / fname
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(audio_bytes)

                return {
                    "success": True,
                    "output": f"Audio generated and saved as '{fname}'.",
                    "audio_filename": fname,
                }
            except Exception as e:
                return {"success": False, "error": f"TTS failed: {e}"}

        return {
            "name": "tool_text_to_speech",
            "description": "Convert text to speech audio using the Text-to-Speech (TTS) binding. Audio is saved as a WAV file.",
            "parameters": [
                {"name": "text", "type": "str", "description": "The text to synthesize."},
                {"name": "voice", "type": "str", "description": "Voice name (binding-specific, optional).", "optional": True},
                {"name": "language", "type": "str", "description": "Language code (default 'en').", "optional": True},
                {"name": "file_name", "type": "str", "description": "Output filename without extension (auto-generated if empty).", "optional": True},
            ],
            "callable": tool_text_to_speech,
        }

    @staticmethod
    def _make_stt_tool(stt_binding, workspace_path: Optional[Path]) -> Dict[str, Any]:
        def tool_speech_to_text(audio_file_name: str) -> dict:
            """
            Transcribe speech from an audio file to text using the STT binding.

            Args:
                audio_file_name (str): Filename of the audio file in the workspace.
            """
            try:
                audio_path = Path(audio_file_name)
                if not audio_path.exists() and workspace_path:
                    audio_path = workspace_path / audio_file_name
                if not audio_path.exists():
                    return {"success": False, "error": f"Audio file '{audio_file_name}' not found."}

                audio_bytes = audio_path.read_bytes()
                transcript = stt_binding.transcribe(audio=audio_bytes)
                return {
                    "success": True,
                    "output": f"Transcription: {transcript}",
                    "transcript": transcript,
                }
            except Exception as e:
                return {"success": False, "error": f"STT failed: {e}"}

        return {
            "name": "tool_speech_to_text",
            "description": "Transcribe speech from an audio file in the workspace to text using the STT binding.",
            "parameters": [
                {"name": "audio_file_name", "type": "str", "description": "Filename of the audio file in the workspace."},
            ],
            "callable": tool_speech_to_text,
        }

    @staticmethod
    def _make_ttm_tool(ttm_binding, workspace_path: Optional[Path]) -> Dict[str, Any]:
        def tool_generate_music(prompt: str, duration: int = 10, file_name: str = "") -> dict:
            """
            Generate music from a text prompt using the TTM binding.

            Args:
                prompt (str): Description of the music to generate.
                duration (int, optional): Duration in seconds. Defaults to 10.
                file_name (str, optional): Output filename (without extension). Auto-generated if empty.
            """
            try:
                audio_bytes = ttm_binding.generate_music(prompt=prompt, duration=duration)
                if not audio_bytes:
                    return {"success": False, "error": "TTM returned no audio data."}

                fname = file_name or f"music_{uuid.uuid4().hex[:6]}"
                if not fname.endswith(".wav"):
                    fname += ".wav"

                save_path = Path(fname)
                if workspace_path:
                    save_path = workspace_path / fname
                save_path.write_bytes(audio_bytes)

                return {
                    "success": True,
                    "output": f"Music generated and saved as '{fname}'.",
                    "audio_filename": fname,
                }
            except Exception as e:
                return {"success": False, "error": f"TTM failed: {e}"}

        return {
            "name": "tool_generate_music",
            "description": "Generate music from a text prompt using the Text-to-Music (TTM) binding.",
            "parameters": [
                {"name": "prompt", "type": "str", "description": "Description of the music to generate."},
                {"name": "duration", "type": "int", "description": "Duration in seconds (default 10).", "optional": True},
                {"name": "file_name", "type": "str", "description": "Output filename without extension.", "optional": True},
            ],
            "callable": tool_generate_music,
        }

    @staticmethod
    def _make_ttv_tool(ttv_binding, workspace_path: Optional[Path]) -> Dict[str, Any]:
        def tool_generate_video(prompt: str, duration: int = 5, file_name: str = "") -> dict:
            """
            Generate a video from a text prompt using the TTV binding.

            Args:
                prompt (str): Description of the video to generate.
                duration (int, optional): Duration in seconds. Defaults to 5.
                file_name (str, optional): Output filename (without extension). Auto-generated if empty.
            """
            try:
                video_bytes = ttv_binding.generate_video(prompt=prompt, duration=duration)
                if not video_bytes:
                    return {"success": False, "error": "TTV returned no video data."}

                fname = file_name or f"video_{uuid.uuid4().hex[:6]}"
                if not fname.endswith(".mp4"):
                    fname += ".mp4"

                save_path = Path(fname)
                if workspace_path:
                    save_path = workspace_path / fname
                save_path.write_bytes(video_bytes)

                return {
                    "success": True,
                    "output": f"Video generated and saved as '{fname}'.",
                    "video_filename": fname,
                }
            except Exception as e:
                return {"success": False, "error": f"TTV failed: {e}"}

        return {
            "name": "tool_generate_video",
            "description": "Generate a video from a text prompt using the Text-to-Video (TTV) binding.",
            "parameters": [
                {"name": "prompt", "type": "str", "description": "Description of the video to generate."},
                {"name": "duration", "type": "int", "description": "Duration in seconds (default 5).", "optional": True},
                {"name": "file_name", "type": "str", "description": "Output filename without extension.", "optional": True},
            ],
            "callable": tool_generate_video,
        }


# ===========================================================================
# ToolsManager — Load and execute lollms-format tool scripts (existing, kept)
# ===========================================================================

class ToolsManager:
    SYSTEM_TOOLS_DIR = Path("app/tools")
    USER_TOOLS_DIR = Path.home() / ".lollms_hub" / "tools"

    def __init__(self, extra_dirs: Optional[List[Union[str, Path]]] = None):
        self._extra_dirs: List[Path] = [Path(d) for d in (extra_dirs or [])]
        self._loaded_modules: Dict[str, ModuleType] = {}

    @classmethod
    def ensure_dirs(cls):
        cls.SYSTEM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        cls.USER_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    def _scan_paths(self) -> List[Path]:
        dirs = [self.SYSTEM_TOOLS_DIR, self.USER_TOOLS_DIR] + self._extra_dirs
        return [d for d in dirs if d.exists()]

    def list_available_files(self) -> List[Path]:
        files: set = set()
        for directory in self._scan_paths():
            for fp in directory.glob("*.py"):
                if fp.name == "__init__.py":
                    continue
                files.add(fp.resolve())
        return sorted(files, key=lambda p: p.name.lower())

    @staticmethod
    def parse_metadata(content: str) -> Dict[str, str]:
        meta = {"name": "Unnamed Tool Library", "description": "No description provided.", "icon": "🔧"}
        try:
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id == "TOOL_LIBRARY_NAME":
                                meta["name"] = ast.literal_eval(node.value)
                            elif target.id == "TOOL_LIBRARY_DESC":
                                meta["description"] = ast.literal_eval(node.value)
                            elif target.id == "TOOL_LIBRARY_ICON":
                                meta["icon"] = ast.literal_eval(node.value)
        except Exception:
            pass
        return meta

    @staticmethod
    def get_tool_definitions(content: str) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        titles: Dict[str, str] = {}
        try:
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "TOOL_TITLES":
                            titles = ast.literal_eval(node.value)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("tool_"):
                    docstring = ast.get_docstring(node) or "No description provided."
                    params: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
                    arg_pattern = re.compile(
                        r'^\s*-\s+([\w_]+)\s*\(([\w_]+)(?:,\s*optional)?\):\s*(.*)',
                        re.MULTILINE | re.IGNORECASE,
                    )
                    for m in arg_pattern.finditer(docstring):
                        name, p_type, desc = m.groups()
                        p_type_map = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object", "list": "array"}
                        params["properties"][name] = {"type": p_type_map.get(p_type.lower(), "string"), "description": desc.strip()}
                        if "optional" not in m.group(0).lower():
                            params["required"].append(name)
                    if not params["properties"]:
                        has_args = any((isinstance(arg, ast.arg) and arg.arg == "args") for arg in node.args.args)
                        if has_args:
                            params["properties"]["args"] = {"type": "object", "description": "Arguments for the tool"}
                    tools.append({"type": "function", "pretty_name": titles.get(node.name), "function": {"name": node.name, "description": docstring.split('\n\n')[0].strip(), "parameters": params}})
        except Exception:
            pass
        return tools

    def load_file(self, file_path: Union[str, Path]) -> ModuleType:
        fp = Path(file_path).resolve()
        key = str(fp)
        if key in self._loaded_modules:
            return self._loaded_modules[key]
        content = fp.read_text(encoding="utf-8")
        module_name = f"lollms_tools_{fp.stem}_{uuid.uuid4().hex[:8]}"
        module = ModuleType(module_name)
        module.__file__ = str(fp)
        exec(compile(content, str(fp), "exec"), module.__dict__)
        if hasattr(module, "init_tools_library"):
            try:
                module.init_tools_library()
            except Exception as e:
                ASCIIColors.warning(f"Tool init failed for {fp.name}: {e}")
        self._loaded_modules[key] = module
        return module

    def get_callable_tools(self, file_path: Union[str, Path]) -> Dict[str, Callable]:
        module = self.load_file(file_path)
        return {name: getattr(module, name) for name in dir(module) if name.startswith("tool_") and callable(getattr(module, name))}

    def execute_tool(self, file_path: Union[str, Path], tool_name: str, args: Dict[str, Any]) -> Any:
        callables = self.get_callable_tools(file_path)
        if tool_name not in callables:
            raise ValueError(f"Tool '{tool_name}' not found in {file_path}")
        return callables[tool_name](args)

    def resolve_tool_file(self, tool_name: str) -> Optional[Path]:
        for fp in self.list_available_files():
            defs = self.get_tool_definitions(fp.read_text(encoding="utf-8"))
            for d in defs:
                if d["function"]["name"] == tool_name:
                    return fp
        return None

    def build_tool_specs(self, sources: List[Union[str, Path, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for src in sources:
            if isinstance(src, dict):
                specs.append(src)
                continue
            fp = Path(src)
            if not fp.exists():
                raise FileNotFoundError(f"Tool file not found: {fp}")
            content = fp.read_text(encoding="utf-8")
            file_specs = self.get_tool_definitions(content)
            for s in file_specs:
                s["_source_file"] = str(fp.resolve())
            specs.extend(file_specs)
        return specs

    def build_inline_tools_dict(self, sources: List[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        tools_dict: Dict[str, Dict[str, Any]] = {}
        for src in sources:
            if isinstance(src, dict):
                name = src.get("name", src.get("function", {}).get("name", "unknown"))
                tools_dict[name] = src
                continue
            fp = Path(src)
            if not fp.exists():
                raise FileNotFoundError(f"Tool file not found: {fp}")
            module = self.load_file(fp)
            callables = self.get_callable_tools(fp)
            for tool_name, fn in callables.items():
                doc = (fn.__doc__ or "").strip()
                params: List[Dict[str, Any]] = []
                arg_pattern = re.compile(r'^\s*-\s+([\w_]+)\s*\(([\w_]+)(?:,\s*optional)?\):\s*(.*)', re.MULTILINE | re.IGNORECASE)
                for m in arg_pattern.finditer(doc):
                    pname, ptype, pdesc = m.groups()
                    is_optional = "optional" in m.group(0).lower()
                    p_entry: Dict[str, Any] = {"name": pname, "type": ptype.lower(), "description": pdesc.strip()}
                    if is_optional:
                        p_entry["optional"] = True
                    params.append(p_entry)
                tools_dict[tool_name] = {"name": tool_name, "callable": fn, "parameters": params, "description": doc.split('\n\n')[0].strip() if doc else f"Execute {tool_name}", "_source_file": str(fp.resolve())}
        return tools_dict


# ===========================================================================
# Helper: Tool output sanitization
# ===========================================================================

_BASE64_RE = re.compile(r'^[A-Za-z0-9+/=\s]{500,}$')
_BINARY_BLOB_KEYS = {"plot_b64", "image_b64", "audio_b64", "video_b64", "file_b64", "screenshot_b64", "pdf_b64", "thumbnail_b64", "base64", "binary", "raw_image", "image_data", "raw_data"}
_MAX_TOOL_RESULT_CHARS = 4000


def _is_large_base64(v: str) -> bool:
    sample = v.replace("\n", "").replace("\r", "").replace(" ", "")
    if len(sample) < 500:
        return False
    return bool(_BASE64_RE.match(sample[:1000]))


def _sanitize_tool_result(tool_res: Any, max_chars: int = _MAX_TOOL_RESULT_CHARS) -> str:
    """Converts an arbitrary tool execution result into a clean, LLM-friendly text representation."""

    def _find_prompt_injection(obj: Any, depth: int = 0) -> Optional[str]:
        if depth > 4:
            return None
        if isinstance(obj, dict):
            pinj = obj.get("prompt_injection")
            if isinstance(pinj, str) and pinj.strip():
                return pinj.strip()
            for v in obj.values():
                hit = _find_prompt_injection(v, depth + 1)
                if hit:
                    return hit
        elif isinstance(obj, list):
            for v in obj:
                hit = _find_prompt_injection(v, depth + 1)
                if hit:
                    return hit
        return None

    def _walk(obj: Any, depth: int = 0) -> Any:
        if depth > 6:
            return "[truncated: depth limit]"
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            if _is_large_base64(obj):
                approx_kb = len(obj) * 3 / 4 / 1024
                return f"[base64 blob stripped: {approx_kb:.1f}KB]"
            if len(obj) > max_chars:
                return obj[:max_chars] + f"\n... [truncated, {len(obj) - max_chars} more chars]"
            return obj
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in _BINARY_BLOB_KEYS:
                    if isinstance(v, str) and v:
                        approx_kb = len(v) * 3 / 4 / 1024
                        cleaned[k] = f"[base64 blob stripped: {approx_kb:.1f}KB]"
                    else:
                        cleaned[k] = None
                else:
                    cleaned[k] = _walk(v, depth + 1)
            return cleaned
        if isinstance(obj, (list, tuple)):
            walked = [_walk(v, depth + 1) for v in obj[:50]]
            if len(obj) > 50:
                walked.append(f"... [truncated, {len(obj) - 50} more items]")
            return walked
        return str(obj)

    if isinstance(tool_res, dict) and tool_res.get("success") is False:
        error_msg = tool_res.get("error", "Unknown error")
        inner = tool_res.get("output")
        if isinstance(inner, dict):
            error_msg = inner.get("error", error_msg)
        return f"⚠ Tool Failed\nError: {error_msg}"

    pinj = _find_prompt_injection(tool_res)
    if pinj:
        success = True
        inner = tool_res.get("output", tool_res) if isinstance(tool_res, dict) else tool_res
        if isinstance(inner, dict):
            success = inner.get("success", True)
        if isinstance(tool_res, dict) and tool_res.get("success") is False:
            success = False
        success_status = "✓ Success" if success else "⚠ Tool Failed"
        error_msg = ""
        if not success and isinstance(tool_res, dict):
            error_msg = tool_res.get("error", "")
        if error_msg:
            return f"{success_status}\nError: {error_msg}\n{pinj}"
        return f"{success_status}\n{pinj}"

    unwrapped = tool_res
    if isinstance(tool_res, dict):
        if "output" in tool_res:
            unwrapped = tool_res["output"]
            if isinstance(unwrapped, dict):
                for key in ("content", "text", "result", "data", "page_content", "summary"):
                    if key in unwrapped:
                        unwrapped = unwrapped[key]
                        break
        elif "content" in tool_res:
            unwrapped = tool_res["content"]
        elif "result" in tool_res:
            unwrapped = tool_res["result"]
        elif "data" in tool_res:
            unwrapped = tool_res["data"]

    if unwrapped is None:
        return "Tool executed successfully but returned no output content."

    def _replace_none(obj):
        if obj is None:
            return "[No output returned by tool]"
        if isinstance(obj, dict):
            return {k: _replace_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_replace_none(v) for v in obj]
        return obj

    sanitized = _walk(_replace_none(unwrapped))
    if isinstance(sanitized, str):
        if len(sanitized) > max_chars:
            return sanitized[:max_chars] + f"\n... [truncated, {len(sanitized) - max_chars} more chars]"
        return sanitized

    try:
        text = json.dumps(sanitized, indent=2, default=str, ensure_ascii=False)
    except Exception:
        text = str(sanitized)

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more chars]"
    return text


# ===========================================================================
# Helper: Workspace context builder
# ===========================================================================

_IGNORED_WS_DIRS = {"__pycache__", ".venv", "venv", ".git", ".idea", ".vscode", "node_modules", ".lollms", "build", "dist", ".next", "env", ".env"}
_IGNORED_WS_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib"}
_TEXT_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss", ".sql", ".md", ".txt", ".json", ".yaml", ".yml", ".xml", ".csv", ".log", ".toml", ".ini", ".cfg", ".sh", ".bash", ".ps1", ".bat", ".rdf", ".ttl", ".rs", ".go", ".rb", ".php", ".java", ".kt", ".swift", ".c", ".cpp", ".h", ".hpp"}
_BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp", ".zip", ".tar", ".gz", ".pdf", ".docx", ".mp3", ".wav", ".mp4", ".avi", ".mov"}


def _build_workspace_context(workspace_path: Path, max_file_size: int = 12000, max_total_chars: int = 30000) -> str:
    """Lists all files in the workspace and includes content for small text files."""
    if not workspace_path or not workspace_path.exists():
        return ""

    lines = ["=== WORKSPACE FILES ==="]
    file_entries = []
    content_entries = []
    total_content_chars = 0

    for f_path in sorted(workspace_path.rglob("*")):
        if not f_path.is_file():
            continue
        rel_parts = f_path.relative_to(workspace_path).parts
        if any(part in _IGNORED_WS_DIRS for part in rel_parts):
            continue
        if any(part.startswith(".") for part in rel_parts[:-1]):
            continue
        file_name = f_path.name
        file_ext = f_path.suffix.lower()
        if file_ext in _IGNORED_WS_EXTS or file_name.startswith("."):
            continue

        size = f_path.stat().st_size
        rel_path = f_path.relative_to(workspace_path)
        size_str = f"{size:,} bytes"

        if file_ext in _TEXT_EXTS and size <= max_file_size:
            file_entries.append(f"- {rel_path} ({size_str}, text)")
            if total_content_chars < max_total_chars:
                try:
                    content = f_path.read_text(encoding="utf-8", errors="ignore")
                    remaining_budget = max_total_chars - total_content_chars
                    if len(content) > remaining_budget:
                        content = content[:remaining_budget] + f"\n... [truncated, {len(content) - remaining_budget} more chars]"
                    content_entries.append(f"\n--- {rel_path} ---\n```{file_ext.lstrip('.')}\n{content}\n```\n")
                    total_content_chars += len(content)
                except Exception:
                    file_entries.append(f"- {rel_path} ({size_str}, unreadable)")
        elif file_ext in _BINARY_EXTS:
            file_entries.append(f"- {rel_path} ({size_str}, binary)")
        else:
            file_entries.append(f"- {rel_path} ({size_str})")

    if not file_entries:
        lines.append("(Workspace is empty)")
        lines.append("=== END WORKSPACE FILES ===")
        return "\n".join(lines)

    lines.append("Files in workspace:")
    lines.extend(file_entries)

    if content_entries:
        lines.append("\nFile Contents:")
        lines.extend(content_entries)

    lines.append("=== END WORKSPACE FILES ===")
    return "\n".join(lines)


# ===========================================================================
# Helper: Message normalization for OpenAI API compliance
# ===========================================================================

def _normalize_messages(messages: List[Dict]) -> List[Dict]:
    """Ensure proper user/assistant alternation for OpenAI API."""
    if not messages:
        return messages

    normalized = []
    system_content_parts = []
    non_system_messages = []

    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                system_content_parts.append("\n".join(text_parts))
            else:
                system_content_parts.append(str(content))
        else:
            non_system_messages.append(msg)

    if system_content_parts:
        fused = "\n\n".join(p for p in system_content_parts if p.strip())
        if fused.strip():
            normalized.append({"role": "system", "content": fused})

    if non_system_messages:
        current_role = None
        current_content = []
        for msg in non_system_messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if not content and not msg.get("images"):
                continue
            if role == current_role:
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            current_content.append(item.get("text", ""))
                else:
                    current_content.append(str(content))
            else:
                if current_role is not None and current_content:
                    merged = "\n\n".join(c for c in current_content if c.strip())
                    if merged.strip():
                        normalized.append({"role": current_role, "content": merged})
                current_role = role
                current_content = []
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            current_content.append(item.get("text", ""))
                else:
                    current_content.append(str(content))
        if current_role is not None and current_content:
            merged = "\n\n".join(c for c in current_content if c.strip())
            if merged.strip():
                normalized.append({"role": current_role, "content": merged})

    non_sys_start = 0
    for i, msg in enumerate(normalized):
        if msg.get("role") != "system":
            non_sys_start = i
            break
    if non_sys_start < len(normalized):
        first_non_sys = normalized[non_sys_start]
        if first_non_sys.get("role") == "assistant":
            normalized.insert(non_sys_start, {"role": "user", "content": "Continue."})

    return normalized


# ===========================================================================
# Built-in workspace tools (always available when workspace is configured)
# ===========================================================================

def _tool_write_file(file_name: str, content: str) -> dict:
    """
    Write text content to a file in the workspace.

    Args:
        file_name (str): Name or relative path of the file to write.
        content (str): The text content to write to the file.
    """
    try:
        path = Path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"success": True, "output": f"File '{file_name}' written successfully ({len(content)} bytes)."}
    except Exception as e:
        return {"success": False, "error": f"Failed to write file '{file_name}': {e}"}


def _tool_read_file(file_name: str) -> dict:
    """
    Read the content of a text file from the workspace.

    Args:
        file_name (str): Name or relative path of the file to read.
    """
    try:
        path = Path(file_name)
        if not path.exists():
            return {"success": False, "error": f"File '{file_name}' not found in workspace."}
        content = path.read_text(encoding="utf-8", errors="ignore")
        return {"success": True, "output": content}
    except Exception as e:
        return {"success": False, "error": f"Failed to read file '{file_name}': {e}"}


def _tool_list_files() -> dict:
    """
    List all files in the workspace directory recursively.
    """
    try:
        files = []
        for f in Path(".").rglob("*"):
            if f.is_file():
                rel = f.relative_to(Path("."))
                if not any(part in _IGNORED_WS_DIRS for part in rel.parts):
                    size = f.stat().st_size
                    files.append(f"{rel} ({size:,} bytes)")
        if not files:
            return {"success": True, "output": "Workspace is empty."}
        return {"success": True, "output": "\n".join(sorted(files))}
    except Exception as e:
        return {"success": False, "error": f"Failed to list files: {e}"}


def _get_builtin_workspace_tools() -> Dict[str, Dict[str, Any]]:
    """Returns the built-in workspace tool specs in the active_tools format."""
    return {
        "tool_write_file": {
            "name": "tool_write_file",
            "description": "Write text content to a file in the workspace. Use this to create or overwrite files.",
            "parameters": [
                {"name": "file_name", "type": "str", "description": "Name or relative path of the file to write."},
                {"name": "content", "type": "str", "description": "The text content to write to the file."}
            ],
            "callable": _tool_write_file
        },
        "tool_read_file": {
            "name": "tool_read_file",
            "description": "Read the content of a text file from the workspace.",
            "parameters": [
                {"name": "file_name", "type": "str", "description": "Name or relative path of the file to read."}
            ],
            "callable": _tool_read_file
        },
        "tool_list_files": {
            "name": "tool_list_files",
            "description": "List all files in the workspace directory recursively with their sizes.",
            "parameters": [],
            "callable": _tool_list_files
        }
    }


# ===========================================================================
# _AgentStreamState — Transactional stream parser for <tool> and <done/> tags
# ===========================================================================

class _AgentStreamState:
    """
    A simplified, non-blocking transactional stream parser.
    Handles: <tool> tags, <done/> termination, code fence protection, <processing> anti-mimicry.
    """

    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.content = ""

        self.tool_trigger = False
        self.tool_json_data = ""
        self._done_detected = False
        self._action_dispatched = False

        self._is_accumulating_tool = False
        self._tool_buffer = ""
        self._pending_buffer = ""

        self._in_code_fence = False
        self._code_fence_buffer = ""
        self._code_fence_hold_buffer = ""
        self._in_inline_code = False

    def _cb(self, text: str, msg_type=None, meta: Optional[Dict] = None):
        if self.callback is None:
            return
        try:
            mt = msg_type if msg_type is not None else MSG_TYPE.MSG_TYPE_CHUNK
            self.callback(text, mt, meta or {})
        except Exception:
            pass

    def feed(self, chunk: str) -> bool:
        if not isinstance(chunk, str) or not chunk:
            return True

        if self._action_dispatched:
            self._pending_buffer += chunk
            return True

        self._pending_buffer += chunk

        # <done/> tag detection
        if not self._is_accumulating_tool and not self._in_code_fence and not self._in_inline_code:
            done_match = re.search(r'(?m)^\s*<done\s*/?>', self._pending_buffer, re.IGNORECASE)
            if done_match:
                ASCIIColors.info("[AgentStreamState] <done/> tag detected. Halting generation.")
                self._done_detected = True
                self._pending_buffer = re.sub(r'(?m)^\s*<done\s*/?>', '', self._pending_buffer, flags=re.IGNORECASE)
                return False

        # Anti-mimicry: prevent LLM from generating <processing> blocks
        if not self._is_accumulating_tool and not self._in_code_fence and not self._in_inline_code:
            proc_match = re.search(r'(?m)^\s*<processing', self._pending_buffer, re.IGNORECASE)
            if proc_match:
                ASCIIColors.warning("[AgentStreamState] LLM attempted to generate a <processing> block. Halting.")
                self._pending_buffer = re.sub(r'(?m)^\s*<processing[^>]*>', '', self._pending_buffer, flags=re.IGNORECASE)
                return False

        # Code fence protection (```)
        if not self._is_accumulating_tool:
            if "```" in self._pending_buffer:
                self._code_fence_buffer += self._pending_buffer
                self._pending_buffer = ""

                while "```" in self._code_fence_buffer:
                    idx = self._code_fence_buffer.find("```")
                    before = self._code_fence_buffer[:idx]
                    self._code_fence_buffer = self._code_fence_buffer[idx + 3:]

                    if not self._in_code_fence:
                        self._in_code_fence = True
                        self.content += before + "```"
                        self._cb(before + "```")
                    else:
                        self._in_code_fence = False
                        if self._code_fence_hold_buffer:
                            self.content += self._code_fence_hold_buffer
                            self._cb(self._code_fence_hold_buffer)
                            self._code_fence_hold_buffer = ""
                        if before:
                            self.content += before
                            self._cb(before)
                        self.content += "```"
                        self._cb("```")

                if self._in_code_fence:
                    self._code_fence_hold_buffer += self._code_fence_buffer
                    self._code_fence_buffer = ""
                    return True
                else:
                    self._pending_buffer = self._code_fence_buffer
                    self._code_fence_buffer = ""
            elif self._in_code_fence:
                self._code_fence_hold_buffer += self._pending_buffer
                self._pending_buffer = ""
                return True

        # Inline code protection (single backtick)
        if not self._is_accumulating_tool and not self._in_code_fence:
            if "`" in self._pending_buffer:
                if self._in_inline_code:
                    idx = self._pending_buffer.find("`")
                    if idx != -1:
                        self._in_inline_code = False
                        inline_content = self._pending_buffer[:idx]
                        self.content += inline_content + "`"
                        self._cb(inline_content + "`")
                        self._pending_buffer = self._pending_buffer[idx + 1:]
                    else:
                        newline_idx = self._pending_buffer.find("\n")
                        if newline_idx != -1:
                            self._in_inline_code = False
                            self.content += self._pending_buffer
                            self._cb(self._pending_buffer)
                            self._pending_buffer = ""
                        else:
                            self.content += self._pending_buffer
                            self._cb(self._pending_buffer)
                            self._pending_buffer = ""
                        return True
                else:
                    idx = self._pending_buffer.find("`")
                    before = self._pending_buffer[:idx]
                    remainder = self._pending_buffer[idx + 1:]
                    closing_idx = remainder.find("`")
                    if closing_idx != -1:
                        inline_content = remainder[:closing_idx]
                        self.content += before + "`" + inline_content + "`"
                        self._cb(before + "`" + inline_content + "`")
                        self._pending_buffer = remainder[closing_idx + 1:]
                    else:
                        newline_idx = remainder.find("\n")
                        if newline_idx != -1:
                            self.content += before + "`" + remainder
                            self._cb(before + "`" + remainder)
                            self._pending_buffer = ""
                        else:
                            self._in_inline_code = True
                            self.content += before + "`"
                            self._cb(before + "`")
                            self._pending_buffer = remainder
                        return True
            elif self._in_inline_code:
                idx = self._pending_buffer.find("`")
                if idx != -1:
                    self._in_inline_code = False
                    inline_content = self._pending_buffer[:idx]
                    self.content += inline_content + "`"
                    self._cb(inline_content + "`")
                    self._pending_buffer = self._pending_buffer[idx + 1:]
                else:
                    newline_idx = self._pending_buffer.find("\n")
                    if newline_idx != -1:
                        self._in_inline_code = False
                        self.content += self._pending_buffer
                        self._cb(self._pending_buffer)
                        self._pending_buffer = ""
                    else:
                        self.content += self._pending_buffer
                        self._cb(self._pending_buffer)
                        self._pending_buffer = ""
                        return True

        # Tool accumulation
        if self._is_accumulating_tool:
            self._tool_buffer += self._pending_buffer
            self._pending_buffer = ""
            if self._try_complete_tool():
                return False
            return True

        # Tool tag detection
        if not self._in_code_fence and not self._in_inline_code:
            tool_match = re.search(r'(?m)^\s*(?!`)(?!.*\|)<tool>', self._pending_buffer, re.IGNORECASE)
            if tool_match:
                tag_start_idx = tool_match.start()
                text_before = self._pending_buffer[:tag_start_idx]
                if text_before:
                    self.content += text_before
                    self._cb(text_before)

                self._is_accumulating_tool = True
                self._tool_buffer = self._pending_buffer[tag_start_idx:]
                self._pending_buffer = ""
                # CRITICAL FIX: Check if closing </tool> is already in the buffer (single-chunk dispatch)
                if self._try_complete_tool():
                    return False
                return True

        # Partial tag detection
        def _ends_with_partial_tag(buffer: str) -> int:
            tags_to_check = ["<tool", "<done"]
            for tag in tags_to_check:
                for i in range(1, len(tag)):
                    if buffer.endswith(tag[:i]):
                        start_idx = len(buffer) - i
                        j = start_idx - 1
                        while j >= 0 and buffer[j] != '\n':
                            if not buffer[j].isspace():
                                return -1
                            j -= 1
                        return start_idx
            return -1

        partial_idx = _ends_with_partial_tag(self._pending_buffer)
        if partial_idx != -1:
            text_before = self._pending_buffer[:partial_idx]
            if text_before:
                self.content += text_before
                self._cb(text_before)
            self._pending_buffer = self._pending_buffer[partial_idx:]
            return True

        self.content += self._pending_buffer
        self._cb(self._pending_buffer)
        self._pending_buffer = ""
        return True

    def _try_complete_tool(self) -> bool:
        """
        Checks if </tool> is present in _tool_buffer and dispatches the tool call if found.
        Returns True if the tool was dispatched, False otherwise.
        Handles both multi-chunk accumulation and single-chunk complete tool calls.
        """
        close_match = re.search(r'</tool>\s*', self._tool_buffer, re.IGNORECASE)
        if not close_match:
            return False

        end_idx = close_match.start()
        end_len = len(close_match.group(0))

        full_tool_call = self._tool_buffer[:end_idx + end_len]
        json_body = re.sub(r'^<tool>', '', full_tool_call, flags=re.IGNORECASE)
        json_body = re.sub(r'</tool>\s*$', '', json_body, flags=re.IGNORECASE).strip()

        self._is_accumulating_tool = False
        remaining = self._tool_buffer[end_idx + end_len:]
        self._tool_buffer = ""
        if remaining:
            self._pending_buffer = remaining + self._pending_buffer

        # Parse and normalize JSON (with repair for incomplete JSON)
        try:
            raw_data = json.loads(json_body)
        except json.JSONDecodeError:
            # Attempt to repair incomplete JSON by closing open braces/brackets
            repaired = json_body
            while repaired.count('{') > repaired.count('}'):
                repaired += '}'
            while repaired.count('[') > repaired.count(']'):
                repaired += ']'
            try:
                raw_data = json.loads(repaired)
                json_body = repaired
            except json.JSONDecodeError:
                raw_data = None

        if isinstance(raw_data, dict):
            tool_name = raw_data.get("name", "")
            if "parameters" in raw_data and isinstance(raw_data["parameters"], dict):
                self.tool_json_data = json_body
            else:
                params = {k: v for k, v in raw_data.items() if k != "name"}
                normalized = {"name": tool_name, "parameters": params}
                self.tool_json_data = json.dumps(normalized)
        else:
            self.tool_json_data = json_body

        self.tool_trigger = True
        self._action_dispatched = True

        # Emit processing block for UI feedback
        try:
            parsed = json.loads(self.tool_json_data)
            ui_tool_name = parsed.get("name", "unknown") if isinstance(parsed, dict) else "unknown"
        except Exception:
            ui_tool_name = "unknown"

        import html
        try:
            parsed_for_ui = json.loads(self.tool_json_data)
            ui_params = parsed_for_ui.get("parameters", {}) if isinstance(parsed_for_ui, dict) else {}
        except Exception:
            ui_params = {}
        escaped_params = html.escape(json.dumps(ui_params, default=str))

        tool_open_tag = f'\n<processing type="tool" title="Tool Execution: {ui_tool_name}" params="{escaped_params}">\n'
        self.content += tool_open_tag
        self._cb(tool_open_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

        status_line = f"* Calling tool '{ui_tool_name}'...\n"
        self.content += status_line
        self._cb(status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

        return True

    def flush_remaining_buffer(self):
        """Flushes any safe text remaining at the end of generation."""
        if self._in_code_fence:
            self._in_code_fence = False
            hold = self._code_fence_hold_buffer
            self._code_fence_hold_buffer = ""
            if hold:
                self.feed(hold)

        if self._is_accumulating_tool:
            self._tool_buffer += self._pending_buffer
            self._pending_buffer = ""
            # Try to complete the tool call normally first
            if not self._try_complete_tool():
                # No closing </tool> found — synthesize dispatch from what we have
                full_tool_call = self._tool_buffer
                json_body = re.sub(r'^<tool>', '', full_tool_call, flags=re.IGNORECASE)
                json_body = re.sub(r'</tool>\s*$', '', json_body, flags=re.IGNORECASE).strip()

                self._is_accumulating_tool = False
                self._tool_buffer = ""

                # Parse and normalize JSON (with repair for incomplete JSON)
                try:
                    raw_data = json.loads(json_body)
                except json.JSONDecodeError:
                    # Attempt to repair incomplete JSON by closing open braces/brackets
                    repaired = json_body
                    while repaired.count('{') > repaired.count('}'):
                        repaired += '}'
                    while repaired.count('[') > repaired.count(']'):
                        repaired += ']'
                    try:
                        raw_data = json.loads(repaired)
                        json_body = repaired
                    except json.JSONDecodeError:
                        raw_data = None

                if isinstance(raw_data, dict):
                    tool_name = raw_data.get("name", "")
                    if "parameters" not in raw_data or not isinstance(raw_data.get("parameters"), dict):
                        params = {k: v for k, v in raw_data.items() if k != "name"}
                        normalized = {"name": tool_name, "parameters": params}
                        self.tool_json_data = json.dumps(normalized)
                    else:
                        self.tool_json_data = json_body
                else:
                    self.tool_json_data = json_body

                self.tool_trigger = True
                self._action_dispatched = True

                try:
                    parsed = json.loads(self.tool_json_data)
                    ui_tool_name = parsed.get("name", "unknown") if isinstance(parsed, dict) else "unknown"
                except Exception:
                    ui_tool_name = "unknown"

                tool_open_tag = f'\n<processing type="tool" title="Tool Execution: {ui_tool_name}">\n'
                self.content += tool_open_tag
                self._cb(tool_open_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                status_line = f"* Calling tool '{ui_tool_name}'...\n"
                self.content += status_line
                self._cb(status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            return

        if self._pending_buffer:
            self.content += self._pending_buffer
            self._cb(self._pending_buffer)
            self._pending_buffer = ""

    def was_done_detected(self) -> bool:
        return self._done_detected

    def was_action_dispatched(self) -> bool:
        return self._action_dispatched

    def get_tool_call_json(self) -> Optional[str]:
        return self.tool_json_data if self.tool_trigger else None

    def get_clean_text(self) -> str:
        return self.content


# ===========================================================================
# Agent — High-Grade Agentic System
# ===========================================================================

@dataclass
class Agent:
    """
    A high-grade agentic system that can work on its own, use tools, spawn sub-agents,
    enhance itself via memory and updatable skills, switch models on the fly, and
    leverage lollms_client's multimodal bindings.

    Key Features:
    - Multi-step agentic reasoning with <done/> termination
    - LCP tool discovery and execution with CWD switching
    - Built-in workspace tools (write_file, read_file, list_files)
    - Skills subsystem (SKILL.md files, always-visible or loadable)
    - Sub-agent delegation with depth/count limits
    - On-the-fly model switching via lollms_client's mount capabilities
    - Binding integration (TTI/TTS/STT/TTM/TTV exposed as tools)
    - Capability flags (code execution, networking, etc.)
    - Virtual history for KV-cache alignment
    - FailureMemory loop prevention and Success-Loop detection
    - Streaming support with proper tag interception
    - Thread-safe cancellation protocol
    - Workspace file context injection
    - Memory enhancement (episodic memory, skill evolution)

    Parameters
    ----------
    lc : LollmsClient
        The LollmsClient instance to use for LLM generation and bindings.
    personality : LollmsPersonality
        The personality defining the agent's system prompt and behavior.
    name : str, optional
        Display name for the agent.
    role : str
        AgentRole constant (PROPOSER, CRITIC, DOMAIN_EXPERT, etc.).
    workspace_path : str or Path, optional
        Path to the agent's workspace directory. Files are physical here.
    capabilities : CapabilityFlags, optional
        Boolean flags controlling what the agent can do. Defaults to safe values.
    skills_manager : SkillsManager, optional
        External skills manager. If None, a default one is created.
    skills_dirs : list of str/Path, optional
        Directories containing SKILL.md files. Used if skills_manager is None.
    skills_files : list of str/Path, optional
        Explicit SKILL.md file paths. Used if skills_manager is None.
    tools : dict, optional
        Explicit tool specifications to always include.
    tool_files : list of str/Path, optional
        File paths to lollms-format tool scripts.
    model_params : dict
        Additional parameters for the LLM (temperature, top_p, etc.).
    max_tokens_per_turn : int
        Maximum tokens per generation turn.
    metadata : dict
        Arbitrary metadata for the agent.
    memory_manager : optional
        LollmsMemoryManager instance for persistent memory.
    handbag_path : str or Path, optional
        Path to a Handbag folder containing ALL agent resources (personalities,
        tools, skills, RAG, memory). When provided, the handbag's resources are
        used as DEFAULTS for personality, tool_files, skills_dirs, memory_manager,
        and workspace_path. Explicit constructor parameters always override
        handbag-provided values.
    _parent_depth : int
        Internal: recursion depth for sub-agent tracking.
    """

    lc:                  Any   # LollmsClient
    personality:         Optional[Any] = None  # LollmsPersonality (optional if handbag_path provides one)
    handbag_path:        Optional[Union[str, Path]] = None  # Path to a Handbag folder containing all agent resources
    name:                Optional[str] = None
    role:                str = AgentRole.FREEFORM
    workspace_path:      Optional[Union[str, Path]] = None
    capabilities:        CapabilityFlags = field(default_factory=CapabilityFlags)
    skills_manager:      Optional[SkillsManager] = None
    skills_dirs:         Optional[List[Union[str, Path]]] = None
    skills_files:        Optional[List[Union[str, Path]]] = None
    tools:               Optional[Dict] = None
    tool_files:          Optional[List[Union[str, Path]]] = None
    model_params:        Dict[str, Any] = field(default_factory=dict)
    max_tokens_per_turn: int = 4096
    metadata:            Dict[str, Any] = field(default_factory=dict)
    memory_manager:      Optional[Any] = None
    _parent_depth:       int = field(default=0, repr=False)

    _agent_id: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)
    _cancel_flag: bool = field(default=False, init=False)
    _failure_memory: Any = field(default=None, init=False)
    _conversation: List[Dict[str, str]] = field(default_factory=list, init=False)
    _resolved_workspace: Optional[Path] = field(default=None, init=False)
    _sub_agent_spawner: Optional[SubAgentSpawner] = field(default=None, init=False)
    _model_switcher: Optional[ModelSwitcher] = field(default=None, init=False)
    _handbag: Any = field(default=None, init=False)

    # ---------------------------------------------------------------- init

    def __post_init__(self):
        # ── Handbag Loading ──
        # A handbag is a folder containing all agent resources (personalities, tools,
        # skills, RAG, memory). If handbag_path is provided, its values are used as
        # DEFAULTS. Explicit constructor parameters always take precedence.
        if self.handbag_path:
            from lollms_client.lollms_agent.handbag import Handbag
            handbag = Handbag(self.handbag_path)
            object.__setattr__(self, '_handbag', handbag)

            # Personality: use handbag default if not explicitly provided
            if self.personality is None:
                hb_personality = handbag.get_default_personality()
                if hb_personality is not None:
                    handbag.attach_rag_to_personality(hb_personality)
                    object.__setattr__(self, 'personality', hb_personality)

            # Workspace: use handbag workspace if not explicitly provided
            if self.workspace_path is None:
                hb_ws = handbag.get_workspace_path()
                if hb_ws:
                    object.__setattr__(self, 'workspace_path', hb_ws)

            # Tool files: append handbag tools to any explicitly provided ones
            hb_tool_files = handbag.get_tool_files()
            if hb_tool_files:
                existing_tools = self.tool_files or []
                object.__setattr__(self, 'tool_files', list(existing_tools) + hb_tool_files)

            # Skills dirs: append handbag skills dir to any explicitly provided ones
            hb_skills_dirs = handbag.get_skills_dirs()
            if hb_skills_dirs:
                existing_dirs = self.skills_dirs or []
                object.__setattr__(self, 'skills_dirs', list(existing_dirs) + hb_skills_dirs)

            # Skills mode from manifest (if not explicitly set via capabilities)
            hb_skills_mode = handbag.get_skills_mode()
            if hb_skills_mode and self.capabilities.skills_mode == "loadable":
                # Only override if the user hasn't explicitly changed the default
                self.capabilities.skills_mode = hb_skills_mode

            # Memory manager: create from handbag if not explicitly provided
            if self.memory_manager is None:
                hb_mem = handbag.create_memory_manager()
                if hb_mem is not None:
                    object.__setattr__(self, 'memory_manager', hb_mem)
        else:
            object.__setattr__(self, '_handbag', None)

        # Validate that we have a personality (from explicit param or handbag)
        if self.personality is None:
            raise ValueError(
                "Agent requires a personality. Either:\n"
                "  1. Pass 'personality' directly, OR\n"
                "  2. Provide 'handbag_path' pointing to a folder with a 'personalities/' subdirectory."
            )

        # Resolve workspace path
        if self.workspace_path:
            ws = Path(self.workspace_path)
            ws.mkdir(parents=True, exist_ok=True)
            object.__setattr__(self, '_resolved_workspace', ws.resolve())
        else:
            object.__setattr__(self, '_resolved_workspace', None)

        # Initialize failure memory
        if FailureMemory is not None:
            object.__setattr__(self, '_failure_memory', FailureMemory())
        else:
            object.__setattr__(self, '_failure_memory', SimpleNamespace(failures=[], _signatures=set()))

        # Initialize skills manager if not provided
        if self.skills_manager is None:
            object.__setattr__(self, 'skills_manager', SkillsManager(
                skills_dirs=self.skills_dirs,
                skills_files=self.skills_files,
                mode=self.capabilities.skills_mode,
            ))

        # Initialize sub-agent spawner
        object.__setattr__(self, '_sub_agent_spawner', SubAgentSpawner(
            parent_agent=self,
            max_depth=self.capabilities.max_sub_agent_depth,
            max_per_turn=self.capabilities.max_sub_agents_per_turn,
        ))
        self._sub_agent_spawner.set_depth(self._parent_depth)

        # Initialize model switcher
        object.__setattr__(self, '_model_switcher', ModelSwitcher(self.lc))

    # ---------------------------------------------------------------- derived

    @property
    def display_name(self) -> str:
        return self.name or getattr(self.personality, "name", "Agent")

    @property
    def system_prompt(self) -> str:
        return getattr(self.personality, "system_prompt", "") or ""

    def has_knowledge(self) -> bool:
        return getattr(self.personality, "has_data", False)

    # ---------------------------------------------------------------- handbag API

    @property
    def handbag(self):
        """Returns the loaded Handbag instance, or None if no handbag was provided."""
        return self._handbag

    def list_handbag_personalities(self) -> Dict[str, Any]:
        """
        Lists all personalities available in the handbag.
        Returns an empty dict if no handbag is loaded.
        """
        if self._handbag:
            return self._handbag.get_personalities()
        return {}

    def switch_handbag_personality(self, name: str) -> bool:
        """
        Switches to a different personality from the handbag.
        The new personality inherits the handbag's RAG data source if it doesn't
        have its own.

        Args:
            name: The name of the personality folder in the handbag.

        Returns:
            True if the switch was successful, False if the personality was not found.
        """
        if not self._handbag:
            return False
        personality = self._handbag.get_personality(name)
        if personality is None:
            return False
        self._handbag.attach_rag_to_personality(personality)
        object.__setattr__(self, 'personality', personality)
        ASCIIColors.info(f"[Agent] Switched to handbag personality: '{name}'")
        return True

    def __repr__(self) -> str:
        return f"<Agent name={self.display_name!r} role={self.role!r} id={self._agent_id[:8]}>"

    # ---------------------------------------------------------------- cancellation

    def cancel_generation(self) -> bool:
        object.__setattr__(self, '_cancel_flag', True)
        if hasattr(self.lc, 'cancel'):
            try:
                self.lc.cancel()
            except Exception:
                pass
        elif hasattr(self.lc, 'llm') and hasattr(self.lc.llm, 'cancel'):
            try:
                self.lc.llm.cancel()
            except Exception:
                pass
        return True

    def is_generation_cancelled(self) -> bool:
        return getattr(self, '_cancel_flag', False)

    def _reset_cancel_state(self):
        object.__setattr__(self, '_cancel_flag', False)

    # ---------------------------------------------------------------- workspace

    def get_workspace_path(self) -> Optional[str]:
        return str(self._resolved_workspace) if self._resolved_workspace else None

    def list_workspace_files(self) -> List[str]:
        if not self._resolved_workspace:
            return []
        result = []
        for f in self._resolved_workspace.rglob("*"):
            if f.is_file():
                rel_parts = f.relative_to(self._resolved_workspace).parts
                if not any(part in _IGNORED_WS_DIRS for part in rel_parts):
                    if not f.suffix.lower() in _IGNORED_WS_EXTS:
                        result.append(str(f.relative_to(self._resolved_workspace)))
        return sorted(result)

    def _sync_workspace(self, files_before: Dict, files_after: Dict) -> List[Dict[str, Any]]:
        changes = []
        new_files = set(files_after.keys()) - set(files_before.keys())
        for rel_path in new_files:
            file_info = files_after[rel_path]
            changes.append({"action": "created", "path": str(rel_path), "size": file_info.get("size", 0)})
        common_files = set(files_after.keys()) & set(files_before.keys())
        for rel_path in common_files:
            if files_before[rel_path].get("hash") != files_after[rel_path].get("hash"):
                changes.append({"action": "modified", "path": str(rel_path), "size": files_after[rel_path].get("size", 0)})
        return changes

    def _take_workspace_snapshot(self) -> Dict:
        snapshot = {}
        if not self._resolved_workspace:
            return snapshot
        for f in self._resolved_workspace.rglob("*"):
            if not f.is_file():
                continue
            rel_parts = f.relative_to(self._resolved_workspace).parts
            if any(part in _IGNORED_WS_DIRS for part in rel_parts):
                continue
            if f.suffix.lower() in _IGNORED_WS_EXTS:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                file_hash = hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()
                snapshot[f.relative_to(self._resolved_workspace)] = {
                    "hash": file_hash,
                    "size": f.stat().st_size,
                    "path": f
                }
            except Exception:
                try:
                    snapshot[f.relative_to(self._resolved_workspace)] = {
                        "hash": None,
                        "size": f.stat().st_size,
                        "path": f
                    }
                except Exception:
                    pass
        return snapshot

    # ---------------------------------------------------------------- skills API

    def list_skills(self) -> List[Dict[str, Any]]:
        """Returns a list of all available skills."""
        return self.skills_manager.list_skills()

    def add_skill(self, title: str, description: str, category: str, content: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Creates a new skill."""
        return self.skills_manager.add_skill(title, description, category, content, tags)

    def update_skill(self, title: str, new_content: str) -> Dict[str, Any]:
        """Updates an existing skill."""
        return self.skills_manager.update_skill(title, new_content)

    def delete_skill(self, title: str) -> Dict[str, Any]:
        """Deletes a skill."""
        return self.skills_manager.delete_skill(title)

    def reload_skills(self):
        """Reloads all skills from disk."""
        self.skills_manager.reload()

    # ---------------------------------------------------------------- model switching API

    def list_available_models(self) -> List[str]:
        """Lists models available for switching."""
        return self._model_switcher.list_models()

    def get_current_model(self) -> str:
        """Returns the current model name."""
        return self._model_switcher.get_current_model()

    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Switches to a different model."""
        return self._model_switcher.switch_model(model_name)

    def restore_original_model(self) -> Dict[str, Any]:
        """Restores the original model if it was switched."""
        return self._model_switcher.restore_original_model()

    # ---------------------------------------------------------------- sub-agent API

    def spawn_sub_agent(self, instruction: str, personality_conditioning: Optional[str] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Spawns a child agent for a sub-task."""
        return self._sub_agent_spawner.spawn(instruction, personality_conditioning, model_name)

    # ---------------------------------------------------------------- tool discovery

    def _discover_tools(self, explicit_tools: Optional[Dict], tool_files: Optional[List], enable_code_execution: bool) -> Dict[str, Dict[str, Any]]:
        """Discovers and merges tools from all sources based on capability flags."""
        active_tools = {}

        # 1. Built-in workspace tools
        if self._resolved_workspace and self.capabilities.enable_workspace_tools:
            active_tools.update(_get_builtin_workspace_tools())

        # 2. Binding tools (TTI, TTS, STT, TTM, TTV)
        binding_tools = BindingToolsBuilder.build_tools(self.lc, self.capabilities, self._resolved_workspace)
        active_tools.update(binding_tools)

        # 3. Skills tools (load, list, create, update, delete)
        if self.capabilities.enable_skill_loading:
            active_tools["tool_load_skill"] = self._make_load_skill_tool()
            active_tools["tool_list_skills"] = self._make_list_skills_tool()
        if self.capabilities.enable_skill_creation:
            active_tools["tool_create_skill"] = self._make_create_skill_tool()
            active_tools["tool_update_skill"] = self._make_update_skill_tool()
            active_tools["tool_delete_skill"] = self._make_delete_skill_tool()

        # 4. Sub-agent tool
        if self.capabilities.enable_sub_agents and self._parent_depth < self.capabilities.max_sub_agent_depth:
            active_tools["tool_spawn_sub_agent"] = self._make_spawn_sub_agent_tool()

        # 5. Model switching tool
        if self.capabilities.enable_model_switching:
            active_tools["tool_switch_model"] = self._make_switch_model_tool()
            active_tools["tool_list_models"] = self._make_list_models_tool()

        # 6. LCP tools (from client binding)
        lcp_binding = getattr(self.lc, 'tools', None)
        if lcp_binding is None and (tool_files or self._resolved_workspace):
            try:
                from lollms_client.tools_bindings.lcp import LCPBinding
                default_tools = Path(__file__).parent.parent / "tools_bindings" / "lcp" / "default_tools"
                lcp_binding = LCPBinding(
                    tools_folders=[str(default_tools)] if default_tools.exists() else []
                )
            except Exception as ex:
                trace_exception(ex)
                lcp_binding = None

        if lcp_binding and hasattr(lcp_binding, 'to_chat_tool_specs'):
            try:
                if enable_code_execution and hasattr(lcp_binding, 'mount_tool_library'):
                    lcp_binding.mount_tool_library('execute_python_code')
                lcp_tools = lcp_binding.to_chat_tool_specs()
                if not enable_code_execution:
                    lcp_tools = {k: v for k, v in lcp_tools.items() if k != 'tool_execute_python_code'}
                active_tools.update(lcp_tools)
            except Exception as ex:
                trace_exception(ex)

        # 7. File-based tools (ToolsManager)
        if tool_files:
            try:
                tools_mgr = ToolsManager()
                file_tools = tools_mgr.build_inline_tools_dict(tool_files)
                active_tools.update(file_tools)
            except Exception as ex:
                trace_exception(ex)

        # 8. Explicit tools (from parameter)
        if explicit_tools:
            active_tools.update(explicit_tools)

        return active_tools

    # ---------------------------------------------------------------- skill tools builders

    def _make_load_skill_tool(self) -> Dict[str, Any]:
        def tool_load_skill(title: str) -> dict:
            """
            Load the full content of a skill by title. Use this to access detailed instructions
            or knowledge capsules that are not always visible in your context.

            Args:
                title (str): The title of the skill to load (case-insensitive).
            """
            content = self.skills_manager.load_skill(title)
            if content:
                return {"success": True, "output": content}
            return {"success": False, "error": f"Skill '{title}' not found."}

        return {
            "name": "tool_load_skill",
            "description": "Load the full content of a skill by title. Skills contain reusable knowledge, instructions, and best practices.",
            "parameters": [
                {"name": "title", "type": "str", "description": "The title of the skill to load."}
            ],
            "callable": tool_load_skill,
        }

    def _make_list_skills_tool(self) -> Dict[str, Any]:
        def tool_list_skills() -> dict:
            """
            List all available skills with their titles, descriptions, and categories.
            """
            skills_list = self.skills_manager.list_skills()
            if not skills_list:
                return {"success": True, "output": "No skills available."}
            lines = []
            for s in skills_list:
                vis = " [always visible]" if s.get("always_visible") else ""
                cat = f" [{s.get('category', '')}]" if s.get("category") else ""
                lines.append(f"- {s['title']}{cat}{vis}: {s.get('description', 'No description')}")
            return {"success": True, "output": "\n".join(lines)}

        return {
            "name": "tool_list_skills",
            "description": "List all available skills with their titles, descriptions, and categories.",
            "parameters": [],
            "callable": tool_list_skills,
        }

    def _make_create_skill_tool(self) -> Dict[str, Any]:
        def tool_create_skill(title: str, description: str, category: str, content: str, tags: str = "") -> dict:
            """
            Create a new skill (knowledge capsule) that persists across sessions.
            Skills are stored outside the workspace in a dedicated skills directory.

            Args:
                title (str): A clear, descriptive title for the skill.
                description (str): A one-line summary of what the skill covers.
                category (str): Domain or category (e.g., 'python', 'data_analysis', 'debugging').
                content (str): The full skill content (Markdown with instructions, examples, rules).
                tags (str, optional): Comma-separated tags for searchability.
            """
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            return self.skills_manager.add_skill(title, description, category, content, tag_list)

        return {
            "name": "tool_create_skill",
            "description": "Create a new persistent skill (knowledge capsule) that survives across sessions. Use this to save reusable knowledge, methodologies, or best practices.",
            "parameters": [
                {"name": "title", "type": "str", "description": "Clear, descriptive title for the skill."},
                {"name": "description", "type": "str", "description": "One-line summary of the skill."},
                {"name": "category", "type": "str", "description": "Domain or category (e.g., 'python', 'data_analysis')."},
                {"name": "content", "type": "str", "description": "Full skill content in Markdown (instructions, examples, rules)."},
                {"name": "tags", "type": "str", "description": "Comma-separated tags for searchability.", "optional": True},
            ],
            "callable": tool_create_skill,
        }

    def _make_update_skill_tool(self) -> Dict[str, Any]:
        def tool_update_skill(title: str, new_content: str) -> dict:
            """
            Update the content of an existing skill. The skill's metadata (title, description) is preserved.

            Args:
                title (str): The title of the skill to update.
                new_content (str): The new full content for the skill.
            """
            return self.skills_manager.update_skill(title, new_content)

        return {
            "name": "tool_update_skill",
            "description": "Update the content of an existing skill. Use this to refine or evolve your knowledge capsules.",
            "parameters": [
                {"name": "title", "type": "str", "description": "Title of the skill to update."},
                {"name": "new_content", "type": "str", "description": "The new full content for the skill."},
            ],
            "callable": tool_update_skill,
        }

    def _make_delete_skill_tool(self) -> Dict[str, Any]:
        def tool_delete_skill(title: str) -> dict:
            """
            Delete a skill permanently.

            Args:
                title (str): The title of the skill to delete.
            """
            return self.skills_manager.delete_skill(title)

        return {
            "name": "tool_delete_skill",
            "description": "Delete a skill permanently. Use with caution.",
            "parameters": [
                {"name": "title", "type": "str", "description": "Title of the skill to delete."},
            ],
            "callable": tool_delete_skill,
        }

    # ---------------------------------------------------------------- sub-agent tool builder

    def _make_spawn_sub_agent_tool(self) -> Dict[str, Any]:
        def tool_spawn_sub_agent(instruction: str, personality_conditioning: str = "", model_name: str = "") -> dict:
            """
            Spawn a focused sub-agent to handle a specific sub-task. The sub-agent shares your workspace
            and can use tools, but cannot spawn further sub-agents (to prevent infinite recursion).

            Use this for complex delegations like:
            - "Write a comprehensive Python script that..."
            - "Research and summarize the key points of..."
            - "Design an HTML presentation about..."

            Args:
                instruction (str): The specific task or instruction for the sub-agent.
                personality_conditioning (str, optional): Custom system prompt for the sub-agent.
                model_name (str, optional): Specific model to use (empty = parent's model).
            """
            return self._sub_agent_spawner.spawn(
                instruction=instruction,
                personality_conditioning=personality_conditioning or None,
                model_name=model_name or None,
            )

        return {
            "name": "tool_spawn_sub_agent",
            "description": "Spawn a focused sub-agent to handle a sub-task. The sub-agent shares your workspace but cannot spawn further sub-agents.",
            "parameters": [
                {"name": "instruction", "type": "str", "description": "The specific task for the sub-agent."},
                {"name": "personality_conditioning", "type": "str", "description": "Custom system prompt for the sub-agent (optional).", "optional": True},
                {"name": "model_name", "type": "str", "description": "Specific model to use (empty = parent's model).", "optional": True},
            ],
            "callable": tool_spawn_sub_agent,
        }

    # ---------------------------------------------------------------- model switching tool builders

    def _make_switch_model_tool(self) -> Dict[str, Any]:
        def tool_switch_model(model_name: str) -> dict:
            """
            Switch to a different LLM model on the fly. Use this when a task requires a different
            model's strengths (e.g., a coding model for code, a reasoning model for analysis).

            Args:
                model_name (str): The exact model name to switch to. Use tool_list_models to see available options.
            """
            return self._model_switcher.switch_model(model_name)

        return {
            "name": "tool_switch_model",
            "description": "Switch to a different LLM model on the fly. Use tool_list_models first to see available options.",
            "parameters": [
                {"name": "model_name", "type": "str", "description": "The exact model name to switch to."},
            ],
            "callable": tool_switch_model,
        }

    def _make_list_models_tool(self) -> Dict[str, Any]:
        def tool_list_models() -> dict:
            """
            List all available models that can be switched to.
            """
            models = self._model_switcher.list_models()
            current = self._model_switcher.get_current_model()
            if not models:
                return {"success": True, "output": "No models available for switching."}
            lines = [f"Current model: {current}", "Available models:"]
            for m in models:
                marker = " (current)" if m == current else ""
                lines.append(f"  - {m}{marker}")
            return {"success": True, "output": "\n".join(lines)}

        return {
            "name": "tool_list_models",
            "description": "List all available LLM models that can be switched to.",
            "parameters": [],
            "callable": tool_list_models,
        }

    # ---------------------------------------------------------------- prompt builders

    def _build_tool_descriptions(self, active_tools: Dict[str, Dict[str, Any]]) -> str:
        if not active_tools:
            return ""

        lines = [
            "\n=== TOOLS AVAILABLE ===",
            "To use a tool, you MUST emit a single <tool> tag on a new line with the tool parameters as a JSON object, and then stop generating.",
            "Do NOT write prose before or after the tag on the same line.",
            "",
            "=== TOOL CALLING DISCIPLINE (CRITICAL) ===",
            "1. **EXACT CLOSING TAG**: The closing tag is `</tool>`. You MUST NOT write ``` or any other variation.",
            "2. **NEW LINE ONLY**: The <tool> tag MUST start on a brand new line.",
            "3. **NO PROSE AROUND IT**: Do NOT write introductory text before the tag.",
            "",
            "=== TASK COMPLETION PROTOCOL ===",
            "1. **TOOL CALLS ARE TEMPORARY**: You call a `<tool>` only to gather data you don't have.",
            "2. **ANSWERING IS THE GOAL**: Once you have the data, writing a comprehensive response IS completion.",
            "3. **HOW TO FINISH**: When your task is complete, end with a `<done/>` tag on a new line.",
            "4. **NEVER LOOP**: If you have already written your final answer, you are DONE.",
            "=== END TASK COMPLETION PROTOCOL ===",
            "",
            "Available tools:",
        ]

        for t_name, t_spec in active_tools.items():
            desc = t_spec.get("description", "")
            params_list = t_spec.get("parameters", [])
            param_desc = ", ".join([f"{p['name']}: {p['type']}" for p in params_list])
            lines.append(f"- {t_name}({param_desc}): {desc}")

        allowed_tool_names = list(active_tools.keys())
        lines.append(f"\n🚨 **STRICT TOOL REGISTRY ENFORCEMENT** 🚨")
        lines.append(f"You are STRICTLY FORBIDDEN from calling any tool not listed above.")
        lines.append(f"The ONLY valid tool names you may use are: {', '.join(allowed_tool_names)}")
        lines.append("=== END TOOLS ===")

        return "\n".join(lines)

    def _build_system_prompt(self, active_tools: Dict, enable_code_execution: bool) -> str:
        sys_prompt = self.system_prompt or ""

        rules = (
            "\n=== ACTION EXECUTION & TERMINATION PROTOCOL (CRITICAL) ===\n"
            "1. **INTENT ≠ EXECUTION**: Stating 'I will search...' in text DOES NOT execute the action. You MUST output the `<tool>` tag.\n"
            "2. **MANDATORY TAG EMISSION**: To execute an action, you MUST output the `<tool>` tag immediately.\n"
            "3. **EXPLICIT TERMINATION WITH `<done/>`**: When finished, end with a `<done/>` tag on a new line.\n"
            "4. **SAME-SESSION CONTINUATION**: When executing a sequence, emit the next action's tag in your IMMEDIATE NEXT response.\n"
            "5. **ROUND 1 SHORT-CIRCUIT**: If the user's request is purely conversational, respond conversationally without `<done/>`.\n"
            "\n=== TOOL CALLING DISCIPLINE (CRITICAL) ===\n"
            "1. **Tool Results ≠ Tool Calls**: When a tool returns JSON, it's a RESULT, not a new call.\n"
            "2. **One Call Per Task**: Once a tool succeeds, analyze and answer.\n"
            "3. **Loop Prevention**: Repeating a successful tool call with identical parameters is a CRITICAL ERROR.\n"
            "4. **File Outputs**: When a tool returns a file, it's ALREADY saved. Do NOT call it again.\n"
            "\n=== SKILLS SYSTEM ===\n"
            "Skills are persistent knowledge capsules stored outside the workspace. They survive across sessions.\n"
            "Use `tool_list_skills` to see available skills, and `tool_load_skill` to load their full content.\n"
            "If you discover a reusable methodology or best practice, use `tool_create_skill` to save it for future use.\n"
            "Use `tool_update_skill` to refine existing skills as you learn more.\n"
            "\n=== SUB-AGENT DELEGATION ===\n"
            "If `tool_spawn_sub_agent` is available, you can delegate complex sub-tasks to a focused child agent.\n"
            "The child shares your workspace but cannot spawn further sub-agents.\n"
            "Use this for heavy tasks like writing large scripts, researching topics, or designing presentations.\n"
            "\n=== MODEL SWITCHING ===\n"
            "If `tool_switch_model` is available, you can switch to a different model mid-task.\n"
            "Use `tool_list_models` to see available models, then `tool_switch_model` to switch.\n"
            "This is useful when a task requires a model with different strengths.\n"
            "\n=== THINKING & REASONING CONSTRAINT ===\n"
            "If you output thoughts enclosed in <think> tags, you MUST output all functional XML tags AFTER the closing tag.\n"
            "\n=== ANTI-MIMICRY PROTOCOL (CRITICAL) ===\n"
            "1. **NEVER OUTPUT SYSTEM MARKERS**: You are STRICTLY FORBIDDEN from generating `<processing>` blocks or `[SYSTEM:` markers.\n"
            "2. **USE REAL TAGS**: To call tools, use the actual `<tool>` XML tags.\n"
        )

        # Workspace context
        workspace_ctx = ""
        if self._resolved_workspace:
            workspace_ctx = "\n" + _build_workspace_context(self._resolved_workspace)

        # Skills context
        skills_ctx = ""
        if self.skills_manager:
            skills_ctx_str = self.skills_manager.build_context()
            if skills_ctx_str:
                skills_ctx = "\n" + skills_ctx_str

        # Tool descriptions
        tool_desc = "\n" + self._build_tool_descriptions(active_tools) if active_tools else ""

        # Memory context
        mem_ctx = ""
        if self.memory_manager:
            try:
                working = self.memory_manager.build_working_zone() if hasattr(self.memory_manager, 'build_working_zone') else ""
                handles = self.memory_manager.build_handles_zone() if hasattr(self.memory_manager, 'build_handles_zone') else ""
                parts = [p for p in (working, handles) if p]
                if parts:
                    mem_ctx = "\n=== ACTIVE MEMORIES ===\n" + "\n".join(parts) + "\n"
            except Exception:
                pass

        # Capability summary
        caps_summary = f"\n=== CAPABILITIES ===\n{json.dumps(self.capabilities.to_dict(), indent=2)}\n=== END CAPABILITIES ==="

        return sys_prompt + "\n" + rules + workspace_ctx + skills_ctx + tool_desc + mem_ctx + caps_summary

    # ---------------------------------------------------------------- tool execution

    def _execute_tool(self, tool_name: str, tool_params: Dict[str, Any], active_tools: Dict) -> Dict[str, Any]:
        old_cwd = os.getcwd()
        if self._resolved_workspace:
            ws_dir = self._resolved_workspace
        else:
            ws_dir = Path(".")
        ws_dir.mkdir(parents=True, exist_ok=True)
        ws_dir_str = str(ws_dir.resolve())

        try:
            os.chdir(ws_dir_str)

            sanitized_params = {}
            for key, value in tool_params.items():
                if isinstance(value, str):
                    sanitized_value = value
                    for prefix in ["workspace/", "data_workspace/", "./workspace/", "./data_workspace/"]:
                        if sanitized_value.lower().startswith(prefix):
                            sanitized_value = sanitized_value[len(prefix):]
                            break
                    sanitized_params[key] = sanitized_value
                else:
                    sanitized_params[key] = value

            lcp_binding = getattr(self.lc, 'tools', None)
            tool_def = active_tools.get(tool_name, {})

            if "callable" in tool_def:
                import inspect as _inspect
                call_kwargs = dict(sanitized_params)
                _tool_sig = _inspect.signature(tool_def["callable"]).parameters
                if "discussion_instance" in _tool_sig:
                    call_kwargs["discussion_instance"] = None
                if "lollms_client_instance" in _tool_sig:
                    call_kwargs["lollms_client_instance"] = self.lc

                try:
                    result = tool_def["callable"](**call_kwargs)
                    return result if isinstance(result, dict) else {"success": True, "output": str(result)}
                except Exception as exec_err:
                    trace_exception(exec_err)
                    return {"success": False, "error": f"Tool '{tool_name}' crashed: {exec_err}", "traceback": traceback.format_exc()}

            elif lcp_binding and hasattr(lcp_binding, 'execute_tool'):
                try:
                    result = lcp_binding.execute_tool(
                        tool_name,
                        sanitized_params,
                        discussion_instance=None,
                        lollms_client_instance=self.lc
                    )
                    if isinstance(result, dict) and "output" in result:
                        return result["output"]
                    return result if isinstance(result, dict) else {"success": True, "output": str(result)}
                except Exception as lcp_err:
                    trace_exception(lcp_err)
                    return {"success": False, "error": f"LCP tool '{tool_name}' crashed: {lcp_err}", "traceback": traceback.format_exc()}

            else:
                return {"success": False, "error": f"Tool '{tool_name}' has no callable and no LCP binding available.", "status_code": 404}

        finally:
            os.chdir(old_cwd)

    # ---------------------------------------------------------------- chat (main agentic loop)

    def chat(
        self,
        prompt: str,
        *,
        conversation: Optional[List[Dict[str, str]]] = None,
        streaming_callback: Optional[Callable] = None,
        images: Optional[List[str]] = None,
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        tool_files: Optional[List[Union[str, Path]]] = None,
        max_reasoning_steps: int = 20,
        temperature: float = 0.7,
        n_predict: int = 4096,
        enable_code_execution: Optional[bool] = None,
        enable_memory: bool = True,
        remove_thinking_blocks: bool = True,
        use_internal_history: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Powerful agentic chat with multi-step reasoning, tools, skills, sub-agents,
        model switching, and binding integration.

        Parameters
        ----------
        prompt : str
            The user's prompt / task description.
        conversation : list of dicts, optional
            External conversation history. If None and use_internal_history is True, uses internal.
        streaming_callback : callable, optional
            Callback (chunk, msg_type, meta) -> bool for real-time token streaming.
        images : list of str, optional
            Base64-encoded images for vision-capable models.
        tools : dict, optional
            Explicit tool specs to include in addition to discovered tools.
        tool_files : list of str/Path, optional
            File paths to lollms-format tool scripts.
        max_reasoning_steps : int
            Maximum agentic reasoning rounds (default 20).
        temperature : float
            Sampling temperature (default 0.7).
        n_predict : int
            Max tokens per generation (default 4096).
        enable_code_execution : bool, optional
            Override the capability flag for this turn only. None = use capabilities.
        enable_memory : bool
            If True and memory_manager is set, performs memory operations.
        remove_thinking_blocks : bool
            If True, strips <think>...</think> from final response.
        use_internal_history : bool
            If True, maintains and updates internal conversation history.

        Returns
        -------
        dict
            {
                "response": str,
                "tool_calls": list,
                "tool_results": list,
                "rounds": int,
                "workspace_changes": list,
                "was_cancelled": bool,
                "skills_created": list,
                "skills_updated": list,
                "sub_agents_spawned": int,
                "model_switches": list,
            }
        """
        self._reset_cancel_state()

        # Reset sub-agent spawner for this turn
        self._sub_agent_spawner.reset_turn()

        # Determine code execution flag
        code_exec = enable_code_execution if enable_code_execution is not None else self.capabilities.enable_code_execution

        # Initialize failure memory
        if self._failure_memory and hasattr(self._failure_memory, '_signatures'):
            self._failure_memory._signatures.clear()
            if hasattr(self._failure_memory, 'failures'):
                self._failure_memory.failures = []

        # Memory pre-turn
        if enable_memory and self.memory_manager:
            try:
                if hasattr(self.memory_manager, 'apply_decay'):
                    self.memory_manager.apply_decay()
                if prompt and hasattr(self.memory_manager, 'auto_pull_deep_memories'):
                    self.memory_manager.auto_pull_deep_memories(prompt)
                if hasattr(self.memory_manager, 'enforce_budget'):
                    token_counter = getattr(self.lc, 'count_tokens', None)
                    self.memory_manager.enforce_budget(token_counter=token_counter)
            except Exception as ex:
                trace_exception(ex)

        # RAG pre-hydration
        rag_context = ""
        if self.has_knowledge() and hasattr(self.personality, 'query_data'):
            try:
                rag_res = self.personality.query_data(prompt)
                if rag_res and rag_res.get("success") and rag_res.get("sources"):
                    sources_text = [f"Source [{s.get('source')}]: {s.get('content')}" for s in rag_res.get("sources", [])]
                    if sources_text:
                        rag_context = "\n=== RETRIEVED RAG CONTEXT ===\n" + "\n\n".join(sources_text[:3]) + "\n=== END RAG CONTEXT ===\n"
            except Exception as ex:
                trace_exception(ex)

        # Discover tools
        active_tools = self._discover_tools(tools, tool_files or self.tool_files, code_exec)

        # Build system prompt
        full_system_prompt = self._build_system_prompt(active_tools, code_exec)
        if rag_context:
            full_system_prompt += "\n" + rag_context

        # Build conversation
        if conversation is not None:
            base_conversation = list(conversation)
        elif use_internal_history:
            base_conversation = list(self._conversation)
        else:
            base_conversation = []

        base_conversation.append({"role": "user", "content": prompt})

        # Agentic loop state
        virtual_history: List[SimpleNamespace] = []
        tool_calls_this_turn: List[Dict[str, Any]] = []
        tool_results_this_turn: List[Dict[str, Any]] = []
        round_count = 0
        was_cancelled = False
        tool_signature_counts: Dict[str, int] = {}
        successful_tool_signatures: set = set()
        final_response = ""
        workspace_changes: List[Dict[str, Any]] = []
        skills_created: List[str] = []
        skills_updated: List[str] = []
        sub_agents_spawned = 0
        model_switches: List[str] = []

        while round_count < max_reasoning_steps:
            if self.is_generation_cancelled():
                was_cancelled = True
                break

            round_count += 1

            if hasattr(self.lc, 'llm') and hasattr(self.lc.llm, 'reset_cancel'):
                try:
                    self.lc.llm.reset_cancel()
                except Exception:
                    pass

            messages = [{"role": "system", "content": full_system_prompt}]
            messages.extend(base_conversation)

            for vh in virtual_history:
                role = "user" if vh.sender_type == "user" else "assistant"
                messages.append({"role": role, "content": vh.content})

            messages = _normalize_messages(messages)

            ss = _AgentStreamState(callback=streaming_callback)

            def _inline_relay(chunk, msg_type=None, meta=None):
                if self.is_generation_cancelled():
                    return False
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss._cb(chunk, msg_type, meta) if streaming_callback else True
                if isinstance(chunk, str):
                    if meta and meta.get("was_processed"):
                        return True
                    return ss.feed(chunk)
                return True

            gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("streaming_callback", "temperature", "n_predict", "stream")}
            gen_kwargs["n_predict"] = min(n_predict, self.max_tokens_per_turn)
            gen_kwargs["temperature"] = temperature

            try:
                self.lc.generate_from_messages(
                    messages=messages,
                    images=images if images else None,
                    stream=True,
                    streaming_callback=_inline_relay,
                    **gen_kwargs
                )
            except Exception as gen_err:
                if self.is_generation_cancelled():
                    was_cancelled = True
                    break
                else:
                    ASCIIColors.error(f"[Agent.chat] Generation error: {gen_err}")
                    final_response = f"[Generation error: {gen_err}]"
                    break

            if self.is_generation_cancelled():
                was_cancelled = True
                break

            ss.flush_remaining_buffer()

            if ss.was_done_detected():
                ASCIIColors.info("[Agent.chat] <done/> tag detected. Terminating agentic loop.")
                final_response = ss.get_clean_text()
                final_response = re.sub(r'(?m)^\s*<done\s*/?>', '', final_response, flags=re.IGNORECASE).strip()
                break

            clean_text = ss.get_clean_text().strip()
            content_without_processing = re.sub(r'<processing[^>]*>.*?</processing>', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()

            if not content_without_processing and not ss.tool_trigger and not ss.was_action_dispatched():
                ASCIIColors.warning("[Agent.chat] Empty LLM response detected. Breaking loop.")
                virtual_history.append(SimpleNamespace(sender_type="assistant", content="[No output generated]"))
                final_response = ""
                break

            if ss.tool_trigger:
                tool_call_json_str = ss.get_tool_call_json()
                if tool_call_json_str:
                    try:
                        call_data = json.loads(tool_call_json_str)
                    except Exception:
                        call_data = {}

                    if not isinstance(call_data, dict) or not call_data.get("name"):
                        ASCIIColors.warning(f"[Agent.chat] Malformed tool call. JSON: {tool_call_json_str[:200]}")

                        malformed_sig = "unknown::malformed"
                        if self._failure_memory and hasattr(self._failure_memory, '_signatures'):
                            self._failure_memory._signatures.add(malformed_sig)

                        correction_msg = (
                            "=== ⚠️ TOOL CALL FORMAT ERROR ===\n"
                            "Your last tool call was malformed or missing the 'name' field. "
                            f"Raw received: `{tool_call_json_str[:150]}`\n"
                            "You MUST output a valid JSON object with a 'name' key. "
                            "Example: <tool>{\"name\": \"tool_write_file\", \"parameters\": {\"file_name\": \"test.txt\", \"content\": \"hello\"}}</tool>\n"
                            "Please output the corrected tool call now."
                        )

                        raw_round_text = ss.get_clean_text()
                        clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                        clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'<tool>.*?</tool>', '', clean_history_text, flags=re.DOTALL | re.IGNORECASE)
                        clean_history_text = clean_history_text.strip() or "[Malformed tool call emitted]"
                        virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text))
                        virtual_history.append(SimpleNamespace(sender_type="user", content=correction_msg))
                        continue

                    tool_name = call_data.get("name", "")
                    tool_params = call_data.get("parameters", {})

                    raw_round_text = ss.get_clean_text()
                    clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                    clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'<tool>.*?</tool>', '', clean_history_text, flags=re.DOTALL | re.IGNORECASE)
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text.strip()))

                    # Phantom tool check
                    if not active_tools or tool_name not in active_tools:
                        ASCIIColors.warning(f"[Agent.chat] Phantom tool call: '{tool_name}' not registered.")

                        if self._failure_memory and hasattr(self._failure_memory, '_signatures'):
                            try:
                                param_sig = json.dumps(tool_params, sort_keys=True, default=str)
                            except Exception:
                                param_sig = str(tool_params)
                            phantom_sig = f"{tool_name}::{param_sig}"
                            self._failure_memory._signatures.add(phantom_sig)

                        status_err = f"* Tool call blocked.\n"
                        details = f"Error Logs:\nTool '{tool_name}' is not available in this session.\n"
                        tool_close = f"{status_err}{details}<!-- status:failure -->\n</processing>\n\n"

                        if streaming_callback:
                            streaming_callback(tool_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                        available_str = ", ".join(f"`{t}`" for t in active_tools.keys()) if active_tools else "No tools available."
                        correction = (
                            f"=== ⚠️ INVALID TOOL CALL ===\n"
                            f"You attempted to call `{tool_name}`, which is NOT REGISTERED.\n"
                            f"Available tools: {available_str}\n"
                            f"If one is suitable, output the corrected `<tool>` call now. "
                            f"If none can accomplish the task, inform the user and complete with `<done/>`."
                        )
                        virtual_history.append(SimpleNamespace(sender_type="user", content=correction))

                        if not hasattr(self, '_phantom_counts'):
                            object.__setattr__(self, '_phantom_counts', {})
                        try:
                            param_sig = json.dumps(tool_params, sort_keys=True, default=str)
                        except Exception:
                            param_sig = str(tool_params)
                        phantom_sig = f"{tool_name}::{param_sig}"
                        self._phantom_counts[phantom_sig] = self._phantom_counts.get(phantom_sig, 0) + 1
                        if self._phantom_counts[phantom_sig] >= 2:
                            final_response = "I was unable to complete the requested task because the required tool is not available."
                            break
                        continue

                    # FailureMemory loop detection
                    try:
                        param_signature = json.dumps(tool_params, sort_keys=True, default=str)
                    except Exception:
                        param_signature = str(tool_params)
                    full_signature = f"{tool_name}::{param_signature}"

                    has_prev_failure = (
                        hasattr(self._failure_memory, '_signatures') and
                        full_signature in self._failure_memory._signatures
                    ) if self._failure_memory else False

                    if has_prev_failure:
                        if self.is_generation_cancelled():
                            was_cancelled = True
                            break

                        result_str = (
                            f"Error executing tool '{tool_name}': This exact parameters configuration failed on a previous round. "
                            f"Execution blocked to prevent infinite loop. Modify your parameters or try a different approach. "
                            f"If you cannot proceed, inform the user and emit `<done/>`."
                        )
                        status_err = f"* Tool call blocked to prevent loop.\n"
                        details = f"Loop Intercepted:\n{result_str}\n"
                        tool_close = f"{status_err}{details}<!-- status:failure -->\n</processing>\n\n"

                        if streaming_callback:
                            streaming_callback(tool_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=(
                                f'<tool_result name="{tool_name}" status="FAILED">\n'
                                f"{result_str}\n"
                                f"</tool_result>\n\n"
                                f"⚠️ **Tool Execution Failed & Loop Blocked.** Write your final response and emit `<done/>`."
                            )
                        ))
                        continue

                    # Success-Loop detection
                    current_file_hashes = {}
                    for k, v in tool_params.items():
                        if isinstance(v, str):
                            p = Path(v)
                            if p.is_file():
                                try:
                                    content = p.read_bytes()
                                    current_file_hashes[k] = hashlib.md5(content).hexdigest()
                                except Exception:
                                    pass
                    context_aware_signature = f"{full_signature}::{json.dumps(current_file_hashes, sort_keys=True)}"

                    if context_aware_signature in successful_tool_signatures:
                        ASCIIColors.warning(f"[Agent.chat] Repetitive SUCCESS loop blocked for '{tool_name}'.")
                        tool_res = {
                            "success": False,
                            "error": f"Repetitive tool call detected. You have already successfully called '{tool_name}' with these exact parameters. The output is already in your context. Do not call it again.",
                            "prompt_injection": f"\n\n🛑 **STOP.** You are calling '{tool_name}' again with the exact same parameters. Analyze the data already retrieved and answer. Emit `<done/>` to finish."
                        }
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=(
                                f'<tool_result name="{tool_name}" status="FAILED">\n'
                                f"Repetitive tool call detected. Output already in context.\n"
                                f"</tool_result>\n\n"
                                f"⚠️ **Tool Execution Blocked.** Write your final response and emit `<done/>`."
                            )
                        ))
                        status_err = f"* Tool call blocked to prevent success loop.\n"
                        details = f"Loop Intercepted:\nRepetitive successful tool call blocked\n<!-- status:failure -->\n</processing>\n\n"
                        if streaming_callback:
                            streaming_callback(status_err + details, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        continue
                    else:
                        tool_signature_counts[full_signature] = tool_signature_counts.get(full_signature, 0) + 1

                    # Take workspace snapshot before execution
                    files_before = self._take_workspace_snapshot()

                    # Execute the tool
                    tool_res = self._execute_tool(tool_name, tool_params, active_tools)

                    # Take workspace snapshot after execution
                    files_after = self._take_workspace_snapshot()

                    # Detect workspace changes
                    changes = self._sync_workspace(files_before, files_after)
                    if changes:
                        workspace_changes.extend(changes)

                    # Track special tool calls
                    if tool_name == "tool_spawn_sub_agent":
                        sub_agents_spawned += 1
                    elif tool_name == "tool_switch_model":
                        if isinstance(tool_res, dict) and tool_res.get("success"):
                            model_switches.append(tool_params.get("model_name", "unknown"))
                    elif tool_name == "tool_create_skill":
                        if isinstance(tool_res, dict) and tool_res.get("success"):
                            skills_created.append(tool_params.get("title", "unknown"))
                    elif tool_name == "tool_update_skill":
                        if isinstance(tool_res, dict) and tool_res.get("success"):
                            skills_updated.append(tool_params.get("title", "unknown"))

                    # Determine success/failure
                    inner_res = tool_res.get("output", tool_res) if isinstance(tool_res, dict) else tool_res
                    is_failure = (
                        (isinstance(inner_res, dict) and inner_res.get("success") is False) or
                        (isinstance(tool_res, dict) and tool_res.get("status_code", 200) != 200) or
                        (isinstance(inner_res, dict) and inner_res.get("error") and not inner_res.get("success", True))
                    )
                    tool_success = not is_failure

                    if tool_success:
                        successful_tool_signatures.add(context_aware_signature)

                    tool_calls_this_turn.append({"round": round_count, "name": tool_name, "parameters": tool_params})
                    tool_results_this_turn.append({"round": round_count, "name": tool_name, "result": tool_res, "success": tool_success})

                    clean_result_str = _sanitize_tool_result(tool_res)

                    # Cognitive checkpoint: offload large outputs
                    if self.lc and hasattr(self.lc, 'count_tokens'):
                        tool_output_tokens = self.lc.count_tokens(clean_result_str)
                    else:
                        tool_output_tokens = len(clean_result_str) // 4

                    if tool_output_tokens > 1500:
                        is_structured = (
                            tool_name.startswith("tool_query") or
                            tool_name.startswith("tool_execute_python_data") or
                            "|" in clean_result_str or
                            "```json" in clean_result_str
                        )
                        if is_structured:
                            clean_result_str = f"[SYSTEM: Tool returned {tool_output_tokens} tokens of structured data. Use aggregation/plotting tools next.]"
                        else:
                            if self._resolved_workspace:
                                log_filename = f"tool_output_{tool_name}_{round_count}.log"
                                log_filepath = self._resolved_workspace / log_filename
                                try:
                                    log_filepath.write_text(clean_result_str, encoding="utf-8", errors="ignore")
                                except Exception:
                                    pass
                                clean_result_str = f"[SYSTEM: Tool returned {tool_output_tokens} tokens. Saved to '{log_filename}'. Use tool_read_file to read it.]"

                    # Build UI processing block
                    if tool_success:
                        status_done = f"* Completed execution of '{tool_name}' successfully.\n"
                        safe_output = str(tool_res.get("output", clean_result_str))[:2000] if isinstance(tool_res, dict) else clean_result_str[:2000]
                        details = f"Output Logs:\n{safe_output}\n"
                    else:
                        status_done = f"* Completed execution with errors.\n"
                        error_msg = tool_res.get("error", "Unknown error") if isinstance(tool_res, dict) else "Unknown error"
                        details = f"Error Logs:\n{error_msg}\n"

                    status_meta = "failure" if is_failure else "success"
                    tool_close = f"{status_done}{details}<!-- status:{status_meta} -->\n</processing>\n\n"

                    if streaming_callback:
                        streaming_callback(tool_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    # Record failure in FailureMemory
                    if is_failure and self._failure_memory and hasattr(self._failure_memory, '_signatures'):
                        try:
                            param_sig = json.dumps(tool_params, sort_keys=True, default=str)
                        except Exception:
                            param_sig = str(tool_params)
                        full_sig = f"{tool_name}::{param_sig}"
                        self._failure_memory._signatures.add(full_sig)

                    # Build virtual history feedback
                    if tool_success:
                        pinj = None
                        if isinstance(tool_res, dict):
                            pinj = tool_res.get("prompt_injection")
                            if not pinj:
                                inner = tool_res.get("output")
                                if isinstance(inner, dict):
                                    pinj = inner.get("prompt_injection")

                        real_filename_instr = ""
                        if isinstance(tool_res, dict) and tool_res.get("image_filename"):
                            p_fn = tool_res["image_filename"]
                            real_filename_instr = (
                                f"🚨 **ACTUAL GENERATED FILE NAME**: `{p_fn}`\n"
                                f"   Reference this exact file in your answer.\n\n"
                            )
                        elif isinstance(tool_res, dict) and tool_res.get("audio_filename"):
                            p_fn = tool_res["audio_filename"]
                            real_filename_instr = f"🚨 **AUDIO FILE**: `{p_fn}`\n\n"
                        elif isinstance(tool_res, dict) and tool_res.get("video_filename"):
                            p_fn = tool_res["video_filename"]
                            real_filename_instr = f"🚨 **VIDEO FILE**: `{p_fn}`\n\n"

                        user_part = (
                            f"=== ✅ TOOL RESULT: {tool_name} ===\n"
                            f"⚠️ **WARNING**: The JSON below is the **RESULT** of your previous tool call. "
                            f"It is **NOT** a new tool call request. Do **NOT** re-execute it.\n\n"
                            f"{real_filename_instr}"
                            f"<tool_result name=\"{tool_name}\" status=\"SUCCESS\">\n"
                            f"{clean_result_str}\n"
                            f"</tool_result>\n\n"
                            f"🚨 **MANDATORY NEXT STEPS**:\n"
                            f"1. ✅ **ACKNOWLEDGE** the data is retrieved.\n"
                            f"2. 🧠 **ANALYZE** the result.\n"
                            f"3. 💬 **RESPOND** to the user.\n"
                            f"4. 🚫 **FORBIDDEN**: Do NOT call '{tool_name}' again with these parameters.\n"
                            f"5. 🏁 **TERMINATION**: When finished, end with `<done/>`.\n"
                        )

                        if pinj:
                            user_part += f"\n{pinj}\n"

                        if changes:
                            new_files_str = ", ".join(f"`{c['path']}`" for c in changes if c['action'] == 'created')
                            modified_str = ", ".join(f"`{c['path']}`" for c in changes if c['action'] == 'modified')
                            parts = []
                            if new_files_str:
                                parts.append(f"New files: {new_files_str}")
                            if modified_str:
                                parts.append(f"Modified files: {modified_str}")
                            if parts:
                                user_part += f"\n[SYSTEM: Workspace changes: {'; '.join(parts)}]\n"

                        tools_so_far = [tc["name"] for tc in tool_calls_this_turn]
                        unique_tools = list(dict.fromkeys(tools_so_far))
                        user_part += (
                            f"\n[SYSTEM: PROGRESS TRACKER — {len(tool_calls_this_turn)} tool call(s): "
                            f"{', '.join(unique_tools)}. Use results already in context.]"
                        )

                        virtual_history.append(SimpleNamespace(sender_type="user", content=user_part))
                    else:
                        user_part = (
                            f'<tool_result name="{tool_name}" status="FAILED">\n'
                            f"{clean_result_str}\n"
                            f"</tool_result>\n\n"
                            f"⚠️ **Tool Execution Failed.**\n"
                            f"1. **Analyze**: Read the error log.\n"
                            f"2. **Explore Alternatives**: Try another approach.\n"
                            f"3. **Inform the User**: If stuck, inform and emit `<done/>`."
                        )
                        virtual_history.append(SimpleNamespace(sender_type="user", content=user_part))

                    continue
                else:
                    break
            else:
                raw_round_text = ss.get_clean_text()

                done_match = re.search(r'(?m)^\s*<done\s*/?>\s*$', raw_round_text.strip())
                if done_match:
                    final_response = re.sub(r'(?m)^\s*<done\s*/?>\s*$', '', raw_round_text, flags=re.MULTILINE).strip()
                    break

                if len(tool_calls_this_turn) > 0:
                    ASCIIColors.info("[Agent.chat] Tool previously executed but no <done/>. Injecting continuation mandate.")
                    clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                    clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text.strip()))
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content="[SYSTEM: You stopped without <done/>. If complete, output final summary and `<done/>`. If not, emit next `<tool>` now.]"
                    ))
                    continue

                # Intent detection
                intent_pattern = re.compile(r'(let me|now i|next i|i will|i need to|we need to).*(query|get|fetch|build|create|analyze|summarize|aggregate|plot)', re.IGNORECASE)
                intent_match = intent_pattern.search(raw_round_text)
                has_intent = False
                if intent_match:
                    matched_line = intent_match.group(0)
                    line_end_idx = raw_round_text.find(matched_line) + len(matched_line)
                    line_end_char = raw_round_text[line_end_idx] if line_end_idx < len(raw_round_text) else ""
                    line_start_idx = raw_round_text.rfind('\n', 0, intent_match.start()) + 1
                    line_start = raw_round_text[line_start_idx:intent_match.start()].strip().lower()
                    is_question = line_end_char == '?' or line_start.startswith(("would you", "do you", "shall i", "should i", "could you"))
                    if not is_question:
                        has_intent = True

                has_tool_tag = "<tool>" in raw_round_text.lower()

                if has_intent and not has_tool_tag and not was_cancelled and round_count < max_reasoning_steps:
                    ASCIIColors.info("[Agent.chat] Detected pending tool intent. Forcing continuation...")
                    clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                    clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text.strip()))
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content="[SYSTEM: CRITICAL. You stopped before executing your stated intent. Output the <tool> tag NOW.]"
                    ))
                    continue

                final_response = raw_round_text
                break

        # Post-processing
        if was_cancelled:
            if final_response.strip():
                final_response += "\n\n[Generation cancelled by user]"
            else:
                final_response = "[Generation cancelled by user]"
        else:
            if remove_thinking_blocks and hasattr(self.lc, 'remove_thinking_blocks'):
                try:
                    final_response = self.lc.remove_thinking_blocks(final_response)
                except Exception:
                    pass
            elif remove_thinking_blocks:
                final_response = re.sub(r'(<think>|<thinking>).*?(|</thinking>)', '', final_response, flags=re.DOTALL | re.IGNORECASE).strip()
                final_response = re.sub(r'(<think>|<thinking>).*$', '', final_response, flags=re.DOTALL | re.IGNORECASE).strip()

        final_response = re.sub(r'(?m)^\s*<done\s*/?>', '', final_response, flags=re.IGNORECASE).strip()

        # Process memory tags
        if enable_memory and self.memory_manager:
            try:
                if hasattr(self.memory_manager, 'process_llm_output'):
                    cleaned, report = self.memory_manager.process_llm_output(final_response)
                    if cleaned != final_response:
                        final_response = cleaned
            except Exception as ex:
                trace_exception(ex)

        # Save episodic memory
        if enable_memory and self.memory_manager:
            try:
                clean_ai = re.sub(r'<processing.*?>.*?</processing>', '', final_response, flags=re.DOTALL)
                clean_ai = re.sub(r'<[^>]+>', '', clean_ai).strip()
                clean_user = prompt.strip()
                if clean_user and clean_ai and len(clean_user) > 10 and len(clean_ai) > 15:
                    from datetime import datetime
                    episode = f"Event/Interaction on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC:\nUser asked: \"{clean_user}\"\nAI responded: \"{clean_ai}\""
                    self.memory_manager.add(content=episode, importance=0.8, tags=["episode", "interaction"], level=1)
            except Exception as ex:
                trace_exception(ex)

        # Update internal conversation history
        if use_internal_history and not was_cancelled:
            self._conversation.append({"role": "user", "content": prompt})
            self._conversation.append({"role": "assistant", "content": final_response})

        self._reset_cancel_state()

        if hasattr(self, '_phantom_counts'):
            object.__setattr__(self, '_phantom_counts', {})

        return {
            "response": final_response,
            "tool_calls": tool_calls_this_turn,
            "tool_results": tool_results_this_turn,
            "rounds": round_count,
            "workspace_changes": workspace_changes,
            "was_cancelled": was_cancelled,
            "skills_created": skills_created,
            "skills_updated": skills_updated,
            "sub_agents_spawned": sub_agents_spawned,
            "model_switches": model_switches,
        }

    # ---------------------------------------------------------------- legacy methods (backward compatibility)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        n_predict: int = 1024,
        streaming_callback: Optional[Callable] = None,
        **extra,
    ) -> str:
        """Direct (non-agentic) generation from this agent's binding."""
        kwargs: Dict[str, Any] = {
            **self.model_params,
            "temperature": temperature,
            "n_predict": min(n_predict, self.max_tokens_per_turn),
            **extra,
        }
        if streaming_callback:
            kwargs["streaming_callback"] = streaming_callback

        full_prompt = ""
        if system_prompt:
            full_prompt = f"!@>system:\n{system_prompt}\n!@>user:\n{prompt}\n!@>assistant:\n"
        else:
            full_prompt = prompt

        try:
            return self.lc.generate_text(full_prompt, **kwargs) or ""
        except Exception as e:
            trace_exception(e)
            return f"[{self.display_name} generation error: {e}]"

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, str],
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Optional[Dict]:
        """Structured JSON generation."""
        try:
            return self.lc.generate_structured_content(
                prompt=prompt,
                schema=schema,
                system_prompt=system_prompt or self.system_prompt,
                temperature=temperature,
            )
        except Exception as e:
            trace_exception(e)
            return None

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Union[str, Path, Dict[str, Any]]],
        system_prompt: str = "",
        temperature: float = 0.7,
        n_predict: int = 4096,
        max_tool_rounds: int = 10,
        streaming_callback: Optional[Callable] = None,
        auto_execute: bool = True,
        **extra,
    ) -> Dict[str, Any]:
        """
        Generate a response with access to tools.
        Delegates to the enhanced chat() method for the full agentic loop.
        """
        if not auto_execute:
            return self._generate_with_tools_manual(prompt, tools, system_prompt, temperature, n_predict, max_tool_rounds, streaming_callback, **extra)

        explicit_tools: Dict[str, Dict[str, Any]] = {}
        file_based_tools: List[Union[str, Path]] = []

        for src in tools:
            if isinstance(src, dict):
                name = src.get("name", src.get("function", {}).get("name", "unknown"))
                explicit_tools[name] = src
            else:
                file_based_tools.append(src)

        original_sys_prompt = None
        if system_prompt:
            original_sys_prompt = getattr(self.personality, 'system_prompt', None)
            if hasattr(self.personality, 'system_prompt'):
                self.personality.system_prompt = system_prompt

        try:
            result = self.chat(
                prompt=prompt,
                streaming_callback=streaming_callback,
                tools=explicit_tools if explicit_tools else None,
                tool_files=file_based_tools if file_based_tools else None,
                max_reasoning_steps=max_tool_rounds,
                temperature=temperature,
                n_predict=n_predict,
                use_internal_history=False,
                **extra
            )
        finally:
            if original_sys_prompt is not None and hasattr(self.personality, 'system_prompt'):
                self.personality.system_prompt = original_sys_prompt

        return {
            "response": result["response"],
            "tool_calls": result["tool_calls"],
            "tool_results": result["tool_results"],
            "rounds": result["rounds"],
        }

    def _generate_with_tools_manual(
        self,
        prompt: str,
        tools: List[Union[str, Path, Dict[str, Any]]],
        system_prompt: str,
        temperature: float,
        n_predict: int,
        max_tool_rounds: int,
        streaming_callback: Optional[Callable],
        **extra,
    ) -> Dict[str, Any]:
        """Manual mode: returns the first tool call without executing it."""
        tools_mgr = ToolsManager()
        inline_tools = tools_mgr.build_inline_tools_dict(tools)

        tool_descriptions: List[str] = []
        for name, spec in inline_tools.items():
            params = spec.get("parameters", [])
            param_str = ", ".join(f"{p['name']}: {p['type']}" + (" (optional)" if p.get("optional") else "") for p in params)
            desc = spec.get("description", f"Execute {name}")
            tool_descriptions.append(f"- {name}({param_str}): {desc}")

        tool_header = (
            "=== TOOL USE — MANDATORY FORMAT ===\n"
            "You have external tools. To use one you MUST use EXACTLY this format:\n"
            "<tool>{\"name\": \"tool_name\", \"parameters\": {\"key\": \"value\"}}</tool>\n\n"
            "CRITICAL RULES:\n"
            "1. The ENTIRE tool call must be wrapped in <tool> tags.\n"
            "2. NO markdown code fences.\n"
            "3. NO raw JSON without the XML wrapper.\n"
            "4. ONLY output the <tool> line when calling a tool.\n"
            "5. One tool call per response turn.\n"
            "=== END TOOL USE RULES ===\n\n"
            "TOOLS AVAILABLE:\n"
        )

        tool_block = tool_header + "\n".join(tool_descriptions)

        full_system = (system_prompt or self.system_prompt).rstrip()
        if full_system:
            full_system += "\n\n"
        full_system += tool_block

        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        kwargs: Dict[str, Any] = {
            **self.model_params,
            "temperature": temperature,
            "n_predict": min(n_predict, self.max_tokens_per_turn),
            **extra,
        }
        if streaming_callback:
            kwargs["streaming_callback"] = streaming_callback

        try:
            raw_response = self.lc.generate_from_messages(messages=conversation, **kwargs)
        except Exception as e:
            ASCIIColors.error(f"generate_with_tools (manual): generation failed: {e}")
            return {"response": f"[Error during generation: {e}]", "tool_calls": [], "tool_results": [], "rounds": 0}

        if not isinstance(raw_response, str):
            raw_response = str(raw_response) if raw_response is not None else ""

        tool_call_pattern = re.compile(r'<tool>(.*?)</tool>', re.DOTALL | re.IGNORECASE)
        matches = list(tool_call_pattern.finditer(raw_response))

        if not matches:
            cleaned = tool_call_pattern.sub('', raw_response).strip()
            return {"response": cleaned, "tool_calls": [], "tool_results": [], "rounds": 1}

        match = matches[0]
        tool_json_str = match.group(1).strip()
        visible_response = raw_response[:match.start()].strip()

        try:
            call_data = json.loads(tool_json_str)
        except json.JSONDecodeError as e:
            ASCIIColors.warning(f"Failed to parse tool call JSON: {e}")
            return {"response": visible_response, "tool_calls": [{"round": 1, "name": "unknown", "parameters": {}, "raw": tool_json_str}], "tool_results": [], "pending_tool": {"round": 1, "name": "unknown", "parameters": {}, "raw": tool_json_str}, "rounds": 1}

        tool_name = call_data.get("name", "")
        tool_params = call_data.get("parameters", {})

        call_record = {"round": 1, "name": tool_name, "parameters": tool_params, "raw": tool_json_str}

        return {"response": visible_response, "tool_calls": [call_record], "tool_results": [], "pending_tool": call_record, "rounds": 1}

    def generate_with_tools_sync(
        self,
        prompt: str,
        tools: List[Union[str, Path, Dict[str, Any]]],
        **kwargs,
    ) -> str:
        """Synchronous convenience wrapper that returns only the final text response."""
        result = self.generate_with_tools(prompt, tools, **kwargs)
        return result.get("response", "")

    # ---------------------------------------------------------------- utility

    def clear_conversation(self):
        """Clears the agent's internal conversation history."""
        self._conversation.clear()

    def get_conversation(self) -> List[Dict[str, str]]:
        """Returns the agent's internal conversation history."""
        return list(self._conversation)

    def set_conversation(self, conversation: List[Dict[str, str]]):
        """Sets the agent's internal conversation history."""
        self._conversation = list(conversation)
