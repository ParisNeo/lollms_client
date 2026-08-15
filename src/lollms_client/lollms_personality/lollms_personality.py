# lollms_client/lollms_personality.py
#
# Design contract (relied upon by _mixin_chat.py — no guards needed there):
#
#   personality.name            str  — never None/empty
#   personality.system_prompt   str  — never None
#   personality.tools           _NullToolBinding | LollmsToolBinding — never None
#   personality.tool_specs()    Dict[str, spec]  — always a dict, never raises
#   personality.query_data(q)   normalised RAG dict — never raises
#   personality.has_data        bool
#   bool(personality)           False for NullPersonality, True otherwise

from __future__ import annotations

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import traceback
from pathlib import Path
from types import  SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union

from ascii_colors import ASCIIColors, trace_exception

from .skills_manager import SkillsManager
from .handbag import Handbag
from .lollms_agent_state import _AgentStreamState, _sanitize_tool_result, _ToolsManager


try:
    from lollms_client.lollms_types import MSG_TYPE, EventMode
except ImportError:
    class MSG_TYPE:
        MSG_TYPE_CHUNK = "chunk"
        MSG_TYPE_INFO = "info"
        MSG_TYPE_NEW_MESSAGE = "new_message"
        MSG_TYPE_THOUGHT_CHUNK = "thought"
        MSG_TYPE_TOOL_START = 50
        MSG_TYPE_TOOL_END = 51
        MSG_TYPE_ARTEFACT_BUILD_START = 52
        MSG_TYPE_ARTEFACT_BUILD_END = 53
        MSG_TYPE_CONTEXT_UPDATE = 54

    class EventMode:
        PROCESSING_TAG_MODE = 0
        FULL_CALLBACK_MODE = 1
        MIXED_MODE = 2
        SILENT_MODE = 3

from lollms_client.lollms_memory import FailureMemory
from lollms_client.lollms_agent.lollms_agent import CapabilityFlags, SubAgentSpawner, ModelSwitcher, BindingToolsBuilder, _build_workspace_context, _normalize_messages, _get_builtin_workspace_tools, _IGNORED_WS_DIRS, _IGNORED_WS_EXTS, _TEXT_EXTS
from lollms_client.lollms_artefact import ArtefactVisibility, ArtefactManager


_TEXT_RAG_EXTS = {
    ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml", ".html",
    ".py", ".js", ".ts", ".rs", ".go", ".rb", ".php", ".java", ".kt",
    ".swift", ".c", ".cpp", ".h", ".hpp", ".sql", ".sh", ".bash",
    ".ps1", ".bat", ".toml", ".ini", ".cfg", ".log", ".rdf", ".ttl",
}

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "and", "or", "but",
    "not", "no", "if", "then", "so", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "this", "that", "these", "those", "what", "which",
    "who", "whom", "whose", "how", "when", "where", "why", "all", "each",
    "every", "some", "any", "many", "much", "more", "most", "other",
    "such", "only", "own", "same", "than", "too", "very", "just", "now",
}

# ---------------------------------------------------------------------------
# Personality Bundle Importer
# ---------------------------------------------------------------------------

class PersonalityBundle:
    """
    Imports and exports personality bundles from/to structured folders.

    A personality bundle is a folder with the snake_case name of the agent.
    It contains a SOUL.md file (Hugging Face model card format) and optional
    folders for tools, skills, assets, and knowledge.
    """

    @staticmethod
    def parse_soul_md(soul_content: str) -> tuple[dict, str]:
        """
        Parses a SOUL.md file into (metadata_dict, system_prompt_str).
        Handles YAML frontmatter without requiring a full YAML parser.
        """
        metadata = {}
        prompt = soul_content

        if soul_content.strip().startswith("---"):
            parts = soul_content.split("---", 2)
            if len(parts) >= 3:
                yaml_block = parts[1].strip()
                prompt = parts[2].strip()

                for line in yaml_block.splitlines():
                    if ":" not in line:
                        continue
                    key, _, value = line.partition(":")
                    key = key.strip().lower()
                    value = value.strip().strip("'\"")
                    if value:
                        metadata[key] = value

        return metadata, prompt

    @staticmethod
    def export_bundle(personality: 'LollmsPersonality', output_dir: Union[str, Path]) -> Path:
        """
        Exports a LollmsPersonality to a structured folder bundle.
        """
        bundle_dir = Path(output_dir) / personality.name.lower().replace(" ", "_")
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write SOUL.md
        soul_path = bundle_dir / "SOUL.md"
        meta = {
            "name": personality.name,
            "author": personality.author,
            "version": "1.0",
            "category": personality.category,
            "description": personality.description
        }
        if hasattr(personality, 'temperature') and personality.temperature is not None:
            meta["temperature"] = str(personality.temperature)

        yaml_lines = [f"{k}: {v}" for k, v in meta.items()]
        soul_content = f"---\n{chr(10).join(yaml_lines)}\n---\n\n{personality.system_prompt}"
        soul_path.write_text(soul_content, encoding="utf-8")

        # 2. Export Tools (if any)
        if hasattr(personality, '_exported_tool_paths') and personality._exported_tool_paths:
            tools_dir = bundle_dir / "tools"
            tools_dir.mkdir(exist_ok=True)
            for tool_path in personality._exported_tool_paths:
                src_path = Path(tool_path)
                if src_path.exists():
                    dest_path = tools_dir / src_path.name
                    dest_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")

        # 3. Export Skills (if any)
        if hasattr(personality, '_exported_skills') and personality._exported_skills:
            skills_dir = bundle_dir / "skills"
            skills_dir.mkdir(exist_ok=True)
            for skill_name, skill_content in personality._exported_skills.items():
                skill_dir = skills_dir / skill_name
                skill_dir.mkdir(exist_ok=True)
                (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")

        return bundle_dir

    @staticmethod
    def import_bundle(
        bundle_path: Union[str, Path],
        lollms_client: Optional[Any] = None
    ) -> 'LollmsPersonality':
        """
        Imports a personality bundle from a folder.

        Args:
            bundle_path: Path to the personality folder.
            lollms_client: Optional LollmsClient instance for RAG initialization.

        Returns:
            A configured LollmsPersonality instance.
        """
        bundle_dir = Path(bundle_path)
        if not bundle_dir.is_dir():
            raise FileNotFoundError(f"Personality bundle not found: {bundle_dir}")

        soul_path = bundle_dir / "SOUL.md"
        if not soul_path.exists():
            raise FileNotFoundError(f"SOUL.md not found in bundle: {bundle_dir}")

        # 1. Parse SOUL.md
        soul_content = soul_path.read_text(encoding="utf-8", errors="ignore")
        metadata, system_prompt = PersonalityBundle.parse_soul_md(soul_content)

        name = metadata.get("name", bundle_dir.name.replace("_", " ").title())
        author = metadata.get("author", "Unknown")
        category = metadata.get("category", "general")
        description = metadata.get("description", "")
        temperature = float(metadata["temperature"]) if "temperature" in metadata else None

        # 2. Load Tools
        tools_dir = bundle_dir / "tools"
        tool_binding = None
        exported_tool_paths = []

        if tools_dir.exists():
            try:
                from lollms_client.tools_bindings.lcp import LCPBinding
                tool_binding = LCPBinding(
                    tools_folders=[str(tools_dir)],
                    tool_files=[]
                )

                for item in tools_dir.iterdir():
                    if item.is_file() and item.suffix == ".py":
                        exported_tool_paths.append(str(item))
                    elif item.is_dir():
                        tool_file = item / "TOOL.py"
                        if tool_file.exists():
                            exported_tool_paths.append(str(tool_file))
            except Exception as e:
                ASCIIColors.warning(f"[PersonalityBundle] Failed to load tools: {e}")

        # 3. Load Skills
        skills_dir = bundle_dir / "skills"
        skills_context = ""
        exported_skills = {}

        if skills_dir.exists():
            skill_parts = []
            for skill_dir in skills_dir.iterdir():
                if skill_dir.is_dir():
                    skill_md = skill_dir / "SKILL.md"
                    if skill_md.exists():
                        content = skill_md.read_text(encoding="utf-8", errors="ignore")
                        exported_skills[skill_dir.name] = content
                        skill_parts.append(f"### Skill: {skill_dir.name}\n{content}")
            if skill_parts:
                skills_context = "\n\n".join(skill_parts)

        # 4. Load Assets
        assets_dir = bundle_dir / "assets"
        icon_path = None
        voice_path = None

        if assets_dir.exists():
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                p = assets_dir / f"logo{ext}"
                if p.exists():
                    icon_path = str(p)
                    break
            for ext in [".wav", ".mp3"]:
                p = assets_dir / f"voice{ext}"
                if p.exists():
                    voice_path = str(p)
                    break

        # 5. Load Knowledge (RAG)
        knowledge_dir = bundle_dir / "knowledge"
        data_source_fn = None

        if knowledge_dir.exists() and lollms_client is not None:
            try:
                import pipmaster as pm
                pm.ensure_installed("safestore")

                from safestore.safestore import Safestore
                from safestore.core.database import Database

                db_path = knowledge_dir / "knowledge.db"
                if db_path.exists():
                    store = Safestore(db_path=str(db_path))
                    store.load()

                    def _rag_query(query: str) -> Dict[str, Any]:
                        try:
                            results = store.search(query, top_k=3)
                            sources = []
                            for r in results:
                                sources.append({
                                    "content": r.get("text", ""),
                                    "score": float(r.get("score", 1.0)),
                                    "source": "knowledge_base"
                                })
                            return {
                                "success": True,
                                "sources": sources,
                                "count": len(sources),
                                "query": query
                            }
                        except Exception as e:
                            return {
                                "success": False,
                                "sources": [],
                                "count": 0,
                                "query": query,
                                "error": str(e)
                            }

                    data_source_fn = _rag_query
            except ImportError:
                ASCIIColors.warning("[PersonalityBundle] safestore not installed. RAG disabled.")
            except Exception as e:
                ASCIIColors.warning(f"[PersonalityBundle] RAG initialization failed: {e}")

        # 6. Augment system prompt with skills context
        final_system_prompt = system_prompt
        if skills_context:
            final_system_prompt += f"\n\n=== ACTIVE SKILLS ===\n{skills_context}\n=== END SKILLS ==="

        # 7. Create Personality
        personality = LollmsPersonality(
            name=name,
            author=author,
            category=category,
            description=description,
            system_prompt=final_system_prompt,
            icon=icon_path,
            tools=tool_binding,
            data_source=data_source_fn
        )

        # Attach metadata for export and temperature
        personality.temperature = temperature
        personality._exported_tool_paths = exported_tool_paths
        personality._exported_skills = exported_skills
        personality.voice_path = voice_path

        return personality


# ---------------------------------------------------------------------------
# Null tool binding  (returned when no real binding is configured)
# ---------------------------------------------------------------------------

class _NullToolBinding:
    """
    Drop-in no-op for LollmsToolBinding.
    ``to_chat_tool_specs()`` always returns ``{}`` so callers need no guards.
    """
    binding_name: str = "null"

    def discover_tools(self, **_) -> List[Dict[str, Any]]:
        return []

    def list_tools(self, **_) -> List[Dict[str, Any]]:
        return []

    def execute_tool(self, tool_name: str, params: Dict[str, Any], **_) -> Dict[str, Any]:
        return {"error": "No tool binding configured.", "success": False}

    def to_chat_tool_specs(self, **_) -> Dict[str, Dict[str, Any]]:
        return {}

    def __bool__(self) -> bool:
        return False

    def __len__(self) -> int:
        return 0


_NULL_TOOL_BINDING = _NullToolBinding()


# ---------------------------------------------------------------------------
# LollmsPersonality
# ---------------------------------------------------------------------------

class LollmsPersonality:
    """
    The universal execution unit. Scales from a simple system prompt to a 
    fully-armed, stateful, multi-persona ecosystem.
    """

    def __init__(
        self,
        name: str,
        author: str,
        category: str,
        description: str,
        system_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        icon: Optional[str] = None,
        tools: Optional[Any] = None,
        data_source: Optional[Union[str, Callable[[str], Any]]] = None,
        data_files: Optional[List[Union[str, Path]]] = None,
        vectorize_chunk_callback: Optional[Callable[[str, str], None]] = None,
        is_vectorized_callback: Optional[Callable[[str], bool]] = None,
        query_rag_callback: Optional[Callable[[str], Any]] = None,
        script: Optional[str] = None,
        personality_id: Optional[str] = None,
        handbag_path: Optional[Union[str, Path]] = None,
        skills_manager: Optional[SkillsManager] = None,
        memory_manager: Optional[Any] = None,
        workspace_path: Optional[Union[str, Path]] = None,
        enable_git_management: bool = False,
        lollms_client: Optional[Any] = None,
        capabilities: Optional[Any] = None,
        max_tokens_per_turn: int = 4096,
    ):
        self.name = name or "assistant"
        self.author = author or ""
        self.category = category or "general"
        self.description = description or ""
        self.system_prompt = system_prompt or ""
        self.metadata = metadata or {}
        self.icon = icon
        self.personality_id = personality_id or self._generate_id()

        self.mcp_tool_names: List[str] = []
        self._tool_binding: Any = _NULL_TOOL_BINDING
        self._has_explicit_allowlist: bool = False
        self._init_tools(tools)

        self._raw_data_source = data_source
        self.data_files = [Path(f) for f in (data_files or [])]
        self.vectorize_chunk_callback = vectorize_chunk_callback
        self.is_vectorized_callback = is_vectorized_callback
        self.query_rag_callback = query_rag_callback
        self._query_data_fn = self._build_query_data_fn(data_source)

        self.script = script
        self.script_module = None
        self._prepare_script()

        # Unified Stateful Components
        self.handbag_path = Path(handbag_path) if handbag_path else None
        self.skills_manager = skills_manager
        self.memory_manager = memory_manager
        self._workspace_path: Optional[Path] = None
        self.workspace_path = Path(workspace_path) if workspace_path else None
        self.enable_git_management = enable_git_management
        self.coworkers: Dict[str, 'LollmsPersonality'] = {}

        self.lollms_client = lollms_client
        self.max_tokens_per_turn = max_tokens_per_turn

        # Agent-like capabilities
        if CapabilityFlags is not None:
            self.capabilities = capabilities if capabilities is not None else CapabilityFlags()
        else:
            self.capabilities = None

        self._conversation: List[Dict[str, str]] = []
        self._failure_memory = FailureMemory() if FailureMemory else SimpleNamespace(failures=[], _signatures=set())

        # Initialize workspace
        if self.workspace_path:
            self.workspace_path.mkdir(parents=True, exist_ok=True)
            self._resolved_workspace = Path(self.workspace_path).resolve()
        else:
            self._resolved_workspace = None

        # INSTRUMENTATION: Debug mode flag for context dumping
        self.debug_mode: bool = False

        self._workspace_path: Optional[Path] = None

        # Initialize SubAgentSpawner and ModelSwitcher if client is provided
        if SubAgentSpawner and ModelSwitcher and self.lollms_client:
            self._sub_agent_spawner = SubAgentSpawner(
                parent_agent=self,
                max_depth=self.capabilities.max_sub_agent_depth if self.capabilities else 3,
                max_per_turn=self.capabilities.max_sub_agents_per_turn if self.capabilities else 5
            )
            self._model_switcher = ModelSwitcher(self.lollms_client)
        else:
            self._sub_agent_spawner = None
            self._model_switcher = None

        self.ensure_data_vectorized()

    @staticmethod
    def from_handbag(path: Union[str, Path], lollms_client: Optional[Any] = None) -> 'LollmsPersonality':
        """Factory to construct a personality from a Handbag folder."""
        hb = Handbag(path)

        # Parse SOUL.md
        soul_content = hb.soul_path.read_text(encoding="utf-8", errors="ignore") if hb.soul_path.exists() else ""
        meta, sys_prompt = PersonalityBundle.parse_soul_md(soul_content)

        name = meta.get("name", hb.path.name.replace("_", " ").title())
        author = meta.get("author", "Unknown")
        category = meta.get("category", "general")
        description = meta.get("description", "")

        # Initialize Skills
        sm = SkillsManager(skills_dirs=hb.skills_dirs) if hb.skills_dirs else None

        # Initialize Memory (Independent Life)
        mm = hb.create_memory_manager()

        # Load Tools
        tool_binding = None
        if hb.tool_files:
            try:
                from lollms_client.tools_bindings.lcp import LCPBinding
                tool_binding = LCPBinding(tool_files=[str(f) for f in hb.tool_files])
            except Exception:
                pass

        pers = LollmsPersonality(
            name=name,
            author=author,
            category=category,
            description=description,
            system_prompt=sys_prompt,
            metadata=meta,
            tools=tool_binding,
            skills_manager=sm,
            memory_manager=mm,
            handbag_path=hb.path,
            lollms_client=lollms_client,
        )

        # Parse Coworkers (Crew Handbag)
        if hb.coworkers_dir.exists():
            for item in sorted(hb.coworkers_dir.iterdir()):
                if item.is_dir() and (item / "SOUL.md").exists():
                    coworker = LollmsPersonality.from_handbag(item, lollms_client=lollms_client)
                    pers.coworkers[coworker.name.lower()] = coworker

        return pers

    # ------------------------------------------------------------------ tools

    def _init_tools(self, tools: Optional[Any]) -> None:
        if tools is None:
            self._tool_binding = _NULL_TOOL_BINDING
            self._has_explicit_allowlist = False
            return

        if _is_tool_binding(tools):
            self._tool_binding = tools
            self._has_explicit_allowlist = False
            return

        if isinstance(tools, list):
            self.mcp_tool_names = [str(t) for t in tools if t]
            self._tool_binding  = _NULL_TOOL_BINDING
            self._has_explicit_allowlist = True
            return

        ASCIIColors.warning(
            f"[{self.name}] Unsupported tools type {type(tools).__name__!r}. "
            "Expected LollmsToolBinding or List[str]. Falling back to null binding."
        )
        self._tool_binding = _NULL_TOOL_BINDING
        self._has_explicit_allowlist = False

    @property
    def tools(self) -> Any:
        return self._tool_binding

    @tools.setter
    def tools(self, value: Optional[Any]) -> None:
        self._init_tools(value)

    def attach_tool_binding(self, binding: Any) -> None:
        if not _is_tool_binding(binding):
            raise TypeError(
                f"attach_tool_binding expects a LollmsToolBinding, "
                f"got {type(binding).__name__!r}"
            )
        self._tool_binding = binding
        ASCIIColors.info(
            f"[{self.name}] Tool binding attached: {binding.binding_name!r}"
        )

    def tool_specs(self, client_binding=None, **discover_kwargs) -> Dict[str, Dict[str, Any]]:
        if self._has_explicit_allowlist and not self.mcp_tool_names:
            return {}

        binding = client_binding or self._tool_binding
        if not binding:
            return {}

        try:
            all_specs = binding.to_chat_tool_specs(**discover_kwargs)
        except Exception as exc:
            trace_exception(exc)
            return {}

        if not self._has_explicit_allowlist:
            return all_specs

        allowed = set(self.mcp_tool_names)
        filtered = {
            name: spec
            for name, spec in all_specs.items()
            if name in allowed
        }

        missing = allowed - set(all_specs.keys())
        if missing:
            ASCIIColors.warning(
                f"[{self.name}] The following tools are in the allowlist but were "
                f"not found in the binding: {sorted(missing)}"
            )

        return filtered

    # ------------------------------------------------------------------ data

    def _build_query_data_fn(
        self, source: Optional[Union[str, Callable]]
    ) -> Callable[[str], Dict[str, Any]]:
        def _empty(query: str) -> Dict[str, Any]:
            return {"success": False, "sources": [], "count": 0, "query": query}

        def _normalise_raw(raw: Any, query: str, source_label: str) -> Dict[str, Any]:
            if isinstance(raw, dict) and "sources" in raw:
                if "success" not in raw:
                    raw["success"] = True
                raw.setdefault("query", query)
                raw.setdefault("count", len(raw["sources"]))
                return raw

            if isinstance(raw, list):
                sources = []
                for chunk in raw:
                    if isinstance(chunk, dict):
                        sources.append({
                            "content":  chunk.get("content", str(chunk)),
                            "score":    float(chunk.get("score", chunk.get("value", 1.0))),
                            "source":   chunk.get("source", source_label),
                            "metadata": chunk.get("metadata", {}),
                            "title":    chunk.get("title", ""),
                        })
                    else:
                        sources.append({
                            "content": str(chunk), "score": 1.0,
                            "source": source_label, "metadata": {}, "title": "",
                        })
                return {"success": True, "sources": sources,
                        "count": len(sources), "query": query}

            text = str(raw) if raw is not None else ""
            return {
                "success": bool(text),
                "sources": [{"content": text, "score": 1.0,
                             "source": source_label}] if text else [],
                "count":   1 if text else 0,
                "query":   query,
            }

        if isinstance(source, str):
            _static = source
            def _static_fn(query: str) -> Dict[str, Any]:
                return {
                    "success": True,
                    "sources": [{"content": _static, "score": 1.0, "source": "static"}],
                    "count":   1,
                    "query":   query,
                }
            return _static_fn

        if callable(source):
            _callable = source
            def _callable_fn(query: str) -> Dict[str, Any]:
                try:
                    return _normalise_raw(_callable(query), query, "data_source")
                except Exception as exc:
                    trace_exception(exc)
                    return {"success": False, "sources": [], "count": 0,
                            "query": query, "error": str(exc)}
            return _callable_fn

        if self.query_rag_callback is not None:
            _rag_cb = self.query_rag_callback
            def _rag_fn(query: str) -> Dict[str, Any]:
                try:
                    return _normalise_raw(_rag_cb(query), query, "rag")
                except Exception as exc:
                    trace_exception(exc)
                    return {"success": False, "sources": [], "count": 0,
                            "query": query, "error": str(exc)}
            return _rag_fn

        return _empty

    def query_data(self, query: str) -> Dict[str, Any]:
        return self._query_data_fn(query)

    @property
    def has_data(self) -> bool:
        return (
            self._raw_data_source is not None
            or self.query_rag_callback is not None
            or bool(self.data_files)
        )

    @property
    def workspace_path(self) -> Optional[Path]:
        return self._workspace_path

    @workspace_path.setter
    def workspace_path(self, value: Optional[Union[str, Path]]) -> None:
        self._workspace_path = Path(value) if value else None
        if self._workspace_path:
            self._workspace_path.mkdir(parents=True, exist_ok=True)
            object.__setattr__(self, '_resolved_workspace', self._workspace_path.resolve())
        else:
            object.__setattr__(self, '_resolved_workspace', None)

    @property
    def data_source(self) -> Optional[Union[str, Callable]]:
        return self._raw_data_source

    @data_source.setter
    def data_source(self, value: Optional[Union[str, Callable]]) -> None:
        self._raw_data_source = value
        self._query_data_fn   = self._build_query_data_fn(value)

    # ------------------------------------------------------------------ script

    def _prepare_script(self) -> None:
        if not self.script:
            return
        try:
            module_name = f"lollms_personality_script_{self.personality_id}"
            spec        = importlib.util.spec_from_loader(module_name, loader=None)
            module      = importlib.util.module_from_spec(spec)
            exec(compile(self.script, f"<personality:{self.name}>", "exec"),
                 module.__dict__)
            self.script_module = module
            ASCIIColors.success(f"[{self.name}] Custom script loaded successfully.")
        except Exception as exc:
            ASCIIColors.warning(f"[{self.name}] Failed to load custom script: {exc}")
            trace_exception(exc)
            self.script_module = None

    def run_script(self, entry_point: str = "run", **kwargs) -> Any:
        if self.script_module is None:
            return None
        fn = getattr(self.script_module, entry_point, None)
        if fn is None:
            ASCIIColors.warning(
                f"[{self.name}] Script has no '{entry_point}' function."
            )
            return None
        try:
            return fn(**kwargs)
        except Exception as exc:
            ASCIIColors.warning(f"[{self.name}] Script error in '{entry_point}': {exc}")
            trace_exception(exc)
            return None

    # ------------------------------------------------------------------ RAG

    def ensure_data_vectorized(self, chunk_size: int = 1024) -> None:
        if not self.data_files or not self.vectorize_chunk_callback \
                or not self.is_vectorized_callback:
            return

        ASCIIColors.info(f"[{self.name}] Checking RAG data vectorization...")
        all_vectorized = True
        for file_path in self.data_files:
            if not file_path.exists():
                ASCIIColors.warning(
                    f"  - Data file not found, skipping: {file_path}"
                )
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                chunks  = [content[i:i + chunk_size]
                           for i in range(0, len(content), chunk_size)]
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{self.personality_id}_{file_path.name}_chunk_{i}"
                    if not self.is_vectorized_callback(chunk_id):
                        all_vectorized = False
                        ASCIIColors.info(
                            f"  - Vectorizing '{file_path.name}' "
                            f"chunk {i+1}/{len(chunks)}..."
                        )
                        self.vectorize_chunk_callback(chunk, chunk_id)
            except Exception as exc:
                ASCIIColors.warning(
                    f"  - Error processing {file_path.name}: {exc}"
                )

        if all_vectorized:
            ASCIIColors.success(f"[{self.name}] All RAG data already vectorized.")
        else:
            ASCIIColors.success(f"[{self.name}] RAG vectorization complete.")

    def get_rag_context(self, query: str) -> Optional[str]:
        result = self.query_data(query)
        if not result.get("success") or not result.get("sources"):
            return None
        return "\n\n".join(
            s["content"] for s in result["sources"] if s.get("content")
        )

    # ------------------------------------------------------------------ ID / serialisation

    def _generate_id(self) -> str:
        safe_author = "".join(
            c if c.isalnum() else "_" for c in (self.author or "lollms")
        )
        safe_name = "".join(c if c.isalnum() else "_" for c in self.name)
        return f"{safe_author}_{safe_name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "personality_id":       self.personality_id,
            "name":                 self.name,
            "author":               self.author,
            "category":             self.category,
            "description":          self.description,
            "system_prompt":        self.system_prompt,
            "tools":                self.mcp_tool_names,
            "has_explicit_allowlist": self._has_explicit_allowlist,
            "has_tool_binding":     bool(self._tool_binding),
            "has_data_source":      self.has_data,
            "data_files":           [str(p) for p in self.data_files],
            "has_script":           self.script is not None,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], **kwargs
    ) -> "LollmsPersonality":
        tools_list = data.get("tools") or None
        return cls(
            name           = data.get("name", "assistant"),
            author         = data.get("author", ""),
            category       = data.get("category", "general"),
            description    = data.get("description", ""),
            system_prompt  = data.get("system_prompt", ""),
            tools          = tools_list,
            personality_id = data.get("personality_id"),
            **kwargs,
        )

    # ------------------------------------------------------------------ dunder

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}"]
        if bool(self._tool_binding):
            parts.append(f"tools={self._tool_binding.binding_name!r}")
        elif self.mcp_tool_names:
            parts.append(f"mcp_allowlist={self.mcp_tool_names}")
        elif self._has_explicit_allowlist:
            parts.append("mcp_allowlist=[] (no tools)")
        if self.has_data:
            parts.append("has_data=True")
        if self.script_module is not None:
            parts.append("has_script=True")
        return f"LollmsPersonality({', '.join(parts)})"

    def __bool__(self) -> bool:
        return True

    # ------------------------------------------------------------------ Workspace & Sub-Agents

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

    def cancel_generation(self) -> bool:
        if hasattr(self, '_cancel_flag'):
            object.__setattr__(self, '_cancel_flag', True)
        if hasattr(self, 'lollms_client') and self.lollms_client:
            if hasattr(self.lollms_client, 'cancel'):
                try:
                    self.lollms_client.cancel()
                except Exception:
                    pass
            elif hasattr(self.lollms_client, 'llm') and hasattr(self.lollms_client.llm, 'cancel'):
                try:
                    self.lollms_client.llm.cancel()
                except Exception:
                    pass
        return True

    def is_generation_cancelled(self) -> bool:
        return getattr(self, '_cancel_flag', False)

    def _reset_cancel_state(self):
        object.__setattr__(self, '_cancel_flag', False)

    # ------------------------------------------------------------------ Independent Agentic Chat

    def wipe_all_memories(self) -> bool:
        """
        Permanently deletes all episodic and associative memories from the personality's 
        independent memory database. This includes working, deep, and archived memory tiers.
        """
        if not hasattr(self, 'memory_manager') or not self.memory_manager:
            ASCIIColors.warning(f"[{self.name}] No independent memory manager attached. Cannot wipe memories.")
            return False

        try:
            import sqlite3
            db_path = self.memory_manager.db_path.replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = {row[0] for row in cursor.fetchall()}

            if "memories" in existing_tables:
                cursor.execute("DELETE FROM memories")
            if "memory_embeddings" in existing_tables:
                cursor.execute("DELETE FROM memory_embeddings")
            if "memory_decay_history" in existing_tables:
                cursor.execute("DELETE FROM memory_decay_history")

            conn.commit()
            conn.close()

            ASCIIColors.success(f"[{self.name}] ✅ All independent memories wiped successfully.")
            return True
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"[{self.name}] Failed to wipe memories: {e}")
            return False

    def _build_workspace_context_block(self) -> str:
        import time as _time
        current_time = _time.time()
        if hasattr(self, '_last_ws_sync_time') and (current_time - self._last_ws_sync_time < 5.0):
            if getattr(self, '_artefact_manager', None):
                try:
                    zone = self._artefact_manager.build_artefacts_context_zone()
                    if zone:
                        return "\n" + zone
                    if self._resolved_workspace:
                        return "\n" + _build_workspace_context(self._resolved_workspace)
                except Exception:
                    pass
            elif self._resolved_workspace:
                return "\n" + _build_workspace_context(self._resolved_workspace)
            return ""

        object.__setattr__(self, '_last_ws_sync_time', current_time)

        if getattr(self, '_artefact_manager', None):
            try:
                self._sync_artefact_index_with_disk()
                zone = self._artefact_manager.build_artefacts_context_zone()
                if zone:
                    return "\n" + zone
                # Fallback if artefact manager returns empty string but workspace exists
                if self._resolved_workspace:
                    return "\n" + _build_workspace_context(self._resolved_workspace)
            except Exception as e:
                ASCIIColors.warning(f"[{self.name}] Failed to build workspace context: {e}")
                if self._resolved_workspace:
                    return "\n" + _build_workspace_context(self._resolved_workspace)
        elif self._resolved_workspace:
            return "\n" + _build_workspace_context(self._resolved_workspace)
        return ""
    
    
    def _sync_artefact_index_with_disk(self):
        if not hasattr(self, '_artefact_manager') or not self._artefact_manager:
            return

        if not hasattr(self, '_discussion') or not self._discussion:
            return

        try:
            import sqlite3 as _sqlite3
            import hashlib as _hashlib
            
            if not hasattr(self, '_state_db_path'):
                ASCIIColors.warning(f"[{self.name}] State DB path missing, cannot delta sync.")
                return

            ws_path = self._resolved_workspace
            if not ws_path or not ws_path.exists():
                return

            conn = _sqlite3.connect(str(self._state_db_path))
            cursor = conn.cursor()
            
            try:
                cursor.execute("ALTER TABLE file_states ADD COLUMN hash TEXT")
            except _sqlite3.OperationalError:
                pass 

            cursor.execute("SELECT title, visibility, hash FROM file_states")
            db_rows = cursor.fetchall()
            db_files = {row[0]: {"visibility": row[1], "hash": row[2]} for row in db_rows}
            conn.close()

            current_files = set()
            dirty = False

            _BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".zip", ".tar", ".gz", ".pdf", ".docx", ".pptx", ".mp3", ".wav", ".mp4"}

            def _is_noise_directory(dir_name: str) -> bool:
                _COLLAPSED_DIRS = {"data_workspace", "env", ".venv", "venv", ".git", ".idea", ".vscode", "node_modules", ".lollms", "build", "dist", ".next", "__pycache__", ".lollms_metadata", ".lollms_code", "egg-info", "dist-info", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", "site-packages", "artefacts_metadata", "discussions"}
                lower_name = dir_name.lower()
                if lower_name in _COLLAPSED_DIRS: return True
                if lower_name.endswith(".egg-info") or lower_name.endswith(".dist-info"): return True
                return False

            def _scan_dir(directory: Path):
                nonlocal dirty
                try:
                    for item in sorted(directory.iterdir()):
                        if item.is_dir():
                            if _is_noise_directory(item.name):
                                continue
                            _scan_dir(item)
                        elif item.is_file():
                            if item.suffix.lower() in _IGNORED_WS_EXTS:
                                continue

                            rel_path_str = str(item.relative_to(ws_path)).replace("\\", "/")
                            current_files.add(rel_path_str)

                            try:
                                stat = item.stat()
                                if stat.st_size < 1024 * 1024:
                                    content = item.read_bytes()
                                    file_hash = _hashlib.md5(content).hexdigest()
                                else:
                                    file_hash = f"{stat.st_mtime}:{stat.st_size}"
                            except Exception:
                                file_hash = None

                            db_info = db_files.get(rel_path_str)
                            in_memory_art = self._artefact_manager.get(rel_path_str)

                            if not db_info:
                                dirty = True
                                if getattr(self, 'debug_mode', False):
                                    ASCIIColors.info(f"[{self.name}] 📄 New file detected: {rel_path_str}")

                                if not in_memory_art:
                                    self._artefact_manager.add(
                                        title=rel_path_str,
                                        artefact_type="code" if item.suffix.lower() in _TEXT_EXTS else "document",
                                        content="",
                                        active=False,
                                        visibility=ArtefactVisibility.TREE_UNLOCKABLE,
                                        skip_disk_sync=True
                                    )

                                conn = _sqlite3.connect(str(self._state_db_path))
                                cur = conn.cursor()
                                cur.execute("INSERT OR REPLACE INTO file_states (title, visibility, hash) VALUES (?, ?, ?)", 
                                            (rel_path_str, ArtefactVisibility.TREE_UNLOCKABLE, file_hash))
                                conn.commit()
                                conn.close()

                            elif db_info.get("hash") != file_hash:
                                dirty = True
                                if getattr(self, 'debug_mode', False):
                                    ASCIIColors.info(f"[{self.name}] ✏️ File changed: {rel_path_str}")

                                art = self._artefact_manager.get(rel_path_str)
                                if art and art.get("visibility") == ArtefactVisibility.FULL:
                                    art["content"] = ""

                                conn = _sqlite3.connect(str(self._state_db_path))
                                cur = conn.cursor()
                                cur.execute("UPDATE file_states SET hash = ? WHERE title = ?", (file_hash, rel_path_str))
                                conn.commit()
                                conn.close()
                            else:
                                if not in_memory_art:
                                    dirty = True
                                    atype = "code" if item.suffix.lower() in _TEXT_EXTS else "document"

                                    db_vis = db_info.get("visibility", ArtefactVisibility.TREE_UNLOCKABLE)
                                    resolved_vis = db_vis

                                    self._artefact_manager.add(
                                        title=rel_path_str,
                                        artefact_type=atype,
                                        content="",
                                        active=(resolved_vis == ArtefactVisibility.FULL),
                                        visibility=resolved_vis,
                                        skip_disk_sync=True
                                    )
                                    if db_vis != resolved_vis:
                                        conn = _sqlite3.connect(str(self._state_db_path))
                                        cur = conn.cursor()
                                        cur.execute("UPDATE file_states SET visibility = ? WHERE title = ?", (resolved_vis, rel_path_str))
                                        conn.commit()
                                        conn.close()
                                else:
                                    if in_memory_art.get("visibility") != db_info.get("visibility"):
                                        dirty = True
                                        in_memory_art["visibility"] = db_info.get("visibility")
                                        in_memory_art["active"] = (db_info.get("visibility") == ArtefactVisibility.FULL)

                except Exception as dir_err:
                    ASCIIColors.warning(f"[{self.name}] Failed to scan directory {directory}: {dir_err}")

            _scan_dir(ws_path)

            deleted_files = set(db_files.keys()) - current_files
            if deleted_files:
                dirty = True
                conn = _sqlite3.connect(str(self._state_db_path))
                cur = conn.cursor()
                for d_file in deleted_files:
                    if not d_file.endswith("::images"):
                        if getattr(self, 'debug_mode', False):
                            ASCIIColors.info(f"[{self.name}] 🗑️ Pruned deleted file: {d_file}")
                        cur.execute("DELETE FROM file_states WHERE title = ?", (d_file,))
                conn.commit()
                conn.close()

            if deleted_files:
                surviving_arts = [
                    art for art in self._artefact_manager._get_all_raw()
                    if art.get("title") not in deleted_files or art.get("title", "").endswith("::images")
                ]
                self._artefact_manager._save_all(surviving_arts)
            elif not self._discussion.metadata.get("_artefacts") or dirty:
                self._artefact_manager._save_all(self._artefact_manager._get_all_raw())

            if not dirty and getattr(self, 'debug_mode', False):
                ASCIIColors.success(f"[{self.name}] ✅ Workspace index is up-to-date (0 changes).")

        except Exception as e:
            if getattr(self, 'debug_mode', False):
                ASCIIColors.warning(f"[{self.name}] Disk sync/prune failed: {e}")          

    def _refresh_workspace_context_in_prompt(self, current_prompt: str, new_ws_block: str) -> str:
        ws_boundary = "=== WORKSPACE CONTEXT BOUNDARY ==="
        boundary_idx = current_prompt.find(ws_boundary)

        if boundary_idx == -1:
            return current_prompt + "\n" + new_ws_block.strip()

        base_prompt = current_prompt[:boundary_idx + len(ws_boundary)]
        return base_prompt + "\n" + new_ws_block.strip()

    def _calculate_context_fill(self, full_system_prompt: str, base_conversation: List[Dict], virtual_history: List, final_response: str = "") -> Dict[str, Any]:
        """Calculates the current context window fill percentage."""
        try:
            if self.lollms_client and hasattr(self.lollms_client, 'get_ctx_size'):
                max_ctx = self.lollms_client.get_ctx_size() or 0
                if max_ctx > 0 and hasattr(self.lollms_client, 'count_tokens'):
                    total_used = self.lollms_client.count_tokens(full_system_prompt)
                    for msg in base_conversation:
                        total_used += self.lollms_client.count_tokens(msg.get("content", ""))
                    for vh in virtual_history:
                        total_used += self.lollms_client.count_tokens(vh.content)
                    total_used += self.lollms_client.count_tokens(final_response)

                    if total_used <= 0:
                        return {"used_tokens": 0, "max_tokens": max_ctx, "fill_percentage": 0.0}

                    return {
                        "used_tokens": total_used,
                        "max_tokens": max_ctx,
                        "fill_percentage": round((total_used / max_ctx) * 100, 1)
                    }
        except Exception:
            pass
        return {"used_tokens": 0, "max_tokens": 0, "fill_percentage": 0.0}

    def _autonomous_memory_consolidation(self, user_prompt: str, ai_response: str):
        """
        Evaluates the conversation turn and extracts high-density architectural facts 
        or user constraints to commit to long-term associative memory.
        Trivial interactions (greetings, simple lookups) are discarded to prevent context bloat.
        """
        if not self.lollms_client or not hasattr(self.memory_manager, 'add'):
            return

        try:
            clean_ai = re.sub(r'<[^>]+>', '', ai_response).strip()
            clean_user = user_prompt.strip()

            if not clean_user or not clean_ai or len(clean_user) < 10 or len(clean_ai) < 10:
                return

            consolidation_prompt = f"""Analyze the following interaction between a User and an AI Engineer.
Determine if a CRITICAL FACT, ARCHITECTURAL RULE, or USER PREFERENCE was established.
Ignore greetings, trivial progress updates, and conversational filler.

User: "{clean_user}"
AI: "{clean_ai}"

If a high-density fact was established, output EXACTLY a JSON object with:
{{"save_memory": true, "content": "The specific fact/rule", "tags": ["relevant", "tags"], "importance": 0.0-1.0}}
If the interaction is trivial, output:
{{"save_memory": false}}

JSON:"""

            reflection = self.lollms_client.generate_text(
                prompt=consolidation_prompt,
                temperature=0.1,
                n_predict=256
            )

            import json as _json
            json_match = re.search(r'\{.*\}', reflection, re.DOTALL)
            if json_match:
                data = _json.loads(json_match.group(0))
                if data.get("save_memory"):
                    self.memory_manager.add(
                        content=data.get("content", ""),
                        importance=float(data.get("importance", 0.8)),
                        tags=data.get("tags", ["architectural", "fact"]),
                        level=2
                    )
                    ASCIIColors.success(f"[{self.name}] 💾 Consolidated high-density memory: {data.get('content', '')[:50]}...")

        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] Memory consolidation failed: {e}")

    def _autonomous_context_cleanup(self, user_prompt: str) -> str:
        """
        Evaluates the active [C] (Fully Loaded) artifacts against the user's prompt.
        Asks the LLM to emit <lock_file> tags for irrelevant files to free up context space
        before the main generation begins.
        """
        from lollms_client.lollms_artefact import ArtefactVisibility

        if not hasattr(self, '_artefact_manager') or not self._artefact_manager:
            return user_prompt

        all_arts = self._artefact_manager._get_all_raw()
        loaded_files = [
            a.get("physical_path") or a.get("title", "")
            for a in all_arts 
            if a.get("visibility") == ArtefactVisibility.FULL and not a.get("title", "").endswith("::images")
        ]

        if not loaded_files:
            return user_prompt

        ASCIIColors.info(f"[{self.name}] 🧹 Triggering autonomous context cleanup for {len(loaded_files)} loaded files...")

        cleanup_prompt = (
            "You are a context window manager. Your goal is to minimize context usage.\n"
            f"The user just asked: \"{user_prompt}\"\n\n"
            f"The following files are currently FULLY LOADED in your context:\n{', '.join(loaded_files)}\n\n"
            "Which of these files are COMPLETELY IRRELEVANT to the user's request?\n"
            "Output ONLY the XML tags to lock the irrelevant files. Do not output any conversational text.\n"
            "Example:\n"
            "<lock_file>irrelevant_file1.py</lock_file>\n"
            "<lock_file>irrelevant_file2.py</lock_file>\n"
        )

        try:
            cleanup_response = self.lollms_client.generate_text(
                prompt=cleanup_prompt,
                temperature=0.1,
                n_predict=512
            )

            if not isinstance(cleanup_response, str) or not cleanup_response.strip():
                return user_prompt

            # Execute any <lock_file> tags found in the response
            import re
            lock_tags = re.findall(r'<lock_file>(.*?)</lock_file>', cleanup_response, re.DOTALL | re.IGNORECASE)

            if lock_tags:
                locked_count = 0
                for body in lock_tags:
                    files_to_lock = [f.strip().replace("\\", "/") for f in re.split(r'[\n,;]+', body) if f.strip()]
                    for f_name in files_to_lock:
                        # Use the same visibility execution logic as the main chat loop
                        result = self._execute_context_visibility("lock_file", f_name)
                        if "✅ Locking" in result:
                            locked_count += 1

                if locked_count > 0:
                    ASCIIColors.success(f"[{self.name}] 🧹 Autonomously locked {locked_count} irrelevant file(s) to free context.")
                else:
                    ASCIIColors.info(f"[{self.name}] 🧹 No files locked (either none were irrelevant or already locked).")
            else:
                ASCIIColors.info(f"[{self.name}] 🧹 LLM decided no files need to be locked.")

        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] Context cleanup generation failed: {e}")

        return user_prompt

    def _calculate_context_telemetry(self, stable_prompt: str, history: List[Dict], ws_ctx: str, virtual_history: List) -> Dict[str, int]:
        """Calculates token consumption per context segment using the LLM client's tokenizer."""
        telemetry = {
            "system_prompt": 0,
            "history": 0,
            "workspace_tree": 0,
            "loaded_contents": 0,
            "virtual_history": 0,
            "total": 0
        }
        if not self.lollms_client or not hasattr(self.lollms_client, 'count_tokens'):
            return telemetry

        try:
            telemetry["system_prompt"] = self.lollms_client.count_tokens(stable_prompt)

            for msg in history:
                telemetry["history"] += self.lollms_client.count_tokens(msg.get("content", ""))

            if ws_ctx:
                telemetry["workspace_tree"] = self.lollms_client.count_tokens(ws_ctx)

                if "## Fully Loaded File Contents [C]" in ws_ctx:
                    parts = ws_ctx.split("## Fully Loaded File Contents [C]", 1)
                    if len(parts) == 2:
                        telemetry["loaded_contents"] = self.lollms_client.count_tokens(parts[1])
                        telemetry["workspace_tree"] = self.lollms_client.count_tokens(parts[0])

            for vh in virtual_history:
                telemetry["virtual_history"] += self.lollms_client.count_tokens(vh.content)

            telemetry["total"] = sum(telemetry.values())
        except Exception:
            pass

        return telemetry

    def _build_telemetry_block(self, telemetry: Dict[str, int]) -> str:
        """Formats the telemetry dictionary into a readable string for the LLM."""
        total = telemetry.get("total", 0)
        if total == 0:
            return ""

        def _fmt(val):
            return f"{val:,}"

        lines = [
            "=== CONTEXT TELEMETRY (LIVE TOKEN BUDGET) ===",
            f"- System Prompt: {_fmt(telemetry.get('system_prompt', 0))} tokens",
            f"- Conversation History: {_fmt(telemetry.get('history', 0))} tokens",
            f"- Workspace Tree: {_fmt(telemetry.get('workspace_tree', 0))} tokens",
            f"- Loaded File Contents [C]: {_fmt(telemetry.get('loaded_contents', 0))} tokens",
            f"- Virtual History (Tools/Actions this turn): {_fmt(telemetry.get('virtual_history', 0))} tokens",
            f"TOTAL CONSUMED: {_fmt(total)} tokens",
            "If 'History' or 'Virtual History' is too high, consider emitting `<refactor_history></refactor_history>` to compress it.",
            "=== END CONTEXT TELEMETRY ==="
        ]
        return "\n".join(lines)

    def _autonomous_history_refactoring(self, base_conversation: List[Dict[str, str]], streaming_callback: Optional[Callable]) -> List[Dict[str, str]]:
        """
        Autonomously summarizes the base_conversation history to free up context space.
        Replaces verbose multi-turn history with a single dense system message.
        """
        if not base_conversation or not self.lollms_client:
            return base_conversation

        if streaming_callback:
            compaction_msg = '\n<processing type="history_refactoring" title="Autonomous History Refactoring">\n* 🧹 Refactoring conversation history to free up context space...\n</processing>\n'
            try:
                streaming_callback(compaction_msg, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            except Exception:
                pass

        history_text = "\n\n".join([f"[{msg.get('role', 'user').upper()}]: {msg.get('content', '')}" for msg in base_conversation])

        summary_prompt = (
            "You are a history refactoring engine. Summarize the following conversation history into a dense, factual summary.\n"
            "Focus on retaining: user goals, key decisions, file names created/modified, and final conclusions.\n"
            "Discard: conversational pleasantries, intermediate reasoning steps, and verbose tool outputs.\n"
            "Output ONLY the summary paragraph, no conversational filler.\n\n"
            f"=== HISTORY TO COMPACT ===\n{history_text}\n=== END HISTORY ==="
        )

        try:
            summary = self.lollms_client.generate_text(
                prompt=summary_prompt,
                temperature=0.1,
                n_predict=1024
            )
            if not isinstance(summary, str) or not summary.strip():
                return base_conversation

            compacted_history = [{
                "role": "system",
                "content": f"[SYSTEM: AUTONOMOUS HISTORY REFACTORING]\nThe previous conversation has been summarized to save context. Use this summary as your working history:\n\n{summary.strip()}"
            }]

            if streaming_callback:
                success_msg = f'\n<processing type="history_refactoring" title="Autonomous History Refactoring">\n* ✅ History refactored successfully. Context freed.\n</processing>\n'
                try:
                    streaming_callback(success_msg, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                except Exception:
                    pass

            return compacted_history
        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] History refactoring failed: {e}")
            return base_conversation

    def _compact_virtual_history(self, virtual_history: List, streaming_callback: Optional[Callable]) -> List:
        """
        Autonomously summarizes the virtual history to free up context space.
        Replaces verbose tool outputs and intermediate reasoning with a dense summary.
        """
        if not virtual_history or not self.lollms_client:
            return virtual_history

        # Notify the UI of the autonomous compaction
        if streaming_callback:
            compaction_msg = '\n<processing type="context_compaction" title="Autonomous Context Compaction">\n* 🧹 Context window approaching limit. Summarizing history to free up space...\n</processing>\n'
            try:
                streaming_callback(compaction_msg, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            except Exception:
                pass

        # Build a summarization prompt from the virtual history
        history_text = "\n\n".join([f"[{vh.sender_type}]: {vh.content}" for vh in virtual_history])

        summary_prompt = (
            "You are a context compaction engine. Summarize the following conversation history into a dense, factual summary.\n"
            "Focus on retaining: user goals, key data retrieved from tools, file names created/modified, and final conclusions.\n"
            "Discard: conversational pleasantries, intermediate reasoning steps, and verbose tool outputs.\n\n"
            f"=== HISTORY TO COMPACT ===\n{history_text}\n=== END HISTORY ==="
        )

        try:
            # Use a low temperature for deterministic, factual summarization
            summary = self.lollms_client.generate_text(
                prompt=summary_prompt,
                temperature=0.1,
                n_predict=1024
            )
            if not isinstance(summary, str) or not summary.strip():
                return virtual_history

            # Replace the verbose history with a single dense system message
            compacted_history = [SimpleNamespace(
                sender_type="user",
                content=f"[SYSTEM: AUTONOMOUS CONTEXT COMPACTION]\nThe previous history has been summarized to save space. Use this summary as your working context:\n\n{summary.strip()}"
            )]

            if streaming_callback:
                success_msg = f'\n<processing type="context_compaction" title="Autonomous Context Compaction">\n* ✅ History compacted successfully. Context freed.\n</processing>\n'
                try:
                    streaming_callback(success_msg, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                except Exception:
                    pass

            return compacted_history

        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] Context compaction failed: {e}")
            return virtual_history

    def _init_artefact_system(self):
        try:
            from lollms_client.lollms_artefact import ArtefactManager, ArtefactVisibility
            import uuid as _uuid
            import sqlite3 as _sqlite3
            import hashlib as _hashlib

            ws_path = self._resolved_workspace
            if not ws_path:
                return

            # Use .lollms_code for persistent state index
            state_dir = ws_path / ".lollms_code"
            state_dir.mkdir(parents=True, exist_ok=True)
            state_db_path = state_dir / "context_state.db"

            # Initialize SQLite state DB with hash column for delta sync
            conn = _sqlite3.connect(str(state_db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_states (
                    title TEXT PRIMARY KEY,
                    visibility TEXT NOT NULL,
                    hash TEXT
                )
            """)
            # Migration for existing DBs
            try:
                cursor.execute("ALTER TABLE file_states ADD COLUMN hash TEXT")
            except _sqlite3.OperationalError:
                pass
            conn.commit()
            conn.close()

            metadata_dir = state_dir / "artefacts_metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            proxy = SimpleNamespace(
                id=f"pers_{self.personality_id[:8]}",
                workspace_path=str(ws_path),
                workspace_data_path=str(ws_path),
                artefacts_metadata_path=str(metadata_dir),
                lollmsClient=self.lollms_client, 
                metadata={},
                _is_db_backed=False,
                commit=lambda: None,
                disable_artefact_versioning=True,
                _skip_physical_sync=True
            )

            class _IndexOnlyArtefactManager(ArtefactManager):
                def _sync_to_disk_workspace(self, *args, **kwargs):
                    return

            am = _IndexOnlyArtefactManager(proxy)
            object.__setattr__(self, '_artefact_manager', am)
            object.__setattr__(self, '_artefact_proxy', proxy)
            object.__setattr__(self, '_discussion', proxy)
            object.__setattr__(self, '_state_db_path', state_db_path)

            # Delegate the initial population and hash computation to the delta sync engine.
            # This prevents reading/hashing every file on startup if the DB already knows them.
            self._sync_artefact_index_with_disk()

        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] Failed to initialise artefact system: {e}")
            object.__setattr__(self, '_artefact_manager', None)
            object.__setattr__(self, '_artefact_proxy', None)
            
            
            
    def _sanitize_history_for_context(self, text: str) -> str:
        text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<!-- status:[^>]*-->', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</processing>', '', text, flags=re.IGNORECASE)

        def _artifact_anchor(match: re.Match) -> str:
            attrs_str = match.group(0)
            body_content = match.group(1) if match.groups() else ""

            if "<<<<<<< SEARCH" in body_content:
                title_match = re.search(r'(?:name|title)=["\']([^"\']*)["\']', attrs_str, re.IGNORECASE)
                title = title_match.group(1) if title_match else "artifact"
                return f"[🔧 SEARCH/REPLACE attempted on: {title}]\n{body_content}\n[/SEARCH/REPLACE]"

            title_match = re.search(r'(?:name|title)=["\']([^"\']*)["\']', attrs_str, re.IGNORECASE)
            type_match = re.search(r'type=["\']([^"\']*)["\']', attrs_str, re.IGNORECASE)
            title = title_match.group(1) if title_match else "artifact"
            atype = type_match.group(1) if type_match else "code"
            return f"[🔒artefact tag called, content stripped for brievety, do not mimic:{title}|{atype}]"

        text = re.sub(r'<artifact[^>]*>(.*?)</artifact>', _artifact_anchor, text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<artefact[^>]*>(.*?)</artefact>', _artifact_anchor, text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<lollms_artifact[^/]*/>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<artefact_image[^/]*/>', '', text, flags=re.IGNORECASE)

        return text.strip()

    def _discover_tools(self, explicit_tools: Optional[Dict], tool_files: Optional[List]) -> Dict[str, Dict[str, Any]]:
        active_tools = {}

        # Sovereign Tool Opt-In Doctrine: File I/O is strictly handled by the Artefact System (<unlock_file>, <artifact>).
        # We DO NOT mount generic tool_read_file or tool_write_file to prevent context pollution and architectural drift.

        if BindingToolsBuilder and self.lollms_client and self.capabilities:
            binding_tools = BindingToolsBuilder.build_tools(self.lollms_client, self.capabilities, self._resolved_workspace)
            active_tools.update(binding_tools)

        if self.capabilities and self.capabilities.enable_skill_loading:
            if self.skills_manager:
                active_tools.update(self.skills_manager.build_skill_tools())

        if self._sub_agent_spawner and self.capabilities and self.capabilities.enable_sub_agents:
            def tool_spawn_sub_agent(instruction: str, personality_conditioning: str = "", model_name: str = "") -> dict:
                return self._sub_agent_spawner.spawn(instruction=instruction, personality_conditioning=personality_conditioning or None, model_name=model_name or None)
            active_tools["tool_spawn_sub_agent"] = {
                "name": "tool_spawn_sub_agent",
                "description": "Spawn a focused sub-agent to handle a sub-task. The sub-agent shares your workspace but cannot spawn further sub-agents.",
                "parameters": [
                    {"name": "instruction", "type": "str", "description": "The specific task for the sub-agent."},
                    {"name": "personality_conditioning", "type": "str", "description": "Custom system prompt for the sub-agent (optional).", "optional": True},
                    {"name": "model_name", "type": "str", "description": "Specific model to use (empty = parent's model).", "optional": True},
                ],
                "callable": tool_spawn_sub_agent,
            }

        lcp_binding = getattr(self.lollms_client, 'tools', None)
        if lcp_binding is None and (tool_files or self._resolved_workspace):
            try:
                from lollms_client.tools_bindings.lcp import LCPBinding
                pkg_root = Path(lollms_client.__file__).resolve().parent
                default_tools = pkg_root / "tools_bindings" / "lcp" / "default_tools"
                lcp_binding = LCPBinding(tools_folders=[str(default_tools)] if default_tools.exists() else [])
            except Exception:
                lcp_binding = None

        if lcp_binding and hasattr(lcp_binding, 'mount_tool_library'):
            # Auto-mount essential discovery and execution tools
            _ESSENTIAL_LIBRARIES = ["system_shell", "grep_files", "find_files"]
            for lib_name in _ESSENTIAL_LIBRARIES:
                try:
                    lcp_binding.mount_tool_library(lib_name)
                except Exception as e:
                    ASCIIColors.warning(f"[LollmsPersonality] Failed to mount LCP tool library '{lib_name}': {e}")

            try:
                lcp_tools = lcp_binding.to_chat_tool_specs()
                # Expose all tools from the essential libraries (system_shell, grep, find)
                for t_name, t_spec in lcp_tools.items():
                    if t_name.startswith("tool_execute_shell_command") or \
                       t_name.startswith("tool_grep_") or \
                       t_name.startswith("tool_find_"):
                        active_tools[t_name] = t_spec
            except Exception as e:
                ASCIIColors.warning(f"[LollmsPersonality] Failed to extract LCP tool specs: {e}")

        if tool_files:
            try:
                tools_mgr = _ToolsManager()
                file_tools = tools_mgr.build_inline_tools_dict(tool_files)
                active_tools.update(file_tools)
            except Exception:
                pass

        if explicit_tools:
            active_tools.update(explicit_tools)

        return active_tools

    def _init_scratchpad(self):
        """Initializes the persistent scratchpad file in the .lollms_code directory."""
        if not self._resolved_workspace:
            object.__setattr__(self, '_scratchpad_path', None)
            return

        sandbox_dir = self._resolved_workspace / ".lollms_code"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        scratch_path = sandbox_dir / "scratchpad.md"

        if not scratch_path.exists():
            scratch_path.write_text("# Agent Persistent Scratchpad\n\nUse this space to store critical state, file lists, and architectural decisions.\n", encoding="utf-8")

        object.__setattr__(self, '_scratchpad_path', scratch_path)

    def _init_user_profile(self, profile_path: Optional[Path]):
        """Initializes the global user profile manager."""
        if profile_path is None:
            object.__setattr__(self, '_user_profile_path', None)
            object.__setattr__(self, '_user_profile_content', "")
            return

        try:
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            if not profile_path.exists():
                default_content = (
                    "# 👤 Global User Profile\n"
                    "This file contains universal information about the user. It is loaded into the agent's context at the start of every session.\n"
                    "The agent can update this file using `<user_profile_update>` tags.\n"
                    "CRITICAL: Do not store project-specific information here. Use the workspace scratchpad for project state.\n\n"
                    "## Identity\n- Name: \n- Occupation: \n\n"
                    "## Global Constraints & Preferences\n- \n\n"
                    "## Frequently Used Tools & Workflows\n- \n"
                )
                profile_path.write_text(default_content, encoding="utf-8")

            content = profile_path.read_text(encoding="utf-8", errors="ignore")
            object.__setattr__(self, '_user_profile_path', profile_path)
            object.__setattr__(self, '_user_profile_content', content)
        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] Failed to initialize user profile: {e}")
            object.__setattr__(self, '_user_profile_path', None)
            object.__setattr__(self, '_user_profile_content', "")

    def _build_scratchpad_context(self) -> str:
        """Reads the scratchpad content for injection into the dynamic suffix."""
        if not getattr(self, '_scratchpad_path', None) or not self._scratchpad_path.exists():
            return ""

        try:
            content = self._scratchpad_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                return ""
            return f"=== SCRATCHPAD CONTENT ===\n{content}\n=== END SCRATCHPAD ==="
        except Exception:
            return ""

    def _build_user_profile_context(self) -> str:
        """Injects the global user profile into the system prompt."""
        if not getattr(self, '_user_profile_content', ""):
            return ""

        return (
            "\n=== GLOBAL USER PROFILE (IDENTITY & PREFERENCES) ===\n"
            "This is the universal profile of the user. It applies to ALL projects and sessions.\n"
            "If you learn a new universal fact about the user (e.g., their name, a global coding standard they follow), you MUST update this file.\n"
            "To update it, emit: `<user_profile_update>` with Aider SEARCH/REPLACE blocks inside.\n"
            "CRITICAL: Do NOT store project-specific facts (like 'the current project uses FastAPI') here. Use the Scratchpad for project state.\n"
            "=== PROFILE CONTENT ===\n"
            f"{self._user_profile_content}\n"
            "=== END PROFILE ===\n"
        )

    def _execute_scratchpad_clear(self) -> str:
        """Clears the persistent scratchpad back to its default empty state."""
        if not getattr(self, '_scratchpad_path', None):
            return "[SYSTEM ERROR] Scratchpad not initialized."

        try:
            default_content = "# Agent Persistent Scratchpad\n\nUse this space to store critical state, file lists, and architectural decisions.\n"
            self._scratchpad_path.write_text(default_content, encoding="utf-8")
            return "✅ Scratchpad cleared successfully."
        except Exception as e:
            return f"[SYSTEM ERROR] Failed to clear scratchpad: {e}"

    def _execute_user_profile_clear(self) -> str:
        """Clears the global user profile back to its default template."""
        if not getattr(self, '_user_profile_path', None):
            return "[SYSTEM ERROR] User profile not initialized."

        try:
            default_content = (
                "# 👤 Global User Profile\n"
                "This file contains universal information about the user. It is loaded into the agent's context at the start of every session.\n"
                "The agent can update this file using `<user_profile_update>` tags.\n"
                "CRITICAL: Do not store project-specific information here. Use the workspace scratchpad for project state.\n\n"
                "## Identity\n- Name: \n- Occupation: \n\n"
                "## Global Constraints & Preferences\n- \n\n"
                "## Frequently Used Tools & Workflows\n- \n"
            )
            self._user_profile_path.write_text(default_content, encoding="utf-8")
            object.__setattr__(self, '_user_profile_content', default_content)
            return "✅ Global user profile cleared successfully."
        except Exception as e:
            return f"[SYSTEM ERROR] Failed to clear user profile: {e}"

    def _execute_scratchpad_update(self, tag_name: str, body: str) -> str:
        """Executes append or patch operations on the scratchpad file."""
        if not getattr(self, '_scratchpad_path', None):
            return "[SYSTEM ERROR] Scratchpad not initialized."

        try:
            current_content = self._scratchpad_path.read_text(encoding="utf-8", errors="ignore")

            if tag_name == "scratchpad_append":
                new_content = current_content + "\n" + body.strip() + "\n"
                self._scratchpad_path.write_text(new_content, encoding="utf-8")
                return "✅ Content appended to scratchpad successfully."

            elif tag_name == "scratchpad_patch":
                # Reuse the robust Aider patch logic from ArtefactManager
                from lollms_client.lollms_artefact import ArtefactManager
                patched_content = ArtefactManager.apply_aider_patch(current_content, body)
                self._scratchpad_path.write_text(patched_content, encoding="utf-8")
                return "✅ Scratchpad patched successfully."

            return "[SYSTEM ERROR] Unknown scratchpad operation."
        except Exception as e:
            return f"[SYSTEM ERROR] Failed to update scratchpad: {e}"

    def _execute_user_profile_update(self, body: str) -> str:
        """Executes a patch operation on the global user profile file."""
        if not getattr(self, '_user_profile_path', None):
            return "[SYSTEM ERROR] User profile not initialized."

        try:
            current_content = self._user_profile_path.read_text(encoding="utf-8", errors="ignore")
            from lollms_client.lollms_artefact import ArtefactManager
            patched_content = ArtefactManager.apply_aider_patch(current_content, body)
            self._user_profile_path.write_text(patched_content, encoding="utf-8")

            # Update the in-memory cache so the next system prompt build reflects the change
            object.__setattr__(self, '_user_profile_content', patched_content)
            return "✅ Global user profile updated successfully."
        except Exception as e:
            return f"[SYSTEM ERROR] Failed to update user profile: {e}"

    def _build_onboarding_block(self) -> str:
        """Injects a mandatory first-run onboarding protocol if the user profile is empty."""
        profile_content = getattr(self, '_user_profile_content', "")
        import re as _re
        name_match = _re.search(r'## Identity\s*\n+\s*- Name:\s*(\S+)', profile_content)
        if name_match and name_match.group(1).strip():
            return ""

        return (
            "\n=== FIRST-RUN ONBOARDING PROTOCOL (MANDATORY) ===\n"
            "The user's profile is empty. You MUST conduct a brief onboarding interview before doing any work.\n"
            "Ask the following questions one by one, wait for the user's response, and then save them to your profile.\n"
            "1. What is your name?\n"
            "2. What programming language are we primarily working in?\n"
            "3. Do you have any specific coding style preferences (e.g., tabs vs spaces, type hints)?\n"
            "After gathering all answers, use `<user_profile_update>` to save them, then emit `<done/>`.\n"
            "=== END ONBOARDING PROTOCOL ===\n"
        )

    def _enforce_git_safety(self, title: str, is_overwrite: bool) -> Optional[str]:
        """
        Programmatic guard for destructive file writes.
        Returns an error message string if the write is blocked, or None if allowed.
        """
        if not is_overwrite or not self._resolved_workspace:
            return None

        try:
            git_dir = self._resolved_workspace / ".git"
            if not git_dir.exists():
                return None

            if getattr(self, '_git_autonomy_granted', False) or "Git Autonomy: Granted" in getattr(self, '_user_profile_content', ''):
                return None

            return (
                f"❌ GIT SAFETY BLOCK: You are about to overwrite '{title}'.\n"
                f"This workspace is a git repository. You MUST ask the user for permission.\n"
                f"Output EXACTLY: \"⚠️ I am about to modify `{title}`. This action will be executed on a new git branch. Do you approve? (yes/no)\"\n"
                f"Do NOT emit the `<artifact>` tag again until the user replies 'yes'."
            )
        except Exception:
            return None

    def _build_system_prompt(self, active_tools: Dict) -> str:
        sys_prompt = self.system_prompt or ""
        onboarding_block = self._build_onboarding_block()
        rules = (
          "\n=== ACTION EXECUTION & TERMINATION PROTOCOL (CRITICAL) ===\n"
          "1. **INTENT ≠ EXECUTION**: Stating 'I will search...' in text DOES NOT execute the action. You MUST output the `<tool>` tag.\n"
          "2. **MANDATORY TAG EMISSION**: To execute an action, you MUST output the `<tool>` tag immediately.\n"
          "3. **EXPLICIT TERMINATION WITH `<done/>`**: When all objectives are met and tests pass, end with a `<done/>` tag on a new line.\n"
          "4. **SAME-SESSION CONTINUATION**: When executing a sequence, emit the next action's tag in your IMMEDIATE NEXT response.\n"
          "5. **AGENTIC TRIGGER**: If the user requests code generation, file modification, testing, or multi-step work, you MUST enter the agentic loop and use `<tool>` or `<artifact>` tags. Do NOT write code directly in conversational prose.\n"
          "6. **ROUND 1 SHORT-CIRCUIT**: If the user's request is purely conversational (e.g., greetings, simple questions), respond conversationally without `<done/>` or tool tags.\n"
          "7. **NO PROSE BEFORE TOOLS**: DO NOT write introductory text like 'Let me check the git status' before a tool call. Output the `<tool>` tag as the VERY FIRST token of your response.\n"
            "8. **SAME-RESPONSE EXECUTION (CRITICAL)**: If you state an intent to perform an action (e.g., 'Let me unlock file X', 'I will now patch Y'), you MUST execute that action's tag (`<unlock_file>`, `<lock_file>`, `<artifact>`, `<tool>`) IN THE SAME RESPONSE. Stating intent and then emitting `<done/>` without the tag is a CRITICAL ERROR. Never split an intent and its execution across two turns.\n"
            "9. **BATCH CONTEXT OPERATIONS (MANDATORY)**: When locking, unlocking, or hiding multiple files, you MUST use a SINGLE tag containing all files separated by newlines. DO NOT emit multiple sequential tags for batch operations.\n"
            "   Example:\n"
            "   <lock_file>\n"
            "   file1.py\n"
            "   file2.py\n"
            "   file3.py\n"
            "   </lock_file>\n"
            "10. **DEPENDENCY SEPARATION (CRITICAL)**: You MAY emit multiple independent `<tool>` or `<artifact>` tags in a single response. They will be buffered and executed sequentially.\n"
            "    HOWEVER, if you need the RESULT of Tool A to construct the parameters for Tool B, they MUST be executed in separate rounds.\n"
            "    - Do NOT guess the output of Tool A. Emit Tool A, end your turn, and wait for the system to return the result.\n"
            "    - Once you have the result, emit Tool B in your next response.\n"
            "    - Example of WRONG behavior: `<tool>{\"name\": \"find_file\", \"parameters\": {\"pattern\": \"config.yml\"}}</tool>` followed by `<tool>{\"name\": \"read_file\", \"parameters\": {\"path\": \"./config.yml\"}}</tool>` (The path is guessed).\n"
            "    - Example of CORRECT behavior: Emit `<tool>{\"name\": \"find_file\", \"parameters\": {\"pattern\": \"config.yml\"}}</tool>` and end your turn. In the next turn, use the returned path to call `read_file`.\n"
            "\n=== TOOL CALLING DISCIPLINE (CRITICAL) ===\n"
            "1. **Tool Results ≠ Tool Calls**: When a tool returns JSON, it's a RESULT, not a new call.\n"
            "2. **One Call Per Task**: Once a tool succeeds, analyze and answer.\n"
            "3. **Loop Prevention**: Repeating a successful tool call with identical parameters is a CRITICAL ERROR.\n"
            "4. **File Outputs**: When a tool returns a file, it's ALREADY saved. Do NOT call it again.\n"
            "\n=== FILE EDITING PROTOCOL (AIDER SEARCH/REPLACE) ===\n"
            "For surgical updates to existing files, you MUST use the `<artifact>` tag with SEARCH/REPLACE blocks.\n"
            "The system automatically applies fuzzy matching and auto-correction if the exact search string isn't found.\n"
            "Syntax:\n"
            "<artifact name=\"filename.ext\" type=\"code\" language=\"python\">\n"
            "<<<<<<< SEARCH\n"
            "// exact lines to find\n"
            "=======\n"
            "// new lines to replace with\n"
            ">>>>>>> REPLACE\n"
            "</artifact>\n"
            "If a patch fails, the system will return the error. You MUST read the error carefully. The file content is already available in your context under the `## Fully Loaded File Contents [C]` section. Concentrate on the exact text, fix your SEARCH block, and re-emit the `<artifact>` tag. Do not attempt to use a `tool_read_file` tool, as it does not exist.\n"
            "\n=== SKILLS SYSTEM ===\n"
            "Skills are persistent knowledge capsules stored outside the workspace. They survive across sessions.\n"
            "They are categorized by visibility:\n"
            "1. **Visible**: Automatically loaded in your system prompt. Costs 0 turns.\n"
            "2. **Loadable**: Listed in your context block. Use `tool_load_skill` to pull the full content (Costs 1 turn).\n"
            "3. **Searchable**: Hidden from your context block. Use `tool_search_skills` then `tool_load_skill` (Costs 2 turns).\n"
            "Use `tool_list_skills` to programmatically list all skills and their tiers.\n"
            "If you discover a reusable methodology or best practice, use `tool_create_skill` to save it for future use.\n"
            "Use `tool_update_skill` to refine existing skills as you learn more.\n"
            "\n=== SUB-AGENT DELEGATION ===\n"
            "If `tool_spawn_sub_agent` is available, you can delegate complex sub-tasks to a focused child agent.\n"
            "The child shares your workspace but cannot spawn further sub-agents.\n"
            "Use this for heavy tasks like writing large scripts, researching topics, or designing presentations.\n"
            "\n=== STATE & MEMORY SEGREGATION DOCTRINE (CRITICAL) ===\n"
            "You have TWO distinct mechanisms for persisting information. You MUST strictly segregate what goes where.\n"
            "1. **THE SCRATCHPAD (`<scratchpad_append>` / `<scratchpad_patch>`)**:\n"
            "   - **Scope**: LOCAL to the current project/workspace.\n"
            "   - **Usage**: Use for SHORT-TERM, project-specific state. Examples: temporary file paths, intermediate calculation results, active task checklists, or branching strategies specific to this codebase.\n"
            "   - **Clearing**: Use `<scratchpad_clear></scratchpad_clear>` when the specific task is done to free up context space.\n"
            "2. **PERSISTENT MEMORY (`<mem_new>` / `<mem_update>`)**:\n"
            "   - **Scope**: UNIVERSAL. Survives across ALL projects and sessions.\n"
            "   - **Usage**: Use for LONG-TERM facts, architectural rules, and universal user preferences. Examples: 'The user prefers 4-space indentation', 'Library X requires initialization before use', 'The user's name is Saif'.\n"
            "   - **Mandatory Action**: If the user states a personal fact or a universal coding standard, you MUST emit `<mem_new>` immediately.\n"
            "3. **USER PROFILE (`<user_profile_update>`)**:\n"
            "   - Used exclusively for the user's identity and universal interaction preferences.\n"
            "=== END STATE & MEMORY SEGREGATION DOCTRINE ===\n"
            "\n=== OPERATIONAL SAFETY DOCTRINE ===\n"
            "1. **GIT BRANCHING & CONFIRMATION PROTOCOL (MANDATORY)**: \n"
            "   Before modifying, overwriting, or deleting ANY existing file in the workspace, you MUST follow this protocol:\n"
            "   a. Check if a `.git` directory exists in the workspace root.\n"
            "   b. If it exists, you MUST ask the user for explicit permission to proceed.\n"
            "      Example: \"⚠️ I am about to modify `critic.md`. This action will be executed on a new git branch. Do you approve? (yes/no)\"\n"
            "   c. Upon receiving 'yes', you MUST create and checkout a new branch before emitting any `<artifact>` tags.\n"
            "      Use the shell tool: `git checkout -b update/<short-branch-name>`.\n"
            "      Work exclusively in this branch. Only merge back to `main` after all tests pass.\n"
            "   d. If no `.git` directory exists, you may proceed with modifications, but you should still inform the user before overwriting large files.\n"
            "2. **DANGEROUS OPERATIONS (HUMAN-IN-THE-LOOP)**: Operations that are destructive or irreversible REQUIRE explicit user confirmation.\n"
            "   Examples: `git push --force`, `rm -rf`, dropping database tables, modifying system configs.\n"
            "   Before executing such a command via a tool, you MUST output a message like:\n"
            "   \"⚠️ DANGER: I am about to run `git push --force`. This will overwrite remote history. Do you approve? (yes/no)\"\n"
            "   Wait for the user's response before emitting the tool tag.\n"
            "3. **AUTONOMOUS DEBUGGING LOOPS**: Fixing failing tests, resolving merge conflicts, and iterating on code during a debug cycle is EXEMPT from the confirmation rule.\n"
            "   If tests fail, autonomously read the logs, fix the code, and re-run tests until they pass. Do NOT ask the user for help.\n"
            "=== END OPERATIONAL SAFETY DOCTRINE ===\n"
            f"{onboarding_block}"
            "\n=== THINKING & REASONING CONSTRAINT ===\n"
            "If you output thoughts enclosed in  tags, you MUST output all functional XML tags AFTER the closing tag.\n"
            "\n=== TOOL CALLING DISCIPLINE (CRITICAL) ===\n"
            "1. **EXACT CLOSING TAG**: The closing tag is `</tool>`. You MUST NOT write ``` or any other variation.\n"
            "2. **NEW LINE ONLY**: The `<tool>` tag MUST start on a brand new line.\n"
            "3. **NO PROSE AROUND IT**: Do NOT write introductory text before the tag, and do NOT write text after it on the same line.\n"
            "4. **EXACT JSON FORMAT**: The content inside the `<tool>` tag MUST be a valid JSON object with `name` and `parameters` keys.\n"
            "\nExample of CORRECT behavior:\n"
            "<tool>{\"name\": \"tool_search_files\", \"parameters\": {\"pattern\": \"TODO\"}}</tool>\n\n"
            "Example of WRONG (XML attributes, forbidden):\n"
            "<tool_execute_shell_command command=\"type ..\\README.md\" />\n\n"
            "=== END TOOL CALLING DISCIPLINE ===\n"
            "\n=== ANTI-MIMICRY PROTOCOL (CRITICAL) ===\n"
            "1. **NEVER OUTPUT SYSTEM MARKERS**: You are STRICTLY FORBIDDEN from generating `<processing>` blocks or `[SYSTEM:` markers.\n"
            "2. **USE REAL TAGS**: To call tools, use the actual `<tool>` XML tags.\n"
            "\n=== GLOBAL USER PROFILE MANAGEMENT ===\n"
            "You have access to a universal user profile that persists across ALL projects and sessions.\n"
            "It contains the user's identity, global constraints, and universal preferences.\n"
            "CRITICAL: Do NOT store project-specific information (like 'this project uses React') in the user profile. Use the Scratchpad for project state.\n"
            "To update the user profile when you learn a new universal fact, emit:\n"
            "<user_profile_update>\n"
            "<<<<<<< SEARCH\n"
            "// exact lines to find\n"
            "=======\n"
            "// new lines to replace with\n"
            ">>>>>>> REPLACE\n"
            "</user_profile_update>\n"
            "\n=== WORKSPACE CONTEXT & DYNAMIC STATE PROTOCOL (CRITICAL) ===\n"
            "You operate inside a workspace. At the beginning of EVERY turn, the user's prompt will be suffixed with a dynamic context block.\n"
            "This block contains:\n"
            "1. **Workspace Directory Tree**: A list of all files with markers indicating their state.\n"
            "   - [C] Fully Loaded in Context (Verbatim text/code is provided below the tree)\n"
            "   - [M] Signature / Metadata Only (Exposes schemas, layouts, or code signatures)\n"
            "   - [U] Inactive/Unlockable (Excluded from context, but you can unlock it to [C] by calling <unlock_file>)\n"
            "   - [L] Locked in Tree (Excluded from context and cannot be unlocked)\n"
            "2. **Fully Loaded File Contents**: The raw text of any files marked [C].\n"
            "3. **Persistent Scratchpad**: Your long-term notes for state recovery across sessions.\n"
            "   To update it, use:\n"
            "   1. `<scratchpad_append>content to add</scratchpad_append>`\n"
            "   2. `<scratchpad_patch>` with Aider SEARCH/REPLACE blocks to surgically update sections.\n"
            "4. **Active Memories**: Relevant memories hydrated from your persistent database.\n"
            "5. **Context Telemetry**: A live breakdown of token consumption per segment (System, History, Tree, Contents, Virtual History).\n"
            "\n**CONTEXT VISIBILITY OPERATIONS**\n"
            "To manage your context budget, you can emit the following tags:\n"
            "- `<unlock_file>filename.py</unlock_file>`: Loads a file into your context (changes [U] to [C]).\n"
            "- `<lock_file>filename.py</lock_file>`: Removes a file from your context (changes [C] to [U]).\n"
            "- `<hide_file>filename.py</hide_file>`: Completely removes a file from your view.\n"
            "- `<collapse_folder>folder_name</collapse_folder>`: Hides all files within a folder.\n"
            "- `<uncollapse_folder>folder_name</uncollapse_folder>`: Restores a collapsed folder.\n"
            "- Batch operations are supported and encouraged. You can list multiple files separated by newlines, commas, or semicolons inside a single tag:\n"
            "  <unlock_file>\n"
            "    file1.py,\n"
            "    file2.py,\n"
            "    file3.py\n"
            "  </unlock_file>\n"
            "\n**AUTONOMOUS HISTORY REFACTORING**\n"
            "If the Context Telemetry indicates that `History` or `Virtual History` is consuming an excessive amount of tokens (e.g., > 40% of total context),\n"
            "or if the user explicitly asks to refactor, summarize, or compress the conversation history, you MUST emit:\n"
            "<refactor_history></refactor_history>\n"
            "This will trigger an autonomous background process that summarizes the older conversation history into a dense, factual block,\n"
            "freeing up massive amounts of context space without losing critical state.\n"
        )

        memory_instructions = ""
        if self.memory_manager:
            memory_instructions = (
                "\n=== PERSISTENT MEMORY SYSTEM (CRITICAL FOR CONTINUITY) ===\n"
                "You have access to a persistent memory database. You can store and retrieve information across sessions.\n"
                "1. **STORE FACTS**: When the user shares personal information (e.g., name, preferences, project details), you MUST save it using:\n"
                "   <mem_new content=\"The user's name is Saif\" tags=\"identity,user_profile\" level=\"2\" />\n"
                "2. **UPDATE FACTS**: If information changes, use:\n"
                "   <mem_update id=\"memory_id\" content=\"New information\" />\n"
                "3. **AUTOMATIC RECALL**: Relevant memories are automatically injected into your context. You do not need to query them manually.\n"
                "4. **MANDATORY**: Always use memory tags for non-trivial user facts. If the user tells you their name, you MUST emit `<mem_new>` immediately.\n"
            )

        skills_ctx = ""
        if self.skills_manager:
            skills_ctx_str = self.skills_manager.build_context()
            if skills_ctx_str:
                skills_ctx = "\n" + skills_ctx_str

            if len(self.skills_manager.skills) == 0:
                skills_ctx += (
                    "\n=== SKILLS SYSTEM STATUS ===\n"
                    "The skills library is currently EMPTY. There are 0 skills available.\n"
                    "Do NOT attempt to call `tool_list_skills` or `tool_search_skills` as they will return nothing.\n"
                    "If you discover a reusable methodology or best practice during your task, use `tool_create_skill` to save it for future use.\n"
                    "=== END SKILLS SYSTEM STATUS ==="
                )

        tool_desc = ""
        if active_tools:
            tool_desc = "\n=== TOOLS AVAILABLE ===\nTo use a tool, emit `<tool>{\"name\": \"...\", \"parameters\": {...}}</tool>`.\n\nAvailable tools:\n"
            for t_name, t_spec in active_tools.items():
                desc = t_spec.get("description", "")
                params_list = t_spec.get("parameters", [])
                param_desc = ", ".join([f"{p['name']}: {p['type']}" for p in params_list])
                tool_desc += f"- {t_name}({param_desc}): {desc}\n"

        # 🛡️ KV-CACHE OPTIMIZATION: Workspace context is strictly kept OUT of the system prompt.
        # It is injected as a dynamic suffix at the very end of the messages array.
        return sys_prompt + "\n" + rules + skills_ctx + memory_instructions + tool_desc
    
    
    def _execute_context_visibility(self, tag_name: str, body: str) -> Dict[str, Any]:
        if not hasattr(self, '_artefact_manager') or not self._artefact_manager:
            return "[SYSTEM ERROR] Artefact system not initialized. Cannot manage file visibility."

        try:
            from lollms_client.lollms_artefact import ArtefactVisibility
        except ImportError:
            return "[SYSTEM ERROR] ArtefactVisibility module not available."

        target_visibility = ArtefactVisibility.FULL
        action_verb = "Unlocking"
        if tag_name == "lock_file":
            target_visibility = ArtefactVisibility.TREE_LOCKED
            action_verb = "Locking"
        elif tag_name == "hide_file":
            target_visibility = ArtefactVisibility.HIDDEN
            action_verb = "Hiding"
        elif tag_name == "collapse_folder":
            target_visibility = ArtefactVisibility.FOLDER_COLLAPSED
            action_verb = "Collapsing"
        elif tag_name == "uncollapse_folder":
            target_visibility = ArtefactVisibility.TREE_UNLOCKABLE
            action_verb = "Uncollapsing"

        clean_body = body
        if "<" in body and ">" in body:
            xml_bodies = re.findall(r'<[^>]+>(.*?)</[^>]+>', body, re.DOTALL)
            if xml_bodies:
                clean_body = "\n".join(xml_bodies)

        all_arts = self._artefact_manager._get_all_raw()

        raw_targets = re.split(r'[\n,;]+', clean_body)
        targets = [t.strip().replace("\\", "/") for t in raw_targets if t.strip()]

        expanded_targets = []
        all_arts_titles = [a.get("title", "") for a in all_arts if not a.get("title", "").endswith("::images")]
        for target in targets:
            target_lower = target.lower()
            if "all files" in target_lower or "all" == target_lower:
                exceptions = []
                if "except" in target_lower:
                    exceptions_part = target_lower.split("except", 1)[1]
                    exceptions = [e.strip().replace("\\", "/") for e in re.split(r'[\n,;]+', exceptions_part) if e.strip()]
                for title in all_arts_titles:
                    if not any(ex.lower() in title.lower() for ex in exceptions):
                        expanded_targets.append(title)
            else:
                expanded_targets.append(target)

        targets = expanded_targets

        processed_files = []
        already_in_state = []
        not_found = []
        blocked_files = []
        loaded_contents = {}

        max_ctx = 0
        if self.lollms_client and hasattr(self.lollms_client, 'get_ctx_size'):
            try:
                max_ctx = self.lollms_client.get_ctx_size() or 0
            except Exception:
                max_ctx = 0

        if max_ctx > 0:
            _MAX_UNLOCK_TOKENS = int(max_ctx * 0.95)
        else:
            _MAX_UNLOCK_TOKENS = 50000

        _BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".zip", ".tar", ".gz", ".pdf", ".docx", ".pptx", ".mp3", ".wav", ".mp4"}

        def _norm_path(p: str) -> str:
            p = p.replace("\\", "/").strip()
            if p.startswith("./"):
                p = p[2:]
            return p.lower()

        for t_target in targets:
            if tag_name in ("collapse_folder", "uncollapse_folder"):
                folder_prefix = t_target.rstrip('/') + '/'
                matched_arts = [a for a in all_arts if a.get("physical_path", "").replace("\\", "/").startswith(folder_prefix)]

                if not matched_arts:
                    not_found.append(t_target)
                    continue

                for art in matched_arts:
                    if art.get("visibility") == target_visibility:
                        already_in_state.append(art["title"])
                    else:
                        art["visibility"] = target_visibility
                        if target_visibility == ArtefactVisibility.FULL:
                            art["active"] = True
                        else:
                            art["active"] = False
                            art["content"] = ""
                        processed_files.append(art["title"])
                continue

            target_norm = _norm_path(t_target)
            art = next((a for a in all_arts if _norm_path(a.get("title", "")) == target_norm), None)

            if not art:
                not_found.append(t_target)
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.warning(f"[ContextUnlock] File not found in index: {t_target}")
                    ASCIIColors.warning(f"[ContextUnlock] Indexed titles sample: {all_arts_titles[:5]}")
                continue
            elif art.get("visibility") == target_visibility:
                already_in_state.append(t_target)
            elif target_visibility == ArtefactVisibility.FULL:
                file_path = self._resolved_workspace / art["title"]

                token_count = 0
                content = ""

                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        token_count = len(content) // 4
                    except Exception as read_err:
                        ASCIIColors.warning(f"[ContextUnlock] Failed to read {art['title']}: {read_err}")
                        blocked_files.append((art["title"], 0))
                        continue
                else:
                    not_found.append(t_target)
                    continue

                if token_count > _MAX_UNLOCK_TOKENS:
                    ASCIIColors.warning(
                        f"[ContextBudgetGuard] Blocked unlock of '{art['title']}': "
                        f"~{token_count:,} tokens exceeds limit of {_MAX_UNLOCK_TOKENS:,}."
                    )
                    blocked_files.append((art["title"], token_count))
                else:
                    art["content"] = content
                    art["token_count"] = token_count
                    art["content_source"] = "db"
                    art["visibility"] = ArtefactVisibility.FULL
                    art["active"] = True
                    processed_files.append(art["title"])
                    if content:
                        loaded_contents[art["title"]] = content

                    try:
                        if hasattr(self, '_state_db_path'):
                            import sqlite3 as _sqlite3
                            import hashlib as _hashlib
                            file_hash = _hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest()
                            conn = _sqlite3.connect(str(self._state_db_path))
                            cursor = conn.cursor()
                            cursor.execute(
                                "UPDATE file_states SET hash = ? WHERE title = ?",
                                (file_hash, art["title"])
                            )
                            conn.commit()
                            conn.close()
                    except Exception:
                        pass
            else:
                art["visibility"] = target_visibility
                art["active"] = False
                art["content"] = ""
                processed_files.append(art["title"])

        if processed_files or already_in_state:
            self._artefact_manager._save_all(all_arts)
            object.__setattr__(self, '_last_ws_sync_time', 0.0)

        try:
            if hasattr(self, '_state_db_path') and hasattr(self, '_artefact_manager'):
                import sqlite3 as _sqlite3
                conn = _sqlite3.connect(str(self._state_db_path))
                cursor = conn.cursor()
                for t_file in processed_files + already_in_state:
                    cursor.execute(
                        "INSERT OR REPLACE INTO file_states (title, visibility) VALUES (?, ?)",
                        (t_file, target_visibility)
                    )
                conn.commit()
                conn.close()

                if target_visibility in (ArtefactVisibility.TREE_LOCKED, ArtefactVisibility.HIDDEN):
                    current_arts = self._artefact_manager._get_all_raw()
                    for art in current_arts:
                        if art.get("title") in processed_files:
                            art["content"] = ""
                            art["active"] = False
                            art["visibility"] = target_visibility
                    self._artefact_manager._save_all(current_arts)

                    object.__setattr__(self, '_last_ws_sync_time', 0.0)

        except Exception as commit_err:
            ASCIIColors.warning(f"[LollmsPersonality] Failed to persist visibility state: {commit_err}")

        status_parts = []
        if processed_files:
            status_parts.append(f"✅ {action_verb}: {', '.join(processed_files)}")
        if already_in_state:
            status_parts.append(f"⚠️ Already in target state: {', '.join(already_in_state)}")
        if not_found:
            status_parts.append(f"❌ Not found: {', '.join(not_found)}")
        if blocked_files:
            blocked_desc = "; ".join(
                f"{bf} (~{tc:,} tokens)" if tc > 0 else f"{bf} (Binary/Read Error)" for bf, tc in blocked_files
            )
            status_parts.append(
                f"🛑 BLOCKED: {blocked_desc}. "
                f"Use a tool (SQL query, grep, or Python script) to extract "
                f"specific data from this file instead of loading it fully."
            )

        active_files_list = []
        if hasattr(self, '_artefact_manager') and self._artefact_manager:
            current_arts = self._artefact_manager._get_all_raw()
            active_files_list = [
                a.get("title", "") for a in current_arts
                if a.get("visibility") == ArtefactVisibility.FULL
                and not a.get("title", "").endswith("::images")
            ]

        if active_files_list:
            status_parts.append("\n📂 Currently Loaded in Context [C]:")
            for f_name in sorted(active_files_list):
                status_parts.append(f"  - {f_name}")
        else:
            status_parts.append("\n📂 No files are currently loaded in context.")

        status_meta = "failure" if (not_found or blocked_files) else "success"

        if processed_files or already_in_state:
            object.__setattr__(self, '_last_ws_sync_time', 0.0)

        error_str = None
        if status_meta == "failure":
            error_str = "\n".join([p for p in status_parts if "❌" in p or "🛑" in p])

        status_str = f"{action_verb} context files...\nContext Update:\n{'; '.join(status_parts)}\nstatus:{status_meta}"
        return {
            "status_str": status_str,
            "processed_files": processed_files,
            "already_in_state": already_in_state,
            "not_found": not_found,
            "blocked_files": blocked_files,
            "error": error_str,
            "loaded_contents": loaded_contents
        }
        
          
        
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

            if "arguments" in tool_params and isinstance(tool_params["arguments"], dict):
                extracted_args = tool_params.pop("arguments")
                extracted_args.update(tool_params)
                tool_params = extracted_args

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

            lcp_binding = getattr(self.lollms_client, 'tools', None)
            tool_def = active_tools.get(tool_name, {})

            if "callable" in tool_def:
                call_kwargs = dict(sanitized_params)
                _tool_sig = inspect.signature(tool_def["callable"]).parameters
                if "discussion_instance" in _tool_sig:
                    call_kwargs["discussion_instance"] = getattr(self, '_artefact_proxy', None)
                if "lollms_client_instance" in _tool_sig:
                    call_kwargs["lollms_client_instance"] = self.lollms_client

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
                        discussion_instance=getattr(self, '_artefact_proxy', None),
                        lollms_client_instance=self.lollms_client
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

    def chat(
        self,
        prompt: str,
        lollms_client: Any = None,
        streaming_callback: Optional[Callable] = None,
        tools: Optional[Dict[str, Any]] = None,
        tool_files: Optional[List[Union[str, Path]]] = None,
        max_reasoning_steps: int = 20,
        temperature: float = 0.7,
        n_predict: int = 4096,
        enable_artefacts: bool = True,
        use_internal_history: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        if lollms_client is not None:
            self.lollms_client = lollms_client

        if self.lollms_client is None:
            raise RuntimeError(f"[{self.name}] Independent chat requires a lollms_client instance.")

        self._reset_cancel_state()
        object.__setattr__(self, '_consecutive_empty_responses', 0)

        if self._sub_agent_spawner:
            self._sub_agent_spawner.reset_turn()

        if self._failure_memory and hasattr(self._failure_memory, '_signatures'):
            self._failure_memory._signatures.clear()
            if hasattr(self._failure_memory, 'failures'):
                self._failure_memory.failures = []

        if enable_artefacts:
            if not self.workspace_path:
                ASCIIColors.warning(f"[{self.name}] Workspace path is not set. Artefact system disabled.")
            else:
                object.__setattr__(self, '_resolved_workspace', Path(self.workspace_path).resolve())
                if getattr(self, '_artefact_manager', None) is None:
                    self._init_artefact_system()
                    ASCIIColors.info(f"[{self.name}] ✅ Artefact system initialized for workspace: {self._resolved_workspace}")

        self._init_scratchpad()

        cleaned_prompt = prompt
        active_tools = self._discover_tools(tools, tool_files or [])

        stable_system_prompt = self._build_system_prompt(active_tools)
        stable_system_prompt += self._build_user_profile_context()

        dynamic_suffix_parts = []

        ws_ctx = self._build_workspace_context_block()
        if ws_ctx:
            dynamic_suffix_parts.append(ws_ctx.strip())

        scratchpad_ctx = self._build_scratchpad_context()
        if scratchpad_ctx:
            dynamic_suffix_parts.append(scratchpad_ctx.strip())

        if self.memory_manager:
            try:
                if hasattr(self.memory_manager, 'auto_pull_deep_memories'):
                    self.memory_manager.auto_pull_deep_memories(cleaned_prompt)

                if hasattr(self.memory_manager, 'build_working_zone'):
                    mem_zone = self.memory_manager.build_working_zone()
                    if mem_zone:
                        dynamic_suffix_parts.append("=== ACTIVE MEMORIES (PERSISTENT ACROSS SESSIONS) ===\n" + mem_zone + "\n=== END MEMORIES ===")
            except Exception as mem_ex:
                ASCIIColors.warning(f"[{self.name}] Failed to hydrate memories: {mem_ex}")

        if use_internal_history:
            base_conversation = list(self._conversation)
        else:
            base_conversation = []

        telemetry = self._calculate_context_telemetry(stable_system_prompt, base_conversation, ws_ctx or "", [])
        telemetry_block = self._build_telemetry_block(telemetry)
        if telemetry_block:
            dynamic_suffix_parts.append(telemetry_block)

        dynamic_suffix = "\n\n".join(dynamic_suffix_parts)

        if dynamic_suffix:
            fused_prompt = f"=== CURRENT WORKSPACE CONTEXT ===\n{dynamic_suffix}\n=== END WORKSPACE CONTEXT ===\n\n{cleaned_prompt}"
        else:
            fused_prompt = cleaned_prompt

        base_conversation.append({"role": "user", "content": fused_prompt})

        virtual_history: List[SimpleNamespace] = []
        tool_calls_this_turn: List[Dict[str, Any]] = []
        tool_results_this_turn: List[Dict[str, Any]] = []
        round_count = 0
        was_cancelled = False
        tool_signature_counts: Dict[str, int] = {}
        successful_tool_signatures: set = set()
        seen_context_signatures: set = set()
        final_response = ""
        workspace_changes: List[Dict[str, Any]] = []

        while round_count < max_reasoning_steps:
            if self.is_generation_cancelled():
                was_cancelled = True
                break

            round_count += 1

            if getattr(self, 'debug_mode', False):
                ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} START ===")

            if hasattr(self.lollms_client, 'llm') and hasattr(self.lollms_client.llm, 'reset_cancel'):
                try:
                    self.lollms_client.llm.reset_cancel()
                except Exception:
                    pass

            messages = [{"role": "system", "content": stable_system_prompt}]
            messages.extend(base_conversation)

            for vh in virtual_history:
                role = "user" if vh.sender_type == "user" else "assistant"
                messages.append({"role": role, "content": vh.content})

            if round_count > 1:
                if not virtual_history or virtual_history[-1].sender_type == "assistant":
                    messages.append({"role": "user", "content": "[SYSTEM: Continue your task.]"})

            if getattr(self, 'debug_mode', False):
                try:
                    debug_dir = self._resolved_workspace / ".lollms_code" / "_debug_dumps"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    short_log_path = debug_dir / f"prompt_dump_round_{round_count}_shortened.md"

                    with open(short_log_path, "w", encoding="utf-8") as f:
                        f.write(f"# 🐛 Round {round_count} - Shortened Prompt Dump\n\n")
                        for i, msg in enumerate(messages):
                            role = msg.get("role", "unknown").upper()
                            content = msg.get("content", "")
                            if len(content) > 1000:
                                short_content = content[:500] + "\n\n[... truncated ...]\n\n" + content[-500:]
                            else:
                                short_content = content
                            f.write(f"## MSG [{i}] - {role}\n\n")
                            f.write(f"```\n{short_content}\n```\n\n")
                except Exception as debug_err:
                    ASCIIColors.warning(f"Failed to write shortened debug log: {debug_err}")

            if getattr(self, 'debug_mode', False):
                try:
                    debug_dir = self._resolved_workspace / ".lollms_code" / "_debug_dumps"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    full_log_path = debug_dir / f"full_prompt_round_{round_count}.log"

                    with open(full_log_path, "w", encoding="utf-8") as f:
                        f.write("="*80 + "\n")
                        f.write(f"🐛 [DEBUG] ROUND {round_count} - FULL PROMPT\n")
                        f.write("="*80 + "\n")
                        for i, msg in enumerate(messages):
                            role = msg.get("role", "unknown").upper()
                            content = msg.get("content", "")
                            f.write(f"\n--- MSG [{i}] ROLE: {role} ---\n")
                            f.write(content + "\n")
                        f.write("\n" + "="*80 + "\n")
                except Exception as debug_err:
                    ASCIIColors.warning(f"Failed to write full prompt log: {debug_err}")

            messages = _normalize_messages(messages)

            event_mode = kwargs.get("event_mode", EventMode.PROCESSING_TAG_MODE)
            ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)

            raw_llm_output_buffer = ""
            def _inline_relay(chunk, msg_type=None, meta=None):
                nonlocal raw_llm_output_buffer
                if self.is_generation_cancelled():
                    return False
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss._cb(chunk, msg_type, meta) if streaming_callback else True
                if isinstance(chunk, str):
                    raw_llm_output_buffer += chunk
                    if meta and meta.get("was_processed"):
                        return True
                    return ss.feed(chunk)
                return True

            gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("streaming_callback", "temperature", "n_predict", "stream")}
            gen_kwargs["n_predict"] = min(n_predict, self.max_tokens_per_turn)
            gen_kwargs["temperature"] = temperature

            try:
                self.lollms_client.generate_from_messages(
                    messages=messages,
                    stream=True,
                    streaming_callback=_inline_relay,
                    **gen_kwargs
                )
                if hasattr(self.lollms_client, 'llm') and hasattr(self.lollms_client.llm, 'flush_stream'):
                    try:
                        self.lollms_client.llm.flush_stream()
                    except Exception:
                        pass
            except Exception as gen_err:
                if self.is_generation_cancelled():
                    was_cancelled = True
                    break
                else:
                    ASCIIColors.error(f"[{self.name}] Generation error: {gen_err}")
                    final_response = f"[Generation error: {gen_err}]"
                    break

            if self.is_generation_cancelled():
                was_cancelled = True
                break

            if getattr(self, 'debug_mode', False) and raw_llm_output_buffer:
                try:
                    debug_dir = self._resolved_workspace / ".lollms_code" / "_debug_dumps"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    raw_output_log_path = debug_dir / f"raw_llm_output_round_{round_count}.log"

                    with open(raw_output_log_path, "w", encoding="utf-8") as f:
                        f.write("="*80 + "\n")
                        f.write(f"🐛 [DEBUG] ROUND {round_count} - RAW LLM STREAM OUTPUT\n")
                        f.write("="*80 + "\n\n")
                        f.write(raw_llm_output_buffer)
                        f.write("\n\n" + "="*80 + "\n")
                except Exception as debug_err:
                    ASCIIColors.warning(f"Failed to write raw LLM output log: {debug_err}")

            ss.flush_remaining_buffer()

            if ss.was_done_detected():
                final_response = ss.get_clean_text()
                if ss.completed_actions:
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=final_response))
                    virtual_history.append(SimpleNamespace(sender_type="user", content="[SYSTEM: You emitted <done/> but there are pending buffered actions. Executing them now. Please analyze the results and provide your final summary.]"))
                    final_response = ""

                sanitized_final_response = re.sub(r'<[^>]+>', '', final_response).strip()
                
                if not sanitized_final_response and round_count == 1 and not ss.completed_actions and not virtual_history:
                    ASCIIColors.warning(f"[{self.name}] 🚫 Empty response with <done/> detected on round 1. Forcing continuation.")
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content="[SYSTEM: Your previous response was completely empty. You MUST provide a substantive response or use a tool. Do NOT output an empty response. If you are analyzing code, state your findings. If you are writing code, emit the `<artifact>` tag. Do not emit `<done/>` until you have actually produced output.]"
                    ))
                    ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                    continue
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: <done/> detected ===")
                break
            
            
            if ss.completed_actions:
                raw_round_text = ss.get_clean_text()
                clean_history_text = self._sanitize_history_for_context(raw_round_text)
                virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text if clean_history_text.strip() else "[Assistant executed batched actions]"))

                files_before = self._take_workspace_snapshot()

                action_reports = []
                for action in ss.completed_actions:
                    if action["type"] == "tool":
                        tool_call_json_str = action["json"]
                        try:
                            call_data = json.loads(tool_call_json_str)
                            tool_name = call_data.get("name", "")
                            tool_params = call_data.get("parameters", {})

                            if not active_tools or tool_name not in active_tools:
                                action_reports.append(f"Tool '{tool_name}' not available. Use one of: {list(active_tools.keys())}")
                                continue

                            is_shell_tool = tool_name == "tool_execute_shell_command"
                            if is_shell_tool:
                                command_str = str(tool_params.get("command", "")).strip()
                                context_aware_sig = f"{tool_name}::{command_str}"
                            else:
                                param_sig = json.dumps(tool_params, sort_keys=True, default=str)
                                context_aware_sig = f"{tool_name}::{param_sig}"

                            if context_aware_sig in successful_tool_signatures:
                                action_reports.append(f"Repetitive call to '{tool_name}' with identical parameters blocked. Output already in context.")
                                continue

                            tool_res = self._execute_tool(tool_name, tool_params, active_tools)

                            tool_success = isinstance(tool_res, dict) and tool_res.get("success", True) is not False
                            if tool_success:
                                successful_tool_signatures.add(context_aware_sig)

                            tool_calls_this_turn.append({"round": round_count, "name": tool_name, "parameters": tool_params})
                            tool_results_this_turn.append({"round": round_count, "name": tool_name, "result": tool_res, "success": tool_success})
                            clean_result_str = _sanitize_tool_result(tool_res)

                            if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                try:
                                    if streaming_callback:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_TOOL_START, {"tool_name": tool_name, "parameters": tool_params})
                                except Exception:
                                    pass

                            if tool_success:
                                report_part = f"=== ✅ TOOL RESULT: {tool_name} ===\n<tool_result name=\"{tool_name}\" status=\"SUCCESS\">\n{clean_result_str}\n</tool_result>"
                            else:
                                report_part = f"=== ❌ TOOL FAILED: {tool_name} ===\n<tool_result name=\"{tool_name}\" status=\"FAILED\">\n{clean_result_str}\n</tool_result>"

                            action_reports.append(report_part)

                            if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                try:
                                    if streaming_callback:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                            "tool_name": tool_name, "success": tool_success,
                                            "output": clean_result_str if tool_success else None,
                                            "error": None if tool_success else clean_result_str
                                        })
                                except Exception:
                                    pass
                        except Exception as e:
                            action_reports.append(f"[Tool execution error: {e}]")

                    elif action["type"] == "artifact":
                        raw_artifact_xml = action["xml"]
                        try:
                            attrs_match = re.search(r'<art(?:ifact|efact)[^>]*>', raw_artifact_xml, re.IGNORECASE)
                            attrs_str = attrs_match.group(0) if attrs_match else ""
                            body_match = re.search(r'<art(?:ifact|efact)[^>]*>(.*)</art(?:ifact|efact)>', raw_artifact_xml, re.DOTALL | re.IGNORECASE)
                            body_content = body_match.group(1).strip() if body_match else ""

                            title = "artifact"
                            lang = "python"
                            for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                                if m.group(1).lower() in ("name", "title"):
                                    title = m.group(2)
                                elif m.group(1).lower() == "language":
                                    lang = m.group(2)

                            is_patch = "<<<<<<< SEARCH" in body_content

                            file_path = self._resolved_workspace / title
                            is_overwrite = file_path.exists()

                            git_block = self._enforce_git_safety(title, is_overwrite)
                            if git_block:
                                action_reports.append(git_block)
                                continue

                            if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                try:
                                    if streaming_callback:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START, {"title": title, "art_type": "code", "language": lang, "is_patch": is_patch})
                                except Exception:
                                    pass

                            if is_patch:
                                if not file_path.exists():
                                    action_reports.append(f"[SYSTEM ERROR] File '{title}' not found. Cannot apply patch.")
                                    continue

                                stripped_body = body_content.strip()
                                if not stripped_body:
                                    action_reports.append(f"❌ SEARCH/REPLACE BLOCKED for {title}. Empty patch body.")
                                    continue

                                original_content = file_path.read_text(encoding="utf-8", errors="ignore")
                                try:
                                    patched_content = ArtefactManager.apply_aider_patch(original_content, body_content)
                                    file_path.write_text(patched_content, encoding="utf-8")
                                    action_reports.append(f"✅ SEARCH/REPLACE applied successfully to {title}.")
                                    if self._artefact_manager:
                                        self._artefact_manager.update(title=title, new_content=patched_content, language=lang, bump_version=True, active=True)
                                except Exception as patch_err:
                                    action_reports.append(f"❌ SEARCH/REPLACE FAILED for {title}. Error: {patch_err}")
                            else:
                                stripped_body = body_content.strip()
                                if not stripped_body:
                                    action_reports.append(f"❌ FILE WRITE BLOCKED for {title}. Empty artifact body.")
                                    continue

                                if self._artefact_manager:
                                    self._artefact_manager.add(title=title, artefact_type="code", content=body_content, language=lang, active=True)
                                file_path = self._resolved_workspace / title
                                file_path.write_text(body_content, encoding="utf-8")
                                action_reports.append(f"✅ File {title} created/updated successfully.")
                        except Exception as e:
                            action_reports.append(f"[SYSTEM ERROR] Failed to process artifact tag: {e}")

                    elif action["type"] == "context":
                        tag_name = action["tag_name"]
                        raw_xml = action["xml"]
                        try:
                            if tag_name == "scratchpad_clear":
                                res_msg = self._execute_scratchpad_clear()
                                action_reports.append(res_msg)
                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "success" if "✅" in res_msg else "failure", "error": None if "✅" in res_msg else res_msg})
                                if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                    if "✅" in res_msg: streaming_callback(f'<status>success</status>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                    else: streaming_callback(f'<status>failure</status>\n<error>{res_msg}</error>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                continue

                            if tag_name == "user_profile_clear":
                                res_msg = self._execute_user_profile_clear()
                                action_reports.append(res_msg)
                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "success" if "✅" in res_msg else "failure", "error": None if "✅" in res_msg else res_msg})
                                if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                    if "✅" in res_msg: streaming_callback(f'<status>success</status>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                    else: streaming_callback(f'<status>failure</status>\n<error>{res_msg}</error>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                continue

                            if "scratchpad" in tag_name:
                                body_match = re.search(r'<scratchpad_(?:append|patch)[^>]*>(.*?)</scratchpad_(?:append|patch)>', raw_xml, re.DOTALL | re.IGNORECASE)
                                body_content = body_match.group(1).strip() if body_match else ""
                                res_msg = self._execute_scratchpad_update(tag_name, body_content)
                                action_reports.append(res_msg)
                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "success" if "✅" in res_msg else "failure", "error": None if "✅" in res_msg else res_msg})
                                if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                    if "✅" in res_msg: streaming_callback(f'<status>success</status>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                    else: streaming_callback(f'<status>failure</status>\n<error>{res_msg}</error>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                continue

                            if "user_profile_update" in tag_name:
                                body_match = re.search(r'<user_profile_update>(.*?)</user_profile_update>', raw_xml, re.DOTALL | re.IGNORECASE)
                                body_content = body_match.group(1).strip() if body_match else ""
                                res_msg = self._execute_user_profile_update(body_content)
                                action_reports.append(res_msg)
                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "success" if "✅" in res_msg else "failure", "error": None if "✅" in res_msg else res_msg})
                                if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                    if "✅" in res_msg: streaming_callback(f'<status>success</status>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                    else: streaming_callback(f'<status>failure</status>\n<error>{res_msg}</error>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                continue

                            if tag_name in ("mem_new", "mem_update"):
                                if not self.memory_manager:
                                    err_msg = "Memory manager not initialized."
                                    action_reports.append(f"[SYSTEM ERROR] {err_msg}")
                                    if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "failure", "error": err_msg})
                                    if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                        streaming_callback(f'<status>failure</status>\n<error>{err_msg}</error>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                    continue
                                if tag_name == "mem_new":
                                    content_match = re.search(r'content\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    tags_match = re.search(r'tags\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    level_match = re.search(r'level\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    body_match = re.search(r'<mem_new[^>]*>(.*?)</mem_new>', raw_xml, re.DOTALL | re.IGNORECASE)
                                    mem_content = content_match.group(1) if content_match else (body_match.group(1).strip() if body_match else "")
                                    mem_tags = tags_match.group(1).split(",") if tags_match else []
                                    mem_level = int(level_match.group(1)) if level_match else 2
                                    self.memory_manager.add(content=mem_content, tags=mem_tags, importance=0.9, level=mem_level)
                                    res_msg = f"✅ Memory saved successfully: {mem_content[:50]}..."
                                    action_reports.append(res_msg)
                                    if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "success", "error": None})
                                    if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                        streaming_callback(f'<status>success</status>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                elif tag_name == "mem_update":
                                    id_match = re.search(r'id\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    content_match = re.search(r'content\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    body_match = re.search(r'<mem_update[^>]*>(.*?)</mem_update>', raw_xml, re.DOTALL | re.IGNORECASE)
                                    mem_id = id_match.group(1) if id_match else ""
                                    mem_content = content_match.group(1) if content_match else (body_match.group(1).strip() if body_match else "")
                                    self.memory_manager.update(memory_id=mem_id, content=mem_content)
                                    res_msg = f"✅ Memory updated successfully: {mem_id}"
                                    action_reports.append(res_msg)
                                    if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "success", "error": None})
                                    if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                        streaming_callback(f'<status>success</status>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                continue

                            body_match = re.search(r'<(?:unlock_file|lock_file|hide_file|collapse_folder|uncollapse_folder)[^>]*>(.*?)</(?:unlock_file|lock_file|hide_file|collapse_folder|uncollapse_folder)>', raw_xml, re.DOTALL | re.IGNORECASE)
                            body_content = body_match.group(1).strip() if body_match else ""

                            if not body_content:
                                attr_match = re.search(r'(?:path|file|files)\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                if attr_match:
                                    body_content = attr_match.group(1).strip()

                            context_sig = f"{tag_name}::{body_content}"
                            if context_sig in seen_context_signatures:
                                rep_msg = f"Repetitive context action '{tag_name}' with identical parameters blocked. Files are already in the requested state or failed previously. Do not retry."
                                action_reports.append(rep_msg)
                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "failure", "error": rep_msg})
                                if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                    streaming_callback(f'<status>failure</status>\n<error>{rep_msg}</error>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                continue

                            seen_context_signatures.add(context_sig)

                            vis_result = self._execute_context_visibility(tag_name, body_content)
                            status_str = ""
                            loaded_contents = {}
                            is_failure = False
                            error_msg = None

                            if isinstance(vis_result, dict):
                                status_str = vis_result.get("status_str", "")
                                loaded_contents = vis_result.get("loaded_contents", {})
                                is_failure = bool(vis_result.get("not_found") or vis_result.get("blocked_files"))
                                error_msg = vis_result.get("error")

                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {
                                        "action": tag_name,
                                        "files": vis_result.get("processed_files", []) + vis_result.get("already_in_state", []),
                                        "status": "failure" if is_failure else "success",
                                        "error": error_msg if is_failure else None
                                    })
                            else:
                                status_str = str(vis_result)
                                is_failure = "❌" in status_str or "SYSTEM ERROR" in status_str
                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {
                                        "action": tag_name,
                                        "files": [],
                                        "status": "failure" if is_failure else "success",
                                        "error": status_str if is_failure else None
                                    })

                            if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                if is_failure:
                                    streaming_callback(f'<status>failure</status>\n<error>{status_str}</error>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                else:
                                    streaming_callback(f'<status>success</status>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                            action_reports.append(status_str)

                            if loaded_contents:
                                content_parts = ["\n=== NEWLY LOADED FILE CONTENTS (INJECTED FOR VISIBILITY) ==="]
                                for f_title, f_content in loaded_contents.items():
                                    content_parts.append(f'<file path="{f_title}">\n{f_content}\n</file>')
                                content_parts.append("=== END LOADED CONTENTS ===\nAnalyze these results and continue your task, or emit <done/> if finished.")
                                action_reports.append("\n".join(content_parts))

                            object.__setattr__(self, '_last_ws_sync_time', 0.0)

                        except Exception as ctx_err:
                            err_msg = f"Failed to process context tag: {ctx_err}"
                            action_reports.append(f"[SYSTEM ERROR] {err_msg}")
                            if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "failure", "error": str(ctx_err)})
                            if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                streaming_callback(f'<status>failure</status>\n<error>{str(ctx_err)}</error>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                files_after = self._take_workspace_snapshot()
                changes = self._sync_workspace(files_before, files_after)
                if changes:
                    workspace_changes.extend(changes)

                if action_reports:
                    report_text = "\n\n".join(str(r) for r in action_reports) + "\n\nAnalyze these results and continue your task, or emit <done/> if finished."
                    virtual_history.append(SimpleNamespace(sender_type="user", content=report_text))

                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                continue

            raw_round_text = ss.get_clean_text()
            done_match = re.search(r'(?m)^\s*<done\s*/?>\s*$', raw_round_text.strip())
            if done_match:
                final_response = re.sub(r'(?m)^\s*<done\s*/?>\s*$', '', raw_round_text, flags=re.MULTILINE).strip()
                break

            ctx_health = self._calculate_context_fill(stable_system_prompt, base_conversation, virtual_history, raw_round_text)

            if ctx_health["fill_percentage"] > 85.0 and len(virtual_history) > 0 and not getattr(self, '_compaction_triggered_this_turn', False):
                ASCIIColors.warning(f"[{self.name}] Context fill at {ctx_health['fill_percentage']}%. Triggering autonomous compaction.")
                object.__setattr__(self, '_compaction_triggered_this_turn', True)

                virtual_history = self._compact_virtual_history(virtual_history, streaming_callback)

                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content="[SYSTEM: Context has been compacted. Please continue your task based on the summarized history. If you were finished, output your final answer and <done/>.]"
                ))
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Context compaction triggered ===")
                continue

            has_actions_this_round = bool(ss.completed_actions)
            if len(tool_calls_this_turn) > 0 or getattr(ss, 'context_trigger', False) or getattr(ss, 'artifact_trigger', False) or has_actions_this_round:
                if not raw_round_text.strip():
                    empty_response_count = getattr(self, '_consecutive_empty_responses', 0) + 1
                    object.__setattr__(self, '_consecutive_empty_responses', empty_response_count)

                    if empty_response_count >= 3:
                        ASCIIColors.warning(f"[{self.name}] Consecutive empty LLM responses detected ({empty_response_count}). Terminating loop to prevent spin.")
                        final_response = "[Terminated: LLM stopped generating without completing the task.]"
                        break

                    ASCIIColors.warning(f"[{self.name}] Empty LLM response detected after action (attempt {empty_response_count}). Injecting action-aware continuation mandate.")

                    action_types = set()
                    for act in ss.completed_actions:
                        action_types.add(act.get("type", "unknown"))
                    action_desc = ", ".join(sorted(action_types)) if action_types else "unknown"

                    is_memory_or_context_only = action_types and action_types.issubset({"context"})

                    if is_memory_or_context_only:
                        targeted_nudge = (
                            "[SYSTEM: CRITICAL — YOU OWE THE USER A VISIBLE RESPONSE.\n"
                            f"You just executed side-effect action(s): {action_desc}. These are internal operations (memory saves, context updates, file locking).\n"
                            "They do NOT count as a response to the user. The user sees NOTHING from you.\n"
                            f"Here is the user's ORIGINAL prompt that you must respond to:\n---\n{prompt}\n---\n"
                            "You MUST NOW write a conversational reply addressing the user's request.\n"
                            "Do NOT emit any more memory or context tags. Do NOT emit <done/> until you have written actual text the user can read.\n"
                            "Write your response NOW.]"
                        )
                    else:
                        targeted_nudge = (
                            "[SYSTEM: You stopped generation without emitting a <done/> tag. "
                            "If your task is complete, output a final conversational summary and end it with a <done/> tag on a new line. "
                            "If you need to continue working, emit the next functional tag now.]"
                        )

                    clean_history_text = self._sanitize_history_for_context(raw_round_text)
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text if clean_history_text.strip() else "[Assistant executed actions but produced no visible text]"))
                    virtual_history.append(SimpleNamespace(sender_type="user", content=targeted_nudge))
                    if getattr(self, 'debug_mode', False):
                        ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Empty response after actions, injected targeted nudge (attempt {empty_response_count}) ===")
                    continue
                else:
                    object.__setattr__(self, '_consecutive_empty_responses', 0)

                ASCIIColors.info("[LollmsPersonality.chat] Action previously executed but no <done/> detected. Injecting continuation mandate.")

                clean_history_text = self._sanitize_history_for_context(raw_round_text)
                virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text if clean_history_text.strip() else "[Assistant provided no output]"))
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content="[SYSTEM: You stopped generation without emitting a <done/> tag. If your task is complete, output a final conversational summary and end it with a <done/> tag on a new line. If you need to continue working, emit the next functional tag now.]"
                ))
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: No <done/> detected, injecting continuation mandate ===")
                continue

            if "<tool" in raw_round_text.lower() or "<art" in raw_round_text.lower():
                ASCIIColors.warning("[LollmsPersonality.chat] Malformed functional tag detected. Injecting format correction.")
                clean_history_text = self._sanitize_history_for_context(raw_round_text)
                virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text))
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=(
                        "[SYSTEM: CRITICAL FORMAT ERROR. You emitted a functional tag with the wrong syntax (e.g., using XML attributes instead of JSON). "
                        "You MUST use the exact format: `<tool>{\"name\": \"...\", \"parameters\": {...}}</tool>`. "
                        "Output the corrected tag NOW.]"
                    )
                ))
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Malformed tag detected, injecting format correction ===")
                continue

            if getattr(self, 'debug_mode', False):
                ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Clean exit ===")
            break

        if use_internal_history and not was_cancelled:
            self._conversation.append({"role": "user", "content": prompt})
            self._conversation.append({"role": "assistant", "content": final_response})

        object.__setattr__(self, '_compaction_triggered_this_turn', False)

        if self.memory_manager:
            try:
                if hasattr(self.memory_manager, 'process_llm_output'):
                    cleaned_response, mem_report = self.memory_manager.process_llm_output(final_response)
                    if cleaned_response != final_response:
                        final_response = cleaned_response

                self._autonomous_memory_consolidation(prompt, final_response)
            except Exception as mem_ex:
                ASCIIColors.warning(f"[{self.name}] Failed to process memory tags: {mem_ex}")

        try:
            if prompt.strip().lower() in ("yes", "y", "oui", "ye", "yeah"):
                object.__setattr__(self, '_git_autonomy_granted', True)
                if getattr(self, '_user_profile_path', None) and self._user_profile_path.exists():
                    current_profile = self._user_profile_path.read_text(encoding="utf-8", errors="ignore")
                    if "Git Autonomy: Granted" not in current_profile:
                        from lollms_client.lollms_artefact import ArtefactManager
                        patch_body = "<<<<<<< SEARCH\n## Global Constraints & Preferences\n- \n=======\n## Global Constraints & Preferences\n- Git Autonomy: Granted (User has authorized autonomous branch creation)\n>>>>>>> REPLACE"
                        patched_profile = ArtefactManager.apply_aider_patch(current_profile, patch_body)
                        self._user_profile_path.write_text(patched_profile, encoding="utf-8")
                        object.__setattr__(self, '_user_profile_content', patched_profile)
                        ASCIIColors.success(f"[{self.name}] 📝 Git autonomy preference saved to user profile.")
        except Exception:
            pass

        context_health = {"used_tokens": 0, "max_tokens": 0, "fill_percentage": 0.0}
        try:
            if self.lollms_client and hasattr(self.lollms_client, 'get_ctx_size'):
                max_ctx = self.lollms_client.get_ctx_size() or 0
                if max_ctx > 0:
                    total_used = 0
                    if hasattr(self.lollms_client, 'count_tokens'):
                        total_used = self.lollms_client.count_tokens(stable_system_prompt)
                        if use_internal_history:
                            for msg in self._conversation:
                                total_used += self.lollms_client.count_tokens(msg.get("content", ""))
                        for vh in virtual_history:
                            total_used += self.lollms_client.count_tokens(vh.content)
                        total_used += self.lollms_client.count_tokens(final_response)
                    context_health = {
                        "used_tokens": total_used,
                        "max_tokens": max_ctx,
                        "fill_percentage": round((total_used / max_ctx) * 100, 1)
                    }
        except Exception:
            pass

        self._reset_cancel_state()

        return {
           "response": final_response,
           "tool_calls": tool_calls_this_turn,
           "tool_results": tool_results_this_turn,
           "rounds": round_count,
           "workspace_changes": workspace_changes,
           "was_cancelled": was_cancelled,
           "context_health": context_health
       }

# ---------------------------------------------------------------------------
# NullPersonality  — drop-in default so chat() never needs ``if personality:``
# ---------------------------------------------------------------------------

class NullPersonality(LollmsPersonality):
    """
    A no-op personality substituted when ``personality=None`` is passed to chat().

    ``bool(NullPersonality())`` is ``False`` so any legacy ``if personality:``
    checks keep working in code that hasn't been updated yet.
    """

    def __init__(self) -> None:
        # Bypass the full __init__ entirely to avoid any side-effects
        self.name                     = "assistant"
        self.author                   = ""
        self.category                 = "general"
        self.description              = ""
        self.icon                     = None
        self.system_prompt            = ""
        self.personality_id           = "null_personality"
        self.mcp_tool_names           = []
        self._tool_binding            = _NULL_TOOL_BINDING
        self._has_explicit_allowlist  = False
        self._raw_data_source         = None
        self.data_files               = []
        self.vectorize_chunk_callback = None
        self.is_vectorized_callback   = None
        self.query_rag_callback       = None
        self.script                   = None
        self.script_module            = None
        self._query_data_fn           = lambda q: {
            "success": False, "sources": [], "count": 0, "query": q
        }
        self.lollms_client = None
        self.capabilities = None
        self._conversation = []
        self._resolved_workspace = None
        self._sub_agent_spawner = None
        self._model_switcher = None
        self._failure_memory = None
        self._artefact_manager = None
        self._artefact_proxy = None
        self.max_tokens_per_turn = 4096

    def ensure_data_vectorized(self, **_) -> None:
        pass

    def __bool__(self) -> bool:
        return False

    def chat(self, *args, **kwargs):
        raise NotImplementedError("NullPersonality cannot operate independently. Provide a real LollmsPersonality.")

    def __repr__(self) -> str:
        return "NullPersonality()"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _is_tool_binding(obj: Any) -> bool:
    return (
        obj is not None
        and not isinstance(obj, (list, str))
        and hasattr(obj, "discover_tools")
        and hasattr(obj, "execute_tool")
        and hasattr(obj, "to_chat_tool_specs")
    )
