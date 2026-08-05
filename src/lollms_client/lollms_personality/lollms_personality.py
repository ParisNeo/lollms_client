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

try:
    from lollms_client.lollms_agent.lollms_agent import CapabilityFlags, SubAgentSpawner, ModelSwitcher, BindingToolsBuilder, _build_workspace_context, _normalize_messages, _get_builtin_workspace_tools, _IGNORED_WS_DIRS, _IGNORED_WS_EXTS, _TEXT_EXTS
except ImportError:
    CapabilityFlags = None
    SubAgentSpawner = None
    ModelSwitcher = None
    BindingToolsBuilder = None
    _build_workspace_context = lambda *args, **kwargs: ""
    _normalize_messages = lambda msgs: msgs
    _get_builtin_workspace_tools = lambda *args, **kwargs: {}
    _IGNORED_WS_DIRS = set()
    _IGNORED_WS_EXTS = set()
    _TEXT_EXTS = set()


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
            self._resolved_workspace = self.workspace_path.resolve()
        else:
            self._resolved_workspace = None

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

    def _init_artefact_system(self):
        try:
            from lollms_client.lollms_artefact import ArtefactManager, ArtefactVisibility
            from lollms_client.lollms_artefact.export import LollmsArtefactPatchApplier
            import uuid as _uuid
            ws_path = self._resolved_workspace
            if not ws_path:
                return

            proxy = SimpleNamespace(
                id=f"pers_{self.personality_id[:8]}",
                workspace_path=str(ws_path),
                workspace_data_path=str(ws_path),
                artefacts_metadata_path=str(ws_path / "artefacts_metadata"),
                lollmsClient=self.lollms_client, 
                metadata={},
                _is_db_backed=False,
                commit=lambda: None,
                disable_artefact_versioning=True
            )

            class _VersionlessArtefactManager(ArtefactManager):
                pass

            am = _VersionlessArtefactManager(proxy)
            object.__setattr__(self, '_artefact_manager', am)
            object.__setattr__(self, '_artefact_proxy', proxy)
            object.__setattr__(self, '_patch_applier', LollmsArtefactPatchApplier(self.lollms_client))

            if ws_path.exists():
                for f_path in sorted(ws_path.rglob("*")):
                    if f_path.is_file():
                        rel_parts = f_path.relative_to(ws_path).parts
                        if any(part in _IGNORED_WS_DIRS for part in rel_parts):
                            continue
                        if f_path.suffix.lower() in _IGNORED_WS_EXTS:
                            continue
                        try:
                            content = f_path.read_text(encoding="utf-8", errors="ignore")
                            am.add(
                                title=f_path.name,
                                artefact_type="code" if f_path.suffix.lower() in _TEXT_EXTS else "document",
                                content=content,
                                active=False,
                                visibility=ArtefactVisibility.TREE_UNLOCKABLE
                            )
                        except Exception:
                            pass
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

    def _discover_tools(self, explicit_tools: Optional[Dict], tool_files: Optional[List], enable_code_execution: bool) -> Dict[str, Dict[str, Any]]:
        active_tools = {}

        # Always include file I/O tools when workspace is active to allow the LLM to read files for SEARCH/REPLACE validation
        if self._resolved_workspace and self.capabilities and self.capabilities.enable_workspace_tools:
            active_tools.update(_get_builtin_workspace_tools(include_file_io=True))

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

        if lcp_binding:
            try:
                if enable_code_execution and hasattr(lcp_binding, 'mount_tool_library'):
                    lcp_binding.mount_tool_library('execute_python_code')
                    lcp_tools = lcp_binding.to_chat_tool_specs()
                    code_tool = {k: v for k, v in lcp_tools.items() if k == 'tool_execute_python_code'}
                    active_tools.update(code_tool)
            except Exception:
                pass

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

    def _build_system_prompt(self, active_tools: Dict, enable_code_execution: bool) -> str:
        sys_prompt = self.system_prompt or ""
        rules = (
           "\n=== ACTION EXECUTION & TERMINATION PROTOCOL (CRITICAL) ===\n"
           "1. **INTENT ≠ EXECUTION**: Stating 'I will search...' in text DOES NOT execute the action. You MUST output the `<tool>` tag.\n"
           "2. **MANDATORY TAG EMISSION**: To execute an action, you MUST output the `<tool>` tag immediately.\n"
           "3. **EXPLICIT TERMINATION WITH `<done/>`**: When all objectives are met and tests pass, end with a `<done/>` tag on a new line.\n"
           "4. **SAME-SESSION CONTINUATION**: When executing a sequence, emit the next action's tag in your IMMEDIATE NEXT response.\n"
           "5. **AGENTIC TRIGGER**: If the user requests code generation, file modification, testing, or multi-step work, you MUST enter the agentic loop and use `<tool>` or `<artifact>` tags. Do NOT write code directly in conversational prose.\n"
           "6. **ROUND 1 SHORT-CIRCUIT**: If the user's request is purely conversational (e.g., greetings, simple questions), respond conversationally without `<done/>` or tool tags.\n"
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
            "If a patch fails, the system will return the error. You MUST read the error, re-read the file using `tool_read_file`, and correct your SEARCH block.\n"
            "\n=== SKILLS SYSTEM ===\n"
            "Skills are persistent knowledge capsules stored outside the workspace. They survive across sessions.\n"
            "Use `tool_list_skills` to see available skills, and `tool_load_skill` to load their full content.\n"
            "If you discover a reusable methodology or best practice, use `tool_create_skill` to save it for future use.\n"
            "Use `tool_update_skill` to refine existing skills as you learn more.\n"
            "\n=== SUB-AGENT DELEGATION ===\n"
            "If `tool_spawn_sub_agent` is available, you can delegate complex sub-tasks to a focused child agent.\n"
            "The child shares your workspace but cannot spawn further sub-agents.\n"
            "Use this for heavy tasks like writing large scripts, researching topics, or designing presentations.\n"
            "\n=== THINKING & REASONING CONSTRAINT ===\n"
            "If you output thoughts enclosed in  tags, you MUST output all functional XML tags AFTER the closing tag.\n"
            "\n=== ANTI-MIMICRY PROTOCOL (CRITICAL) ===\n"
            "1. **NEVER OUTPUT SYSTEM MARKERS**: You are STRICTLY FORBIDDEN from generating `<processing>` blocks or `[SYSTEM:` markers.\n"
            "2. **USE REAL TAGS**: To call tools, use the actual `<tool>` XML tags.\n"
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

        workspace_ctx = (""
            "2. **MANDATORY TAG EMISSION**: To execute an action, you MUST output the `<tool>` tag immediately.\n"
            "3. **EXPLICIT TERMINATION WITH `<done/>`**: When finished, end with a `<done/>` tag on a new line.\n"
            "4. **SAME-SESSION CONTINUATION**: When executing a sequence, emit the next action's tag in your IMMEDIATE NEXT response.\n"
            "5. **ROUND 1 SHORT-CIRCUIT**: If the user's request is purely conversational, respond conversationally without `<done/>`.\n"
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
            "If a patch fails, the system will return the error. You MUST read the error, re-read the file using `tool_read_file`, and correct your SEARCH block.\n"
            "\n=== SKILLS SYSTEM ===\n"
            "Skills are persistent knowledge capsules stored outside the workspace. They survive across sessions.\n"
            "Use `tool_list_skills` to see available skills, and `tool_load_skill` to load their full content.\n"
            "If you discover a reusable methodology or best practice, use `tool_create_skill` to save it for future use.\n"
            "Use `tool_update_skill` to refine existing skills as you learn more.\n"
            "\n=== SUB-AGENT DELEGATION ===\n"
            "If `tool_spawn_sub_agent` is available, you can delegate complex sub-tasks to a focused child agent.\n"
            "The child shares your workspace but cannot spawn further sub-agents.\n"
            "Use this for heavy tasks like writing large scripts, researching topics, or designing presentations.\n"
            "\n=== THINKING & REASONING CONSTRAINT ===\n"
            "If you output thoughts enclosed in  tags, you MUST output all functional XML tags AFTER the closing tag.\n"
            "\n=== ANTI-MIMICRY PROTOCOL (CRITICAL) ===\n"
            "1. **NEVER OUTPUT SYSTEM MARKERS**: You are STRICTLY FORBIDDEN from generating `<processing>` blocks or `[SYSTEM:` markers.\n"
            "2. **USE REAL TAGS**: To call tools, use the actual `<tool>` XML tags.\n"
        )

        workspace_ctx = ""
        if getattr(self, '_artefact_manager', None):
            try:
                zone = self._artefact_manager.build_artefacts_context_zone()
                if zone:
                    workspace_ctx = "\n" + zone
            except Exception:
                pass
        elif self._resolved_workspace:
            workspace_ctx = "\n" + _build_workspace_context(self._resolved_workspace)

        skills_ctx = ""
        if self.skills_manager:
            skills_ctx_str = self.skills_manager.build_context()
            if skills_ctx_str:
                skills_ctx = "\n" + skills_ctx_str

        tool_desc = ""
        if active_tools:
            tool_desc = "\n=== TOOLS AVAILABLE ===\nTo use a tool, emit `<tool>{\"name\": \"...\", \"parameters\": {...}}</tool>`.\n\nAvailable tools:\n"
            for t_name, t_spec in active_tools.items():
                desc = t_spec.get("description", "")
                params_list = t_spec.get("parameters", [])
                param_desc = ", ".join([f"{p['name']}: {p['type']}" for p in params_list])
                tool_desc += f"- {t_name}({param_desc}): {desc}\n"

        return sys_prompt + "\n" + rules + workspace_ctx + skills_ctx + tool_desc

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
        enable_code_execution: Optional[bool] = None,
        enable_artefacts: bool = True,
        use_internal_history: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Independent agentic chat loop. Used when the personality operates outside of a LollmsDiscussion.
        """
        if lollms_client is not None:
            self.lollms_client = lollms_client

        if self.lollms_client is None:
            raise RuntimeError(f"[{self.name}] Independent chat requires a lollms_client instance.")

        self._reset_cancel_state()

        if self._sub_agent_spawner:
            self._sub_agent_spawner.reset_turn()

        code_exec = enable_code_execution if enable_code_execution is not None else (self.capabilities.enable_code_execution if self.capabilities else False)

        if self._failure_memory and hasattr(self._failure_memory, '_signatures'):
            self._failure_memory._signatures.clear()
            if hasattr(self._failure_memory, 'failures'):
                self._failure_memory.failures = []

        if self._resolved_workspace and enable_artefacts and not hasattr(self, '_artefact_manager'):
            self._init_artefact_system()

        active_tools = self._discover_tools(tools, tool_files or [], code_exec)
        full_system_prompt = self._build_system_prompt(active_tools, code_exec)

        # ── Pre-Turn Memory Hydration ──
        # Pull relevant memories from the LollmsMemoryManager and inject them into the system prompt
        # so the LLM can "remember" facts across sessions (e.g., the user's name).
        if self.memory_manager:
            try:
                # Pull deep memories relevant to the current prompt
                if hasattr(self.memory_manager, 'auto_pull_deep_memories'):
                    self.memory_manager.auto_pull_deep_memories(prompt)

                # Build the memory context block and append it to the system prompt
                if hasattr(self.memory_manager, 'build_working_zone'):
                    mem_zone = self.memory_manager.build_working_zone()
                    if mem_zone:
                        full_system_prompt += "\n=== ACTIVE MEMORIES (PERSISTENT ACROSS SESSIONS) ===\n" + mem_zone + "\n=== END MEMORIES ===\n"
            except Exception as mem_ex:
                ASCIIColors.warning(f"[{self.name}] Failed to hydrate memories: {mem_ex}")

        if use_internal_history:
            base_conversation = list(self._conversation)
        else:
            base_conversation = []
        base_conversation.append({"role": "user", "content": prompt})

        virtual_history: List[SimpleNamespace] = []
        tool_calls_this_turn: List[Dict[str, Any]] = []
        tool_results_this_turn: List[Dict[str, Any]] = []
        round_count = 0
        was_cancelled = False
        tool_signature_counts: Dict[str, int] = {}
        successful_tool_signatures: set = set()
        final_response = ""
        workspace_changes: List[Dict[str, Any]] = []

        while round_count < max_reasoning_steps:
            if self.is_generation_cancelled():
                was_cancelled = True
                break

            round_count += 1

            if hasattr(self.lollms_client, 'llm') and hasattr(self.lollms_client.llm, 'reset_cancel'):
                try:
                    self.lollms_client.llm.reset_cancel()
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
                self.lollms_client.generate_from_messages(
                    messages=messages,
                    stream=True,
                    streaming_callback=_inline_relay,
                    **gen_kwargs
                )
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

            ss.flush_remaining_buffer()

            if ss.was_done_detected():
                final_response = ss.get_clean_text()
                final_response = re.sub(r'(?m)^\s*<done\s*/?>', '', final_response, flags=re.IGNORECASE).strip()
                break

            if ss.tool_trigger:
                tool_call_json_str = ss.get_tool_call_json()
                if tool_call_json_str:
                    try:
                        call_data = json.loads(tool_call_json_str)
                        tool_name = call_data.get("name", "")
                        tool_params = call_data.get("parameters", {})

                        raw_round_text = ss.get_clean_text()
                        clean_history_text = self._sanitize_history_for_context(raw_round_text)
                        virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text))

                        if not active_tools or tool_name not in active_tools:
                            virtual_history.append(SimpleNamespace(sender_type="user", content=f"Tool '{tool_name}' not available. Use one of: {list(active_tools.keys())}"))
                            continue

                        param_sig = json.dumps(tool_params, sort_keys=True, default=str)
                        context_aware_sig = f"{tool_name}::{param_sig}"
                        if context_aware_sig in successful_tool_signatures:
                            virtual_history.append(SimpleNamespace(
                                sender_type="user",
                                content=f"Repetitive call to '{tool_name}' blocked. Output already in context. Finish with <done/>."
                            ))
                            continue

                        files_before = self._take_workspace_snapshot()
                        tool_res = self._execute_tool(tool_name, tool_params, active_tools)
                        files_after = self._take_workspace_snapshot()

                        changes = self._sync_workspace(files_before, files_after)
                        if changes:
                            workspace_changes.extend(changes)

                        tool_success = isinstance(tool_res, dict) and tool_res.get("success", True) is not False
                        if tool_success:
                            successful_tool_signatures.add(context_aware_sig)

                        tool_calls_this_turn.append({"round": round_count, "name": tool_name, "parameters": tool_params})
                        tool_results_this_turn.append({"round": round_count, "name": tool_name, "result": tool_res, "success": tool_success})
                        clean_result_str = _sanitize_tool_result(tool_res)

                        if tool_success:
                            user_part = (
                                f"=== ✅ TOOL RESULT: {tool_name} ===\n"
                                f"⚠️ This is a RESULT, not a new call.\n"
                                f"<tool_result name=\"{tool_name}\" status=\"SUCCESS\">\n{clean_result_str}\n</tool_result>\n"
                                f"Analyze this data and respond, or emit <done/> if finished."
                            )
                        else:
                            user_part = (
                                f"=== ❌ TOOL FAILED: {tool_name} ===\n"
                                f"<tool_result name=\"{tool_name}\" status=\"FAILED\">\n{clean_result_str}\n</tool_result>\n"
                                f"Analyze the error and try a different approach, or emit <done/> if stuck."
                            )
                        virtual_history.append(SimpleNamespace(sender_type="user", content=user_part))
                        continue
                    except Exception as e:
                        final_response = f"[Tool execution error: {e}]"
                        break
            elif ss.artifact_trigger:
                raw_artifact_xml = ss.get_artifact_xml()
                if not raw_artifact_xml:
                    final_response = ss.get_clean_text()
                    break

                virtual_history.append(SimpleNamespace(sender_type="assistant", content=raw_artifact_xml))

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

                    if "<<<<<<< SEARCH" in body_content:
                        if not hasattr(self, '_patch_applier') or not self._patch_applier:
                            user_part = "[SYSTEM ERROR] Patch applier not initialized. Cannot apply SEARCH/REPLACE."
                            virtual_history.append(SimpleNamespace(sender_type="user", content=user_part))
                            continue

                        file_path = self._resolved_workspace / title
                        if not file_path.exists():
                            user_part = f"[SYSTEM ERROR] File '{title}' not found. Cannot apply patch."
                            virtual_history.append(SimpleNamespace(sender_type="user", content=user_part))
                            continue

                        original_content = file_path.read_text(encoding="utf-8", errors="ignore")
                        result = self._patch_applier.apply_aider_patch(original_content, body_content, file_name=title, language=lang)

                        if result.get("success"):
                            file_path.write_text(result["patched_content"], encoding="utf-8")
                            user_part = f"✅ SEARCH/REPLACE applied successfully to {title}."
                            if self._artefact_manager:
                                self._artefact_manager.update(title=title, new_content=result["patched_content"], language=lang, bump_version=True, active=True)
                        else:
                            user_part = f"❌ SEARCH/REPLACE FAILED for {title}.\nError: {result.get('error')}\n\nPlease read the file using tool_read_file and correct your SEARCH block."

                        virtual_history.append(SimpleNamespace(sender_type="user", content=user_part))
                        continue
                    else:
                        if self._artefact_manager:
                            self._artefact_manager.add(title=title, artefact_type="code", content=body_content, language=lang, active=True)
                        file_path = self._resolved_workspace / title
                        file_path.write_text(body_content, encoding="utf-8")
                        user_part = f"✅ File {title} created/updated successfully."
                        virtual_history.append(SimpleNamespace(sender_type="user", content=user_part))
                        continue
                except Exception as e:
                    user_part = f"[SYSTEM ERROR] Failed to process artifact tag: {e}"
                    virtual_history.append(SimpleNamespace(sender_type="user", content=user_part))
                    continue
            else:
                raw_round_text = ss.get_clean_text()
                done_match = re.search(r'(?m)^\s*<done\s*/?>\s*$', raw_round_text.strip())
                if done_match:
                    final_response = re.sub(r'(?m)^\s*<done\s*/?>\s*$', '', raw_round_text, flags=re.MULTILINE).strip()
                    break

                # 🛑 CONTINUATION MANDATE (NO <done/>)
                if len(tool_calls_this_turn) > 0:
                    ASCIIColors.info("[LollmsPersonality.chat] Tool previously executed but no <done/> detected. Injecting continuation mandate.")
                    clean_history_text = self._sanitize_history_for_context(raw_round_text)
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text))
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content="[SYSTEM: You stopped generation without emitting a <done/> tag. If your task is complete, output a final conversational summary and end it with a <done/> tag on a new line. If you need to continue working, emit the next functional tag now.]"
                    ))
                    continue

                # 🛑 INTENT DETECTION (Migrated from LollmsDiscussion)
                # If the LLM states an intent to use a tool but stops without emitting the tag, force a continuation.
                intent_pattern = re.compile(r'(let me|now i|next i|i will|i need to|we need to).*(query|get|fetch|build|create|analyze|summarize|aggregate|plot|explore|check|read|list|write|update|fix|run|execute)', re.IGNORECASE)
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
                    ASCIIColors.info(f"[LollmsPersonality.chat] Detected pending tool intent without XML tag. Forcing continuation...")

                    clean_history_text = self._sanitize_history_for_context(raw_round_text)
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=clean_history_text))

                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content="[SYSTEM: CRITICAL. You stopped generation before executing your stated intent. Output the <tool> or <artifact> tag NOW. Do not write any more prose.]"
                    ))

                    continue

                final_response = raw_round_text
                break

        if use_internal_history and not was_cancelled:
            self._conversation.append({"role": "user", "content": prompt})
            self._conversation.append({"role": "assistant", "content": final_response})

        # ── 12. Memory Post-Processing ──
        if self.memory_manager:
            try:
                if hasattr(self.memory_manager, 'process_llm_output'):
                    cleaned_response, mem_report = self.memory_manager.process_llm_output(final_response)
                    if cleaned_response != final_response:
                        final_response = cleaned_response
 
                # Save episodic memory (interaction history)
                if hasattr(self.memory_manager, 'add'):
                    from datetime import datetime
                    clean_ai = re.sub(r'<[^>]+>', '', final_response).strip()
                    clean_user = prompt.strip()
                    if clean_user and clean_ai and len(clean_user) > 5 and len(clean_ai) > 5:
                        episode = f"Event/Interaction on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC:\nUser asked: \"{clean_user}\"\nAI responded: \"{clean_ai}\""
                        self.memory_manager.add(content=episode, importance=0.6, tags=["episode", "interaction"], level=1)
            except Exception as mem_ex:
                ASCIIColors.warning(f"[{self.name}] Failed to process memory tags: {mem_ex}")

        # ── 13. Context Health Telemetry ──
        context_health = {"used_tokens": 0, "max_tokens": 0, "fill_percentage": 0.0}
        try:
            if self.lollms_client and hasattr(self.lollms_client, 'get_ctx_size'):
                max_ctx = self.lollms_client.get_ctx_size() or 0
                if max_ctx > 0:
                    total_used = 0
                    if hasattr(self.lollms_client, 'count_tokens'):
                        total_used = self.lollms_client.count_tokens(full_system_prompt)
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
