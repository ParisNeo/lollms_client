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

import ast
import builtins
import base64
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import traceback
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ascii_colors import ASCIIColors, trace_exception

from .skills_manager import SkillsManager
from .handbag import Handbag
from .lollms_agent_state import _AgentStreamState, _sanitize_tool_result, _ToolsManager

if not callable(getattr(builtins, 'compile', None)) or builtins.compile.__module__ != 'builtins':
    import importlib as _importlib
    _builtins_mod = _importlib.import_module('builtins')
    if hasattr(_builtins_mod, 'compile') and _builtins_mod.compile.__module__ == 'builtins':
        builtins.compile = _builtins_mod.compile
    else:
        ASCIIColors.error("[LollmsPersonality] CRITICAL: builtins.compile is shadowed or missing. Tool execution may fail.")

_compile = getattr(builtins, 'compile', None)
if _compile is None or not callable(_compile) or _compile.__module__ != 'builtins':
    ASCIIColors.error("[LollmsPersonality] CRITICAL: Could not restore native compile(). exec() fallback will be used.")
    _compile = None


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
from lollms_client.lollms_artefact import ArtefactVisibility, ArtefactManager
from lollms_client.lollms_artefact.lollms_artefact import ArtefactManager as _ArtefactManager
from lollms_client.lollms_history import HistoryManager
from lollms_client.lollms_history.lollms_history import HistoryManager


_TEXT_RAG_EXTS = {
    ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml", ".html",
    ".py", ".js", ".ts", ".rs", ".go", ".rb", ".php", ".java", ".kt",
    ".swift", ".c", ".cpp", ".h", ".hpp", ".sql", ".sh", ".bash",
    ".ps1", ".bat", ".toml", ".ini", ".cfg", ".log", ".rdf", ".ttl",
}

class _NullArtefactManager:
    """Null-safe stand-in for ArtefactManager when no workspace is configured."""
    def get_context_images(self) -> list:
        return []


class _HistoryContextAdapter:
    """
    Adapter to provide LollmsDiscussion-like interface to HistoryManager
    for LollmsPersonality context generation.
    """
    def __init__(self, personality: 'LollmsPersonality', stable_system_prompt: str):
        self._personality = personality
        self._system_prompt_ref = stable_system_prompt
        self.lollmsClient = personality.lollms_client
        self.scratchpad = getattr(personality, '_scratchpad_content', '')
        self.pruning_summary = None
        self.pruning_point_id = None
        self.artefacts = getattr(personality, '_artefact_manager', None) or _NullArtefactManager()
        self.memory_manager = personality.memory_manager
        self.workspace_data_path = str(personality._resolved_workspace) if personality._resolved_workspace else "."

    @property
    def _system_prompt(self) -> str:
        return self._system_prompt_ref

    def get_full_data_zone(self) -> str:
        return ""

    def get_discussion_images(self) -> list:
        return []

    def _apply_three_view_protocol(self, msg, raw_content: str, distance_from_end: int = 0) -> str:
        return raw_content

    def _build_memory_context_block(self, memory_manager, token_counter=None) -> str:
        if not memory_manager:
            return ""
        try:
            if hasattr(memory_manager, 'build_working_zone'):
                return memory_manager.build_working_zone(token_counter=token_counter)
        except Exception:
            pass
        return ""

    def _inject_memory_into_messages(self, messages, memory_manager, format_type, token_counter):
        if not memory_manager:
            return messages
        try:
            if hasattr(memory_manager, 'inject_into_messages'):
                return memory_manager.inject_into_messages(messages, format_type, token_counter=token_counter)
        except Exception:
            pass
        return messages

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

_IGNORED_WS_DIRS = {"__pycache__", ".venv", "venv", ".git", ".idea", ".vscode", "node_modules", ".lollms", "build", "dist", ".next", "env", ".env", ".lollms_code", ".lollms_metadata", "egg-info", "dist-info", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", "site-packages", "artefacts_metadata", "discussions", ".git"}
_IGNORED_WS_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib"}
_TEXT_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss", ".sql", ".md", ".txt", ".json", ".yaml", ".yml", ".xml", ".csv", ".log", ".toml", ".ini", ".cfg", ".sh", ".bash", ".ps1", ".bat", ".rdf", ".ttl", ".rs", ".go", ".rb", ".php", ".java", ".kt", ".swift", ".c", ".cpp", ".h", ".hpp"}
_BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp", ".zip", ".tar", ".gz", ".pdf", ".docx", ".mp3", ".wav", ".mp4", ".avi", ".mov"}

_MAX_TREE_DEPTH = 2
_MAX_DIR_ITEMS = 15


def _build_workspace_tree_r(directory: Path, workspace_root: Path, current_depth: int, collapsed_set: set, max_depth: int, max_items: int) -> List[str]:
    if current_depth >= max_depth:
        return []

    entries = []
    try:
        sorted_items = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except Exception:
        return []

    if len(sorted_items) > max_items:
        remaining = len(sorted_items) - max_items
        sorted_items = sorted_items[:max_items]
        entries.append(f"{'  ' * current_depth}... ({remaining} more items in this folder. Use <uncollapse_folder> to see them.)")

    for item in sorted_items:
        if item.name in _IGNORED_WS_DIRS or item.name.startswith("."):
            continue
        if item.is_dir():
            rel_dir_path = str(item.relative_to(workspace_root)).replace("\\", "/")
            if rel_dir_path in collapsed_set:
                entries.append(f"{'  ' * current_depth}[📁 COLLAPSED] {item.name}/ ({_count_files_recursive(item, collapsed_set, max_depth, current_depth)} items)")
            elif current_depth + 1 >= max_depth:
                entries.append(f"{'  ' * current_depth}[📁 DEEP] {item.name}/ (Use <uncollapse_folder> to explore)")
            else:
                entries.append(f"{'  ' * current_depth}[📁] {item.name}/")
                entries.extend(_build_workspace_tree_r(item, workspace_root, current_depth + 1, collapsed_set, max_depth, max_items))
        elif item.is_file():
            if item.suffix.lower() in _IGNORED_WS_EXTS:
                continue
            rel_path = str(item.relative_to(workspace_root)).replace("\\", "/")
            size = item.stat().st_size
            entries.append(f"{'  ' * current_depth}- {rel_path} ({size:,} bytes)")

    return entries

def _count_files_recursive(directory: Path, collapsed_set: set, max_depth: int, current_depth: int) -> int:
    if current_depth >= max_depth:
        return 0
    count = 0
    try:
        for item in directory.iterdir():
            if item.name in _IGNORED_WS_DIRS or item.name.startswith("."):
                continue
            if item.is_dir():
                count += _count_files_recursive(item, collapsed_set, max_depth, current_depth + 1)
            elif item.is_file():
                if item.suffix.lower() not in _IGNORED_WS_EXTS:
                    count += 1
    except Exception:
        pass
    return count

def _build_workspace_context(workspace_path: Path, max_file_size: int = 12000, max_total_chars: int = 30000, collapsed_folders: Optional[set] = None) -> str:
    if not workspace_path or not workspace_path.exists():
        return ""

    collapsed_set = collapsed_folders or set()
    lines = ["=== WORKSPACE TREE ==="]

    tree_entries = _build_workspace_tree_r(
        directory=workspace_path,
        workspace_root=workspace_path,
        current_depth=0,
        collapsed_set=collapsed_set,
        max_depth=_MAX_TREE_DEPTH,
        max_items=_MAX_DIR_ITEMS
    )

    if not tree_entries:
        lines.append("(Workspace is empty)")
        lines.append("=== END WORKSPACE TREE ===")
        return "\n".join(lines)

    lines.extend(tree_entries)
    lines.append("=== END WORKSPACE TREE ===")
    return "\n".join(lines)




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
# RAGDataSource — Multi-source RAG Knowledge Base Schema
# ===========================================================================

@dataclass
class RAGDataSource:
    """
    Represents a named, described RAG data source with query resolution.
    """
    name: str
    description: str = ""
    query_fn: Optional[Callable] = None
    store: Optional[Any] = None
    auto_query: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
        if not self.query_fn:
            return {
                "success": False,
                "sources": [],
                "count": 0,
                "query": query_text,
                "datasource_name": self.name
            }
        try:
            raw = _call_query_engine(self.query_fn, query_text, store=self.store, **kwargs)
            return _normalise_raw(raw, query_text, self.name)
        except Exception as e:
            trace_exception(e)
            return {
                "success": False,
                "sources": [],
                "count": 0,
                "query": query_text,
                "error": str(e),
                "datasource_name": self.name
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "auto_query": self.auto_query,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGDataSource":
        return cls(
            name=data.get("name", "knowledge_base"),
            description=data.get("description", ""),
            query_fn=data.get("query_fn") or data.get("source") or data.get("callable"),
            store=data.get("store") or data.get("ss"),
            auto_query=data.get("auto_query", True),
            metadata=data.get("metadata", {})
        )


def _call_query_engine(query_fn: Callable, query: str, store: Any = None, **kwargs) -> Any:
    """
    Dynamically calls query_fn matching signatures such as:
      - query_fn(query)
      - query_fn(query, ss, ...)
      - query_fn(query, store, **kwargs)
    """
    if not callable(query_fn):
        return str(query_fn)

    sig = None
    try:
        sig = inspect.signature(query_fn)
    except Exception:
        pass

    if sig:
        param_names = list(sig.parameters.keys())
        call_kwargs = {}

        if len(param_names) >= 2 and param_names[1] in ("ss", "store", "data_store", "storage", "database"):
            positional_args = [query, store]
            for k, v in kwargs.items():
                if k in param_names[2:]:
                    call_kwargs[k] = v
            try:
                return query_fn(*positional_args, **call_kwargs)
            except TypeError:
                pass

        for k, v in kwargs.items():
            if k in param_names:
                call_kwargs[k] = v

        if "ss" in param_names and "ss" not in call_kwargs and store is not None:
            call_kwargs["ss"] = store
        elif "store" in param_names and "store" not in call_kwargs and store is not None:
            call_kwargs["store"] = store

        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_keyword:
            call_kwargs.update(kwargs)
            if store is not None and "ss" not in call_kwargs:
                call_kwargs["ss"] = store

        try:
            return query_fn(query, **call_kwargs)
        except TypeError:
            if store is not None:
                try:
                    return query_fn(query, store)
                except TypeError:
                    return query_fn(query)
            return query_fn(query)
    else:
        if store is not None:
            try:
                return query_fn(query, store, **kwargs)
            except TypeError:
                try:
                    return query_fn(query, store)
                except TypeError:
                    return query_fn(query)
        try:
            return query_fn(query, **kwargs)
        except TypeError:
            return query_fn(query)


def _normalise_raw(raw: Any, query: str, source_label: str) -> Dict[str, Any]:
    """Normalizes raw RAG outputs (dicts, lists of chunks, strings) into standard format."""
    if isinstance(raw, dict) and "sources" in raw:
        if "success" not in raw:
            raw["success"] = True
        raw.setdefault("query", query)
        raw.setdefault("count", len(raw["sources"]))
        raw.setdefault("datasource_name", source_label)
        return raw

    if isinstance(raw, list):
        sources = []
        for chunk in raw:
            if isinstance(chunk, dict):
                if "error" in chunk and len(chunk) == 1 and not chunk.get("content"):
                    continue
                content = (
                    chunk.get("content") or
                    chunk.get("chunk_text") or
                    chunk.get("text") or
                    chunk.get("snippet") or
                    str(chunk)
                )
                title = (
                    chunk.get("title") or
                    chunk.get("name") or
                    (Path(chunk.get("file_path", "")).name if chunk.get("file_path") else "") or
                    source_label
                )
                score = chunk.get("score", chunk.get("similarity_percent", chunk.get("fused_score", chunk.get("value", 1.0))))
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 1.0

                sources.append({
                    "content":  content,
                    "score":    score,
                    "source":   title or source_label,
                    "metadata": chunk.get("document_metadata", chunk.get("metadata", {})),
                    "title":    title,
                    "datasource_name": source_label
                })
            else:
                sources.append({
                    "content": str(chunk),
                    "score": 1.0,
                    "source": source_label,
                    "metadata": {},
                    "title": source_label,
                    "datasource_name": source_label
                })
        return {
            "success": True,
            "sources": sources,
            "count": len(sources),
            "query": query,
            "datasource_name": source_label
        }

    text = str(raw) if raw is not None else ""
    return {
        "success": bool(text),
        "sources": [{"content": text, "score": 1.0, "source": source_label, "title": source_label, "datasource_name": source_label}] if text else [],
        "count":   1 if text else 0,
        "query":   query,
        "datasource_name": source_label
    }


# ===========================================================================
# AgentRole & CapabilityFlags
# ===========================================================================

class AgentRole:
    PROPOSER = "proposer"
    CRITIC = "critic"
    DEVIL_ADVOCATE = "devil_advocate"
    DOMAIN_EXPERT = "domain_expert"
    SYNTHESIZER = "synthesizer"
    MODERATOR = "moderator"
    IMPLEMENTER = "implementer"
    TESTER = "tester"
    NARRATOR = "narrator"
    PLAYER = "player"
    FREEFORM = "freeform"


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
        try:
            exec(compile(content, str(fp), "exec"), module.__dict__)
        except TypeError as te:
            if "compile()" in str(te):
                ASCIIColors.warning(f"[ToolsManager] compile() signature error for {fp.name}. Falling back to direct exec. Error: {te}")
                exec(content, module.__dict__)
            else:
                raise
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
            child_caps = CapabilityFlags(
                enable_code_execution=self.parent.capabilities.enable_code_execution if self.parent.capabilities else False,
                enable_image_generation=False,
                enable_image_editing=False,
                enable_sub_agents=False,  # Prevent infinite recursion
                enable_model_switching=False,
                enable_skill_loading=self.parent.capabilities.enable_skill_loading if self.parent.capabilities else True,
                enable_skill_creation=False,
                skills_mode="loadable",
                max_sub_agent_depth=0,
            )

            child_agent = LollmsPersonality(
                name=f"SubAgent_{self._spawned_this_turn}",
                author="lollms_personality",
                category="sub_agent",
                description="A focused sub-agent spawned for a specific task.",
                system_prompt=personality_conditioning or (
                    "You are a focused sub-agent. Execute the given task precisely and return the result. "
                    "Do not engage in conversational pleasantries. Focus solely on the task."
                ),
                role=AgentRole.IMPLEMENTER,
                workspace_path=self.parent.get_workspace_path(),
                capabilities=child_caps,
                skills_manager=self.parent.skills_manager,
                model_params=self.parent.model_params,
                max_tokens_per_turn=self.parent.max_tokens_per_turn,
                memory_manager=None,
                lollms_client=self.parent.lollms_client,
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
                pm.ensure_packages("safestore")

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
        name: str = "assistant",
        author: str = "",
        category: str = "general",
        description: str = "",
        system_prompt: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        icon: Optional[str] = None,
        tools: Optional[Any] = None,
        data_source: Optional[Union[str, Callable, Dict[str, Any], List[Any], RAGDataSource]] = None,
        data_sources: Optional[Union[List[Any], Dict[str, Any]]] = None,
        data_files: Optional[List[Union[str, Path]]] = None,
        vectorize_chunk_callback: Optional[Callable[[str, str], None]] = None,
        is_vectorized_callback: Optional[Callable[[str], bool]] = None,
        query_rag_callback: Optional[Callable] = None,
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
        role: str = AgentRole.IMPLEMENTER,
        model_params: Optional[Dict[str, Any]] = None,
        enable_artefact_system: bool = False,
        disable_artefact_versioning: bool = False,
        skills_dirs: Optional[List[Union[str, Path]]] = None,
        _parent_depth: int = 0,
        lc: Optional[Any] = None,
        personality: Optional[Any] = None,
    ):
        if personality is not None:
            if name == "assistant" and hasattr(personality, "name"):
                name = personality.name
            if not author and hasattr(personality, "author"):
                author = personality.author
            if category == "general" and hasattr(personality, "category"):
                category = personality.category
            if not description and hasattr(personality, "description"):
                description = personality.description
            if not system_prompt and hasattr(personality, "system_prompt"):
                system_prompt = personality.system_prompt
            if metadata is None and hasattr(personality, "metadata"):
                metadata = personality.metadata
            if icon is None and hasattr(personality, "icon"):
                icon = personality.icon
            if tools is None and hasattr(personality, "tools"):
                tools = personality.tools
            if data_source is None and hasattr(personality, "data_source"):
                data_source = personality.data_source
            if data_sources is None and hasattr(personality, "data_sources"):
                data_sources = personality.data_sources
            if skills_manager is None and hasattr(personality, "skills_manager"):
                skills_manager = personality.skills_manager
            if memory_manager is None and hasattr(personality, "memory_manager"):
                memory_manager = personality.memory_manager
            if capabilities is None and hasattr(personality, "capabilities"):
                capabilities = personality.capabilities
            if model_params is None and hasattr(personality, "model_params"):
                model_params = personality.model_params
            if lollms_client is None and hasattr(personality, "lollms_client"):
                lollms_client = personality.lollms_client

        resolved_client = lc or lollms_client

        self.name = name or "assistant"
        self.author = author or ""
        self.category = category or "general"
        self.description = description or ""
        self.system_prompt = system_prompt or ""
        self.metadata = metadata or {}
        self.icon = icon
        self.personality_id = personality_id or self._generate_id()
        self.role = role
        self.model_params = model_params or {}

        self.mcp_tool_names: List[str] = []
        self._tool_binding: Any = _NULL_TOOL_BINDING
        self._has_explicit_allowlist: bool = False
        self._init_tools(tools)

        self._raw_data_source = data_source
        self.data_files = [Path(f) for f in (data_files or [])]
        self.vectorize_chunk_callback = vectorize_chunk_callback
        self.is_vectorized_callback = is_vectorized_callback
        self.query_rag_callback = query_rag_callback
        self.data_sources: List[RAGDataSource] = []
        self._init_data_sources(data_source, data_sources, query_rag_callback)
        self._query_data_fn = self._build_query_data_fn(data_source)

        self.script = script
        self.script_module = None
        self._prepare_script()

        # Unified Stateful Components
        self.handbag_path = Path(handbag_path) if handbag_path else None

        # Skills initialization
        if skills_manager:
            self.skills_manager = skills_manager
            if skills_dirs:
                self.skills_manager._skills_dirs.extend([Path(d).resolve() for d in skills_dirs if Path(d).exists()])
                self.skills_manager.reload()
        elif skills_dirs:
            self.skills_manager = SkillsManager(skills_dirs=skills_dirs, mode="mixed", max_visible_tokens=8000)
        else:
            self.skills_manager = None

        # Pre-inject skills context into the system prompt so that
        # external callers (like LollmsDiscussion) inherit the skills
        # without needing to invoke the private _build_system_prompt().
        if self.skills_manager:
            skills_ctx_str = self.skills_manager.build_context()
            if skills_ctx_str:
                if "=== SKILLS SYSTEM ===" not in self.system_prompt:
                    self.system_prompt = f"{self.system_prompt}\n\n{skills_ctx_str}".strip()
                    self._skills_context_injected = True
                else:
                    self._skills_context_injected = True
            else:
                self._skills_context_injected = False
        else:
            self._skills_context_injected = False

        self.memory_manager = memory_manager
        self._workspace_path: Optional[Path] = None
        self.workspace_path = Path(workspace_path) if workspace_path else None
        self.enable_git_management = enable_git_management
        self.coworkers: Dict[str, 'LollmsPersonality'] = {}

        self.lollms_client = resolved_client
        self.max_tokens_per_turn = max_tokens_per_turn
        self.enable_artefact_system = enable_artefact_system
        self.disable_artefact_versioning = disable_artefact_versioning

        # Capabilities
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

        # Initialize SubAgentSpawner and ModelSwitcher if client is provided
        if SubAgentSpawner and ModelSwitcher and self.lollms_client:
            self._sub_agent_spawner = SubAgentSpawner(
                parent_agent=self,
                max_depth=self.capabilities.max_sub_agent_depth if self.capabilities else 3,
                max_per_turn=self.capabilities.max_sub_agents_per_turn if self.capabilities else 5
            )
            self._sub_agent_spawner.set_depth(_parent_depth)
            self._model_switcher = ModelSwitcher(self.lollms_client)
        else:
            self._sub_agent_spawner = None
            self._model_switcher = None

        if self.enable_artefact_system and self._resolved_workspace:
            self._init_artefact_system()

        self.ensure_data_vectorized()

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def _agent_id(self) -> str:
        return self.personality_id

    @property
    def lc(self) -> Optional[Any]:
        return self.lollms_client

    @lc.setter
    def lc(self, value: Optional[Any]) -> None:
        self.lollms_client = value

    def clear_conversation(self) -> None:
        """Clears the agent's internal multi-turn conversation memory."""
        self._conversation = []

    def save_history_to_disk(self, history_file: Path) -> None:
        """Persists the internal conversation history to a JSON file."""
        if not history_file:
            return
        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            history_file.write_text(
                _json.dumps(self._conversation, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] Failed to save history to disk: {e}")

    def load_history_from_disk(self, history_file: Path) -> None:
        """Loads the internal conversation history from a JSON file."""
        if not history_file or not history_file.exists():
            self._conversation = []
            return
        try:
            import json as _json
            data = _json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._conversation = data
            else:
                self._conversation = []
        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] Failed to load history from disk: {e}")
            self._conversation = []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        n_predict: Optional[int] = None,
        streaming_callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """Direct text generation proxy."""
        if not self.lollms_client:
            raise RuntimeError("lollms_client is required for text generation.")
        return self.lollms_client.generate_text(
            prompt=prompt,
            system_prompt=system_prompt if system_prompt is not None else self.system_prompt,
            temperature=temperature if temperature is not None else self.model_params.get("temperature", 0.7),
            n_predict=None,
            streaming_callback=streaming_callback,
            **kwargs
        )

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """Direct structured JSON generation proxy."""
        if not self.lollms_client:
            raise RuntimeError("lollms_client is required for structured generation.")
        return self.lollms_client.generate_structured_content(
            prompt=prompt,
            schema=schema,
            temperature=temperature,
            **kwargs
        )

    def generate_with_tools(
        self,
        prompt: str,
        tools: Optional[List[Union[str, Path, Dict[str, Any]]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        n_predict: Optional[int] = None,
        max_tool_rounds: int = 10,
        streaming_callback: Optional[Callable] = None,
        auto_execute: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Executes an agentic tool reasoning loop."""
        if not auto_execute:
            # Single-pass manual mode: delegate to client.generate_from_messages
            messages = [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            response_text = self.lollms_client.generate_from_messages(
                messages=messages,
                temperature=temperature or 0.7,
                n_predict=None,
                **kwargs
            )
            tool_calls = []
            m = re.search(r'<tool>(.*?)</tool>', response_text, re.DOTALL | re.IGNORECASE)
            if m:
                try:
                    tool_data = json.loads(m.group(1).strip())
                    tool_calls.append(tool_data)
                except Exception:
                    pass
            return {
                "response": response_text,
                "tool_calls": tool_calls,
                "rounds": 1
            }

    def generate_with_tools_sync(self, prompt: str, tools: Optional[List] = None, **kwargs) -> str:
        """Synchronous wrapper returning only the final response string."""
        res = self.generate_with_tools(prompt=prompt, tools=tools, **kwargs)
        return res.get("response", "")

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
        skills_mode = meta.get("skills_mode") or hb.manifest.get("skills_mode", "mixed")
        # CRITICAL: Explicitly pass the handbag's skills directory as the primary target.
        # This ensures tool_create_skill and tool_update_skill route file writes to the
        # correct physical handbag folder, even if external dirs are merged later.
        handbag_skills_dir = [hb.skills_dir.resolve()] if hb.skills_dir.exists() else []
        sm = SkillsManager(skills_dirs=handbag_skills_dir, mode=skills_mode) if handbag_skills_dir else None

        # Initialize RAG data sources from handbag rag/ folder
        rag_data_sources: List[RAGDataSource] = []
        if hb.rag_files:
            for rf in hb.rag_files:
                try:
                    content = rf.read_text(encoding="utf-8", errors="ignore")
                    doc_title = rf.stem.replace("_", " ").title()
                    rag_data_sources.append(
                        RAGDataSource(
                            name=rf.name,
                            description=f"Knowledge document: {doc_title}",
                            query_fn=lambda q, doc_text=content: doc_text,
                            auto_query=True
                        )
                    )
                except Exception as ex:
                    ASCIIColors.warning(f"[Handbag] Failed to load RAG document {rf.name}: {ex}")

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
            skills_dirs=handbag_skills_dir,
            memory_manager=mm,
            handbag_path=hb.path,
            data_sources=rag_data_sources if rag_data_sources else None,
            workspace_path=hb.workspace_dir if hb.workspace_dir.exists() else None,
            lollms_client=lollms_client,
        )

        # CRITICAL: Explicitly bind the resolved handbag path to the personality instance.
        # This guarantees that downstream systems (like _StreamState in _mixin_chat.py)
        # can access the physical handbag directory for tool/skill file updates.
        object.__setattr__(pers, 'handbag_path', hb.path.resolve())

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

    def _init_data_sources(
        self,
        data_source: Optional[Any],
        data_sources: Optional[Any],
        query_rag_callback: Optional[Callable]
    ) -> None:
        self.data_sources = []

        if data_sources:
            if isinstance(data_sources, dict):
                for name, val in data_sources.items():
                    self._register_data_source_item(val, default_name=name)
            elif isinstance(data_sources, list):
                for item in data_sources:
                    self._register_data_source_item(item)

        if data_source:
            if isinstance(data_source, dict) and not any(k in data_source for k in ("query_fn", "source", "callable", "engine")):
                for name, val in data_source.items():
                    self._register_data_source_item(val, default_name=name)
            elif isinstance(data_source, list):
                for item in data_source:
                    self._register_data_source_item(item)
            else:
                self._register_data_source_item(data_source, default_name="primary_knowledge_base")

        if query_rag_callback and not self.data_sources:
            self._register_data_source_item(query_rag_callback, default_name="rag_callback")

    def _register_data_source_item(self, item: Any, default_name: Optional[str] = None) -> Optional[RAGDataSource]:
        if isinstance(item, RAGDataSource):
            if not any(ds.name == item.name for ds in self.data_sources):
                self.data_sources.append(item)
            return item

        if isinstance(item, dict):
            name = item.get("name") or default_name or f"datasource_{len(self.data_sources)+1}"
            desc = item.get("description") or item.get("desc") or ""
            query_fn = item.get("query_fn") or item.get("source") or item.get("callable") or item.get("engine")
            store = item.get("store") or item.get("ss")
            auto_q = item.get("auto_query", True)
            meta = item.get("metadata", {})

            if isinstance(query_fn, str):
                static_text = query_fn
                query_fn = lambda q: static_text

            ds = RAGDataSource(name=name, description=desc, query_fn=query_fn, store=store, auto_query=auto_q, metadata=meta)
            if not any(existing.name == ds.name for existing in self.data_sources):
                self.data_sources.append(ds)
            return ds

        if callable(item):
            name = default_name or getattr(item, "__name__", f"datasource_{len(self.data_sources)+1}")
            if name == "<lambda>":
                name = default_name or f"datasource_{len(self.data_sources)+1}"
            doc = getattr(item, "__doc__", "") or ""
            desc = doc.strip().split("\n")[0] if doc else "RAG query engine"
            ds = RAGDataSource(name=name, description=desc, query_fn=item, auto_query=True)
            if not any(existing.name == ds.name for existing in self.data_sources):
                self.data_sources.append(ds)
            return ds

        if isinstance(item, str):
            name = default_name or "static_knowledge"
            static_text = item
            ds = RAGDataSource(name=name, description="Static knowledge base", query_fn=lambda q: static_text, auto_query=True)
            if not any(existing.name == ds.name for existing in self.data_sources):
                self.data_sources.append(ds)
            return ds

        return None

    def add_data_source(
        self,
        name: str,
        description: str = "",
        query_fn: Optional[Callable] = None,
        store: Optional[Any] = None,
        auto_query: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RAGDataSource:
        """Register a new RAG datasource dynamically."""
        ds = RAGDataSource(
            name=name,
            description=description,
            query_fn=query_fn,
            store=store,
            auto_query=auto_query,
            metadata=metadata or {}
        )
        self.data_sources = [existing for existing in self.data_sources if existing.name != name]
        self.data_sources.append(ds)
        return ds

    def remove_data_source(self, name: str) -> bool:
        before = len(self.data_sources)
        self.data_sources = [ds for ds in self.data_sources if ds.name != name]
        return len(self.data_sources) < before

    def get_data_source(self, name: str) -> Optional[RAGDataSource]:
        return next((ds for ds in self.data_sources if ds.name.lower() == name.lower()), None)

    def list_data_sources(self) -> List[Dict[str, Any]]:
        return [ds.to_dict() for ds in self.data_sources]

    def _build_query_data_fn(
        self, source: Optional[Union[str, Callable]]
    ) -> Callable[[str], Dict[str, Any]]:
        def _runner(query: str, **kwargs) -> Dict[str, Any]:
            return self.query_data(query, **kwargs)
        return _runner

    def query_data(self, query: str, datasource_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if not self.data_sources:
            if self.query_rag_callback:
                try:
                    raw = _call_query_engine(self.query_rag_callback, query, **kwargs)
                    return _normalise_raw(raw, query, "rag_callback")
                except Exception as e:
                    trace_exception(e)
                    return {"success": False, "sources": [], "count": 0, "query": query, "error": str(e)}
            return {"success": False, "sources": [], "count": 0, "query": query}

        if datasource_name:
            target_ds = next((ds for ds in self.data_sources if ds.name.lower() == datasource_name.lower()), None)
            if not target_ds:
                return {
                    "success": False,
                    "sources": [],
                    "count": 0,
                    "query": query,
                    "error": f"Datasource '{datasource_name}' not found. Available: {[ds.name for ds in self.data_sources]}"
                }
            return target_ds.query(query, **kwargs)

        active_sources = [ds for ds in self.data_sources if ds.auto_query]
        if not active_sources:
            active_sources = self.data_sources

        all_sources = []
        for ds in active_sources:
            res = ds.query(query, **kwargs)
            if res.get("success") and res.get("sources"):
                for src in res["sources"]:
                    src.setdefault("datasource_name", ds.name)
                    all_sources.append(src)

        return {
            "success": bool(all_sources),
            "sources": all_sources,
            "count": len(all_sources),
            "query": query
        }

    def build_rag_system_block(self) -> str:
        if not self.has_data:
            return ""
        lines = ["=== RAG KNOWLEDGE BASES ==="]
        lines.append("You have access to the following RAG knowledge base data source(s):")
        for ds in self.data_sources:
            desc = f": {ds.description}" if ds.description else ""
            lines.append(f"- **{ds.name}**{desc}")
        lines.append(
            "Relevant excerpts are automatically pre-hydrated into your context under "
            "'=== RETRIEVED RAG CONTEXT ==='.\n"
            "You can also query specific data sources on demand using `tool_query_rag`."
        )
        lines.append("=== END RAG KNOWLEDGE BASES ===\n")
        return "\n".join(lines)

    def build_rag_tools(self) -> Dict[str, Dict[str, Any]]:
        if not self.has_data:
            return {}

        ds_descriptions = []
        for ds in self.data_sources:
            desc = f"'{ds.name}': {ds.description}" if ds.description else f"'{ds.name}'"
            ds_descriptions.append(desc)

        ds_list_str = "; ".join(ds_descriptions) if ds_descriptions else "Default Knowledge Base"

        def tool_query_rag(query: str, datasource_name: str = "") -> dict:
            """
            Query the attached RAG knowledge base data source(s) for relevant document excerpts, citations, or facts.

            Args:
                query (str): The search query or question to retrieve information for.
                datasource_name (str, optional): The name of the specific data source to query. If omitted, queries available data sources.
            """
            try:
                ds_target = datasource_name.strip() or None
                res = self.query_data(query, datasource_name=ds_target)
                if not res or not res.get("success") or not res.get("sources"):
                    return {
                        "success": True,
                        "output": f"No relevant content found in RAG datasource for query: '{query}'."
                    }

                output_parts = []
                for idx, src in enumerate(res.get("sources", []), 1):
                    title = src.get("title") or src.get("source") or "Document"
                    score_val = src.get("score")
                    score_str = f" (Score: {score_val:.2f})" if isinstance(score_val, (int, float)) and score_val <= 1.0 else (f" (Score: {score_val})" if score_val is not None else "")
                    output_parts.append(f"[{idx}] {title}{score_str}:\n{src.get('content')}")

                return {
                    "success": True,
                    "sources_count": len(res.get("sources", [])),
                    "output": "\n\n".join(output_parts)
                }
            except Exception as e:
                return {"success": False, "error": f"RAG query failed: {e}"}

        ds_names = [ds.name for ds in self.data_sources]
        return {
            "tool_query_rag": {
                "name": "tool_query_rag",
                "description": f"Query external RAG knowledge bases for information. Available data sources: {ds_list_str}",
                "parameters": [
                    {"name": "query", "type": "str", "description": "The search query or keywords."},
                    {"name": "datasource_name", "type": "str", "description": f"Specific data source name to query (options: {', '.join(ds_names)}).", "optional": True}
                ],
                "callable": tool_query_rag
            }
        }

    @property
    def has_data(self) -> bool:
        return (
            bool(self.data_sources)
            or self._raw_data_source is not None
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
        import builtins as _builtins_mod
        _current_compile = getattr(_builtins_mod, 'compile', None)
        if _current_compile is None or getattr(_current_compile, '__module__', '') != 'builtins':
            ASCIIColors.error(f"[{self.name}] CRITICAL SHADOW DETECTED: builtins.compile is not the native function (module: {getattr(_current_compile, '__module__', 'None')}). Restoring it.")
            import importlib as _importlib
            _real_builtins = _importlib.import_module('builtins')
            _builtins_mod.compile = _real_builtins.compile

        if not self.script:
            return
        try:
            module_name = f"lollms_personality_script_{self.personality_id}"
            spec        = importlib.util.spec_from_loader(module_name, loader=None)
            module      = importlib.util.module_from_spec(spec)
            exec(_builtins_mod.compile(self.script, f"<personality:{self.name}>", "exec"),
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

    @staticmethod
    def _build_progressive_continuation_prompt(stall_count: int, recent_tools: Optional[List[str]] = None) -> str:
        recent_ctx = f" Recent actions executed: {recent_tools}." if recent_tools else ""
        if stall_count <= 1:
            return (
                f"[SYSTEM: You stopped generation without emitting a <done/> tag.{recent_ctx}\n"
                "CRITICAL: Do NOT assume any action succeeded unless you see its result in the conversation history above.\n"
                "If your previous response stated an intent to perform an action (e.g., 'I will commit', 'I will create a branch'), "
                "you MUST execute that action's tag NOW. Stating intent and then emitting `<done/>` without executing is a CRITICAL ERROR.\n"
                "If your task is truly complete, output a final conversational summary and end it with a <done/> tag on a new line. "
                "If you need to continue working, emit the next functional tag now.]"
            )
        elif stall_count == 2:
            return (
                f"[SYSTEM: You have stopped without <done/> or any new action for 2 consecutive rounds.{recent_ctx}\n"
                "This is wasting context. You MUST either:\n"
                "1. Emit the next <tool> or <artifact> tag to continue your task, OR\n"
                "2. Write your final response and emit <done/> to terminate.\n"
                "Do NOT produce another preamble without following through.]"
            )
        else:
            return (
                f"[SYSTEM: CRITICAL — You have stalled {stall_count} times without producing output or <done/>.\n"
                "You are wasting tokens and context. You MUST emit <done/> NOW on a new line.\n"
                "Write a brief summary of the current situation and immediately emit <done/>.]"
            )

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

    def _get_collapsed_folders_from_db(self) -> set:
        if not hasattr(self, '_state_db_path'):
            return set()
        try:
            import sqlite3 as _sqlite3
            conn = _sqlite3.connect(str(self._state_db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT path FROM collapsed_folders")
            return {row[0] for row in cursor.fetchall()}
        except Exception:
            return set()
        finally:
            if 'conn' in locals():
                conn.close()

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
                        collapsed = self._get_collapsed_folders_from_db()
                        return "\n" + _build_workspace_context(self._resolved_workspace, collapsed_folders=collapsed)
                except Exception:
                    pass
            elif self._resolved_workspace:
                collapsed = self._get_collapsed_folders_from_db()
                return "\n" + _build_workspace_context(self._resolved_workspace, collapsed_folders=collapsed)
            return ""

        object.__setattr__(self, '_last_ws_sync_time', current_time)

        if getattr(self, '_artefact_manager', None):
            try:
                self._sync_artefact_index_with_disk()
                zone = self._artefact_manager.build_artefacts_context_zone()
                if zone:
                    return "\n" + zone
                if self._resolved_workspace:
                    collapsed = self._get_collapsed_folders_from_db()
                    return "\n" + _build_workspace_context(self._resolved_workspace, collapsed_folders=collapsed)
            except Exception as e:
                ASCIIColors.warning(f"[{self.name}] Failed to build workspace context: {e}")
                if self._resolved_workspace:
                    collapsed = self._get_collapsed_folders_from_db()
                    return "\n" + _build_workspace_context(self._resolved_workspace, collapsed_folders=collapsed)
        elif self._resolved_workspace:
            collapsed = self._get_collapsed_folders_from_db()
            return "\n" + _build_workspace_context(self._resolved_workspace, collapsed_folders=collapsed)
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

            def _is_pinned(path_str: str) -> bool:
                if not hasattr(self, '_artefact_manager') or not self._artefact_manager:
                    return False
                try:
                    norm_path = path_str.replace("\\", "/").strip().lower()
                    if norm_path.startswith("./"):
                        norm_path = norm_path[2:]
                    arts = self._artefact_manager._get_all_raw()
                    for art in arts:
                        title = art.get("title", "").replace("\\", "/").strip().lower()
                        if title == norm_path:
                            return art.get("visibility") == ArtefactVisibility.PINNED
                except Exception:
                    pass
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

            max_ctx = 0
            if hasattr(self.lollms_client, 'get_ctx_size'):
                max_ctx = self.lollms_client.get_ctx_size() or 0

            if max_ctx > 0:
                threshold = int(max_ctx * 0.50)
                if telemetry["loaded_contents"] > threshold:
                    ASCIIColors.warning(f"[{self.name}] 🚨 Hard Context Budget Guard: Loaded files consume {telemetry['loaded_contents']:,} tokens (> 50% of {max_ctx:,}). Autonomously locking large non-pinned files to prevent collapse.")

                    if hasattr(self, '_artefact_manager') and self._artefact_manager:
                        from lollms_client.lollms_artefact import ArtefactVisibility
                        all_arts = self._artefact_manager._get_all_raw()

                        loaded_files = []
                        for art in all_arts:
                            if art.get("visibility") == ArtefactVisibility.FULL and not art.get("title", "").endswith("::images"):
                                try:
                                    size = art.get("size", 0)
                                    if not size:
                                        fp = self._resolved_workspace / art["title"]
                                        if fp.exists():
                                            size = fp.stat().st_size
                                    loaded_files.append({"title": art["title"], "size": size or 0})
                                except Exception:
                                    loaded_files.append({"title": art["title"], "size": 0})

                        loaded_files.sort(key=lambda x: x.get("size", 0), reverse=True)

                        if loaded_files:
                            targets_to_lock = [f["title"] for f in loaded_files[:3]]
                            if targets_to_lock:
                                self._execute_context_visibility("lock_file", "\n".join(targets_to_lock))
                                object.__setattr__(self, '_last_ws_sync_time', 0.0)
                                ws_ctx = self._build_workspace_context_block()
                                if ws_ctx:
                                    telemetry["workspace_tree"] = self.lollms_client.count_tokens(ws_ctx)
                                    if "## Fully Loaded File Contents [C]" in ws_ctx:
                                        parts = ws_ctx.split("## Fully Loaded File Contents [C]", 1)
                                        if len(parts) == 2:
                                            telemetry["loaded_contents"] = self.lollms_client.count_tokens(parts[1])
                                            telemetry["workspace_tree"] = self.lollms_client.count_tokens(parts[0])
                                    else:
                                        telemetry["loaded_contents"] = 0
                                    telemetry["total"] = sum(telemetry.values())

        except Exception:
            pass

        return telemetry

    def _apply_rolling_artifact_compaction(self, virtual_history: List, base_conversation: List[Dict[str, str]]) -> List:
        """
        Enforces the Rolling Window Compaction Protocol.
        Keeps only the last 4 consecutive artifact operations in virtual_history.
        Evicts older ones and syncs their final state into the Base Context.
        Pinned files are exempt from eviction.
        """
        if not virtual_history:
            return virtual_history

        artifact_indices = [
            i for i, vh in enumerate(virtual_history)
            if vh.sender_type == "assistant" and ("<artifact" in vh.content.lower() or "<artefact" in vh.content.lower())
        ]

        if len(artifact_indices) <= 4:
            return virtual_history

        oldest_artifact_idx = artifact_indices[0]
        next_user_idx = oldest_artifact_idx + 1
        while next_user_idx < len(virtual_history) and virtual_history[next_user_idx].sender_type != "user":
            next_user_idx += 1

        if next_user_idx < len(virtual_history):
            next_user_idx += 1

        evicted_history = virtual_history[:next_user_idx]
        surviving_history = virtual_history[next_user_idx:]

        self._sync_base_context_artifacts(base_conversation, evicted_history)

        return surviving_history

    def _compact_virtual_history(self, virtual_history: List, base_conversation: List[Dict[str, str]], streaming_callback: Optional[Callable]) -> List:
        """
        Autonomously summarizes the virtual history to free up context space.
        Pinned files are exempt from eviction.
        """
        if not virtual_history or not self.lollms_client:
            return virtual_history

        self._sync_base_context_artifacts(base_conversation, virtual_history)

        if streaming_callback:
            compaction_msg = '\n<processing type="context_compaction" title="Autonomous Context Compaction">\n* 🧹 Context window approaching limit. Summarizing history to free up space...\n</processing>\n'
            try:
                streaming_callback(compaction_msg, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            except Exception:
                pass

        history_text = "\n\n".join([f"[{vh.sender_type}]: {vh.content}" for vh in virtual_history])

        summary_prompt = (
            "You are a context compaction engine. Summarize the following conversation history into a dense, factual summary.\n"
            "Focus on retaining: user goals, key data retrieved from tools, file names created/modified, and final conclusions.\n"
            "Discard: conversational pleasantries, intermediate reasoning steps, and verbose tool outputs.\n\n"
            f"=== HISTORY TO COMPACT ===\n{history_text}\n=== END HISTORY ==="
        )

        try:
            summary = self.lollms_client.generate_text(
                prompt=summary_prompt,
                temperature=0.1,
                n_predict=1024
            )
            if not isinstance(summary, str) or not summary.strip():
                return virtual_history

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

    def _compact_virtual_history(self, virtual_history: List, base_conversation: List[Dict[str, str]], streaming_callback: Optional[Callable]) -> List:
        """
        Autonomously summarizes the virtual history to free up context space.
        Replaces verbose tool outputs and intermediate reasoning with a dense summary.
        Syncs the base context to preserve artifact state before history is discarded.
        """
        if not virtual_history or not self.lollms_client:
            return virtual_history

        self._sync_base_context_artifacts(base_conversation, virtual_history)

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
            try:
                cursor.execute("ALTER TABLE file_states ADD COLUMN hash TEXT")
            except _sqlite3.OperationalError:
                pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collapsed_folders (
                    path TEXT PRIMARY KEY
                )
            """)
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
                disable_artefact_versioning=self.disable_artefact_versioning,
            )

            am = ArtefactManager(proxy)
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
            
            
            
    def _dump_error(self, error: Exception, context_desc: str, round_count: int, extra_data: Optional[Dict[str, Any]] = None):
        """Writes a detailed error log to the debug dumps directory."""
        if not getattr(self, 'debug_mode', False):
            return

        try:
            import traceback as _traceback
            from pathlib import Path

            ws_path = getattr(self, '_resolved_workspace', None)
            if not ws_path:
                return

            debug_dir = ws_path / ".lollms_code" / "_debug_dumps"
            debug_dir.mkdir(parents=True, exist_ok=True)

            error_log_path = debug_dir / f"error_round_{round_count}_{context_desc.replace(' ', '_').lower()}.log"

            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"🐛 [DEBUG] ERROR DUMP - ROUND {round_count}\n")
                f.write(f"Context: {context_desc}\n")
                f.write("=" * 80 + "\n\n")

                f.write("--- EXCEPTION ---\n")
                f.write(f"Type: {type(error).__name__}\n")
                f.write(f"Message: {str(error)}\n\n")

                f.write("--- TRACEBACK ---\n")
                f.write(_traceback.format_exc())
                f.write("\n\n")

                if extra_data:
                    f.write("--- EXTRA DATA ---\n")
                    import json as _json
                    try:
                        f.write(_json.dumps(extra_data, indent=2, default=str, ensure_ascii=False))
                    except Exception:
                        f.write(str(extra_data))
                    f.write("\n\n")

            ASCIIColors.error(f"[{self.name}] 🐛 Error dumped to: {error_log_path}")

        except Exception as dump_err:
            ASCIIColors.warning(f"[{self.name}] Failed to write error dump: {dump_err}")

    def _sanitize_history_for_context(self, text: str, round_index: int = 0, distance_from_end: int = 99) -> str:
        """
        Sanitizes assistant message content for LLM context export.
        Enforces a strict non-placeholder strategy for the last 4 actions to prevent
        cognitive thread loss, while aggressively compressing older history.
        """
        if distance_from_end < 4:
            # ── 🔒 STRICT PRESERVATION ZONE (Last 4 Actions) ──
            # Preserve the LLM's raw conversational text and intent statements VERBATIM.
            # Only strip bulky XML tag *bodies* (full artifact content, full tool JSON)
            # to save context, but keep the LLM's own words exactly as it wrote them.
            # NEVER replace with synthetic placeholders or [Action: ...] summaries here.
            # CRITICAL: Preserve <tool_result> content inside <processing> blocks so the
            # model retains access to its own tool outputs in subsequent reasoning rounds.

            text = re.sub(r'<!-- status:[^>]*-->', '', text, flags=re.IGNORECASE)
            text = re.sub(r'<lollms_artifact[^/]*/>', '', text, flags=re.IGNORECASE)
            text = re.sub(r'<artefact_image[^/]*/>', '', text, flags=re.IGNORECASE)

            def _preserve_tool_results_in_processing(m):
                block = m.group(0)
                tool_result_match = re.search(r'<tool_result[^>]*>.*?</tool_result>', block, re.DOTALL | re.IGNORECASE)
                if tool_result_match:
                    return tool_result_match.group(0)
                return ''

            text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', _preserve_tool_results_in_processing, text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'</processing>', '', text, flags=re.IGNORECASE)

            # Strip bulky tag BODIES but preserve surrounding conversational text
            text = re.sub(
                r'<art(?:ifact|efact)[^>]*>.*?</art(?:ifact|efact)>',
                lambda m: re.sub(r'<art(?:ifact|efact)[^>]*>', '[🔒 Artifact body stripped]', m.group(0), flags=re.IGNORECASE),
                text, flags=re.DOTALL | re.IGNORECASE
            )
            text = re.sub(
                r'<tool>.*?</tool>',
                lambda m: re.sub(r'<tool>', '[🔒 Tool body stripped]', m.group(0), flags=re.IGNORECASE),
                text, flags=re.DOTALL | re.IGNORECASE
            )
            text = re.sub(
                r'<(?:unlock_file|lock_file|hide_file|collapse_folder|uncollapse_folder)>.*?</(?:unlock_file|lock_file|hide_file|collapse_folder|uncollapse_folder)>',
                lambda m: re.sub(r'<(unlock_file|lock_file|hide_file|collapse_folder|uncollapse_folder)>', r'[🔒 \1 body stripped]', m.group(0), flags=re.IGNORECASE),
                text, flags=re.DOTALL | re.IGNORECASE
            )
            text = re.sub(r'<scratchpad_(?:append|patch)>.*?</scratchpad_(?:append|patch)>', '[🔒 Scratchpad update]', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<scratchpad_clear\s*/?>', '[🔒 Scratchpad cleared]', text, flags=re.IGNORECASE)
            text = re.sub(r'<mem_new[^/]*/?>', '[🔒 Memory created]', text, flags=re.IGNORECASE)
            text = re.sub(r'<mem_update[^/]*/?>', '[🔒 Memory updated]', text, flags=re.IGNORECASE)
            text = re.sub(r'<user_profile_update>.*?</user_profile_update>', '[🔒 User profile updated]', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<user_profile_clear\s*/?>', '[🔒 User profile cleared]', text, flags=re.IGNORECASE)
            text = re.sub(r'<refactor_history\s*/?>', '[🔒 History refactored]', text, flags=re.IGNORECASE)

            stripped_for_check = re.sub(r'\[🔒[^\]]*\]', '', text).strip()
            if not stripped_for_check:
                text = "[Action executed with no conversational text]"
        else:
            # ── 🧹 AGGRESSIVE COMPRESSION ZONE (Older Actions, distance >= 4) ──
            # CRITICAL: Even in the compression zone, preserve <tool_result> content
            # so the model can reference data from older rounds without re-querying.
            text = re.sub(r'<!-- status:[^>]*-->', '', text, flags=re.IGNORECASE)
            text = re.sub(r'<lollms_artifact[^/]*/>', '', text, flags=re.IGNORECASE)
            text = re.sub(r'<artefact_image[^/]*/>', '', text, flags=re.IGNORECASE)

            def _preserve_tool_results_in_processing_old(m):
                block = m.group(0)
                tool_result_match = re.search(r'<tool_result[^>]*>.*?</tool_result>', block, re.DOTALL | re.IGNORECASE)
                if tool_result_match:
                    return tool_result_match.group(0)
                return ''

            text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', _preserve_tool_results_in_processing_old, text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'</processing>', '', text, flags=re.IGNORECASE)
            text = re.sub(r'<art(?:ifact|efact)[^>]*>.*?</art(?:ifact|efact)>', '[🔒 Artifact stripped]', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<tool>.*?</tool>', '[🔒 Tool stripped]', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<(?:unlock_file|lock_file|hide_file|collapse_folder|uncollapse_folder)>.*?</(?:unlock_file|lock_file|hide_file|collapse_folder|uncollapse_folder)>', '[🔒 Context op stripped]', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<scratchpad_(?:append|patch)>.*?</scratchpad_(?:append|patch)>', '[🔒 Scratchpad update]', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<scratchpad_clear\s*/?>', '[🔒 Scratchpad cleared]', text, flags=re.IGNORECASE)
            text = re.sub(r'<mem_new[^/]*/?>', '[🔒 Memory created]', text, flags=re.IGNORECASE)
            text = re.sub(r'<mem_update[^/]*/?>', '[🔒 Memory updated]', text, flags=re.IGNORECASE)
            text = re.sub(r'<user_profile_update>.*?</user_profile_update>', '[🔒 User profile updated]', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<user_profile_clear\s*/?>', '[🔒 User profile cleared]', text, flags=re.IGNORECASE)
            text = re.sub(r'<refactor_history\s*/?>', '[🔒 History refactored]', text, flags=re.IGNORECASE)
        return text.strip()

    def _sync_base_context_artifacts(self, base_conversation: List[Dict[str, str]], virtual_history: List) -> None:
        """
        Rebuilds the Base Context (initial user message) by injecting the latest workspace tree.
        This ensures the LLM sees the full content of recently evicted artifacts.
        """
        if not base_conversation:
            return

        try:
            evicted_artifact_titles = []
            if hasattr(self, '_artefact_manager') and self._artefact_manager:
                for vh in virtual_history:
                    content = getattr(vh, "content", "")
                    if getattr(vh, "sender_type", "") == "assistant":
                        matches = re.findall(r'<art(?:ifact|efact)[^>]*name=["\']([^"\']+)["\']', content, re.IGNORECASE)
                        evicted_artifact_titles.extend(matches)

                from lollms_client.lollms_artefact import ArtefactVisibility
                for title in evicted_artifact_titles:
                    try:
                        art = self._artefact_manager.get(title)
                        if art and art.get("visibility") != ArtefactVisibility.FULL:
                            self._execute_context_visibility("unlock_file", title)
                    except Exception:
                        pass

            ws_block = self._build_workspace_context_block()
            if not ws_block:
                return

            ws_boundary = "=== CURRENT WORKSPACE CONTEXT ==="
            end_boundary = "=== END CURRENT WORKSPACE CONTEXT ==="

            for i, msg in enumerate(base_conversation):
                if msg.get("role") == "user" and ws_boundary in msg.get("content", ""):
                    start_idx = msg["content"].find(ws_boundary)
                    end_idx = msg["content"].find(end_boundary) + len(end_boundary)
                    prefix = msg["content"][:start_idx].strip()
                    suffix = msg["content"][end_idx:].strip()
                    msg["content"] = f"{prefix}\n\n{ws_boundary}\n{ws_block.strip()}\n{end_boundary}\n\n{suffix}".strip()
                    break
        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] Failed to sync base context artifacts: {e}")

    def _apply_rolling_artifact_compaction(self, virtual_history: List, base_conversation: List[Dict[str, str]]) -> List:
        """
        Enforces the Rolling Window Compaction Protocol.
        Keeps only the last 4 consecutive artifact operations in virtual_history.
        Evicts older ones and syncs their final state into the Base Context.
        """
        if not virtual_history:
            return virtual_history

        artifact_indices = [
            i for i, vh in enumerate(virtual_history)
            if vh.sender_type == "assistant" and ("<artifact" in vh.content.lower() or "<artefact" in vh.content.lower())
        ]

        if len(artifact_indices) <= 4:
            return virtual_history

        oldest_artifact_idx = artifact_indices[0]
        next_user_idx = oldest_artifact_idx + 1
        while next_user_idx < len(virtual_history) and virtual_history[next_user_idx].sender_type != "user":
            next_user_idx += 1

        if next_user_idx < len(virtual_history):
            next_user_idx += 1

        evicted_history = virtual_history[:next_user_idx]
        surviving_history = virtual_history[next_user_idx:]

        self._sync_base_context_artifacts(base_conversation, evicted_history)

        return surviving_history

    def _discover_tools(
        self,
        explicit_tools: Optional[Dict] = None,
        tool_files: Optional[List] = None,
        enable_data_tools: bool = False,
        enable_workspace_tools: bool = True,
        enable_shell: bool = False,
        enable_python_exec: bool = False,
        enable_web_tools: bool = False,
        *args, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        active_tools = {}

        try:
            import getpass
            current_user_name = getpass.getuser()
        except Exception:
            current_user_name = "Unknown User"

        lcp_binding = getattr(self.lollms_client, 'tools', None)
        if not _is_tool_binding(lcp_binding) or hasattr(lcp_binding, "_mock_return_value"):
            lcp_binding = None

        if lcp_binding is None and self._resolved_workspace:
            try:
                from lollms_client.tools_bindings.lcp import LCPBinding
                lcp_binding = LCPBinding()
                if hasattr(self.lollms_client, 'tools'):
                    self.lollms_client.tools = lcp_binding
            except Exception as e:
                ASCIIColors.warning(f"[{self.name}] Failed to initialize LCPBinding: {e}")
                lcp_binding = None

        if lcp_binding and hasattr(lcp_binding, 'mount_tool_library'):
            _libraries_to_mount: List[str] = []

            if enable_workspace_tools and self.capabilities and self.capabilities.enable_workspace_tools and self._resolved_workspace:
                _libraries_to_mount.append("workspace_tools")
            if enable_shell:
                _libraries_to_mount.append("system_shell")
            if enable_python_exec:
                _libraries_to_mount.append("execute_python_code")

            for lib_name in _libraries_to_mount:
                try:
                    lcp_binding.mount_tool_library(lib_name)
                except Exception as e:
                    ASCIIColors.warning(f"[{self.name}] Failed to mount LCP library '{lib_name}': {e}")

            if _libraries_to_mount:
                try:
                    all_lcp_tools = lcp_binding.to_chat_tool_specs(
                        discussion_instance=getattr(self, '_artefact_proxy', None),
                        lollms_client_instance=self.lollms_client
                    )
                    _WS_TOOL_NAMES = {
                        "tool_write_file", "tool_read_file", "tool_list_files",
                        "tool_find_files", "tool_grep_files"
                    }
                    _SHELL_TOOL_NAMES = {"tool_execute_shell_command"}
                    _PY_EXEC_TOOL_NAMES = {"tool_execute_python_code"}

                    allowed_tool_names = set()
                    if enable_workspace_tools and self.capabilities and self.capabilities.enable_workspace_tools and self._resolved_workspace:
                        allowed_tool_names.update(_WS_TOOL_NAMES)
                    if enable_shell:
                        allowed_tool_names.update(_SHELL_TOOL_NAMES)
                    if enable_python_exec:
                        allowed_tool_names.update(_PY_EXEC_TOOL_NAMES)

                    for t_name, t_spec in all_lcp_tools.items():
                        if t_name in allowed_tool_names:
                            active_tools[t_name] = t_spec
                except Exception as e:
                    ASCIIColors.warning(f"[{self.name}] Failed to extract LCP tool specs: {e}")

        if BindingToolsBuilder and self.lollms_client and self.capabilities:
            binding_tools = BindingToolsBuilder.build_tools(self.lollms_client, self.capabilities, self._resolved_workspace)
            active_tools.update(binding_tools)

        if self.capabilities and self.capabilities.enable_skill_loading:
            if self.skills_manager:
                active_tools.update(self.skills_manager.build_skill_tools())

        if self.capabilities and self.capabilities.enable_skill_creation:
            if self.skills_manager:
                skill_tools = self.skills_manager.build_skill_tools()
                for t_name, t_spec in skill_tools.items():
                    if t_name in ("tool_create_skill", "tool_update_skill", "tool_append_to_skill", "tool_remove_skill"):
                        active_tools[t_name] = t_spec

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

        if self._model_switcher and self.capabilities and self.capabilities.enable_model_switching:
            def tool_switch_model(model_name: str) -> dict:
                return self._model_switcher.switch_model(model_name)

            def tool_list_models() -> dict:
                models = self._model_switcher.list_models()
                return {"success": True, "models": models, "output": ", ".join(models)}

            active_tools["tool_switch_model"] = {
                "name": "tool_switch_model",
                "description": "Switch to a different model.",
                "parameters": [{"name": "model_name", "type": "str", "description": "The name of the model to switch to."}],
                "callable": tool_switch_model
            }
            active_tools["tool_list_models"] = {
                "name": "tool_list_models",
                "description": "List available models.",
                "parameters": [],
                "callable": tool_list_models
            }

        # Merge Handbag / Personality-level Tools (LCPBinding / tool_specs)
        if self._tool_binding and _is_tool_binding(self._tool_binding):
            try:
                handbag_tools = self._tool_binding.to_chat_tool_specs(
                    discussion_instance=getattr(self, '_artefact_proxy', None),
                    lollms_client_instance=self.lollms_client
                )
                active_tools.update(handbag_tools)
            except Exception as e:
                ASCIIColors.warning(f"[{self.name}] Failed to extract handbag tools: {e}")

        # Mount RAG Query Tool if personality has RAG data sources
        if self.has_data:
            active_tools.update(self.build_rag_tools())

        lcp_binding = getattr(self.lollms_client, 'tools', None)
        if not _is_tool_binding(lcp_binding) or hasattr(lcp_binding, "_mock_return_value"):
            lcp_binding = None


        if enable_data_tools and lcp_binding is None and (tool_files or self._resolved_workspace):
            try:
                import lollms_client as _lollms_client_pkg
                from lollms_client.tools_bindings.lcp import LCPBinding
                pkg_root = Path(_lollms_client_pkg.__file__).resolve().parent
                default_tools = pkg_root / "tools_bindings" / "lcp" / "default_tools"
                lcp_binding = LCPBinding(tools_folders=[str(default_tools)] if default_tools.exists() else [])
            except Exception:
                lcp_binding = None

        if enable_data_tools and lcp_binding and hasattr(lcp_binding, 'mount_tool_library'):
            ws_path = self._resolved_workspace
            has_data_files = False
            has_document_files = False
            if ws_path and ws_path.exists():
                _DATA_EXTS = {".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"}
                _DOC_EXTS = {".pdf", ".docx", ".pptx", ".odt", ".doc", ".txt", ".md"}

                try:
                    for f in ws_path.rglob("*"):
                        if f.is_file():
                            ext = f.suffix.lower()
                            if ext in _DATA_EXTS:
                                has_data_files = True
                            elif ext in _DOC_EXTS:
                                has_document_files = True
                            if has_data_files and has_document_files:
                                break
                except Exception:
                    pass

            _LIBRARIES_TO_MOUNT: List[str] = []
            if has_document_files:
                _LIBRARIES_TO_MOUNT.extend(["as_is_document_tools", "document_editor"])
            if has_data_files:
                _LIBRARIES_TO_MOUNT.append("semantic_data_engineer")

            for lib_name in _LIBRARIES_TO_MOUNT:
                try:
                    lcp_binding.mount_tool_library(lib_name)
                except Exception as e:
                    ASCIIColors.warning(f"[LollmsPersonality] Failed to mount LCP tool library '{lib_name}': {e}")

            if _LIBRARIES_TO_MOUNT:
                try:
                    lcp_tools = lcp_binding.to_chat_tool_specs()
                    for t_name, t_spec in lcp_tools.items():
                        if has_document_files and (
                            t_name.startswith("tool_inspect_document") or
                            t_name.startswith("tool_read_document_content") or
                            t_name.startswith("tool_grep_document") or
                            t_name.startswith("tool_modify_docx") or
                            t_name.startswith("tool_modify_excel") or
                            t_name.startswith("tool_modify_pptx_slide") or
                            t_name.startswith("tool_edit_document_text") or
                            t_name.startswith("tool_annotate_document")
                        ):
                            active_tools[t_name] = t_spec
                        if has_data_files and t_name == "tool_execute_python_data_query":
                            active_tools[t_name] = t_spec
                except Exception as e:
                    ASCIIColors.warning(f"[LollmsPersonality] Failed to extract LCP tool specs: {e}")

            if has_document_files and current_user_name and current_user_name != "Unknown User":
                user_annotation_rule = (
                    f"\n\n**CRITICAL ANNOTATION RULE**: When using `tool_annotate_document` to add comments to a PDF or DOCX, "
                    f"you MUST set the `commenter_name` parameter to '{current_user_name}' (the current OS user account)."
                )
                if "tool_annotate_document" in active_tools:
                    active_tools["tool_annotate_document"]["description"] += user_annotation_rule

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
            try:
                _cb = getattr(self, '_active_streaming_callback', None)
                if _cb:
                    _cb("", MSG_TYPE.MSG_TYPE_SCRATCHPAD_UPDATE, {"action": "scratchpad_clear", "status": "success", "message": "Scratchpad cleared successfully."})
            except Exception:
                pass
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
            try:
                _cb = getattr(self, '_active_streaming_callback', None)
                if _cb:
                    _cb("", MSG_TYPE.MSG_TYPE_SCRATCHPAD_UPDATE, {"action": "user_profile_clear", "status": "success", "message": "User profile cleared successfully."})
            except Exception:
                pass
            return "✅ Global user profile cleared successfully."
        except Exception as e:
            return f"[SYSTEM ERROR] Failed to clear user profile: {e}"

    def _execute_scratchpad_update(self, tag_name: str, body: str) -> str:
        """Executes append or patch operations on the scratchpad file."""
        if not getattr(self, '_scratchpad_path', None):
            return "[SYSTEM ERROR] Scratchpad not initialized."

        stripped_body = body.strip() if body else ""
        if not stripped_body:
            return "⚠️ Scratchpad update ignored: No content provided. Provide text to append or a valid SEARCH/REPLACE block."

        try:
            current_content = self._scratchpad_path.read_text(encoding="utf-8", errors="ignore")
            action_verb = "updated"
            preview = stripped_body[:200].replace('\n', ' | ')

            if tag_name == "scratchpad_append":
                new_content = current_content + "\n" + stripped_body + "\n"
                self._scratchpad_path.write_text(new_content, encoding="utf-8")
                action_verb = "appended to"
            elif tag_name == "scratchpad_patch":
                from lollms_client.lollms_artefact import ArtefactManager
                patched_content = ArtefactManager.apply_aider_patch(current_content, body)
                self._scratchpad_path.write_text(patched_content, encoding="utf-8")
                action_verb = "patched"
            else:
                return "[SYSTEM ERROR] Unknown scratchpad operation."

            try:
                _cb = getattr(self, '_active_streaming_callback', None)
                if _cb:
                    _cb("", MSG_TYPE.MSG_TYPE_SCRATCHPAD_UPDATE, {"action": tag_name, "status": "success", "message": f"Scratchpad {action_verb} successfully.", "preview": preview})
            except Exception:
                pass
            return f"✅ Content {action_verb} scratchpad successfully."
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
            object.__setattr__(self, '_user_profile_content', patched_content)

            preview = body[:200].replace('\n', ' | ')
            try:
                _cb = getattr(self, '_active_streaming_callback', None)
                if _cb:
                    _cb("", MSG_TYPE.MSG_TYPE_SCRATCHPAD_UPDATE, {"action": "user_profile_update", "status": "success", "message": "User profile updated successfully.", "preview": preview})
            except Exception:
                pass
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
        New file creation is ALWAYS exempt from blocking to prevent autonomous loop deadlocks.
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

    def _build_system_prompt(self, active_tools: Optional[Dict] = None) -> str:
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
            "11. **EXPLANATION BEFORE ARTIFACTS (MANDATORY)**: Before emitting an `<artifact>` tag, you MUST provide a brief, 2-3 sentence explanation of what you are about to write and why. Do NOT emit the `<artifact>` tag as the very first token of your response.\n"
            "\n=== TOOL CALLING DISCIPLINE (XML HYBRID PROTOCOL) ===\n"
            "To call a tool, you have TWO options. **Option 1 is strongly preferred for code execution** to avoid escaping errors.\n\n"
            "OPTION 1: Raw XML Parameters (No JSON escaping needed)\n"
            "Use this for tools with code or long text parameters. Wrap each parameter in its own tag.\n"
            "<tool>\n"
            "  <tool_name name=\"tool_execute_python_code\" />\n"
            "  <parameter name=\"code\">\n"
            "import sys\n"
            "print(\"Hello World\")\n"
            "  </parameter>\n"
            "</tool>\n\n"
            "OPTION 2: JSON Parameters (For simple, non-code parameters)\n"
            "Use this for simple parameters (filenames, booleans, numbers).\n"
            "<tool>\n"
            "  <tool_name name=\"tool_find_files\" />\n"
            "  <parameters>{\"pattern\": \"*.py\", \"path\": \".\"}</parameters>\n"
            "</tool>\n\n"
            "1. **Tool Results ≠ Tool Calls**: When a tool returns JSON, it's a RESULT, not a new call.\n"
            "2. **One Call Per Task**: Once a tool succeeds, analyze and answer.\n"
            "3. **Loop Prevention**: Repeating a successful tool call with identical parameters is a CRITICAL ERROR.\n"
            "4. **File Outputs**: When a tool returns a file, it's ALREADY saved. Do NOT call it again.\n"
            "\n=== FILE EDITING & WRITING PROTOCOL ===\n"
            "You have a massive output token limit. Write complete files whenever possible.\n"
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
            "\n**SEGMENTED WRITING (APPEND OPERATION)**\n"
            "If you are writing a very large file and prefer to write it in chunks (or if you hit a generation limit), you can use the `operation=\"append\"` attribute.\n"
            "This adds the content inside the tag to the end of the specified file without overwriting what is already there.\n"
            "Syntax:\n"
            "<artifact name=\"filename.ext\" type=\"code\" language=\"python\" operation=\"append\">\n"
            "// content to add to the end of the file\n"
            "</artifact>\n"
            "You MUST ensure the file exists before appending to it. Use `operation=\"append\"` sequentially to build massive files piece by piece.\n"
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
            "4. **GIT STATE PRESERVATION (ABSOLUTE PROHIBITION)**: \n"
            "   Before creating a new branch (`git checkout -b`), you MUST ensure the working tree is clean.\n"
            "   a. Run `git status`.\n"
            "   b. If there are uncommitted changes, you are STRICTLY FORBIDDEN from running `git stash` or `git commit` autonomously.\n"
            "   c. You MUST stop and output EXACTLY: \"⚠️ I need to create a new branch, but you have uncommitted changes. Do you want me to `git stash` them (temporary) or `git commit` them (permanent) before I switch branches? (stash/commit/cancel)\n</done>\" the done tag is important, it stops the loop and gives the hand back to user"
            "   d. You MUST wait for the user's explicit response ('stash' or 'commit') before executing either command.\n"
            "   e. NEVER execute `git checkout -b` on a dirty working tree. This carries changes to the new branch and pollutes it.\n"
            "   f. Before stashing or switching, use `<scratchpad_append>` to save your current plan and reasoning so you don't lose your train of thought.\n"
            "=== END OPERATIONAL SAFETY DOCTRINE ===\n"
            f"{onboarding_block}"
            "\n=== THINKING & REASONING CONSTRAINT ===\n"
            "If you output thoughts enclosed in  tags, you MUST output all functional XML tags AFTER the closing tag.\n"
            "\n=== TOOL CALLING SYNTAX (STRICT) ===\n"
            "1. **EXACT CLOSING TAG**: The closing tag is `</tool>`. You MUST NOT write ``` or any other variation.\n"
            "2. **NEW LINE ONLY**: The `<tool>` tag MUST start on a brand new line.\n"
            "3. **NO PROSE AROUND IT**: Do NOT write introductory text before the tag, and do NOT write text after it on the same line.\n"
            "4. **XML HYBRID FORMAT**: You MUST use `<tool_name name=\"...\" />` inside the `<tool>` tag.\n"
            "=== END TOOL CALLING SYNTAX ===\n"
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
            "\n=== EPHEMERAL CONTEXT & SCRATCHPAD ENFORCEMENT (CRITICAL) ===\n"
            "When you load a file into your context using `<unlock_file>`, its content is visible to you ONLY for the current turn.\n"
            "Once you emit `<done/>` or the turn ends, the file content is EVICTED from the active context window.\n"
            "If you need to remember specific details from that file for future turns (e.g., a variable name, a function signature, a configuration value),\n"
            "you MUST extract those notes and write them to your Scratchpad using `<scratchpad_append>` BEFORE finishing your turn.\n"
            "Do not rely on your ability to 're-read' the file later, as context budget may prevent re-loading.\n"
            "=== END EPHEMERAL CONTEXT ENFORCEMENT ===\n"
            "\n=== WORKSPACE TREE COMPACTION PROTOCOL ===\n"
            "The workspace tree is COMPACTED to save context tokens. Deep directories are marked as `[📁 DEEP]`.\n"
            "Large directories are auto-collapsed and marked as `[📁 COLLAPSED]` with an item count.\n"
            "To see the contents of a collapsed or deep folder, emit:\n"
            "<uncollapse_folder>folder_name/</uncollapse_folder>\n"
            "To collapse it again (saving context), emit:\n"
            "<collapse_folder>folder_name/</collapse_folder>\n"
            "\n=== STICKY CONTEXT & PINNING (CRITICAL FOR CODING) ===\n"
            "When performing complex coding tasks or cross-file refactoring, you need to keep the exact source code of the files you are editing in your context.\n"
            "To prevent the system from auto-locking or evicting these files, you can PIN them:\n"
            "- `<pin_file>filename.py</pin_file>`: Pins a file. It will be marked as [📌 Pinned] and its content will NEVER be evicted or auto-locked.\n"
            "- `<unpin_file>filename.py</unpin_file>`: Unpins a file, returning it to normal [C] loaded state (subject to budget guards).\n"
            "You can pin multiple files simultaneously to see them side-by-side in your context.\n"
            "=== END STICKY CONTEXT & PINNING ===\n"
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

        document_annotation_workflow = ""
        has_annotation_tools = "tool_annotate_document" in active_tools or "tool_edit_document_text" in active_tools
        has_reading_tools = "tool_read_document_content" in active_tools or "tool_inspect_document" in active_tools
        if has_annotation_tools and has_reading_tools:
            document_annotation_workflow = (
                "\n=== DOCUMENT ANNOTATION WORKFLOW (MANDATORY FOR PROOFREADING TASKS) ===\n"
                "When asked to annotate, proofread, or correct a document (PDF, DOCX, PPTX), you MUST follow this workflow:\n"
                "1. **INSPECT**: Call `tool_inspect_document` to get the page/slide count.\n"
                "2. **READ IN BATCHES**: Call `tool_read_document_content` with `page_or_sheet` set to a 10-page range (e.g., \"1-10\") and `max_chars` set to at least 20000.\n"
                "3. **COLLECT EXACT QUOTES**: As you read, note the EXACT text of each issue (spelling, grammar, clarity, logic, structure). You need the exact text for the `search_text` parameter of the annotation tool.\n"
                "4. **ANNOTATE IMMEDIATELY**: After reading each batch, call `tool_annotate_document` (for comments) or `tool_edit_document_text` (for corrections) for EVERY issue you found in that batch. Do NOT wait until you have read the entire document to start annotating.\n"
                "5. **BE CONSTRUCTIVE**: Your comments should explain WHY something is wrong and suggest a fix. For example: \"Grammar: 'start' should be 'starts' (subject-verb agreement).\"\n"
                "6. **COVER ALL PAGES**: Continue reading and annotating in 10-page batches until you have covered the entire document.\n"
                "7. **SUMMARIZE**: After annotating all pages, provide a summary of the main issues found and emit `<done/>`.\n"
                "**CRITICAL**: You MUST call `tool_annotate_document` or `tool_edit_document_text` at least once per batch of issues found. Reading without annotating is a failure.\n"
                "=== END DOCUMENT ANNOTATION WORKFLOW ===\n"
            )

        return sys_prompt + "\n" + rules + skills_ctx + memory_instructions + tool_desc + document_annotation_workflow
    
    
    def change_file_visibility(self, targets: List[str], action: str) -> Dict[str, Any]:
        action_map = {
            "load": "unlock_file",
            "unload": "lock_file",
            "lock": "lock_file",
            "hide": "hide_file",
            "unhide": "uncollapse_folder"
        }
        tag_name = action_map.get(action.lower())
        if not tag_name:
            return {"status_str": f"❌ Unknown action: {action}", "loaded_contents": {}}

        body_content = "\n".join(targets)
        return self._execute_context_visibility(tag_name, body_content)

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
        elif tag_name == "pin_file":
            target_visibility = ArtefactVisibility.PINNED
            action_verb = "Pinning"
        elif tag_name == "unpin_file":
            target_visibility = ArtefactVisibility.FULL
            action_verb = "Unpinning"

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
                import sqlite3 as _sqlite3

                folder_path_normalized = t_target.strip().replace("\\", "/").rstrip("/")
                if not folder_path_normalized:
                    continue

                try:
                    conn = _sqlite3.connect(str(self._state_db_path))
                    cursor = conn.cursor()
                    if tag_name == "collapse_folder":
                        cursor.execute("INSERT OR REPLACE INTO collapsed_folders (path) VALUES (?)", (folder_path_normalized,))
                        action_verb = "Collapsing"
                    else:
                        cursor.execute("DELETE FROM collapsed_folders WHERE path = ?", (folder_path_normalized,))
                        action_verb = "Uncollapsing"
                    conn.commit()
                    conn.close()

                    processed_files.append(folder_path_normalized)
                    object.__setattr__(self, '_last_ws_sync_time', 0.0)
                    continue
                except Exception as db_err:
                    ASCIIColors.warning(f"[{self.name}] Failed to update collapsed_folders DB: {db_err}")
                    continue

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
                if art.get("visibility") == ArtefactVisibility.FULL:
                    already_in_state.append(t_target)
                    continue

                file_path = self._resolved_workspace / art["title"]

                token_count = 0
                content = ""

                if file_path.exists():
                    try:
                        if art.get("content") and art.get("content_source") == "db" and len(art["content"]) > 0:
                            content = art["content"]
                        else:
                            import_art = self._artefact_manager.import_file(
                                file_path=file_path,
                                title=art["title"],
                                active=False
                            )
                            content = import_art.get("content", "")
                            art["content_source"] = "db"

                            if art.get("type") == "data":
                                content = self._artefact_manager._get_lam_content(import_art).strip()

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
        pinned_files_list = []
        if hasattr(self, '_artefact_manager') and self._artefact_manager:
            current_arts = self._artefact_manager._get_all_raw()
            for a in current_arts:
                title = a.get("title", "")
                if title.endswith("::images"):
                    continue
                vis = a.get("visibility")
                if vis == ArtefactVisibility.PINNED:
                    pinned_files_list.append(title)
                elif vis == ArtefactVisibility.FULL:
                    active_files_list.append(title)

        if pinned_files_list:
            status_parts.append("\n📌 Pinned in Context (Sticky):")
            for f_name in sorted(pinned_files_list):
                status_parts.append(f"  - {f_name}")
        if active_files_list:
            status_parts.append("\n📂 Loaded in Context [C]:")
            for f_name in sorted(active_files_list):
                status_parts.append(f"  - {f_name}")
        if not active_files_list and not pinned_files_list:
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

            if tool_name == "tool_execute_shell_command":
                git_block_msg = self._enforce_git_branch_safety(str(tool_params.get("command", "")))
                if git_block_msg:
                    return {"success": False, "error": git_block_msg, "output": git_block_msg}

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
                    if isinstance(result, dict):
                        if result.get("success") is False and not result.get("error"):
                            result["error"] = (
                                f"Tool '{tool_name}' returned success=False with no error message. "
                                f"Raw keys: {list(result.keys())}. "
                                f"This may indicate a library initialization failure, a missing dependency, or an import error."
                            )
                            ASCIIColors.error(f"[{self.name}] Tool '{tool_name}' returned bare success=False. Synthesized error: {result['error']}")
                        return result
                    return {"success": True, "output": str(result)}
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
                    if isinstance(result, dict):
                        inner = result.get("output")
                        if isinstance(inner, dict):
                            if inner.get("success") is False:
                                if not inner.get("error"):
                                    inner["error"] = f"LCP tool '{tool_name}' returned success=False without a descriptive error. Raw keys: {list(inner.keys())}"
                                return inner
                            if "error" in inner:
                                return inner
                            return inner if "output" in inner else {"success": True, "output": inner}
                        if result.get("status_code", 200) not in (200, 201):
                            err_msg = result.get("error") or f"LCP tool '{tool_name}' returned status_code {result.get('status_code')} with no error message."
                            if result.get("traceback"):
                                err_msg = f"{err_msg}\n\nTraceback:\n{result.get('traceback')}"
                            return {
                                "success": False,
                                "error": err_msg,
                                "traceback": result.get("traceback"),
                            }
                        if inner is not None:
                            return {"success": True, "output": inner}
                        if "error" in result:
                            err_msg = result["error"]
                            if result.get("traceback"):
                                err_msg = f"{err_msg}\n\nTraceback:\n{result.get('traceback')}"
                            return {"success": False, "error": err_msg, "traceback": result.get("traceback")}
                        return {"success": True, "output": str(result)}
                    if result is None:
                        return {"success": False, "error": f"LCP tool '{tool_name}' returned None (no output). This may indicate a crash in the tool's initialization or execution."}
                    return {"success": True, "output": str(result)}
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
        max_nb_rounds: Optional[int] = None,
        max_reasoning_steps: Optional[int] = None,
        temperature: float = 0.7,
        n_predict: Optional[int] = None,
        enable_artefacts: bool = True,
        use_internal_history: bool = True,
        enable_workspace_tools: bool = True,
        enable_shell: bool = False,
        enable_python_exec: bool = False,
        enable_web_tools: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        resolved_max_rounds = max_nb_rounds if max_nb_rounds is not None else max_reasoning_steps
        if resolved_max_rounds is None:
            resolved_max_rounds = 20

        if lollms_client is not None:
            self.lollms_client = lollms_client

        if self.lollms_client is None:
            raise RuntimeError(f"[{self.name}] Independent chat requires a lollms_client instance.")

        self._reset_cancel_state()
        object.__setattr__(self, '_consecutive_empty_responses', 0)
        object.__setattr__(self, '_consecutive_stall_count', 0)
        object.__setattr__(self, '_consecutive_artifact_rounds', 0)
        object.__setattr__(self, '_max_rounds', resolved_max_rounds)

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

        import builtins as _builtins_mod_check
        _current_compile = getattr(_builtins_mod_check, 'compile', None)
        if _current_compile is None or _current_compile.__module__ != 'builtins':
            ASCIIColors.error(f"[{self.name}] CRITICAL SHADOW DETECTED in chat(): builtins.compile is not native (module: {_current_compile.__module__ if _current_compile else 'None'}). Restoring it.")
            import importlib as _importlib_check
            _real_builtins = _importlib_check.import_module('builtins')
            _builtins_mod_check.compile = _real_builtins.compile

        import builtins as _builtins_mod_check
        _current_compile = getattr(_builtins_mod_check, 'compile', None)
        if _current_compile is None or getattr(_current_compile, '__module__', '') != 'builtins':
            ASCIIColors.error(f"[{self.name}] CRITICAL SHADOW DETECTED in chat(): builtins.compile is not native (module: {getattr(_current_compile, '__module__', 'None')}). Restoring it.")
            import importlib as _importlib_check
            _real_builtins = _importlib_check.import_module('builtins')
            _builtins_mod_check.compile = _real_builtins.compile

        self._init_scratchpad()
        object.__setattr__(self, '_active_streaming_callback', streaming_callback)

        cleaned_prompt = prompt
        enable_data_tools_flag = kwargs.get("enable_data_tools", True)
        active_tools = self._discover_tools(
            tools,
            tool_files or [],
            enable_data_tools=enable_data_tools_flag,
            enable_workspace_tools=enable_workspace_tools,
            enable_shell=enable_shell,
            enable_python_exec=enable_python_exec,
            enable_web_tools=enable_web_tools
        )

        stable_system_prompt = self._build_system_prompt(active_tools)
        stable_system_prompt += self._build_user_profile_context()

        # Pre-hydrate RAG knowledge base context into prompt
        if self.has_data:
            rag_sys_block = self.build_rag_system_block()
            if rag_sys_block:
                stable_system_prompt += "\n" + rag_sys_block

            try:
                rag_res = self.query_data(cleaned_prompt)
                if rag_res and rag_res.get("success") and rag_res.get("sources"):
                    sources_text = []
                    for src in rag_res.get("sources", []):
                        title = src.get("title") or src.get("source") or "Document"
                        ds_label = f" [{src.get('datasource_name')}]" if src.get('datasource_name') else ""
                        sources_text.append(f"--- Document [{title}]{ds_label} ---\n{src.get('content')}")
                    if sources_text:
                        stable_system_prompt += "\n=== RETRIEVED RAG CONTEXT ===\n" + "\n\n".join(sources_text) + "\n=== END RAG CONTEXT ===\n"
            except Exception as rag_err:
                ASCIIColors.warning(f"[{self.name}] RAG pre-hydration warning: {rag_err}")


        dynamic_suffix_parts = []

        ws_ctx = self._build_workspace_context_block()
        if ws_ctx:
            dynamic_suffix_parts.append(ws_ctx.strip())

        scratchpad_ctx = self._build_scratchpad_context()
        if scratchpad_ctx:
            dynamic_suffix_parts.append(scratchpad_ctx.strip())
            object.__setattr__(self, '_scratchpad_content', scratchpad_ctx)

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

        if telemetry.get("total", 0) > 0 and telemetry.get("fill_percentage", 0) > 90.0:
            ASCIIColors.warning(f"[{self.name}] 🚨 PRE-GENERATION CONTEXT OVERFLOW: {telemetry.get('fill_percentage', 0):.1f}% fill detected before LLM generation. Triggering emergency context recovery.")

            if hasattr(self, '_artefact_manager') and self._artefact_manager:
                try:
                    from lollms_client.lollms_artefact import ArtefactVisibility
                    all_arts = self._artefact_manager._get_all_raw()
                    loaded_files = [
                        a.get("title", "") for a in all_arts
                        if a.get("visibility") == ArtefactVisibility.FULL
                        and a.get("visibility") != ArtefactVisibility.PINNED
                        and not a.get("title", "").endswith("::images")
                    ]
                    if loaded_files:
                        ASCIIColors.warning(f"[{self.name}] 🚨 Emergency-locking {len(loaded_files)} non-pinned loaded file(s) to prevent context collapse.")
                        self._execute_context_visibility("lock_file", "\n".join(loaded_files))
                        object.__setattr__(self, '_last_ws_sync_time', 0.0)
                        ws_ctx = self._build_workspace_context_block()
                        telemetry = self._calculate_context_telemetry(stable_system_prompt, base_conversation, ws_ctx or "", [])
                        telemetry_block = self._build_telemetry_block(telemetry)
                except Exception as emergency_err:
                    ASCIIColors.warning(f"[{self.name}] Emergency context recovery failed: {emergency_err}")

        if telemetry_block:
            dynamic_suffix_parts.append(telemetry_block)

        dynamic_suffix = "\n\n".join(dynamic_suffix_parts)

        if dynamic_suffix:
            stable_system_prompt += "\n\n" + dynamic_suffix
        fused_prompt = cleaned_prompt

        base_conversation.append({"role": "user", "content": fused_prompt})

        virtual_history: List[SimpleNamespace] = []
        tool_calls_this_turn: List[Dict[str, Any]] = []
        tool_results_this_turn: List[Dict[str, Any]] = []
        round_count = 0
        was_cancelled = False
        successful_tool_signatures: set = set()
        seen_context_signatures: set = set()
        final_response = ""
        workspace_changes: List[Dict[str, Any]] = []

        while round_count < resolved_max_rounds:
            if self.is_generation_cancelled():
                was_cancelled = True
                break

            round_count += 1

            if getattr(self, 'debug_mode', False):
                ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count}/{self._max_rounds} START ===")

            pre_gen_telemetry = self._calculate_context_telemetry(
                stable_system_prompt, base_conversation,
                self._build_workspace_context_block() if hasattr(self, '_build_workspace_context_block') else "",
                virtual_history
            )
            pre_gen_fill = pre_gen_telemetry.get("fill_percentage", 0.0)
            if pre_gen_fill > 98.0 and round_count == 1:
                ASCIIColors.error(f"[{self.name}] 🛑 CONTEXT WINDOW EXHAUSTED ({pre_gen_fill:.1f}% fill). Cannot generate — the system prompt + workspace context exceeds the model's context window ({pre_gen_telemetry.get('total', 0):,} / {pre_gen_telemetry.get('max_tokens', 0):,} tokens). Refusing to generate to prevent silent empty-response exit.")

                if streaming_callback:
                    try:
                        diagnostic_msg = (
                            f"\n⚠️ **Context Window Exhausted** ({pre_gen_fill:.1f}% fill)\n\n"
                            f"The combined system prompt, workspace tree, and loaded file contents "
                            f"({pre_gen_telemetry.get('total', 0):,} tokens) exceed your model's context "
                            f"window ({pre_gen_telemetry.get('max_tokens', 0):,} tokens).\n\n"
                            f"**Breakdown:**\n"
                            f"- System Prompt: {pre_gen_telemetry.get('system_prompt', 0):,} tokens\n"
                            f"- History: {pre_gen_telemetry.get('history', 0):,} tokens\n"
                            f"- Workspace Tree: {pre_gen_telemetry.get('workspace_tree', 0):,} tokens\n"
                            f"- Loaded Files: {pre_gen_telemetry.get('loaded_contents', 0):,} tokens\n"
                            f"- Virtual History: {pre_gen_telemetry.get('virtual_history', 0):,} tokens\n\n"
                            f"**Suggested actions:**\n"
                            f"1. Use `/clear-files` to unload all files from context\n"
                            f"2. Use `/clear-history` to clear conversation history\n"
                            f"3. Lock or hide large directories (e.g., `exports/`)\n"
                            f"4. Switch to a model with a larger context window\n"
                        )
                        streaming_callback(diagnostic_msg, MSG_TYPE.MSG_TYPE_CHUNK, {})
                    except Exception:
                        pass

                final_response = (
                    f"[Context Window Exhausted: The system prompt + workspace context ({pre_gen_telemetry.get('total', 0):,} tokens) "
                    f"exceeds the model's context window ({pre_gen_telemetry.get('max_tokens', 0):,} tokens). "
                    f"Please unload files, clear history, or use a model with a larger context window.]"
                )
                break

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

            context_adapter = _HistoryContextAdapter(self, stable_system_prompt)
            messages = HistoryManager.export(
                context=context_adapter,
                format_type="openai_chat",
                branch=base_conversation,
                virtual_history=virtual_history,
                system_prompt_override=stable_system_prompt
            )

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
                            if isinstance(content, list):
                                content = "\n".join([item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"])
                            if not isinstance(content, str):
                                content = str(content)
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
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        f.write(item.get("text", "") + "\n")
                                    elif isinstance(item, dict) and item.get("type") == "image_url":
                                        f.write("[IMAGE ATTACHED]\n")
                                    else:
                                        f.write(str(item) + "\n")
                            else:
                                f.write(str(content) + "\n")
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
                    if meta and meta.get("live_tool_chunk"):
                        return True
                    if meta and meta.get("was_processed"):
                        return True
                    return ss.feed(chunk)
                return True

            gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("streaming_callback", "temperature", "n_predict", "stream")}

            gen_kwargs["n_predict"] = None

            gen_kwargs["temperature"] = temperature

            _max_retries = 3
            _retry_delay = 2.0
            _generation_succeeded = False

            for _retry_attempt in range(_max_retries):
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
                    _generation_succeeded = True
                    break
                except Exception as gen_err:
                    if self.is_generation_cancelled():
                        was_cancelled = True
                        break

                    ss.completed_actions = []
                    ss._is_accumulating_tool = False
                    ss._is_accumulating_artifact = False
                    ss._is_accumulating_context = False
                    ss._tool_buffer = ""
                    ss._pending_buffer = ""

                    is_transient = False
                    try:
                        err_type_name = type(gen_err).__name__
                        err_module = type(gen_err).__module__
                        if "RemoteProtocolError" in err_type_name or "ConnectionError" in err_type_name or "TimeoutError" in err_type_name or "APIConnectionError" in err_type_name:
                            is_transient = True
                    except Exception:
                        pass

                    if is_transient and _retry_attempt < _max_retries - 1:
                        ASCIIColors.warning(f"[{self.name}] Transient network error during generation (attempt {_retry_attempt + 1}/{_max_retries}). Retrying in {_retry_delay}s... Error: {gen_err}")
                        try:
                            import time as _time
                            _time.sleep(_retry_delay)
                        except Exception:
                            pass
                        _retry_delay *= 2
                        ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                        continue
                    else:
                        if getattr(self, 'debug_mode', False):
                            self._dump_error(
                                error=gen_err,
                                context_desc="LLM Generation Error",
                                round_count=round_count,
                                extra_data={"messages": messages}
                            )
                        ASCIIColors.error(f"[{self.name}] Generation error: {gen_err}")
                        final_response = f"[Generation error: The LLM server connection failed. Please check your server and retry. Details: {gen_err}]"
                        break

            if was_cancelled:
                break

            if not _generation_succeeded and not final_response:
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

            has_truncated_artifact = False
            truncated_artifact_title = None

            if ss.was_done_detected():
                final_response = re.sub(r'(?i)<done\s*/?>', '', ss.get_clean_text()).strip()

                if not ss.completed_actions and not tool_calls_this_turn and not workspace_changes and round_count == 1:
                    sanitized_final_response_check = re.sub(r'<[^>]+>', '', final_response).strip()
                    if sanitized_final_response_check:
                        ASCIIColors.warning(f"[{self.name}] Round 1 preamble stall with <done/> (text produced, no actions). Injecting continuation mandate.")
                        virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=(
                                "[SYSTEM: CRITICAL. You emitted <done/> after writing a conversational preamble, "
                                "but you did NOT execute any actions. Stating intent DOES NOT execute it.\n\n"
                                "MANDATORY ACTION: You MUST NOW emit the functional tag to perform the action you just described.\n"
                                "- If you said you would unlock files, emit: <unlock_file>filename.pdf</unlock_file>\n"
                                "- If you said you would read a document, emit the appropriate <tool> tag.\n"
                                "- If your task is truly complete, output your final answer and end with <done/>.\n\n"
                                "Do NOT write another preamble. Emit the functional tag NOW.]"
                            )
                        ))
                        ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                        continue

                if ss.completed_actions:
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
                    files_before = self._take_workspace_snapshot()
                    action_reports = []
                    actions_executed_count = 0
                    has_truncated_artifact = False
                    truncated_artifact_title = None

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
                                file_name = ""
                                if is_shell_tool:
                                    command_str = str(tool_params.get("command", "")).strip()
                                    context_aware_sig = f"{tool_name}::{command_str}"
                                else:
                                    normalized_params = dict(tool_params)
                                    param_sig = json.dumps(normalized_params, sort_keys=True, default=str)
                                    context_aware_sig = f"{tool_name}::{param_sig}"
                                    file_name = tool_params.get("file_name", "")

                                    stripped_params = dict(normalized_params)
                                    if "page_or_sheet" in stripped_params:
                                        stripped_params.pop("page_or_sheet", None)
                                    if "max_chars" in stripped_params:
                                        stripped_params.pop("max_chars", None)
                                    stripped_sig = f"{tool_name}::{json.dumps(stripped_params, sort_keys=True, default=str)}"

                                    if stripped_sig in successful_tool_signatures:
                                        action_reports.append(f"Repetitive call to '{tool_name}' with identical file/base parameters blocked. Output already in context. If you need a different page or sheet, change the page_or_sheet parameter.")
                                        continue

                                if context_aware_sig in successful_tool_signatures:
                                    action_reports.append(f"Repetitive call to '{tool_name}' with identical parameters blocked. Output already in context.")
                                    continue

                                if file_name and tool_name in ("tool_read_document_content", "tool_inspect_document", "tool_grep_document"):
                                    file_tool_key = f"__file_consumed__::{tool_name}::{file_name}"
                                    if file_tool_key in successful_tool_signatures:
                                        action_reports.append(
                                            f"🛑 BLOCKED: You have already read '{file_name}' via '{tool_name}'. The tool returned truncated output, meaning the PDF extraction may be limited. "
                                            f"Retrying with different page ranges will NOT help — the extraction returns the same pages. "
                                            f"Do NOT call this tool again for this file. Instead, proceed with what you have, or inform the user that the PDF cannot be fully read."
                                        )
                                        continue

                                tool_res = self._execute_tool(tool_name, tool_params, active_tools)

                                tool_success = isinstance(tool_res, dict) and tool_res.get("success", True) is not False
                                inner_res = tool_res.get("output", tool_res) if isinstance(tool_res, dict) else tool_res
                                is_failure = (
                                    (isinstance(inner_res, dict) and inner_res.get("success") is False)
                                    or (isinstance(tool_res, dict) and tool_res.get("status_code", 200) not in (200, 201))
                                    or (isinstance(tool_res, dict) and bool(tool_res.get("error")))
                                    or (isinstance(inner_res, dict) and bool(inner_res.get("error")) and not inner_res.get("success", True))
                                    or (isinstance(tool_res, dict) and tool_res.get("return_code", 0) != 0)
                                    or (isinstance(inner_res, dict) and inner_res.get("return_code", 0) != 0)
                                )
                                tool_success = not is_failure

                                if tool_success:
                                    successful_tool_signatures.add(context_aware_sig)

                                tool_calls_this_turn.append({"round": round_count, "name": tool_name, "parameters": tool_params})
                                tool_results_this_turn.append({"round": round_count, "name": tool_name, "result": tool_res, "success": tool_success})
                                clean_result_str = _sanitize_tool_result(tool_res, client=self.lollms_client)

                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                    try:
                                        if streaming_callback:
                                            streaming_callback("", MSG_TYPE.MSG_TYPE_TOOL_START, {"tool_name": tool_name, "parameters": tool_params})
                                    except Exception:
                                        pass

                                if tool_success:
                                    report_part = f"=== ✅ TOOL RESULT: {tool_name} ===\n<tool_result name=\"{tool_name}\" status=\"SUCCESS\">\n{clean_result_str}\n</tool_result>"
                                else:
                                    report_part = f"=== ❌ TOOL FAILED: {tool_name} ===\n<tool_result name=\"{tool_name}\" status=\"FAILED\">\n{clean_result_str}\n</tool_result>\n\n⚠️ **Error Analysis Guidance:** Read the error details above carefully to understand what failed. Fix the parameters or try an alternative approach."

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
                                if getattr(self, 'debug_mode', False):
                                    self._dump_error(
                                        error=e,
                                        context_desc="Tool Execution Error",
                                        round_count=round_count,
                                        extra_data={"tool_name": tool_name, "parameters": tool_params, "raw_json": tool_call_json_str}
                                    )
                                action_reports.append(f"[Tool execution error: {e}]")

                        elif action["type"] == "artifact":
                            raw_artifact_xml = action["xml"]
                            was_truncated = action.get("was_truncated", False)
                            try:
                                attrs_match = re.search(r'<art(?:ifact|efact)[^>]*>', raw_artifact_xml, re.IGNORECASE)
                                attrs_str = attrs_match.group(0) if attrs_match else ""
                                body_match = re.search(r'<art(?:ifact|efact)[^>]*>(.*)</art(?:ifact|efact)>', raw_artifact_xml, re.DOTALL | re.IGNORECASE)

                                if not body_match:
                                    action_reports.append(f"❌ TRUNCATED ARTIFACT REJECTED. Missing closing tag for '{title}'. Retry generation.")
                                    continue

                                body_content = body_match.group(1).strip()

                                title = "artifact"
                                lang = "python"
                                operation_type = "full_rewrite"
                                for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                                    if m.group(1).lower() in ("name", "title"):
                                        title = m.group(2)
                                    elif m.group(1).lower() == "language":
                                        lang = m.group(2)
                                    elif m.group(1).lower() == "operation":
                                        operation_type = m.group(2).lower()

                                is_patch = "<<<<<<< SEARCH" in body_content
                                is_append = operation_type == "append"

                                if was_truncated and not is_patch and not is_append:
                                    has_truncated_artifact = True
                                    truncated_artifact_title = title
                                    action_reports.append(
                                        f"❌ GENERATION TRUNCATED for artifact '{title}'. "
                                        "You hit the token generation limit before finishing the file. "
                                        "The file was NOT saved to disk to prevent corruption. "
                                        "You MUST use `operation=\"append\"` in your next <artifact> tag to add the remaining content to the file, or use a SEARCH/REPLACE patch. "
                                        "Start your append/patch from the last few lines you managed to generate."
                                    )
                                    continue

                                file_path = self._resolved_workspace / title
                                is_overwrite = file_path.exists()

                                # Git safety only applies to overwrites of existing files via full_rewrite or patch.
                                # Appending is a modification, but we bypass the strict "are you sure you want to overwrite?" block for appends.
                                if not is_append:
                                    git_block = self._enforce_git_safety(title, is_overwrite)
                                    if git_block:
                                        action_reports.append(git_block)
                                        continue

                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                    try:
                                        if streaming_callback:
                                            streaming_callback("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START, {
                                                "title": title, 
                                                "art_type": "code", 
                                                "language": lang, 
                                                "is_patch": is_patch, 
                                                "operation": "patch" if is_patch else ("append" if is_append else "full_rewrite"),
                                                "execution_phase": True
                                            })
                                    except Exception:
                                        pass

                                if is_patch:
                                    if not file_path.exists():
                                        action_reports.append(f"[SYSTEM ERROR] File '{title}' not found. Cannot apply patch.")
                                        continue

                                    stripped_body = body_content.strip()
                                    if not stripped_body:
                                        action_reports.append(
                                            f"❌ SEARCH/REPLACE BLOCKED for {title}. The patch body is empty. "
                                            "You MUST provide a valid SEARCH/REPLACE block inside the <artifact> tag. "
                                            "Do not output an empty artifact. Retry immediately with the correct format."
                                        )
                                        continue

                                    original_content = file_path.read_text(encoding="utf-8", errors="ignore")
                                    try:
                                        patched_content = _ArtefactManager.apply_aider_patch(original_content, body_content)
                                        file_path.write_text(patched_content, encoding="utf-8")
                                        action_reports.append(f"✅ SEARCH/REPLACE applied successfully to {title}.")
                                        if self._artefact_manager:
                                            self._artefact_manager.update(title=title, new_content=patched_content, language=lang, bump_version=True, active=True)
                                    except Exception as patch_err:
                                        if getattr(self, 'debug_mode', False):
                                            self._dump_error(
                                                error=patch_err,
                                                context_desc="Artifact Patch Error (Block 1)",
                                                round_count=round_count,
                                                extra_data={"title": title, "original_length": len(original_content), "patch_body": body_content[:500]}
                                            )
                                        action_reports.append(f"❌ SEARCH/REPLACE FAILED for {title}. Error: {patch_err}")
                                elif is_append:
                                    if not file_path.exists():
                                        action_reports.append(f"[SYSTEM ERROR] File '{title}' not found. Cannot append. Create it first without operation='append'.")
                                        continue

                                    stripped_body = body_content.strip()
                                    if not stripped_body:
                                        action_reports.append(f"❌ APPEND BLOCKED for {title}. The body is empty.")
                                        continue

                                    original_content = file_path.read_text(encoding="utf-8", errors="ignore")
                                    # Ensure a newline separation if the original file doesn't end with one
                                    sep = "" if original_content.endswith("\n") else "\n"
                                    new_content = original_content + sep + stripped_body + "\n"

                                    file_path.write_text(new_content, encoding="utf-8")
                                    action_reports.append(f"✅ Content appended successfully to {title}.")
                                    if self._artefact_manager:
                                        self._artefact_manager.update(title=title, new_content=new_content, language=lang, bump_version=True, active=True)

                                    actions_executed_count += 1
                                    file_ext = file_path.suffix.lower()
                                    _BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".zip", ".tar", ".gz", ".pdf", ".docx", ".pptx", ".mp3", ".wav", ".mp4"}
                                    if file_ext not in _BINARY_EXTS and len(new_content) < 50000:
                                        try:
                                            self._execute_context_visibility("unlock_file", title)
                                            action_reports.append(f"📂 Auto-loaded '{title}' into context [C].")
                                        except Exception:
                                            pass
                                else:
                                    stripped_body = body_content.strip()
                                    if not stripped_body:
                                        action_reports.append(f"❌ FILE WRITE BLOCKED for {title}. Empty artifact body.")
                                        continue

                                    if self._artefact_manager:
                                        self._artefact_manager.add(title=title, artefact_type="code", content=body_content, language=lang, active=True)
                                    file_path = self._resolved_workspace / title
                                    file_path.parent.mkdir(parents=True, exist_ok=True)
                                    file_path.write_text(body_content, encoding="utf-8")
                                    action_reports.append(f"✅ File {title} created/updated successfully.")
                                    actions_executed_count += 1

                                    _BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".zip", ".tar", ".gz", ".pdf", ".docx", ".pptx", ".mp3", ".wav", ".mp4"}
                                    file_ext = file_path.suffix.lower()
                                    if file_ext not in _BINARY_EXTS and len(body_content) < 50000:
                                        try:
                                            self._execute_context_visibility("unlock_file", title)
                                            action_reports.append(f"📂 Auto-loaded '{title}' into context [C].")
                                        except Exception:
                                            pass
                                    elif file_ext in _BINARY_EXTS:
                                        action_reports.append(f"🚫 Skipped auto-loading binary file '{title}' from context.")
                            except Exception as e:
                                if getattr(self, 'debug_mode', False):
                                    self._dump_error(
                                        error=e,
                                        context_desc="Artifact Processing Error",
                                        round_count=round_count,
                                        extra_data={"raw_xml": raw_artifact_xml}
                                    )
                                action_reports.append(f"[SYSTEM ERROR] Failed to process artifact tag: {e}")

                        elif action["type"] == "context":
                            tag_name = action["tag_name"]
                            raw_xml = action["xml"]
                            try:
                                if tag_name == "scratchpad_clear":
                                    action_reports.append(self._execute_scratchpad_clear())
                                    continue
                                if tag_name == "user_profile_clear":
                                    action_reports.append(self._execute_user_profile_clear())
                                    continue
                                if "scratchpad" in tag_name:
                                    body_match = re.search(r'<scratchpad_(?:append|patch)>(.*?)</scratchpad_(?:append|patch)>', raw_xml, re.DOTALL | re.IGNORECASE)
                                    body_content = body_match.group(1).strip() if body_match else ""
                                    action_reports.append(self._execute_scratchpad_update(tag_name, body_content))
                                    continue
                                if "user_profile_update" in tag_name:
                                    body_match = re.search(r'<user_profile_update>(.*?)</user_profile_update>', raw_xml, re.DOTALL | re.IGNORECASE)
                                    body_content = body_match.group(1).strip() if body_match else ""
                                    action_reports.append(self._execute_user_profile_update(body_content))
                                    continue
                                if tag_name in ("mem_new", "mem_update"):
                                    if not self.memory_manager:
                                        action_reports.append("[SYSTEM ERROR] Memory manager not initialized.")
                                        continue
                                    if tag_name == "mem_new":
                                        content_match = re.search(r'content="([^"]*)"', raw_xml)
                                        tags_match = re.search(r'tags="([^"]*)"', raw_xml)
                                        level_match = re.search(r'level="([^"]*)"', raw_xml)
                                        body_match = re.search(r'<mem_new[^>]*>(.*?)</mem_new>', raw_xml, re.DOTALL | re.IGNORECASE)
                                        mem_content = content_match.group(1) if content_match else (body_match.group(1).strip() if body_match else "")
                                        mem_tags = tags_match.group(1).split(",") if tags_match else []
                                        mem_level = int(level_match.group(1)) if level_match else 2
                                        self.memory_manager.add(content=mem_content, tags=mem_tags, importance=0.9, level=mem_level)
                                        action_reports.append(f"✅ Memory saved successfully: {mem_content[:50]}...")
                                    elif tag_name == "mem_update":
                                        id_match = re.search(r'id="([^"]*)"', raw_xml)
                                        content_match = re.search(r'content="([^"]*)"', raw_xml)
                                        body_match = re.search(r'<mem_update[^>]*>(.*?)</mem_update>', raw_xml, re.DOTALL | re.IGNORECASE)
                                        mem_id = id_match.group(1) if id_match else ""
                                        mem_content = content_match.group(1) if content_match else (body_match.group(1).strip() if body_match else "")
                                        self.memory_manager.update(memory_id=mem_id, content=mem_content)
                                        action_reports.append(f"✅ Memory updated successfully: {mem_id}")
                                    continue

                                body_match = re.search(r'<(?:unlock_file|lock_file|hide_file|pin_file|unpin_file|collapse_folder|uncollapse_folder)[^>]*>(.*?)</(?:unlock_file|lock_file|hide_file|pin_file|unpin_file|collapse_folder|uncollapse_folder)>', raw_xml, re.DOTALL | re.IGNORECASE)
                                body_content = body_match.group(1).strip() if body_match else ""

                                context_sig = f"{tag_name}::{body_content}"
                                if context_sig in seen_context_signatures:
                                    rep_msg = f"Repetitive context action '{tag_name}' with identical parameters blocked. Files are already in the requested state or failed previously. Do not retry."
                                    action_reports.append(rep_msg)
                                    continue

                                seen_context_signatures.add(context_sig)

                                vis_result = self._execute_context_visibility(tag_name, body_content)
                                status_str = ""
                                loaded_contents = {}
                                is_failure = False

                                if isinstance(vis_result, dict):
                                    status_str = vis_result.get("status_str", "")
                                    loaded_contents = vis_result.get("loaded_contents", {})
                                    is_failure = bool(vis_result.get("not_found") or vis_result.get("blocked_files"))

                                    if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {
                                            "action": tag_name,
                                            "files": vis_result.get("processed_files", []) + vis_result.get("already_in_state", []),
                                            "status": "failure" if is_failure else "success",
                                            "error": vis_result.get("error") if is_failure else None
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
                                if getattr(self, 'debug_mode', False):
                                    self._dump_error(
                                        error=ctx_err,
                                        context_desc="Context Visibility Error",
                                        round_count=round_count,
                                        extra_data={"tag_name": tag_name, "raw_xml": raw_xml}
                                    )
                                action_reports.append(f"[SYSTEM ERROR] Failed to process context tag: {ctx_err}")

                    files_after = self._take_workspace_snapshot()
                    changes = self._sync_workspace(files_before, files_after)
                    if changes:
                        workspace_changes.extend(changes)

                    ss.completed_actions = []

                if has_truncated_artifact:
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content=(
                            f"[SYSTEM: CRITICAL ERROR. Your previous generation of '{truncated_artifact_title}' was TRUNCATED because you hit the token limit. "
                            "The file was NOT saved. You MUST rewrite the COMPLETE file from scratch using a standard <artifact> tag (NOT a SEARCH/REPLACE patch). "
                            "Reproduce the existing content exactly and append the missing ending. Do NOT emit `<done/>` until the file is complete.]"
                        )
                    ))
                    ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                    continue

                if not ss.completed_actions and not tool_calls_this_turn and not workspace_changes and not ss.was_done_detected() and round_count == 1:
                    if raw_round_text.strip():
                        ASCIIColors.warning(f"[{self.name}] Round 1 preamble stall (text produced, no actions, no <done/>). Injecting continuation mandate.")
                        virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=(
                                "[SYSTEM: CRITICAL. You wrote a conversational preamble but you STOPPED without executing "
                                "the actual action. Stating intent DOES NOT execute it.\n\n"
                                "MANDATORY ACTION: You MUST NOW emit the functional tag to perform the action you just described.\n"
                                "- If you said you would unlock files, emit: <unlock_file>filename.pdf</unlock_file>\n"
                                "- If your task is truly complete, output your final answer and end with <done/>.\n\n"
                                "Do NOT write another preamble. Emit the functional tag NOW.]"
                            )
                        ))
                        ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                        continue

                if not final_response.strip():
                    ASCIIColors.warning(f"[{self.name}] Empty response after <done/> with no prior actions. Terminating.")
                    final_response = "[Task terminated: The agent produced no actionable output.]"
                    break

                sanitized_final_response = re.sub(r'<[^>]+>', '', final_response).strip()

                if not sanitized_final_response and not tool_calls_this_turn and not workspace_changes and not ss.completed_actions and round_count == 1:
                    if getattr(self, 'debug_mode', False):
                        self._dump_error(
                            error=Exception("Empty response with <done/> on round 1"),
                            context_desc="Empty Response Interception",
                            round_count=round_count,
                            extra_data={"virtual_history": [vh.content for vh in virtual_history]}
                        )
                    ASCIIColors.warning(f"[{self.name}] 🚫 Empty response with <done/> detected on round 1. Forcing continuation.")
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content="[SYSTEM: Your previous response was empty. Continue your task or emit <done/> if you are truly finished.]"
                    ))
                    ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                    continue
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: <done/> detected ===")
                break

            if ss.completed_actions:
                raw_round_text = ss.get_clean_text()
                virtual_history.append(SimpleNamespace(sender_type="assistant", content=raw_round_text))

                files_before = self._take_workspace_snapshot()
                actions_executed_count = 0
                has_truncated_artifact = False
                truncated_artifact_title = None

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
                            file_name = ""
                            if is_shell_tool:
                                command_str = str(tool_params.get("command", "")).strip()
                                context_aware_sig = f"{tool_name}::{command_str}"
                            else:
                                normalized_params = dict(tool_params)
                                param_sig = json.dumps(normalized_params, sort_keys=True, default=str)
                                context_aware_sig = f"{tool_name}::{param_sig}"
                                file_name = tool_params.get("file_name", "")

                                stripped_params = dict(normalized_params)
                                if "page_or_sheet" in stripped_params:
                                    stripped_params.pop("page_or_sheet", None)
                                if "max_chars" in stripped_params:
                                    stripped_params.pop("max_chars", None)
                                stripped_sig = f"{tool_name}::{json.dumps(stripped_params, sort_keys=True, default=str)}"

                                if stripped_sig in successful_tool_signatures:
                                    action_reports.append(f"Repetitive call to '{tool_name}' with identical file/base parameters blocked. Output already in context. If you need a different page or sheet, change the page_or_sheet parameter.")
                                    continue

                            if context_aware_sig in successful_tool_signatures:
                                action_reports.append(f"Repetitive call to '{tool_name}' with identical parameters blocked. Output already in context.")
                                continue

                            if file_name and tool_name in ("tool_read_document_content", "tool_inspect_document", "tool_grep_document"):
                                file_tool_key = f"__file_consumed__::{tool_name}::{file_name}"
                                if file_tool_key in successful_tool_signatures:
                                    action_reports.append(
                                        f"🛑 BLOCKED: You have already read '{file_name}' via '{tool_name}'. The tool returned truncated output, meaning the PDF extraction may be limited. "
                                        f"Retrying with different page ranges will NOT help — the extraction returns the same pages. "
                                        f"Do NOT call this tool again for this file. Instead, proceed with what you have, or inform the user that the PDF cannot be fully read."
                                    )
                                    continue

                            tool_res = self._execute_tool(tool_name, tool_params, active_tools)

                            inner_res = tool_res.get("output", tool_res) if isinstance(tool_res, dict) else tool_res
                            is_failure = (
                                (isinstance(inner_res, dict) and inner_res.get("success") is False)
                                or (isinstance(tool_res, dict) and tool_res.get("status_code", 200) not in (200, 201))
                                or (isinstance(tool_res, dict) and bool(tool_res.get("error")))
                                or (isinstance(inner_res, dict) and bool(inner_res.get("error")) and not inner_res.get("success", True))
                                or (isinstance(tool_res, dict) and tool_res.get("return_code", 0) != 0)
                                or (isinstance(inner_res, dict) and inner_res.get("return_code", 0) != 0)
                            )
                            tool_success = not is_failure

                            result_text = ""
                            if isinstance(tool_res, dict):
                                result_text = str(tool_res.get("output", "")) + str(tool_res.get("error", ""))
                            else:
                                result_text = str(tool_res)

                            is_truncated = "truncated" in result_text.lower() and "more lines" in result_text.lower()
                            if is_truncated and file_name and tool_name in ("tool_read_document_content", "tool_inspect_document", "tool_grep_document"):
                                file_tool_key = f"__file_consumed__::{tool_name}::{file_name}"
                                successful_tool_signatures.add(file_tool_key)
                                ASCIIColors.warning(f"[{self.name}] Tool returned truncated output for '{file_name}'. Marking as consumed to prevent retry loops.")

                            if tool_success:
                                successful_tool_signatures.add(context_aware_sig)

                            tool_calls_this_turn.append({"round": round_count, "name": tool_name, "parameters": tool_params})
                            tool_results_this_turn.append({"round": round_count, "name": tool_name, "result": tool_res, "success": tool_success})
                            clean_result_str = _sanitize_tool_result(tool_res, client=self.lollms_client)
                            actions_executed_count += 1

                            if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                try:
                                    if streaming_callback:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_TOOL_START, {"tool_name": tool_name, "parameters": tool_params})
                                except Exception:
                                    pass

                            if tool_success:
                                report_part = f"=== ✅ TOOL RESULT: {tool_name} ===\n<tool_result name=\"{tool_name}\" status=\"SUCCESS\">\n{clean_result_str}\n</tool_result>"
                            else:
                                report_part = f"=== ❌ TOOL FAILED: {tool_name} ===\n<tool_result name=\"{tool_name}\" status=\"FAILED\">\n{clean_result_str}\n</tool_result>\n\n⚠️ **Error Analysis Guidance:** Read the error details above carefully to understand what failed. Fix the parameters or try an alternative approach."

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

                            if not body_match:
                                action_reports.append(f"❌ TRUNCATED ARTIFACT REJECTED. Missing closing tag for '{title}'. Retry generation.")
                                continue

                            body_content = body_match.group(1).strip()

                            title = "artifact"
                            lang = "python"
                            operation_type = "full_rewrite"
                            for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                                if m.group(1).lower() in ("name", "title"):
                                    title = m.group(2)
                                elif m.group(1).lower() == "language":
                                    lang = m.group(2)
                                elif m.group(1).lower() == "operation":
                                    operation_type = m.group(2).lower()

                            is_patch = "<<<<<<< SEARCH" in body_content
                            is_append = operation_type == "append"

                            file_path = self._resolved_workspace / title
                            is_overwrite = file_path.exists()

                            if not is_append:
                                git_block = self._enforce_git_safety(title, is_overwrite)
                                if git_block:
                                    action_reports.append(git_block)
                                    continue

                            art_type_match = re.search(r'type=["\']([^"\']*)["\']', attrs_str, re.IGNORECASE)
                            resolved_art_type = art_type_match.group(1) if art_type_match else "code"

                            if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                try:
                                    if streaming_callback:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START, {
                                            "title": title, 
                                            "art_type": resolved_art_type, 
                                            "language": lang, 
                                            "is_patch": is_patch, 
                                            "operation": "patch" if is_patch else ("append" if is_append else "full_rewrite"),
                                            "execution_phase": True
                                        })
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
                                    patched_content = _ArtefactManager.apply_aider_patch(original_content, body_content)
                                    file_path.write_text(patched_content, encoding="utf-8")
                                    action_reports.append(f"✅ SEARCH/REPLACE applied successfully to {title}.")
                                    if self._artefact_manager:
                                        self._artefact_manager.update(title=title, new_content=patched_content, language=lang, bump_version=True, active=True)
                                except Exception as patch_err:
                                    if getattr(self, 'debug_mode', False):
                                        self._dump_error(
                                            error=patch_err,
                                            context_desc="Artifact Patch Error (Block 2)",
                                            round_count=round_count,
                                            extra_data={"title": title, "original_length": len(original_content), "patch_body": body_content[:500]}
                                        )
                                    action_reports.append(f"❌ SEARCH/REPLACE FAILED for {title}. Error: {patch_err}")
                            elif is_append:
                                if not file_path.exists():
                                    action_reports.append(f"[SYSTEM ERROR] File '{title}' not found. Cannot append. Create it first without operation='append'.")
                                    continue

                                stripped_body = body_content.strip()
                                if not stripped_body:
                                    action_reports.append(f"❌ APPEND BLOCKED for {title}. The body is empty.")
                                    continue

                                original_content = file_path.read_text(encoding="utf-8", errors="ignore")
                                sep = "" if original_content.endswith("\n") else "\n"
                                new_content = original_content + sep + stripped_body + "\n"

                                file_path.write_text(new_content, encoding="utf-8")
                                action_reports.append(f"✅ Content appended successfully to {title}.")
                                if self._artefact_manager:
                                    self._artefact_manager.update(title=title, new_content=new_content, language=lang, bump_version=True, active=True)

                                actions_executed_count += 1
                                file_ext = file_path.suffix.lower()
                                _BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".zip", ".tar", ".gz", ".pdf", ".docx", ".pptx", ".mp3", ".wav", ".mp4"}
                                if file_ext not in _BINARY_EXTS and len(new_content) < 50000:
                                    try:
                                        self._execute_context_visibility("unlock_file", title)
                                        action_reports.append(f"📂 Auto-loaded '{title}' into context [C].")
                                    except Exception:
                                        pass
                            else:
                                stripped_body = body_content.strip()
                                if not stripped_body:
                                    action_reports.append(f"❌ FILE WRITE BLOCKED for {title}. Empty artifact body.")
                                    continue

                                if self._artefact_manager:
                                    self._artefact_manager.add(title=title, artefact_type="code", content=body_content, language=lang, active=True)
                                file_path = self._resolved_workspace / title
                                file_path.parent.mkdir(parents=True, exist_ok=True)
                                file_path.write_text(body_content, encoding="utf-8")
                                action_reports.append(f"✅ File {title} created/updated successfully.")
                                actions_executed_count += 1

                                _BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".zip", ".tar", ".gz", ".pdf", ".docx", ".pptx", ".mp3", ".wav", ".mp4"}
                                file_ext = file_path.suffix.lower()
                                if file_ext not in _BINARY_EXTS and len(body_content) < 50000:
                                    try:
                                        self._execute_context_visibility("unlock_file", title)
                                        action_reports.append(f"📂 Auto-loaded '{title}' into context [C].")
                                    except Exception:
                                        pass
                                elif file_ext in _BINARY_EXTS:
                                    action_reports.append(f"🚫 Skipped auto-loading binary file '{title}' from context.")
                        except Exception as e:
                            action_reports.append(f"[SYSTEM ERROR] Failed to process artifact tag: {e}")

                    elif action["type"] == "context":
                        tag_name = action["tag_name"]
                        raw_xml = action["xml"]
                        try:
                            if tag_name == "scratchpad_clear":
                                res_msg = self._execute_scratchpad_clear()
                                action_reports.append(res_msg)
                                actions_executed_count += 1
                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "success" if "✅" in res_msg else "failure", "error": None if "✅" in res_msg else res_msg})
                                if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                    if "✅" in res_msg: streaming_callback(f'<status>success</status>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                    else: streaming_callback(f'<status>failure</status>\n<error>{res_msg}</error>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                continue

                            if tag_name == "user_profile_clear":
                                res_msg = self._execute_user_profile_clear()
                                action_reports.append(res_msg)
                                actions_executed_count += 1
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
                                actions_executed_count += 1
                                if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                    if "✅" in res_msg: streaming_callback(f'<status>success</status>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                    else: streaming_callback(f'<status>failure</status>\n<error>{res_msg}</error>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                continue

                            if "user_profile_update" in tag_name:
                                body_match = re.search(r'<user_profile_update>(.*?)</user_profile_update>', raw_xml, re.DOTALL | re.IGNORECASE)
                                body_content = body_match.group(1).strip() if body_match else ""
                                res_msg = self._execute_user_profile_update(body_content)
                                action_reports.append(res_msg)
                                actions_executed_count += 1
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
                                        actions_executed_count += 1
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
                                    actions_executed_count += 1
                                    if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {"action": tag_name, "files": [], "status": "success", "error": None})
                                    if event_mode == EventMode.PROCESSING_TAG_MODE and streaming_callback:
                                        streaming_callback(f'<status>success</status>\n', MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                continue

                            body_match = re.search(r'<(?:unlock_file|lock_file|hide_file|pin_file|unpin_file|collapse_folder|uncollapse_folder)[^>]*>(.*?)</(?:unlock_file|lock_file|hide_file|pin_file|unpin_file|collapse_folder|uncollapse_folder)>', raw_xml, re.DOTALL | re.IGNORECASE)
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
                            actions_executed_count += 1

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
                elif not raw_round_text.strip():
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content="[SYSTEM: Your context visibility operation was executed. Continue your task or emit <done/> if finished.]"
                    ))

                ss.completed_actions = []
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Actions dispatched, continuing ===")
                continue

            if ss.was_done_detected() and not ss.completed_actions:
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: <done/> detected (no actions) ===")
                break

            raw_round_text = ss.get_clean_text()

            # ── 🧹 DYNAMIC HISTORY SANITIZATION (Strict Non-Placeholder Strategy) ──
            if virtual_history:
                history_len = len(virtual_history)
                for idx, vh in enumerate(virtual_history):
                    if vh.sender_type == "assistant":
                        distance = history_len - 1 - idx
                        vh.content = HistoryManager._sanitize_for_context(vh.content, distance_from_end=distance)

            # ── 🛑 TERTIARY <done/> / <end/> FALLBACK ──
            raw_round_text = ss.get_clean_text()
            done_pattern = re.compile(r'(?i)<(?:done|end)\s*/?>')
            done_match = done_pattern.search(raw_round_text)
            if done_match:
                final_response = done_pattern.sub('', raw_round_text).strip()
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: <done/> detected (fallback) ===")
                break

            # ── 🛡️ SAFETY NET: Detect phantom artifact processing ──
            # If the LLM emitted <processing> or <artifact> markers in the raw stream
            # but completed_actions is empty (artifact was never fully parsed/dispatched),
            # we must NOT exit. Force a continuation to prevent silent termination.
            if not ss.completed_actions and not was_cancelled:
                _has_artifact_evidence = bool(re.search(
                    r'<(?:processing|artifact|artefact)\b',
                    raw_llm_output_buffer or "",
                    re.IGNORECASE
                ))
                if _has_artifact_evidence:
                    ASCIIColors.warning(f"[{self.name}] Phantom artifact detected (processing markers in stream but no completed actions). Forcing continuation.")
                    virtual_history.append(SimpleNamespace(
                            sender_type="assistant",
                            content=raw_round_text.strip()
                        ))
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content="[SYSTEM: Your previous artifact was detected but not fully processed. If you intended to write a file, emit the <artifact> tag again with the complete content. If your task is complete, output your final answer and end with <done/>.]"
                    ))
                    continue

            text_is_repetitive = False
            has_new_actions_this_round = bool(ss.completed_actions) or bool(raw_round_text.strip())

            _xml_tool_pattern = re.compile(r'^\s*<tool_\w+[\s/>]', re.MULTILINE | re.IGNORECASE)
            if _xml_tool_pattern.search(raw_round_text):
                ASCIIColors.warning(f"[{self.name}] Malformed XML tool syntax detected (Round {round_count}). Injecting format correction.")
                virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=(
                        "[SYSTEM: CRITICAL FORMAT ERROR. You emitted a tool call using XML self-closing syntax like `<tool_read_document_content file_name=\"...\" />`. "
                        "This is WRONG. The system does NOT execute XML-attribute tool calls. "
                        "You MUST use the JSON format inside a `<tool>` tag. The correct syntax is:\n"
                        "<tool>{\"name\": \"tool_read_document_content\", \"parameters\": {\"file_name\": \"...\", \"page_or_sheet\": \"...\", \"max_chars\": 15000}}</tool>\n"
                        "Output the corrected tool call NOW using the JSON format. Do NOT repeat the XML syntax.]"
                    )
                ))
                object.__setattr__(self, '_consecutive_stall_count', 0)
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                continue

            stripped_round_text = raw_round_text.strip()
            if stripped_round_text and len(virtual_history) > 0:
                last_assistant_text = None
                for vh in reversed(virtual_history):
                    if vh.sender_type == "assistant" and vh.content.strip():
                        candidate = vh.content.strip()
                        if not candidate.startswith("[Assistant executed batched actions]"):
                            last_assistant_text = candidate
                        break
                if last_assistant_text:
                    _last_clean = re.sub(r'<[^>]+>', '', last_assistant_text).strip()
                    _current_clean = re.sub(r'<[^>]+>', '', stripped_round_text).strip()
                    if _current_clean and _last_clean:
                        if _current_clean == _last_clean:
                            text_is_repetitive = True
                        elif len(_current_clean) > 80 and _current_clean in _last_clean:
                            text_is_repetitive = True
                        elif len(_last_clean) > 80 and _last_clean in _current_clean:
                            text_is_repetitive = True
                        elif len(_current_clean) > 60:
                            words_last = set(_last_clean.split())
                            words_current = set(_current_clean.split())
                            if len(words_last) > 0 and len(words_current) > 0:
                                overlap = len(words_last & words_current) / max(len(words_last), len(words_current))
                                if overlap > 0.85:
                                    text_is_repetitive = True

            if not text_is_repetitive and stripped_round_text and not has_new_actions_this_round:
                lines_in_response = stripped_round_text.splitlines()
                non_empty_lines = [l.strip() for l in lines_in_response if l.strip()]
                if len(non_empty_lines) >= 2:
                    from collections import Counter as _Counter
                    line_counts = _Counter(non_empty_lines)
                    most_common_line, most_common_count = line_counts.most_common(1)[0]
                    if most_common_count >= 2 and len(most_common_line) > 20:
                        repetition_ratio = most_common_count / len(non_empty_lines)
                        if repetition_ratio >= 0.5:
                            text_is_repetitive = True
                            deduplicated_lines = []
                            seen_lines = set()
                            for l in non_empty_lines:
                                if l not in seen_lines:
                                    deduplicated_lines.append(l)
                                    seen_lines.add(l)
                            if deduplicated_lines:
                                stripped_round_text = "\n".join(deduplicated_lines)
                                raw_round_text = stripped_round_text
                                ss.content = stripped_round_text
                            ASCIIColors.warning(f"[{self.name}] Intra-round text duplication detected (line repeated {most_common_count}x, ratio: {repetition_ratio:.0%}). Deduplicated to {len(deduplicated_lines)} unique line(s).")
                        elif most_common_count >= 2:
                            consecutive_dup_count = 0
                            for i in range(1, len(non_empty_lines)):
                                if non_empty_lines[i] == non_empty_lines[i - 1]:
                                    consecutive_dup_count += 1
                            if consecutive_dup_count >= 1:
                                text_is_repetitive = True
                                deduplicated_lines = []
                                prev_line = None
                                for l in non_empty_lines:
                                    if l != prev_line:
                                        deduplicated_lines.append(l)
                                    prev_line = l
                                if deduplicated_lines:
                                    stripped_round_text = "\n".join(deduplicated_lines)
                                    raw_round_text = stripped_round_text
                                    ss.content = stripped_round_text
                                ASCIIColors.warning(f"[{self.name}] Intra-round consecutive text duplication detected ({consecutive_dup_count} consecutive duplicate lines). Deduplicated to {len(deduplicated_lines)} unique line(s).")

            if text_is_repetitive:
                consecutive_stall_count = getattr(self, '_consecutive_stall_count', 0) + 1
                object.__setattr__(self, '_consecutive_stall_count', consecutive_stall_count)

                if consecutive_stall_count >= 2:
                    ASCIIColors.error(f"[{self.name}] Breaking after {consecutive_stall_count} consecutive repetition+stall cycles.")
                    final_response = re.sub(r'(?i)<done\s*/?>', '', ss.get_clean_text()).strip()
                    if not final_response:
                        final_response = "[Task terminated: The agent produced repetitive text multiple times due to a tool failure or sandbox restriction. The last tool call may have been blocked.]"
                    break

                ASCIIColors.warning(f"[{self.name}] Repetitive text preamble detected (Round {round_count}, streak: {consecutive_stall_count}/2). Injecting correction — NOT terminating.")
                virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=(
                        "[SYSTEM: CRITICAL ERROR. You are producing repetitive conversational preambles without emitting functional tags. "
                        "This usually happens when a tool call failed or the stream was interrupted.\n\n"
                        "MANDATORY ACTION: Stop writing prose entirely. You must output the RAW JSON object for the tool inside `<tool>` tags immediately. "
                        "Do not include any conversational text before or after the tag. "
                        "Example:\n"
                        "<tool>{\"name\": \"tool_annotate_document\", \"parameters\": {\"file_name\": \"file.pdf\", \"annotation_type\": \"comment\", \"search_text\": \"text\", \"comment\": \"comment\"}}</tool>"
                    )
                ))
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                continue

            # ── 🧹 AUTONOMOUS CONTEXT COMPACTION ──
            ctx_health = self._calculate_context_fill(stable_system_prompt, base_conversation, virtual_history, raw_round_text)
            if ctx_health["fill_percentage"] > 85.0 and len(virtual_history) > 0 and not getattr(self, '_compaction_triggered_this_turn', False):
                ASCIIColors.warning(f"[{self.name}] Context fill at {ctx_health['fill_percentage']}%. Triggering autonomous compaction.")
                object.__setattr__(self, '_compaction_triggered_this_turn', True)

                virtual_history = self._compact_virtual_history(virtual_history, base_conversation, streaming_callback)

                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content="[SYSTEM: Context has been compacted. Please continue your task based on the summarized history. If you were finished, output your final answer and <done/>.]"
                ))
                continue

            if round_count > 1 and tool_calls_this_turn and not was_cancelled and not has_new_actions_this_round:
                consecutive_stall_count = getattr(self, '_consecutive_stall_count', 0) + 1
                object.__setattr__(self, '_consecutive_stall_count', consecutive_stall_count)

                if consecutive_stall_count >= 3:
                    ASCIIColors.error(f"[{self.name}] Terminating after {consecutive_stall_count} consecutive stalls. The LLM appears unable to proceed.")
                    final_response = re.sub(r'(?i)<done\s*/?>', '', ss.get_clean_text()).strip()
                    if not final_response:
                        final_response = "[Task terminated: The agent stalled repeatedly without producing actionable output. This may indicate the context window is full or the task is too complex for the current model.]"
                    break

                ASCIIColors.warning(f"[{self.name}] Mid-task stall detected (Round {round_count}, consecutive: {consecutive_stall_count}). LLM stopped without <done/> or new actions. Forcing continuation.")
                virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
                recent_tool_names = [tc.get("name", "") for tc in tool_calls_this_turn[-3:]]
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=self._build_progressive_continuation_prompt(consecutive_stall_count, recent_tool_names)
                ))
                continue
            elif has_new_actions_this_round and not text_is_repetitive:
                object.__setattr__(self, '_consecutive_stall_count', 0)
                if raw_round_text.strip():
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
            if not text_is_repetitive and stripped_round_text:
                object.__setattr__(self, '_consecutive_stall_count', 0)

            ctx_health = self._calculate_context_fill(stable_system_prompt, base_conversation, virtual_history, raw_round_text)

            if getattr(self, 'debug_mode', False):
                gen_tokens = len(raw_round_text) // 4
                raw_tokens = len(raw_llm_output_buffer) // 4
                ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: No <done/> detected. Generated ~{gen_tokens} tokens (raw buffer: ~{raw_tokens} tokens). Total context fill: {ctx_health.get('fill_percentage', 0.0):.1f}% ===")

            if ctx_health["fill_percentage"] > 85.0 and len(virtual_history) > 0 and not getattr(self, '_compaction_triggered_this_turn', False):
                ASCIIColors.warning(f"[{self.name}] Context fill at {ctx_health['fill_percentage']}%. Triggering autonomous compaction.")
                object.__setattr__(self, '_compaction_triggered_this_turn', True)

                virtual_history = self._compact_virtual_history(virtual_history, base_conversation, streaming_callback)

                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content="[SYSTEM: Context has been compacted. Please continue your task based on the summarized history. If you were finished, output your final answer and <done/>.]"
                ))
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Context compaction triggered ===")
                continue

            has_actions_this_round = bool(ss.completed_actions)
            if len(tool_calls_this_turn) > 0 or getattr(ss, 'context_trigger', False) or getattr(ss, 'artifact_trigger', False) or has_actions_this_round:
                if not raw_round_text.strip() and not has_actions_this_round:
                    empty_response_count = getattr(self, '_consecutive_empty_responses', 0) + 1
                    object.__setattr__(self, '_consecutive_empty_responses', empty_response_count)

                    if empty_response_count >= 2:
                        ASCIIColors.warning(f"[{self.name}] Consecutive empty LLM responses detected ({empty_response_count}). Terminating loop to prevent spin.")
                        final_response = "[Terminated: LLM stopped generating without completing the task.]"
                        break

                    ASCIIColors.warning(f"[{self.name}] Empty LLM response detected after action (attempt {empty_response_count}). Injecting continuation mandate.")
                else:
                    object.__setattr__(self, '_consecutive_empty_responses', 0)
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))

                recent_tool_names = [tc.get("name", "") for tc in tool_calls_this_turn[-3:]]

                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=self._build_progressive_continuation_prompt(
                        getattr(self, '_consecutive_stall_count', 0),
                        recent_tool_names
                    )
                ))
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: No <done/> detected, injecting continuation mandate ===")
                continue

            has_artifact_this_round = any(act.get("type") == "artifact" for act in ss.completed_actions)
            if has_artifact_this_round:
                virtual_history = self._apply_rolling_artifact_compaction(virtual_history, base_conversation)

            # ── 🛡️ ROUND 1 PREAMBLE STALL INTERCEPTOR ──
            # If the LLM produced text on Round 1 but emitted NO functional tags
            # and NO <done/>, it wrote a conversational preface and stopped.
            # We MUST inject a continuation mandate instead of breaking cleanly.
            # This prevents the agent from terminating before doing any work.
            if (
                round_count == 1
                and not ss.was_done_detected()
                and not ss.was_action_dispatched()
                and not has_new_actions_this_round
                and not tool_calls_this_turn
                and stripped_round_text
                and not text_is_repetitive
            ):
                ASCIIColors.warning(f"[{self.name}] Round 1 preamble stall detected (text produced, no actions, no <done/>). Injecting continuation mandate.")

                virtual_history.append(SimpleNamespace(
                    sender_type="assistant",
                    content=ss.get_clean_text().strip()
                ))

                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=(
                        "[SYSTEM: CRITICAL. You wrote a conversational preamble stating your intent "
                        "(e.g., 'Let me check...' or 'I'll create...') but you STOPPED without executing "
                        "the actual action. Stating intent DOES NOT execute it.\n\n"
                        "MANDATORY ACTION: You MUST NOW emit the functional tag to perform the action you just described.\n"
                        "- If you said you would check skills, emit: <tool>{\"name\": \"tool_list_skills\", \"parameters\": {}}</tool>\n"
                        "- If you said you would create a skill, emit: <tool>{\"name\": \"tool_create_skill\", \"parameters\": {...}}</tool>\n"
                        "- If your task is truly complete, output your final answer and end with <done/>.\n\n"
                        "Do NOT write another preamble. Emit the functional tag NOW.]"
                    )
                ))
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Round 1 preamble stall intercepted, continuing ===")
                continue

            has_malformed_tag = "<tool" in raw_round_text.lower() or "<art" in raw_round_text.lower()

            if has_malformed_tag:
                if not raw_round_text.strip():
                    empty_response_count = getattr(self, '_consecutive_empty_responses', 0) + 1
                    object.__setattr__(self, '_consecutive_empty_responses', empty_response_count)
                    if empty_response_count >= 3:
                        ASCIIColors.warning(f"[{self.name}] Consecutive empty responses with malformed tags ({empty_response_count}). Terminating.")
                        final_response = "[Terminated: LLM repeatedly produced malformed tags without content.]"
                        break
                else:
                    object.__setattr__(self, '_consecutive_empty_responses', 0)
                    object.__setattr__(self, '_consecutive_stall_count', 0)

                ASCIIColors.warning("[LollmsPersonality.chat] Malformed functional tag detected. Injecting format correction.")
                virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
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

            if len(tool_calls_this_turn) > 0 and not was_cancelled:
                consecutive_stall_count = getattr(self, '_consecutive_stall_count', 0) + 1
                object.__setattr__(self, '_consecutive_stall_count', consecutive_stall_count)

                if consecutive_stall_count >= 3:
                    ASCIIColors.warning(f"[{self.name}] Terminating after {consecutive_stall_count} consecutive text-only stalls after tools. LLM is stuck in preamble mode.")
                    final_response = re.sub(r'(?i)<done\s*/?>', '', ss.get_clean_text()).strip()
                    if not final_response:
                        final_response = "[Task terminated: The agent repeatedly produced text preambles without executing any actions.]"
                    break

                ASCIIColors.warning(f"[{self.name}] LLM stopped without <done/> after tools were executed (stall #{consecutive_stall_count}). Injecting continuation mandate.")
                virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
                recent_tool_names = [tc.get("name", "") for tc in tool_calls_this_turn[-3:]]
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=self._build_progressive_continuation_prompt(consecutive_stall_count, recent_tool_names)
                ))
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Text-only after tools, injecting continuation mandate ===")
                continue

            if not final_response:
                final_response = re.sub(r'(?i)<done\s*/?>', '', ss.get_clean_text()).strip()

            stripped_final_check = re.sub(r'<[^>]+>', '', final_response).strip()

            if (
                not was_cancelled
                and not tool_calls_this_turn
                and not workspace_changes
                and not getattr(self, '_consecutive_stall_count', 0) >= 3
                and not stripped_final_check
                and round_count == 1
                and not ss.was_done_detected()
                and not ss.was_action_dispatched()
            ):
                empty_response_count = getattr(self, '_consecutive_empty_responses', 0) + 1
                object.__setattr__(self, '_consecutive_empty_responses', empty_response_count)

                if empty_response_count >= 2:
                    ASCIIColors.warning(f"[{self.name}] Consecutive empty responses with no actions or <done/> ({empty_response_count}). Likely context exhaustion or model failure. Terminating.")
                    final_response = (
                        "[Empty response: The LLM produced 0 tokens. This typically indicates the context window is exhausted "
                        "(input exceeds the model's maximum context length). Try unloading files with /clear-files, "
                        "clearing history with /clear-history, or switching to a model with a larger context window.]"
                    )
                    break

                ASCIIColors.warning(f"[{self.name}] Empty LLM response on round 1 (no actions, no <done/>). Possible context exhaustion. Injecting continuation mandate (attempt {empty_response_count}).")
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=(
                        "[SYSTEM: Your previous response was completely empty (0 tokens generated). "
                        "This usually means the context window is full. "
                        "If you can see this message, respond with a brief status and emit <done/>. "
                        "If you cannot generate any text, the user needs to reduce the context load.]"
                    )
                ))
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                continue

            if (
                not was_cancelled
                and not tool_calls_this_turn
                and not workspace_changes
                and not getattr(self, '_consecutive_stall_count', 0) >= 3
                and stripped_final_check
                and len(stripped_final_check) > 10
            ):
                consecutive_stall_count = getattr(self, '_consecutive_stall_count', 0) + 1
                object.__setattr__(self, '_consecutive_stall_count', consecutive_stall_count)

                if consecutive_stall_count >= 3:
                    ASCIIColors.warning(f"[{self.name}] Terminating after {consecutive_stall_count} consecutive text-only stalls. LLM is stuck in preamble mode.")
                    if not final_response:
                        final_response = "[Task terminated: The agent repeatedly produced text preambles without executing any actions.]"
                    break

                ASCIIColors.warning(f"[{self.name}] Text-only stall detected (Round {round_count}, consecutive: {consecutive_stall_count}). No actions, no <done/>. Injecting continuation mandate.")

                virtual_history.append(SimpleNamespace(
                    sender_type="assistant",
                    content=ss.get_clean_text()
                ))

                recent_tool_names = [tc.get("name", "") for tc in tool_calls_this_turn[-3:]]
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=self._build_progressive_continuation_prompt(consecutive_stall_count, recent_tool_names)
                ))
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Text-only stall intercepted, enforcing continuation ===")
                continue

            if getattr(self, 'debug_mode', False):
                ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Clean exit ===")
            break

        context_health = {"used_tokens": 0, "max_tokens": 0, "fill_percentage": 0.0}
        try:
            if self.lollms_client and hasattr(self.lollms_client, 'get_ctx_size'):
                max_ctx = self.lollms_client.get_ctx_size() or 0
                if max_ctx > 0:
                    total_used = 0
                    if hasattr(self.lollms_client, 'count_tokens'):
                        total_used = self.lollms_client.count_tokens(stable_system_prompt)
                        for msg in base_conversation:
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

        if not final_response and ss:
            final_response = re.sub(r'(?i)<done\s*/?>', '', ss.get_clean_text()).strip()

        if ss.completed_actions and not was_cancelled:
            ASCIIColors.warning(f"[{self.name}] ⚠️ Generation ended with {len(ss.completed_actions)} unexecuted buffered action(s). Flushing now.")

            files_before = self._take_workspace_snapshot()
            action_reports = []

            for action in ss.completed_actions:
                if action["type"] == "artifact":
                    raw_artifact_xml = action["xml"]
                    try:
                        attrs_match = re.search(r'<art(?:ifact|efact)[^>]*>', raw_artifact_xml, re.IGNORECASE)
                        attrs_str = attrs_match.group(0) if attrs_match else ""
                        body_match = re.search(r'<art(?:ifact|efact)[^>]*>(.*)</art(?:ifact|efact)>', raw_artifact_xml, re.DOTALL | re.IGNORECASE)

                        if not body_match:
                            action_reports.append(f"❌ TRUNCATED ARTIFACT REJECTED. Missing closing tag for '{title}'. Retry generation.")
                            continue

                        body_content = body_match.group(1).strip()

                        title = "artifact"
                        lang = "python"
                        operation_type = "full_rewrite"
                        for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                            if m.group(1).lower() in ("name", "title"):
                                title = m.group(2)
                            elif m.group(1).lower() == "language":
                                lang = m.group(2)
                            elif m.group(1).lower() == "operation":
                                operation_type = m.group(2).lower()

                        is_patch = "<<<<<<< SEARCH" in body_content
                        is_append = operation_type == "append"
                        file_path = self._resolved_workspace / title
                        is_overwrite = file_path.exists()

                        if not is_append:
                            git_block = self._enforce_git_safety(title, is_overwrite)
                            if git_block:
                                action_reports.append(git_block)
                                continue

                        event_mode = kwargs.get("event_mode", EventMode.PROCESSING_TAG_MODE)
                        if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                            try:
                                if streaming_callback:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START, {
                                        "title": title, 
                                        "art_type": "code", 
                                        "language": lang, 
                                        "is_patch": is_patch, 
                                        "operation": "patch" if is_patch else ("append" if is_append else "full_rewrite"),
                                        "execution_phase": True
                                    })
                            except Exception:
                                pass

                        if is_patch:
                            if not file_path.exists():
                                action_reports.append(f"[SYSTEM ERROR] File '{title}' not found. Cannot apply patch.")
                                continue
                            original_content = file_path.read_text(encoding="utf-8", errors="ignore")
                            try:
                                patched_content = _ArtefactManager.apply_aider_patch(original_content, body_content)
                                file_path.write_text(patched_content, encoding="utf-8")
                                action_reports.append(f"✅ SEARCH/REPLACE applied successfully to {title}.")
                                if self._artefact_manager:
                                    self._artefact_manager.update(title=title, new_content=patched_content, language=lang, bump_version=True, active=True)
                            except Exception as patch_err:
                                if getattr(self, 'debug_mode', False):
                                    self._dump_error(
                                        error=patch_err,
                                        context_desc="Artifact Patch Error (Block 4)",
                                        round_count=round_count,
                                        extra_data={"title": title, "original_length": len(original_content), "patch_body": body_content[:500]}
                                    )
                                action_reports.append(f"❌ SEARCH/REPLACE FAILED for {title}. Error: {patch_err}")
                        elif is_append:
                            if not file_path.exists():
                                action_reports.append(f"[SYSTEM ERROR] File '{title}' not found. Cannot append.")
                                continue
                            if not body_content.strip():
                                action_reports.append(f"❌ APPEND BLOCKED for {title}. Empty body.")
                                continue
                            original_content = file_path.read_text(encoding="utf-8", errors="ignore")
                            sep = "" if original_content.endswith("\n") else "\n"
                            new_content = original_content + sep + body_content.strip() + "\n"
                            file_path.write_text(new_content, encoding="utf-8")
                            action_reports.append(f"✅ Content appended successfully to {title}.")
                            if self._artefact_manager:
                                self._artefact_manager.update(title=title, new_content=new_content, language=lang, bump_version=True, active=True)
                        else:
                            if not body_content.strip():
                                action_reports.append(f"❌ FILE WRITE BLOCKED for {title}. Empty artifact body.")
                                continue
                            if self._artefact_manager:
                                self._artefact_manager.add(title=title, artefact_type="code", content=body_content, language=lang, active=True)
                            file_path = self._resolved_workspace / title
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            file_path.write_text(body_content, encoding="utf-8")
                            action_reports.append(f"✅ File {title} created/updated successfully.")

                            _BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".zip", ".tar", ".gz", ".pdf", ".docx", ".pptx", ".mp3", ".wav", ".mp4"}
                            file_ext = file_path.suffix.lower()
                            if file_ext not in _BINARY_EXTS and len(body_content) < 50000:
                                try:
                                    self._execute_context_visibility("unlock_file", title)
                                    action_reports.append(f"📂 Auto-loaded '{title}' into context [C].")
                                except Exception:
                                    pass
                            elif file_ext in _BINARY_EXTS:
                                action_reports.append(f"🚫 Skipped auto-loading binary file '{title}' from context.")

                        if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                            try:
                                streaming_callback("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END, {"title": title, "art_type": "code", "success": True, "stream_complete": True})
                            except Exception:
                                pass
                    except Exception as e:
                        action_reports.append(f"[SYSTEM ERROR] Failed to process stranded artifact tag: {e}")

            files_after = self._take_workspace_snapshot()
            changes = self._sync_workspace(files_before, files_after)
            if changes:
                workspace_changes.extend(changes)

            ss.completed_actions = []

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

Agent = LollmsPersonality

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
