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
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ascii_colors import ASCIIColors, trace_exception

from .skills_manager import SkillsManager
from .handbag import Handbag
from .lollms_agent_state import _AgentStreamState, _sanitize_tool_result, _ToolsManager

from lollms_client.lollms_chat_core import (
    calculate_dynamic_tool_char_limit as _calculate_dynamic_tool_char_limit,
    repair_llm_json as _repair_llm_tool_json,
    is_large_base64 as _is_large_base64,
    sanitize_tool_result as _core_sanitize_tool_result,
    detect_structural_symbols as _detect_structural_symbols,
    extract_artefact_meta as _extract_artefact_meta,
    build_progressive_continuation_prompt as _build_progressive_continuation_prompt,
    inject_tool_images_for_vlm as _inject_tool_images_for_vlm_core,
    dump_error as _core_dump_error,
    take_workspace_snapshot as _core_take_workspace_snapshot,
    sync_workspace_diff as _core_sync_workspace_diff,
    execute_tool_call as _core_execute_tool_call,
    execute_context_visibility_operation as _core_execute_context_visibility,
    is_connection_or_server_error as _core_is_connection_or_server_error,
    run_fast_context_compaction as _core_run_fast_context_compaction,
)

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


from lollms_client.lollms_types import MSG_TYPE, EventMode, normalize_event_mode

from lollms_client.lollms_memory import FailureMemory
from lollms_client.lollms_artefact import ArtefactVisibility, ArtefactManager
from lollms_client.lollms_artefact.lollms_artefact import ArtefactManager as _ArtefactManager
from lollms_client.lollms_history import HistoryManager


_TEXT_RAG_EXTS = {
    ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml", ".html",
    ".py", ".js", ".ts", ".rs", ".go", ".rb", ".php", ".java", ".kt",
    ".swift", ".c", ".cpp", ".h", ".hpp", ".sql", ".sh", ".bash",
    ".ps1", ".bat", ".toml", ".ini", ".cfg", ".log", ".rdf", ".ttl",
}

_SYNTHETIC_RESPONSE_PREFIXES = (
    "[Task terminated:",
    "[Terminated:",
    "[Empty response:",
    "[Generation error:",
    "[Context Window Exhausted:",
)

_INTENT_ANNOUNCEMENT_RE = re.compile(
    r'(?im)(?:'
    r'^\s*(?:'
    r'i\s+will\b'
    r'|i\s+am\s+going\s+to\b'
    r'|i\'?m\s+going\s+to\b'
    r'|i\'?ll\b'
    r'|let\s+me\b'
    r'|let\'?s\b'
    r'|allow\s+me\b'
    r'|first[,.]?\s+(?:i\s+will|let\s+me|allow\s+me|i\'?ll)\b'
    r'|now\s+(?:i\s+will|let\s+me|allow\s+me|i\'?ll)\b'
    r'|next[,.]?\s+(?:i\s+will|let\s+me|allow\s+me|i\'?ll)\b'
    r'|je\s+vais\b'
    r'|permettez[- ]moi\b'
    r'|laissez[- ]moi\b'
    r'|laisse[- ]moi\b'
    r'|je\s+commence\b'
    r')'
    r'|\b(?:i\'?ll|i\s+will|let\s+me|let\'?s)\s+(?:first\s+)?(?:copy|create|check|run|write|build|execute|delete|modify|update|search|read|inspect|list|look|verify|test|fix|find)\b'
    r')'
)


def _is_synthetic_agent_response(text: str) -> bool:
    stripped = (text or "").strip()
    return any(stripped.startswith(prefix) for prefix in _SYNTHETIC_RESPONSE_PREFIXES)

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
        # Both scratchpad and active memories are already curated into stable_system_prompt.
        # Leaving them empty here prevents HistoryManager.export() from duplicating the scratchpad
        # inside the user's message and re-injecting memories into the system prompt.
        self.scratchpad = ""
        self.memory_manager = None
        self.pruning_summary = None
        self.pruning_point_id = None
        self.artefacts = getattr(personality, '_artefact_manager', None) or _NullArtefactManager()
        self.workspace_data_path = str(personality._resolved_workspace) if personality._resolved_workspace else "."

    @property
    def _system_prompt(self) -> str:
        return self._system_prompt_ref

    def get_full_data_zone(self) -> str:
        return ""

    def get_discussion_images(self) -> list:
        return []

    def _apply_three_view_protocol(self, msg, raw_content: str, distance_from_end: int = 0) -> str:
        from lollms_client.lollms_discussion._context_sanitizer import (
            scrub_processing_and_status_blocks,
        )

        if getattr(msg, "sender_type", "") != "assistant":
            return raw_content

        return scrub_processing_and_status_blocks(raw_content)

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
_MAX_DIR_ITEMS = 12


def _format_compact_size(size_bytes: int) -> str:
    """Formats bytes into compact 1-2 token strings (e.g. 11.7 MB, 124 KB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _cluster_similar_files(files: List[Path]) -> Tuple[List[str], int, int]:
    """
    Groups sequences of similarly named/numbered files (e.g. img_dalle__1.png..img_dalle__16.png)
    to save tokens while preserving structural inventory.
    """
    from collections import defaultdict
    clusters = defaultdict(list)
    standalone = []

    for f in files:
        m = re.match(r'^(.*?)(\d+)(\.[a-zA-Z0-9]+)$', f.name)
        if m:
            prefix, _, ext = m.groups()
            clusters[(prefix, ext)].append(f)
        else:
            standalone.append(f)

    lines = []
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    for (prefix, ext), cl_files in sorted(clusters.items()):
        if len(cl_files) >= 4:
            cl_size = sum(f.stat().st_size for f in cl_files)
            nums = []
            for cf in cl_files:
                m = re.search(r'\d+', cf.name)
                if m:
                    nums.append(int(m.group(0)))
            if nums:
                min_n, max_n = min(nums), max(nums)
                lines.append(f"{prefix}{min_n}{ext} .. {prefix}{max_n}{ext} ({len(cl_files)} {ext.lstrip('.').upper()} files, {_format_compact_size(cl_size)})")
            else:
                lines.append(f"{prefix}*{ext} ({len(cl_files)} files, {_format_compact_size(cl_size)})")
        else:
            standalone.extend(cl_files)

    for sf in standalone:
        s_sz = sf.stat().st_size if sf.is_file() else 0
        lines.append(f"{sf.name} ({_format_compact_size(s_sz)})")

    return lines, len(files), total_size


def _build_workspace_tree_r(
    directory: Path,
    workspace_root: Path,
    current_depth: int,
    collapsed_set: set,
    max_depth: int,
    max_items: int
) -> List[str]:
    if current_depth >= max_depth:
        return []

    lines = []
    try:
        raw_items = [p for p in directory.iterdir() if p.name not in _IGNORED_WS_DIRS and not p.name.startswith(".")]
    except Exception:
        return []

    subdirs = sorted([p for p in raw_items if p.is_dir()], key=lambda p: p.name.lower())
    files = sorted([p for p in raw_items if p.is_file() and p.suffix.lower() not in _IGNORED_WS_EXTS], key=lambda p: p.name.lower())

    indent = "  " * current_depth

    # 1. Render Subdirectories
    for d in subdirs:
        rel_dir = str(d.relative_to(workspace_root)).replace("\\", "/")
        try:
            d_files = [p for p in d.rglob("*") if p.is_file() and p.name not in _IGNORED_WS_DIRS and not p.name.startswith(".")]
            d_count = len(d_files)
            d_size = sum(p.stat().st_size for p in d_files)
            meta_str = f" ({d_count} items, {_format_compact_size(d_size)})" if d_count else ""
        except Exception:
            meta_str = ""

        if rel_dir in collapsed_set:
            lines.append(f"{indent}[📁 COLLAPSED] {d.name}/{meta_str}")
        elif current_depth + 1 >= max_depth:
            lines.append(f"{indent}[📁 DEEP] {d.name}/{meta_str}")
        else:
            lines.append(f"{indent}[📁] {d.name}/{meta_str}")
            lines.extend(_build_workspace_tree_r(d, workspace_root, current_depth + 1, collapsed_set, max_depth, max_items))

    # 2. Render Clustered Files
    if files:
        file_lines, total_f_cnt, total_f_sz = _cluster_similar_files(files)
        displayed_lines = file_lines[:max_items]

        for fl in displayed_lines:
            lines.append(f"{indent}├── {fl}")

        if len(file_lines) > max_items:
            overflow = len(file_lines) - max_items
            lines.append(f"{indent}└── ... (+{overflow} more files in this directory. Use tool_list_files to view)")

    return lines


def _build_workspace_context(workspace_path: Path, max_file_size: int = 12000, max_total_chars: int = 30000, collapsed_folders: Optional[set] = None) -> str:
    if not workspace_path or not workspace_path.exists():
        return ""

    collapsed_set = collapsed_folders or set()

    try:
        all_root_items = [p for p in workspace_path.iterdir() if p.name not in _IGNORED_WS_DIRS and not p.name.startswith(".")]
        total_items_count = len(all_root_items)
    except Exception:
        total_items_count = 0

    tree_entries = _build_workspace_tree_r(
        directory=workspace_path,
        workspace_root=workspace_path,
        current_depth=0,
        collapsed_set=collapsed_set,
        max_depth=_MAX_TREE_DEPTH,
        max_items=_MAX_DIR_ITEMS
    )

    if not tree_entries:
        return "=== WORKSPACE TREE ===\n(Workspace is empty)\n=== END WORKSPACE TREE ==="

    content_str = "\n".join(tree_entries)
    # Wrap inside preformatted ```text to guarantee literal line returns and prevent markdown horizontal collapsing
    return (
        f"=== WORKSPACE TREE ===\n"
        f"```text\n"
        f"Root: ./ ({total_items_count} items total)\n"
        f"{content_str}\n"
        f"```\n"
        f"=== END WORKSPACE TREE ==="
    )




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

    non_sys_start = -1
    for i, msg in enumerate(normalized):
        if msg.get("role") != "system":
            non_sys_start = i
            break
    if non_sys_start != -1 and non_sys_start < len(normalized):
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

    # Desktop UI automation (computer use)
    allow_computer_use: bool = False
    enable_computer_use: bool = False

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

    def __post_init__(self):
        if self.allow_computer_use or self.enable_computer_use:
            self.allow_computer_use = True
            self.enable_computer_use = True

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
            "allow_computer_use": self.allow_computer_use,
            "enable_computer_use": self.enable_computer_use,
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
    Enforces recursion depth, per-turn spawn count limits, and cooperative cancellation.
    """

    def __init__(self, parent_agent: 'Agent', max_depth: int = 3, max_per_turn: int = 5):
        self.parent = parent_agent
        self.max_depth = max_depth
        self.max_per_turn = max_per_turn
        self._current_depth = 0
        self._spawned_this_turn = 0
        self.active_child_agent: Optional[LollmsPersonality] = None

    def reset_turn(self):
        self._spawned_this_turn = 0
        self.active_child_agent = None

    def set_depth(self, depth: int):
        self._current_depth = depth

    def cancel_active_child(self):
        """Immediately terminates any active running sub-agent."""
        if self.active_child_agent is not None:
            try:
                self.active_child_agent.cancel_generation()
            except Exception:
                pass

    def can_spawn(self) -> bool:
        return (
            not self.parent.is_generation_cancelled() and
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
        effort: Optional[str] = None,
        dynamic_effort: bool = False,
        **kwargs,
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
            effort: Reasoning effort tier for the child ('none', 'low', 'medium', 'high').
            dynamic_effort: Whether child can adjust its reasoning effort dynamically.
        """
        if not self.can_spawn():
            return {
                "success": False,
                "error": f"Sub-agent spawn limit reached (depth: {self._current_depth}/{self.max_depth}, spawned: {self._spawned_this_turn}/{self.max_per_turn})."
            }

        self._spawned_this_turn += 1

        try:
            if self.parent.is_generation_cancelled():
                return {"success": False, "error": "Operation cancelled before sub-agent execution."}

            child_caps = CapabilityFlags(
                enable_code_execution=self.parent.capabilities.enable_code_execution if self.parent.capabilities else True,
                enable_image_generation=False,
                enable_image_editing=False,
                enable_sub_agents=False,  # Prevent infinite recursion
                enable_model_switching=False,
                enable_skill_loading=self.parent.capabilities.enable_skill_loading if self.parent.capabilities else True,
                enable_skill_creation=False,
                enable_workspace_tools=True,
                skills_mode="loadable",
                max_sub_agent_depth=0,
            )

            # Authoritative Worker Sub-Agent Doctrine
            base_worker_prompt = (
                "=== WORKER SUB-AGENT OPERATING DOCTRINE (STRICT & MANDATORY) ===\n"
                "1. YOU ARE AN AUTONOMOUS SPECIALIST SUB-AGENT SPAWNED BY THE PRIMARY ORCHESTRATOR.\n"
                "2. NO HUMAN IN THE LOOP: You are running headlessly inside an isolated sub-task. You are strictly FORBIDDEN from asking the user questions, requesting human confirmation, or pausing for feedback. Make sound engineering assumptions and proceed!\n"
                "3. AUTONOMOUS ACTION MANDATE: Stating intent does nothing. You MUST immediately emit the functional XML tags (`<tool>`, `<artifact>`, `<unlock_file>`) in the same turn to execute code, read files, or write artifacts.\n"
                "4. FULL COMPLETION IN SILENCE: Complete all steps required for your assigned task thoroughly. Do not stop midway.\n"
                "5. REPORTING CONTRACT: When your task is finished, summarize all your actions, code written, and test results inside `<report>...</report>` and conclude with `<done/>` on a new line.\n"
                "=== END WORKER SUB-AGENT DOCTRINE ==="
            )

            conditioned_prompt = (
                f"{personality_conditioning.strip()}\n\n{base_worker_prompt}"
                if personality_conditioning
                else base_worker_prompt
            )

            # ── 🔄 DYNAMIC SUBTASK MODEL SELECTION & RESTORATION ──
            client = self.parent.lollms_client
            orig_model_alias = getattr(client, "_active_llm_alias", None)
            orig_model_name = getattr(getattr(client, "llm", None), "model_name", None)

            target_model_alias = (model_name or "").strip()
            model_switched = False

            if target_model_alias and client:
                try:
                    if hasattr(client, "llm_model_profiles_registry") and target_model_alias in client.llm_model_profiles_registry:
                        model_switched = client.switch_model(target_model_alias)
                        if model_switched:
                            ASCIIColors.success(f"[SubAgentSpawner] Subtask assigned model profile '{target_model_alias}'.")
                    elif hasattr(client, "switch_active_model"):
                        model_switched = client.switch_active_model(target_model_alias)
                        if model_switched:
                            ASCIIColors.success(f"[SubAgentSpawner] Subtask active model switched to '{target_model_alias}'.")
                except Exception as switch_err:
                    ASCIIColors.warning(f"[SubAgentSpawner] Subtask model switch failed ({switch_err}). Using parent model.")

            child_model_label = target_model_alias if model_switched else (orig_model_alias or orig_model_name or "parent model")

            try:
                child_agent = LollmsPersonality(
                    name=f"SubAgent_{self._spawned_this_turn}",
                    author="lollms_personality",
                    category="sub_agent",
                    description="A focused sub-agent spawned for a specific task.",
                    system_prompt=conditioned_prompt,
                    role=AgentRole.IMPLEMENTER,
                    workspace_path=self.parent.get_workspace_path(),
                    capabilities=child_caps,
                    skills_manager=self.parent.skills_manager,
                    model_params=self.parent.model_params,
                    max_tokens_per_turn=self.parent.max_tokens_per_turn,
                    memory_manager=None,
                    lollms_client=client,
                    _parent_depth=self._current_depth + 1,
                )

                # Grant workspace autonomy to child agent
                object.__setattr__(child_agent, "_git_autonomy_granted", True)

                self.active_child_agent = child_agent

                parent_cb = getattr(self.parent, '_active_streaming_callback', None)
                spawn_start_time = time.time()
                if parent_cb:
                    try:
                        parent_cb(
                            f"🤖 Spawning sub-agent '{child_agent.name}' (Model: {child_model_label}) for task:\n{instruction[:250]}...",
                            getattr(MSG_TYPE, "MSG_TYPE_WORKER_SPAWN_START", MSG_TYPE.MSG_TYPE_INFO),
                            {
                                "agent_name": child_agent.name,
                                "task": instruction,
                                "worker_index": self._spawned_this_turn,
                                "depth": self._current_depth + 1,
                                "max_depth": self.max_depth,
                                "max_steps": max_steps,
                                "model_name": child_model_label,
                                "effort": effort or "default",
                                "dynamic_effort": dynamic_effort,
                                "personality_conditioning": personality_conditioning or "Autonomous Worker Specialist",
                            }
                        )
                    except Exception:
                        pass

                def child_stream_relay(chunk: str, msg_type=None, meta=None) -> bool:
                    if self.parent.is_generation_cancelled() or child_agent.is_generation_cancelled():
                        child_agent.cancel_generation()
                        return False
                    if parent_cb is None:
                        return True
                    try:
                        m = dict(meta or {})
                        m["sub_agent"] = child_agent.name
                        m["model_name"] = child_model_label
                        # Relay tool, artifact, and info events live to the UI
                        if msg_type in (
                            MSG_TYPE.MSG_TYPE_TOOL_START, MSG_TYPE.MSG_TYPE_TOOL_END,
                            MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START, MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END,
                            MSG_TYPE.MSG_TYPE_INFO, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK
                        ):
                            return parent_cb(chunk, msg_type, m)
                        elif msg_type == MSG_TYPE.MSG_TYPE_CHUNK and not m.get("live_tool_chunk") and not m.get("was_processed"):
                            return parent_cb(chunk, MSG_TYPE.MSG_TYPE_CHUNK, m)
                    except Exception:
                        return True
                    return True

                task_wrapped_prompt = (
                    f"[ASSIGNED SUB-AGENT TASK]\n{instruction}\n\n"
                    "[EXECUTION DIRECTIVE]\n"
                    "Execute the task autonomously using your available tools and artifact tags.\n"
                    "Do NOT ask questions or await human approval. When complete, provide your summary inside `<report>...</report>` and end with `<done/>`."
                )

                # Execute child chat with live streaming relay
                result = child_agent.chat(
                    prompt=task_wrapped_prompt,
                    streaming_callback=child_stream_relay,
                    max_reasoning_steps=max_steps,
                    temperature=temperature,
                    use_internal_history=False,
                    reasoning_effort=effort,
                    dynamic_effort=dynamic_effort,
                    enable_shell=getattr(self.parent.capabilities, "enable_code_execution", True),
                    enable_workspace_tools=True,
                    enable_python_exec=True,
                )
            finally:
                self.active_child_agent = None
                # Restore parent model
                if model_switched and client:
                    try:
                        if orig_model_alias and hasattr(client, "switch_model"):
                            client.switch_model(orig_model_alias)
                        elif orig_model_name and hasattr(client, "switch_active_model"):
                            client.switch_active_model(orig_model_name)
                        ASCIIColors.info(f"[SubAgentSpawner] Restored primary model to '{orig_model_alias or orig_model_name}'.")
                    except Exception:
                        pass

            spawn_elapsed = time.time() - spawn_start_time
            if parent_cb:
                try:
                    parent_cb(
                        f"✅ Sub-agent '{child_agent.name}' completed in {result.get('rounds', 0)} round(s).",
                        getattr(MSG_TYPE, "MSG_TYPE_WORKER_SPAWN_END", MSG_TYPE.MSG_TYPE_INFO),
                        {
                            "agent_name": child_agent.name,
                            "task": instruction,
                            "rounds": result.get("rounds", 0),
                            "tools_count": len(result.get("tool_calls", [])),
                            "success": not result.get("was_cancelled", False),
                            "report_digest": result.get("response", ""),
                            "elapsed_seconds": round(spawn_elapsed, 1),
                            "worker_index": self._spawned_this_turn,
                        }
                    )
                except Exception:
                    pass

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
        if tti is None:
            tti_registry = getattr(client, 'tti_model_profiles_registry', None)
            if tti_registry:
                tti = True
        if tti is not None:
            if caps.enable_image_generation:
                tools["tool_generate_image"] = BindingToolsBuilder._make_tti_generate_tool(client, workspace_path)
            if caps.enable_image_editing:
                tools["tool_edit_image"] = BindingToolsBuilder._make_tti_edit_tool(client, workspace_path)

        # TTS (Text-to-Speech)
        tts = getattr(client, 'tts', None)
        if tts is not None and caps.enable_tts:
            tools["tool_text_to_speech"] = BindingToolsBuilder._make_tts_tool(tts, workspace_path)

        # STT (Speech-to-Text)
        stt = getattr(client, 'stt', None)
        if stt is not None and caps.enable_stt:
            tools["tool_speech_to_text"] = BindingToolsBuilder._make_stt_tool(stt, workspace_path)

        # TTM (Text-to-Music & Songs)
        ttm = getattr(client, 'ttm', None)
        if ttm is not None and caps.enable_ttm:
            tools["tool_generate_music"] = BindingToolsBuilder._make_ttm_tool(ttm, workspace_path)
            tools["tool_generate_song"] = BindingToolsBuilder._make_song_tool(ttm, workspace_path)

        # TTV (Text-to-Video)
        ttv = getattr(client, 'ttv', None)
        if ttv is not None and caps.enable_ttv:
            tools["tool_generate_video"] = BindingToolsBuilder._make_ttv_tool(ttv, workspace_path)

        # CONNECTION (Communication channels: Discord, Telegram, Slack, Webhook, etc.)
        connection_registry = getattr(client, 'connection_model_profiles_registry', None)
        has_connections = bool(connection_registry) and not hasattr(connection_registry, "_mock_return_value")
        conn_binding = getattr(client, 'connection', None)
        if has_connections or (conn_binding is not None and not hasattr(conn_binding, "_mock_return_value")):
            tools["tool_send_connection"] = BindingToolsBuilder._make_connection_tool(client)

        # RAG (Knowledge base & semantic vector/graph store)
        rag_registry = getattr(client, 'rag_model_profiles_registry', None)
        has_rag = bool(rag_registry) and not hasattr(rag_registry, "_mock_return_value")
        rag_binding = getattr(client, 'rag', None)
        if has_rag or (rag_binding is not None and not hasattr(rag_binding, "_mock_return_value")):
            tools.update(BindingToolsBuilder._make_rag_binding_tools(client, workspace_path))

        return tools

    @staticmethod
    def _make_rag_binding_tools(client, workspace_path: Optional[Path]) -> Dict[str, Any]:
        """Exposes native RAG data store operations as callable agent tools."""
        tools = {}

        def tool_query_rag(query: str, store_alias: str = "", top_k: int = 5, hybrid: bool = True) -> dict:
            """
            Query the active RAG knowledge base for semantically relevant document excerpts,
            facts, or technical guidelines.

            Args:
                query (str): The search keywords or question.
                store_alias (str, optional): Target knowledge store profile alias. Uses default store if empty.
                top_k (int, optional): Maximum number of matching excerpts to return. Defaults to 5.
                hybrid (bool, optional): If True (default), fuses dense semantic vector search with sparse BM25.
            """
            try:
                results = client.query_rag(query, top_k=top_k, store_alias=store_alias or None, hybrid=hybrid)
                if not results:
                    return {"success": True, "output": f"No matching documents found in RAG store for '{query}'."}

                output_parts = [f"Found {len(results)} relevant excerpt(s) in RAG store:"]
                for idx, r in enumerate(results, 1):
                    src = r.get("source") or r.get("title") or "Document"
                    score = r.get("relevance_percent", r.get("score", ""))
                    score_str = f" ({score:.1f}% relevance)" if isinstance(score, (int, float)) else ""
                    content = r.get("content", "").strip()
                    output_parts.append(f"[{idx}] {src}{score_str}:\n{content}\n")

                return {
                    "success": True,
                    "count": len(results),
                    "output": "\n".join(output_parts)
                }
            except Exception as e:
                return {"success": False, "error": f"RAG query failed: {e}"}

        def tool_sparql_query(sparql_query: str, store_alias: str = "") -> dict:
            """
            Execute a W3C SPARQL 1.1 query (SELECT, ASK, CONSTRUCT) against the RAG knowledge graph.

            Args:
                sparql_query (str): The W3C SPARQL 1.1 query string.
                store_alias (str, optional): Target knowledge store profile alias.
            """
            try:
                res = client.query_sparql(sparql_query, store_alias=store_alias or None)
                if isinstance(res, dict) and "results" in res and "bindings" in res["results"]:
                    bindings = res["results"]["bindings"]
                    if not bindings:
                        return {"success": True, "output": "SPARQL query executed successfully (0 matching bindings)."}
                    lines = [f"SPARQL Results ({len(bindings)} binding(s)):"]
                    for b in bindings[:25]:
                        row_items = [f"{var}: {info.get('value', '')}" for var, info in b.items()]
                        lines.append("  • " + " | ".join(row_items))
                    if len(bindings) > 25:
                        lines.append(f"  ... (+{len(bindings) - 25} more bindings)")
                    return {"success": True, "output": "\n".join(lines)}
                elif isinstance(res, dict) and "boolean" in res:
                    return {"success": True, "output": f"SPARQL ASK Result: {res['boolean']}"}
                else:
                    import json
                    return {"success": True, "output": json.dumps(res, indent=2, default=str)}
            except Exception as e:
                return {"success": False, "error": f"SPARQL execution failed: {e}"}

        def tool_add_document_to_rag(file_name: str, store_alias: str = "") -> dict:
            """
            Ingests a file from the workspace into the RAG knowledge base.
            Supports .pdf, .docx, .xlsx, .csv, .md, .txt, and code files.

            Args:
                file_name (str): Path or filename of the document in the workspace.
                store_alias (str, optional): Target knowledge store profile alias.
            """
            try:
                p = Path(file_name)
                if not p.exists() and workspace_path:
                    p = workspace_path / file_name

                if not p.exists():
                    return {"success": False, "error": f"File '{file_name}' not found in workspace."}

                ok = client.add_document_to_rag(p, store_alias=store_alias or None)
                if ok:
                    return {"success": True, "output": f"Document '{p.name}' successfully indexed into RAG store."}
                return {"success": False, "error": f"Failed to ingest document '{p.name}'."}
            except Exception as e:
                return {"success": False, "error": f"RAG ingestion failed: {e}"}

        def tool_get_rag_info(store_alias: str = "") -> dict:
            """
            Inspects diagnostic information about the active RAG store (total documents, chunks, vectorizer, and graph stats).

            Args:
                store_alias (str, optional): Target knowledge store profile alias.
            """
            try:
                info_data = client.get_rag_info(store_alias=store_alias or None)
                import json
                return {"success": True, "output": json.dumps(info_data, indent=2, default=str)}
            except Exception as e:
                return {"success": False, "error": f"Failed to get RAG info: {e}"}

        tools["tool_query_rag"] = {
            "name": "tool_query_rag",
            "description": "Query the RAG knowledge store for semantically relevant document excerpts and facts.",
            "parameters": [
                {"name": "query", "type": "str", "description": "The search query or keywords."},
                {"name": "store_alias", "type": "str", "description": "Target store alias (default = active).", "optional": True},
                {"name": "top_k", "type": "int", "description": "Maximum excerpts to return (default 5).", "optional": True},
                {"name": "hybrid", "type": "bool", "description": "Fuse dense vectors with sparse BM25 (default True).", "optional": True},
            ],
            "callable": tool_query_rag,
        }

        tools["tool_sparql_query"] = {
            "name": "tool_sparql_query",
            "description": "Execute a W3C SPARQL 1.1 query (SELECT, ASK, CONSTRUCT) against the RAG knowledge graph.",
            "parameters": [
                {"name": "sparql_query", "type": "str", "description": "W3C SPARQL 1.1 query string."},
                {"name": "store_alias", "type": "str", "description": "Target store alias (default = active).", "optional": True},
            ],
            "callable": tool_sparql_query,
        }

        tools["tool_add_document_to_rag"] = {
            "name": "tool_add_document_to_rag",
            "description": "Ingest and index a document file from the workspace into the RAG knowledge store.",
            "parameters": [
                {"name": "file_name", "type": "str", "description": "Path of the file to index."},
                {"name": "store_alias", "type": "str", "description": "Target store alias (default = active).", "optional": True},
            ],
            "callable": tool_add_document_to_rag,
        }

        tools["tool_get_rag_info"] = {
            "name": "tool_get_rag_info",
            "description": "Inspect diagnostic metadata and statistics of the RAG knowledge store.",
            "parameters": [
                {"name": "store_alias", "type": "str", "description": "Target store alias (default = active).", "optional": True},
            ],
            "callable": tool_get_rag_info,
        }

        return tools

    @staticmethod
    def _make_tti_generate_tool(client, workspace_path: Optional[Path]) -> Dict[str, Any]:
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
                tti_binding = getattr(client, 'tti', None)
                if tti_binding is None:
                    tti_registry = getattr(client, 'tti_model_profiles_registry', None)
                    if tti_registry:
                        default_alias = next((a for a, p in tti_registry.items() if p.is_default), None)
                        if default_alias and hasattr(client, 'switch_tti'):
                            client.switch_tti(default_alias)
                        tti_binding = getattr(client, 'tti', None)
                if tti_binding is None:
                    return {"success": False, "error": "No TTI binding available. Configure tti_binding_name or tti_model_profiles."}
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
    def _make_tti_edit_tool(client, workspace_path: Optional[Path]) -> Dict[str, Any]:
        def tool_edit_image(prompt: str, image_file_name: str = "") -> dict:
            """
            Edit an existing image in the workspace using a text prompt.

            Args:
                prompt (str): Detailed English prompt describing the edits to apply.
                image_file_name (str): Filename of the image to edit (in the workspace).
            """
            try:
                tti_binding = getattr(client, 'tti', None)
                if tti_binding is None:
                    tti_registry = getattr(client, 'tti_model_profiles_registry', None)
                    if tti_registry:
                        default_alias = next((a for a, p in tti_registry.items() if p.is_default), None)
                        if default_alias and hasattr(client, 'switch_tti'):
                            client.switch_tti(default_alias)
                        tti_binding = getattr(client, 'tti', None)
                if tti_binding is None:
                    return {"success": False, "error": "No TTI binding available. Configure tti_binding_name or tti_model_profiles."}
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

                img_bytes = tti_binding.edit_image(images=source_b64, prompt=prompt)
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
    def _make_song_tool(ttm_binding, workspace_path: Optional[Path]) -> Dict[str, Any]:
        def tool_generate_song(prompt: str, lyrics: str = "", duration: int = 60, file_name: str = "") -> dict:
            """
            Generate a full song with vocals and music conditioned on lyrics and style descriptions.

            Args:
                prompt (str): Description of the musical style, mood, genre, tempo, instruments.
                lyrics (str, optional): The song lyrics, optionally formatted with tags like [Verse], [Chorus].
                duration (int, optional): Duration in seconds (default 60).
                file_name (str, optional): Output filename (without extension). Auto-generated if empty.
            """
            try:
                audio_bytes = ttm_binding.generate_song_from_lyrics(prompt=prompt, lyrics=lyrics, duration=duration)
                if not audio_bytes:
                    return {"success": False, "error": "TTM song generation returned no audio data."}

                fname = file_name or f"song_{uuid.uuid4().hex[:6]}"
                if not fname.endswith(".wav"):
                    fname += ".wav"

                save_path = Path(fname)
                if workspace_path:
                    save_path = workspace_path / fname
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(audio_bytes)

                return {
                    "success": True,
                    "output": f"Song generated and saved as '{fname}'.",
                    "audio_filename": fname,
                }
            except Exception as e:
                return {"success": False, "error": f"Song generation failed: {e}"}

        return {
            "name": "tool_generate_song",
            "description": "Generate a complete song with vocals and arrangement from lyrics and musical description using the TTM binding.",
            "parameters": [
                {"name": "prompt", "type": "str", "description": "Musical style, genre, tempo, and arrangement description."},
                {"name": "lyrics", "type": "str", "description": "Lyrics for the song with section tags like [Verse] and [Chorus].", "optional": True},
                {"name": "duration", "type": "int", "description": "Song duration in seconds (default 60).", "optional": True},
                {"name": "file_name", "type": "str", "description": "Output filename without extension.", "optional": True},
            ],
            "callable": tool_generate_song,
        }

    @staticmethod
    def _make_connection_tool(client) -> Dict[str, Any]:
        def tool_send_connection(content: str, channel_alias: str = "", sender_name: str = "") -> dict:
            """
            Send a message to a communication channel via an active connection binding.

            Args:
                content (str): The message text to send.
                channel_alias (str, optional): The connection profile alias to use
                    (e.g., "slack-alerts", "discord-general"). If empty, uses the active/default connection.
                sender_name (str, optional): Display name override.
            """
            try:
                conn_binding = getattr(client, 'connection', None)
                if conn_binding is None:
                    conn_registry = getattr(client, 'connection_model_profiles_registry', None)
                    if conn_registry:
                        target_alias = next(
                            (a for a, p in conn_registry.items() if p.is_default), None
                        )
                        if target_alias and hasattr(client, 'switch_connection'):
                            client.switch_connection(target_alias)
                            conn_binding = getattr(client, 'connection', None)

                if conn_binding is None:
                    return {
                        "success": False,
                        "error": "No connection binding available. Configure connection_binding_name or connection_model_profiles.",
                    }

                result = conn_binding.send_message(
                    content=content,
                    sender_name=sender_name or None,
                )

                if result.get("sent"):
                    return {
                        "success": True,
                        "output": f"Message sent successfully via {conn_binding.binding_name} to channel '{result.get('channel', 'unknown')}'.",
                        "channel": result.get("channel"),
                        "message_id": result.get("message_id"),
                        "prompt_injection": f"\n\n📤 Message sent to {result.get('channel', 'channel')} via {conn_binding.binding_name}.",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to send message: {result.get('error', 'Unknown error')}",
                    }

            except Exception as e:
                return {"success": False, "error": f"Connection send failed: {e}"}

        return {
            "name": "tool_send_connection",
            "description": (
                "Send a message to a communication channel (Discord, Telegram, Slack, webhook, etc.) "
                "via an active connection binding. Optionally specify a channel_alias to pick a different "
                "connection profile from the registered set."
            ),
            "parameters": [
                {"name": "content", "type": "str", "description": "The message text to send."},
                {"name": "channel_alias", "type": "str", "description": "Connection profile alias (default = active).", "optional": True},
                {"name": "sender_name", "type": "str", "description": "Display name override (optional).", "optional": True},
            ],
            "callable": tool_send_connection,
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
        disable_artefact_versioning: bool = True,
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

        # ── Initialize Core Attributes Early (Before Properties & Setters) ──
        object.__setattr__(self, '_lollms_client', resolved_client)
        self.disable_artefact_versioning = disable_artefact_versioning
        self.enable_artefact_system = enable_artefact_system

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
            self.skills_manager = SkillsManager(skills_dirs=skills_dirs, mode="loadable", max_visible_tokens=1200)
        else:
            self.skills_manager = None

        # Do not pre-bake dynamic skills context into the immutable base system_prompt.
        # Skills context is rendered dynamically per turn in _build_system_prompt().
        self._skills_context_injected = False

        self.memory_manager = memory_manager
        self._workspace_path: Optional[Path] = None
        self.workspace_path = Path(workspace_path) if workspace_path else None
        self.enable_git_management = enable_git_management
        self.coworkers: Dict[str, 'LollmsPersonality'] = {}

        self.max_tokens_per_turn = max_tokens_per_turn

        # Capabilities
        if CapabilityFlags is not None:
            self.capabilities = capabilities if capabilities is not None else CapabilityFlags()
        else:
            self.capabilities = None

        self._conversation: List[Dict[str, str]] = []
        self._failure_memory = FailureMemory() if FailureMemory else SimpleNamespace(failures=[], _signatures=set())
        self._pinned_lessons: str = ""

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
    def lollms_client(self) -> Optional[Any]:
        return getattr(self, '_lollms_client', None)

    @lollms_client.setter
    def lollms_client(self, value: Optional[Any]) -> None:
        object.__setattr__(self, '_lollms_client', value)
        proxy = getattr(self, '_artefact_proxy', None)
        if proxy is not None:
            setattr(proxy, 'lollmsClient', value)

    @property
    def lc(self) -> Optional[Any]:
        return self.lollms_client

    @lc.setter
    def lc(self, value: Optional[Any]) -> None:
        self.lollms_client = value

    def clear_conversation(self) -> None:
        """Clears the agent's internal multi-turn conversation memory and ephemeral session scratchpad."""
        self._conversation = []
        object.__setattr__(self, '_scratchpad_content', '')
        if getattr(self, '_scratchpad_path', None) and self._scratchpad_path.exists():
            try:
                self._scratchpad_path.write_text("# Scratchpad\n\n(Empty - session notes only)\n", encoding="utf-8")
            except Exception:
                pass

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
        """Loads the internal conversation history from a JSON file, validating message schemas."""
        if not history_file or not history_file.exists():
            self._conversation = []
            return
        try:
            import json as _json
            data = _json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                valid_msgs = []
                for item in data:
                    if isinstance(item, dict) and "role" in item and "content" in item:
                        role = str(item["role"])
                        content = str(item["content"])
                        if role == "assistant" and _is_synthetic_agent_response(content):
                            continue
                        valid_msgs.append({"role": role, "content": content})
                self._conversation = valid_msgs
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

        return self.chat(
            prompt=prompt,
            tools=tools,
            max_nb_rounds=max_tool_rounds,
            temperature=temperature if temperature is not None else 0.7,
            streaming_callback=streaming_callback,
            **kwargs
        )

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
        skills_mode = meta.get("skills_mode") or hb.manifest.get("skills_mode", "loadable")
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
                discovered_count = len(tool_binding.discovered_tools) if tool_binding else 0
                if discovered_count == 0:
                    ASCIIColors.error(
                        f"[Handbag] Tool binding created but discovered 0 tools from {len(hb.tool_files)} "
                        f"file(s): {[f.name for f in hb.tool_files]}. Check for syntax errors or missing "
                        f"'tool_' function definitions in each file."
                    )
                else:
                    ASCIIColors.success(
                        f"[Handbag] Loaded {discovered_count} tool(s) from handbag: "
                        f"{[t.get('name') for t in tool_binding.discovered_tools]}"
                    )
            except Exception as tool_load_err:
                ASCIIColors.error(f"[Handbag] FAILED to create tool binding from {len(hb.tool_files)} file(s): {tool_load_err}")
                trace_exception(tool_load_err)

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
        # 1. Query client RAG binding if available and no specific custom datasource was requested
        if self.lollms_client and getattr(self.lollms_client, "rag", None) and (not datasource_name or datasource_name == "rag_binding"):
            try:
                res = self.lollms_client.query_rag(query, store_alias=datasource_name if datasource_name != "rag_binding" else None, **kwargs)
                if res:
                    return {
                        "success": True,
                        "sources": res,
                        "count": len(res),
                        "query": query,
                        "datasource_name": getattr(self.lollms_client.rag, "store_name", "rag_binding")
                    }
            except Exception as ex:
                ASCIIColors.warning(f"[{self.name}] Client RAG query error: {ex}")

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
    def artefacts(self) -> Optional[ArtefactManager]:
        """Exposes the internal ArtefactManager instance."""
        return getattr(self, '_artefact_manager', None)

    @property
    def has_data(self) -> bool:
        return (
            bool(self.data_sources)
            or self._raw_data_source is not None
            or self.query_rag_callback is not None
            or bool(self.data_files)
            or (self.lollms_client is not None and (getattr(self.lollms_client, "rag", None) is not None or bool(getattr(self.lollms_client, "rag_model_profiles_registry", None))))
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
            if getattr(self, '_artefact_manager', None) is None:
                self._init_artefact_system()
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

    def _sync_artefact_index_with_disk(self) -> None:
        """
        No-op: Eager recursive scanning and importing of the entire workspace tree on startup
        is disabled to eliminate startup hangs on large codebases. Files are discovered via the workspace tree
        and indexed on-demand when unlocked, loaded, or modified.
        """
        return

    def get_workspace_path(self) -> Optional[str]:
        return str(self._resolved_workspace) if self._resolved_workspace else None

    def list_workspace_files(self) -> List[str]:
        if not self._resolved_workspace or not self._resolved_workspace.exists():
            return []
        result = []
        try:
            for root, dirs, files in os.walk(self._resolved_workspace):
                dirs[:] = [d for d in dirs if d not in _IGNORED_WS_DIRS and not d.startswith(".")]
                for fname in files:
                    if fname.startswith("."):
                        continue
                    p = Path(root) / fname
                    if p.suffix.lower() not in _IGNORED_WS_EXTS:
                        result.append(str(p.relative_to(self._resolved_workspace)))
        except Exception:
            pass
        return sorted(result)

    def _take_workspace_snapshot(self) -> Dict:
        if not self._resolved_workspace:
            return {}
        return _core_take_workspace_snapshot(self._resolved_workspace)

    def _sync_workspace(self, files_before: Dict, files_after: Dict) -> List[Dict[str, Any]]:
        return _core_sync_workspace_diff(files_before, files_after)

    def cancel_generation(self) -> bool:
        object.__setattr__(self, '_cancel_flag', True)
        if hasattr(self, '_sub_agent_spawner') and self._sub_agent_spawner:
            try:
                self._sub_agent_spawner.cancel_active_child()
            except Exception:
                pass
        if hasattr(self, 'lollms_client') and self.lollms_client:
            if hasattr(self.lollms_client, 'cancel'):
                try:
                    self.lollms_client.cancel()
                except Exception:
                    pass
            if hasattr(self.lollms_client, 'llm') and hasattr(self.lollms_client.llm, 'cancel'):
                try:
                    self.lollms_client.llm.cancel()
                except Exception:
                    pass
        return True

    def cancel(self) -> bool:
        return self.cancel_generation()

    def is_generation_cancelled(self) -> bool:
        return getattr(self, '_cancel_flag', False)

    def _reset_cancel_state(self):
        object.__setattr__(self, '_cancel_flag', False)

    @staticmethod
    def _build_progressive_continuation_prompt(stall_count: int, recent_tools: Optional[List[str]] = None) -> str:
        recent_ctx = f" Recent actions executed: {recent_tools}." if recent_tools else ""
        if stall_count <= 1:
            return (
                f"[SYSTEM DIRECTIVE: You wrote conversational text without executing an action tag or emitting `<done/>`.{recent_ctx}\n"
                "Conversational declarations and apologies DO NOT execute tools or create files.\n"
                "MANDATORY: Output the functional XML tag (`<tool>`, `<generate_image>`, `<artifact>`, `<unlock_file>`) as the FIRST token of your reply NOW.\n"
                "- If generating an image: `<generate_image>prompt</generate_image>` or `<tool>{\"name\": \"tool_generate_image\", \"parameters\": {\"prompt\": \"...\"}}</tool>`\n"
                "- If running tests/commands: `<tool>{\"name\": \"tool_execute_shell_command\", \"parameters\": {\"command\": \"...\"}}</tool>`\n"
                "- If modifying code: `<artifact name=\"file.py\">...</artifact>`\n"
                "DO NOT apologize. DO NOT write another introductory sentence. Output the XML tag NOW.]"
            )
        elif stall_count == 2:
            return (
                f"[SYSTEM: ACTION REQUIRED — You have produced conversational text without action tags or `<done/>` for 2 consecutive turns.{recent_ctx}\n"
                "You MUST output the functional tag (`<generate_image>`, `<tool>`, `<artifact>`) IMMEDIATELY as your first token, or output `<done/>` to terminate.\n"
                "Do NOT output conversational apologies or introductory preambles.]"
            )
        else:
            return (
                f"[SYSTEM: CRITICAL — You have stalled {stall_count} times without producing an action tag or `<done/>`.\n"
                "Emit `<done/>` on a new line NOW to terminate the turn.]"
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
        if not self._resolved_workspace:
            return ""

        collapsed = self._get_collapsed_folders_from_db()
        tree_block = _build_workspace_context(self._resolved_workspace, collapsed_folders=collapsed)

        loaded_parts = ["=== FULLY LOADED FILE CONTENTS [C] ==="]
        has_loaded = False

        if getattr(self, '_artefact_manager', None):
            try:
                all_arts = self._artefact_manager._get_all_raw()
                for art in all_arts:
                    title = art.get("title", "")
                    if title.endswith("::images"):
                        continue

                    vis = art.get("visibility")
                    if vis in (ArtefactVisibility.FULL, ArtefactVisibility.PINNED):
                        content = art.get("content", "")
                        phys_target = art.get("physical_path") or title
                        file_path = self._resolved_workspace / phys_target
                        if file_path.exists() and file_path.is_file():
                            ext = file_path.suffix.lower()
                            is_bin = ext in _BINARY_EXTS
                            if not is_bin:
                                try:
                                    with open(file_path, "rb") as f_bin:
                                        is_bin = b"\x00" in f_bin.read(4096)
                                except Exception:
                                    is_bin = True

                            if is_bin:
                                lam = getattr(self._artefact_manager, "_get_lam_content", lambda a: "")(art)
                                content = lam or f"[Non-textual file: {phys_target} ({file_path.stat().st_size:,} bytes). Raw binary content is withheld to protect the context window.]"
                            elif not content or art.get("content_source") == "disk":
                                try:
                                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                                    art["content"] = content
                                except Exception:
                                    pass

                        if content:
                            has_loaded = True
                            loaded_parts.append(f"--- File: {title} ---\n{content}\n--- End File: {title} ---")
            except Exception as e:
                ASCIIColors.warning(f"[{self.name}] Failed to extract loaded contents: {e}")

        loaded_block = "\n".join(loaded_parts) + "\n=== END FULLY LOADED FILE CONTENTS ===" if has_loaded else ""

        # ── 📚 SUB-WORKSPACE (REFERENCE & DOCUMENTATION) INGESTION ──
        sub_ws_block = ""
        try:
            from lollms_client.apps.lollms_code.sub_workspace import SubWorkspaceManager
            sub_ws = SubWorkspaceManager(self._resolved_workspace)
            if sub_ws.has_files():
                sub_ws_block = sub_ws.build_context_block(self.lollms_client)
        except Exception:
            pass

        parts = [tree_block]
        if loaded_block:
            parts.append(loaded_block)
        if sub_ws_block:
            parts.append(sub_ws_block)

        return "\n\n".join(parts)

    def _refresh_workspace_context_in_prompt(self, current_prompt: str, new_ws_block: str) -> str:
        ws_boundary = "=== WORKSPACE CONTEXT BOUNDARY ==="
        boundary_idx = current_prompt.find(ws_boundary)

        if boundary_idx == -1:
            return current_prompt + "\n" + new_ws_block.strip()

        base_prompt = current_prompt[:boundary_idx + len(ws_boundary)]
        return base_prompt + "\n" + new_ws_block.strip()

    def _calculate_context_fill(self, full_system_prompt: str, base_conversation: List[Dict], virtual_history: List, final_response: str = "") -> Dict[str, Any]:
        """Calculates current context window fill percentage using fast cached token lookups."""
        try:
            max_ctx = 0
            if self.lollms_client and hasattr(self.lollms_client, 'get_ctx_size'):
                max_ctx = self.lollms_client.get_ctx_size() or 0
            if max_ctx <= 0:
                max_ctx = 8192

            total_used = self._count_tokens_cached(full_system_prompt)
            for msg in base_conversation:
                total_used += self._count_tokens_cached(msg.get("content", ""))
            for vh in virtual_history:
                content = getattr(vh, "content", "")
                if content:
                    total_used += self._count_tokens_cached(content)
            if final_response:
                total_used += self._count_tokens_cached(final_response)

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
        Trivial interactions and failed turns are discarded.
        """
        if not self.lollms_client or not hasattr(self.memory_manager, 'add'):
            return

        # Do not run memory consolidation on failed or error responses
        if not ai_response or isinstance(ai_response, dict):
            return

        ai_response_str = str(ai_response)
        if "[Generation error:" in ai_response_str or "Server Connection Failure" in ai_response_str or "Connection error" in ai_response_str:
            return

        try:
            user_prompt_str = str(user_prompt) if not isinstance(user_prompt, str) else user_prompt
            clean_ai = re.sub(r'<[^>]+>', '', ai_response_str).strip()
            clean_user = user_prompt_str.strip()

            if not clean_user or not clean_ai or len(clean_user) < 10 or len(clean_ai) < 10:
                return

            consolidation_prompt = f"""Analyze the following interaction between a User and an AI Engineer.
Determine if a CRITICAL USER PREFERENCE, PERMANENT ARCHITECTURAL RULE, or IDENTITY FACT was established.
CRITICAL EXCLUSIONS:
- DO NOT save ephemeral task requests (e.g., 'User wants to add tool X', 'User asked to build Y').
- DO NOT save to-do lists, progress updates, greetings, or conversational filler.
- ONLY save permanent rules (e.g., 'User prefers 4 spaces indentation', 'Command alias c&p means commit and push').

User: "{clean_user}"
AI: "{clean_ai}"

If a permanent rule/fact was established, output EXACTLY a JSON object with:
{{"save_memory": true, "content": "The specific fact/rule (written as a passive fact, NOT an imperative task)", "tags": ["relevant", "tags"], "importance": 0.0-1.0}}
If the interaction is a task, greeting, or ephemeral request, output:
{{"save_memory": false}}

JSON:"""

            reflection = self.lollms_client.generate_text(
                prompt=consolidation_prompt,
                temperature=0.1,
                n_predict=256
            )

            if not isinstance(reflection, str):
                return

            import json as _json
            json_match = re.search(r'\{.*\}', reflection, re.DOTALL)
            if json_match:
                data = _json.loads(json_match.group(0))
                if data.get("save_memory"):
                    content_to_save = data.get("content", "").strip()
                    task_reject_regex = re.compile(r'^(?:user\s+wants\s+to|user\s+asked\s+to|organize|create|implement|build|fix|add)\b', re.IGNORECASE)
                    if not task_reject_regex.search(content_to_save):
                        self.memory_manager.add(
                            content=content_to_save,
                            importance=float(data.get("importance", 0.8)),
                            tags=data.get("tags", ["architectural", "fact"]),
                            level=2
                        )
                        ASCIIColors.success(f"[{self.name}] 💾 Consolidated high-density memory: {content_to_save[:50]}...")
                        cb = getattr(self, '_active_streaming_callback', None)
                        if cb:
                            try:
                                cb(
                                    f"💾 **Memory Consolidated**: {content_to_save}",
                                    MSG_TYPE.MSG_TYPE_INFO,
                                    {
                                        "type": "memory_consolidated",
                                        "content": content_to_save,
                                        "tags": data.get("tags", ["architectural", "fact"]),
                                        "importance": float(data.get("importance", 0.8))
                                    }
                                )
                            except Exception:
                                pass

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

            lock_tags = re.findall(r'<lock_file>(.*?)</lock_file>', cleanup_response, re.DOTALL | re.IGNORECASE)

            if lock_tags:
                locked_count = 0
                for body in lock_tags:
                    files_to_lock = [f.strip().replace("\\", "/") for f in re.split(r'[\n,;]+', body) if f.strip()]
                    for f_name in files_to_lock:
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

    def _count_tokens_cached(self, text: str) -> int:
        """Fast cached token counter: caches by string length and hash to eliminate redundant tokenizations."""
        if not text:
            return 0
        if not hasattr(self, "_token_cache"):
            object.__setattr__(self, "_token_cache", {})
        cache_key = f"{len(text)}:{hash(text)}"
        cached = self._token_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.lollms_client and hasattr(self.lollms_client, "count_tokens"):
            count = self.lollms_client.count_tokens(text) or 0
        else:
            count = len(text) // 4
        self._token_cache[cache_key] = count
        return count

    def _calculate_context_telemetry(self, stable_prompt: str, history: List[Dict], ws_ctx: str, virtual_history: List) -> Dict[str, int]:
        """Calculates token consumption per context segment using cached token counts."""
        telemetry = {
            "system_prompt": 0,
            "history": 0,
            "workspace_tree": 0,
            "loaded_contents": 0,
            "virtual_history": 0,
            "total": 0
        }
        if not self.lollms_client:
            return telemetry

        try:
            telemetry["system_prompt"] = self._count_tokens_cached(stable_prompt)

            for msg in history:
                telemetry["history"] += self._count_tokens_cached(msg.get("content", ""))

            if ws_ctx:
                if "## Fully Loaded File Contents [C]" in ws_ctx:
                    parts = ws_ctx.split("## Fully Loaded File Contents [C]", 1)
                    if len(parts) == 2:
                        telemetry["loaded_contents"] = self._count_tokens_cached(parts[1])
                        telemetry["workspace_tree"] = self._count_tokens_cached(parts[0])
                else:
                    telemetry["workspace_tree"] = self._count_tokens_cached(ws_ctx)

            for vh in virtual_history:
                content = getattr(vh, "content", "")
                if content:
                    telemetry["virtual_history"] += self._count_tokens_cached(content)

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

            return compacted_history
        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] History refactoring failed: {e}")
            return base_conversation

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

            conn = _sqlite3.connect(str(state_db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_states (
                    title TEXT PRIMARY KEY,
                    visibility TEXT NOT NULL,
                    hash TEXT,
                    mtime REAL,
                    size INTEGER
                )
            """)
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
                id=f"pers_{getattr(self, 'personality_id', 'unknown')[:8]}",
                workspace_path=str(ws_path),
                workspace_data_path=str(ws_path),
                artefacts_metadata_path=str(metadata_dir),
                lollmsClient=getattr(self, 'lollms_client', None), 
                metadata={},
                _is_db_backed=False,
                commit=lambda: None,
                disable_artefact_versioning=getattr(self, 'disable_artefact_versioning', True),
            )

            am = ArtefactManager(proxy)
            object.__setattr__(self, '_artefact_manager', am)
            object.__setattr__(self, '_artefact_proxy', proxy)
            object.__setattr__(self, '_discussion', proxy)
            object.__setattr__(self, '_state_db_path', state_db_path)
            object.__setattr__(self, '_last_ws_sync_time', 0.0)

            # Sync existing files from disk into the index
            self._sync_artefact_index_with_disk()

        except Exception as e:
            ASCIIColors.warning(f"[{self.name}] Failed to initialise artefact system: {e}")
            object.__setattr__(self, '_artefact_manager', None)
            object.__setattr__(self, '_artefact_proxy', None)
            
            
            
    def _dump_error(self, error: Exception, context_desc: str, round_count: int, extra_data: Optional[Dict[str, Any]] = None):
        """Writes a detailed error log to the debug dumps directory."""
        ws_path = getattr(self, '_resolved_workspace', None)
        if ws_path:
            debug_dir = ws_path / ".lollms_code"
            _core_dump_error(
                error=error,
                context_desc=context_desc,
                round_count=round_count,
                workspace_dir=debug_dir,
                extra_data=extra_data,
                debug_mode=getattr(self, 'debug_mode', False)
            )

    def _sanitize_history_for_context(self, text: str, round_index: int = 0, distance_from_end: int = 99) -> str:
        """
        Sanitizes assistant message content for LLM context export.
        Preserves real tags without synthetic bracketed placeholders to prevent mimicry loops.
        """
        from lollms_client.lollms_history import HistoryManager
        return HistoryManager._sanitize_for_context(text, distance_from_end)

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
                        if art is None or art.get("visibility") != ArtefactVisibility.FULL:
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

    def list_skills_structured(self, include_content: bool = False) -> List[Dict[str, Any]]:
        """
        Returns a structured list of all skills registered with this personality,
        annotating each skill with its source provenance (handbag, workspace, global, bundled)
        and indicating whether it originated from the active handbag.
        """
        if not self.skills_manager:
            return []

        handbag_dir = Path(self.handbag_path).resolve() if self.handbag_path else None
        ws_dir = Path(self.workspace_path).resolve() if self.workspace_path else None

        skills_list = []
        seen_keys = set()
        unique_skills = self.skills_manager.get_unique_skills() if hasattr(self.skills_manager, "get_unique_skills") else self.skills_manager.skills.values()
        for s in unique_skills:
            canonical_key = str(s.file_path.resolve()) if s.file_path else s.title.lower().strip()
            if canonical_key in seen_keys:
                continue
            seen_keys.add(canonical_key)

            fp = s.file_path.resolve() if s.file_path else None
            source = "custom"
            is_handbag = False

            if fp and handbag_dir:
                try:
                    fp.relative_to(handbag_dir)
                    source = "handbag"
                    is_handbag = True
                except ValueError:
                    pass

            if not is_handbag and fp and ws_dir:
                try:
                    fp.relative_to(ws_dir)
                    source = "workspace"
                except ValueError:
                    pass

            if not is_handbag and source == "custom" and fp:
                fp_str = str(fp).lower()
                if ".lollms_client" in fp_str or ".lollms_hub" in fp_str or ".lollms_code" in fp_str:
                    source = "global"
                elif "skills" in fp_str:
                    source = "bundled"

            entry = {
                "title": s.title,
                "description": s.description,
                "category": s.category,
                "tags": s.tags,
                "visibility": s.visibility,
                "modifiable": s.modifiable,
                "source": source,
                "is_handbag": is_handbag,
                "handbag_name": handbag_dir.name if is_handbag and handbag_dir else None,
                "file_path": str(s.file_path) if s.file_path else None,
                "content_preview": (s.content[:200] + "...") if len(s.content) > 200 else s.content,
            }
            if include_content:
                entry["content"] = s.content
            skills_list.append(entry)

        skills_list.sort(key=lambda s: (not s["is_handbag"], s["source"], s["title"].lower()))
        return skills_list

    def list_skills(self, include_content: bool = False) -> List[Dict[str, Any]]:
        """Alias for list_skills_structured()."""
        return self.list_skills_structured(include_content=include_content)

    def get_skills_structured(self, include_content: bool = False) -> List[Dict[str, Any]]:
        """Alias for list_skills_structured()."""
        return self.list_skills_structured(include_content=include_content)

    def list_tools_structured(self) -> List[Dict[str, Any]]:
        """
        Returns a structured, categorized list of all tools available to this personality,
        indicating tool provenance (handbag, lcp_default, multimodal binding, rag, sub_agent, etc.),
        descriptions, parameter schemas, and whether the tool originates from the active handbag.
        """
        active_tools = self._discover_tools(
            enable_data_tools=True,
            enable_workspace_tools=True,
            enable_shell=True,
            enable_python_exec=True,
            enable_web_tools=True,
            auto_load_document_editor=True,
            enable_computer_use=True,
        )

        handbag_tool_names = set()
        if self._tool_binding and hasattr(self._tool_binding, "discovered_tools"):
            for t in self._tool_binding.discovered_tools:
                handbag_tool_names.add(t.get("name"))

        structured_tools = []
        for name, spec in active_tools.items():
            desc = spec.get("description", "")
            params = spec.get("parameters", [])
            source_file = spec.get("_source_file") or spec.get("_python_file_path")

            if not source_file and "callable" in spec and callable(spec["callable"]):
                try:
                    import inspect
                    source_file = inspect.getfile(spec["callable"])
                except Exception:
                    pass

            if name in handbag_tool_names or (self.handbag_path and source_file and str(self.handbag_path) in str(source_file)):
                source = "handbag"
                category = "handbag_tools"
                is_handbag = True
            elif name.startswith("tool_computer_"):
                source = "computer_use"
                category = "desktop_automation"
                is_handbag = False
            elif name in ("tool_execute_python_code", "tool_execute_python_file"):
                source = "execute_python"
                category = "execution"
                is_handbag = False
            elif name == "tool_execute_shell_command":
                source = "system_shell"
                category = "system"
                is_handbag = False
            elif name in ("tool_generate_image", "tool_edit_image", "tool_text_to_speech", "tool_speech_to_text", "tool_generate_music", "tool_generate_video", "tool_send_connection"):
                source = "multimodal_binding"
                category = "multimodal"
                is_handbag = False
            elif name.startswith("tool_git_"):
                source = "git_manager"
                category = "version_control"
                is_handbag = False
            elif name in ("tool_write_file", "tool_read_file", "tool_list_files", "tool_find_files", "tool_grep_files"):
                source = "workspace_tools"
                category = "workspace"
                is_handbag = False
            elif name in ("tool_load_skill", "tool_search_skills", "tool_list_skills", "tool_create_skill", "tool_update_skill", "tool_append_to_skill", "tool_remove_skill"):
                source = "skills_manager"
                category = "skills"
                is_handbag = False
            elif name in ("tool_spawn_sub_agent", "tool_spinoff_code_specialist", "tool_spinoff_presentation_designer"):
                source = "sub_agent"
                category = "delegation"
                is_handbag = False
            elif name in ("tool_switch_model", "tool_list_models"):
                source = "model_switcher"
                category = "model_management"
                is_handbag = False
            elif name == "tool_query_rag":
                source = "rag"
                category = "knowledge_retrieval"
                is_handbag = False
            elif name.startswith(("tool_inspect_document", "tool_read_document_content", "tool_grep_document", "tool_modify_docx", "tool_modify_excel", "tool_edit_document_text", "tool_annotate_document")):
                source = "document_editor"
                category = "documents"
                is_handbag = False
            elif name == "tool_execute_python_data_query":
                source = "semantic_data_engineer"
                category = "data_engineering"
                is_handbag = False
            else:
                source = "custom"
                category = "custom_tools"
                is_handbag = False

            structured_tools.append({
                "name": name,
                "description": desc,
                "parameters": params,
                "source": source,
                "category": category,
                "is_handbag": is_handbag,
                "source_file": str(source_file) if source_file else None,
            })

        structured_tools.sort(key=lambda t: (not t["is_handbag"], t["category"], t["name"]))
        return structured_tools

    def list_tools(self) -> List[Dict[str, Any]]:
        """Alias for list_tools_structured()."""
        return self.list_tools_structured()

    def get_tools_structured(self) -> List[Dict[str, Any]]:
        """Alias for list_tools_structured()."""
        return self.list_tools_structured()

    def _discover_tools(
        self,
        explicit_tools: Optional[Dict] = None,
        tool_files: Optional[List] = None,
        enable_data_tools: bool = False,
        enable_workspace_tools: bool = True,
        enable_shell: bool = False,
        enable_python_exec: bool = False,
        enable_web_tools: bool = False,
        auto_load_document_editor: bool = True,
        enable_computer_use: bool = False,
        allow_computer_use: Optional[bool] = None,
        shell_autonomy_level: Optional[str] = "safe",
        python_autonomy_level: Optional[str] = "safe",
        auto_approve_python: bool = False,
        confirm_handler: Optional[Callable] = None,
        *args, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        if confirm_handler is None and "confirm_handler" in kwargs:
            confirm_handler = kwargs.get("confirm_handler")
        active_tools = {}

        try:
            import getpass
            current_user_name = getpass.getuser()
        except Exception:
            current_user_name = "Unknown User"

        lcp_binding = getattr(self.lollms_client, 'tools', None)
        if not _is_tool_binding(lcp_binding) or hasattr(lcp_binding, "_mock_return_value"):
            lcp_binding = None

        if lcp_binding is None:
            try:
                from lollms_client.tools_bindings.lcp import LCPBinding
                default_tools_dir = Path(__file__).resolve().parent.parent / "tools_bindings" / "lcp" / "default_tools"
                lcp_binding = LCPBinding(tools_folders=[default_tools_dir] if default_tools_dir.exists() else [])
                if hasattr(self.lollms_client, 'tools') and not hasattr(self.lollms_client.tools, "_mock_return_value"):
                    self.lollms_client.tools = lcp_binding
            except Exception as e:
                ASCIIColors.warning(f"[{self.name}] Failed to initialize LCPBinding: {e}")
                lcp_binding = None

        if lcp_binding and hasattr(lcp_binding, 'mount_tool_library'):
            # Update host configs dynamically
            if not hasattr(lcp_binding, "host_tool_configs") or lcp_binding.host_tool_configs is None:
                lcp_binding.host_tool_configs = {}

            if confirm_handler:
                if hasattr(lcp_binding, "set_confirm_handler"):
                    lcp_binding.set_confirm_handler(confirm_handler)
                lcp_binding.host_tool_configs.setdefault("execute_python", {})["confirm_handler"] = confirm_handler
                lcp_binding.host_tool_configs.setdefault("system_shell", {})["confirm_handler"] = confirm_handler

            if shell_autonomy_level:
                lcp_binding.host_tool_configs.setdefault("system_shell", {})["autonomy_level"] = shell_autonomy_level
            if python_autonomy_level or auto_approve_python is not None:
                py_cfg = lcp_binding.host_tool_configs.setdefault("execute_python", {})
                if python_autonomy_level:
                    py_cfg["autonomy_level"] = python_autonomy_level
                if auto_approve_python is not None:
                    py_cfg["auto_approve"] = auto_approve_python

            _libraries_to_mount: List[str] = []

            if enable_workspace_tools and self.capabilities and self.capabilities.enable_workspace_tools and self._resolved_workspace:
                _libraries_to_mount.append("workspace_tools")
            if enable_shell:
                _libraries_to_mount.append("system_shell")
            if enable_python_exec:
                _libraries_to_mount.append("execute_python")

            if getattr(self, "enable_git_management", False):
                _libraries_to_mount.append("git_manager")

            if allow_computer_use is None:
                allow_computer_use = kwargs.get("allow_computer_use")
            if allow_computer_use is None and self.capabilities:
                allow_computer_use = getattr(self.capabilities, "allow_computer_use", False) or getattr(self.capabilities, "enable_computer_use", False)

            computer_use_requested = bool(enable_computer_use or allow_computer_use)

            _has_vision = False
            if self.lollms_client:
                if hasattr(self.lollms_client, "has_vision_capability"):
                    try:
                        _has_vision = bool(self.lollms_client.has_vision_capability())
                    except Exception:
                        _has_vision = False
                if not _has_vision:
                    active_llm = getattr(self.lollms_client, "llm", None)
                    _has_vision = bool(getattr(active_llm, "vision_enabled", False))

            if _has_vision:
                _libraries_to_mount.append("vlm_query")

            _computer_use_vision_ready = computer_use_requested and _has_vision
            if _computer_use_vision_ready:
                _libraries_to_mount.append("computer_use")
            elif computer_use_requested and not _has_vision:
                ASCIIColors.warning(
                    f"[{self.name}] allow_computer_use is True, but active model does NOT support vision. "
                    "Computer use tools cannot be loaded without vision capability."
                )

            for lib_name in _libraries_to_mount:
                try:
                    if hasattr(lcp_binding, 'mount_tool_library_if_absent'):
                        lcp_binding.mount_tool_library_if_absent(lib_name)
                    else:
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
                    _PY_EXEC_TOOL_NAMES = {"tool_execute_python_code", "tool_execute_python_file"}
                    _GIT_TOOL_NAMES = {
                        "tool_git_status", "tool_git_diff", "tool_git_commit",
                        "tool_git_create_branch", "tool_git_checkout", "tool_git_log",
                        "tool_git_config_get", "tool_git_config_set"
                    }
                    _COMPUTER_USE_TOOL_NAMES = {
                        "tool_computer_desktop_info", "tool_computer_screenshot",
                        "tool_computer_click", "tool_computer_move_cursor",
                        "tool_computer_mouse_down", "tool_computer_mouse_up",
                        "tool_computer_drag",
                        "tool_computer_type", "tool_computer_key", "tool_computer_scroll",
                        "tool_computer_wait", "tool_computer_cursor_position"
                    }

                    allowed_tool_names = set()
                    if enable_workspace_tools and self.capabilities and self.capabilities.enable_workspace_tools and self._resolved_workspace:
                        allowed_tool_names.update(_WS_TOOL_NAMES)
                    if enable_shell:
                        allowed_tool_names.update(_SHELL_TOOL_NAMES)
                    if enable_python_exec:
                        allowed_tool_names.update(_PY_EXEC_TOOL_NAMES)

                    if getattr(self, "enable_git_management", False):
                        allowed_tool_names.update(_GIT_TOOL_NAMES)

                    if _has_vision:
                        allowed_tool_names.update({"tool_inspect_image", "tool_vlm_query"})

                    if _computer_use_vision_ready:
                        allowed_tool_names.update(_COMPUTER_USE_TOOL_NAMES)

                    for t_name, t_spec in all_lcp_tools.items():
                        if t_name in allowed_tool_names:
                            active_tools[t_name] = t_spec
                except Exception as e:
                    ASCIIColors.warning(f"[{self.name}] Failed to extract LCP tool specs: {e}")

        # ── 1. CORE MINIMUM TOOLS ONLY (File search/grep, Python, Shell) ──
        # Git, Multimodal, Computer Use, Sub-agents, and Memory tools are kept LOADABLE on demand.

        # ── 2. CONTEXTUAL DETECTION: Data vs Document files in workspace ──
        ws_path = self._resolved_workspace
        has_data_files = False
        has_document_files = False
        if ws_path and ws_path.exists():
            _DATA_EXTS = {".csv", ".tsv", ".db", ".sqlite", ".sqlite3", ".parquet"}
            _DOC_EXTS = {".pdf", ".docx", ".pptx", ".odt", ".epub", ".doc"}
            try:
                for root, dirs, files in os.walk(ws_path):
                    dirs[:] = [d for d in dirs if d not in _IGNORED_WS_DIRS and not d.startswith(".")]
                    for fname in files:
                        ext = Path(fname).suffix.lower()
                        if ext in _DATA_EXTS:
                            has_data_files = True
                        elif ext in _DOC_EXTS:
                            has_document_files = True
                        if has_data_files and has_document_files:
                            break
                    if has_data_files and has_document_files:
                        break
            except Exception:
                pass

        # ── 3. MOUNT CONTEXTUAL TOOLS ONLY WHEN MATCHING FILES EXIST ──
        if lcp_binding and hasattr(lcp_binding, 'mount_tool_library'):
            if has_data_files:
                lcp_binding.mount_tool_library_if_absent("semantic_data_engineer")
                try:
                    specs = lcp_binding.to_chat_tool_specs(
                        discussion_instance=getattr(self, '_artefact_proxy', None),
                        lollms_client_instance=self.lollms_client
                    )
                    for t_name in ("tool_execute_python_data_query", "tool_get_table_schema", "tool_query_database_sql"):
                        if t_name in specs:
                            active_tools[t_name] = specs[t_name]
                    ASCIIColors.info(f"[{self.name}] Mounted database & tabular query tools (data files detected).")
                except Exception as ex:
                    ASCIIColors.warning(f"Failed to mount data query tools: {ex}")

            if has_document_files:
                lcp_binding.mount_tool_library_if_absent("as_is_document_tools")
                lcp_binding.mount_tool_library_if_absent("document_editor")
                try:
                    specs = lcp_binding.to_chat_tool_specs(
                        discussion_instance=getattr(self, '_artefact_proxy', None),
                        lollms_client_instance=self.lollms_client
                    )
                    for t_name in ("tool_inspect_document", "tool_read_document_content", "tool_annotate_document", "tool_edit_document_text"):
                        if t_name in specs:
                            active_tools[t_name] = specs[t_name]
                    ASCIIColors.info(f"[{self.name}] Mounted document annotation & reading tools (rich documents detected).")
                except Exception as ex:
                    ASCIIColors.warning(f"Failed to mount document tools: {ex}")

        # ── 4. WIRE DYNAMIC TOOL LOADER & UNLOADER (TOOL-ON-DEMAND) ──
        loadable_tool_index = {
            "file_organizer": "Automated workspace file migration from YAML/JSON plans (tool_organize_files_from_plan)",
            "git_manager": "Git version control operations (tool_git_status, tool_git_commit, tool_git_diff, tool_git_branch, tool_git_checkout)",
            "generate_image": "Text-to-Image creation and editing (tool_generate_image, tool_edit_image)",
            "speech_to_text": "Audio transcription (tool_speech_to_text)",
            "text_to_speech": "Audio synthesis (tool_text_to_speech)",
            "generate_music": "Music and song generation (tool_generate_music, tool_generate_song)",
            "send_connection": "Message dispatch to Slack, Discord, webhooks (tool_send_connection)",
            "computer_use": "Desktop UI automation and clicking (tool_computer_screenshot, tool_computer_click, tool_computer_type)",
            "vlm_query": "Vision model inspection of images (tool_inspect_image, tool_vlm_query)",
            "model_switcher": "On-the-fly model switching (tool_switch_model, tool_list_models)",
            "memory_tools": "Cognitive memory persistence (tool_save_memory, tool_search_memory, tool_load_memory)",
        }
        object.__setattr__(self, "_loadable_tool_index", loadable_tool_index)

        def _resolve_and_mount_tool(target_name: str) -> List[str]:
            """Helper that resolves, mounts, and registers a tool into active_tools."""
            target = target_name.strip()
            loaded = []

            # 1. Already in active_tools
            if target in active_tools:
                return [target]

            # 2. Direct check in LCP binding
            if lcp_binding:
                if target in ("file_organizer", "tool_organize_files_from_plan") or "organize" in target.lower():
                    lcp_binding.mount_tool_library_if_absent("file_organizer")
                    specs = lcp_binding.to_chat_tool_specs(
                        discussion_instance=getattr(self, '_artefact_proxy', None),
                        lollms_client_instance=self.lollms_client
                    )
                    for tn in ("tool_organize_files_from_plan",):
                        if tn in specs:
                            active_tools[tn] = specs[tn]
                            loaded.append(tn)

                lib_name = lcp_binding.find_library_for_tool(target)
                if lib_name:
                    lcp_binding.mount_tool_library_if_absent(lib_name)
                    all_specs = lcp_binding.to_chat_tool_specs(
                        discussion_instance=getattr(self, '_artefact_proxy', None),
                        lollms_client_instance=self.lollms_client
                    )
                    for s_name, s_def in all_specs.items():
                        if s_name == target or target in s_name or s_name.endswith(target.replace("tool_", "")):
                            active_tools[s_name] = s_def
                            loaded.append(s_name)

            # 3. Check special categories (git, multimodal, sub_agents)
            if ("git" in target.lower() or target.startswith("tool_git_")) and lcp_binding:
                lcp_binding.mount_tool_library_if_absent("git_manager")
                specs = lcp_binding.to_chat_tool_specs()
                for gn in ("tool_git_status", "tool_git_diff", "tool_git_commit", "tool_git_branch", "tool_git_checkout", "tool_git_log"):
                    if gn in specs:
                        active_tools[gn] = specs[gn]
                        loaded.append(gn)

            elif ("image" in target.lower() or target in ("tool_generate_image", "tool_edit_image")) and BindingToolsBuilder and self.lollms_client:
                b_tools = BindingToolsBuilder.build_tools(self.lollms_client, self.capabilities, self._resolved_workspace)
                for iname in ("tool_generate_image", "tool_edit_image"):
                    if iname in b_tools:
                        active_tools[iname] = b_tools[iname]
                        loaded.append(iname)

            elif "sub_agent" in target.lower() or "spinoff" in target.lower():
                if self._sub_agent_spawner:
                    active_tools["tool_spawn_sub_agent"] = {
                        "name": "tool_spawn_sub_agent",
                        "description": "Spawn a focused sub-agent to perform a specific sub-task in the workspace.",
                        "parameters": [
                            {"name": "instruction", "type": "str", "description": "The specific task instructions for the sub-agent."},
                            {"name": "personality_conditioning", "type": "str", "description": "System prompt conditioning the sub-agent's behavior.", "optional": True},
                            {"name": "model_name", "type": "str", "description": "Specific model name to use for the sub-agent.", "optional": True},
                        ],
                        "callable": lambda instruction, personality_conditioning="", model_name="": self._sub_agent_spawner.spawn(
                            instruction=instruction, personality_conditioning=personality_conditioning or None, model_name=model_name or None
                        )
                    }
                    loaded.append("tool_spawn_sub_agent")

            elif "computer" in target.lower() and lcp_binding:
                if _has_vision:
                    lcp_binding.mount_tool_library_if_absent("computer_use")
                    specs = lcp_binding.to_chat_tool_specs()
                    for cn in (
                        "tool_computer_desktop_info", "tool_computer_screenshot",
                        "tool_computer_click", "tool_computer_move_cursor",
                        "tool_computer_mouse_down", "tool_computer_mouse_up",
                        "tool_computer_drag", "tool_computer_type",
                        "tool_computer_key", "tool_computer_scroll",
                        "tool_computer_wait", "tool_computer_cursor_position"
                    ):
                        if cn in specs:
                            active_tools[cn] = specs[cn]
                            loaded.append(cn)
                else:
                    ASCIIColors.warning(f"[{self.name}] Cannot load computer_use: active model lacks vision.")

            # 4. Search and auto-install from tools zoo if missing
            if not loaded and self._resolved_workspace:
                try:
                    from lollms_client.apps.lollms_code.zoo import ZooManager
                    zm = ZooManager(self._resolved_workspace)
                    zoo_tool = zm.find_tool_for_requirement(target)
                    if zoo_tool:
                        ok, _ = zm.install_item(zoo_tool, scope="project")
                        if ok and lcp_binding:
                            lcp_binding._discover_local_tools()
                            specs = lcp_binding.to_chat_tool_specs()
                            for s_name, s_def in specs.items():
                                if s_name == target or target in s_name or s_name.endswith(target.replace("tool_", "")):
                                    active_tools[s_name] = s_def
                                    loaded.append(s_name)
                except Exception as zoo_ex:
                    ASCIIColors.warning(f"[{self.name}] Zoo search for tool '{target}' failed: {zoo_ex}")

            return list(dict.fromkeys(loaded))

        def tool_load_tool(tool_name: str) -> dict:
            """
            Loads and activates an on-demand tool or toolset into your active tool registry for this session.

            Args:
                tool_name (str): Name of the tool or toolkit to load (e.g. 'file_organizer', 'git_manager', 'generate_image', 'computer_use', 'sub_agents', 'vlm_query').
            """
            target = tool_name.strip()
            loaded_specs = _resolve_and_mount_tool(target)

            if loaded_specs:
                return {
                    "success": True,
                    "output": f"Successfully loaded and activated tool(s): {', '.join(loaded_specs)}. Their schemas are now available in your active tool registry.",
                    "loaded_tools": loaded_specs
                }

            return {
                "success": False,
                "error": f"Tool '{target}' could not be resolved or loaded. Check available toolkits in your prompt."
            }

        def tool_unload_tool(tool_name: str) -> dict:
            """Unloads a tool from active context to free token space."""
            target = tool_name.strip()
            removed = []
            for k in list(active_tools.keys()):
                if k == target or target in k:
                    # Do not allow unloading the strict core minimum
                    if k in ("tool_find_files", "tool_grep_files", "tool_list_files", "tool_read_file", "tool_write_file", "tool_execute_python_code", "tool_execute_python_file", "tool_execute_shell_command", "tool_load_tool", "tool_load_skill"):
                        continue
                    del active_tools[k]
                    removed.append(k)
            if removed:
                return {"success": True, "output": f"Unloaded tool(s): {', '.join(removed)} from active context."}
            return {"success": False, "error": f"Tool '{target}' not found in active tools or is part of the protected core."}

        active_tools["tool_load_tool"] = {
            "name": "tool_load_tool",
            "description": "Loads on-demand toolsets (git, image generation, speech, computer use, sub-agents, etc.) into active context when needed.",
            "parameters": [
                {"name": "tool_name", "type": "str", "description": "The name of the tool or toolkit to load (e.g. 'git_manager', 'generate_image', 'computer_use', 'sub_agents')."}
            ],
            "callable": tool_load_tool
        }
        active_tools["tool_unload_tool"] = {
            "name": "tool_unload_tool",
            "description": "Unloads an active tool from context to save tokens when no longer needed.",
            "parameters": [
                {"name": "tool_name", "type": "str", "description": "The name of the tool to unload."}
            ],
            "callable": tool_unload_tool
        }

        # ── 5. SKILLS MANAGER REGISTRATION WITH AUTO-TOOL LOADING ──
        if self.capabilities and self.capabilities.enable_skill_loading and self.skills_manager:
            def _skill_tool_availability_checker(t_name: str) -> bool:
                clean = t_name.lower().strip()
                if clean in active_tools:
                    return True
                if lcp_binding and (lcp_binding.is_tool_available(clean) or lcp_binding.is_tool_available(f"tool_{clean}")):
                    return True
                return clean in loadable_tool_index or any(clean in k for k in loadable_tool_index)

            def _skill_tool_loader(t_name: str) -> Optional[Dict[str, Any]]:
                clean = t_name.strip()
                mounted_names = _resolve_and_mount_tool(clean)
                for m_name in mounted_names:
                    if m_name in active_tools:
                        return active_tools[m_name]
                if clean in active_tools:
                    return active_tools[clean]
                return None

            self.skills_manager.tool_availability_checker = _skill_tool_availability_checker
            self.skills_manager.tool_loader = _skill_tool_loader

            all_skill_tools = self.skills_manager.build_skill_tools()
            for t_name in ("tool_load_skill", "tool_unload_skill", "tool_search_skills", "tool_list_skills"):
                if t_name in all_skill_tools:
                    active_tools[t_name] = all_skill_tools[t_name]

        lcp_binding = getattr(self.lollms_client, 'tools', None)
        if not _is_tool_binding(lcp_binding) or hasattr(lcp_binding, "_mock_return_value"):
            lcp_binding = None


        if enable_data_tools and lcp_binding is None and (tool_files or self._resolved_workspace):
            try:
                from lollms_client.tools_bindings.lcp import LCPBinding
                lcp_binding = LCPBinding(tools_folders=[])
            except Exception:
                lcp_binding = None

        if enable_data_tools and auto_load_document_editor and lcp_binding and hasattr(lcp_binding, 'mount_tool_library'):
            ws_path = self._resolved_workspace
            has_data_files = False
            has_document_files = False
            if ws_path and ws_path.exists():
                _DATA_EXTS = {".csv", ".db", ".sqlite", ".sqlite3", ".parquet"}
                _DOC_EXTS = {".pdf", ".docx", ".pptx", ".odt", ".doc"}

                try:
                    for root, dirs, files in os.walk(ws_path):
                        dirs[:] = [d for d in dirs if d not in _IGNORED_WS_DIRS and not d.startswith(".")]
                        for fname in files:
                            if fname.startswith("."):
                                continue
                            ext = Path(fname).suffix.lower()
                            if ext in _DATA_EXTS:
                                has_data_files = True
                            elif ext in _DOC_EXTS:
                                has_document_files = True
                            if has_data_files and has_document_files:
                                break
                        if has_data_files and has_document_files:
                            break
                except Exception:
                    pass

            _LIBRARIES_TO_MOUNT: List[str] = []
            if has_document_files:
                _LIBRARIES_TO_MOUNT.extend(["as_is_document_tools"])
                if auto_load_document_editor:
                    _LIBRARIES_TO_MOUNT.append("document_editor")
            if has_data_files:
                _LIBRARIES_TO_MOUNT.append("semantic_data_engineer")

            for lib_name in _LIBRARIES_TO_MOUNT:
                try:
                    if hasattr(lcp_binding, 'mount_tool_library_if_absent'):
                        lcp_binding.mount_tool_library_if_absent(lib_name)
                    else:
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
                        if has_data_files and t_name in (
                            "tool_execute_python_data_query",
                            "tool_get_table_schema",
                            "tool_filter_and_slice_data",
                            "tool_get_unique_values",
                            "tool_compute_column_aggregations",
                            "tool_query_database_sql",
                        ):
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

        # ── 6. MOUNT SUB-AGENTS & MODEL SWITCHING WHEN ENABLED IN CAPABILITIES ──
        if self.capabilities and self.capabilities.enable_sub_agents and self._sub_agent_spawner:
            avail_profiles = []
            if self.lollms_client and hasattr(self.lollms_client, "llm_model_profiles_registry"):
                avail_profiles = list(self.lollms_client.llm_model_profiles_registry.keys())
            prof_hint = f" Available models: {', '.join(avail_profiles)}." if avail_profiles else ""

            active_tools["tool_spawn_sub_agent"] = {
                "name": "tool_spawn_sub_agent",
                "description": f"Spawn a focused sub-agent to perform a specific sub-task in the workspace. You can assign a specific model to this subtask.{prof_hint}",
                "parameters": [
                    {"name": "instruction", "type": "str", "description": "The specific task instructions for the sub-agent."},
                    {"name": "personality_conditioning", "type": "str", "description": "System prompt conditioning the sub-agent's behavior.", "optional": True},
                    {"name": "model_name", "type": "str", "description": f"Specific model name or profile alias to run this subtask.{prof_hint}", "optional": True},
                ],
                "callable": lambda instruction, personality_conditioning="", model_name="", **kwargs: self._sub_agent_spawner.spawn(
                    instruction=instruction, personality_conditioning=personality_conditioning or None, model_name=model_name or None, **kwargs
                )
            }

        if self.capabilities and self.capabilities.enable_model_switching and self._model_switcher:
            active_tools["tool_switch_model"] = {
                "name": "tool_switch_model",
                "description": "Switch the active model for subsequent rounds.",
                "parameters": [
                    {"name": "model_name", "type": "str", "description": "The target model name to switch to."}
                ],
                "callable": lambda model_name: self._model_switcher.switch_model(model_name)
            }
            active_tools["tool_list_models"] = {
                "name": "tool_list_models",
                "description": "List available models for model switching.",
                "parameters": [],
                "callable": lambda: self._model_switcher.list_models()
            }

        # ── 7. MOUNT MULTIMODAL BINDINGS TOOLS (TTI, TTS, STT, TTM, TTV) ──
        if BindingToolsBuilder and self.lollms_client and self.capabilities:
            try:
                b_tools = BindingToolsBuilder.build_tools(self.lollms_client, self.capabilities, self._resolved_workspace)
                active_tools.update(b_tools)
            except Exception as b_err:
                ASCIIColors.warning(f"[{self.name}] Failed building multimodal tools: {b_err}")

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
        """Reads the ephemeral scratchpad content for injection into the dynamic suffix."""
        if not getattr(self, '_scratchpad_path', None) or not self._scratchpad_path.exists():
            return ""

        try:
            content = self._scratchpad_path.read_text(encoding="utf-8", errors="ignore")
            # If the scratchpad has no meaningful notes (only headers or blank lines), do not inject
            meaningful_lines = [
                l.strip() for l in content.splitlines()
                if l.strip() and not l.startswith("#") and "(Empty - " not in l and "Use this space to store" not in l
            ]
            if not meaningful_lines:
                return ""
            return f"=== SCRATCHPAD CONTENT (CURRENT SESSION ONLY) ===\n{content.strip()}\n=== END SCRATCHPAD ==="
        except Exception:
            return ""

    def _build_current_plan_context(self) -> str:
        """Reads the CURRENT.md plan for injection into the dynamic suffix."""
        if not self._resolved_workspace:
            return ""
        current_path = self._resolved_workspace / ".lollms_code" / "CURRENT.md"
        if not current_path.exists():
            return ""
        try:
            content = current_path.read_text(encoding="utf-8", errors="ignore")
            clean_content = content.strip()
            if not clean_content or clean_content.startswith("# Current Task\n\nNo active task plan"):
                return ""
            return f"=== CURRENT TASK PLAN (CURRENT.md) ===\n{clean_content}\n=== END CURRENT TASK PLAN ==="
        except Exception:
            return ""

    def _should_compress_context(self, new_prompt: str) -> bool:
        """
        Determines whether the incoming prompt represents a major shift in objective
        or a new task, warranting compression of the previous context into pinned lessons.
        """
        if not self._conversation or len(self._conversation) < 2:
            return False

        new_prompt_lower = new_prompt.strip().lower()

        # 1. Explicit new-task indicators
        new_task_patterns = [
            r'^(?:new\s+task|next\s+task|different\s+task|another\s+task)\b',
            r'^(?:now\s+)?(?:let\'?s\s+)?(?:switch\s+to|start\s+(?:a\s+)?new|move\s+on\s+to)\b',
            r'^(?:forget\s+(?:about\s+)?(?:that|the\s+previous|earlier)|instead\s+of\s+that)\b',
            r'^(?:nouvelle\s+t\u00e2che|passons\s+\u00e0|autre\s+chose)\b',
        ]
        for pat in new_task_patterns:
            if re.search(pat, new_prompt_lower):
                return True

        # 2. Strong continuation indicators (DO NOT compress)
        continuation_patterns = [
            r'^(?:continue|proceed|go\s+on|keep\s+going|more|next\s+step)\b',
            r'^(?:fix|debug|repair|correct|modify|change|update|delete|run|test|execute)\s+(?:it|this|that|the\s+file|the\s+script|the\s+bug|the\s+code)\b',
            r'^(?:why|what\s+happened|explain|how\s+come)\b',
            r'^(?:yes|no|y|n|ok|okay|sure|do\s+it)\b',
        ]
        for pat in continuation_patterns:
            if re.search(pat, new_prompt_lower):
                return False

        # 3. Lexical overlap analysis against previous task keywords
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', new_prompt_lower))
        meaningful_words = {w for w in words if w not in _STOP_WORDS}
        if not meaningful_words:
            return False

        prev_words: set[str] = set()
        for msg in self._conversation:
            if msg.get("role") == "user":
                content = str(msg.get("content", "")).lower()
                for w in re.findall(r'\b[a-zA-Z]{3,}\b', content):
                    if w not in _STOP_WORDS:
                        prev_words.add(w)

        if not prev_words:
            return False

        overlap = len(meaningful_words & prev_words)
        overlap_ratio = overlap / len(meaningful_words)

        # If keyword overlap is very low (< 15%), it represents a distinct new task
        if overlap_ratio < 0.15:
            return True

        return False

    def _compress_previous_task_context(self, new_prompt: str) -> None:
        """
        Compresses previous conversation context into persistent lessons learned,
        pins them to the system prompt, and resets working history so the agent starts
        the new task with clean context while preserving all acquired knowledge.
        """
        if not self._conversation:
            return

        ASCIIColors.info(f"[{self.name}] 🧠 Objective shift detected. Compressing previous context into pinned lessons...")

        history_lines = []
        for msg in self._conversation:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            content = re.sub(r'<think\b[^>]*>.*?(?:</think>|$)', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<thought\b[^>]*>.*?(?:</thought>|$)', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<processing\b[^>]*>.*?(?:</processing>|$)', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = content.strip()
            if content:
                history_lines.append(f"{role.upper()}: {content}")

        history_text = "\n\n".join(history_lines)
        if not history_text.strip():
            self._conversation.clear()
            return

        extracted_lessons = ""

        if self.lollms_client:
            try:
                extraction_prompt = (
                    "You are a Senior Software Architect. The user is starting a new task.\n"
                    "Analyze the previous session history below and extract a concise list of KEY LESSONS LEARNED, "
                    "CODE/ENVIRONMENT RULES, and ARCHITECTURAL CONSTRAINTS established during that session.\n"
                    "Requirements:\n"
                    "1. Focus ONLY on actionable facts, technical constraints, bugs resolved, and tool findings.\n"
                    "2. Exclude conversational filler, pleasantries, apologies, and intermediate step-by-step logs.\n"
                    "3. Format as clean bullet points.\n"
                    "4. Maximum 5 bullet points.\n\n"
                    f"=== PREVIOUS SESSION ===\n{history_text[:4000]}\n=== END PREVIOUS SESSION ===\n\n"
                    "KEY LESSONS & CONSTRAINTS:"
                )
                raw_extracted = self.lollms_client.generate_text(
                    prompt=extraction_prompt,
                    temperature=0.1,
                    n_predict=512
                )
                if isinstance(raw_extracted, str) and raw_extracted.strip():
                    extracted_lessons = re.sub(r'<think\b[^>]*>.*?(?:</think>|$)', '', raw_extracted, flags=re.DOTALL | re.IGNORECASE).strip()
            except Exception as ex:
                ASCIIColors.warning(f"[{self.name}] LLM lesson extraction failed: {ex}")

        if not extracted_lessons:
            summary_bullets = []
            for msg in self._conversation:
                if msg.get("role") == "user":
                    summary_bullets.append(f"- Previous task: {str(msg.get('content', ''))[:120]}...")
            extracted_lessons = "\n".join(summary_bullets[:4])

        if extracted_lessons:
            timestamp = time.strftime("%Y-%m-%d %H:%M")
            lesson_entry = f"### Session ({timestamp}):\n{extracted_lessons}"
            if getattr(self, "_pinned_lessons", ""):
                combined = f"{self._pinned_lessons}\n\n{lesson_entry}"
                if len(combined) > 3000:
                    combined = combined[-3000:]
                self._pinned_lessons = combined
            else:
                self._pinned_lessons = lesson_entry

            ASCIIColors.success(f"[{self.name}] 📌 Pinned lessons from previous task to system prompt.")

        # Clear working conversation history; pinned lessons are preserved in system prompt
        self._conversation.clear()
        if hasattr(self, "_project_history_file") and self._project_history_file:
            try:
                self.save_history_to_disk(self._project_history_file)
            except Exception:
                pass

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

    def _build_system_prompt(self, active_tools: Optional[Dict] = None, dynamic_effort: bool = False) -> str:
        sys_prompt = (self.system_prompt or "").strip()
        onboarding_block = self._build_onboarding_block()

        has_authoritative_protocol = (
            "=== AUTHORITATIVE OPERATING PROTOCOL" in sys_prompt
            or "=== CORE OPERATING PROTOCOL" in sys_prompt
            or "CORE OPERATIONAL DIRECTIVES" in sys_prompt
        )

        operating_protocol = ""
        if not has_authoritative_protocol:
            operating_protocol = (
                "\n=== AUTHORITATIVE OPERATING PROTOCOL ===\n"
                "1. **ACTION FIRST (NO PREAMBLE)**: When performing a task, output the action tag (`<tool>`, `<artifact>`, `<unlock_file>`) as your FIRST tokens. Never write prose checklists explaining what you plan to do.\n"
                "2. **TERMINATION CONTRACT (<done/>)**: Conclude completed tasks with `<done/>` on a new line. On casual greetings (e.g. 'hi'), reply conversationally and finish with `<done/>`.\n"
                "3. **PASSIVE MEMORY BOUNDARY**: Memories provide passive background facts only. Your active objective is determined exclusively by the latest user message.\n"
                "4. **SURGICAL PATCHES**: For existing files, use Aider SEARCH/REPLACE patches with exact verbatim lines.\n\n"
                "### FEW-SHOT EXECUTION PATTERNS (MANDATORY DEMONSTRATIONS):\n"
                "User: \"Execute the migration plan mapping.yaml\"\n"
                "Assistant:\n"
                "<tool>{\"name\": \"tool_organize_files_from_plan\", \"parameters\": {\"plan_file\": \"mapping.yaml\", \"move_files\": true}}</tool>\n\n"
                "User: \"Create hello.py\"\n"
                "Assistant:\n"
                "<artifact name=\"hello.py\" type=\"code\" language=\"python\">\n"
                "print(\"Hello world!\")\n"
                "</artifact>\n\n"
                "User: \"Read doc.txt\"\n"
                "Assistant:\n"
                "<unlock_file>doc.txt</unlock_file>\n\n"
                "=== END OPERATING PROTOCOL ===\n"
            )

        memory_instructions = ""
        if self.memory_manager:
            if hasattr(self.memory_manager, "build_system_instructions"):
                memory_instructions = self.memory_manager.build_system_instructions()
            else:
                memory_instructions = (
                    "\n=== PERSISTENT COGNITIVE MEMORY ===\n"
                    "Store facts with `<mem_new content=\"...\" tags=\"...\" level=\"1\" />`. Update with `<mem_update id=\"ID\" content=\"...\" />`.\n"
                    "When the user says 'remember this', execute the memory action immediately in that response.\n"
                    "=== END MEMORY ===\n"
                )

        skills_ctx = ""
        has_skills_already = "=== AVAILABLE SKILLS" in sys_prompt or "=== ACTIVE SKILLS" in sys_prompt
        if self.skills_manager and not has_skills_already:
            active_names = set(active_tools.keys()) if active_tools else None
            query_hint = getattr(self, '_current_chat_prompt', None)
            current_r = getattr(self, '_current_round_num', 1)
            skills_ctx_str = self.skills_manager.build_context(active_tool_names=active_names, current_query=query_hint, round_count=current_r)
            if skills_ctx_str:
                skills_ctx = "\n" + skills_ctx_str

        tool_desc = ""
        if active_tools:
            tool_sections = [
                "\n=== ACTIVE TOOLS (Lean Schema) ===",
                "Emit on a clean line: `<tool>{\"name\": \"...\", \"parameters\": {...}}</tool>`\n"
            ]
            for t_name, t_spec in sorted(active_tools.items()):
                desc = (t_spec.get("description") or "").strip().split("\n\n")[0].strip()
                params_list = t_spec.get("parameters") or []
                param_sig = ", ".join([f"{p.get('name')}: {p.get('type', 'any')}{'?' if p.get('optional') else ''}" for p in params_list]) if params_list else ""
                tool_sections.append(f"• **`{t_name}({param_sig})`**: {desc}")

            loadable_index = getattr(self, "_loadable_tool_index", None)
            if loadable_index:
                tool_sections.append("\n=== ON-DEMAND TOOLSETS (via `tool_load_tool`) ===")
                for l_name, l_desc in sorted(loadable_index.items()):
                    tool_sections.append(f"- `{l_name}`: {l_desc}")

            tool_sections.append("=== END TOOLS ===\n")
            tool_desc = "\n".join(tool_sections)

        dynamic_effort_prompt = ""
        if dynamic_effort:
            dynamic_effort_prompt = (
                "\n=== DYNAMIC REASONING EFFORT PROTOCOL ===\n"
                "You can dynamically adjust your reasoning effort across rounds based on task complexity.\n"
                "Your initial effort setting is DEACTIVATED (none).\n"
                "When a task is simple, routine, or conversational, keep effort at 'none' for speed.\n"
                "If you face complex logic, deep refactoring, difficult bugs, or architectural decisions, you can adjust your thinking effort for the NEXT round by emitting:\n"
                "<effort level=\"none|low|medium|high\"/>\n"
                "- level=\"none\": Disable reasoning for fast execution, simple lookups, or final responses.\n"
                "- level=\"low\": Light reasoning for moderate checks.\n"
                "- level=\"medium\": Standard deep reasoning for non-trivial logic.\n"
                "- level=\"high\": Maximum reasoning depth for complex problems, mathematical proofs, or challenging refactoring.\n"
                "The effort level applies starting from your very next round. Emit the tag on its own line when adjusting effort.\n"
                "=== END DYNAMIC REASONING EFFORT PROTOCOL ===\n"
            )

        computer_use_workflow = ""
        _has_computer_use_tools = any(t_name.startswith("tool_computer_") for t_name in active_tools)
        if _has_computer_use_tools:
            computer_use_workflow = (
                "\n=== COMPUTER USE OPERATING DOCTRINE (MANDATORY) ===\n"
                "Desktop automation is enabled. Follow this loop STRICTLY:\n"
                "1. **OBSERVE**: Call `tool_computer_screenshot` to SEE the current screen state. NEVER act blind.\n"
                "2. **LOCATE**: Use `tool_computer_click` with either explicit x/y coordinates read from the screenshot, OR a natural-language `description` (visually grounded automatically).\n"
                "3. **ACT**: Click, then `tool_computer_type` / `tool_computer_key` / `tool_computer_scroll` as needed. Click into a text field BEFORE typing.\n"
                "4. **VERIFY**: After EVERY action, take another screenshot to confirm the effect before proceeding.\n"
                "5. **TERMINATE**: When the goal is achieved, describe the result and emit `<done/>`. Do NOT loop screenshots indefinitely.\n"
                "Start every computer use task with `tool_computer_desktop_info` to learn the screen geometry.\n"
                "=== END COMPUTER USE OPERATING DOCTRINE ===\n"
            )

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

        parts = [sys_prompt]
        if onboarding_block:
            parts.append(onboarding_block)
        if operating_protocol:
            parts.append(operating_protocol)
        if skills_ctx:
            parts.append(skills_ctx)
        if memory_instructions:
            parts.append(memory_instructions)
        if tool_desc:
            parts.append(tool_desc)
        if dynamic_effort_prompt:
            parts.append(dynamic_effort_prompt)
        if computer_use_workflow:
            parts.append(computer_use_workflow)
        if document_annotation_workflow:
            parts.append(document_annotation_workflow)

        return "\n\n".join(p.strip() for p in parts if p and p.strip())
    
    
    def change_file_visibility(self, targets: List[str], action: str) -> Dict[str, Any]:
        if getattr(self, '_artefact_manager', None) is None and (self._resolved_workspace or self.workspace_path):
            self._init_artefact_system()

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

    def _register_unindexed_workspace_files(self, targets: List[str], all_arts: List[Dict[str, Any]]) -> List[str]:
        """
        Resolves targets that exist on disk but are missing from the artefact index.

        The tool layer addresses files relative to the workspace root (matching
        Path.cwd()), while the artefact index keys files by workspace-relative
        titles. When a target matches a real file on disk, it is imported into
        the artefact manager on demand so visibility operations can proceed.
        Returns the target list with any workspace-prefix ambiguity removed.
        """
        if not self._resolved_workspace:
            return targets

        indexed_titles = {a.get("title", "") for a in all_arts}
        resolved: List[str] = []

        for target in targets:
            if target in indexed_titles:
                resolved.append(target)
                continue

            candidate = self._resolved_workspace / target
            if candidate.is_file():
                try:
                    imported = self._artefact_manager.import_file(file_path=candidate, title=target, active=False, parse_data_schema=False)
                    if imported:
                        resolved.append(imported.get("title", target))
                    else:
                        resolved.append(target)
                except Exception as import_err:
                    ASCIIColors.warning(f"[{self.name}] Failed to import '{target}' into artefact index: {import_err}")
                    resolved.append(target)
                continue

            resolved.append(target)

        return resolved

    def _execute_context_visibility(self, tag_name: str, body: str) -> Dict[str, Any]:
        if getattr(self, '_artefact_manager', None) is None and (self._resolved_workspace or self.workspace_path):
            self._init_artefact_system()

        # Check if targets reference sub_workspace/
        if self._resolved_workspace and ("sub_workspace" in body or (self._resolved_workspace / ".lollms_code" / "sub_workspace").exists()):
            try:
                from lollms_client.apps.lollms_code.sub_workspace import SubWorkspaceManager
                sub_ws = SubWorkspaceManager(self._resolved_workspace)
                targets = [t.strip().replace("\\", "/") for t in body.splitlines() if t.strip()]
                handled = []
                for t in targets:
                    clean = t[len("sub_workspace/"):].lstrip("/") if t.startswith("sub_workspace/") else t.lstrip("/")
                    if (sub_ws.sub_ws_dir / clean).exists():
                        if tag_name in ("unlock_file", "load_file"):
                            sub_ws.load_file(clean)
                            handled.append(f"sub_workspace/{clean}")
                        elif tag_name in ("lock_file", "unload_file"):
                            sub_ws.unload_file(clean)
                            handled.append(f"sub_workspace/{clean}")
                if handled:
                    status_action = "Unlocked" if tag_name in ("unlock_file", "load_file") else "Locked"
                    return {
                        "status_str": f"✅ {status_action} reference files: {', '.join(handled)}",
                        "processed_files": handled,
                        "already_in_state": [],
                        "not_found": [],
                        "blocked_files": [],
                        "loaded_contents": {h: sub_ws.peek_file(h.split('/', 1)[-1]) for h in handled} if tag_name in ("unlock_file", "load_file") else {},
                        "success": True
                    }
            except Exception as e:
                ASCIIColors.warning(f"Sub-workspace visibility handling failed: {e}")

        return _core_execute_context_visibility(
            tag_name=tag_name,
            body=body,
            artefact_manager=getattr(self, '_artefact_manager', None),
            workspace_dir=self._resolved_workspace,
            client=self.lollms_client,
            state_db_path=getattr(self, '_state_db_path', None)
        )
        
          
        
    def _inject_tool_images_for_vlm(self, tool_result: Dict[str, Any]) -> List[Dict[str, str]]:
        return _inject_tool_images_for_vlm_core(tool_result, self.lollms_client)

    def _execute_tool(self, tool_name: str, tool_params: Dict[str, Any], active_tools: Dict) -> Dict[str, Any]:
        ws_dir = self._resolved_workspace or Path(".")
        return _core_execute_tool_call(
            tool_name=tool_name,
            tool_params=tool_params,
            active_tools=active_tools,
            workspace_dir=ws_dir,
            lollms_client=self.lollms_client,
            discussion_instance=getattr(self, '_artefact_proxy', None)
        )

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
        auto_load_document_editor: bool = True,
        enable_computer_use: bool = False,
        allow_computer_use: Optional[bool] = None,
        enforce_end_tag: bool = True,
        orchestrator_mode: bool = False,
        context_compaction_threshold: float = 0.85,
        event_mode: EventMode = EventMode.PROCESSING_TAG_MODE,
        think: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        shell_autonomy_level: Optional[str] = "safe",
        python_autonomy_level: Optional[str] = "safe",
        auto_approve_python: bool = False,
        confirm_handler: Optional[Callable] = None,
        dynamic_effort: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        if confirm_handler is None and "confirm_handler" in kwargs:
            confirm_handler = kwargs.get("confirm_handler")
        resolved_max_rounds = max_nb_rounds if max_nb_rounds is not None else max_reasoning_steps
        if resolved_max_rounds is None:
            resolved_max_rounds = 20

        is_infinite_rounds = resolved_max_rounds <= 0 or resolved_max_rounds == float("inf")
        if is_infinite_rounds:
            ASCIIColors.warning(
                f"[{self.name}] ⚠️ WARNING: Infinite reasoning rounds enabled (max_rounds <= 0). "
                "The agent will continue running until <done/> is emitted or cancelled by user."
            )

        event_mode = normalize_event_mode(kwargs.pop("event_mode", event_mode))

        if orchestrator_mode:
            from lollms_client.lollms_agentic.runner import AgenticRunner
            auto_load_doc_editor_flag = kwargs.get("auto_load_document_editor", True)
            enable_data_tools_flag = kwargs.get("enable_data_tools", True)
            active_tools = self._discover_tools(
                tools,
                tool_files or [],
                enable_data_tools=enable_data_tools_flag,
                enable_workspace_tools=enable_workspace_tools,
                enable_shell=enable_shell,
                enable_python_exec=enable_python_exec,
                enable_web_tools=enable_web_tools,
                auto_load_document_editor=auto_load_doc_editor_flag,
                enable_computer_use=enable_computer_use,
                allow_computer_use=allow_computer_use,
            )
            runner = AgenticRunner(
                context=self,
                tools_registry=active_tools,
                callback=streaming_callback,
                event_mode=event_mode,
                max_orchestrator_rounds=resolved_max_rounds,
                max_worker_rounds=max(2, resolved_max_rounds // 2),
            )
            return runner.run(user_message=prompt)

        if lollms_client is not None:
            self.lollms_client = lollms_client

        if self.lollms_client is None:
            raise RuntimeError(f"[{self.name}] Independent chat requires a lollms_client instance.")

        self._reset_cancel_state()
        object.__setattr__(self, '_consecutive_empty_responses', 0)
        object.__setattr__(self, '_consecutive_stall_count', 0)
        object.__setattr__(self, '_consecutive_artifact_rounds', 0)
        object.__setattr__(self, '_max_rounds', "∞" if is_infinite_rounds else resolved_max_rounds)

        if self._sub_agent_spawner:
            self._sub_agent_spawner.reset_turn()

        if self._failure_memory and hasattr(self._failure_memory, '_signatures'):
            self._failure_memory._signatures.clear()
            if hasattr(self._failure_memory, 'failures'):
                self._failure_memory.failures = []

        kwargs["auto_load_document_editor"] = auto_load_document_editor
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
        if _current_compile is None or getattr(_current_compile, '__module__', '') != 'builtins':
            ASCIIColors.error(f"[{self.name}] CRITICAL SHADOW DETECTED in chat(): builtins.compile is not native (module: {getattr(_current_compile, '__module__', 'None')}). Restoring it.")
            import importlib as _importlib_check
            _real_builtins = _importlib_check.import_module('builtins')
            _builtins_mod_check.compile = _real_builtins.compile

        self._init_scratchpad()
        object.__setattr__(self, '_active_streaming_callback', streaming_callback)
        auto_load_doc_editor_flag = kwargs.get("auto_load_document_editor", True)

        cleaned_prompt = prompt
        enable_data_tools_flag = kwargs.get("enable_data_tools", True)
        if allow_computer_use is None:
            allow_computer_use = kwargs.get("allow_computer_use")
        if allow_computer_use is None and self.capabilities:
            allow_computer_use = getattr(self.capabilities, "allow_computer_use", False) or getattr(self.capabilities, "enable_computer_use", False)

        resolved_computer_use = bool(enable_computer_use or allow_computer_use)

        active_tools = self._discover_tools(
            tools,
            tool_files or [],
            enable_data_tools=enable_data_tools_flag,
            enable_workspace_tools=enable_workspace_tools,
            enable_shell=enable_shell,
            enable_python_exec=enable_python_exec,
            enable_web_tools=enable_web_tools,
            auto_load_document_editor=auto_load_doc_editor_flag,
            enable_computer_use=resolved_computer_use,
            allow_computer_use=resolved_computer_use,
            shell_autonomy_level=shell_autonomy_level,
            python_autonomy_level=python_autonomy_level,
            auto_approve_python=auto_approve_python,
            confirm_handler=confirm_handler
        )

        if active_tools:
            for t_name, t_spec in active_tools.items():
                if t_name in ("tool_create_skill", "tool_update_skill"):
                    t_spec["description"] += "\n\n**VISIBILITY CONTROL**: You can control how this skill is stored in your workspace by setting the `output_visibility` parameter.\n- `output_visibility=\"context\"` (default): Loads the skill directly into your context [C].\n- `output_visibility=\"artefact\"`: Saves the skill as a file [U] without loading it, saving context space."

        stable_system_prompt = self._build_system_prompt(active_tools, dynamic_effort=dynamic_effort)
        stable_system_prompt += self._build_user_profile_context()

        # Pre-hydrate RAG knowledge base context into prompt
        collected_sources: List[Dict[str, Any]] = []
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
                        score_val = src.get("score")
                        score_str = f" (Score: {score_val:.2f})" if isinstance(score_val, (int, float)) and score_val <= 1.0 else (f" (Score: {score_val})" if score_val is not None else "")

                        src_idx = len(collected_sources) + 1
                        raw_snippet = src.get("snippet") or (src.get("content", "")[:300] if src.get("content") else "")
                        src_entry = {
                            "id": src_idx,
                            "index": src_idx,
                            "title": title,
                            "source": src.get("source") or title,
                            "url": src.get("url") or src.get("link") or src.get("file_path") or "",
                            "snippet": str(raw_snippet)[:500],
                            "score": score_val,
                            "type": "rag",
                            "datasource_name": src.get("datasource_name", "")
                        }
                        collected_sources.append(src_entry)

                        sources_text.append(f"--- Document [{src_idx}] {title}{ds_label}{score_str} ---\n{src.get('content')}")
                    if sources_text:
                        stable_system_prompt += "\n=== RETRIEVED RAG CONTEXT ===\n" + "\n\n".join(sources_text) + "\n=== END RAG CONTEXT ===\n"
            except Exception as rag_err:
                ASCIIColors.warning(f"[{self.name}] RAG pre-hydration warning: {rag_err}")


        dynamic_suffix_parts = []

        object.__setattr__(self, '_current_chat_prompt', cleaned_prompt)
        clean_input = cleaned_prompt.strip().lower()
        words = set(re.findall(r'\b[a-zA-Z]+\b', clean_input))
        GREETING_WORDS = {"hi", "hello", "hey", "salut", "bonjour", "coucou", "yo", "greetings", "sup"}
        GREETING_PHRASES = {
            "hi", "hello", "hey", "salut", "bonjour", "coucou", "yo", "sup",
            "hi there", "hello there", "good morning", "good evening", "good afternoon",
            "how are you", "ca va", "comment ca va", "thanks", "thank you", "merci"
        }
        # Explicitly exempt continuation commands from ever being classified as greetings
        is_continuation = bool(re.search(r'\b(?:continue|resume|proceed|go\s+on)\b', clean_input))
        is_approval = clean_input in ("yes", "y", "oui", "proceed", "approved", "ok", "do it", "sure", "go ahead")
        is_greeting = (
            not is_continuation and not is_approval and (
                clean_input in GREETING_PHRASES
                or (len(words) <= 2 and bool(words & GREETING_WORDS))
            )
        )

        ws_ctx = self._build_workspace_context_block()
        if ws_ctx:
            dynamic_suffix_parts.append(ws_ctx.strip())

        # Suppress stale scratchpad, old CURRENT.md roadmaps, and deep memories on casual greetings
        if not is_greeting:
            scratchpad_ctx = self._build_scratchpad_context()
            if scratchpad_ctx:
                dynamic_suffix_parts.append(scratchpad_ctx.strip())
                object.__setattr__(self, '_scratchpad_content', scratchpad_ctx)

            current_plan_ctx = self._build_current_plan_context()
            if current_plan_ctx:
                dynamic_suffix_parts.append(current_plan_ctx.strip())

        if self.memory_manager:
            try:
                if not is_greeting and hasattr(self.memory_manager, 'auto_pull_deep_memories'):
                    self.memory_manager.auto_pull_deep_memories(cleaned_prompt)

                if hasattr(self.memory_manager, 'build_working_zone'):
                    mem_zone = self.memory_manager.build_working_zone()
                    if mem_zone:
                        dynamic_suffix_parts.append(mem_zone.strip())

                if not is_greeting and hasattr(self.memory_manager, 'build_handles_zone'):
                    handles_zone = self.memory_manager.build_handles_zone()
                    if handles_zone:
                        dynamic_suffix_parts.append(handles_zone.strip())
            except Exception as mem_ex:
                ASCIIColors.warning(f"[{self.name}] Failed to hydrate memories: {mem_ex}")

        # Check for objective shift / new task and compress context into pinned lessons if needed
        if use_internal_history and self._conversation:
            if self._should_compress_context(cleaned_prompt):
                self._compress_previous_task_context(cleaned_prompt)

        # Pin acquired lessons to the system prompt
        if getattr(self, "_pinned_lessons", ""):
            pinned_block = (
                "\n=== PINNED LESSONS & CONSTRAINTS FROM PREVIOUS SESSIONS ===\n"
                f"{self._pinned_lessons}\n"
                "=== END PINNED LESSONS ===\n"
            )
            stable_system_prompt += pinned_block

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

        # ── 🚀 APPROVAL HYDRATION: Ensure user 'yes' executes pending migration plans ──
        if is_approval and self._resolved_workspace:
            plan_file = None
            for cand_plan in ("mapping.yaml", "mapping.json", "mapping.md"):
                if (self._resolved_workspace / cand_plan).exists():
                    plan_file = cand_plan
                    break
            if plan_file:
                fused_prompt = (
                    f"{cleaned_prompt}\n\n"
                    f"[SYSTEM DIRECTIVE: USER CONFIRMED PLAN APPROVAL]\n"
                    f"The user approved the migration plan '{plan_file}'.\n"
                    f"You MUST now execute Phase 4 immediately:\n"
                    f"Call `<tool>{{\"name\": \"tool_organize_files_from_plan\", \"parameters\": {{\"plan_file\": \"{plan_file}\", \"move_files\": true}}}}</tool>` as your very first token now!\n"
                    f"Do NOT output conversational preambles or step lists without the tool call tag."
                )

        base_conversation.append({"role": "user", "content": fused_prompt})

        virtual_history: List[SimpleNamespace] = []
        tool_calls_this_turn: List[Dict[str, Any]] = []
        tool_results_this_turn: List[Dict[str, Any]] = []
        round_count = 0
        was_cancelled = False
        successful_tool_signatures: set = set()
        failed_tool_signatures: Dict[str, int] = {}
        object.__setattr__(self, "_failed_tool_signatures", failed_tool_signatures)
        seen_context_signatures: set = set()
        final_response = ""
        workspace_changes: List[Dict[str, Any]] = []
        base_temperature = temperature
        active_temperature = temperature

        consecutive_connection_errors = 0
        last_connection_error_desc = ""

        if dynamic_effort:
            if reasoning_effort:
                active_reasoning_effort = str(reasoning_effort).strip().lower()
                active_think_flag = active_reasoning_effort not in ("none", "off", "0", "disabled", "false")
            else:
                active_reasoning_effort = "none"
                active_think_flag = False
        else:
            active_reasoning_effort = reasoning_effort
            active_think_flag = think

        while is_infinite_rounds or round_count < resolved_max_rounds:
            if self.is_generation_cancelled():
                was_cancelled = True
                if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                    try:
                        streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                            "round_id": round_count,
                            "status": "cancelled"
                        })
                    except Exception:
                        pass
                break

            round_count += 1
            object.__setattr__(self, '_current_round_num', round_count)

            if getattr(self, 'debug_mode', False):
                ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count}/{self._max_rounds} START ===")

            if event_mode.has_tags and streaming_callback:
                try:
                    round_tag = f'<round id="{round_count}"/>\n'
                    streaming_callback(round_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True, "round": round_count})
                except Exception:
                    pass

            if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                try:
                    streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_START, {
                        "round_id": round_count,
                        "max_rounds": self._max_rounds
                    })
                except Exception:
                    pass

            pre_gen_telemetry = self._calculate_context_telemetry(
                stable_system_prompt, base_conversation,
                self._build_workspace_context_block() if hasattr(self, '_build_workspace_context_block') else "",
                virtual_history
            )
            pre_gen_fill = pre_gen_telemetry.get("fill_percentage", 0.0)

            # ── 🧹 AUTONOMOUS CONTEXT COMPACTION (FAST AGENT PRE-GENERATION GATE) ──
            norm_threshold = context_compaction_threshold if context_compaction_threshold <= 1.0 else context_compaction_threshold / 100.0
            threshold_pct = norm_threshold * 100.0

            if pre_gen_fill >= threshold_pct and not was_cancelled:
                ASCIIColors.warning(
                    f"[{self.name}] 🚨 Context fill at {pre_gen_fill:.1f}% (>= {threshold_pct:.0f}% threshold). "
                    "Running Fast Context Compaction Agent to lock non-essential files and compact history before generation..."
                )

                all_raw_arts = self._artefact_manager._get_all_raw() if getattr(self, "_artefact_manager", None) else []
                loaded_arts = [
                    a for a in all_raw_arts
                    if a.get("visibility") == ArtefactVisibility.FULL and not a.get("title", "").endswith("::images")
                ]
                pinned_titles = {
                    a.get("title", "") for a in all_raw_arts
                    if a.get("visibility") == ArtefactVisibility.PINNED
                }

                history_dicts = [
                    {"role": vh.sender_type, "content": getattr(vh, "content", "")}
                    for vh in virtual_history
                ]

                compaction_res = _core_run_fast_context_compaction(
                    client=self.lollms_client,
                    task=prompt,
                    loaded_files=loaded_arts,
                    pinned_files=pinned_titles,
                    history_items=history_dicts,
                    fill_pct=pre_gen_fill,
                    threshold_pct=threshold_pct,
                )

                files_to_lock = compaction_res.get("files_to_lock", [])
                history_summary = compaction_res.get("history_summary", "").strip()

                if files_to_lock:
                    self._execute_context_visibility("lock_file", "\n".join(files_to_lock))
                    object.__setattr__(self, "_last_ws_sync_time", 0.0)
                    ASCIIColors.success(f"[{self.name}] 🔒 Locked {len(files_to_lock)} non-essential file(s): {', '.join(files_to_lock)}")

                if history_summary and len(virtual_history) > 1:
                    virtual_history.clear()
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content=(
                            f"[SYSTEM: CONTEXT AUTO-COMPACTED ({pre_gen_fill:.1f}% >= {threshold_pct:.0f}%)]\n"
                            f"Previous conversation and tool executions were summarized to prevent server disconnect:\n\n"
                            f"{history_summary}\n\n"
                            f"Continue your task based on this summary. If finished, output <done/>."
                        )
                    ))
                    ASCIIColors.success(f"[{self.name}] 📝 Compacted virtual history into concise state summary.")

                if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                    try:
                        streaming_callback(
                            f"\n\n🧹 [Context Health Guard: {pre_gen_fill:.1f}% >= {threshold_pct:.0f}%] "
                            f"Locked {len(files_to_lock)} file(s) and compacted history to prevent backend disconnect.\n\n",
                            MSG_TYPE.MSG_TYPE_INFO,
                            {"type": "context_compression", "locked_files": files_to_lock, "fill_pct": pre_gen_fill}
                        )
                    except Exception:
                        pass

                # Re-calculate workspace context block and telemetry after compaction
                ws_ctx = self._build_workspace_context_block() if hasattr(self, '_build_workspace_context_block') else ""
                pre_gen_telemetry = self._calculate_context_telemetry(
                    stable_system_prompt, base_conversation, ws_ctx, virtual_history
                )
                pre_gen_fill = pre_gen_telemetry.get("fill_percentage", 0.0)
                ASCIIColors.info(f"[{self.name}] ✅ Post-compaction context fill: {pre_gen_fill:.1f}% ({pre_gen_telemetry.get('total', 0):,} tokens).")

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

            pending_vlm = getattr(self, '_pending_vlm_images', [])
            if pending_vlm:
                last_user_idx = None
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        last_user_idx = i
                        break
                if last_user_idx is not None:
                    orig_content = messages[last_user_idx].get("content", "")
                    if isinstance(orig_content, str):
                        content_blocks = [{"type": "text", "text": orig_content}]
                    elif isinstance(orig_content, list):
                        content_blocks = list(orig_content)
                    else:
                        content_blocks = [{"type": "text", "text": str(orig_content)}]
                    content_blocks.extend(pending_vlm)
                    messages[last_user_idx]["content"] = content_blocks
                object.__setattr__(self, '_pending_vlm_images', [])

            ss = _AgentStreamState(
                callback=streaming_callback,
                event_mode=event_mode,
                workspace_path=self._resolved_workspace
            )

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

            gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("streaming_callback", "temperature", "n_predict", "stream", "think", "reasoning_effort", "reasoning_summary")}

            # Auto Max Generation Tokens resolution:
            # If n_predict or max_tokens_per_turn <= 0 (Auto), calculate safe max remaining context window
            resolved_n_predict = n_predict if (n_predict is not None and n_predict > 0) else self.max_tokens_per_turn
            if resolved_n_predict <= 0:
                max_ctx = 8192
                if self.lollms_client and hasattr(self.lollms_client, "get_ctx_size"):
                    max_ctx = self.lollms_client.get_ctx_size() or 8192
                prompt_used = pre_gen_telemetry.get("total", 0) if "pre_gen_telemetry" in locals() else 2048
                available_tokens = max(1024, max_ctx - prompt_used - 256)
                resolved_n_predict = available_tokens

            gen_kwargs["n_predict"] = resolved_n_predict

            # Auto Temperature resolution:
            # If temperature is None or auto, dynamically adapt: 0.15 for code/tools/patches, 0.7 for conversation
            if active_temperature is None:
                has_code_intent = bool(ss.artifact_trigger or ss.tool_trigger or "<artifact" in raw_llm_output_buffer or "<tool" in raw_llm_output_buffer)
                gen_kwargs["temperature"] = 0.15 if has_code_intent else 0.7
            else:
                gen_kwargs["temperature"] = active_temperature
            if think is not None:
                gen_kwargs["think"] = think
            if reasoning_effort is not None:
                gen_kwargs["reasoning_effort"] = reasoning_effort
            if reasoning_summary is not None:
                gen_kwargs["reasoning_summary"] = reasoning_summary

            if dynamic_effort:
                gen_kwargs["reasoning_effort"] = active_reasoning_effort
                gen_kwargs["think"] = active_think_flag

            ASCIIColors.info(
                f"[LollmsPersonality.chat] Round {round_count}: think={gen_kwargs.get('think')}, "
                f"reasoning_effort={gen_kwargs.get('reasoning_effort')}"
            )

            _max_retries = 2
            _retry_delay = 1.0
            _generation_succeeded = False
            round_connection_error = None

            for _retry_attempt in range(_max_retries):
                try:
                    gen_res = self.lollms_client.generate_from_messages(
                        messages=messages,
                        stream=True,
                        streaming_callback=_inline_relay,
                        **gen_kwargs
                    )
                    if isinstance(gen_res, dict):
                        is_conn_dict, conn_reason_dict = _core_is_connection_or_server_error(gen_res)
                        if is_conn_dict:
                            round_connection_error = conn_reason_dict
                            raise ConnectionError(conn_reason_dict)

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

                    is_conn, conn_reason = _core_is_connection_or_server_error(gen_err)
                    if is_conn:
                        round_connection_error = conn_reason

                    ss.completed_actions = []
                    ss._is_accumulating_tool = False
                    ss._is_accumulating_artifact = False
                    ss._is_accumulating_context = False
                    ss._tool_buffer = ""
                    ss._pending_buffer = ""

                    if is_conn and _retry_attempt < _max_retries - 1:
                        ASCIIColors.warning(f"[{self.name}] Transient network error during generation (attempt {_retry_attempt + 1}/{_max_retries}): {conn_reason}")
                        try:
                            import time as _time
                            _time.sleep(_retry_delay)
                        except Exception:
                            pass
                        _retry_delay *= 2
                        ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                        continue
                    else:
                        break

            if was_cancelled:
                break

            # ── 🛑 BROKEN SERVER & CONSECUTIVE CONNECTION ERROR CIRCUIT BREAKER ──
            if not _generation_succeeded or round_connection_error:
                if not round_connection_error:
                    is_conn_raw, conn_reason_raw = _core_is_connection_or_server_error(raw_llm_output_buffer)
                    if is_conn_raw:
                        round_connection_error = conn_reason_raw

                if round_connection_error:
                    consecutive_connection_errors += 1
                    last_connection_error_desc = round_connection_error
                    ASCIIColors.warning(
                        f"[{self.name}] ⚠️ LLM Server connection failure on round {round_count} "
                        f"(consecutive failures: {consecutive_connection_errors}/3): {round_connection_error}"
                    )

                    if consecutive_connection_errors >= 3:
                        ASCIIColors.error(
                            f"[{self.name}] 🛑 Server unreachable for 3 consecutive rounds ({last_connection_error_desc}). "
                            f"Aborting agentic loop immediately to preserve round budget."
                        )
                        final_response = (
                            f"❌ **Server Connection Failure**: Unable to communicate with the LLM backend after 3 consecutive failed attempts.\n\n"
                            f"**Details**: {last_connection_error_desc}\n\n"
                            f"**Action Required**: The LLM server is unresponsive or disconnected. "
                            f"Please check that your backend engine (Ollama, vLLM, OpenAI, LoLLMs, etc.) is running, reachable, and has sufficient memory."
                        )
                        if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                            try:
                                streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                                    "round_id": round_count,
                                    "status": "connection_error"
                                })
                            except Exception:
                                pass
                        break

                    try:
                        import time as _time
                        _time.sleep(2.0)
                    except Exception:
                        pass
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content=f"[SYSTEM: Server connection error on round {round_count} ({last_connection_error_desc}). Retrying attempt {consecutive_connection_errors + 1}/3...]"
                    ))
                    continue
                else:
                    ASCIIColors.error(f"[{self.name}] Generation failed on round {round_count}.")
                    final_response = "[Generation error: The LLM failed to produce a response.]"
                    break
            else:
                consecutive_connection_errors = 0

            if self.is_generation_cancelled():
                was_cancelled = True
                if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                    try:
                        streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                            "round_id": round_count,
                            "status": "cancelled"
                        })
                    except Exception:
                        pass
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

            if is_greeting and not enforce_end_tag:
                # 🛡️ GREETING IMMUNITY SHIELD: Completely discard any unprompted tool calls or file writes
                if ss.completed_actions:
                    ASCIIColors.warning(f"[{self.name}] 🛡️ Blocked {len(ss.completed_actions)} unprompted hallucinated action(s) on greeting '{cleaned_prompt}'.")
                    ss.completed_actions = []
                final_response = "Hello! How can I help you with your project today?"
                if streaming_callback:
                    streaming_callback(final_response, MSG_TYPE.MSG_TYPE_CHUNK, {})
                break

            if ss.was_done_detected():
                final_response = re.sub(r'(?i)</?(?:done|end)\s*/?>', '', ss.get_clean_text()).strip()

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

                                # ── 🛑 FAILURE LOOP GUARD (PRE-EXECUTION) ──
                                fail_count = failed_tool_signatures.get(context_aware_sig, 0)
                                if fail_count >= 2:
                                    rep_msg = (
                                        f"🛑 BLOCKED: Tool '{tool_name}' with identical parameters has already failed {fail_count} times in this turn. "
                                        f"Execution was blocked to prevent an infinite loop. "
                                        f"Do NOT call this tool again with the same parameters. Adapt your approach or inform the user."
                                    )
                                    action_reports.append(rep_msg)
                                    if (event_mode.has_callbacks or event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE)) and streaming_callback:
                                        try:
                                            streaming_callback("", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                                "tool_name": tool_name,
                                                "parameters": tool_params,
                                                "success": False,
                                                "output": None,
                                                "error": rep_msg
                                            })
                                        except Exception:
                                            pass
                                    continue

                                if (event_mode.has_callbacks or event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE)) and streaming_callback:
                                    try:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_TOOL_START, {"tool_name": tool_name, "parameters": tool_params})
                                    except Exception as ex:
                                        ASCIIColors.warning(f"Failed to emit tool start: {ex}")

                                tool_res = self._execute_tool(tool_name, tool_params, active_tools)

                                vlm_images = self._inject_tool_images_for_vlm(tool_res)
                                if vlm_images:
                                    if not hasattr(self, '_pending_vlm_images'):
                                        object.__setattr__(self, '_pending_vlm_images', [])
                                    self._pending_vlm_images.extend(vlm_images)

                                inner_res = tool_res.get("output", tool_res) if isinstance(tool_res, dict) else tool_res
                                is_failure = (
                                    (isinstance(tool_res, dict) and tool_res.get("success") is False)
                                    or (isinstance(inner_res, dict) and inner_res.get("success") is False)
                                    or (isinstance(tool_res, dict) and tool_res.get("status_code", 200) not in (200, 201))
                                    or (isinstance(inner_res, dict) and inner_res.get("status_code", 200) not in (200, 201))
                                    or (isinstance(tool_res, dict) and bool(tool_res.get("error")))
                                    or (isinstance(inner_res, dict) and bool(inner_res.get("error")) and not inner_res.get("success", True))
                                    or (isinstance(tool_res, dict) and tool_res.get("return_code") is not None and tool_res.get("return_code") != 0)
                                    or (isinstance(inner_res, dict) and inner_res.get("return_code") is not None and inner_res.get("return_code") != 0)
                                )
                                tool_success = not is_failure

                                if tool_success:
                                    successful_tool_signatures.add(context_aware_sig)
                                    if context_aware_sig in failed_tool_signatures:
                                        del failed_tool_signatures[context_aware_sig]
                                else:
                                    failed_tool_signatures[context_aware_sig] = failed_tool_signatures.get(context_aware_sig, 0) + 1
                                    if self._failure_memory:
                                        try:
                                            if hasattr(self._failure_memory, "record_failure_by_signature"):
                                                self._failure_memory.record_failure_by_signature(context_aware_sig, str(tool_res.get("error", "")))
                                            elif hasattr(self._failure_memory, "_signatures"):
                                                self._failure_memory._signatures.add(context_aware_sig)
                                        except Exception:
                                            pass

                                if tool_success and tool_name in ("tool_create_skill", "tool_update_skill"):
                                    if isinstance(tool_res, dict):
                                        output_vis = tool_params.get("output_visibility", "context").lower()
                                        skill_title = tool_params.get("title", tool_params.get("skill_name", ""))
                                        if output_vis == "artefact" and skill_title:
                                            try:
                                                from lollms_client.lollms_artefact import ArtefactVisibility
                                                self._execute_context_visibility("lock_file", f"skills/{skill_title}.md")
                                                tool_res["output"] = tool_res.get("output", "") + f"\n\n[SYSTEM: Skill '{skill_title}' saved as an artefact [U]. It will not consume context space until you explicitly unlock it.]"
                                            except Exception:
                                                pass

                                tool_calls_this_turn.append({"round": round_count, "name": tool_name, "parameters": tool_params})
                                tool_results_this_turn.append({"round": round_count, "name": tool_name, "result": tool_res, "success": tool_success})
                                clean_result_str = _sanitize_tool_result(tool_res, client=self.lollms_client)

                                if tool_success and tool_name == "tool_load_skill":
                                    report_part = (
                                        f"=== ✅ TOOL RESULT: {tool_name} ===\n"
                                        f"<tool_result name=\"{tool_name}\" status=\"SUCCESS\">\n{clean_result_str}\n</tool_result>\n\n"
                                        f"[SYSTEM DIRECTIVE: The skill methodology has been loaded into your context above. "
                                        f"You have fulfilled the Skill-First mandate. DO NOT call tool_load_skill again. "
                                        f"You MUST now proceed directly to executing Phase 1 of the skill protocol (scan the workspace and emit `<artifact name=\"classes.md\">` and `<artifact name=\"mapping.md\">`).]"
                                    )
                                elif tool_success:
                                    report_part = f"=== ✅ TOOL RESULT: {tool_name} ===\n<tool_result name=\"{tool_name}\" status=\"SUCCESS\">\n{clean_result_str}\n</tool_result>"
                                else:
                                    report_part = f"=== ❌ TOOL FAILED: {tool_name} ===\n<tool_result name=\"{tool_name}\" status=\"FAILED\">\n{clean_result_str}\n</tool_result>\n\n⚠️ **Error Analysis Guidance:** Read the error details above carefully to understand what failed. Fix the parameters or try an alternative approach."

                                action_reports.append(report_part)

                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                    try:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                            "tool_name": tool_name,
                                            "parameters": tool_params,
                                            "success": tool_success,
                                            "output": clean_result_str if tool_success else None,
                                            "error": None if tool_success else clean_result_str
                                        })
                                    except Exception as ex:
                                        ASCIIColors.warning(f"Failed to emit tool end: {ex}")
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
                                    action_reports.append("❌ TRUNCATED ARTIFACT REJECTED. Missing closing tag. Retry generation.")
                                    continue

                                body_content = body_match.group(1).strip()

                                title = "artifact"
                                lang = "python"
                                operation_type = "full_rewrite"
                                resolved_art_type = "code"
                                for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                                    if m.group(1).lower() in ("name", "title"):
                                        title = m.group(2)
                                    elif m.group(1).lower() == "language":
                                        lang = m.group(2)
                                    elif m.group(1).lower() == "operation":
                                        operation_type = m.group(2).lower()
                                    elif m.group(1).lower() == "type":
                                        resolved_art_type = m.group(2).lower()

                                has_patch_markers = bool(
                                    re.search(r'^\s*<{5,10}\s*SEARCH\b', body_content, re.MULTILINE | re.IGNORECASE)
                                    or re.search(r'^\s*={5,10}\s*$', body_content, re.MULTILINE)
                                )
                                is_patch = ("<<<<<<< SEARCH" in body_content) or has_patch_markers
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
                                        has_truncated_artifact = True
                                        truncated_artifact_title = title
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
                                        fail_msg = (
                                            f"❌ SEARCH/REPLACE FAILED for '{title}'. Error: {patch_err}\n"
                                            f"⚠️ CRITICAL LOOP RECOVERY INSTRUCTION:\n"
                                            f"The lines in your SEARCH block did not match the file on disk.\n"
                                            f"DO NOT retry with another guessed SEARCH block!\n"
                                            f"Instead, emit a FULL FILE REWRITE using standard <artifact> syntax:\n"
                                            f'<artifact name="{title}" type="{resolved_art_type}">\n'
                                            f"[Write the complete updated file content from line 1 to end]\n"
                                            f"</artifact>\n"
                                        )
                                        action_reports.append(fail_msg)
                                        has_truncated_artifact = True
                                        truncated_artifact_title = title
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
                                    try:
                                        self._execute_context_visibility("lock_file", title)
                                    except Exception:
                                        pass
                                else:
                                    stripped_body = body_content.strip()
                                    if not stripped_body:
                                        action_reports.append(f"❌ FILE WRITE BLOCKED for {title}. Empty artifact body.")
                                        continue

                                    if has_patch_markers:
                                        action_reports.append(f"❌ FILE WRITE BLOCKED for {title}. Content contains raw patch markers (=======).")
                                        continue

                                    if self._artefact_manager:
                                        self._artefact_manager.add(title=title, artefact_type=resolved_art_type, content=body_content, language=lang, active=True)
                                    file_path = self._resolved_workspace / title
                                    file_path.parent.mkdir(parents=True, exist_ok=True)
                                    file_path.write_text(body_content, encoding="utf-8")
                                    action_reports.append(f"✅ File {title} created/updated successfully.")
                                    actions_executed_count += 1

                                    if (event_mode.has_callbacks or event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE)) and streaming_callback:
                                        try:
                                            streaming_callback("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END, {
                                                "title": title,
                                                "art_type": resolved_art_type,
                                                "language": lang,
                                                "version": 1,
                                                "is_patch": False,
                                                "operation": "create",
                                                "content": body_content,
                                                "success": True,
                                                "error": None
                                            })
                                        except Exception:
                                            pass
                            except Exception as e:
                                if getattr(self, 'debug_mode', False):
                                    self._dump_error(
                                        error=e,
                                        context_desc="Artifact Processing Error",
                                        round_count=round_count,
                                        extra_data={"raw_xml": raw_artifact_xml}
                                    )
                                action_reports.append(f"[SYSTEM ERROR] Failed to process artifact tag: {e}")

                        elif action["type"] == "malformed_json":
                            raw_body = action.get("raw_body", "")
                            fail_msg = (
                                f"❌ MALFORMED TOOL CALL: The tool payload could not be parsed as valid JSON.\n"
                                f"Raw payload:\n```\n{raw_body[:400]}\n```\n"
                                "You MUST use valid JSON inside the tool tag: `<tool>{\"name\": \"...\", \"parameters\": {...}}</tool>`."
                            )
                            action_reports.append(fail_msg)
                            if (event_mode.has_callbacks or event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE)) and streaming_callback:
                                try:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                        "tool_name": "malformed_tool_call",
                                        "parameters": {},
                                        "success": False,
                                        "output": None,
                                        "error": fail_msg
                                    })
                                except Exception:
                                    pass

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
                                if tag_name in ("mem_new", "mem_update", "mem_load", "mem_delete", "mem_search", "mem_tag"):
                                    if not self.memory_manager:
                                        action_reports.append("[SYSTEM ERROR] Memory manager not initialized.")
                                        continue
                                    if tag_name == "mem_new":
                                        content_match = re.search(r'content="([^"]*)"', raw_xml) or re.search(r"content='([^']*)'", raw_xml)
                                        tags_match = re.search(r'tags="([^"]*)"', raw_xml) or re.search(r"tags='([^']*)'", raw_xml)
                                        level_match = re.search(r'level="([^"]*)"', raw_xml) or re.search(r"level='([^']*)'", raw_xml)
                                        body_match = re.search(r'<mem_new[^>]*>(.*?)</mem_new>', raw_xml, re.DOTALL | re.IGNORECASE)
                                        mem_content = content_match.group(1) if content_match else (body_match.group(1).strip() if body_match else "")
                                        mem_tags = tags_match.group(1).split(",") if tags_match else []
                                        mem_level = int(level_match.group(1)) if level_match else 1
                                        if mem_content:
                                            self.memory_manager.add(content=mem_content, tags=mem_tags, importance=0.85, level=mem_level)
                                            action_reports.append(f"✅ Memory saved successfully: {mem_content[:50]}...")
                                        else:
                                            action_reports.append("⚠️ Memory tag had empty content.")
                                    elif tag_name == "mem_update":
                                        id_match = re.search(r'id="([^"]*)"', raw_xml) or re.search(r"id='([^']*)'", raw_xml)
                                        content_match = re.search(r'content="([^"]*)"', raw_xml) or re.search(r"content='([^']*)'", raw_xml)
                                        body_match = re.search(r'<mem_update[^>]*>(.*?)</mem_update>', raw_xml, re.DOTALL | re.IGNORECASE)
                                        mem_id = id_match.group(1) if id_match else ""
                                        mem_content = content_match.group(1) if content_match else (body_match.group(1).strip() if body_match else "")
                                        if mem_id and mem_content:
                                            self.memory_manager.update(memory_id=mem_id, content=mem_content)
                                            action_reports.append(f"✅ Memory updated successfully: [{mem_id[:8]}]")
                                    elif tag_name == "mem_load":
                                        id_match = re.search(r'id="([^"]*)"', raw_xml) or re.search(r"id='([^']*)'", raw_xml)
                                        mem_id = id_match.group(1) if id_match else ""
                                        if mem_id:
                                            full_id = getattr(self.memory_manager, "_resolve_id", lambda x: x)(mem_id) or mem_id
                                            res = self.memory_manager.load_to_working(full_id)
                                            if res:
                                                action_reports.append(f"✅ Loaded memory [{res['id'][:8]}] into Working Memory: {res.get('content', '')}")
                                            else:
                                                action_reports.append(f"❌ Memory [{mem_id}] not found in Deep Memory.")
                                    elif tag_name == "mem_delete":
                                        id_match = re.search(r'id="([^"]*)"', raw_xml) or re.search(r"id='([^']*)'", raw_xml)
                                        mem_id = id_match.group(1) if id_match else ""
                                        if mem_id:
                                            full_id = getattr(self.memory_manager, "_resolve_id", lambda x: x)(mem_id) or mem_id
                                            self.memory_manager.delete(full_id)
                                            action_reports.append(f"✅ Archived/deleted memory [{mem_id[:8]}].")
                                    elif tag_name == "mem_search":
                                        q_match = re.search(r'query="([^"]*)"', raw_xml) or re.search(r"query='([^']*)'", raw_xml)
                                        search_q = q_match.group(1) if q_match else ""
                                        if search_q:
                                            results = self.memory_manager.query(text=search_q, top_k=5)
                                            if results:
                                                hits = [f"[{r['id'][:8]}] (Imp: {r.get('importance', 0):.0%}) {r.get('content', '')}" for r in results]
                                                action_reports.append(f"🔍 Memory Search Results for '{search_q}':\n" + "\n".join(hits))
                                            else:
                                                action_reports.append(f"🔍 No memories found matching '{search_q}'.")
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

                    if action_reports:
                        report_text = "\n\n".join(str(r) for r in action_reports) + "\n\nAnalyze these results and continue your task, or emit <done/> if finished."
                        virtual_history.append(SimpleNamespace(sender_type="user", content=report_text))

                    # Reset stall tracking since concrete actions were executed
                    object.__setattr__(self, '_consecutive_stall_count', 0)
                    active_temperature = base_temperature

                    has_tool_actions = any(action.get("type") == "tool" for action in ss.completed_actions)
                    ss.completed_actions = []

                    # If tool actions were executed, continue to the next round to let the agent use the results
                    if has_tool_actions and action_reports:
                        ss = _AgentStreamState(
                            callback=streaming_callback,
                            event_mode=event_mode,
                            workspace_path=self._resolved_workspace
                        )
                        continue

                    final_response = re.sub(r'(?i)</?(?:done|end)\s*/?>', '', ss.get_clean_text()).strip()
                    if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                        try:
                            streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                                "round_id": round_count,
                                "status": "done"
                            })
                        except Exception:
                            pass
                    break

                if has_truncated_artifact:
                    consecutive_artifact_failures = getattr(self, '_consecutive_artifact_failures', 0) + 1
                    object.__setattr__(self, '_consecutive_artifact_failures', consecutive_artifact_failures)

                    if consecutive_artifact_failures >= 3:
                        ASCIIColors.error(f"[{self.name}] Breaking after {consecutive_artifact_failures} consecutive artifact failures (truncation/empty patch).")
                        final_response = f"[Task terminated: The agent repeatedly failed to generate a valid artifact for '{truncated_artifact_title}'. Check token limits or patch syntax.]"
                        if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                            try:
                                streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                                    "round_id": round_count,
                                    "status": "loop_break"
                                })
                            except Exception:
                                pass
                        break

                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content=(
                            f"[SYSTEM: CRITICAL ERROR. Your previous generation of '{truncated_artifact_title}' was TRUNCATED or FAILED. "
                            "The file was NOT saved. You MUST rewrite the COMPLETE file from scratch using a standard <artifact> tag (NOT a SEARCH/REPLACE patch). "
                            "Reproduce the existing content exactly and append the missing ending. Do NOT emit `<done/>` until the file is complete.]"
                        )
                    ))
                    ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                    continue
                object.__setattr__(self, '_consecutive_artifact_failures', 0)

                if not ss.completed_actions and not tool_calls_this_turn and not workspace_changes and not ss.was_done_detected() and round_count == 1:
                    pass

                if not final_response.strip():
                    for vh in reversed(virtual_history):
                        if getattr(vh, "sender_type", "") == "assistant" and getattr(vh, "content", "").strip():
                            recovered = re.sub(r'<[^>]+>', '', vh.content).strip()
                            if recovered and not _is_synthetic_agent_response(recovered):
                                final_response = recovered
                                ASCIIColors.info(f"[{self.name}] Recovered conversational response from previous round assistant text.")
                                break

                if not final_response.strip():
                    clean_input = cleaned_prompt.strip().lower()
                    is_greeting = clean_input in (
                        "hi", "hello", "hey", "salut", "bonjour", "coucou", "yo",
                        "good morning", "good evening", "good afternoon", "greetings",
                        "how are you", "ca va", "comment ca va", "sup"
                    ) or len(clean_input) < 4

                    if is_greeting:
                        final_response = "Hello! How can I help you with your project today?"
                        ASCIIColors.info(f"[{self.name}] Synthesized friendly greeting for '{cleaned_prompt}'.")
                        if streaming_callback:
                            streaming_callback(final_response, MSG_TYPE.MSG_TYPE_CHUNK, {})
                    elif round_count == 1:
                        ASCIIColors.warning(f"[{self.name}] 🚫 Empty response with <done/> detected on round 1. Forcing continuation.")
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content="[SYSTEM: You emitted `<done/>` without providing any answer or conversational response to the user. You MUST answer the user's message in conversational text first before emitting `<done/>`.]"
                        ))
                        ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                        continue
                    else:
                        if getattr(self, 'debug_mode', False):
                            self._dump_error(
                                error=Exception("Empty clean text after <done/>"),
                                context_desc="Empty Response After Done",
                                round_count=round_count,
                                extra_data={
                                    "raw_stream_chars": len(raw_llm_output_buffer or ""),
                                    "think_buffer_chars": len(getattr(ss, '_think_buffer', '') or ""),
                                    "pending_buffer_chars": len(getattr(ss, '_pending_buffer', '') or ""),
                                    "in_think_block": bool(getattr(ss, '_in_think_block', False)),
                                    "raw_stream_tail": (raw_llm_output_buffer or "")[-2000:],
                                }
                            )
                        ASCIIColors.warning(f"[{self.name}] Empty response after <done/> with no prior actions. Terminating.")
                        final_response = "[Task terminated: The agent produced no actionable output.]"
                        if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                            try:
                                streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                                    "round_id": round_count,
                                    "status": "text_stall"
                                })
                            except Exception:
                                pass
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
                if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                    try:
                        streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                            "round_id": round_count,
                            "status": "done"
                        })
                    except Exception:
                        pass
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

                            vlm_images = self._inject_tool_images_for_vlm(tool_res)
                            if vlm_images:
                                if not hasattr(self, '_pending_vlm_images'):
                                    object.__setattr__(self, '_pending_vlm_images', [])
                                self._pending_vlm_images.extend(vlm_images)

                            if isinstance(tool_res, dict) and tool_res.get("success") is False and not tool_res.get("error"):
                                raw_preview = ""
                                for k, v in tool_res.items():
                                    if k not in ("error", "traceback", "success"):
                                        raw_preview += f"  {k}: {str(v)[:300]}\n"
                                tool_res["error"] = (
                                    f"Tool '{tool_name}' returned success=False with no error message. "
                                    f"Raw keys: {list(tool_res.keys())}.\n"
                                    f"Raw content:\n{raw_preview}"
                                    if raw_preview else
                                    f"Tool '{tool_name}' returned success=False with no error message. "
                                    f"Raw keys: {list(tool_res.keys())}."
                                )
                                ASCIIColors.error(f"[{self.name}] Tool '{tool_name}' returned bare success=False. Synthesized error: {tool_res['error']}")

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

                            if tool_success:
                                successful_tool_signatures.add(context_aware_sig)
                                if context_aware_sig in failed_tool_signatures:
                                    del failed_tool_signatures[context_aware_sig]
                            else:
                                failed_tool_signatures[context_aware_sig] = failed_tool_signatures.get(context_aware_sig, 0) + 1
                                if self._failure_memory:
                                    try:
                                        if hasattr(self._failure_memory, "record_failure_by_signature"):
                                            self._failure_memory.record_failure_by_signature(context_aware_sig, str(tool_res.get("error", "")))
                                        elif hasattr(self._failure_memory, "_signatures"):
                                            self._failure_memory._signatures.add(context_aware_sig)
                                    except Exception:
                                        pass

                            if tool_success:
                                report_part = f"=== ✅ TOOL RESULT: {tool_name} ===\n<tool_result name=\"{tool_name}\" status=\"SUCCESS\">\n{clean_result_str}\n</tool_result>"
                            else:
                                report_part = f"=== ❌ TOOL FAILED: {tool_name} ===\n<tool_result name=\"{tool_name}\" status=\"FAILED\">\n{clean_result_str}\n</tool_result>\n\n⚠️ **Error Analysis Guidance:** Read the error details above carefully to understand what failed. Fix the parameters or try an alternative approach."

                            action_reports.append(report_part)

                            if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                try:
                                    streaming_callback("", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                        "tool_name": tool_name,
                                        "parameters": tool_params,
                                        "success": tool_success,
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
                                action_reports.append("❌ TRUNCATED ARTIFACT REJECTED. Missing closing tag. Retry generation.")
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
                                    if (event_mode.has_callbacks or event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE)) and streaming_callback:
                                        try:
                                            streaming_callback("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END, {
                                                "title": title,
                                                "art_type": resolved_art_type,
                                                "language": lang,
                                                "version": 1,
                                                "is_patch": True,
                                                "operation": "patch",
                                                "content": patched_content,
                                                "success": True,
                                                "error": None
                                            })
                                        except Exception:
                                            pass
                                except Exception as patch_err:
                                    if getattr(self, 'debug_mode', False):
                                        self._dump_error(
                                            error=patch_err,
                                            context_desc="Artifact Patch Error (Block 2)",
                                            round_count=round_count,
                                            extra_data={"title": title, "original_length": len(original_content), "patch_body": body_content[:500]}
                                        )
                                    fail_msg = (
                                        f"❌ SEARCH/REPLACE FAILED for '{title}'. Error: {patch_err}\n"
                                        f"⚠️ CRITICAL LOOP RECOVERY INSTRUCTION:\n"
                                        f"The lines in your SEARCH block did not match the file on disk.\n"
                                        f"DO NOT retry with another guessed SEARCH block!\n"
                                        f"Instead, emit a FULL FILE REWRITE using standard <artifact> syntax:\n"
                                        f'<artifact name="{title}" type="{resolved_art_type}">\n'
                                        f"[Write the complete updated file content from line 1 to end]\n"
                                        f"</artifact>\n"
                                    )
                                    action_reports.append(fail_msg)
                                    if (event_mode.has_callbacks or event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE)) and streaming_callback:
                                        try:
                                            streaming_callback("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END, {
                                                "title": title,
                                                "art_type": resolved_art_type,
                                                "language": lang,
                                                "version": 1,
                                                "is_patch": True,
                                                "operation": "patch",
                                                "content": body_content,
                                                "success": False,
                                                "error": str(patch_err)
                                            })
                                        except Exception:
                                            pass
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
                                try:
                                    self._execute_context_visibility("lock_file", title)
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

                                if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE) and streaming_callback:
                                    try:
                                        streaming_callback("", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END, {
                                            "title": title,
                                            "art_type": resolved_art_type,
                                            "language": lang,
                                            "version": 1,
                                            "is_patch": False,
                                            "operation": "create",
                                            "content": body_content,
                                            "success": True,
                                            "error": None
                                        })
                                    except Exception:
                                        pass

                                try:
                                    self._execute_context_visibility("lock_file", title)
                                except Exception:
                                    pass
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

                            if tag_name in ("generate_image", "edit_image"):
                                prompt_match = re.search(r'prompt\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                body_match = re.search(r'<(?:generate_image|edit_image)[^>]*>(.*?)</(?:generate_image|edit_image)>', raw_xml, re.DOTALL | re.IGNORECASE)
                                img_prompt = prompt_match.group(1) if prompt_match else (body_match.group(1).strip() if body_match else "")
                                img_file_match = re.search(r'name\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                img_file = img_file_match.group(1) if img_file_match else ""

                                tti_tool_def = active_tools.get("tool_generate_image") if tag_name == "generate_image" else active_tools.get("tool_edit_image")
                                if tti_tool_def and "callable" in tti_tool_def:
                                    if tag_name == "generate_image":
                                        img_res = tti_tool_def["callable"](prompt=img_prompt)
                                    else:
                                        img_res = tti_tool_def["callable"](prompt=img_prompt, image_file_name=img_file)
                                    out_msg = img_res.get("output") or img_res.get("error") or "Image processed."
                                    action_reports.append(f"🎨 Image Generation: {out_msg}")
                                    actions_executed_count += 1
                                    continue
                                else:
                                    action_reports.append(f"❌ TTI (Image Generation) binding is not available in current configuration.")
                                    continue

                            if tag_name in ("mem_new", "mem_update", "mem_load", "mem_delete", "mem_search", "mem_tag"):
                                if not self.memory_manager:
                                    err_msg = "Memory manager not initialized."
                                    action_reports.append(f"[SYSTEM ERROR] {err_msg}")
                                    continue

                                if tag_name == "mem_new":
                                    content_match = re.search(r'content\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    tags_match = re.search(r'tags\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    level_match = re.search(r'level\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    body_match = re.search(r'<mem_new[^>]*>(.*?)</mem_new>', raw_xml, re.DOTALL | re.IGNORECASE)
                                    mem_content = content_match.group(1) if content_match else (body_match.group(1).strip() if body_match else "")
                                    mem_tags = tags_match.group(1).split(",") if tags_match else []
                                    mem_level = int(level_match.group(1)) if level_match else 1
                                    if mem_content:
                                        self.memory_manager.add(content=mem_content, tags=mem_tags, importance=0.85, level=mem_level)
                                        res_msg = f"✅ Memory saved successfully: {mem_content[:50]}..."
                                    else:
                                        res_msg = "⚠️ Memory tag received with empty content."
                                    action_reports.append(res_msg)
                                    actions_executed_count += 1
                                elif tag_name == "mem_update":
                                    id_match = re.search(r'id\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    content_match = re.search(r'content\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    body_match = re.search(r'<mem_update[^>]*>(.*?)</mem_update>', raw_xml, re.DOTALL | re.IGNORECASE)
                                    mem_id = id_match.group(1) if id_match else ""
                                    mem_content = content_match.group(1) if content_match else (body_match.group(1).strip() if body_match else "")
                                    if mem_id and mem_content:
                                        self.memory_manager.update(memory_id=mem_id, content=mem_content)
                                        res_msg = f"✅ Memory updated successfully: [{mem_id[:8]}]"
                                    else:
                                        res_msg = "⚠️ Memory update received with missing id or content."
                                    action_reports.append(res_msg)
                                    actions_executed_count += 1
                                elif tag_name == "mem_load":
                                    id_match = re.search(r'id\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    mem_id = id_match.group(1) if id_match else ""
                                    if mem_id:
                                        full_id = getattr(self.memory_manager, "_resolve_id", lambda x: x)(mem_id) or mem_id
                                        res = self.memory_manager.load_to_working(full_id)
                                        if res:
                                            action_reports.append(f"✅ Loaded memory [{res['id'][:8]}] into Working Memory: {res.get('content', '')}")
                                        else:
                                            action_reports.append(f"❌ Memory [{mem_id}] not found in Deep Memory.")
                                        actions_executed_count += 1
                                elif tag_name == "mem_delete":
                                    id_match = re.search(r'id\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    mem_id = id_match.group(1) if id_match else ""
                                    if mem_id:
                                        full_id = getattr(self.memory_manager, "_resolve_id", lambda x: x)(mem_id) or mem_id
                                        self.memory_manager.delete(full_id)
                                        action_reports.append(f"✅ Archived/deleted memory [{mem_id[:8]}].")
                                        actions_executed_count += 1
                                elif tag_name == "mem_search":
                                    q_match = re.search(r'query\s*=\s*["\']([^"\']*)["\']', raw_xml, re.IGNORECASE)
                                    search_q = q_match.group(1) if q_match else ""
                                    if search_q:
                                        results = self.memory_manager.query(text=search_q, top_k=5)
                                        if results:
                                            hits = [f"[{r['id'][:8]}] (Imp: {r.get('importance', 0):.0%}) {r.get('content', '')}" for r in results]
                                            action_reports.append(f"🔍 Memory Search Results for '{search_q}':\n" + "\n".join(hits))
                                        else:
                                            action_reports.append(f"🔍 No memories found matching '{search_q}'.")
                                        actions_executed_count += 1
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

                # Reset stall tracking since actions were executed in this round
                object.__setattr__(self, '_consecutive_stall_count', 0)
                active_temperature = base_temperature
                ss.completed_actions = []
                ss = _AgentStreamState(
                    callback=streaming_callback,
                    event_mode=event_mode,
                    workspace_path=self._resolved_workspace
                )
                if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                    try:
                        streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                            "round_id": round_count,
                            "status": "action"
                        })
                    except Exception:
                        pass

                # ── 💾 CHECKPOINT SAVE: Persist status at round end ──
                self._save_round_checkpoint(
                    round_count=round_count,
                    prompt=prompt,
                    virtual_history=virtual_history,
                    tool_calls=tool_calls_this_turn,
                    tool_results=tool_results_this_turn,
                    workspace_changes=workspace_changes,
                    status="in_progress"
                )

                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Actions dispatched, checkpoint saved ===")
                continue

            # ── ⚡ UPDATE DYNAMIC EFFORT FOR NEXT ROUND ──
            if dynamic_effort and getattr(ss, "next_reasoning_effort", None) is not None:
                new_lvl = ss.next_reasoning_effort.lower().strip()
                if new_lvl in ("none", "off", "disabled", "false", "0"):
                    active_reasoning_effort = "none"
                    active_think_flag = False
                else:
                    active_reasoning_effort = new_lvl
                    active_think_flag = True
                ASCIIColors.info(f"[{self.name}] Dynamic reasoning effort set to '{active_reasoning_effort}' for next round.")
                ss.next_reasoning_effort = None

            if ss.was_done_detected() and not ss.completed_actions:
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: <done/> detected (no actions) ===")
                if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                    try:
                        streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                            "round_id": round_count,
                            "status": "done"
                        })
                    except Exception:
                        pass
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
                if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                    try:
                        streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                            "round_id": round_count,
                            "status": "done"
                        })
                    except Exception:
                        pass
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

            if ss.completed_actions and any(act.get("type") == "context" for act in ss.completed_actions):
                object.__setattr__(self, '_consecutive_stall_count', 0)
                text_is_repetitive = False
                if raw_round_text.strip():
                    virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
                ss.completed_actions = []
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                continue

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

            if not text_is_repetitive and stripped_round_text:
                lines_in_response = stripped_round_text.splitlines()
                non_empty_lines = [l.strip() for l in lines_in_response if l.strip()]
                if len(non_empty_lines) >= 4:
                    from collections import Counter as _Counter
                    line_counts = _Counter(non_empty_lines)
                    most_common_line, most_common_count = line_counts.most_common(1)[0]
                    # Only treat as true repetition if identical multi-sentence blocks are looping
                    if most_common_count >= 3 and len(most_common_line) > 30:
                        repetition_ratio = most_common_count / len(non_empty_lines)
                        if repetition_ratio >= 0.6:
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
                            ASCIIColors.warning(f"[{self.name}] Intra-round text duplication detected ({most_common_count}x). Deduplicated.")

            if text_is_repetitive:
                consecutive_stall_count = getattr(self, '_consecutive_stall_count', 0) + 1
                object.__setattr__(self, '_consecutive_stall_count', consecutive_stall_count)

                # Dynamically increase temperature to shake the greedy decoding out of the repetition trap
                active_temperature = min(0.95, base_temperature + (0.2 * consecutive_stall_count))
                ASCIIColors.warning(f"[{self.name}] Repetitive text preamble detected (Round {round_count}, streak: {consecutive_stall_count}/4). Shaking decoding temperature to {active_temperature:.2f}.")

                if consecutive_stall_count >= 4:
                    ASCIIColors.error(f"[{self.name}] Breaking after {consecutive_stall_count} consecutive repetition+stall cycle(s). Repetitive text detected.")
                    final_response = re.sub(r'(?i)<done\s*/?>', '', ss.get_clean_text()).strip()
                    if not final_response:
                        final_response = "[Task terminated: The agent produced repetitive text due to a tool failure or sandbox restriction. The last tool call may have been blocked.]"
                    if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                        try:
                            streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                                "round_id": round_count,
                                "status": "loop_break"
                            })
                        except Exception:
                            pass
                    break

                virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text()))
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=(
                        "[SYSTEM GUIDANCE: Please proceed directly with your task.\n"
                        "- If creating or updating files, emit a valid `<artifact name=\"filename.ext\" type=\"code\">...</artifact>` tag.\n"
                        "- If calling a tool, use `<tool>{\"name\": \"...\", \"parameters\": {...}}</tool>`.\n"
                        "- If awaiting user confirmation or replying conversationally, conclude your response with `<done/>`.\n"
                        "Do not apologize or output introductory apologies.]"
                    )
                ))
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                continue
            else:
                # Reset temperature back to base once repetition breaks
                active_temperature = base_temperature

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

            had_prior_actions = bool(tool_calls_this_turn or workspace_changes or len(virtual_history) > 0)

            # ── 🛡️ MID-TASK STALL INTERCEPTOR ──
            # If the model had prior actions (e.g. created a file in round 1), but in this round
            # outputted prose intent (e.g. "Now I'll execute it...") without emitting an action tag
            # or <done/>, it must NOT exit. Intercept and mandate action tag execution.
            if round_count > 1 and had_prior_actions and not was_cancelled and not has_new_actions_this_round and not ss.was_done_detected():
                consecutive_stall_count = getattr(self, '_consecutive_stall_count', 0) + 1
                object.__setattr__(self, '_consecutive_stall_count', consecutive_stall_count)

                active_temperature = min(0.95, base_temperature + (0.15 * consecutive_stall_count))

                if consecutive_stall_count >= 4:
                    ASCIIColors.warning(f"[{self.name}] Terminating after {consecutive_stall_count} consecutive stalls. The LLM is stuck in preamble mode.")
                    final_response = re.sub(r'(?i)<done\s*/?>', '', ss.get_clean_text()).strip()
                    if not final_response:
                        final_response = "[Task terminated: The agent stalled repeatedly without producing actionable output.]"
                    if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                        try:
                            streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                                "round_id": round_count,
                                "status": "text_stall"
                            })
                        except Exception:
                            pass
                    break

                ASCIIColors.warning(f"[{self.name}] Mid-task stall detected (Round {round_count}, consecutive: {consecutive_stall_count}). Adjusting temperature to {active_temperature:.2f} and forcing continuation.")
                virtual_history.append(SimpleNamespace(sender_type="assistant", content=ss.get_clean_text().strip()))
                recent_tool_names = [tc.get("name", "") for tc in tool_calls_this_turn[-3:]]
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=self._build_progressive_continuation_prompt(consecutive_stall_count, recent_tool_names)
                ))
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
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
                        if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                            try:
                                streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                                    "round_id": round_count,
                                    "status": "text_stall"
                                })
                            except Exception:
                                pass
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
            if (
                round_count == 1
                and not ss.was_done_detected()
                and not ss.was_action_dispatched()
                and not has_new_actions_this_round
                and not tool_calls_this_turn
                and stripped_round_text
                and not text_is_repetitive
                and bool(_INTENT_ANNOUNCEMENT_RE.search(stripped_round_text))
            ):
                ASCIIColors.info(f"[{self.name}] Round 1 action intent announcement without `<done/>` or action tag. Forcing action continuation.")
                virtual_history.append(SimpleNamespace(
                    sender_type="assistant",
                    content=ss.get_clean_text().strip()
                ))
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=(
                        "[SYSTEM DIRECTIVE: You stated an intent to perform an action but did not execute a tag or finish with `<done/>`.\n"
                        "Conversational text DOES NOT execute actions. You MUST emit the functional XML tag as the FIRST token of your reply:\n"
                        "- To generate an image: `<generate_image>detailed prompt</generate_image>` or `<tool>{\"name\": \"tool_generate_image\", \"parameters\": {\"prompt\": \"...\"}}</tool>`\n"
                        "- To create/edit code: `<artifact name=\"filename.ext\" type=\"code\">content</artifact>`\n"
                        "- To read files: `<unlock_file>filename</unlock_file>`\n"
                        "- To run commands: `<tool>{\"name\": \"tool_execute_shell_command\", \"parameters\": {\"command\": \"...\"}}</tool>`\n"
                        "DO NOT apologize. DO NOT write another introductory sentence. Output the XML tag NOW.]"
                    )
                ))
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
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
                        "[SYSTEM GUIDANCE: Ensure all functional tags use standard syntax:\n"
                        "- To call a tool: `<tool>{\"name\": \"...\", \"parameters\": {...}}</tool>`\n"
                        "- To write a file: `<artifact name=\"...\" type=\"...\">...</artifact>`\n"
                        "Do not apologize or explain formatting errors — output your response directly.]"
                    )
                ))
                if getattr(self, 'debug_mode', False):
                    ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Malformed tag detected, injecting format correction ===")
                continue

            # ── 🛑 ENFORCE END TAG MANDATE (UNIVERSAL TERMINATION CONTRACT) ──
            # Only when enforce_end_tag is FALSE can round 1 exit on a pure conversational greeting without intent
            if not enforce_end_tag:
                if (
                    round_count == 1
                    and not tool_calls_this_turn
                    and not ss.completed_actions
                    and not ss.was_action_dispatched()
                    and not ss.tool_trigger
                    and stripped_round_text
                ):
                    final_response = re.sub(r'(?i)</?(?:done|end)\s*/?>', '', stripped_round_text).strip()
                    if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                        try:
                            streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                                "round_id": round_count,
                                "status": "done"
                            })
                        except Exception:
                            pass
                    break

            if not ss.was_done_detected() and not was_cancelled:
                if not enforce_end_tag and round_count == 1 and not had_prior_actions and not has_new_actions_this_round:
                    final_response = stripped_round_text
                    break
                consecutive_stall_count = getattr(self, '_consecutive_stall_count', 0) + 1
                object.__setattr__(self, '_consecutive_stall_count', consecutive_stall_count)
                if consecutive_stall_count >= 5:
                    ASCIIColors.warning(f"[{self.name}] Terminating after {consecutive_stall_count} consecutive rounds without <done/> or <end/>.")
                    final_response = re.sub(r'(?i)</?(?:done|end)\s*/?>', '', ss.get_clean_text()).strip()
                    break

                ASCIIColors.warning(f"[{self.name}] No <done/> or <end/> detected (Round {round_count}, streak {consecutive_stall_count}/5). Enforcing continuation.")
                clean_round_text = ss.get_clean_text().strip()
                if clean_round_text:
                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=clean_round_text
                    ))
                recent_tool_names = [tc.get("name", "") for tc in tool_calls_this_turn[-3:]]
                recent_ctx = f" Recent actions executed: {recent_tool_names}." if recent_tool_names else ""
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=(
                        f"[SYSTEM DIRECTIVE: TERMINATION REQUIREMENT{recent_ctx}\n"
                        "Your previous response did NOT execute any action tag and did NOT emit `<done/>` or `<end/>`.\n"
                        "The conversation does NOT stop until you explicitly finish or take an action.\n"
                        "You have two options:\n"
                        "1. If you need to perform more work, emit the appropriate functional tag (`<tool>`, `<artifact>`, `<unlock_file>`, etc.) NOW.\n"
                        "2. If you have completely finished answering the user's request, provide your final response and you MUST append `<done/>` on a new line.\n"
                        "Do not output conversational preambles without action tags or `<done/>`.]"
                    )
                ))
                ss = _AgentStreamState(callback=streaming_callback, event_mode=event_mode)
                continue

            if not final_response:
                final_response = re.sub(r'(?i)</?(?:done|end)\s*/?>', '', ss.get_clean_text()).strip()

            if getattr(self, 'debug_mode', False):
                ASCIIColors.info(f"[{self.name}] 🐛 === ROUND {round_count} END: Clean exit with <done/> or <end/> ===")
            if streaming_callback and event_mode.has_callbacks and not event_mode.is_silent:
                try:
                    streaming_callback("", MSG_TYPE.MSG_TYPE_ROUND_END, {
                        "round_id": round_count,
                        "status": "done"
                    })
                except Exception:
                    pass
            break

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
                            action_reports.append("❌ TRUNCATED ARTIFACT REJECTED. Missing closing tag. Retry generation.")
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

                            try:
                                self._execute_context_visibility("lock_file", title)
                            except Exception:
                                pass

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

        if use_internal_history:
            # Recover assistant content even if interrupted/cancelled so the turn is never lost
            resp_to_save = final_response.strip() if final_response else ""
            if not resp_to_save:
                # Recover from virtual history (tool results and partial assistant responses)
                for vh in reversed(virtual_history):
                    if getattr(vh, "sender_type", "") == "assistant" and getattr(vh, "content", "").strip():
                        resp_to_save = getattr(vh, "content", "").strip()
                        break

            clean_persisted = re.sub(r'<think\b[^>]*>.*?(?:</think>|$)', '', resp_to_save, flags=re.DOTALL | re.IGNORECASE).strip()
            clean_persisted = re.sub(r'<thought\b[^>]*>.*?(?:</thought>|$)', '', clean_persisted, flags=re.DOTALL | re.IGNORECASE).strip()
            clean_persisted = re.sub(r'<round\s+id=["\'][^"\']*["\']\s*/?>\n?', '', clean_persisted, flags=re.IGNORECASE).strip()

            if was_cancelled:
                clean_persisted = (clean_persisted + "\n\n[⏹️ Generation stopped by user]").strip() if clean_persisted else "[⏹️ Generation stopped by user]"

            if clean_persisted and not _is_synthetic_agent_response(clean_persisted):
                self._conversation.append({"role": "user", "content": prompt})
                self._conversation.append({"role": "assistant", "content": clean_persisted})
                if hasattr(self, "_project_history_file") and self._project_history_file:
                    self.save_history_to_disk(self._project_history_file)

        object.__setattr__(self, '_compaction_triggered_this_turn', False)

        if self.memory_manager and not was_cancelled and consecutive_connection_errors < 3:
            final_resp_str = str(final_response) if final_response is not None else ""
            if not final_resp_str.startswith("[Generation error:") and "Server Connection Failure" not in final_resp_str:
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
                if max_ctx <= 0:
                    max_ctx = 8192

                total_used = self._count_tokens_cached(stable_system_prompt)
                if use_internal_history:
                    for msg in self._conversation:
                        total_used += self._count_tokens_cached(msg.get("content", ""))
                for vh in virtual_history:
                    content = getattr(vh, "content", "")
                    if content:
                        total_used += self._count_tokens_cached(content)
                if final_response:
                    total_used += self._count_tokens_cached(final_response)

                context_health = {
                    "used_tokens": total_used,
                    "max_tokens": max_ctx,
                    "fill_percentage": round((total_used / max_ctx) * 100, 1)
                }
        except Exception:
            pass

        self._reset_cancel_state()

        _has_tti = False
        if self.lollms_client:
            _has_tti = getattr(self.lollms_client, 'tti', None) is not None or bool(getattr(self.lollms_client, 'tti_model_profiles_registry', None))

        all_personality_artefacts = []
        if hasattr(self, '_artefact_manager') and self._artefact_manager:
            all_personality_artefacts = self._artefact_manager.list(active_only=True)

        # Persist final turn checkpoint
        self._save_round_checkpoint(
            round_count=round_count,
            prompt=prompt,
            virtual_history=virtual_history,
            tool_calls=tool_calls_this_turn,
            tool_results=tool_results_this_turn,
            workspace_changes=workspace_changes,
            status="completed" if not was_cancelled else "cancelled"
        )

        return {
           "response": final_response,
           "sources": collected_sources,
           "tool_calls": tool_calls_this_turn,
           "tool_results": tool_results_this_turn,
           "rounds": round_count,
           "workspace_changes": workspace_changes,
           "artefacts": all_personality_artefacts,
           "was_cancelled": was_cancelled,
           "context_health": context_health,
           "tti_available": _has_tti
       }

    def _get_checkpoint_path(self) -> Optional[Path]:
        if not self._resolved_workspace:
            return None
        return self._resolved_workspace / ".lollms_code" / "turn_checkpoint.json"

    def _save_round_checkpoint(
        self,
        round_count: int,
        prompt: str,
        virtual_history: List[Any],
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        workspace_changes: List[Dict[str, Any]],
        status: str = "in_progress"
    ) -> None:
        """Saves turn checkpoint to disk at each round end."""
        chk_path = self._get_checkpoint_path()
        if not chk_path:
            return
        try:
            chk_path.parent.mkdir(parents=True, exist_ok=True)
            vh_data = []
            for vh in virtual_history:
                vh_data.append({
                    "sender_type": getattr(vh, "sender_type", "user"),
                    "content": getattr(vh, "content", "")
                })
            checkpoint = {
                "round_count": round_count,
                "prompt": prompt,
                "status": status,
                "virtual_history": vh_data,
                "tool_calls": tool_calls,
                "tool_results": [
                    {"name": tr.get("name"), "success": tr.get("success")} for tr in tool_results
                ],
                "workspace_changes": workspace_changes,
                "timestamp": time.time()
            }
            object.__setattr__(self, "_active_turn_checkpoint", checkpoint)
            chk_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as ex:
            ASCIIColors.warning(f"[{self.name}] Failed to save round checkpoint: {ex}")

    def has_resumable_turn(self) -> bool:
        """Checks if a resumable turn checkpoint exists on disk or in memory."""
        chk_path = self._get_checkpoint_path()
        if not chk_path or not chk_path.exists():
            return False
        try:
            data = json.loads(chk_path.read_text(encoding="utf-8"))
            return data.get("status") in ("in_progress", "cancelled")
        except Exception:
            return False

    def load_turn_checkpoint(self) -> Optional[Dict[str, Any]]:
        chk_path = self._get_checkpoint_path()
        if not chk_path or not chk_path.exists():
            return None
        try:
            return json.loads(chk_path.read_text(encoding="utf-8"))
        except Exception:
            return None

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

is_tool_binding = _is_tool_binding
