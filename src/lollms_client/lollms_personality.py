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

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ascii_colors import ASCIIColors, trace_exception


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
    Self-contained, null-safe personality unit for lollms-client.

    Accepted forms for ``tools``
    ─────────────────────────────
    * ``None``                    → _NullToolBinding (no tools)
    * ``LollmsToolBinding``       → used directly
    * ``List[str]`` of MCP names  → stored in ``mcp_tool_names``; a binding
                                    can be attached later via ``attach_tool_binding()``

    Accepted forms for ``data_source``
    ────────────────────────────────────
    All three are normalised into ``query_data(query) → dict`` at init time:
    * ``None``                    → returns empty RAG dict
    * ``str``                     → returns that string as static context
    * ``callable(query) → any``   → wrapped and normalised

    ``query_rag_callback`` is still accepted for backward compatibility and
    used as a fallback when no ``data_source`` is given.
    """

    def __init__(
        self,
        # Identity
        name:        str,
        author:      str,
        category:    str,
        description: str,

        # Core behaviour
        system_prompt: str,

        icon:  Optional[str]       = None,

        # Tools — accepts None, LollmsToolBinding, or List[str] of MCP names
        tools: Optional[Any]       = None,

        # Data / RAG — accepts None, str, or callable
        data_source: Optional[Union[str, Callable[[str], Any]]] = None,

        # Legacy RAG callbacks
        data_files:               Optional[List[Union[str, Path]]] = None,
        vectorize_chunk_callback: Optional[Callable[[str, str], None]] = None,
        is_vectorized_callback:   Optional[Callable[[str], bool]]  = None,
        query_rag_callback:       Optional[Callable[[str], Any]]   = None,

        # Custom script
        script: Optional[str] = None,

        # Unique identifier
        personality_id: Optional[str] = None,
    ):
        # ── Identity ──────────────────────────────────────────────────────────
        self.name           = name or "assistant"
        self.author         = author or ""
        self.category       = category or "general"
        self.description    = description or ""
        self.icon           = icon
        self.system_prompt  = system_prompt or ""
        self.personality_id = personality_id or self._generate_id()

        # ── Tools ─────────────────────────────────────────────────────────────
        self.mcp_tool_names: List[str] = []
        self._tool_binding: Any        = _NULL_TOOL_BINDING
        self._init_tools(tools)

        # ── Data source ───────────────────────────────────────────────────────
        self._raw_data_source             = data_source
        self.data_files                   = [Path(f) for f in (data_files or [])]
        self.vectorize_chunk_callback     = vectorize_chunk_callback
        self.is_vectorized_callback       = is_vectorized_callback
        self.query_rag_callback           = query_rag_callback
        self._query_data_fn               = self._build_query_data_fn(data_source)

        # ── Script ────────────────────────────────────────────────────────────
        self.script        = script
        self.script_module = None
        self._prepare_script()

        # ── RAG pre-vectorisation ─────────────────────────────────────────────
        self.ensure_data_vectorized()

    # ------------------------------------------------------------------ tools

    def _init_tools(self, tools: Optional[Any]) -> None:
        if tools is None:
            self._tool_binding = _NULL_TOOL_BINDING
            return

        if _is_tool_binding(tools):
            self._tool_binding = tools
            return

        if isinstance(tools, list):
            self.mcp_tool_names = [str(t) for t in tools if t]
            self._tool_binding  = _NULL_TOOL_BINDING
            return

        ASCIIColors.warning(
            f"[{self.name}] Unsupported tools type {type(tools).__name__!r}. "
            "Expected LollmsToolBinding or List[str]. Falling back to null binding."
        )
        self._tool_binding = _NULL_TOOL_BINDING

    @property
    def tools(self) -> Any:
        """Always returns a binding-compatible object (never None)."""
        return self._tool_binding

    @tools.setter
    def tools(self, value: Optional[Any]) -> None:
        self._init_tools(value)

    def attach_tool_binding(self, binding: Any) -> None:
        """
        Attach a real LollmsToolBinding after construction.

        Useful when the personality was built with a List[str] of MCP names
        and the binding is only available after app initialisation.
        """
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
        """
        Resolve the personality's tool allowlist against the client's binding.

        If mcp_tool_names is empty → return all tools from the binding (no filter).
        If mcp_tool_names is set   → return only those tools.
        If no binding              → return {}.
        """
        binding = client_binding or self._tool_binding
        if not binding:
            return {}

        try:
            all_specs = binding.to_chat_tool_specs(**discover_kwargs)
        except Exception as exc:
            trace_exception(exc)
            return {}

        if not self.mcp_tool_names:
            return all_specs  # no filter — personality gets everything

        return {
            name: spec
            for name, spec in all_specs.items()
            if name in self.mcp_tool_names
        }

    # ------------------------------------------------------------------ data

    def _build_query_data_fn(
        self, source: Optional[Union[str, Callable]]
    ) -> Callable[[str], Dict[str, Any]]:
        """
        Build a unified ``(query: str) → normalised-RAG-dict`` function.

        Return schema:
            {
              "success":  bool,
              "sources":  [{"content": str, "score": float, "source": str}, …],
              "count":    int,
              "query":    str,
            }
        """
        def _empty(query: str) -> Dict[str, Any]:
            return {"success": False, "sources": [], "count": 0, "query": query}

        def _normalise_raw(raw: Any, query: str, source_label: str) -> Dict[str, Any]:
            """Convert any callable return value into the normalised dict."""
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

        # ── static string ────────────────────────────────────────────────────
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

        # ── callable data source ─────────────────────────────────────────────
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

        # ── legacy query_rag_callback ────────────────────────────────────────
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
        """
        Unified RAG interface. Always returns a normalised dict; never raises.

            {
              "success":  bool,
              "sources":  [{"content": str, "score": float, "source": str}, …],
              "count":    int,
              "query":    str,
            }
        """
        return self._query_data_fn(query)

    @property
    def has_data(self) -> bool:
        """True when any data source or RAG callback is configured."""
        return (
            self._raw_data_source is not None
            or self.query_rag_callback is not None
            or bool(self.data_files)
        )

    # backward compat: expose data_source as the raw value
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
        """
        Call a named function in the loaded script module.
        Returns None if the script is not loaded or the function doesn't exist.
        """
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
        """
        Legacy helper — returns a plain string for a given query.
        Prefer ``query_data()`` for new code.
        """
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
            "personality_id":    self.personality_id,
            "name":              self.name,
            "author":            self.author,
            "category":          self.category,
            "description":       self.description,
            "system_prompt":     self.system_prompt,
            "tools":             self.mcp_tool_names,
            "has_tool_binding":  bool(self._tool_binding),
            "has_data_source":   self.has_data,
            "data_files":        [str(p) for p in self.data_files],
            "has_script":        self.script is not None,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], **kwargs
    ) -> "LollmsPersonality":
        """
        Reconstruct from a ``to_dict()`` snapshot.
        Callbacks / bindings must be re-injected via ``kwargs`` or
        ``attach_tool_binding()``.
        """
        return cls(
            name           = data.get("name", "assistant"),
            author         = data.get("author", ""),
            category       = data.get("category", "general"),
            description    = data.get("description", ""),
            system_prompt  = data.get("system_prompt", ""),
            tools          = data.get("tools") or None,
            personality_id = data.get("personality_id"),
            **kwargs,
        )

    # ------------------------------------------------------------------ dunder

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}"]
        if bool(self._tool_binding):
            parts.append(f"tools={self._tool_binding.binding_name!r}")
        elif self.mcp_tool_names:
            parts.append(f"mcp_names={self.mcp_tool_names}")
        if self.has_data:
            parts.append("has_data=True")
        if self.script_module is not None:
            parts.append("has_script=True")
        return f"LollmsPersonality({', '.join(parts)})"

    def __bool__(self) -> bool:
        return True


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

    def ensure_data_vectorized(self, **_) -> None:
        pass

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "NullPersonality()"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _is_tool_binding(obj: Any) -> bool:
    """
    Duck-type check for LollmsToolBinding without importing it directly
    (avoids circular dependency at module load time).
    """
    return (
        obj is not None
        and not isinstance(obj, (list, str))
        and hasattr(obj, "discover_tools")
        and hasattr(obj, "execute_tool")
        and hasattr(obj, "to_chat_tool_specs")
    )