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
import re
import os
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ascii_colors import ASCIIColors, trace_exception

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
    Self-contained, null-safe personality unit for lollms-client.

    Accepted forms for ``tools``
    ─────────────────────────────
    * ``None``                    → _NullToolBinding (no tools configured,
                                    but if a client_binding is supplied to
                                    tool_specs(), ALL its tools are exposed)
    * ``LollmsToolBinding``       → used directly; all of its tools exposed
    * ``List[str]`` of MCP names  → explicit allowlist; only the named tools
                                    are exposed when tool_specs() is called.
                                    An empty list means "no tools at all".

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
        # _has_explicit_allowlist distinguishes:
        #   tools=None  → False → expose all tools from client_binding (unrestricted)
        #   tools=[]    → True  → expose NO tools (empty allowlist)
        #   tools=[...] → True  → expose only the listed tools
        self._has_explicit_allowlist: bool = False
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
            # No restriction — client_binding tools will all be exposed
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
            # An explicit list (even if empty) is an allowlist.
            # Empty list → personality has NO tools at all.
            # Non-empty list → only those named tools are allowed.
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
        Resolve the personality's tool allowlist against the available binding.

        Behaviour matrix
        ────────────────
        tools=None (no allowlist)
            → expose ALL tools from client_binding (or self._tool_binding)
        tools=[] (empty allowlist)
            → expose NO tools
        tools=['A::x', 'B::y', ...] (explicit allowlist)
            → expose ONLY the listed tool names, looked up from the binding
        tools=LollmsToolBinding instance
            → expose all tools from that binding (no name filter applied)

        The ``client_binding`` argument (typically ``lollmsClient.tools``) is
        preferred over ``self._tool_binding`` so the personality can piggyback
        on whatever MCP binding the host application has configured, while still
        enforcing its own per-name allowlist.
        """
        # If an explicit allowlist was given and it is empty → no tools at all
        if self._has_explicit_allowlist and not self.mcp_tool_names:
            return {}

        # Prefer the caller-supplied binding (e.g. lollmsClient.tools)
        binding = client_binding or self._tool_binding
        if not binding:
            return {}

        try:
            all_specs = binding.to_chat_tool_specs(**discover_kwargs)
        except Exception as exc:
            trace_exception(exc)
            return {}

        # No explicit allowlist → personality is unrestricted, return everything
        if not self._has_explicit_allowlist:
            return all_specs

        # Apply the allowlist using a set for O(1) membership checks
        allowed = set(self.mcp_tool_names)
        filtered = {
            name: spec
            for name, spec in all_specs.items()
            if name in allowed
        }

        # Warn about names that were requested but not found in the binding
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
        """
        Reconstruct from a ``to_dict()`` snapshot.
        Callbacks / bindings must be re-injected via ``kwargs`` or
        ``attach_tool_binding()``.
        """
        # Restore the tools list so _init_tools can set _has_explicit_allowlist correctly
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
        self._has_explicit_allowlist  = False   # NullPersonality is unrestricted by design
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
