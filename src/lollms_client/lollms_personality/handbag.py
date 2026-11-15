from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ascii_colors import ASCIIColors


class Handbag:
    """
    A self-contained folder carrying ALL of an agent's resources.
    """
    def __init__(
        self,
        handbag_path: Union[str, Path],
        skills_output_path: Optional[Union[str, Path]] = None,
        tools_output_path: Optional[Union[str, Path]] = None,
    ):
        self.path = Path(handbag_path).resolve()
        if not self.path.exists() or not self.path.is_dir():
            raise ValueError(f"Handbag path does not exist or is not a directory: {self.path}")

        self.soul_path = self.path / "SOUL.md"
        self.coworkers_dir = self.path / "coworkers"
        self.tools_dir = self.path / "tools"
        self._skills_dir = self.path / "skills"
        self.assets_dir = self.path / "assets"
        self.memory_dir = self.path / "memory"
        self.rag_dir = self.path / "rag"
        self.docs_dir = self.path / "docs"
        self.workspace_dir = self.path / "workspace"

        self.manifest = self._load_manifest()

        self.tool_files: List[Path] = self._load_tools()
        self.skills_dirs: List[Path] = [self._skills_dir.resolve()] if self._skills_dir.exists() else []
        # Diagnostic: log EXACTLY which rag/ folder is being scanned, whether
        # it exists, and every file discovered — BEFORE the .db split. Without
        # this the "no RAG database" symptom is untraceable when the user has
        # the wrong handbag open (JML vs Ltspice_design_optimizer) or a path
        # mismatch between the on-screen tree and disk.
        try:
            _rag_dir_exists = self.rag_dir.exists()
            _rag_scan_preview = []
            if _rag_dir_exists:
                try:
                    _rag_scan_preview = [
                        str(p.relative_to(self.path))
                        for p in self.rag_dir.rglob("*") if p.is_file()
                    ][:20]
                except Exception:
                    _rag_scan_preview = ["<scan failed>"]
            ASCIIColors.info(
                f"[Handbag-SCAN] handbag='{self.path.name}' "
                f"rag_dir='{self.rag_dir}' exists={_rag_dir_exists} "
                f"files_found={len(_rag_scan_preview)} preview={_rag_scan_preview}"
            )
        except Exception:
            pass
        self.rag_files: List[Path] = self._load_rag_files()

        # ── AUTO-DISCOVER .db DATASTORES FROM <handbag>/rag/ ──
        # A `.db` file lying anywhere under `<handbag>/rag/` (e.g.
        # `rag/ltspice/ltspice.db`) is a safe_store datastore. It has NOTHING
        # to do with plain-text RAG documents (which are `.md`/`.txt`/etc.),
        # and pretending it is one causes `read_text()` to silently fail on
        # the binary blob — the store then never reaches the catalog and the
        # agent correctly-but-uselessly reports "no RAG database". Auto-classify
        # every `.db` under `rag/` as a SafeStore so a handbag shipping a
        # datastore Just Works without ANY handbag.yaml entry.
        _DB_LIKE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
        _rag_db_files: List[Path] = [
            f for f in self.rag_files if f.suffix.lower() in _DB_LIKE_SUFFIXES
        ]
        # Leave rag_files containing ONLY non-datastore files for the
        # plain-text pipeline (which is what `_load_rag_files()` conceptually
        # returns). Every recognised datastore suffix is routed to SafeStore.
        self.rag_files = [
            f for f in self.rag_files if f.suffix.lower() not in _DB_LIKE_SUFFIXES
        ]

        # Merge the auto-discovered .db entries into the raw extra list so the
        # unified classification loop below handles them identically to a
        # yaml-declared store (dedup by resolved path, deferred fallback if
        # safe_store can't be imported, named by the .db's stem).
        _all_extra_from_yaml = self._load_extra_knowledge_bases_rich()
        _auto_rag_entries: List[Dict[str, Any]] = [
            {
                "path": f,
                "name": f.stem,
                "description": f"Knowledge base '{f.stem}' (auto-discovered at {f.relative_to(self.path)}).",
            }
            for f in _rag_db_files
        ]
        # Rebuild unified list, keeping yaml entries first (highest priority)
        # and appending auto entries not already covered by yaml.
        _seen_paths = {str(e.get("path")) for e in _all_extra_from_yaml if e.get("path")}
        _all_extra = list(_all_extra_from_yaml)
        for _a in _auto_rag_entries:
            if str(_a["path"]) not in _seen_paths:
                _all_extra.append(_a)
                _seen_paths.add(str(_a["path"]))

        if _rag_db_files:
            try:
                ASCIIColors.success(
                    f"[Handbag] Auto-discovered {len(_rag_db_files)} .db datastore(s) "
                    f"under rag/: {[str(f.relative_to(self.path)) for f in _rag_db_files]}"
                )
            except Exception:
                pass
        else:
            # Belt-and-suspenders: if NO .db was found under rag/ but the folder
            # exists, emit a clear diagnostic. This is the exact scenario the
            # user hits when looking at the WRONG agent's file tree (e.g. viewing
            # `Ltspice_design_optimizer/rag/ltspice/Ltspice.db` in the editor but
            # chatting with `JML`, whose handbag has no rag/ folder at all).
            try:
                if self.rag_dir.exists():
                    _rag_files_all = list(self.rag_files)  # non-.db (already filtered)
                    if _rag_files_all:
                        ASCIIColors.info(
                            f"[Handbag] rag/ contains {len(_rag_files_all)} plain-text "
                            f"document(s) but ZERO .db datastores: "
                            f"{[str(f.relative_to(self.path)) for f in _rag_files_all][:10]}"
                        )
                    else:
                        ASCIIColors.warning(
                            f"[Handbag] rag/ exists at '{self.rag_dir}' but contains "
                            f"NO files (no .db datastores, no plain-text documents). "
                            f"The agent will report 'no RAG database' — check that the "
                            f".db is really in THIS handbag, not another one."
                        )
                else:
                    # ── SIBLING-AGENT CROSS-REFERENCE ──
                    # The single most common cause of "my agent has no RAG" is
                    # loading the wrong agent. When our own `rag/` is empty, scan
                    # the PARENT directory (the agents folder) for sibling agent
                    # handbags that DO contain `.db` files, and name them in the
                    # warning. Turns a 30-minute debug into a 2-second fix: the
                    # user immediately sees "JML has no rag/, but
                    # Ltspice_design_optimizer/rag/ltspice/Ltspice.db exists".
                    sibling_hits: List[str] = []
                    try:
                        parent = self.path.parent
                        if parent.exists():
                            for sib in sorted(parent.iterdir()):
                                if (not sib.is_dir() or sib == self.path):
                                    continue
                                sib_rag = sib / "rag"
                                if not sib_rag.exists():
                                    continue
                                try:
                                    sib_dbs = [
                                        str(p.relative_to(parent))
                                        for p in sib_rag.rglob("*")
                                        if p.is_file() and p.suffix.lower() in (".db", ".sqlite", ".sqlite3")
                                    ]
                                except Exception:
                                    sib_dbs = []
                                if sib_dbs:
                                    sibling_hits.append(f"{sib.name} → {sib_dbs[0]}")
                    except Exception:
                        sibling_hits = []

            except Exception:
                pass

        # Split extra KBs: plain files/dirs vs. safe_store .db datastores.
        # Classification is driven by the DECLARED suffix (intent), not disk state,
        # so a missing .db is reported (warning) instead of being silently demoted
        # to a plain KB path. A broken/missing .db never crashes the handbag build.
        # NOTE: `_all_extra` was already built above (yaml entries + auto-discovered
        # rag/ .db entries, deduped by resolved path). Do NOT re-load here — that
        # would re-introduce the "yaml-only, no auto-db" bug.
        try:
            ASCIIColors.info(
                f"[Handbag-DBG] _load_extra_knowledge_bases_rich() returned "
                f"{len(_all_extra)} entry(ies): "
                f"{[(str(e.get('path')), e.get('name'), e.get('description')) for e in _all_extra]}"
            )
        except Exception:
            pass
        self.extra_knowledge_bases: List[Path] = []
        self.extra_safe_stores: List[Any] = []
        for entry in _all_extra:
            p = entry["path"]
            declared_name = entry.get("name")
            declared_desc = entry.get("description")
            _is_db = self._is_safe_store_db(p)

            try:
                ASCIIColors.info(
                    f"[Handbag-DBG] Classifying KB entry path='{p}' "
                    f"suffix='{p.suffix}' declared_name={declared_name!r} "
                    f"is_safe_store_db={_is_db} exists_on_disk={p.exists()} "
                    f"→ routing to {'extra_safe_stores' if _is_db else 'extra_knowledge_bases'}"
                )
            except Exception:
                pass

            # A declared path that does NOT exist as a readable text file but
            # IS a data-store suffix (or named after one) is still treated as a
            # SafeStore so its deferred-config fallback is registered — the
            # catalog then surfaces it to the LLM by declared name, even when
            # safe_store cannot be loaded eagerly. This closes the "declared
            # but silently demoted to plain .db KB path" gap that made an
            # LTspice-style store invisible to the agent.


            if self._is_safe_store_db(p):
                binding = self._open_safe_store(p)
                try:
                    ASCIIColors.info(
                        f"[Handbag-DBG] _open_safe_store('{p}') returned "
                        f"{type(binding).__name__ if binding is not None else 'None'}"
                    )
                except Exception:
                    pass
                if binding is not None:
                    # Attach human-authored identity (provenance) so the personality's
                    # tool builder can surface the real name/description to the LLM.
                    display = declared_name or p.stem
                    desc = declared_desc or f"Search knowledge base '{display}' (from {p.name})."
                    try:
                        binding._handbag_label = display
                        binding._handbag_description = desc
                        binding._handbag_source_path = str(p)
                    except Exception:
                        pass
                    self.extra_safe_stores.append(binding)
                    try:
                        ASCIIColors.info(
                            f"[Handbag] SafeStore '{display}' ← '{p.name}' "
                            f"(description={'set' if declared_desc else 'auto'})"
                        )
                    except Exception:
                        pass
                else:
                    # Fallback: the .db was declared but could not be opened
                    # (missing file, safe_store not installed, or init failure).
                    # Register a NAMED store config so the RAG resolver can still
                    # address it by datasource_name — the binding will be built
                    # lazily on first query, and any real error will surface there
                    # with a clear message instead of being silently dropped.
                    display = declared_name or p.stem
                    desc = declared_desc or f"Search knowledge base '{display}' (from {p.name})."
                    self.extra_safe_stores.append({
                        "name": display,
                        "db_path": str(p),
                        "vectorizer_name": "st",
                        "description": desc,
                    })
                    try:
                        ASCIIColors.warning(
                            f"[Handbag] SafeStore '{display}' ← '{p.name}' could not be "
                            f"opened eagerly; registered as deferred store config "
                            f"(will be built on first query)."
                        )
                    except Exception:
                        pass
            else:
                self.extra_knowledge_bases.append(p)

        try:
            ASCIIColors.info(
                f"[Handbag] extra_knowledge_bases={len(self.extra_knowledge_bases)} "
                f"plain path(s) + extra_safe_stores={len(self.extra_safe_stores)} "
                f"auto-instantiated SafeStore(s) — handbag='{self.path.name}'"
            )
            # Detail: log the final store list so the user can visually confirm
            # each `.db` under rag/ was registered (by name + resolved path).
            if self.extra_safe_stores:
                _store_summary = []
                for _s in self.extra_safe_stores:
                    if isinstance(_s, dict):
                        _store_summary.append(f"{_s.get('name')!r}→{_s.get('db_path')}")
                    else:
                        _store_summary.append(
                            f"{getattr(_s, '_handbag_label', getattr(_s, 'db_path', '?'))}"
                        )
                ASCIIColors.success(
                    f"[Handbag] ✅ RAG SafeStores registered: {_store_summary} — "
                    f"the agent WILL see these databases in its system prompt."
                )
            elif self.rag_dir.exists():
                _db_lookaround = [str(p.relative_to(self.path)) for p in self.rag_dir.rglob("*.db")]
                if _db_lookaround:
                    ASCIIColors.warning(
                        f"[Handbag] ⚠️ Found .db file(s) under rag/ but ZERO stores were "
                        f"registered: {_db_lookaround}. This means the auto-discovery "
                        f"failed — please report this state."
                    )
        except Exception:
            pass

        self.memory_db_path: Optional[str] = f"sqlite:///{self.memory_dir / 'memory.db'}" if self.memory_dir.exists() else None
        self.assets: Dict[str, str] = self._load_assets()



    def _resolve_manifest_path(self, key: str) -> Optional[Path]:
        """Resolves an optional manifest-declared path relative to the handbag root."""
        raw = self.manifest.get(key) if isinstance(self.manifest, dict) else None
        if not raw:
            return None
        try:
            p = Path(str(raw)).expanduser()
            if not p.is_absolute():
                p = (self.path / p).resolve()
            else:
                p = p.resolve()
            return p
        except Exception:
            return None

    @property
    def global_memory_db_path(self) -> Optional[str]:
        """Returns the path to the global handbag memory database."""
        return self.memory_db_path

    @property
    def skills_dir(self) -> Path:
        """Returns the canonical skills directory for this handbag."""
        return self._skills_dir

    @property
    def builtin_skills(self) -> Dict[str, str]:
        """
        Returns a ``{skill_title: skill_file_path}`` map for every skill shipped
        natively with this handbag.

        A skill is "builtin" when its SKILL.md lives under one of this handbag's
        registered ``skills_dirs`` (which always includes ``<handbag>/skills``
        but may also include additional app-registered dirs — see
        ``register_skills_dir``).

        A skill whose SKILL.md cannot be parsed or has no title is skipped.
        Only the file path is recorded (no content); the value is the absolute
        path string of the ``SKILL.md`` file.
        """
        from .skill import parse_skill_md  # local import: avoid circular import at module load

        builtin: Dict[str, str] = {}
        for skills_root in self.skills_dirs:
            try:
                root = Path(skills_root).resolve()
            except Exception:
                continue
            if not root.exists() or not root.is_dir():
                continue

            # A handbag can host skills in two layouts:
            #   1. <root>/<slug>/SKILL.md
            #   2. <root>/<file>.md  (single-file skill)
            for item in sorted(root.iterdir()):
                try:
                    if item.is_dir():
                        skill_file = item / "SKILL.md"
                        if skill_file.exists():
                            skill = parse_skill_md(skill_file)
                            if skill and skill.title:
                                builtin[skill.title] = str(skill_file.resolve())
                    elif item.is_file() and item.suffix.lower() == ".md" and item.name != "README.md":
                        skill = parse_skill_md(item)
                        if skill and skill.title:
                            builtin[skill.title] = str(item.resolve())
                except Exception:
                    continue
        return builtin

    def register_skills_dir(self, skills_dir: Union[str, Path]) -> Path:
        """
        Registers an additional directory of application-grade skills.

        Hosting apps call this to inject their own SKILL.md libraries
        (app-shipped, per-user, or per-discussion) alongside the handbag's
        native skills. The directory is created if missing and appended to
        `skills_dirs` (idempotent — re-registering the same path is a no-op).

        Returns:
            The resolved directory path.
        """
        resolved = Path(skills_dir).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        if resolved not in self.skills_dirs:
            self.skills_dirs.append(resolved)
        return resolved

    def _load_manifest(self) -> Dict[str, Any]:
        manifest_path = self.path / "handbag.yaml"
        if not manifest_path.exists():
            return {}
        try:
            import yaml
            content = manifest_path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_tools(self) -> List[Path]:
        """Returns one entry per tool library shipped in ``<handbag>/tools/``.

        The layout mirrors the skills standard: **the entry stored here is the
        toolset FOLDER** (which the LCP loader resolves to ``TOOLS.py``,
        ``<folder>.py``, or the first ``.py``). This lets individual tool files
        be renamed to ``TOOLS.py`` without breaking handbags.

        Flat ``.py`` files sitting directly under ``<handbag>/tools/`` remain
        supported for backward compatibility: the FILE path is returned, and
        the resolver treats it as a single-file library.
        """
        if not self.tools_dir.exists():
            return []
        entries: List[Path] = []
        for item in sorted(self.tools_dir.iterdir()):
            if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
                # Legacy flat single-file tool library.
                entries.append(item.resolve())
            elif item.is_dir():
                # Skip asset / metadata folders that are not libraries.
                if item.name in ("assets", "references", "__pycache__") or item.name.startswith("."):
                    continue
                # Only emit the folder if it contains at least one .py — the
                # LCP resolver will fail-closed otherwise, so we avoid
                # cluttering the list with empty folders.
                if any(f.suffix == ".py" and f.stem != "__init__" for f in item.iterdir() if f.is_file()):
                    entries.append(item.resolve())
        return entries

    def _load_rag_files(self) -> List[Path]:
        if not self.rag_dir.exists():
            return []
        files = []
        for f in sorted(self.rag_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                files.append(f.resolve())
        return files

    def _load_extra_knowledge_bases(self) -> List[Path]:
        """Back-compat wrapper: returns only the resolved paths (drops name/description)."""
        return [entry["path"] for entry in self._load_extra_knowledge_bases_rich()]

    # YAML keys accepted for knowledge-base declarations. The canonical key is
    # ``extra_knowledge_bases`` (backward compat), but users frequently write
    # ``rag_databases`` / ``knowledge_bases`` / ``rag`` / ``knowledge`` — those
    # were previously SILENTLY IGNORED, producing an agent that "has no RAG
    # database" despite a declaration in handbag.yaml. All accepted keys are
    # merged (deduplicated by resolved path). See [Handbag] DBG logs at load.
    _KB_MANIFEST_KEYS = (
        "extra_knowledge_bases",
        "rag_databases",
        "knowledge_bases",
        "knowledge",
        "rag",
    )

    def _load_extra_knowledge_bases_rich(self) -> List[Dict[str, Any]]:
        """
        Reads knowledge-base declarations from handbag.yaml. Every accepted YAML
        key (see ``_KB_MANIFEST_KEYS``) is merged. Each yielded item is a
        normalised dict: {"path": Path, "name": str|None, "description": str|None}.

        Accepted yaml shapes (ALL forms work):
            extra_knowledge_bases:
              - "Auto design.db"                    # bare string  → name = stem
              - "docs/notes.txt"
              - path: "Auto design.db"              # dict form    → honours name/description
                name: "auto_design"
                description: "ASIC project specs."

            rag_databases:                          # ALIAS — same semantics
              - path: "LTspice.lib.db"
                name: "ltspice"
                description: "LTspice component models."

            knowledge_bases: [...]                  # ALIAS
            rag:            [...]                  # ALIAS
            knowledge:      [...]                  # ALIAS

        Also accepts a mapping form where the KEY is the display name:
            rag_databases:
              "ltspice": {path: "LTspice.lib.db", description: "..."}
              "asic": "asic_specs.db"              # shorthand
        """
        if not isinstance(self.manifest, dict):
            return []

        # Gather every declared key with a dual-form normalisation:
        #   * list items → {"path", "name", "description"}
        #   * mapping form → {"path", "name": <key>, "description"}
        declared: List[Any] = []
        _found_keys: List[str] = []
        for key in self._KB_MANIFEST_KEYS:
            val = self.manifest.get(key)
            if not val:
                continue
            _found_keys.append(key)
            if isinstance(val, (str, Path)):
                declared.append(val)
            elif isinstance(val, dict):
                # Mapping form: {display_name: path_or_dict}
                for disp_name, inner in val.items():
                    if isinstance(inner, dict):
                        merged = dict(inner)
                        merged.setdefault("name", disp_name)
                        declared.append(merged)
                    else:
                        declared.append({"path": inner, "name": disp_name})
            elif isinstance(val, (list, tuple)):
                declared.extend(val)
            else:
                try:
                    ASCIIColors.warning(
                        f"[Handbag] Ignored manifest key '{key}': expected list/mapping, "
                        f"got {type(val).__name__}."
                    )
                except Exception:
                    pass

        try:
            ASCIIColors.info(
                f"[Handbag-DBG] KB manifest keys found={_found_keys or '(none)'} "
                f"→ {len(declared)} candidate(s) to normalise."
            )
        except Exception:
            pass

        if not declared:
            return []

        out: List[Dict[str, Any]] = []
        seen: set = set()
        for item in declared:
            try:
                if isinstance(item, str):
                    raw_path = item
                    declared_name = None
                    declared_desc = None
                elif isinstance(item, dict):
                    raw_path = item.get("path")
                    if not raw_path:
                        continue
                    declared_name = item.get("name")
                    declared_desc = item.get("description")
                elif isinstance(item, Path):
                    raw_path = item
                    declared_name = None
                    declared_desc = None
                else:
                    ASCIIColors.warning(
                        f"[Handbag] Skipped malformed KB declaration entry: {item!r}"
                    )
                    continue

                p = Path(raw_path).expanduser()
                if not p.is_absolute():
                    p = (self.path / p).resolve()
                else:
                    p = p.resolve()

                if p in seen:
                    continue
                seen.add(p)

                out.append({
                    "path": p,
                    "name": declared_name,
                    "description": declared_desc,
                })
            except Exception as e:
                try:
                    ASCIIColors.warning(
                        f"[Handbag] Skipped malformed KB declaration entry: {item!r} ({e})"
                    )
                except Exception:
                    pass
        return out

    @staticmethod
    def _is_safe_store_db(path: Path) -> bool:
        """
        True if the declared path is intended as a safe_store datastore (a .db file).

        Classification is driven by the DECLARED suffix (intent), NOT by disk state:
        a `.db` declared in handbag.yaml that does not yet exist on disk is still
        routed to the SafeStore auto-instantiation path, where it is reported with
        a clear warning instead of being silently demoted to a plain KB path.
        """
        return path.suffix.lower() in (".db", ".sqlite", ".sqlite3")

    @staticmethod
    def _open_safe_store(db_path: Path) -> Optional[Any]:
        """
        Auto-instantiate a SafeStoreRAGBinding from a `.db` path alone.

        The `.db` file IS the source of truth for the store's configuration
        (vectorizer, chunk size, model, encryption state) — it is read back
        transparently by `safe_store.SafeStore(db_path=...)` inside
        `SafeStoreRAGBinding.get_store()`. NO external RAG binding profile,
        `lollms_client.rag` registry entry, or `rag_model_profiles_registry`
        entry is consulted.

        The binding is LAZY: `.get_store()` is deferred until the first query,
        so no SentenceTransformer is loaded at handbag build time.

        Returns:
            The SafeStoreRAGBinding, or None if the .db file is missing or
            safe_store cannot be imported. Never raises.
        """
        try:
            ASCIIColors.info(
                f"[Handbag-DBG] _open_safe_store ENTRY db_path='{db_path}' "
                f"exists={db_path.exists()} suffix='{db_path.suffix}'"
            )
        except Exception:
            pass
        try:
            if not db_path.exists():
                ASCIIColors.warning(
                    f"[Handbag] Declared safe_store '.db' not found on disk: {db_path} — "
                    f"the store cannot be auto-instantiated. Create the file or remove "
                    f"the declaration from handbag.yaml."
                )
                return None

            try:
                ASCIIColors.info(
                    f"[Handbag-DBG] Importing SafeStoreRAGBinding from "
                    f"lollms_client.rag_bindings.safe_store ..."
                )
            except Exception:
                pass
            from lollms_client.rag_bindings.safe_store import SafeStoreRAGBinding
            try:
                ASCIIColors.info(
                    f"[Handbag-DBG] SafeStoreRAGBinding imported OK: {SafeStoreRAGBinding}"
                )
            except Exception:
                pass
            binding = SafeStoreRAGBinding(
                db_path=str(db_path),
                vectorizer_name="st",
                use_shared_server=True,
                port=8765,
            )
            # Do NOT force .get_store() here — keep the lazy contract.
            ASCIIColors.info(
                f"[Handbag] Auto-instantiated SafeStoreRAGBinding from '.db' file: {db_path} "
                f"(vectorizer={getattr(binding, 'vectorizer_name', '?')}, "
                f"chunk_size={getattr(binding, 'chunk_size', '?')}, "
                f"encrypted={bool(getattr(binding, 'encryption_key', None))}) — "
                f"config will be read from the .db on first query."
            )
            return binding
        except ImportError as ie:
            ASCIIColors.warning(
                f"[Handbag] Cannot auto-instantiate SafeStore for '{db_path}': "
                f"`safe_store` package is not installed ({ie}). "
                f"Install it with: pip install safe_store"
            )
            return None
        except Exception as e:
            ASCIIColors.warning(
                f"[Handbag] Failed to auto-instantiate SafeStore for '{db_path}': {e} — "
                f"the store will be skipped for this handbag build."
            )
            return None

    def close_extra_safe_stores(self) -> None:
        """
        Releases any resources held by extra safe_store datastores.

        Note: the shared vectorizer daemon itself is NOT shut down here — it is
        a process-wide singleton. Call `SafeStore.shutdown_shared_vectorizer(port=8765)`
        explicitly when terminating the host application.
        """
        for binding in getattr(self, "extra_safe_stores", []) or []:
            try:
                store = getattr(binding, "_store_instance", None)
                if store is not None and hasattr(store, "close"):
                    store.close()
            except Exception:
                pass
        self.extra_safe_stores = []

    @staticmethod
    def normalize_extra_knowledge_bases(
        extra_knowledge_bases: Optional[Union[str, Path, List[Union[str, Path]]]],
    ) -> List[Path]:
        """
        Normalizes the `extra_knowledge_bases` parameter into a deduplicated list
        of resolved, existing paths.

        Accepted formats:
          - None                          -> []
          - "path/to/file_or_dir"         -> [Path(...)]
          - Path("path/to/file_or_dir")   -> [Path(...)]
          - ["a.md", Path("b_dir"), ...]  -> [Path(a.md), Path(b_dir), ...]

        Each entry may be a file (any extension) or a directory (recursively
        scanned by the RAG layer). Non-existent paths are silently skipped so
        that a handbag remains loadable even when an optional KB is missing.

        Special case: entries ending in `.db` are treated as safe_store
        datastores and opened via the shared vectorizer daemon (see
        `_open_safe_store`). They are exposed separately on
        `handbag.extra_safe_stores` and NOT included in
        `handbag.extra_knowledge_bases`.
        """
        if extra_knowledge_bases is None:
            return []
        if isinstance(extra_knowledge_bases, (str, Path)):
            candidates: List[Union[str, Path]] = [extra_knowledge_bases]
        elif isinstance(extra_knowledge_bases, (list, tuple)):
            candidates = list(extra_knowledge_bases)
        else:
            raise TypeError(
                "extra_knowledge_bases must be None, a str/Path, or a list of str/Path; "
                f"got {type(extra_knowledge_bases).__name__}"
            )

        resolved: List[Path] = []
        seen: set = set()
        for entry in candidates:
            if not isinstance(entry, (str, Path)):
                raise TypeError(
                    f"extra_knowledge_bases entries must be str or Path; got {type(entry).__name__}"
                )
            p = Path(entry).expanduser().resolve()
            if not p.exists():
                ASCIIColors.warning(f"extra_knowledge_bases entry not found, skipping: {p}")
                continue
            if p in seen:
                continue
            seen.add(p)
            resolved.append(p)
        return resolved

    def has_documents(self) -> bool:
        """Returns True when the handbag carries a non-empty docs/ library."""
        if not self.docs_dir.exists() or not self.docs_dir.is_dir():
            return False
        try:
            return any(item.name != ".gitkeep" for item in self.docs_dir.iterdir())
        except Exception:
            return False

    def add_document(
        self,
        file_path: Union[str, Path],
        use_llm: bool = False,
        lollms_client: Optional[Any] = None,
    ) -> Path:
        """
        Ingests a document (txt/md/pdf/docx/pptx) into the handbag's docs/ library
        as a heading-chunked tree with an INDEX.md (title, authors, abstract,
        ordered section map with token estimates).

        The LLM is only consulted for metadata when use_llm=True AND the
        deterministic heuristics failed.
        """
        from .doc_navigator import DocIngestor
        return DocIngestor().add_document(
            file_path, self.path, use_llm=use_llm, lollms_client=lollms_client
        )

    @staticmethod
    def create_structure(target_path: Union[str, Path], name: str = "My Handbag") -> Path:
        p = Path(target_path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        (p / "coworkers").mkdir(exist_ok=True)
        (p / "tools").mkdir(exist_ok=True)
        (p / "skills").mkdir(exist_ok=True)
        (p / "rag").mkdir(exist_ok=True)
        (p / "docs").mkdir(exist_ok=True)
        (p / "memory").mkdir(exist_ok=True)
        (p / "assets").mkdir(exist_ok=True)
        (p / "workspace").mkdir(exist_ok=True)
        return p

    def _load_assets(self) -> Dict[str, str]:
        if not self.assets_dir.exists():
            return {}
        assets = {}
        for f in self.assets_dir.rglob("*"):
            if f.is_file():
                assets[f.name] = str(f.resolve())
        return assets

    def create_memory_manager(self) -> Optional[Any]:
        if not self.memory_db_path:
            return None
        try:
            from lollms_client.lollms_memory import LollmsMemoryManager, MemoryConfig
            db_file = self.memory_db_path.replace("sqlite:///", "")
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)
            manager = LollmsMemoryManager(
                db_path=self.memory_db_path,
                owner_id=f"handbag_{self.path.name}",
                config=MemoryConfig(working_token_budget=2000),
            )
            return manager
        except Exception:
            return None