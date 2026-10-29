from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ascii_colors import ASCIIColors


class Handbag:
    """
    A self-contained folder carrying ALL of an agent's resources.
    """
    def __init__(self, handbag_path: Union[str, Path]):
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
        self.rag_files: List[Path] = self._load_rag_files()

        # Split extra KBs: plain files/dirs vs. safe_store .db datastores.
        _all_extra = self._load_extra_knowledge_bases()
        self.extra_knowledge_bases: List[Path] = [
            p for p in _all_extra if not self._is_safe_store_db(p)
        ]
        self.extra_safe_stores: List[Any] = [
            self._open_safe_store(p) for p in _all_extra if self._is_safe_store_db(p)
        ]

        self.memory_db_path: Optional[str] = f"sqlite:///{self.memory_dir / 'memory.db'}" if self.memory_dir.exists() else None
        self.assets: Dict[str, str] = self._load_assets()

    @property
    def global_memory_db_path(self) -> Optional[str]:
        """Returns the path to the global handbag memory database."""
        return self.memory_db_path

    @property
    def skills_dir(self) -> Path:
        """Returns the canonical skills directory for this handbag."""
        return self._skills_dir

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
        if not self.tools_dir.exists():
            return []
        files = []
        for item in sorted(self.tools_dir.iterdir()):
            if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
                files.append(item.resolve())
            elif item.is_dir():
                for py_file in sorted(item.glob("*.py")):
                    if py_file.stem != "__init__":
                        files.append(py_file.resolve())
        return files

    def _load_rag_files(self) -> List[Path]:
        if not self.rag_dir.exists():
            return []
        files = []
        for f in sorted(self.rag_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                files.append(f.resolve())
        return files

    def _load_extra_knowledge_bases(self) -> List[Path]:
        """Reads `extra_knowledge_bases` declared in handbag.yaml (list of paths)."""
        declared = self.manifest.get("extra_knowledge_bases") if isinstance(self.manifest, dict) else None
        return self.normalize_extra_knowledge_bases(declared)

    @staticmethod
    def _is_safe_store_db(path: Path) -> bool:
        """A `.db` file in extra_knowledge_bases is treated as a safe_store datastore."""
        return path.is_file() and path.suffix.lower() == ".db"

    @staticmethod
    def _open_safe_store(db_path: Path) -> Any:
        """
        Opens a safe_store datastore via the SHARED vectorizer daemon so that
        multiple processes (agents, workers, sub-agents) reuse a single
        SentenceTransformer instance in VRAM/RAM.

        Returns a `SafeStoreRAGBinding` (lazy — the underlying SafeStore is
        instantiated on first `get_store()` call). Returns None on failure so
        the handbag remains loadable even if a datastore is corrupt.
        """
        try:
            from lollms_client.rag_bindings.safe_store import SafeStoreRAGBinding
            binding = SafeStoreRAGBinding(
                db_path=str(db_path),
                vectorizer_name="st",
                use_shared_server=True,
                port=8765,
            )
            ASCIIColors.info(
                f"[Handbag] Attached safe_store datastore (shared daemon :8765): {db_path.name}"
            )
            return binding
        except Exception as e:
            ASCIIColors.warning(
                f"[Handbag] Failed to open safe_store datastore '{db_path}': {e}"
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