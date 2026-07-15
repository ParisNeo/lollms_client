# handbag.py
# ─────────────────────────────────────────────────────────────────────────────
# Handbag — A self-contained folder that carries ALL of an agent's resources.
#
# A Handbag aggregates:
#   - Personalities (SOUL.md bundles with their own tools and RAG knowledge)
#   - Extra tools (LCP .py files the agent can grab from)
#   - Skills (SKILL.md files the agent can add/fix over time)
#   - RAG sources (text documents for retrieval-augmented generation)
#   - Memory database (SQLite for persistent episodic/semantic memory)
#   - Workspace (optional isolated working directory)
#
# Folder Structure:
#   my_handbag/
#   ├── handbag.yaml           # Optional manifest (name, default_personality, skills_mode)
#   ├── personalities/         # Personality bundles (subdirs with SOUL.md)
#   │   ├── researcher/
#   │   │   └── SOUL.md
#   │   └── coder/
#   │       └── SOUL.md
#   ├── tools/                 # Extra LCP tools (.py files or subdirs)
#   │   ├── my_custom_tool.py
#   │   └── another_tool/
#   │       └── another_tool.py
#   ├── skills/                # SKILL.md files (agent creates/updates these)
#   │   ├── python_patterns/
#   │   │   └── SKILL.md
#   │   └── ...
#   ├── rag/                   # RAG documents (text files for retrieval)
#   │   ├── doc1.txt
#   │   └── doc2.md
#   ├── memory/                # Memory database
#   │   └── memory.db
#   └── workspace/             # Optional isolated workspace
#
# Usage:
#   agent = Agent(
#       lc=client,
#       handbag_path="./my_handbag",
#       # personality, skills_dirs, tool_files, memory_manager, workspace_path
#       # are all auto-configured from the handbag. Explicit params override.
#   )
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ascii_colors import ASCIIColors

# File extensions considered as text for RAG ingestion
_TEXT_RAG_EXTS = {
    ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml", ".html",
    ".py", ".js", ".ts", ".rs", ".go", ".rb", ".php", ".java", ".kt",
    ".swift", ".c", ".cpp", ".h", ".hpp", ".sql", ".sh", ".bash",
    ".ps1", ".bat", ".toml", ".ini", ".cfg", ".log", ".rdf", ".ttl",
}

# Simple English stop words for keyword-based RAG scoring
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


class Handbag:
    """
    A self-contained folder that carries ALL of an agent's resources.

    The Handbag is designed to be:
    - Portable: copy the folder, and the agent has everything it needs.
    - Version-controllable: all resources are plain files on disk.
    - Extensible: the agent can add/fix skills over time in the skills/ dir.
    - Graceful: missing subdirectories are silently skipped.

    Parameters
    ----------
    handbag_path : str or Path
        Path to the handbag folder.
    """

    def __init__(self, handbag_path: Union[str, Path]):
        self.path = Path(handbag_path).resolve()
        if not self.path.exists():
            raise ValueError(f"Handbag path does not exist: {self.path}")
        if not self.path.is_dir():
            raise ValueError(f"Handbag path is not a directory: {self.path}")

        # Subdirectory paths
        self.personalities_dir = self.path / "personalities"
        self.tools_dir = self.path / "tools"
        self.skills_dir = self.path / "skills"
        self.rag_dir = self.path / "rag"
        self.memory_dir = self.path / "memory"
        self.workspace_dir = self.path / "workspace"

        # Loaded resources
        self._personalities: Dict[str, Any] = {}  # name -> LollmsPersonality
        self._tool_files: List[Path] = []
        self._skills_dirs: List[Path] = []
        self._rag_data_source: Optional[Callable[[str], Dict[str, Any]]] = None
        self._memory_db_path: Optional[str] = None
        self._default_personality_name: Optional[str] = None

        # Load manifest
        self.manifest: Dict[str, Any] = self._load_manifest()

        # Load all resources
        self._load()

    # ------------------------------------------------------------------ manifest

    def _load_manifest(self) -> Dict[str, Any]:
        """Loads the optional handbag.yaml manifest."""
        manifest_path = self.path / "handbag.yaml"
        if not manifest_path.exists():
            return {}

        try:
            import yaml
            content = manifest_path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                return data
        except Exception as e:
            ASCIIColors.warning(f"[Handbag] Failed to load handbag.yaml: {e}")
        return {}

    # ------------------------------------------------------------------ loading

    def _load(self):
        """Loads all resources from the handbag folder."""
        # 1. Personalities
        if self.personalities_dir.exists():
            self._load_personalities()

        # 2. Tools
        if self.tools_dir.exists():
            self._load_tools()

        # 3. Skills directory
        if self.skills_dir.exists():
            self._skills_dirs.append(self.skills_dir.resolve())

        # 4. RAG
        if self.rag_dir.exists():
            self._rag_data_source = self._build_rag_data_source()

        # 5. Memory
        if self.memory_dir.exists():
            db_path = self.memory_dir / "memory.db"
            self._memory_db_path = f"sqlite:///{db_path}"

        # 6. Default personality name from manifest
        self._default_personality_name = self.manifest.get("default_personality")

        ASCIIColors.success(
            f"[Handbag] Loaded from '{self.path.name}': "
            f"{len(self._personalities)} personalities, "
            f"{len(self._tool_files)} tool files, "
            f"{len(self._skills_dirs)} skills dirs, "
            f"{'RAG enabled' if self._rag_data_source else 'no RAG'}, "
            f"{'memory enabled' if self._memory_db_path else 'no memory'}."
        )

    def _load_personalities(self):
        """Loads all personality bundles from the personalities/ directory."""
        try:
            from lollms_client.lollms_personality import PersonalityBundle
        except ImportError:
            ASCIIColors.warning("[Handbag] lollms_personality not available. Skipping personalities.")
            return

        for item in sorted(self.personalities_dir.iterdir()):
            if not item.is_dir():
                continue
            soul_path = item / "SOUL.md"
            if not soul_path.exists():
                continue
            try:
                personality = PersonalityBundle.import_bundle(item)
                self._personalities[item.name] = personality
                ASCIIColors.info(f"[Handbag] Loaded personality: '{item.name}'")
            except Exception as e:
                ASCIIColors.warning(f"[Handbag] Failed to load personality '{item.name}': {e}")

    def _load_tools(self):
        """Scans the tools/ directory for LCP tool files."""
        for item in sorted(self.tools_dir.iterdir()):
            if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
                self._tool_files.append(item.resolve())
            elif item.is_dir():
                # LCP convention: directory name matches .py file
                tool_file = item / f"{item.name}.py"
                if tool_file.exists():
                    self._tool_files.append(tool_file.resolve())
                else:
                    # Fallback: scan for any .py file (excluding __init__)
                    for py_file in sorted(item.glob("*.py")):
                        if py_file.stem != "__init__":
                            self._tool_files.append(py_file.resolve())

    # ------------------------------------------------------------------ RAG

    def _build_rag_data_source(self) -> Optional[Callable[[str], Dict[str, Any]]]:
        """
        Builds a RAG data source callable from the rag/ directory.

        Tries safestore for semantic search if available, otherwise falls back
        to keyword-based scoring.
        """
        # Collect all text documents
        docs: List[Dict[str, str]] = []
        for f in sorted(self.rag_dir.rglob("*")):
            if not f.is_file():
                continue
            # Skip hidden files and ignored extensions
            if f.name.startswith("."):
                continue
            if f.suffix.lower() not in _TEXT_RAG_EXTS:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                if content.strip():
                    rel_path = str(f.relative_to(self.rag_dir))
                    docs.append({"content": content, "source": rel_path})
            except Exception:
                pass

        if not docs:
            return None

        ASCIIColors.info(f"[Handbag] RAG: {len(docs)} documents indexed from rag/")

        # Try safestore for better semantic search
        safestore_fn = self._try_safestore_rag(docs)
        if safestore_fn is not None:
            return safestore_fn

        # Fallback: keyword-based scoring
        return self._build_keyword_rag(docs)

    def _try_safestore_rag(self, docs: List[Dict[str, str]]) -> Optional[Callable]:
        """Attempts to use safestore for semantic RAG. Returns None if unavailable."""
        try:
            import pipmaster as pm
            pm.ensure_installed("safestore")
            from safestore.safestore import Safestore

            db_path = self.memory_dir / "rag_index.db" if self.memory_dir.exists() else self.path / "rag_index.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            store = Safestore(db_path=str(db_path))
            store.load()

            # Index documents if not already indexed
            for doc in docs:
                chunk_id = f"handbag_rag_{doc['source']}"
                if not store.exists(chunk_id):
                    store.add(text=doc["content"], metadata={"source": doc["source"]}, doc_id=chunk_id)
            store.save()

            def _safestore_query(query: str) -> Dict[str, Any]:
                try:
                    results = store.search(query, top_k=5)
                    sources = []
                    for r in results:
                        sources.append({
                            "content": r.get("text", ""),
                            "score": float(r.get("score", 1.0)),
                            "source": r.get("metadata", {}).get("source", "rag"),
                        })
                    return {
                        "success": bool(sources),
                        "sources": sources,
                        "count": len(sources),
                        "query": query,
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "sources": [],
                        "count": 0,
                        "query": query,
                        "error": str(e),
                    }

            ASCIIColors.success("[Handbag] RAG: Using safestore for semantic search.")
            return _safestore_query

        except Exception:
            return None

    def _build_keyword_rag(self, docs: List[Dict[str, str]]) -> Callable[[str], Dict[str, Any]]:
        """Builds a simple keyword-based RAG callable."""

        def _keyword_query(query: str) -> Dict[str, Any]:
            query_lower = query.lower()
            query_words = set(re.findall(r"\b\w{3,}\b", query_lower)) - _STOP_WORDS

            if not query_words:
                # If no meaningful words, return first few docs
                sources = [
                    {"content": d["content"][:4000], "score": 0.1, "source": d["source"]}
                    for d in docs[:3]
                ]
                return {"success": bool(sources), "sources": sources, "count": len(sources), "query": query}

            scored = []
            for doc in docs:
                content_lower = doc["content"].lower()
                # Count how many query words appear in the document
                hits = sum(1 for w in query_words if w in content_lower)
                if hits > 0:
                    # Normalize by document length to avoid bias toward long docs
                    score = hits / max(len(content_lower) / 1000, 1)
                    scored.append((score, doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            sources = [
                {
                    "content": d["content"][:4000],
                    "score": round(s, 4),
                    "source": d["source"],
                }
                for s, d in scored[:5]
            ]
            return {
                "success": bool(sources),
                "sources": sources,
                "count": len(sources),
                "query": query,
            }

        ASCIIColors.info("[Handbag] RAG: Using keyword-based search (safestore not available).")
        return _keyword_query

    # ------------------------------------------------------------------ memory

    def create_memory_manager(self) -> Optional[Any]:
        """Creates a LollmsMemoryManager from the handbag's memory directory."""
        if not self._memory_db_path:
            return None
        try:
            from lollms_client.lollms_memory import LollmsMemoryManager, MemoryConfig

            db_file = self._memory_db_path.replace("sqlite:///", "")
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)

            manager = LollmsMemoryManager(
                db_path=self._memory_db_path,
                owner_id=f"handbag_{self.path.name}",
                config=MemoryConfig(working_token_budget=2000),
            )
            ASCIIColors.success(f"[Handbag] Memory manager created from '{db_file}'.")
            return manager
        except Exception as e:
            ASCIIColors.warning(f"[Handbag] Failed to create memory manager: {e}")
            return None

    # ------------------------------------------------------------------ accessors

    def get_default_personality(self) -> Optional[Any]:
        """Returns the default LollmsPersonality from the handbag."""
        if not self._personalities:
            return None
        # Use manifest-specified default, else first personality
        if self._default_personality_name and self._default_personality_name in self._personalities:
            return self._personalities[self._default_personality_name]
        return next(iter(self._personalities.values()))

    def get_personalities(self) -> Dict[str, Any]:
        """Returns all loaded personalities as {name: LollmsPersonality}."""
        return dict(self._personalities)

    def get_personality(self, name: str) -> Optional[Any]:
        """Returns a specific personality by name, or None if not found."""
        return self._personalities.get(name)

    def get_tool_files(self) -> List[str]:
        """Returns the list of tool file paths as strings."""
        return [str(f) for f in self._tool_files]

    def get_skills_dirs(self) -> List[str]:
        """Returns the list of skills directory paths as strings."""
        return [str(d) for d in self._skills_dirs]

    def get_skills_mode(self) -> Optional[str]:
        """Returns the skills mode from the manifest, or None."""
        return self.manifest.get("skills_mode")

    def get_rag_data_source(self) -> Optional[Callable[[str], Dict[str, Any]]]:
        """Returns the RAG data source callable, or None if no RAG configured."""
        return self._rag_data_source

    def get_memory_db_path(self) -> Optional[str]:
        """Returns the SQLite DB path for memory, or None."""
        return self._memory_db_path

    def get_workspace_path(self) -> Optional[str]:
        """Returns the workspace path if the workspace/ directory exists."""
        if self.workspace_dir.exists():
            return str(self.workspace_dir)
        return None

    # ------------------------------------------------------------------ helpers

    def attach_rag_to_personality(self, personality: Any) -> None:
        """
        Attaches the handbag's RAG data source to a personality if it doesn't
        already have one. This allows the personality to benefit from the
        handbag's shared RAG knowledge.
        """
        if self._rag_data_source is None:
            return
        if personality is None:
            return
        # Check if personality already has data
        has_data = getattr(personality, "has_data", False)
        if has_data:
            return  # Personality already has its own RAG, don't override
        # Attach our RAG
        try:
            personality.data_source = self._rag_data_source
            ASCIIColors.info("[Handbag] Attached RAG data source to personality.")
        except Exception as e:
            ASCIIColors.warning(f"[Handbag] Failed to attach RAG to personality: {e}")

    # ------------------------------------------------------------------ factory

    @staticmethod
    def create_structure(handbag_path: Union[str, Path], name: str = "My Agent Handbag") -> Path:
        """
        Creates a new handbag folder structure on disk.

        Args:
            handbag_path: Path where the handbag folder should be created.
            name: Name for the handbag manifest.

        Returns:
            Path to the created handbag folder.
        """
        hb_path = Path(handbag_path)
        hb_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["personalities", "tools", "skills", "rag", "memory", "workspace"]:
            (hb_path / subdir).mkdir(exist_ok=True)

        # Create manifest
        manifest_path = hb_path / "handbag.yaml"
        if not manifest_path.exists():
            try:
                import yaml
                manifest = {
                    "name": name,
                    "version": "1.0",
                    "description": "A handbag containing all agent resources.",
                    "default_personality": None,
                    "skills_mode": "mixed",
                }
                manifest_path.write_text(
                    yaml.dump(manifest, default_flow_style=False, sort_keys=False),
                    encoding="utf-8",
                )
            except ImportError:
                # No yaml available, write a simple text file
                manifest_path.write_text(
                    f"name: {name}\nversion: '1.0'\nskills_mode: mixed\n",
                    encoding="utf-8",
                )

        # Create a README
        readme_path = hb_path / "README.md"
        if not readme_path.exists():
            readme_path.write_text(
                f"# {name}\n\n"
                "This is a Lollms Agent Handbag — a self-contained folder with all agent resources.\n\n"
                "## Structure\n\n"
                "- `personalities/` — Personality bundles (subdirs with SOUL.md)\n"
                "- `tools/` — Extra LCP tool files (.py)\n"
                "- `skills/` — SKILL.md files (agent creates/updates these over time)\n"
                "- `rag/` — Text documents for retrieval-augmented generation\n"
                "- `memory/` — SQLite memory database\n"
                "- `workspace/` — Isolated working directory\n"
                "- `handbag.yaml` — Manifest with configuration\n",
                encoding="utf-8",
            )

        ASCIIColors.success(f"[Handbag] Created structure at '{hb_path}'")
        return hb_path

    # ------------------------------------------------------------------ repr

    def __repr__(self) -> str:
        return (
            f"Handbag(path='{self.path.name}', "
            f"personalities={len(self._personalities)}, "
            f"tools={len(self._tool_files)}, "
            f"skills_dirs={len(self._skills_dirs)}, "
            f"rag={'yes' if self._rag_data_source else 'no'}, "
            f"memory={'yes' if self._memory_db_path else 'no'})"
        )
