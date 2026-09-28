from __future__ import annotations

from abc import abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

try:
    import yaml
except ImportError:
    yaml = None

from ascii_colors import trace_exception, ASCIIColors
from lollms_client.lollms_base_binding import LollmsBaseBinding

__all__ = ["LollmsRAGBinding", "LollmsRAGBindingManager"]


class LollmsRAGBinding(LollmsBaseBinding):
    """
    Abstract base class for all LOLLMS RAG (Retrieval-Augmented Generation) & Knowledge Store bindings.
    In the two-tier profile architecture, RAG model profiles represent data stores
    (e.g., SQLite vector/graph databases, indices, or knowledge vaults).
    """

    def __init__(
        self,
        binding_name: str = "unknown",
        debug: Optional[bool] = False,
        **kwargs: Any
    ):
        super().__init__(binding_name=binding_name, debug=debug, **kwargs)
        self.store_name: Optional[str] = (
            kwargs.get("store_name")
            or kwargs.get("model_name")
            or kwargs.get("db_path")
        )
        self.config: Dict[str, Any] = kwargs

    # ── Core Retrieval Contract ───────────────────────────────────────────────

    @abstractmethod
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Executes a dense vector similarity query against the active store.
        Returns a list of standardized chunk hit dictionaries.
        """
        pass

    @abstractmethod
    def hybrid_query(
        self,
        query_text: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        min_relevance_percent: float = 0.0,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Executes a tri-modal / hybrid search combining dense vectors and sparse lexical search (e.g., BM25).
        """
        pass

    # ── Ingestion Contract ───────────────────────────────────────────────────

    @abstractmethod
    def add_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> bool:
        """
        Ingests and chunks a file (PDF, DOCX, XLSX, CSV, Markdown, Code, etc.) into the store.
        """
        pass

    @abstractmethod
    def add_text(
        self,
        text: str,
        unique_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> bool:
        """
        Ingests a raw text snippet into the store with optional unique ID and metadata.
        """
        pass

    # ── Advanced Context & Neighborhood Contract ──────────────────────────────

    def query_full_documents(
        self,
        query_text: str,
        top_k_docs: int = 3,
        min_relevance_percent: float = 0.0,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Aggregates chunk hits up to complete document texts.
        """
        return []

    def query_document_content_window(
        self,
        query_text: str,
        top_k_hits: int = 3,
        window_before: int = 1,
        window_after: int = 1,
        min_relevance_percent: float = 0.0,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Expands chunk hits to include surrounding contiguous chunk neighborhoods.
        """
        return []

    # ── Knowledge Graph & SPARQL 1.1 Contract ─────────────────────────────────

    def query_sparql(
        self,
        sparql_query: str,
        enable_reasoning: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes a standard W3C SPARQL 1.1 query (SELECT, ASK, CONSTRUCT, DESCRIBE).
        """
        return {"head": {"vars": []}, "results": {"bindings": []}}

    def execute_sparql_update(
        self,
        sparql_update: str,
        **kwargs: Any
    ) -> bool:
        """
        Executes a SPARQL 1.1 Update (INSERT DATA, DELETE DATA, etc.).
        """
        return False

    def query_graph_hybrid(
        self,
        query_text: str,
        top_k: int = 5,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes unified Graph Subgraph + Vector + Lexical retrieval.
        """
        return {"ranked_chunks": [], "subgraph": {"nodes": [], "relationships": []}}

    # ── Introspection & Diagnostics Contract ──────────────────────────────────

    def get_database_info(self) -> Dict[str, Any]:
        """
        Returns structured diagnostic information about the store state,
        including document counts, chunks, vectorizer, and graph nodes/edges.
        """
        return {
            "store_name": self.store_name,
            "binding": self.binding_name,
            "total_documents": 0,
            "total_chunks": 0,
        }

    def info(self) -> None:
        """
        Prints a diagnostic inspection panel to stdout.
        """
        info_data = self.get_database_info()
        ASCIIColors.info(f"[{self.binding_name}] Store '{self.store_name}': {info_data}")

    def list_models(self) -> List[str]:
        """
        Lists discovered stores/databases associated with this binding.
        """
        return [self.store_name] if self.store_name else []

    def list_stores(self) -> List[str]:
        """Alias for list_models for conceptual clarity with RAG data stores."""
        return self.list_models()


class LollmsRAGBindingManager:
    """Manages discovery and instantiation of RAG bindings."""

    def __init__(
        self,
        rag_bindings_dir: Union[str, Path] = Path(__file__).parent / "rag_bindings"
    ):
        self.rag_bindings_dir = Path(rag_bindings_dir)
        self.available_bindings: Dict[str, Any] = {}

    def _load_binding(self, binding_name: str) -> None:
        binding_dir = self.rag_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.rag_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName)
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                ASCIIColors.error(f"Failed to load RAG binding '{binding_name}': {e}")

    def create_binding(
        self,
        binding_name: str,
        **kwargs: Any
    ) -> Optional[LollmsRAGBinding]:
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)

        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            return binding_class(**kwargs)
        return None

    def get_available_bindings(self) -> List[Dict[str, Any]]:
        bindings_dir = Path(self.rag_bindings_dir)
        if not bindings_dir.is_dir():
            return []

        bindings_list = []
        for binding_folder in bindings_dir.iterdir():
            if binding_folder.is_dir() and (binding_folder / "__init__.py").exists():
                binding_name = binding_folder.name
                desc_file = binding_folder / "description.yaml"
                binding_info: Dict[str, Any] = {}
                if desc_file.exists():
                    try:
                        with open(desc_file, "r", encoding="utf-8") as f:
                            binding_info = yaml.safe_load(f) if yaml else {}
                    except Exception:
                        pass
                binding_info.setdefault("binding_name", binding_name)
                binding_info.setdefault("title", binding_name.replace("_", " ").title())
                bindings_list.append(binding_info)

        return sorted(bindings_list, key=lambda b: b.get("title", b["binding_name"]))