from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

import pipmaster as pm
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_rag_binding import LollmsRAGBinding

BindingName = "SafeStoreRAGBinding"


class SafeStoreRAGBinding(LollmsRAGBinding):
    """
    RAG & Knowledge Store binding for safe_store: The Local Multi-Modal Vector,
    Graph & Semantic Engine.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(binding_name="safe_store", **kwargs)

        pm.ensure_packages(["safe_store"])

        raw_db_path = (
            kwargs.get("db_path")
            or kwargs.get("store_name")
            or kwargs.get("model_name")
            or "knowledge.db"
        )
        self.db_path = self.resolve_system_path(raw_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_name = str(self.db_path)

        self.vectorizer_name = kwargs.get("vectorizer_name", "st")
        self.chunk_size = int(kwargs.get("chunk_size", 128))
        self.chunk_overlap = int(kwargs.get("chunk_overlap", 16))
        self.chunking_strategy = kwargs.get("chunking_strategy", "token")
        self.encryption_key = kwargs.get("encryption_key") or None

        vectorizer_config = kwargs.get("vectorizer_config", {})
        if "model" not in vectorizer_config:
            m_name = kwargs.get("model_name") or kwargs.get("model")
            if m_name and m_name != str(raw_db_path):
                vectorizer_config["model"] = m_name

        if kwargs.get("use_shared_server"):
            vectorizer_config["use_shared_server"] = True
            vectorizer_config["port"] = int(kwargs.get("port", 8765))

        self.vectorizer_config = vectorizer_config

        self._store_instance = None
        self._graph_instance = None
        self._llm_generator: Optional[Callable] = kwargs.get("llm_generator")
        self._lollms_client = kwargs.get("lollms_client")

    def _build_llm_callable(self) -> Optional[Callable]:
        if self._llm_generator and callable(self._llm_generator):
            return self._llm_generator

        if self._lollms_client:
            client = self._lollms_client

            def _client_generator(prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False, **gen_kw) -> str:
                try:
                    kwargs_call: Dict[str, Any] = {"temperature": 0.1}
                    if json_mode:
                        kwargs_call["response_format"] = {"type": "json_object"}
                    kwargs_call.update(gen_kw)

                    if hasattr(client, "generate_text"):
                        res = client.generate_text(prompt=prompt, system_prompt=system_prompt or "", **kwargs_call)
                        return str(res) if res is not None else ""
                    elif hasattr(client, "generate"):
                        res = client.generate(prompt=prompt, system_prompt=system_prompt or "", **kwargs_call)
                        return str(res) if res is not None else ""
                except Exception as ex:
                    ASCIIColors.warning(f"[SafeStoreRAG] LLM generation bridge failed: {ex}")
                return ""

            return _client_generator

        return None

    def get_store(self):
        """Lazily instantiates the SafeStore database connection."""
        if self._store_instance is not None:
            return self._store_instance

        try:
            import safe_store
            llm_call = self._build_llm_callable()

            self._store_instance = safe_store.SafeStore(
                db_path=str(self.db_path),
                vectorizer_name=self.vectorizer_name,
                vectorizer_config=self.vectorizer_config,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                chunking_strategy=self.chunking_strategy,
                encryption_key=self.encryption_key,
                llm_generator=llm_call,
            )
            return self._store_instance
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"[SafeStoreRAG] Failed to initialize SafeStore on {self.db_path}: {e}")
            raise

    def get_graph(self):
        """Lazily instantiates the GraphStore connection."""
        if self._graph_instance is not None:
            return self._graph_instance

        try:
            import safe_store
            store = self.get_store()
            llm_call = self._build_llm_callable()
            self._graph_instance = safe_store.GraphStore(
                store=store,
                llm_executor_callback=llm_call,
            )
            return self._graph_instance
        except Exception as e:
            trace_exception(e)
            ASCIIColors.warning(f"[SafeStoreRAG] GraphStore initialization failed: {e}")
            return None

    # ── Retrieval Implementation ─────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        reconstruct_overlapping_chunks: bool = True,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        store = self.get_store()
        try:
            with store:
                results = store.query(
                    query_text=query_text,
                    top_k=top_k,
                    reconstruct_overlapping_chunks=reconstruct_overlapping_chunks,
                    **kwargs
                )
            return self._normalize_results(results)
        except Exception as e:
            trace_exception(e)
            return []

    def hybrid_query(
        self,
        query_text: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        min_relevance_percent: float = 0.0,
        reconstruct_overlapping_chunks: bool = True,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        store = self.get_store()
        try:
            with store:
                results = store.hybrid_query(
                    query_text=query_text,
                    top_k=top_k,
                    dense_weight=dense_weight,
                    bm25_weight=bm25_weight,
                    min_relevance_percent=min_relevance_percent,
                    **kwargs
                )
                if reconstruct_overlapping_chunks and hasattr(store, "reconstruct_overlapping_chunks"):
                    results = store.reconstruct_overlapping_chunks(results, add_metadata=True)
            return self._normalize_results(results)
        except Exception as e:
            trace_exception(e)
            return []

    def _normalize_results(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for r in raw_results:
            text = r.get("chunk_text") or r.get("content") or r.get("text") or ""
            doc_title = r.get("document_title") or r.get("title") or r.get("file_path") or "Knowledge Document"
            score = r.get("relevance_score", r.get("score", 1.0))
            if isinstance(score, (int, float)) and score > 1.0:
                score_norm = score / 100.0
            else:
                score_norm = float(score)

            item = {
                "content": text,
                "score": score_norm,
                "relevance_percent": r.get("relevance_score", round(score_norm * 100, 1)),
                "source": doc_title,
                "title": doc_title,
                "metadata": r.get("document_metadata", r.get("metadata", {})),
                "store_name": self.store_name,
            }
            if "chunk_seqs" in r:
                item["chunk_seqs"] = r["chunk_seqs"]
            normalized.append(item)
        return normalized

    # ── Ingestion Implementation ─────────────────────────────────────────────

    def add_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> bool:
        store = self.get_store()
        p = Path(file_path).resolve()
        if not p.exists():
            ASCIIColors.warning(f"[SafeStoreRAG] Document not found on disk: {p}")
            return False

        try:
            with store:
                store.add_document(p, metadata=metadata or {}, **kwargs)
            ASCIIColors.success(f"[SafeStoreRAG] Successfully ingested document: '{p.name}'")
            return True
        except Exception as e:
            trace_exception(e)
            return False

    def add_text(
        self,
        text: str,
        unique_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> bool:
        store = self.get_store()
        try:
            with store:
                uid = unique_id or f"text_{hash(text)}"
                store.add_text(unique_id=uid, text=text, metadata=metadata or {}, **kwargs)
            return True
        except Exception as e:
            trace_exception(e)
            return False

    # ── Context Windows & Full Document Retrieval ─────────────────────────────

    def query_full_documents(
        self,
        query_text: str,
        top_k_docs: int = 3,
        min_relevance_percent: float = 40.0,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        store = self.get_store()
        try:
            with store:
                if hasattr(store, "query_full_documents"):
                    return store.query_full_documents(
                        query_text=query_text,
                        top_k_docs=top_k_docs,
                        min_relevance_percent=min_relevance_percent,
                        **kwargs
                    )
        except Exception as e:
            trace_exception(e)
        return []

    def query_document_content_window(
        self,
        query_text: str,
        top_k_hits: int = 3,
        window_before: int = 1,
        window_after: int = 1,
        min_relevance_percent: float = 40.0,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        store = self.get_store()
        try:
            with store:
                if hasattr(store, "query_document_content_window"):
                    return store.query_document_content_window(
                        query_text=query_text,
                        top_k_hits=top_k_hits,
                        window_before=window_before,
                        window_after=window_after,
                        min_relevance_percent=min_relevance_percent,
                        **kwargs
                    )
        except Exception as e:
            trace_exception(e)
        return []

    # ── Knowledge Graph & SPARQL Implementation ──────────────────────────────

    def query_sparql(
        self,
        sparql_query: str,
        enable_reasoning: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        graph = self.get_graph()
        if not graph:
            return {"head": {"vars": []}, "results": {"bindings": []}, "error": "Knowledge graph not initialized"}

        try:
            return graph.query_sparql(sparql_query, enable_reasoning=enable_reasoning, **kwargs)
        except Exception as e:
            trace_exception(e)
            return {"head": {"vars": []}, "results": {"bindings": []}, "error": str(e)}

    def execute_sparql_update(
        self,
        sparql_update: str,
        **kwargs: Any
    ) -> bool:
        graph = self.get_graph()
        if not graph:
            return False

        try:
            return bool(graph.execute_sparql_update(sparql_update, **kwargs))
        except Exception as e:
            trace_exception(e)
            return False

    def query_graph_hybrid(
        self,
        query_text: str,
        top_k: int = 5,
        **kwargs: Any
    ) -> Dict[str, Any]:
        graph = self.get_graph()
        if not graph:
            return {"ranked_chunks": [], "subgraph": {"nodes": [], "relationships": []}}

        try:
            return graph.query_graph_hybrid(query_text, top_k=top_k, **kwargs)
        except Exception as e:
            trace_exception(e)
            return {"ranked_chunks": [], "subgraph": {"nodes": [], "relationships": []}, "error": str(e)}

    # ── Introspection & Diagnostics ──────────────────────────────────────────

    def get_database_info(self) -> Dict[str, Any]:
        store = self.get_store()
        try:
            with store:
                if hasattr(store, "get_database_info"):
                    return store.get_database_info()
        except Exception:
            pass

        return {
            "store_name": str(self.db_path),
            "vectorizer": self.vectorizer_name,
            "chunk_size": self.chunk_size,
            "encrypted": bool(self.encryption_key)
        }

    def info(self) -> None:
        store = self.get_store()
        try:
            with store:
                if hasattr(store, "info"):
                    store.info()
                    return
        except Exception:
            pass
        super().info()

    def revectorize_database(
        self,
        new_vectorizer_name: str,
        new_vectorizer_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> bool:
        store = self.get_store()
        try:
            with store:
                if hasattr(store, "revectorize_database"):
                    store.revectorize_database(
                        new_vectorizer_name=new_vectorizer_name,
                        new_vectorizer_config=new_vectorizer_config or {},
                        **kwargs
                    )
                    return True
        except Exception as e:
            trace_exception(e)
        return False

    def export_db_file(self, output_path: Union[str, Path]) -> bool:
        store = self.get_store()
        try:
            with store:
                if hasattr(store, "export_db_file"):
                    store.export_db_file(str(output_path))
                    return True
        except Exception as e:
            trace_exception(e)
        return False

    def cluster_documents(self, method: str = "kmeans", **kwargs: Any) -> List[Dict[str, Any]]:
        store = self.get_store()
        try:
            with store:
                if hasattr(store, "cluster_documents"):
                    return store.cluster_documents(method=method, **kwargs)
        except Exception as e:
            trace_exception(e)
        return []

    def get_datalake_view(self, method: str = "umap", n_components: int = 2, **kwargs: Any) -> Any:
        store = self.get_store()
        try:
            with store:
                if hasattr(store, "get_datalake_view"):
                    return store.get_datalake_view(method=method, n_components=n_components, **kwargs)
        except Exception as e:
            trace_exception(e)
        return []

    def export_datalake_html(self, output_file: str, method: str = "umap", n_components: int = 3, **kwargs: Any) -> bool:
        store = self.get_store()
        try:
            with store:
                if hasattr(store, "export_datalake_html"):
                    store.export_datalake_html(output_file=output_file, method=method, n_components=n_components, **kwargs)
                    return True
        except Exception as e:
            trace_exception(e)
        return False