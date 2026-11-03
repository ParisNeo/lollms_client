"""
rag_bindings — LOLLMS Retrieval-Augmented Generation & Knowledge Store Bindings.

Hosts concrete RAG binding implementations. Each sub-package exposes a
`LollmsRAGBinding` subclass that the `LollmsRAGBindingManager` can instantiate
by name (e.g. `"safe_store"`).

Available bindings:
    - safe_store : Local multi-modal vector, graph & semantic engine
                   (SQLite-backed, supports dense/hybrid/lexical/graph retrieval,
                   SPARQL, clustering, datalake views, and encryption).

This package intentionally does not eagerly import any submodule; bindings are
loaded on demand by the RAG binding manager.
"""