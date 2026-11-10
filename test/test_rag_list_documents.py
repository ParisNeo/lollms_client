"""Tests for ``LollmsRAGBinding.list_documents`` and the SafeStore implementation.

Coverage matrix
---------------
1. Abstract contract: a bare ``LollmsRAGBinding`` subclass that does NOT override
   ``list_documents`` must return ``[]`` (never raise) — callers rely on the
   empty-list contract for bindings that cannot enumerate their contents.
2. SafeStore SQL strategy: given a database whose schema mirrors safe_store's
   (``documents`` + ``chunks`` tables), ``list_documents`` returns one normalised
   entry per document with ``doc_id``, ``document_title``, ``file_path``,
   ``chunk_count`` and ``metadata``.
3. SafeStore fallback strategy: when the live store connection is unavailable,
   the method falls back to ``get_database_info()['documents']['list']``.
4. SafeStore resilience: a store whose ``get_store()`` raises must yield ``[]``
   (never raise to the caller).
5. End-to-end (optional): when the real ``safe_store`` package is installed, a
   freshly ingested text document appears in ``list_documents``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from lollms_client.lollms_rag_binding import LollmsRAGBinding
from lollms_client.rag_bindings.safe_store import SafeStoreRAGBinding


# ── 1. Abstract contract ────────────────────────────────────────────────────


class _MinimalRAGBinding(LollmsRAGBinding):
    """Bare concrete binding that satisfies the abstract methods but does NOT
    override ``list_documents`` — exercises the base-class default."""

    def _query_dense_impl(self, query_text, top_k=5, min_relevance_percent=0.0, **kwargs):
        return []

    def hybrid_query(self, query_text, top_k=5, dense_weight=0.5, bm25_weight=0.5,
                     min_relevance_percent=0.0, **kwargs):
        return []

    def add_document(self, file_path, metadata=None, **kwargs):
        return False

    def add_text(self, text, unique_id=None, metadata=None, **kwargs):
        return False


def test_abstract_list_documents_returns_empty_list():
    """Bindings that do not override ``list_documents`` must return ``[]``."""
    binding = _MinimalRAGBinding(binding_name="minimal", store_name="dummy.db")
    assert binding.list_documents() == []
    # Must always be a fresh list instance, never shared mutable state.
    assert binding.list_documents() is not binding.list_documents()


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sqlite_safe_store_db(tmp_path: Path) -> Path:
    """Creates a minimal SQLite file whose schema mirrors safe_store's core
    ``documents`` and ``chunks`` tables, seeded with two documents."""
    db_path = tmp_path / "test_store.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                metadata TEXT,
                full_text TEXT
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_seq INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            );
            """
        )
        conn.execute(
            "INSERT INTO documents (file_path, metadata, full_text) VALUES (?, ?, ?)",
            ("docs/intro.md", '{"title": "Introduction"}', "intro text"),
        )
        conn.execute(
            "INSERT INTO documents (file_path, metadata, full_text) VALUES (?, ?, ?)",
            ("/abs/path/report.pdf", None, "report text"),
        )
        for doc_id, chunk_count in ((1, 3), (2, 1)):
            for seq in range(chunk_count):
                conn.execute(
                    "INSERT INTO chunks (doc_id, chunk_seq, chunk_text) VALUES (?, ?, ?)",
                    (doc_id, seq, f"chunk {seq} of {doc_id}"),
                )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _build_binding_with_stub_store(
    db_path: Path,
    *,
    store_factory: Any = None,
) -> SafeStoreRAGBinding:
    """Constructs a SafeStoreRAGBinding without booting safe_store, and injects a
    stub ``_store_instance`` so ``get_store()`` returns it directly."""
    binding = SafeStoreRAGBinding.__new__(SafeStoreRAGBinding)
    LollmsRAGBinding.__init__(binding, binding_name="safe_store", db_path=str(db_path))
    binding.db_path = db_path
    binding.store_name = str(db_path)
    binding.vectorizer_name = "st"
    binding.chunk_size = 128
    binding.encryption_key = None
    binding._store_instance = store_factory if store_factory is not None else None
    binding._graph_instance = None
    return binding


class _StubContextStore:
    """Minimal context-manager store exposing a live sqlite connection."""

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.conn.commit()
        finally:
            self.conn.close()
        return False

    def get_database_info(self) -> Dict[str, Any]:
        return {"documents": {"list": []}}


# ── 2. SQL strategy ────────────────────────────────────────────────────────


def test_list_documents_sql_strategy(sqlite_safe_store_db: Path):
    """The direct-SQL strategy enumerates every indexed document with chunk
    counts, metadata and a best-effort title."""
    binding = _build_binding_with_stub_store(sqlite_safe_store_db)
    binding._store_instance = _StubContextStore(sqlite_safe_store_db)

    docs = binding.list_documents()

    assert len(docs) == 2
    by_id = {d["doc_id"]: d for d in docs}

    assert by_id[1]["document_title"] == "Introduction"  # from metadata
    assert by_id[1]["file_path"] == "docs/intro.md"
    assert by_id[1]["chunk_count"] == 3
    assert by_id[1]["metadata"] == {"title": "Introduction"}

    assert by_id[2]["document_title"] == "report.pdf"  # filename fallback
    assert by_id[2]["file_path"] == "/abs/path/report.pdf"
    assert by_id[2]["chunk_count"] == 1
    assert by_id[2]["metadata"] == {}


def test_list_documents_empty_database(sqlite_safe_store_db: Path, tmp_path: Path):
    """An empty database returns ``[]`` — not an error."""
    empty_db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(empty_db))
    try:
        conn.executescript(
            """
            CREATE TABLE documents (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    file_path TEXT NOT NULL, metadata TEXT,
                                    full_text TEXT);
            CREATE TABLE chunks (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                 doc_id INTEGER NOT NULL, chunk_seq INTEGER NOT NULL,
                                 chunk_text TEXT NOT NULL);
            """
        )
        conn.commit()
    finally:
        conn.close()

    binding = _build_binding_with_stub_store(empty_db)
    binding._store_instance = _StubContextStore(empty_db)

    assert binding.list_documents() == []


# ── 3. Fallback strategy ──────────────────────────────────────────────────


def test_list_documents_falls_back_to_get_database_info(sqlite_safe_store_db: Path):
    """When no live sqlite connection is reachable, ``list_documents`` falls
    back to the documented ``get_database_info()['documents']['list']`` shape."""
    binding = _build_binding_with_stub_store(sqlite_safe_store_db)

    class _NoConnStore:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_database_info(self):
            return {
                "documents": {
                    "list": [
                        {"document_title": "alpha.md", "chunk_count": 5, "doc_id": 10},
                        {"title": "beta.pdf", "chunk_count": 2},
                        {"file_path": "gamma/code.py"},  # title falls back to path
                        "not a dict — must be skipped",
                    ]
                }
            }

    binding._store_instance = _NoConnStore()

    docs = binding.list_documents()

    assert len(docs) == 3
    assert docs[0]["document_title"] == "alpha.md"
    assert docs[0]["chunk_count"] == 5
    assert docs[0]["doc_id"] == 10
    assert docs[1]["document_title"] == "beta.pdf"
    assert docs[2]["document_title"] == "gamma/code.py"
    # Entries that are not dicts are skipped, never crash.
    assert all(isinstance(d, dict) for d in docs)


# ── 4. Resilience ─────────────────────────────────────────────────────────


def test_list_documents_returns_empty_when_store_raises(tmp_path: Path):
    """A broken store must yield ``[]`` — ``list_documents`` never raises."""
    binding = _build_binding_with_stub_store(tmp_path / "broken.db")
    binding._store_instance = MagicMock()
    binding._store_instance.__enter__.side_effect = RuntimeError("store down")

    assert binding.list_documents() == []


def test_list_documents_returns_empty_when_get_store_raises(tmp_path: Path):
    """If ``get_store()`` itself fails, ``list_documents`` returns ``[]``."""
    binding = _build_binding_with_stub_store(tmp_path / "broken.db")

    def _boom():
        raise RuntimeError("cannot open database")

    binding.get_store = _boom  # type: ignore[method-assign]
    assert binding.list_documents() == []


# ── 5. End-to-end (optional) ──────────────────────────────────────────────


def test_list_documents_end_to_end_with_real_safe_store(tmp_path: Path):
    """End-to-end: ingest a text document, then confirm it appears in the list.

    Skipped when safe_store or its embedding stack is unavailable in the
    environment (CI/dev machines without the ML dependencies).
    """
    pytest.importorskip("safe_store", reason="safe_store package not installed")

    db_path = tmp_path / "e2e.db"
    try:
        binding = SafeStoreRAGBinding(
            db_path=str(db_path),
            vectorizer_name="tfidf",
            chunk_size=32,
            chunk_overlap=4,
            use_shared_server=False,
        )
        store = binding.get_store()
        with store:
            store.add_text(
                unique_id="hello.md",
                text="# Greeting\n\nHello world, this is a smoke test.",
            )
    except Exception as exc:  # noqa: BLE001 — environment-dependent skip
        pytest.skip(f"safe_store unavailable in this environment: {exc}")

    docs = binding.list_documents()
    assert isinstance(docs, list)
    titles = [d["document_title"] for d in docs]
    assert any("hello" in t.lower() for t in titles), (
        f"expected the ingested document in list_documents, got {docs!r}"
    )