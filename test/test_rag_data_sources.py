"""Regression tests for RAG data source handling.

RAG tools are optional. When no RAG sources are configured, the personality
must return an empty tool list silently (no WARNING noise).
"""

import logging
from unittest.mock import MagicMock

from lollms_client.lollms_personality.lollms_personality import LollmsPersonality


def test_get_rag_tools_silent_when_no_sources():
    """get_rag_tools must return [] silently when no RAG sources are configured.

    RAG tools are optional; the absence of extra_knowledge_bases / extra_safe_stores
    must NOT emit a WARNING (it is a normal, expected state).
    """
    personality = LollmsPersonality.__new__(LollmsPersonality)
    # Minimal state required by `has_data` and `build_rag_tools` when bypassing __init__.
    personality.data_sources = []
    personality._raw_data_source = None
    personality.query_rag_callback = None
    personality.data_files = []
    personality.extra_knowledge_bases = []
    personality.extra_safe_stores = []
    personality._lollms_client = MagicMock()
    personality._lollms_client.rag = None
    personality._lollms_client.rag_model_profiles_registry = None

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        tools = LollmsPersonality.build_rag_tools(personality)
    finally:
        root.removeHandler(handler)

    assert tools == {}, "build_rag_tools must return an empty dict when no RAG sources exist"
    rag_warnings = [r for r in records if "NO RAG tools" in r.getMessage()]
    assert not rag_warnings, (
        "build_rag_tools must NOT emit a WARNING when no RAG sources are configured; "
        f"got: {[r.getMessage() for r in rag_warnings]}"
    )