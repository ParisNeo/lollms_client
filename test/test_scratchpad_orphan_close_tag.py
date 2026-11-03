"""
Regression test: orphan `</artefact` / `</artifact` closing tags must never
leak to the user-visible surface (scratchpad or any other artefact).

Reproduces the reported bug:
    "The scratchpad is leaking a `</artefact`"

Two independent layers are exercised:
  1. ArtefactManager._apply_artefact_xml — strips orphan closers after
     regex substitution.
  2. lollms_chat_core.format_user_view — strips orphan closers from the
     user-facing text after rendering.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lollms_client.lollms_chat_core import format_user_view


@pytest.fixture
def artefact_manager_stub(monkeypatch):
    """Instantiate a minimal ArtefactManager without touching disk/DB."""
    from lollms_client.lollms_artefact import lollms_artefact as la

    class _StubDiscussion:
        id = "test-disc"
        metadata = {}
        workspace_path = "."
        workspace_data_path = None
        lollmsClient = None
        discussion_data_zone = ""
        deactivated_contents: set = set()
        sequential_summaries: dict = {}

        def commit(self):
            pass

        def touch(self):
            pass

        def get_discussion_images(self):
            return []

    manager = la.ArtefactManager(  # type: ignore[arg-type]
        _StubDiscussion(), skip_sync_on_load=True
    )
    return manager, la


# ── Layer 1: orphan closer inside a scratchpad body ────────────────────────
def test_scratchpad_orphan_closing_tag_stripped_from_surface(artefact_manager_stub):
    manager, la = artefact_manager_stub

    # Simulates the leak: the scratchpad artefact body has been processed,
    # but an orphaned closing tag survived (no matching opening tag in
    # the buffered chunk). The cleaned text must not contain `</artefact`.
    leaked_text = (
        "I'll process the pages in batches: unlock a few, read them into "
        "the scratchpad, then lock them.\n\n"
        "<artefact title=\"scratchpad\" type=\"scratchpad\">"
        "batch 1: pages 1-3 loaded"
        "</artefact>"
        "\n\n</artefact"  # ← the bug: orphan closer
    )

    cleaned, _affected = manager._apply_artefact_xml(
        leaked_text, default_type="scratchpad", auto_activate=False
    )

    # Sanity: opening tag itself must be stripped (as before) …
    assert "<artefact title=\"scratchpad\"" not in cleaned
    # Regression: orphan closer must also be stripped.
    assert "</artefact" not in cleaned, f"Orphan closer leaked: {cleaned!r}"
    assert "</artifact" not in cleaned


# ── Layer 2: format_user_view strips orphan artefacts too ──────────────────
def test_format_user_view_strips_orphan_closing_tag():
    raw = (
        "I'll process the pages in batches.\n\n"
        "scratchpad.md\n</artefact"
    )
    out = format_user_view(raw)
    assert "</artefact" not in out, f"Orphan closer leaked via user view: {out!r}"
    assert "I'll process the pages in batches." in out


def test_format_user_view_strips_paired_closing_tag():
    raw = "Done.\n</artifact>"
    out = format_user_view(raw)
    assert "</artifact" not in out


# ── Guard: no false positive on well-formed prefixes ───────────────────────
def test_format_user_view_keeps_legitimate_text():
    raw = "Here is a comparison: A </artefactX> is invalid text."
    out = format_user_view(raw)
    # The exact tag `</artefact>` must be stripped, but `</artefactX>`
    # (different token) should remain since it's not the artefact tag.
    assert "</artefact>" not in out
    assert "</artefactX>" in out