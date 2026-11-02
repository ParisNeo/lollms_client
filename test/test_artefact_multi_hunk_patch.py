"""
Tests for multi-hunk Aider patch application with partial-success semantics.

Covers:
  - Partial success: some hunks match, some don't → applied hunks persisted,
    failed hunks reported with a reprompt.
  - All hunks fail → success=False, reprompt present, content unchanged.
  - All hunks succeed → applied_hunks == N, partial=False, no reprompt.
  - No sentinels → falls through to full-content behavior (unchanged).
  - Empty SEARCH block → reported as failed hunk.
"""
import pytest

from lollms_client.lollms_artefact.lollms_artefact import ArtefactManager


def _hunk(search: str, replace: str) -> str:
    return (
        "<<<<<<< SEARCH\n"
        f"{search}\n"
        "=======\n"
        f"{replace}\n"
        ">>>>>>> REPLACE\n"
    )


def test_partial_hunk_application():
    original = "line one\nline two\nline three\n"
    patch = (
        _hunk("line one", "LINE ONE")
        + _hunk("this text does not exist", "SHOULD NOT APPLY")
        + _hunk("line three", "LINE THREE")
    )

    result = ArtefactManager.apply_aider_patch_ex(original, patch)

    assert result["success"] is True
    assert result["applied_hunks"] == 2
    assert len(result["failed_hunks"]) == 1
    assert result["partial"] is True
    assert result["reprompt"] is not None
    assert "this text does not exist" in result["reprompt"]
    # Applied replaces are present, failed one is not.
    assert "LINE ONE" in result["content"]
    assert "LINE THREE" in result["content"]
    assert "SHOULD NOT APPLY" not in result["content"]
    assert "line two" in result["content"]


def test_all_hunks_fail_reports_reprompt():
    original = "alpha\nbeta\n"
    patch = (
        _hunk("missing one", "X")
        + _hunk("missing two", "Y")
    )

    result = ArtefactManager.apply_aider_patch_ex(original, patch)

    assert result["success"] is False
    assert result["applied_hunks"] == 0
    assert len(result["failed_hunks"]) == 2
    assert result["partial"] is False
    assert result["reprompt"] is not None
    assert "missing one" in result["reprompt"]
    assert "missing two" in result["reprompt"]
    assert result["content"] == original
    assert "error" in result


def test_all_hunks_succeed():
    original = "aaa\nbbb\nccc\n"
    patch = (
        _hunk("aaa", "AAA")
        + _hunk("bbb", "BBB")
        + _hunk("ccc", "CCC")
    )

    result = ArtefactManager.apply_aider_patch_ex(original, patch)

    assert result["success"] is True
    assert result["applied_hunks"] == 3
    assert result["failed_hunks"] == []
    assert result["partial"] is False
    assert result["reprompt"] is None
    assert result["content"] == "AAA\nBBB\nCCC\n"


def test_no_sentinels_full_content():
    # No SEARCH/REPLACE sentinels → not a patch; must report failure (caller
    # falls through to full-content write path in _apply_artefact_xml).
    original = "existing content"
    result = ArtefactManager.apply_aider_patch_ex(original, "just plain text, no markers")

    assert result["success"] is False
    assert result["applied_hunks"] == 0
    assert result["content"] == original
    assert "error" in result


def test_empty_search_block_reported_as_failed():
    original = "keep me\n"
    # A hunk with an empty SEARCH body.
    patch = (
        "<<<<<<< SEARCH\n"
        "=======\n"
        "replacement\n"
        ">>>>>>> REPLACE\n"
    )

    result = ArtefactManager.apply_aider_patch_ex(original, patch)

    assert result["success"] is False
    assert result["applied_hunks"] == 0
    assert len(result["failed_hunks"]) == 1
    assert "Empty SEARCH" in result["failed_hunks"][0]["reason"]
    assert result["content"] == original


def test_backward_compatible_wrapper_raises_on_total_failure():
    original = "abc\n"
    patch = _hunk("nonexistent", "X")
    with pytest.raises(ValueError):
        ArtefactManager.apply_aider_patch(original, patch)


def test_backward_compatible_wrapper_returns_content_on_success():
    original = "abc\n"
    patch = _hunk("abc", "xyz")
    out = ArtefactManager.apply_aider_patch(original, patch)
    assert out == "xyz\n"