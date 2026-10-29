"""
CLI attachment helpers for lollms-code.

Bridges the shared `upload_utils` primitives into the CLI's batch and REPL
flows. Both `attach_paths` and `materialize_attachments` are thin wrappers
that:

  1. Copy each source file into ``<workspace>/uploads/`` (sanitized name,
     collision-free) via :func:`upload_utils.copy_into_uploads`.
  2. Write each inline text as a timestamped ``.md`` note via
     :func:`upload_utils.write_text_note`.
  3. Hand the resulting absolute paths to the caller-supplied
     ``load_into_context`` callable — which MUST be the same loader the
     ``/load`` REPL command uses (or, in the current CLI, the artefact-manager
     bridge exposed by ``_attach_paths_to_context``).
  4. Print a ``✅`` line per success and a ``⚠️`` line per failure, then
     return the count of successfully materialized attachments.

The caller is responsible for the actual context-loading semantics; this
module never touches the personality directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Union

from ascii_colors import ASCIIColors

from lollms_client.apps.lollms_code.upload_utils import (
    copy_into_uploads,
    describe_result,
    write_text_note,
)


def attach_paths(
    workspace: Union[str, Path],
    paths: Iterable[Union[str, Path]],
    load_into_context: Callable[[List[Path]], object],
) -> List[Path]:
    """
    Copy each source path into ``<workspace>/uploads/`` and load the resulting
    files into the agent's context via ``load_into_context``.

    Returns the list of successfully materialized absolute paths.
    """
    final_paths: List[Path] = []
    for raw in paths:
        if raw is None:
            continue
        try:
            target = copy_into_uploads(raw, workspace)
        except FileNotFoundError:
            ASCIIColors.red(f"⚠️  Skipped (not found): {raw}")
            continue
        except (ValueError, OSError) as exc:
            ASCIIColors.red(f"⚠️  Skipped ({exc}): {raw}")
            continue
        ASCIIColors.green(f"✅ Attached {describe_result(target)}")
        final_paths.append(target)

    if final_paths:
        try:
            load_into_context(final_paths)
        except Exception as exc:  # noqa: BLE001 — surface, don't abort the batch
            ASCIIColors.red(f"⚠️  Context load failed: {exc}")

    return final_paths


def materialize_attachments(
    workspace: Union[str, Path],
    attach_files: Iterable[Union[str, Path]],
    attach_texts: Iterable[str],
    load_into_context: Callable[[List[Path]], object],
) -> int:
    """
    Materialize both file attachments and inline text notes into
    ``<workspace>/uploads/``, then load them into context in a single batch.

    Returns the number of successfully materialized attachments (files + notes).
    """
    final_paths: List[Path] = []

    # 1. File attachments
    for raw in attach_files or []:
        if raw is None:
            continue
        try:
            target = copy_into_uploads(raw, workspace)
        except FileNotFoundError:
            ASCIIColors.red(f"⚠️  Skipped (not found): {raw}")
            continue
        except (ValueError, OSError) as exc:
            ASCIIColors.red(f"⚠️  Skipped ({exc}): {raw}")
            continue
        ASCIIColors.green(f"✅ Attached {describe_result(target)}")
        final_paths.append(target)

    # 2. Inline text notes
    for text in attach_texts or []:
        if text is None or not str(text).strip():
            continue
        try:
            target = write_text_note(workspace, str(text).strip(), stem="attach")
        except (ValueError, OSError) as exc:
            ASCIIColors.red(f"⚠️  Skipped note ({exc})")
            continue
        ASCIIColors.green(f"✅ Attached {describe_result(target)}")
        final_paths.append(target)

    # 3. Single batch load into context
    if final_paths:
        try:
            load_into_context(final_paths)
        except Exception as exc:  # noqa: BLE001 — surface, don't abort the batch
            ASCIIColors.red(f"⚠️  Context load failed: {exc}")

    return len(final_paths)