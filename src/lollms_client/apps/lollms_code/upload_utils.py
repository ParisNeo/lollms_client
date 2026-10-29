"""
Shared upload/attachment utilities for the lollms-code app.

Pure standard library — no NiceGUI, no LollmsPersonality dependency — so both
the GUI (``gui/chat_page.py``) and the CLI (``cli.py``) can import it without
pulling in heavy optional dependencies.

Invariants enforced here:
  * Every uploaded/attached file lands under ``<workspace>/uploads/``.
  * Filenames are sanitized to ``[A-Za-z0-9._-]`` and de-duplicated with ``_N``.
  * Writes are atomic (temp file + ``os.replace``) — no half-written uploads.
  * Path traversal (``..``, ``/``, ``\\``, ``~``, NUL) is impossible.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "sanitize_upload_name",
    "uploads_dir",
    "safe_target_path",
    "write_bytes_atomic",
    "write_text_note",
    "copy_into_uploads",
    "describe_result",
]

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_UPLOADS_SUBDIR = "uploads"


def _coerce_workspace(workspace: Union[str, Path]) -> Path:
    """Return a resolved workspace Path, raising ValueError if falsy."""
    if not workspace:
        raise ValueError("Workspace path is required for uploads.")
    return Path(workspace).expanduser().resolve()


def sanitize_upload_name(name: str, *, taken: Optional[set[str]] = None) -> str:
    """
    Produce a traversal-safe, collision-resistant filename.

    ``taken`` is an optional set of already-used names (case-sensitive) used to
    append ``_1``, ``_2``, ... suffixes until a free name is found.
    """
    base = Path(str(name or "")).name  # strip any directory components
    clean = _SAFE_NAME_RE.sub("_", base).strip("._")
    if not clean:
        clean = f"upload_{int(time.time())}.bin"

    if not taken:
        return clean

    if clean not in taken:
        return clean

    stem, dot, suffix = clean.partition(".")
    if not dot:
        stem, suffix = clean, ""
    counter = 1
    while True:
        candidate = f"{stem}_{counter}{('.' + suffix) if suffix else ''}"
        if candidate not in taken:
            return candidate
        counter += 1


def uploads_dir(workspace: Union[str, Path]) -> Path:
    """Return ``<workspace>/uploads/``, creating it if needed."""
    target = _coerce_workspace(workspace) / _UPLOADS_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_target_path(workspace: Union[str, Path], original_name: str) -> Path:
    """
    Return a collision-free path INSIDE ``<workspace>/uploads/``.

    Guarantees containment via a resolve-based check — never escapes uploads/.
    """
    up_dir = uploads_dir(workspace)
    existing = {p.name for p in up_dir.iterdir() if p.is_file()}
    safe_name = sanitize_upload_name(original_name, taken=existing)
    target = (up_dir / safe_name).resolve()

    if not target.is_relative_to(up_dir):
        raise ValueError(
            f"Refusing to write outside uploads directory: {original_name!r}"
        )
    return target


def write_bytes_atomic(target: Path, data: bytes) -> Path:
    """Write ``data`` to ``target`` atomically (temp file + os.replace)."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
    return target


def write_text_note(
    workspace: Union[str, Path], text: str, *, stem: str = "note"
) -> Path:
    """Write ``text`` as ``<workspace>/uploads/<stem>_<utc>.md`` (UTF-8)."""
    up_dir = uploads_dir(workspace)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_stem = _SAFE_NAME_RE.sub("_", stem).strip("._") or "note"
    target = up_dir / f"{safe_stem}_{stamp}.md"
    # Avoid clobbering if two notes land in the same second.
    if target.exists():
        target = up_dir / f"{safe_stem}_{stamp}_{int(time.time() * 1000) % 1000:03d}.md"
    return write_bytes_atomic(target, text.encode("utf-8"))


def copy_into_uploads(
    source: Union[str, Path], workspace: Union[str, Path]
) -> Path:
    """
    Copy ``source`` into ``<workspace>/uploads/`` with a sanitized unique name.

    Raises:
        FileNotFoundError: source missing or not a regular file.
        ValueError: workspace invalid or path escapes uploads/.
        OSError: copy failure.
    """
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(
            f"Attach target not found or is not a file: {source}"
        )
    target = safe_target_path(workspace, src.name)
    shutil.copy2(src, target)
    return target


def describe_result(path: Path) -> str:
    """One-line human-readable summary for GUI/CLI notifications."""
    try:
        size = path.stat().st_size
        return f"📎 {path.name} ({size:,} bytes) → {path}"
    except OSError:
        return f"📎 {path.name} → {path}"