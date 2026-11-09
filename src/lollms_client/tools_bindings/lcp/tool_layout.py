"""Tool library layout -- the canonical entry-point resolver.

Standard layout (mirrors the skills folder layout):

    <library_name>/
        TOOLS.py                 <- entry point (canonical)
        helper_module.py         <- optional, imported by TOOLS.py
        README.md                <- optional
        LICENSE.md               <- optional
        assets/
            icon.png             <- optional
            <other assets>       <- optional

Legacy layout still supported (backward compatible):

    <library_name>/
        <library_name>.py        <- entry point (legacy pre-TOOLS.py standard)

Resolution order (first match wins):
    1. <library_dir>/TOOLS.py                    (canonical)
    2. <library_dir>/<library_dir.name>.py       (legacy)
    3. First non-__init__ *.py sorted by name    (last-resort fallback)

A flat <folder>.py file sitting directly in a tools folder is ALSO treated
as a single-file library (its parent folder plays the role of the library dir
and its stem is used as the library name).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

# Canonical entry-point filename for tool libraries.
TOOLS_ENTRYPOINT: str = "TOOLS.py"

# Legacy entry-point filename template: <library_name>.py
LEGACY_ENTRYPOINT_TEMPLATE: str = "{library_name}.py"

# Folder names that are NOT considered Python library sources (they would
# otherwise be picked up when scanning for entry points or siblings).
NON_SOURCE_DIRS = frozenset({"assets", "references", "__pycache__"})

def resolve_library_entry(library_dir: Path) -> Optional[Path]:
    """Resolve the entry-point Python file for a library directory.

    Args:
        library_dir: The library folder (e.g. ``.../default_tools/websearch``).

    Returns:
        Path to the entry-point ``.py`` file, or ``None`` when the folder
        contains no usable entry point.
    """
    if not library_dir or not library_dir.is_dir():
        return None

    canonical = library_dir / TOOLS_ENTRYPOINT
    if canonical.is_file():
        return canonical

    legacy = library_dir / LEGACY_ENTRYPOINT_TEMPLATE.format(
        library_name=library_dir.name
    )
    if legacy.is_file():
        return legacy

    # Fallback: any non-__init__ .py file, sorted for determinism.
    for candidate in sorted(library_dir.glob("*.py")):
        if candidate.stem == "__init__":
            continue
        if candidate.is_file():
            return candidate

    return None

def resolve_library(library_path: Path) -> Optional[Tuple[Path, Path]]:
    """Resolve any tool candidate path into ``(library_dir, entry_py)``.

    Handles BOTH shapes:
        * Directory (folder-per-library): ``.../websearch`` -> entry inside it.
        * Flat file: ``.../websearch.py`` -> ``(parent, that file)``.

    Returns ``None`` when ``library_path`` is neither a library folder with a
    resolvable entry point, nor a standalone ``.py`` file.
    """
    if library_path is None:
        return None

    if library_path.is_dir():
        entry = resolve_library_entry(library_path)
        if entry is None:
            return None
        return library_path, entry

    if (
        library_path.is_file()
        and library_path.suffix == ".py"
        and library_path.stem != "__init__"
    ):
        return library_path.parent, library_path

    return None

def iter_sibling_modules(
    library_dir: Path, entry_point: Optional[Path]
) -> Iterator[Path]:
    """Yield sibling ``.py`` modules of a library.

    Excludes the entry point and ``__init__`` files. Used to mount multi-file
    libraries that split their tools across helper modules.
    """
    if not library_dir or not library_dir.is_dir():
        return
    entry_resolved = entry_point.resolve() if entry_point else None
    for candidate in sorted(library_dir.glob("*.py")):
        if candidate.stem == "__init__":
            continue
        if not candidate.is_file():
            continue
        if entry_resolved is not None and candidate.resolve() == entry_resolved:
            continue
        yield candidate

def iter_library_candidates(
    tools_folder: Path,
) -> Iterator[Tuple[str, Path, Path]]:
    """Yield ``(library_name, library_dir, entry_py)`` for every library found
    directly under ``tools_folder``.

    Both folder-per-library and flat ``.py`` libraries are yielded. Sub-folders
    named ``assets`` / ``references`` / ``__pycache__`` and hidden folders are
    skipped.
    """
    if not tools_folder or not tools_folder.is_dir():
        return
    for item in sorted(tools_folder.iterdir()):
        if item.is_dir():
            if item.name in NON_SOURCE_DIRS or item.name.startswith("."):
                continue
            entry = resolve_library_entry(item)
            if entry is not None:
                yield item.name, item, entry
        elif item.suffix == ".py" and item.stem != "__init__" and item.is_file():
            yield item.stem, item.parent, item