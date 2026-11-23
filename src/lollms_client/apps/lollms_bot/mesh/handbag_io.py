
"""
Handbag bundling: import/export a mesh agent as a portable zip.

A "handbag" is lollms-client's identity bundle for a personality: SOUL.md,
personalities/, tools/, skills/, rag/, docs/, memory/, mcp/, etc. For the mesh
Mission Control we want operators to SEND an agent (handbag + BotSpec) to
another machine as a single zip file. This module is that format.

Two operations, both side-effect clean:

  * ``export_handbag(handbag_path, output_zip, spec=None)`` — copies the
    handbag into a zip along with an optional ``mesh_bot_spec.json`` at the
    archive root so the receiving machine can auto-register the bot.

  * ``import_handbag(input_zip, dest_root, *, allow_overwrite=False)`` —
    extracts a bundle into ``dest_root/<bot_id>/`` after validating:
      1. Zip-bomb guard (max entries, max uncompressed bytes).
      2. Path-traversal guard (each entry must resolve INSIDE dest_root).
      3. Manifest presence (a valid ``handbag.yaml`` OR the ``mesh_bot_spec.json``
         guarantees the extracted folder is a usable agent source).

Motivation for the guards: a handbag bundle is operator-supplied content that
may traverse machines. A malicious or corrupt zip must never write outside its
target folder, and must never exhaust RAM/disk on extraction. We reject (never
"sanitize") any archive that violates the guards — silent rewriting would make
the bundle non-deterministic across machines.

Format
------
    agent_bundle.zip
    ├── mesh_bot_spec.json         (optional, produced by export_handbag)
    ├── handbag.yaml               (the canonical handbag manifest)
    ├── SOUL.md
    ├── personalities/...
    ├── skills/...
    ├── tools/...
    ├── rag/...
    ├── docs/...
    ├── memory/                    (excluded by default: not portable state)
    └── ...other handbag content
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from lollms_client.apps.lollms_bot.mesh.types import BotSpec, validate_bot_id


class HandbagBundleError(RuntimeError):
    """Raised when an import or export fails a validity/safety check."""


# Extraction guard defaults. Tuned for typical agent handbags (a few MB of
# markdown + code) with generous headroom; deliberately NOT huge so a hostile
# bundle can't fill the disk.
_DEFAULT_MAX_ENTRIES = 2000
_DEFAULT_MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024   # 200 MiB

# Paths under a handbag that should never travel between machines: they hold
# machine-specific state (episodic memory DB, chat caches, sandbox caches).
_NON_PORTABLE_TOP_LEVEL = ("memory", ".sandbox_cache", "workspace")

_MANIFEST_NAME = "mesh_bot_spec.json"
_HANDBAG_MANIFEST = "handbag.yaml"


@dataclass
class BundleInfo:
    """Result of a successful import."""

    bot_spec: BotSpec
    handbag_path: Path
    extracted_files: int
    total_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_spec": self.bot_spec.to_dict(),
            "handbag_path": str(self.handbag_path),
            "extracted_files": self.extracted_files,
            "total_bytes": self.total_bytes,
        }


def _iter_handbag_files(root: Path) -> Iterable[Path]:
    """Yields every file inside ``root`` except non-portable subtrees."""
    root = root.resolve()
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        try:
            rel = entry.relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in _NON_PORTABLE_TOP_LEVEL:
            continue
        yield entry


def export_handbag(
    handbag_path: Path | str,
    output_zip: Path | str,
    *,
    spec: Optional[BotSpec] = None,
) -> Path:
    """
    Bundles ``handbag_path`` into ``output_zip``. When ``spec`` is provided,
    writes it as ``mesh_bot_spec.json`` at the archive root so ``import_handbag``
    can auto-rebuild the BotSpec.

    Returns the resolved output path. Raises ``HandbagBundleError`` if the
    source is not a folder or the destination cannot be written.
    """
    src = Path(handbag_path).expanduser().resolve()
    if not src.is_dir():
        raise HandbagBundleError(f"handbag_path {src} is not a directory")

    dst = Path(output_zip).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    files = list(_iter_handbag_files(src))
    if not files:
        raise HandbagBundleError(f"handbag {src} contains no portable files")

    try:
        with zipfile.ZipFile(
            dst, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            if spec is not None:
                zf.writestr(
                    _MANIFEST_NAME,
                    json.dumps(spec.to_dict(), indent=2, sort_keys=True),
                )
            for file_path in files:
                arcname = file_path.relative_to(src).as_posix()
                zf.write(file_path, arcname=arcname)
    except OSError as e:
        raise HandbagBundleError(f"failed to write bundle {dst}: {e}") from e
    return dst


def _validate_zip_members(
    zf: zipfile.ZipFile,
    dest_root: Path,
    *,
    max_entries: int,
    max_uncompressed_bytes: int,
) -> None:
    """
    Rejects a bundle BEFORE extraction if any entry is unsafe. Computes the
    resolved target of every entry and asserts it stays inside ``dest_root``.
    Also enforces entry-count and total-uncompressed-size caps so a zip bomb
    is neutralized cheaply.
    """
    infos = zf.infolist()
    if len(infos) > max_entries:
        raise HandbagBundleError(
            f"bundle has {len(infos)} entries (> {max_entries} limit)"
        )
    total = 0
    dest_resolved = dest_root.resolve()
    for info in infos:
        # Windows-style backslashes in zip names are normalized by zipfile,
        # but guard anyway.
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ".." in Path(name).parts:
            raise HandbagBundleError(
                f"unsafe entry name in bundle: {info.filename!r}"
            )
        target = (dest_resolved / name).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError:
            raise HandbagBundleError(
                f"entry {info.filename!r} resolves outside the destination"
            ) from None
        total += int(info.file_size)
        if total > max_uncompressed_bytes:
            raise HandbagBundleError(
                f"bundle exceeds {max_uncompressed_bytes} uncompressed bytes"
            )


def _read_manifest(zf: zipfile.ZipFile) -> Optional[BotSpec]:
    try:
        raw = zf.read(_MANIFEST_NAME)
    except KeyError:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HandbagBundleError(f"{_MANIFEST_NAME} is not valid JSON: {e}") from e
    try:
        return BotSpec.from_dict(data)
    except Exception as e:
        raise HandbagBundleError(f"{_MANIFEST_NAME} is not a valid BotSpec: {e}") from e


def import_handbag(
    input_zip: Path | str,
    dest_root: Path | str,
    *,
    allow_overwrite: bool = False,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_uncompressed_bytes: int = _DEFAULT_MAX_UNCOMPRESSED_BYTES,
    bot_id_override: Optional[str] = None,
) -> BundleInfo:
    """
    Extracts a bundle into ``dest_root/<bot_id>/``.

    ``bot_id`` is taken (in order of precedence) from ``bot_id_override``,
    the bundle manifest (``mesh_bot_spec.json``), or the archive filename
    stem (last resort). It is validated as a filesystem-safe slug either way.

    The destination folder must not already exist unless ``allow_overwrite``
    is True — in which case a timestamped ``.old.<ts>`` sibling is created
    first (atomically renamed) so a failed import never destroys a working
    bot.
    """
    src = Path(input_zip).expanduser().resolve()
    if not src.is_file():
        raise HandbagBundleError(f"input bundle {src} not found")
    dest_root_p = Path(dest_root).expanduser().resolve()
    dest_root_p.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(src, mode="r") as zf:
            _validate_zip_members(
                zf,
                dest_root_p,
                max_entries=max_entries,
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            manifest_spec = _read_manifest(zf)
            bot_id_raw = (
                bot_id_override
                or (manifest_spec.bot_id if manifest_spec else None)
                or src.stem
            )
            try:
                bot_id = validate_bot_id(bot_id_raw)
            except ValueError as e:
                raise HandbagBundleError(
                    f"cannot derive a valid bot_id ({bot_id_raw!r}): {e}"
                ) from e

            target = dest_root_p / bot_id
            if target.exists():
                if not allow_overwrite:
                    raise HandbagBundleError(
                        f"destination {target} already exists "
                        "(use allow_overwrite=True to replace)"
                    )
                backup = target.with_name(f"{bot_id}.old.{int(__import__('time').time())}")
                target.rename(backup)

            try:
                zf.extractall(target)
            except Exception:
                # Roll back to a clean state: remove the half-extracted target.
                shutil.rmtree(target, ignore_errors=True)
                raise

            # Enforce manifest presence check AFTER extraction — a zip with no
            # handbag.yaml and no BotSpec is not a usable agent source.
            has_manifest = (target / _HANDBAG_MANIFEST).exists()
            if not has_manifest and manifest_spec is None:
                shutil.rmtree(target, ignore_errors=True)
                raise HandbagBundleError(
                    f"bundle {src} contains no handbag.yaml and no "
                    f"{_MANIFEST_NAME}; it is not a usable agent source"
                )

            # If the bundle had no BotSpec, synthesize one pointing at the
            # extracted handbag so the receiver can register the bot immediately.
            if manifest_spec is None:
                spec = BotSpec(
                    bot_id=bot_id,
                    handbag_path=str(target),
                    metadata={"imported_from": src.name},
                )
            else:
                spec = manifest_spec
                # Override handbag_path to the extracted location with the
                # SAME bot_id (the manifest path is hub-relative, meaningless
                # on the receiver).
                spec = BotSpec.from_dict(
                    {**spec.to_dict(), "handbag_path": str(target)}
                )

            extracted_files = sum(1 for _ in target.rglob("*") if _.is_file())
            total_bytes = sum(
                _.stat().st_size for _ in target.rglob("*") if _.is_file()
            )

    except zipfile.BadZipFile as e:
        raise HandbagBundleError(f"{src} is not a valid zip archive: {e}") from e

    return BundleInfo(
        bot_spec=spec,
        handbag_path=target,
        extracted_files=extracted_files,
        total_bytes=total_bytes,
    )


def list_bundled_specs(bundle_path: Path | str) -> Optional[Dict[str, Any]]:
    """
    Returns the parsed ``mesh_bot_spec.json`` from a bundle WITHOUT extracting
    it. Used by the hub's import dialog to preview what it's about to install.
    """
    src = Path(bundle_path).expanduser().resolve()
    if not src.is_file():
        return None
    try:
        with zipfile.ZipFile(src, mode="r") as zf:
            try:
                raw = zf.read(_MANIFEST_NAME)
            except KeyError:
                return None
            return json.loads(raw.decode("utf-8"))
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError):
        return None
