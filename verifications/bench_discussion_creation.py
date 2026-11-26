
"""Benchmark: how fast is a LollmsDiscussion created inside a pre-populated folder?

Builds a realistic project tree (nested subdirs + many files), then times
`LollmsDiscussion.create_new(workspace_path=<tree>)`, which triggers the
construction-time workspace ingestion in agentic mode.

The package is loaded with an explicit regular-package spec from
`src/lollms_client/__init__.py`. This bypasses the namespace-package confusion
the persistent benchmark interpreter develops after repeated failed imports
(it resolves `src/lollms_client` as a namespace even though `__init__.py`
exists). Forcing a SourceFileLoader spec is deterministic and env-independent.

Run:  python verifications/bench_discussion_creation.py [--dirs N] [--files F]
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
_PKG_INIT = _SRC / "lollms_client" / "__init__.py"

def _load_lollms_package() -> None:
    """Loads `lollms_client` as a REGULAR package from its explicit __init__.py.

    Idempotent: if the package is already correctly loaded, this is a no-op.
    """
    existing = sys.modules.get("lollms_client")
    # A NAMESPACE package also carries `__path__` but has `__spec__.loader is None`.
    # Only treat the package as already-loaded when it was loaded by a REAL
    # loader (regular package). This defeats the poisoned-namespace trap.
    if (
        existing is not None
        and getattr(existing, "__path__", None)
        and getattr(getattr(existing, "__spec__", None), "loader", None) is not None
    ):
        return
    for name in [m for m in list(sys.modules) if m == "lollms_client" or m.startswith("lollms_client.")]:
        del sys.modules[name]
    sys.path_importer_cache.clear()
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    importlib.invalidate_caches()

    loader = importlib.machinery.SourceFileLoader("lollms_client", str(_PKG_INIT))
    spec = importlib.util.spec_from_file_location(
        "lollms_client",
        str(_PKG_INIT),
        loader=loader,
        submodule_search_locations=[str(_SRC / "lollms_client")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["lollms_client"] = module
    spec.loader.exec_module(module)

def _make_stub_client():
    """Cheap deterministic client stand-in (no LLM binding needed)."""
    class _StubClient:
        def count_tokens(self, text: str) -> int:
            return len(text or "") // 4

        def count_image_tokens(self, image) -> int:
            return 0

    return _StubClient()

def _build_tree(root: Path, n_dirs: int, files_per_dir: int) -> None:
    """Creates `n_dirs` subdirectories, each with `files_per_dir` files,
    plus pruned dirs (venv/.git/node_modules) that the walk must skip."""
    root.mkdir(parents=True, exist_ok=True)
    payload = "x = 1\n" * 40  # ~240 bytes of text per file

    for d in range(n_dirs):
        sub = root / f"pkg_{d:03d}"
        sub.mkdir(parents=True, exist_ok=True)
        for f in range(files_per_dir):
            ext = ".py" if f % 5 == 0 else (".md" if f % 5 == 1 else ".txt")
            # Globally-unique basename: artefact titles are flat keys, so files
            # sharing a basename across directories would collide and only one
            # survivor would be registered — masking the real per-file cost.
            (sub / f"pkg{d:03d}_file_{f:03d}{ext}").write_text(
                f"# {d}/{f}\n{payload}", encoding="utf-8"
            )

    (root / "venv" / "lib").mkdir(parents=True, exist_ok=True)
    (root / "venv" / "lib" / "heavy.py").write_text("print(1)\n" * 100, encoding="utf-8")
    (root / ".git" / "objects").mkdir(parents=True, exist_ok=True)
    (root / ".git" / "objects" / "blob").write_text("deadbeef\n" * 100, encoding="utf-8")
    (root / "node_modules" / "leftpad").mkdir(parents=True, exist_ok=True)
    (root / "node_modules" / "leftpad" / "index.js").write_text("module.exports=1\n", encoding="utf-8")

def _tree_stats(root: Path) -> tuple[int, int, int]:
    ndirs = nfiles = nbytes = 0
    for r, dirs, files in os.walk(root):
        ndirs += len(dirs)
        for fn in files:
            nfiles += 1
            try:
                nbytes += (Path(r) / fn).stat().st_size
            except OSError:
                pass
    return ndirs, nfiles, nbytes

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", type=int, default=50)
    parser.add_argument("--files", type=int, default=20, help="files per dir")
    args = parser.parse_args()

    _load_lollms_package()
    from lollms_client.lollms_discussion import LollmsDiscussion

    tmp_root = Path(tempfile.mkdtemp(prefix="bench_disc_"))
    workspace = tmp_root / "project"
    _build_tree(workspace, args.dirs, args.files)
    ndirs, nfiles, nbytes = _tree_stats(workspace)

    print(f"workspace         : {workspace}")
    print(f"tree              : {ndirs} dirs, {nfiles} files, {nbytes/1024:.1f} KiB")
    print(f"files per dir     : {args.files}")

    client = _make_stub_client()

    t0 = time.perf_counter()
    disc1 = LollmsDiscussion.create_new(
        lollms_client=client,
        workspace_path=str(workspace),
        auto_create_db=True,
    )
    cold_s = time.perf_counter() - t0
    arts1 = len(disc1.artefacts._get_all_raw())
    print(f"\nCOLD creation     : {cold_s*1000:8.1f} ms   ({arts1} artefacts registered)", flush=True)
    try:
        disc1.close()
    except Exception:
        pass

    try:
        t0 = time.perf_counter()
        disc2 = LollmsDiscussion.create_new(
            lollms_client=client,
            workspace_path=str(workspace),
            auto_create_db=True,
        )
        warm_s = time.perf_counter() - t0
        arts2 = len(disc2.artefacts._get_all_raw())
        print(f"WARM creation     : {warm_s*1000:8.1f} ms   ({arts2} artefacts registered)", flush=True)
        try:
            disc2.close()
        except Exception:
            pass
    except Exception as warm_ex:
        print(f"WARM creation     : FAILED ({type(warm_ex).__name__}: {warm_ex})", flush=True)

    print(f"\nper-file cold     : {cold_s/max(nfiles,1)*1000:.2f} ms/file", flush=True)
    print(f"throughput cold   : {nfiles/max(cold_s,1e-9):.0f} files/s", flush=True)
    if arts1 < nfiles:
        print(f"NOTE              : {nfiles - arts1} files NOT registered "
              f"(agentic enumeration cap = 2000)", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
