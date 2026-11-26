
"""Standalone repro/verification for the agentic-mode workspace sync purge gap.

Scenario
--------
1. Create a temp workspace directory containing three text files.
2. Create a discussion pointed at that workspace (custom path => agentic mode:
   versioning off, bounded enumeration).
3. Force a full workspace sync so all three files are registered as artefacts.
4. Delete one file from disk and re-sync.
5. Assert the deleted file's artefact is purged from the registry.

Expected BEFORE the fix: sync skips the purge in agentic mode, so the deleted
file's artefact lingers forever -> report "LEAK CONFIRMED".
Expected AFTER the fix:  the deleted file's artefact is removed from the
registry -> report "PURGE OK".
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the repo importable whether this is run from the repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _make_client():
    """Smallest client stand-in accepted by LollmsDiscussion."""
    from lollms_client.lollms_discussion import LollmsDiscussion  # noqa: F401

    class _StubClient:
        def count_tokens(self, text: str) -> int:
            return len(text or "") // 4

        def count_image_tokens(self, image) -> int:
            return 0

    return _StubClient()


def main() -> int:
    from lollms_client.lollms_discussion import LollmsDiscussion

    tmp_root = Path(tempfile.mkdtemp(prefix="verify_sync_purge_"))
    workspace = tmp_root / "agentic_ws"
    workspace.mkdir(parents=True, exist_ok=True)

    keep_files = ["alpha.txt", "beta.txt"]
    deleted_file = "gamma.txt"
    for name in keep_files + [deleted_file]:
        (workspace / name).write_text(f"content of {name}\n", encoding="utf-8")

    client = _make_client()
    disc = LollmsDiscussion.create_new(
        lollms_client=client,
        workspace_path=str(workspace),
        auto_create_db=True,
    )

    print(f"workspace         : {workspace}")
    print(f"agentic mode      : {getattr(disc, 'disable_artefact_versioning', None)}")

    titles_before = {a.get("title") for a in disc.artefacts._get_all_raw()}
    print(f"registered before : {sorted(titles_before)}")

    missing_before = [n for n in keep_files + [deleted_file] if n not in titles_before]
    if missing_before:
        print(f"ERROR: files not registered before sync: {missing_before}")
        return 2

    (workspace / deleted_file).unlink()
    report = disc.sync_workspace_to_artefacts(enumerate_workspace=True)
    print(f"sync report       : {report}")

    titles_after = {a.get("title") for a in disc.artefacts._get_all_raw()}
    print(f"registered after  : {sorted(titles_after)}")

    print()
    if deleted_file in titles_after:
        print(f"LEAK CONFIRMED: '{deleted_file}' was deleted from disk but is still "
              f"registered as an artefact.")
        return 1

    kept_missing = [n for n in keep_files if n not in titles_after]
    if kept_missing:
        print(f"ERROR: surviving files were wrongly purged: {kept_missing}")
        return 3

    print(f"PURGE OK: '{deleted_file}' was correctly removed from the registry and "
          f"surviving files ({keep_files}) remain registered.")
    print(f"report deleted_artefacts = {report.get('deleted_artefacts')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
