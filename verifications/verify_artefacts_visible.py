
"""Verify that a custom (agentic) workspace registers its files as artefacts
without creating a .versions/ folder.

Because creation-time auto-sync now registers workspace files eagerly in
agentic mode, artefacts may already exist before an explicit sync call. The
explicit sync is then a no-op (0 new) but the artefacts must still be present.

Run:  python verify_artefacts_visible.py
Expected: RESULT: PASS
"""
import json
import tempfile
from pathlib import Path

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion


def main() -> int:
    ws = Path(tempfile.mkdtemp(prefix="verify_ws_"))
    for i in range(15):
        (ws / f"f{i}.py").write_text("print(1)", encoding="utf-8")
    (ws / "notes.md").write_text("# Notes", encoding="utf-8")
    (ws / "data.csv").write_text("a,b\n1,2", encoding="utf-8")

    file_count = len([p for p in ws.rglob("*") if p.is_file()])

    client = LollmsClient(llm_binding_name="litellm")
    d = LollmsDiscussion.create_new(
        lollms_client=client, workspace_path=str(ws), auto_create_db=True
    )

    # Explicit sync: may be a no-op if creation-time auto-sync already ran.
    report = d.sync_workspace_to_artefacts()
    arts = d.artefacts.list()
    artefact_count = len(arts)

    versions_created = (ws / ".versions").exists()

    ok = (
        artefact_count == file_count
        and not versions_created
        and d.disable_artefact_versioning is True
    )

    print(json.dumps({
        "file_count": file_count,
        "artefact_count": artefact_count,
        "versions_created": versions_created,
        "sync_report": report,
        "agentic": d.disable_artefact_versioning,
        "sample_titles": [a["title"] for a in arts[:5]],
        "RESULT": "PASS" if ok else "FAIL",
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())