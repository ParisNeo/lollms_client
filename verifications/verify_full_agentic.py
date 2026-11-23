
"""Consolidated verification: agentic workspace ⇒ artefacts visible, no .versions,
and agentic mode survives a discussion_id reload."""
import json
import tempfile
from pathlib import Path

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion._db import LollmsDataManager


def has_versions(root: Path) -> bool:
    return root.exists() and any(p.is_dir() and p.name == ".versions" for p in root.rglob("*"))


def main() -> int:
    ws = Path(tempfile.mkdtemp(prefix="full_ws_"))
    db_dir = Path(tempfile.mkdtemp(prefix="full_db_"))
    for i in range(12):
        (ws / f"m{i}.py").write_text("print(1)", encoding="utf-8")
    (ws / "readme.md").write_text("# hi", encoding="utf-8")
    file_count = len([p for p in ws.rglob("*") if p.is_file()])

    client = LollmsClient(llm_binding_name="litellm")
    dm = LollmsDataManager("sqlite:///" + str(db_dir / "d.db"))

    d1 = LollmsDiscussion.create_new(
        lollms_client=client, db_manager=dm, workspace_path=str(ws)
    )
    did = d1.id
    d1.sync_workspace_to_artefacts()
    count1 = len(d1.artefacts.list())
    d1.close()

    d2 = LollmsDiscussion(lollmsClient=client, db_manager=dm, discussion_id=did)
    d2.sync_workspace_to_artefacts()
    count2 = len(d2.artefacts.list())

    result = {
        "file_count": file_count,
        "create_count": count1,
        "reload_count": count2,
        "reload_agentic": d2.disable_artefact_versioning,
        "reload_versioning_enabled": d2.artefacts._is_versioning_enabled(),
        "versions_created": has_versions(ws),
    }
    ok = (
        count1 == file_count
        and count2 == file_count
        and d2.disable_artefact_versioning is True
        and d2.artefacts._is_versioning_enabled() is False
        and not result["versions_created"]
    )
    result["RESULT"] = "PASS" if ok else "FAIL"
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
