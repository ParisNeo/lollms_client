
"""
Tests for the lollms-bot Mission Control hub (phase 6).

Covers the three properties that keep the hub SAFE to expose on a network:

  * Handbag bundle round-trips losslessly (import(export(h)) == h modulo the
    non-portable subtrees that are intentionally excluded).
  * A malicious bundle with directory traversal entries is REJECTED before
    any file is written outside its destination folder.
  * The remote-control token gate REFUSES every /mesh/... call when expected
    token is empty or does not match — the fail-closed posture.

The FastAPI app itself is exercised via TestClient only when fastapi is
installed; on a bare client environment these tests are skipped cleanly so
the rest of the suite keeps running. This keeps the hub an OPTIONAL feature.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from lollms_client.apps.lollms_bot.mesh.handbag_io import (
    HandbagBundleError,
    export_handbag,
    import_handbag,
)
from lollms_client.apps.lollms_bot.mesh.recorder import (
    Replayer,
    SessionRecorder,
    list_sessions,
)
from lollms_client.apps.lollms_bot.mesh.remote import (
    RemoteCoordinationError,
    check_inbound_token,
)
from lollms_client.apps.lollms_bot.mesh.types import BotSpec, MeshLayout


# ---------------------------------------------------------------------------
# Handbag IO
# ---------------------------------------------------------------------------

def _make_fake_handbag(root: Path, *, with_manifest: bool = True) -> Path:
    hb = root / "my_handbag"
    hb.mkdir(parents=True)
    (hb / "SOUL.md").write_text("name: TestBot\nyou are a test.", encoding="utf-8")
    if with_manifest:
        (hb / "handbag.yaml").write_text("name: TestBot\n", encoding="utf-8")
    (hb / "skills").mkdir()
    (hb / "skills" / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    # Non-portable: memory DB should NOT be exported.
    (hb / "memory").mkdir()
    (hb / "memory" / "memory.db").write_bytes(b"\x00\x01")
    return hb


class TestHandbagBundle:
    def test_roundtrip(self, tmp_path: Path):
        src = _make_fake_handbag(tmp_path)
        spec = BotSpec(bot_id="testbot", handbag_path=str(src))
        bundle = tmp_path / "testbot.zip"
        export_handbag(src, bundle, spec=spec)
        assert bundle.is_file()

        dst_root = tmp_path / "install"
        info = import_handbag(bundle, dst_root)

        assert info.bot_spec.bot_id == "testbot"
        assert (info.handbag_path / "SOUL.md").is_file()
        assert (info.handbag_path / "skills" / "SKILL.md").is_file()
        # Memory must NOT come back — it's non-portable machine state.
        assert not (info.handbag_path / "memory" / "memory.db").exists()

    def test_traversal_entries_rejected(self, tmp_path: Path):
        evil = tmp_path / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../../etc/passwd", b"pwned")
            zf.writestr("handbag.yaml", b"name: Evil\n")
        with pytest.raises(HandbagBundleError):
            import_handbag(evil, tmp_path / "install")
        # Nothing extracted.
        assert not (tmp_path / "install" / "etc" / "passwd").exists()

    def test_no_handbag_no_manifest_rejected(self, tmp_path: Path):
        empty = tmp_path / "empty.zip"
        with zipfile.ZipFile(empty, "w") as zf:
            zf.writestr("random.txt", b"hello")
        with pytest.raises(HandbagBundleError):
            import_handbag(empty, tmp_path / "install")


# ---------------------------------------------------------------------------
# Remote token gate
# ---------------------------------------------------------------------------

class TestRemoteTokenGate:
    def test_empty_expected_rejects_everything(self):
        with pytest.raises(RemoteCoordinationError):
            check_inbound_token("", "anything")

    def test_wrong_token_rejected(self):
        with pytest.raises(RemoteCoordinationError):
            check_inbound_token("secret", "notsecret")

    def test_correct_token_accepted(self):
        # Must not raise.
        check_inbound_token("secret", "secret")

    def test_no_presented_token_rejected(self):
        with pytest.raises(RemoteCoordinationError):
            check_inbound_token("secret", None)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class TestRecorder:
    def test_record_and_list(self, tmp_path: Path):
        layout = MeshLayout.under(tmp_path, "m1")
        layout.materialize()
        rec = SessionRecorder(layout)
        session_id = rec.start()
        assert rec.is_recording()
        rec.stop()
        sessions = list_sessions(layout)
        assert any(s["session_id"] == session_id for s in sessions)

    def test_replay_yields_events(self, tmp_path: Path):
        layout = MeshLayout.under(tmp_path, "m2")
        layout.materialize()
        rec = SessionRecorder(layout)
        sid = rec.start()
        rec.stop()
        rep = Replayer(layout, sid)
        assert rep.exists()
        events = list(rep.iter_events(block=False))
        # session_started + session_stopped at minimum.
        assert len(events) >= 1
        kinds = {e.kind for _, e in events}
        assert "session_started" in kinds


# ---------------------------------------------------------------------------
# Optional: Hub app boots (skipped when fastapi missing)
# ---------------------------------------------------------------------------

fastapi_missing = pytest.importorskip is None   # placeholder, resolved below
try:
    from fastapi.testclient import TestClient  # type: ignore
    _HAS_FASTAPI = True
except Exception:  # pragma: no cover
    _HAS_FASTAPI = False


@pytest.fixture()
def hub_client(tmp_path: Path):
    if not _HAS_FASTAPI:
        pytest.skip("fastapi not installed")
    from lollms_client.apps.lollms_bot.mesh.gui.server import create_hub_app

    layout = MeshLayout.under(tmp_path, "hubtest")
    layout.materialize()
    app = create_hub_app(layout, token="secret", allow_remote=True)
    return TestClient(app)


class TestHubApp:
    def test_env_endpoint(self, hub_client):
        r = hub_client.get("/api/env")
        assert r.status_code == 200
        assert r.json()["mesh_id"] == "hubtest"

    def test_remote_gate_rejects_without_token(self, hub_client):
        r = hub_client.get("/mesh/hubtest/info")
        assert r.status_code == 401

    def test_remote_gate_accepts_with_token(self, hub_client):
        r = hub_client.get(
            "/mesh/hubtest/info", headers={"X-Mesh-Token": "secret"}
        )
        assert r.status_code == 200
        assert r.json()["mesh_id"] == "hubtest"
