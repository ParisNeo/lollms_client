"""Tests for the handbag MCP hooks subsystem.

A handbag MAY ship an optional ``mcp/`` folder containing one YAML manifest per
MCP server. This test file pins four contracts that other modules rely on:

1. **Optionality** — a handbag with no ``mcp/`` folder yields zero specs and
   no warnings (importing/loading a handbag never has side effects).
2. **Fail-soft parsing** — malformed manifests are skipped with a warning,
   never aborting handbag construction; valid ones are loaded and sorted by
   ``server_id`` for deterministic catalogs.
3. **In-memory registration** — host apps can register specs dynamically by
   replacing any existing spec sharing the same ``server_id``.
4. **Secret hygiene** — ``mcp_manifest_summary`` redacts ``env`` and ``headers``
   values so they can never leak into logs or model prompts.

All tests use ``tmp_path`` fixtures and never touch real handbags on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lollms_client.lollms_personality.handbag import Handbag
from lollms_client.lollms_personality.mcp_hooks import (
    MCPServerSpec,
    MCPConfigError,
    parse_mcp_manifest,
)


# ── Fixtures & helpers ──────────────────────────────────────────────────────


STDIO_MANIFEST = """\
title: Filesystem
description: Filesystem tools
transport: stdio
command: npx
args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
env:
  NODE_ENV: production
security:
  allow_file_write: true
  allowed_paths: ["/workspace"]
timeout_seconds: 45
"""

HTTP_MANIFEST = """\
title: Remote
description: Remote HTTP server
transport: http
url: http://localhost:8080/mcp
headers:
  Authorization: "Bearer token-value"
security:
  approval_required: true
"""


def _make_handbag(tmp_path: Path, manifests: dict[str, str] | None = None) -> Handbag:
    """Builds a minimal handbag directory, optionally seeded with mcp/*.yaml."""
    root = tmp_path / "hb"
    root.mkdir()
    if manifests:
        mcp_dir = root / "mcp"
        mcp_dir.mkdir()
        for filename, content in manifests.items():
            (mcp_dir / filename).write_text(content, encoding="utf-8")
    return Handbag(root)


def _make_spec(server_id: str, **overrides) -> MCPServerSpec:
    """Builds an MCPServerSpec with sensible defaults for registration tests."""
    fields = dict(
        server_id=server_id,
        source_path=Path("virtual") / f"{server_id}.yaml",
        title=server_id,
        description="",
        enabled=True,
        transport="stdio",
        command="echo",
    )
    fields.update(overrides)
    return MCPServerSpec(**fields)


# ── Optionality ─────────────────────────────────────────────────────────────


def test_handbag_without_mcp_dir_has_no_servers(tmp_path):
    hb = _make_handbag(tmp_path)
    assert hb.mcp_servers == []
    assert hb.mcp_warnings == []
    assert hb.enabled_mcp_servers == []


# ── Parsing & loading ───────────────────────────────────────────────────────


def test_loads_valid_stdio_manifest(tmp_path):
    hb = _make_handbag(tmp_path, {"filesystem.yaml": STDIO_MANIFEST})
    assert len(hb.mcp_servers) == 1
    spec = hb.mcp_servers[0]
    assert spec.server_id == "filesystem"
    assert spec.transport == "stdio"
    assert spec.command == "npx"
    assert spec.args == ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    assert spec.env == {"NODE_ENV": "production"}
    assert spec.timeout_seconds == 45
    assert spec.security["allow_file_write"] is True
    # Untouched keys inherit the SAFE default.
    assert spec.security["allow_shell"] is False
    assert spec.is_stdio and not spec.is_network
    assert hb.mcp_warnings == []


def test_loads_http_manifest_fields(tmp_path):
    hb = _make_handbag(tmp_path, {"remote.yaml": HTTP_MANIFEST})
    spec = hb.mcp_servers[0]
    assert spec.server_id == "remote"
    assert spec.transport == "http"
    assert spec.url == "http://localhost:8080/mcp"
    assert spec.headers == {"Authorization": "Bearer token-value"}
    assert spec.security["approval_required"] is True
    assert spec.is_network and not spec.is_stdio


def test_manifests_sorted_by_server_id(tmp_path):
    hb = _make_handbag(
        tmp_path,
        {
            "zeta.yaml": STDIO_MANIFEST,
            "alpha.yml": HTTP_MANIFEST,
            "mike.yaml": HTTP_MANIFEST,
        },
    )
    assert [s.server_id for s in hb.mcp_servers] == ["alpha", "mike", "zeta"]


def test_malformed_manifest_is_skipped_with_warning(tmp_path):
    bad = "transport: carrier-pigeon\ncommand: nope\n"
    hb = _make_handbag(tmp_path, {"broken.yaml": bad, "ok.yaml": STDIO_MANIFEST})
    assert [s.server_id for s in hb.mcp_servers] == ["ok"]
    assert len(hb.mcp_warnings) == 1
    assert "broken.yaml" in hb.mcp_warnings[0]


def test_unknown_security_key_is_ignored(tmp_path):
    manifest = STDIO_MANIFEST.replace(
        "  allow_file_write: true",
        "  allow_file_write: true\n  bogus_key: should_be_dropped",
    )
    hb = _make_handbag(tmp_path, {"custom.yaml": manifest})
    spec = hb.mcp_servers[0]
    assert "bogus_key" not in spec.security
    assert spec.security["allow_file_write"] is True


# ── enabled_mcp_servers filter ──────────────────────────────────────────────


def test_enabled_mcp_servers_filters_disabled(tmp_path):
    disabled_http = "enabled: false\n" + HTTP_MANIFEST
    hb = _make_handbag(
        tmp_path,
        {"filesystem.yaml": STDIO_MANIFEST, "remote.yaml": disabled_http},
    )
    assert [s.server_id for s in hb.enabled_mcp_servers] == ["filesystem"]
    assert len(hb.mcp_servers) == 2


# ── In-memory registration ──────────────────────────────────────────────────


def test_register_mcp_server_adds_then_replaces(tmp_path):
    hb = _make_handbag(tmp_path)

    hb.register_mcp_server(_make_spec("custom"))
    assert [s.server_id for s in hb.mcp_servers] == ["custom"]

    replacement = _make_spec("custom", title="Replaced")
    hb.register_mcp_server(replacement)

    assert len(hb.mcp_servers) == 1
    assert hb.mcp_servers[0] is replacement
    assert hb.mcp_servers[0].title == "Replaced"


def test_register_mcp_server_rejects_non_spec(tmp_path):
    hb = _make_handbag(tmp_path)
    with pytest.raises(TypeError):
        hb.register_mcp_server({"server_id": "nope"})  # type: ignore[arg-type]


# ── Secret hygiene ─────────────────────────────────────────────────────────


def test_manifest_summary_redacts_secret_values(tmp_path):
    hb = _make_handbag(
        tmp_path,
        {"filesystem.yaml": STDIO_MANIFEST, "remote.yaml": HTTP_MANIFEST},
    )
    summary = {entry["server_id"]: entry for entry in hb.mcp_manifest_summary()}

    # Values are replaced with a placeholder; keys are preserved.
    assert summary["filesystem"]["env"] == {"NODE_ENV": "<redacted>"}
    assert summary["remote"]["headers"] == {"Authorization": "<redacted>"}

    # Non-secret metadata stays intact and paths are JSON-serialisable.
    assert summary["filesystem"]["transport"] == "stdio"
    assert summary["remote"]["url"] == "http://localhost:8080/mcp"
    assert isinstance(summary["remote"]["source_path"], str)


# ── Low-level parser edge cases ─────────────────────────────────────────────


def test_parse_mcp_manifest_requires_command_for_stdio(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("transport: stdio\n", encoding="utf-8")
    with pytest.raises(MCPConfigError):
        parse_mcp_manifest(p)


def test_parse_mcp_manifest_requires_url_for_http(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("transport: http\n", encoding="utf-8")
    with pytest.raises(MCPConfigError):
        parse_mcp_manifest(p)


def test_parse_mcp_manifest_rejects_unknown_transport(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("transport: telepathy\ncommand: x\n", encoding="utf-8")
    with pytest.raises(MCPConfigError):
        parse_mcp_manifest(p)


def test_parse_mcp_manifest_applies_secure_defaults(tmp_path):
    p = tmp_path / "minimal.yaml"
    p.write_text("transport: stdio\ncommand: echo\n", encoding="utf-8")
    spec = parse_mcp_manifest(p)

    assert spec.enabled is True
    assert spec.timeout_seconds == 30
    assert spec.security == {
        "allow_file_write": False,
        "allow_shell": False,
        "allowed_paths": [],
        "denied_tools": [],
        "approval_required": False,
    }


# ── Structure creation ──────────────────────────────────────────────────────


def test_create_structure_creates_mcp_dir_with_readme(tmp_path):
    root = Handbag.create_structure(tmp_path / "new_hb")
    mcp_dir = root / "mcp"

    assert mcp_dir.is_dir()
    readme = mcp_dir / "README.md"
    assert readme.is_file()

    content = readme.read_text(encoding="utf-8")
    assert "MCP Hooks" in content
    assert "filesystem.yaml" in content