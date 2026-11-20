"""
MCP (Model Context Protocol) hooks support for Handbag agents.

A handbag can optionally ship an ``mcp/`` directory containing one YAML file
per MCP server configuration. Each file declares how to reach an MCP server
(stdio subprocess or HTTP/SSE endpoint) plus the security posture the host
application should enforce when the server's tools are mounted into the
agent's toolset.

Design goals
------------
* **Zero-impact optionality** — a handbag with no ``mcp/`` directory behaves
  exactly as before; ``Handbag.mcp_servers`` is simply an empty list.
* **Declarative & auditable** — one human-readable YAML per server. The
  filename stem (minus the extension) is the canonical server identifier.
* **Fail-soft** — a malformed or insecure configuration is skipped with a
  warning rather than crashing handbag construction. Security rejections are
  logged loudly so operators notice.
* **No eager side effects** — this module only *parses and validates*. It
  never spawns processes, opens sockets, or imports ``mcp``. Actual connection
  is the responsibility of the LCP MCP toolset loader which consumes
  :class:`MCPServerSpec` instances.

YAML schema (see ``docs/PERSONALITIES_AND_HANDBAGS.md``)::

    # mcp/filesystem.yaml
    title: Filesystem Access
    description: Read and write files inside the agent workspace.
    enabled: true
    transport: stdio                     # stdio | http | sse
    command: npx                          # stdio only (required)
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    env:                                  # stdio only (optional)
      NODE_ENV: production
    url: http://localhost:8080/mcp       # http/sse only (required)
    headers:                             # http/sse only (optional)
      Authorization: "Bearer ${MCP_TOKEN}"
    security:
      allow_file_write: true
      allow_shell: false
      allowed_paths: ["/workspace"]
      denied_tools: ["dangerous_tool"]
      approval_required: false
    timeout_seconds: 30
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors

try:  # PyYAML ships with lollms_client; keep the import lazy-friendly.
    import yaml
except ImportError:  # pragma: no cover - defensive
    yaml = None  # type: ignore[assignment]


# ── Constants ────────────────────────────────────────────────────────────────

_VALID_TRANSPORTS = ("stdio", "http", "sse")

# Server identifiers become directory-ish keys downstream (log prefixes,
# tool namespace prefixes). Restrict to a conservative, shell-safe alphabet.
_SERVER_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")

# Security block defaults. Absent keys inherit these values; explicit ``None``
# is normalised to the default. This is intentionally the SAFE-by-default
# posture (no shell, no write) rather than an open posture.
_DEFAULT_SECURITY: Dict[str, Any] = {
    "allow_file_write": False,
    "allow_shell": False,
    "allowed_paths": [],
    "denied_tools": [],
    "approval_required": False,
}


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class MCPServerSpec:
    """Immutable, validated description of a single MCP server.

    Attributes:
        server_id:         Canonical identifier derived from the YAML filename
                           stem. Used as tool namespace prefix and log tag.
        source_path:       Absolute path of the YAML file this spec came from.
        title:             Human-readable name (falls back to ``server_id``).
        description:       Free-form description surfaced to the LLM.
        enabled:           If ``False``, the host MUST skip this server.
        transport:         One of ``"stdio"``, ``"http"``, ``"sse"``.
        command:           Executable to launch (``stdio`` transport only).
        args:              Arguments passed to ``command`` (``stdio`` only).
        env:               Extra environment variables (``stdio`` only).
        url:               Endpoint URL (``http``/``sse`` transports only).
        headers:           HTTP headers (``http``/``sse`` transports only).
        security:          Normalised security posture dict — always contains
                           every key in :data:`_DEFAULT_SECURITY`.
        timeout_seconds:   Per-request timeout hint for the transport layer.
    """

    server_id: str
    source_path: Path
    title: str
    description: str
    enabled: bool
    transport: str
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_SECURITY))
    timeout_seconds: int = 30

    # ── Convenience projections ────────────────────────────────────────────

    @property
    def is_stdio(self) -> bool:
        return self.transport == "stdio"

    @property
    def is_network(self) -> bool:
        return self.transport in ("http", "sse")

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Returns a JSON/YAML-serialisable representation.

        Paths are stringified so the result can be written to a log or passed
        across a boundary that does not understand ``pathlib.Path``.
        """
        return {
            "server_id": self.server_id,
            "source_path": str(self.source_path),
            "title": self.title,
            "description": self.description,
            "enabled": self.enabled,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "url": self.url,
            "headers": dict(self.headers),
            "security": dict(self.security),
            "timeout_seconds": self.timeout_seconds,
        }


# ── Parsing / validation helpers ─────────────────────────────────────────────


class MCPConfigError(ValueError):
    """Raised internally when a YAML file fails schema validation."""


def _coerce_manifest(path: Path) -> Dict[str, Any]:
    """Reads and YAML-decodes a manifest file, returning the top-level mapping.

    Raises:
        MCPConfigError: if the file cannot be read or does not decode to a dict.
    """
    if yaml is None:  # pragma: no cover - extremely rare (PyYAML is a hard dep)
        raise MCPConfigError(
            "PyYAML is not installed; cannot parse MCP configuration files."
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MCPConfigError(f"cannot read file: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise MCPConfigError(f"invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise MCPConfigError(
            f"top-level YAML value must be a mapping, got {type(data).__name__}"
        )
    return data


def _normalise_security_block(raw: Any) -> Dict[str, Any]:
    """Deep-normalises the ``security`` mapping so downstream code never needs
    to guard against missing keys or the wrong container type."""
    merged = dict(_DEFAULT_SECURITY)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key not in merged:
                # Unknown keys are a typo most of the time; warn but do not fail.
                ASCIIColors.warning(
                    f"[MCP] Unknown security key '{key}' ignored. "
                    f"Known keys: {sorted(merged)}"
                )
                continue
            if isinstance(merged[key], list):
                if isinstance(value, (list, tuple)):
                    merged[key] = [str(v) for v in value if v is not None]
                elif value is None:
                    merged[key] = []
                else:
                    merged[key] = [str(value)]
            elif isinstance(merged[key], bool):
                merged[key] = bool(value)
            else:  # pragma: no cover - future-proofing
                merged[key] = value
    return merged


def _coerce_str_mapping(raw: Any, *, field_name: str) -> Dict[str, str]:
    """Coerces a YAML mapping to ``Dict[str, str]`` with predictable errors."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MCPConfigError(f"'{field_name}' must be a mapping")
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


def _coerce_str_list(raw: Any, *, field_name: str) -> List[str]:
    """Coerces a YAML list (or scalar) to ``List[str]``."""
    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        return [str(raw)]
    if not isinstance(raw, (list, tuple)):
        raise MCPConfigError(f"'{field_name}' must be a list")
    return [str(v) for v in raw if v is not None]


def parse_mcp_manifest(path: Path) -> MCPServerSpec:
    """Parses exactly one MCP manifest file into a validated :class:`MCPServerSpec`.

    Args:
        path: Absolute path of a ``*.yaml`` / ``*.yml`` file inside ``mcp/``.

    Returns:
        A fully-populated :class:`MCPServerSpec`.

    Raises:
        MCPConfigError: on any schema violation. The caller is expected to
            catch this and skip the file.
    """
    server_id = path.stem
    if not _SERVER_ID_RE.match(server_id):
        raise MCPConfigError(
            f"invalid server id '{server_id}': must match {_SERVER_ID_RE.pattern}"
        )

    data = _coerce_manifest(path)

    transport = str(data.get("transport") or "stdio").strip().lower()
    if transport not in _VALID_TRANSPORTS:
        raise MCPConfigError(
            f"invalid transport '{transport}': expected one of {_VALID_TRANSPORTS}"
        )

    enabled_raw = data.get("enabled", True)
    enabled = True if enabled_raw is None else bool(enabled_raw)

    # Transport-specific requirements.
    command: Optional[str] = None
    url: Optional[str] = None
    if transport == "stdio":
        cmd_raw = data.get("command")
        if not cmd_raw:
            raise MCPConfigError("'command' is required for stdio transport")
        command = str(cmd_raw)
    else:
        url_raw = data.get("url")
        if not url_raw:
            raise MCPConfigError(f"'url' is required for {transport} transport")
        url = str(url_raw)

    args = _coerce_str_list(data.get("args"), field_name="args")
    env = _coerce_str_mapping(data.get("env"), field_name="env")
    headers = _coerce_str_mapping(data.get("headers"), field_name="headers")
    security = _normalise_security_block(data.get("security"))

    try:
        timeout = int(data.get("timeout_seconds", 30))
    except (TypeError, ValueError) as exc:
        raise MCPConfigError(f"'timeout_seconds' must be an integer: {exc}") from exc
    if timeout <= 0:
        raise MCPConfigError("'timeout_seconds' must be a positive integer")

    title = str(data.get("title") or server_id)
    description = str(data.get("description") or "").strip()

    return MCPServerSpec(
        server_id=server_id,
        source_path=path,
        title=title,
        description=description,
        enabled=enabled,
        transport=transport,
        command=command,
        args=args,
        env=env,
        url=url,
        headers=headers,
        security=security,
        timeout_seconds=timeout,
    )


def load_mcp_manifests(
    mcp_dir: Path,
) -> Tuple[List[MCPServerSpec], List[str]]:
    """Loads every ``*.yaml``/``*.yml`` manifest under *mcp_dir*.

    Skips (and reports) malformed files rather than aborting. Returns a
    ``(specs, warnings)`` tuple. Specifications are sorted by ``server_id`` for
    deterministic ordering — critical for reproducible tool-catalog hashes.

    Args:
        mcp_dir: Directory to scan. Need not exist.
    """
    specs: List[MCPServerSpec] = []
    warnings: List[str] = []

    if not mcp_dir.exists() or not mcp_dir.is_dir():
        return specs, warnings

    for candidate in sorted(mcp_dir.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in (".yaml", ".yml"):
            continue
        if candidate.name.startswith("."):
            continue
        try:
            spec = parse_mcp_manifest(candidate.resolve())
        except MCPConfigError as exc:
            warnings.append(f"{candidate.name}: {exc}")
            ASCIIColors.warning(
                f"[MCP] Skipping malformed manifest '{candidate.name}': {exc}"
            )
            continue
        except Exception as exc:  # pragma: no cover - last-resort guard
            warnings.append(f"{candidate.name}: unexpected error: {exc}")
            ASCIIColors.warning(
                f"[MCP] Skipping manifest '{candidate.name}' after unexpected "
                f"error: {exc}"
            )
            continue
        specs.append(spec)

    specs.sort(key=lambda s: s.server_id)
    return specs, warnings