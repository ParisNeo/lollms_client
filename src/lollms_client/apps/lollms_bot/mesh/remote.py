
"""
Remote mesh control protocol (phase 6a).

Lets a "hub" (the FastAPI Mission Control app) launch/kill/monitor bots that
live on ANOTHER PC/server. The design intentionally reuses the mesh registry
and supervisor primitives — a remote bot is just a registry entry whose
``host`` differs from the hub's hostname, with an ``endpoint`` pointing at a
small HTTP control server running on that machine.

Wire contract (all JSON over HTTP, no auth framing beyond a shared token):

    GET  /mesh/<mesh_id>/info          → {"host": ..., "mesh": ..., "bots": [...]}
    POST /mesh/<mesh_id>/bots          → register a BotSpec on the remote; the
                                         remote supervisor then launches it.
    POST /mesh/<mesh_id>/bots/<id>/start
    POST /mesh/<mesh_id>/bots/<id>/stop
    POST /mesh/<mesh_id>/bots/<id>/kill
    DELETE /mesh/<mesh_id>/bots/<id>   → unregister (terminal only)
    GET  /mesh/<mesh_id>/bots/<id>/logs?tail=N

Authentication: a bearer token in ``X-Mesh-Token``. Every server (local or
remote) is given the SAME token by the operator; the token identifies the
"trusted hub cluster". This is deliberately simple — a mesh is a trusted
federation, not a public API.

The endpoint must be reachable from the hub; typical deployment is
``lollms-bot hub serve --bind 0.0.0.0 --port 8123 --token <secret>`` on the
remote, and ``POST /remote/register`` on the local hub with the remote URL.

This module provides:
  * ``RemoteMeshClient`` — hub-side client (used by ``gui/server.py``).
  * ``RemoteCoordinationError`` — precise, actionable errors.
  * URL/host sanitization helpers shared with the UI.

It does NOT start an HTTP server — that lives in ``gui/server.py`` so the
same FastAPI app can serve both the local hub UI AND the remote control API
from one port, avoiding a second server per bot machine.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry
from lollms_client.apps.lollms_bot.mesh.types import (
    BotSpec,
    BotStatus,
    MeshLayout,
)


class RemoteCoordinationError(RuntimeError):
    """Raised for any failed remote-mesh interaction (network, auth, protocol)."""


@dataclass
class RemoteNode:
    """
    One reachable remote machine, as the hub knows it.

    ``node_id`` is a hub-local identifier (safe for URLs/UI); ``url`` is the
    base URL (no trailing slash); ``token`` is the bearer token configured on
    that node. ``label`` is a human display name from the operator.
    """

    node_id: str
    url: str
    token: str
    label: str = ""
    mesh_id: Optional[str] = None
    last_seen: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "url": self.url,
            "token": "***redacted***",
            "label": self.label,
            "mesh_id": self.mesh_id,
            "last_seen": self.last_seen,
            "metadata": dict(self.metadata),
        }


def _sanitize_host_url(url: str) -> str:
    """
    Normalizes an operator-supplied base URL. Accepts ``host:port``,
    ``http://...``, ``https://...``. Rejects anything that could be abused as
    a scheme (``file://``, ``ftp://``) — only http(s) is permitted. This is a
    hard boundary: a remote node is reachable ONLY over HTTP(S).
    """
    raw = (url or "").strip()
    if not raw:
        raise RemoteCoordinationError("empty remote URL")
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in ("http", "https"):
        raise RemoteCoordinationError(
            f"remote URL scheme {parsed.scheme!r} not allowed; use http or https"
        )
    if not parsed.hostname:
        raise RemoteCoordinationError(f"remote URL {raw!r} has no host")
    return f"{parsed.scheme}://{parsed.netloc}"


def _is_loopback_hostname(host: str) -> bool:
    if not host:
        return False
    if host.lower() in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _current_hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


class RemoteMeshClient:
    """
    Hub-side HTTP client for a mesh node running on another machine.

    Each call is a single synchronous HTTP request — the hub is a supervisory
    UI backed by FastAPI, which runs these in a thread pool (FastAPI's
    default for ``def`` endpoint handlers), so there is no need for an async
    HTTP client. Reasoning: correctness and debuggability beat speculative
    performance here; a mesh has tens of bots, not thousands.
    """

    _DEFAULT_TIMEOUT = 15.0

    def __init__(self, node: RemoteNode):
        self.node = node
        self.base = _sanitize_host_url(node.url)

    # ------------------------------------------------------------ transport
    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        stream: bool = False,
    ) -> Any:
        url = f"{self.base}{path}"
        data = None
        headers = {
            "X-Mesh-Token": self.node.token or "",
            "Accept": "application/json",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            ctx = None
            if self.base.startswith("https://"):
                ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                raw = resp.read()
                if stream:
                    return raw
                if not raw:
                    return {}
                try:
                    return json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raise RemoteCoordinationError(
                        f"Invalid JSON from {url}: {e}"
                    ) from e
        except urllib.error.HTTPError as e:
            text = ""
            try:
                text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RemoteCoordinationError(
                f"{method} {url} → HTTP {e.code}: {text[:200]}"
            ) from e
        except urllib.error.URLError as e:
            raise RemoteCoordinationError(
                f"{method} {url} → network error: {e.reason}"
            ) from e
        except (ssl.SSLError, OSError) as e:
            raise RemoteCoordinationError(f"{method} {url} → connection error: {e}") from e

    # -------------------------------------------------------------- queries
    def info(self) -> Dict[str, Any]:
        return self._request("GET", f"/mesh/{self.node.mesh_id or '_auto'}/info")

    def list_bots(self) -> List[Dict[str, Any]]:
        data = self.info()
        bots = data.get("bots")
        return list(bots) if isinstance(bots, list) else []

    # ------------------------------------------------------------ mutations
    def register_bot(self, spec: BotSpec) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/mesh/{self.node.mesh_id or '_auto'}/bots",
            body=spec.to_dict(),
        )

    def start_bot(self, bot_id: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/mesh/{self.node.mesh_id or '_auto'}/bots/{bot_id}/start",
        )

    def stop_bot(self, bot_id: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/mesh/{self.node.mesh_id or '_auto'}/bots/{bot_id}/stop",
        )

    def kill_bot(self, bot_id: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/mesh/{self.node.mesh_id or '_auto'}/bots/{bot_id}/kill",
        )

    def delete_bot(self, bot_id: str) -> Dict[str, Any]:
        return self._request(
            "DELETE", f"/mesh/{self.node.mesh_id or '_auto'}/bots/{bot_id}"
        )

    def fetch_logs(self, bot_id: str, *, tail: int = 400) -> str:
        raw = self._request(
            "GET",
            f"/mesh/{self.node.mesh_id or '_auto'}/bots/{bot_id}/logs?tail={int(tail)}",
            stream=True,
        )
        return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# For the server side: verify an inbound control request's token.
# ---------------------------------------------------------------------------

def check_inbound_token(expected: str, presented: Optional[str]) -> None:
    """
    Constant-time token comparison. Raises ``RemoteCoordinationError`` on
    mismatch. If ``expected`` is empty, EVERY inbound call is rejected — the
    control API is closed by default; the operator must set a token to open
    the mesh. This is fail-closed by design.
    """
    if not expected:
        raise RemoteCoordinationError(
            "Remote control is disabled: no token configured on this node. "
            "Start the hub with `--token <secret>` to enable it."
        )
    if not presented or not _constant_time_eq(expected, presented):
        raise RemoteCoordinationError("Invalid or missing X-Mesh-Token.")


def _constant_time_eq(a: str, b: str) -> bool:
    """Timing-attack-resistant equality (length-safe + byte-loop)."""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a.encode("utf-8"), b.encode("utf-8")):
        result |= x ^ y
    return result == 0
