
"""
Mission Control hub server (phase 6b).

A FastAPI app that launches, monitors, records, replays, imports, exports,
and remotely controls bots on a mesh. This is the backend driving the "watch
an animation" UI in ``static/index.html``.

It intentionally exposes TWO API surfaces on the same port:

  * ``/api/...``   — hub-facing REST/WebSocket consumed by the browser.
  * ``/mesh/...``  — the remote-control protocol from ``remote.py``.

That way an operator runs ONE ``lollms-bot hub launch --token X`` per machine
and each machine's hub is directly consumable by every other hub in the
cluster, which makes "cross-PC" deployment a one-command operation per node.

Security posture
----------------
* The browser API is loopback-only by default (``--bind 127.0.0.1``); binding
  to ``0.0.0.0`` is required for cross-PC and prints a loud warning.
* The remote-control API is **closed unless a token is configured** (see
  ``remote.check_inbound_token``). No token → remote spawn is refused.
* The browser API never accepts arbitrary filesystem paths for handbag
  import/export; it always resolves them under the mesh's ``bots/`` root.
* Logs are streamed by ``GET /api/bots/{id}/logs?tail=N`` — no shell escapes
  involved; we read the bot's stdout file directly.

Startup
-------
Call ``create_hub_app(layout, token=None, allow_remote=False)`` and run it via
``uvicorn``. The ``run_hub`` helper wraps both.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors

from lollms_client.apps.lollms_bot.mesh.agents import (
    AgentCatalog,
    AgentDefinition,
    AgentError,
    CapabilityEditor,
    discover_handbags as _canonical_discover_handbags,
    discover_model_profiles as _canonical_discover_model_profiles,
)
from lollms_client.apps.lollms_bot.mesh.blueprints import (
    BlueprintError,
    ensure_blueprint_handbag,
    get_blueprint,
    list_blueprints,
)
from lollms_client.apps.lollms_bot.mesh.bus import MessageBus
from lollms_client.apps.lollms_bot.mesh.connection_adapter import (
    ConnectionBindingAdapter,
    build_adapters_from_client,
)
from lollms_client.apps.lollms_bot.mesh.connector import (
    SwarmConnector,
    SwarmConnectorError,
    load_connector_config,
    save_connector_config,
)
from lollms_client.apps.lollms_bot.mesh.handbag_io import (
    BundleInfo,
    HandbagBundleError,
    export_handbag,
    import_handbag,
    list_bundled_specs,
)
from lollms_client.apps.lollms_bot.mesh.recorder import (
    Replayer,
    SessionRecorder,
    delete_session,
    list_sessions,
)
from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry, RegistryError
from lollms_client.apps.lollms_bot.mesh.remote import (
    RemoteCoordinationError,
    RemoteNode,
    check_inbound_token,
)
from lollms_client.apps.lollms_bot.mesh.supervisor import (
    MeshSupervisor,
    SupervisorError,
)
from lollms_client.apps.lollms_bot.mesh.types import (
    BotSpec,
    BotStatus,
    MeshLayout,
    validate_bot_id,
)


# ---------------------------------------------------------------------------
# Optional dependency guard — the hub is a separate feature, not a hard
# dependency of the lollms-bot daemon.
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import guard exercised by error paths
    from fastapi import (
        FastAPI,
        HTTPException,
        Request,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
        PlainTextResponse,
    )
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    _HUB_DEPS_OK = True
except Exception:  # pragma: no cover
    _HUB_DEPS_OK = False


if not _HUB_DEPS_OK:  # pragma: no cover
    def create_hub_app(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "The hub requires `fastapi` and `uvicorn`. "
            "Install them with: pip install fastapi uvicorn"
        )

    def run_hub(*args: Any, **kwargs: Any) -> int:
        ASCIIColors.red(
            "The hub requires `fastapi` and `uvicorn`.\n"
            "Install them with:\n    pip install fastapi uvicorn"
        )
        return 2


# ---------------------------------------------------------------------------
# Hub context: shared state for one running hub instance.
# ---------------------------------------------------------------------------

@dataclass
class HubContext:
    """
    Everything one hub process owns. Passed to the FastAPI app as ``app.state.hub``
    so every endpoint has typed, injectable access.

    Why ``SessionRecorder`` lives here (not per-request): recording is a
    singleton activity per mesh — the operator presses Record once and the
    whole mesh is captured. A per-request recorder would be undefined.
    """

    layout: MeshLayout
    token: Optional[str]
    allow_remote: bool
    registry: MeshRegistry
    bus: MessageBus
    recorder: SessionRecorder
    # Phase 3: the supervisor that lazily spawns/kills bot subprocesses. The
    # hub is the natural owner — its process is long-lived, has the mesh
    # layout, and drives all lifecycle REST calls. The autostarted watchdog
    # gives restart-on-crash for free.
    supervisor: MeshSupervisor

    # In-memory "known remotes" list. Persisted to disk (remotes.json) so a
    # restart re-connects. Simplicity → file, not DB.
    remotes: Dict[str, RemoteNode]

    # The swarm connector bridges external connection bindings (Discord,
    # Slack, lollms.com, ...) onto the mesh bus so a human on those platforms
    # can talk to the mesh. It is OPTIONAL: the hub runs fine without it.
    # ``connector`` is None until the operator enables it; the config lives
    # in ``swarm_connector.json`` so the choice survives restarts.
    connector: Optional[SwarmConnector] = None
    connector_config: Optional[Dict[str, Any]] = None
    # The live connection-binding adapter attached to ``connector`` (or None
    # when the connector was enabled without a specific binding — e.g. a
    # headless node that only ingests via /api/swarm/ingest).
    connector_adapter: Optional[ConnectionBindingAdapter] = None
    # The LollmsClient used to resolve connection bindings for the swarm
    # connector. Lazily built on first use (a headless node may never chat)
    # and cached so binding discovery is cheap on repeat open.
    swarm_client: Any = None

    # A single writer lock serializes log-file access across endpoints,
    # avoiding torn reads on Windows where concurrent file handles are
    # stricter than than on POSIX.
    log_lock: asyncio.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.log_lock is None:
            self.log_lock = asyncio.Lock()
        if self.connector_config is None:
            self.connector_config = load_connector_config(self.layout)


# ---------------------------------------------------------------------------
# Structural sanity check for index.html
# ---------------------------------------------------------------------------


def _validate_index_html(text: str) -> List[str]:
    """
    Returns a list of human-readable problems found in the served SPA file.

    Only checks invariants whose violation produces a *silent* failure in
    the browser (JS parser aborts → nothing runs → blank page). Cosmetic
    issues are intentionally out of scope.

    Checks:
      1. `<script>` and `</script>` tag counts match exactly.
      2. A small set of load-bearing function declarations exist.
      3. The boot IIFE exists (the last line of the script that actually
         starts the app).
      4. Braces are balanced inside the first <script> block — measured with
         a STRING/COMMENT/REGEX-AWARE scanner so braces that legitimately
         appear inside string literals, template-literal text, line/block
         comments or regex character classes do not produce false alarms.
    """
    problems: List[str] = []

    open_scripts = len(re.findall(r"<script\b", text, re.IGNORECASE))
    close_scripts = len(re.findall(r"</script\s*>", text, re.IGNORECASE))
    if open_scripts != close_scripts:
        problems.append(
            f"Unbalanced script tags: <script> x{open_scripts}, "
            f"</script> x{close_scripts}. The browser will treat the rest of "
            f"the file as script text and abort."
        )

    required_functions = [
        "function renderGraph",
        "function renderAgents",
        "function openLiveSocket",
        "function fullRefresh",
        "async function refreshAgents",
    ]
    for marker in required_functions:
        if marker not in text:
            problems.append(f"Missing required declaration: {marker!r}")

    if "(async () => {" not in text or "setInterval(refreshBots" not in text:
        problems.append(
            "Boot section missing: expected the final IIFE that calls "
            "refresh + openLiveSocket + setInterval(refreshBots, ...). "
            "The page will render but never connect to the hub."
        )

    # Brace balance inside <script>…</script> is a strong signal the file is
    # truncated mid-function — the specific bug class we've hit before. But
    # a NAIVE body.count("{") is wrong: braces appear legitimately inside
    # string literals, template-literal text, comments and regex classes.
    # Counting those produced false "truncated file" errors and blocked
    # serving a perfectly valid SPA. We measure brace balance with a small
    # JS-aware scanner that ignores those contexts.
    m = re.search(r"<script\b[^>]*>(.*?)</script\s*>", text, re.DOTALL | re.IGNORECASE)
    if m:
        depth, ok = _js_brace_balance(m.group(1))
        if ok and depth != 0:
            problems.append(
                f"Script brace mismatch: net {depth:+d} '{{' imbalance. "
                f"Likely a truncated function body."
            )
    return problems


def _js_brace_balance(src: str) -> tuple:
    """
    Returns ``(net_depth, reliable)`` for a JS source snippet.

    A small state machine walks the source once, tracking whether we are
    inside a single-quote string, double-quote string, template literal
    (with nested ``${ … }`` expression depth), line comment, block comment,
    or regex literal — and only counts ``{``/``}`` when they are *structural*
    (i.e. real code braces, not characters inside those contexts).

    ``reliable`` is False when the scanner cannot trust its own result — e.g.
    the source ends while still inside an unterminated string/comment/regex,
    which is itself the signature of a truncated file. Callers treat an
    unreliable scan as "do not report a brace problem" so a genuinely
    truncated file is caught by the OTHER checks (missing boot IIFE,
    unbalanced script tags) rather than producing a confusing brace count.

    This deliberately does NOT attempt to fully implement the ECMAScript
    lexical grammar (that would be absurd for a file-integrity guard). It
    handles exactly the constructs the SPA actually contains: quoted strings
    with backslash escapes, template literals with ``${}`` interpolation,
    ``//`` line comments, ``/* */`` block comments, and regex literals with
    character classes. That is sufficient to make brace counting immune to
    the one failure mode that matters here: false positives from string
    content.
    """
    depth = 0
    i = 0
    n = len(src)
    # Stack of contexts while inside template literals. Each entry is the
    # brace-depth INSIDE that template's ${ … } expression. When we close the
    # matching }, we resume template TEXT (not code).
    template_expr_depths: List[int] = []

    # Tracks the last significant code token's class so we can tell a regex
    # literal `/…/` apart from a division operator. 'value' means the previous
    # code token ended an expression (identifier, literal, ')', ']'), after
    # which '/' is division; anything else ('op', 'start') makes '/' a regex.
    prev_significant = "start"

    while i < n:
        ch = src[i]

        # ── Line comment ──────────────────────────────────────────────────
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i + 2)
            i = n if j == -1 else j + 1
            continue

        # ── Block comment ─────────────────────────────────────────────────
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            if j == -1:
                return depth, False   # unterminated → truncated
            i = j + 2
            continue

        # ── Single / double quoted string ─────────────────────────────────
        if ch in ("'", '"'):
            quote = ch
            i += 1
            terminated = False
            while i < n:
                c = src[i]
                if c == "\\":
                    i += 2
                    continue
                if c == quote:
                    terminated = True
                    i += 1
                    break
                if c == "\n":
                    # Unterminated string at end of line → truncated.
                    break
                i += 1
            if not terminated:
                return depth, False
            prev_significant = "value"
            continue

        # ── Template literal text ─────────────────────────────────────────
        if ch == "`":
            i += 1
            terminated = False
            while i < n:
                c = src[i]
                if c == "\\":
                    i += 2
                    continue
                if c == "`":
                    terminated = True
                    i += 1
                    break
                if c == "$" and i + 1 < n and src[i + 1] == "{":
                    # Enter a ${ … } interpolation expression. From here we
                    # fall back to normal code scanning; when the brace depth
                    # returns to the entry level, we resume template TEXT.
                    template_expr_depths.append(depth)
                    i += 2
                    prev_significant = "start"
                    break
                i += 1
            else:
                pass
            if i >= n and not terminated:
                return depth, False
            continue

        # ── Regex literal ─────────────────────────────────────────────────
        if ch == "/" and prev_significant in ("start", "op"):
            i += 1
            in_class = False
            terminated = False
            while i < n:
                c = src[i]
                if c == "\\":
                    i += 2
                    continue
                if c == "[":
                    in_class = True
                elif c == "]":
                    in_class = False
                elif c == "/" and not in_class:
                    terminated = True
                    i += 1
                    break
                elif c == "\n":
                    break
                i += 1
            if not terminated:
                # Not really a regex (e.g. a lone slash); treat as operator.
                prev_significant = "op"
                i += 1
                continue
            # Consume trailing flags.
            while i < n and src[i].isalpha():
                i += 1
            prev_significant = "value"
            continue

        # ── Structural braces ─────────────────────────────────────────────
        if ch == "{":
            depth += 1
            prev_significant = "op"
            i += 1
            continue
        if ch == "}":
            # Closing a ${ … } interpolation? If so, this brace is the
            # template-expression terminator, not a structural one, and we
            # resume template text right after it.
            if template_expr_depths and depth == template_expr_depths[-1] + 1:
                template_expr_depths.pop()
                depth = template_expr_depths[-1] if template_expr_depths else depth - 1
                # We just closed the interpolation; continue scanning the
                # template text from the next character. We re-enter the
                # template-text loop by NOT decrementing depth again.
                depth = template_expr_depths[-1] if template_expr_depths else depth
                i += 1
                prev_significant = "value"
                # Re-scan remaining template text until the closing backtick
                # or the next ${.
                terminated = False
                while i < n:
                    c = src[i]
                    if c == "\\":
                        i += 2
                        continue
                    if c == "`":
                        terminated = True
                        i += 1
                        break
                    if c == "$" and i + 1 < n and src[i + 1] == "{":
                        template_expr_depths.append(depth)
                        i += 2
                        prev_significant = "start"
                        break
                    i += 1
                if i >= n and not terminated and not template_expr_depths:
                    return depth, False
                continue
            depth -= 1
            prev_significant = "value"
            i += 1
            continue

        # ── Any other character ───────────────────────────────────────────
        if not ch.isspace():
            if ch.isalnum() or ch in "_$":
                # A run of identifier characters counts as a value.
                j = i + 1
                while j < n and (src[j].isalnum() or src[j] in "_$"):
                    j += 1
                prev_significant = "value"
                i = j
                continue
            if ch in ")]":
                prev_significant = "value"
            elif ch in "([,=:+-*%&|!<>?;":
                prev_significant = "op"
            i += 1
            continue

    # If we finish while still inside a template interpolation the file is
    # truncated — flag it.
    if template_expr_depths:
        return depth, False
    return depth, True


# ---------------------------------------------------------------------------
# Persistence of known remotes
# ---------------------------------------------------------------------------

def _remotes_path(layout: MeshLayout) -> Path:
    return layout.root / "remotes.json"


def _load_remotes(layout: MeshLayout) -> Dict[str, RemoteNode]:
    p = _remotes_path(layout)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, RemoteNode] = {}
    for node_id, raw in (data or {}).items():
        if not isinstance(raw, dict):
            continue
        try:
            out[node_id] = RemoteNode(
                node_id=node_id,
                url=str(raw.get("url", "")),
                token=str(raw.get("token", "")),
                label=str(raw.get("label", "")),
                mesh_id=raw.get("mesh_id") or layout.root.name,
                last_seen=raw.get("last_seen"),
                metadata=dict(raw.get("metadata") or {}),
            )
        except Exception:
            continue
    return out


def _save_remotes(layout: MeshLayout, remotes: Dict[str, RemoteNode]) -> None:
    p = _remotes_path(layout)
    payload = {
        node_id: {
            "url": node.url,
            "token": node.token,
            "label": node.label,
            "mesh_id": node.mesh_id,
            "last_seen": node.last_seen,
            "metadata": node.metadata,
        }
        for node_id, node in remotes.items()
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Local discovery: handbags (=> potential agents) + LLM profiles.
# ---------------------------------------------------------------------------

def _discover_handbags(layout: MeshLayout) -> List[Dict[str, Any]]:
    """
    Enumerates candidate agent sources. Sources, in order of priority:

      1. ``<mesh_root>/bots/*/handbag/`` — meshes already installed here.
      2. ``~/.lollms-bot/handbag``       — the classic bot's home.
      3. ``~/.lollms-bot/personas/*``    — personas registered with the bot.
      4. ``~/.lollms-bot/handbag/coworkers/*`` — the bot's coworker folder.

    A folder counts as a handbag when it contains ``handbag.yaml`` OR
    ``SOUL.md`` (mirrors the mesh's first-class identity requirement, keeps
    loose folders from showing up as agents).
    """
    from lollms_client.apps.lollms_bot.config import get_bot_home

    candidates: List[Path] = []
    for entry in sorted((layout.root / "bots").glob("*/handbag")):
        candidates.append(entry)
    home = get_bot_home()
    candidates.append(home / "handbag")
    personas_dir = home / "personas"
    if personas_dir.is_dir():
        candidates.extend(sorted(p for p in personas_dir.iterdir() if p.is_dir()))
    coworkers = home / "handbag" / "coworkers"
    if coworkers.is_dir():
        candidates.extend(sorted(p for p in coworkers.iterdir() if p.is_dir()))

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for path in candidates:
        if not path.is_dir():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        is_handbag = (path / "handbag.yaml").is_file() or (path / "SOUL.md").is_file()
        if not is_handbag:
            continue
        entry: Dict[str, Any] = {
            "path": str(resolved),
            "name": _display_name_for(path),
        }
        out.append(entry)
    return out


def _display_name_for(handbag_path: Path) -> str:
    manifest = handbag_path / "handbag.yaml"
    if manifest.is_file():
        try:
            import yaml

            data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            for k in ("name", "title", "display_name"):
                if isinstance(data.get(k), str) and data[k].strip():
                    return data[k].strip()
        except Exception:
            pass
    soul = handbag_path / "SOUL.md"
    if soul.is_file():
        try:
            for line in soul.read_text(encoding="utf-8", errors="ignore").splitlines()[:20]:
                stripped = line.strip()
                if stripped.lower().startswith("name:"):
                    return stripped.split(":", 1)[1].strip()
        except Exception:
            pass
    return handbag_path.name


def _discover_model_profiles() -> List[Dict[str, Any]]:
    """
    Lists the LLM profiles the operator can assign to a bot, using the SAME
    resolution path as lollms-code / the bot daemon. Returns an empty list
    when the global config is missing — the UI degrades to "no models yet".
    """
    try:
        from lollms_client.lollms_config_cli_env import get_client_from_env
        client = get_client_from_env(
            create_llm=True, create_connection=False, run_wizard_if_fail=False
        )
    except Exception:
        return []
    profiles: List[Dict[str, Any]] = []
    # Try the profile registry first; fall back to the active LLM only.
    for attr in ("llm_profiles", "profiles", "llm_profile_registry"):
        reg = getattr(client, attr, None)
        if isinstance(reg, dict):
            for name, profile in reg.items():
                profiles.append({
                    "name": str(name),
                    "binding": getattr(profile, "binding_name", None),
                    "model": getattr(profile, "model_name", None),
                })
            break
    if not profiles:
        llm = getattr(client, "llm", None)
        if llm is not None:
            profiles.append({
                "name": "default",
                "binding": getattr(llm, "binding_name", None),
                "model": getattr(llm, "model_name", None),
            })
    return profiles


# ---------------------------------------------------------------------------
# Swarm connector helpers (used by /api/swarm/*)
# ---------------------------------------------------------------------------

def _swarm_client(ctx: "HubContext") -> Any:
    """
    Lazily builds and caches a LollmsClient for swarm binding discovery.

    Rationale: a headless hub node may not chat on its own and the operator
    does not want an LLM warmup on boot, so we defer client construction
    until the swarm config dialog opens (or the connector tries to attach a
    binding). Building an LLM is expensive enough it's worth the deferral.
    """
    if ctx.swarm_client is not None:
        return ctx.swarm_client
    try:
        from lollms_client.lollms_config_cli_env import get_client_from_env
        ctx.swarm_client = get_client_from_env(
            create_llm=True, create_connection=True, run_wizard_if_fail=False
        )
    except Exception as e:
        ASCIIColors.warning(f"[hub] could not build swarm client: {e}")
        ctx.swarm_client = None
    return ctx.swarm_client


def _safe_call(fn: Any) -> Any:
    """Calls ``fn`` if callable; returns None on any exception. For probes."""
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:
        return None


def _pid_alive(pid: Any) -> bool:
    """
    Cross-platform "is this pid a live process?" probe, without spawning
    anything or sending a signal the process could observe.

    * POSIX  — ``os.kill(pid, 0)`` raises ``ProcessLookupError`` for a dead
               pid, ``PermissionError`` for a live process we don't own
               (which still means ALIVE), and returns None otherwise.
    * Windows— ``os.kill`` is destructive, so we must NOT use signal 0.
               Instead we open a handle with full query rights via the ctypes
               OpenProcess API; a NULL handle means gone, and
               ``STILL_ACTIVE`` (259) is the exit-code probe.

    Returns False for a non-numeric pid so a corrupt registry entry is
    treated as "definitely not running" rather than raising.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid_int
            )
            if not handle:
                return False
            try:
                code = ctypes.c_ulong(0)
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return code.value == STILL_ACTIVE
                # Handle opened but exit code unreadable — assume alive
                # (safer than wrongly reporting a live process as dead and
                # letting a duplicate spawn).
                return True
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            # ctypes failure: fail-safe to "unknown = alive" only when the
            # process clearly exists. Without a reliable probe we prefer
            # NOT to risk a duplicate spawn, but we also don't want to
            # permanently block a start on a probe bug. Return False so the
            # operator can proceed (documented in /api/supervisor hints).
            return False
    try:
        os.kill(pid_int, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _enable_swarm_connector(ctx: "HubContext", cfg: Dict[str, Any]) -> None:
    """
    Starts the swarm connector (or reconciles it to the new config) and, when
    ``connection_name`` names a live connection binding, attaches a
    :class:`ConnectionBindingAdapter` so real platform messages flow through.

    Idempotent: calling with a running connector + same config is a no-op;
    changing ``connection_name`` while running rebuilds the adapter so the
    operator can switch from Discord to Slack without a hub restart.
    """
    requested_name = str(cfg.get("connection_name") or "").strip().lower()

    # Resolve the adapter BEFORE constructing the connector so the connector
    # can be built with it (this keeps ``connector.adapter`` authoritative
    # for status reporting and eliminates the window where the connector
    # exists without its bridge).
    current = ctx.connector_adapter
    matched: Optional[ConnectionBindingAdapter] = None

    if requested_name:
        # If the current adapter already matches, keep it as-is.
        if current is not None and current.platform.lower() == requested_name:
            matched = current
        else:
            # Detach the previous adapter first (so its pull loop stops
            # calling the connector while we swap it out).
            if current is not None:
                current.stop()
                current.attach(None)
                ctx.connector_adapter = None
                current = None

            client = _swarm_client(ctx)
            if client is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"connection_name={requested_name!r} requested but no "
                        f"lollms-client is configured. Configure one first: "
                        f"python -m lollms_client.lollms_config_cli_env"
                    ),
                )
            adapters = build_adapters_from_client(client)
            matched = next(
                (a for a in adapters if a.platform.lower() == requested_name),
                None,
            )
            if matched is None:
                known = ", ".join(a.platform for a in adapters) or "(none configured)"
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"connection binding {requested_name!r} is not available. "
                        f"Configured: {known}"
                    ),
                )
    else:
        # No binding requested — drop any existing adapter; the connector
        # may still run headless (via /api/swarm/ingest).
        if current is not None:
            current.stop()
            current.attach(None)
            ctx.connector_adapter = None
            current = None

    # Construct or update the connector.
    if ctx.connector is None:
        ctx.connector = SwarmConnector(
            ctx.layout, adapter=matched, config=cfg
        )
        try:
            ctx.connector.start()
        except SwarmConnectorError as e:
            # If start fails after we constructed the adapter, unwind it so
            # we don't leave an orphan pull loop.
            if matched is not None:
                try:
                    matched.stop()
                    matched.attach(None)
                except Exception:
                    pass
            ctx.connector = None
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Already running — refresh the config the connector uses.
        ctx.connector.cfg = cfg
        ctx.connector.ingress_topic = str(cfg.get("ingress_topic") or "swarm.inbound")
        ctx.connector.reply_topic = f"{ctx.connector.ingress_topic}.reply"
        ctx.connector.reply_timeout = float(cfg.get("reply_timeout_seconds", 120))

    # If we swapped adapters, start the new one.
    if matched is not None and matched is not current:
        # The connector's __init__ auto-attaches when constructed with the
        # adapter, but for swaps (existing connector) we attach here.
        matched.attach(ctx.connector)
        try:
            matched.start()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"adapter start failed: {e}")
        ctx.connector_adapter = matched
    elif matched is not None:
        # Already attached + running.
        ctx.connector_adapter = matched


def _disable_swarm_connector(ctx: "HubContext") -> None:
    """Stops the adapter, then the connector. Idempotent."""
    if ctx.connector_adapter is not None:
        try:
            ctx.connector_adapter.stop()
            ctx.connector_adapter.attach(None)
        except Exception:
            pass
        ctx.connector_adapter = None
    if ctx.connector is not None:
        try:
            ctx.connector.stop()
        except Exception:
            pass
        ctx.connector = None


# ---------------------------------------------------------------------------
# Hub app factory
# ---------------------------------------------------------------------------

def create_hub_app(
    layout: MeshLayout,
    *,
    token: Optional[str] = None,
    allow_remote: bool = False,
) -> Any:
    """
    Builds the FastAPI app for one mesh. Does NOT start uvicorn — see
    ``run_hub`` for that.

    ``allow_remote=False`` disables the ``/mesh/...`` control endpoints even
    when a token is provided (defense-in-depth: remote-control is opt-in).
    """
    if not _HUB_DEPS_OK:  # pragma: no cover
        raise RuntimeError("fastapi/uvicorn not installed")

    layout.materialize()
    registry = MeshRegistry(layout)
    bus = MessageBus(layout)
    remote_store = _load_remotes(layout)
    ctx = HubContext(
        layout=layout,
        token=token,
        allow_remote=allow_remote,
        registry=registry,
        bus=bus,
        recorder=SessionRecorder(layout),
        supervisor=MeshSupervisor(layout, autostart_watchdog=True),
        remotes=remote_store,
        connector_config=load_connector_config(layout),
        log_lock=asyncio.Lock(),
    )

    app = FastAPI(
        title=f"lollms-bot hub · {layout.root.name}",
        description="Mission Control for the lollms-bot mesh",
    )
    app.state.hub = ctx

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _hub() -> HubContext:
        return app.state.hub

    # ---------------------------------------------------------- root + UI
    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        """
        Serves the SPA, but first checks the file is structurally intact.

        Rationale
        ---------
        A truncated `index.html` (missing `</script>`, missing closing `}`,
        or cut-off mid-function) produces a *silent* blank page in the
        browser — the JS parser fails, no JS runs, and the operator sees
        "nothing works" without a clue why. This has happened several times
        during mesh development due to partial file writes.

        Rather than serve a broken page, we validate the file against a
        minimal structural contract and, if it fails, render a LOUD dev
        error page that names the failing invariant. The operator can fix
        the file and reload; there is no ambiguity about what is wrong.
        """
        index_html = static_dir / "index.html"
        if not index_html.is_file():
            return HTMLResponse(
                f"<h1>lollms-bot hub · mesh {layout.root.name}</h1>"
                "<p>UI assets not installed.</p>"
            )
        try:
            text = index_html.read_text(encoding="utf-8")
        except OSError as e:
            return HTMLResponse(
                f"<h1>UI read error</h1><p>{e}</p>", status_code=500
            )

        problems = _validate_index_html(text)
        if problems:
            listing = "".join(f"<li>{html.escape(p)}</li>" for p in problems)
            return HTMLResponse(
                "<!DOCTYPE html><html><head><title>Hub UI integrity error</title>"
                "<style>body{font-family:monospace;background:#0e1117;color:#f85149;padding:24px;}"
                "h1{color:#f85149}li{margin:4px 0}code{background:#161b22;padding:1px 4px;}"
                "</style></head><body>"
                f"<h1>✗ Hub UI integrity error</h1>"
                f"<p>The file <code>{html.escape(str(index_html))}</code> failed "
                f"structural validation. This is why the page appears blank.</p>"
                f"<ul>{listing}</ul>"
                "<p>Fix the file (usually a truncated patch left an unbalanced tag or "
                "a missing closing brace), then reload this page.</p>"
                "</body></html>",
                status_code=500,
            )
        return HTMLResponse(text)

    # ----------------------------------------------------------- agents
    # The Agent concept (Handbag + Model profile) is the operator-facing
    # layer over BotSpec. These endpoints are the CRUD/control surface the
    # Agents panel in the UI consumes. Implementation is entirely delegated
    # to AgentCatalog (catalog/validate/register) and MeshSupervisor
    # (spawn/stop/restart) — no logic is duplicated here.
    @app.get("/api/agents/catalog")
    def api_agents_catalog() -> Dict[str, Any]:
        """One round-trip for both pickers (handbags + model profiles)."""
        try:
            catalog = AgentCatalog(layout).catalog()
            catalog["blueprints"] = [b.to_dict() for b in list_blueprints()]
            return catalog
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/blueprints")
    def api_blueprints_list() -> Dict[str, Any]:
        """Returns the curated starter blueprint catalog."""
        return {"blueprints": [b.to_dict() for b in list_blueprints()]}

    @app.post("/api/blueprints/{blueprint_id}/materialize")
    def api_blueprint_materialize(blueprint_id: str) -> Dict[str, Any]:
        """
        Materializes a blueprint's starter handbag on disk and returns its
        path. Idempotent — calling twice returns the same folder without
        touching an existing SOUL.md (so operator customizations survive).
        """
        try:
            handbag_path = ensure_blueprint_handbag(layout, blueprint_id)
        except BlueprintError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {
            "blueprint_id": blueprint_id,
            "handbag_path": str(handbag_path),
        }

    @app.get("/api/agents")
    def api_agents_list() -> Dict[str, Any]:
        try:
            return {"agents": AgentCatalog(layout).list_agents()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agents")
    def api_agents_create(body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registers a new agent. When ``autostart`` is true the spawn is
        attempted immediately; a spawn failure does NOT undo the registration
        (the agent stays PENDING so the operator can inspect its logs and
        retry start without re-entering everything).
        """
        body = body or {}
        ctx = _hub()

        # Blueprint path: when ``blueprint_id`` is supplied and no explicit
        # handbag_path, synthesize the starter handbag from the blueprint
        # catalog and inherit its default topics/capabilities. Explicit
        # fields still win — the operator may have edited them in the form.
        blueprint_id = str(body.get("blueprint_id") or "").strip()
        handbag_path = str(body.get("handbag_path") or "").strip()
        mission_topics = list(body.get("mission_topics") or [])
        capabilities = dict(body.get("capabilities") or {})
        tags = list(body.get("tags") or [])
        if blueprint_id:
            try:
                bp = get_blueprint(blueprint_id)
                if not handbag_path:
                    handbag_path = str(ensure_blueprint_handbag(layout, blueprint_id))
                if not mission_topics:
                    mission_topics = list(bp.default_topics)
                if not capabilities:
                    capabilities = dict(bp.suggested_capabilities)
                if not tags:
                    tags = list(bp.tags)
            except BlueprintError as e:
                raise HTTPException(status_code=404, detail=str(e))

        definition = AgentDefinition(
            agent_id=str(body.get("agent_id") or "").strip(),
            handbag_path=handbag_path,
            llm_profile=(body.get("llm_profile") or None),
            mission_topics=mission_topics,
            capabilities=capabilities,
            tags=tags,
            metadata=dict(body.get("metadata") or {}),
        )
        try:
            entry = AgentCatalog(layout).create_agent(definition)
        except AgentError as e:
            # Same agent_id → 409; anything else → 422.
            code = 409 if "already exists" in str(e) else 422
            raise HTTPException(status_code=code, detail=str(e))
        result: Dict[str, Any] = {"agent": entry, "started": None}
        if body.get("autostart"):
            try:
                result["started"] = ctx.supervisor.spawn(definition.agent_id)
            except SupervisorError as e:
                result["start_error"] = str(e)
        return result

    @app.post("/api/agents/{agent_id}/start")
    def api_agents_start(agent_id: str) -> Dict[str, Any]:
        ctx = _hub()
        try:
            validate_bot_id(agent_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if ctx.registry.get_bot(agent_id) is None:
            raise HTTPException(status_code=404, detail="agent not found")
        try:
            return ctx.supervisor.spawn(agent_id)
        except SupervisorError as e:
            code = 409 if "already running" in str(e) else 400
            raise HTTPException(status_code=code, detail=str(e))

    @app.post("/api/agents/{agent_id}/stop")
    def api_agents_stop(agent_id: str) -> Dict[str, Any]:
        ctx = _hub()
        try:
            return ctx.supervisor.stop(agent_id)
        except SupervisorError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/agents/{agent_id}/restart")
    def api_agents_restart(agent_id: str) -> Dict[str, Any]:
        ctx = _hub()
        try:
            return ctx.supervisor.restart(agent_id)
        except SupervisorError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/agents/{agent_id}/logs", response_class=PlainTextResponse)
    async def api_agents_logs(agent_id: str, tail: int = 400) -> str:
        ctx = _hub()
        log_path = ctx.layout.bot_logs_dir(agent_id) / "stdout.log"
        async with ctx.log_lock:
            if not log_path.is_file():
                return ""
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if tail > 0:
                lines = lines[-tail:]
            return "\n".join(lines)

    # ------------------------------------------------ agent capabilities
    # An agent is only as useful as the tools it can reach. A Mail Agent
    # without an MCP server descended to Gmail is useless. These endpoints
    # let the operator author the handbag's mcp/*.yaml files and inspect its
    # tool libraries from the UI. Credentials live in the descriptors'
    # env/headers blocks — that's where every MCP server expects them, and
    # they never come back out through the read endpoint.
    @app.get("/api/agents/{agent_id}/capabilities")
    def api_agents_capabilities(agent_id: str) -> Dict[str, Any]:
        """Returns MCP servers + tools for an agent (secret values redacted)."""
        ctx = _hub()
        try:
            editor = CapabilityEditor(ctx.layout)
            return {
                "mcp": editor.list_mcp(agent_id),
                "tools": editor.list_tools(agent_id),
            }
        except AgentError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/agents/{agent_id}/mcp/{server_id}")
    def api_agents_mcp_get(agent_id: str, server_id: str) -> Dict[str, Any]:
        """
        Returns ONE raw MCP descriptor YAML (for editing). Secret VALUES are
        replaced with a sentinel so the operator can see WHAT is configured
        without the raw credential round-tripping through the browser.
        """
        ctx = _hub()
        try:
            editor = CapabilityEditor(ctx.layout)
            handbag = editor._handbag_for(agent_id)
        except AgentError as e:
            raise HTTPException(status_code=404, detail=str(e))
        path = editor._mcp_dir(handbag) / f"{server_id}.yaml"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="mcp descriptor not found")
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        for block in ("env", "headers"):
            values = data.get(block)
            if isinstance(values, dict):
                data[block] = {str(k): "***" for k in values.keys()}
        return {"server_id": server_id, "descriptor": data}

    @app.put("/api/agents/{agent_id}/mcp/{server_id}")
    def api_agents_mcp_save(agent_id: str, server_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates or replaces one MCP descriptor.

        Secret handling: the UI GETs the descriptor first (secrets sentinel'd
        as ``***``). When it PUTs back, any env/header value that is exactly
        ``***`` is REMOVED from the incoming descriptor so it does NOT
        overwrite the real stored secret — instead we merge the pre-existing
        value back in. This lets an operator edit a channel/command without
        re-pasting their API key.
        """
        ctx = _hub()
        try:
            editor = CapabilityEditor(ctx.layout)
            handbag = editor._handbag_for(agent_id)
        except AgentError as e:
            raise HTTPException(status_code=404, detail=str(e))

        path = editor._mcp_dir(handbag) / f"{server_id}.yaml"
        incoming = dict(body or {})

        # Preserve secrets the UI sentinel'd rather than expose the real value.
        if path.is_file():
            try:
                import yaml
                previous = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                previous = {}
            for block in ("env", "headers"):
                incoming_block = incoming.get(block)
                if not isinstance(incoming_block, dict):
                    continue
                prev_block = previous.get(block) if isinstance(previous.get(block), dict) else {}
                for k, v in list(incoming_block.items()):
                    if v == "***":
                        if k in prev_block:
                            incoming_block[k] = prev_block[k]
                        else:
                            incoming_block.pop(k, None)

        try:
            result = editor.save_mcp(agent_id, server_id, incoming)
        except AgentError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return result

    @app.delete("/api/agents/{agent_id}/mcp/{server_id}")
    def api_agents_mcp_delete(agent_id: str, server_id: str) -> Dict[str, Any]:
        ctx = _hub()
        try:
            editor = CapabilityEditor(ctx.layout)
            deleted = editor.delete_mcp(agent_id, server_id)
        except AgentError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not deleted:
            raise HTTPException(status_code=404, detail="mcp descriptor not found")
        return {"deleted": server_id}

    # -------------------------------------------------------- environment
    @app.get("/api/env")
    def api_env() -> Dict[str, Any]:
        ctx = _hub()
        return {
            "mesh_id": layout.root.name,
            "host": os.environ.get("COMPUTERNAME") or os.uname().nodename
                    if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "?"),
            "allow_remote": ctx.allow_remote,
            "remote_enabled": bool(ctx.token) and ctx.allow_remote,
            "recording": ctx.recorder.is_recording(),
            "recording_session": ctx.recorder.current_session_id(),
        }

    @app.get("/api/models")
    def api_models() -> Dict[str, Any]:
        # Canonical implementation lives in mesh.agents (single source of
        # truth); the UI's "New Bot" and "New Agent" pickers must agree.
        return {"profiles": _canonical_discover_model_profiles()}

    @app.get("/api/handbags")
    def api_handbags() -> Dict[str, Any]:
        return {"handbags": _canonical_discover_handbags(layout)}

    # -------------------------------------------------------- bot lifecycle
    @app.get("/api/bots")
    def api_list_bots() -> Dict[str, Any]:
        ctx = _hub()
        return {"bots": ctx.registry.list_bots()}

    @app.get("/api/supervisor")
    def api_supervisor_state() -> Dict[str, Any]:
        """
        Reports the supervisor's LIVE view of spawned processes. This is the
        remedy for the very common "why did start return 409?" question:

          * ``running``  — processes alive in THIS hub process
          * ``stale``    — registry entries marked RUNNING whose pid is dead
          * ``orphans``  — registry entries marked RUNNING whose pid is alive
                           but which this hub did NOT spawn (a survivor of a
                           previous hub process; it must be adopted or killed
                           before a new start can proceed)

        All three are needed: a 409 on start is caused by EITHER ``running``
        (handled) OR ``orphans`` (which the operator must resolve).
        """
        ctx = _hub()
        live = {p["bot_id"]: p for p in ctx.supervisor.list_running()}
        registry_view = ctx.registry.list_bots()

        running: List[Dict[str, Any]] = []
        stale: List[Dict[str, Any]] = []
        orphans: List[Dict[str, Any]] = []
        for bot_id, entry in registry_view.items():
            status = entry.get("status", "")
            pid = entry.get("pid")
            if bot_id in live and live[bot_id].get("alive"):
                running.append({
                    "bot_id": bot_id,
                    "pid": live[bot_id]["pid"],
                    "status": status,
                    "restarts": live[bot_id].get("restarts", 0),
                })
                continue
            if status != BotStatus.RUNNING or pid is None:
                continue
            if _pid_alive(pid):
                orphans.append({
                    "bot_id": bot_id,
                    "pid": pid,
                    "host": entry.get("host"),
                    "hint": "alive but not tracked by this hub — kill it or "
                            "restart the hub to adopt it",
                })
            else:
                stale.append({
                    "bot_id": bot_id,
                    "pid": pid,
                    "hint": "registry says RUNNING but the pid is dead",
                })
        return {"running": running, "stale": stale, "orphans": orphans}

    @app.post("/api/bots/{bot_id}/adopt")
    def api_adopt_bot(bot_id: str) -> Dict[str, Any]:
        """
        Repairs a stale registry entry so the bot can be started again.

        Two cases are handled by the SAME endpoint because both have the
        operator intent "I want to be able to start this bot":
          * the pid is dead → clear the RUNNING status back to STOPPED;
          * the pid is alive but untracked (orphan from a previous hub) →
            mark it FAILED so the operator can kill it (via ``/kill``) then
            start fresh. We do NOT silently ``os.kill`` it here: an orphan
            may still be serving live traffic and killing it without the
            operator's consent would lose in-flight answers.
        """
        ctx = _hub()
        entry = ctx.registry.get_bot(bot_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="bot not found")
        status = entry.get("status", "")
        pid = entry.get("pid")
        if status != BotStatus.RUNNING or pid is None:
            return {"bot_id": bot_id, "action": "none",
                    "detail": f"bot is {status!r}, nothing to adopt"}
        live = {p["bot_id"]: p for p in ctx.supervisor.list_running()}
        if bot_id in live and live[bot_id].get("alive"):
            return {"bot_id": bot_id, "action": "none",
                    "detail": "bot is live in this hub; use stop/kill"}
        alive = _pid_alive(pid)
        if alive:
            ctx.registry.update_status(
                bot_id,
                status=BotStatus.FAILED,
                error="orphan process from a previous hub; kill it to continue",
            )
            return {"bot_id": bot_id, "action": "marked_failed",
                    "detail": "orphan pid is alive; POST /kill to terminate it"}
        ctx.registry.update_status(bot_id, status=BotStatus.STOPPED, pid=None)
        return {"bot_id": bot_id, "action": "cleared_stale",
                "detail": "dead pid cleared; bot is now startable"}

    @app.post("/api/bots")
    def api_create_bot(body: Dict[str, Any]) -> Dict[str, Any]:
        """
        LOW-LEVEL / SCRIPTING surface — registers a raw ``BotSpec``.

        Prefer ``POST /api/agents`` for interactive use: in this mesh an
        "agent" IS a bot (handbag + llm_profile + capabilities + inbound
        routes), and ``/api/agents`` validates the handbag and model choices
        against the live catalog. This endpoint exists for
        programmatic/automation clients that construct a BotSpec directly
        (e.g. a remote peer's ``/mesh/...`` handler already does). It is NOT
        wired to any UI button — the "New Bot" dialog was removed because it
        duplicated "New Agent" exactly.
        """
        try:
            spec = BotSpec.from_dict(body or {})
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"invalid bot spec: {e}")
        ctx = _hub()
        try:
            ctx.registry.register(spec, status=BotStatus.PENDING)
        except RegistryError as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"bot": ctx.registry.get_bot(spec.bot_id)}

    @app.post("/api/bots/{bot_id}/start")
    def api_start_bot(bot_id: str) -> Dict[str, Any]:
        ctx = _hub()
        entry = ctx.registry.get_bot(bot_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="bot not found")
        try:
            return ctx.supervisor.spawn(bot_id)
        except SupervisorError as e:
            # 409 for "already running", 400 for anything else — actionable
            # to the UI rather than a generic 500.
            code = 409 if "already running" in str(e) else 400
            raise HTTPException(status_code=code, detail=str(e))

    @app.post("/api/bots/{bot_id}/stop")
    def api_stop_bot(bot_id: str) -> Dict[str, Any]:
        ctx = _hub()
        try:
            return ctx.supervisor.stop(bot_id)
        except SupervisorError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/bots/{bot_id}/kill")
    def api_kill_bot(bot_id: str) -> Dict[str, Any]:
        ctx = _hub()
        try:
            return ctx.supervisor.kill(bot_id)
        except SupervisorError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/bots/{bot_id}/restart")
    def api_restart_bot(bot_id: str) -> Dict[str, Any]:
        ctx = _hub()
        try:
            return ctx.supervisor.restart(bot_id)
        except SupervisorError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/bots/{bot_id}")
    def api_delete_bot(bot_id: str) -> Dict[str, Any]:
        ctx = _hub()
        try:
            validate_bot_id(bot_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        entry = ctx.registry.get_bot(bot_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="bot not found")
        if not BotStatus.is_terminal(entry.get("status", "")):
            raise HTTPException(
                status_code=409, detail="stop or kill the bot before deleting it"
            )
        ctx.registry.unregister(bot_id)
        return {"deleted": bot_id}

    # Backward-compatible alias: some UI builds (and older bookmarks) call
    # ``DELETE /api/bots/{id}/delete``. Both paths share one handler so there
    # is a single delete implementation; this alias exists only to keep those
    # clients working without a second code path.
    @app.delete("/api/bots/{bot_id}/delete", include_in_schema=False)
    def api_delete_bot_legacy(bot_id: str) -> Dict[str, Any]:
        return api_delete_bot(bot_id)

    @app.get("/api/bots/{bot_id}/logs", response_class=PlainTextResponse)
    async def api_bot_logs(bot_id: str, tail: int = 400) -> str:
        ctx = _hub()
        log_path = ctx.layout.bot_logs_dir(bot_id) / "stdout.log"
        async with ctx.log_lock:
            if not log_path.is_file():
                return ""
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if tail > 0:
                lines = lines[-tail:]
            return "\n".join(lines)

    # ------------------------------------------------------------- bus view
    @app.get("/api/bus/topics")
    def api_bus_topics() -> Dict[str, Any]:
        ctx = _hub()
        try:
            return {"topics": ctx.bus.list_topics()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/bus/publish")
    def api_bus_publish(body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publishes an operator-authored message on a topic. Used by the hub's
        "Send a test message" affordance: an operator clicks it to make an
        otherwise-idle agent react, so they can SEE that the agent is wired up
        and correctly subscribed to at least one mission topic.

        Security: this is intentionally a thin pass-through with no
        filtering. The hub is loopback-only by default; binding it to a
        network interface is the operator's explicit decision. The token gate
        protects /mesh/... (remote control), while /api/bus/publish is a
        UI action available on the hub process itself.
        """
        ctx = _hub()
        topic = str((body or {}).get("topic") or "").strip()
        if not topic:
            raise HTTPException(status_code=422, detail="topic is required")
        payload = (body or {}).get("payload")
        sender = str((body or {}).get("sender") or "hub")
        try:
            msg_id = ctx.bus.send_message(topic, payload, sender=sender)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"published": msg_id, "topic": topic}

    @app.get("/api/bus/messages/{topic}")
    def api_bus_messages(topic: str, limit: int = 100) -> Dict[str, Any]:
        """
        Peek — not consume — the newest messages on a topic. We open an
        EPHEMERAL pub/sub subscription with ``from_now=False`` and immediately
        unsubscribe, so the operator never steals messages from a real bot's
        queue.
        """
        ctx = _hub()
        peek_name = f"hub-peek::{os.getpid()}::{time.time_ns()}"
        try:
            ctx.bus.subscribe(peek_name, topic, from_now=False)
            msgs = ctx.bus.poll_topic(peek_name, topic, max_batch=limit)
            return {"messages": [m.to_dict() for m in msgs]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            try:
                ctx.bus.unsubscribe(peek_name, topic)
            except Exception:
                pass

    # ------------------------------------------------------------ recording
    @app.post("/api/recording/start")
    def api_recording_start() -> Dict[str, Any]:
        ctx = _hub()
        try:
            session_id = ctx.recorder.start()
        except Exception as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"session_id": session_id}

    @app.post("/api/recording/stop")
    def api_recording_stop() -> Dict[str, Any]:
        ctx = _hub()
        session_id = ctx.recorder.stop()
        return {"session_id": session_id}

    @app.get("/api/recording/status")
    def api_recording_status() -> Dict[str, Any]:
        ctx = _hub()
        return {
            "recording": ctx.recorder.is_recording(),
            "session_id": ctx.recorder.current_session_id(),
        }

    @app.get("/api/sessions")
    def api_sessions() -> Dict[str, Any]:
        return {"sessions": list_sessions(layout)}

    @app.delete("/api/sessions/{session_id}")
    def api_session_delete(session_id: str) -> Dict[str, Any]:
        return {"deleted": delete_session(layout, session_id)}

    @app.get("/api/sessions/{session_id}/events")
    def api_session_events(session_id: str, block: bool = False) -> Dict[str, Any]:
        """
        Non-blocking static export of a session for the UI's replay panel.
        The browser drives its own animation timing; the server just hands
        over the whole stream at once (session files are small).
        """
        rep = Replayer(layout, session_id)
        if not rep.exists():
            raise HTTPException(status_code=404, detail="session not found")
        events = []
        for delay, evt in rep.iter_events(block=block):
            events.append({**evt.to_dict(), "delay": delay})
        return {"events": events}

    # -------------------------------------------------- handbag import/export
    @app.get("/api/handbags/{bot_id}/export")
    def api_handbag_export(bot_id: str, download: bool = False) -> Any:
        """
        Bundles a bot's handbag as a zip. When ``download=True`` the response
        is a file download (used by the UI button). Otherwise returns the
        path so a script can move it.
        """
        ctx = _hub()
        try:
            validate_bot_id(bot_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        entry = ctx.registry.get_bot(bot_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="bot not found")
        spec = BotSpec.from_dict(entry["spec"])
        handbag_path = (
            Path(spec.handbag_path).expanduser().resolve()
            if spec.handbag_path
            else ctx.layout.bot_handbag_dir(bot_id)
        )
        if not handbag_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"handbag path {handbag_path} does not exist",
            )
        out_path = ctx.layout.bot_logs_dir(bot_id) / f"{bot_id}.bundle.zip"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            export_handbag(handbag_path, out_path, spec=spec)
        except HandbagBundleError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if download:
            return FileResponse(
                str(out_path),
                media_type="application/zip",
                filename=f"{bot_id}.bundle.zip",
            )
        return {"bundle": str(out_path)}

    @app.post("/api/handbags/import")
    async def api_handbag_import(request: Request) -> Dict[str, Any]:
        """
        Accepts a zip upload and installs it as a new bot source under
        ``<mesh_root>/bots/``. Returns the synthesized/restored BotSpec so
        the UI can immediately show the new agent in the graph.

        NOTE: we stream the body to a temp file rather than buffering in RAM —
        handbag bundles can be hundreds of MB. The temp file lives inside the
        mesh's ``bots/`` root so the extraction guard can validate it.
        """
        ctx = _hub()
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        incoming = ctx.layout.bots_dir / f"_import_{time.time_ns()}.zip"
        try:
            incoming.write_bytes(raw)
            try:
                info: BundleInfo = import_handbag(
                    incoming, ctx.layout.bots_dir, allow_overwrite=False
                )
            except HandbagBundleError as e:
                raise HTTPException(status_code=400, detail=str(e))
            ctx.registry.register(info.bot_spec, status=BotStatus.PENDING)
            return {"bundle": info.to_dict()}
        finally:
            try:
                incoming.unlink(missing_ok=True)
            except OSError:
                pass

    @app.post("/api/handbags/inspect")
    async def api_handbag_inspect(request: Request) -> Dict[str, Any]:
        """Returns the ``mesh_bot_spec.json`` from an uploaded bundle (preview)."""
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        ctx = _hub()
        temp = ctx.layout.bots_dir / f"_inspect_{time.time_ns()}.zip"
        try:
            temp.write_bytes(raw)
            spec = list_bundled_specs(temp)
            return {"bot_spec": spec}
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    # ---------------------------------------------------------- remote nodes
    @app.get("/api/remotes")
    def api_remotes() -> Dict[str, Any]:
        ctx = _hub()
        return {"remotes": [n.to_dict() for n in ctx.remotes.values()]}

    @app.post("/api/remotes")
    def api_remotes_add(body: Dict[str, Any]) -> Dict[str, Any]:
        ctx = _hub()
        node_id = str((body or {}).get("node_id") or "").strip()
        url = str((body or {}).get("url") or "").strip()
        token = str((body or {}).get("token") or "").strip()
        label = str((body or {}).get("label") or "").strip()
        mesh_id = str((body or {}).get("mesh_id") or ctx.layout.root.name).strip()
        if not node_id or not url:
            raise HTTPException(status_code=422, detail="node_id and url required")
        if node_id in ctx.remotes:
            raise HTTPException(status_code=409, detail="node_id already registered")
        try:
            node = RemoteNode(
                node_id=node_id,
                url=url,
                token=token,
                label=label or node_id,
                mesh_id=mesh_id,
            )
        except RemoteCoordinationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        ctx.remotes[node_id] = node
        _save_remotes(ctx.layout, ctx.remotes)
        return {"node": node.to_dict()}

    @app.delete("/api/remotes/{node_id}")
    def api_remotes_delete(node_id: str) -> Dict[str, Any]:
        ctx = _hub()
        if node_id not in ctx.remotes:
            raise HTTPException(status_code=404, detail="node not found")
        del ctx.remotes[node_id]
        _save_remotes(ctx.layout, ctx.remotes)
        return {"deleted": node_id}

    @app.get("/api/remotes/{node_id}/bots")
    def api_remotes_bots(node_id: str) -> Dict[str, Any]:
        ctx = _hub()
        node = ctx.remotes.get(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="node not found")
        from lollms_client.apps.lollms_bot.mesh.remote import RemoteMeshClient

        try:
            bots = RemoteMeshClient(node).list_bots()
        except RemoteCoordinationError as e:
            raise HTTPException(status_code=502, detail=str(e))
        return {"bots": bots}

    @app.post("/api/remotes/{node_id}/bots/{bot_id}/start")
    def api_remotes_start(node_id: str, bot_id: str) -> Dict[str, Any]:
        ctx = _hub()
        node = ctx.remotes.get(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="node not found")
        from lollms_client.apps.lollms_bot.mesh.remote import RemoteMeshClient

        try:
            return RemoteMeshClient(node).start_bot(bot_id)
        except RemoteCoordinationError as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/remotes/{node_id}/bots/{bot_id}/stop")
    def api_remotes_stop(node_id: str, bot_id: str) -> Dict[str, Any]:
        ctx = _hub()
        node = ctx.remotes.get(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="node not found")
        from lollms_client.apps.lollms_bot.mesh.remote import RemoteMeshClient

        try:
            return RemoteMeshClient(node).stop_bot(bot_id)
        except RemoteCoordinationError as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/remotes/{node_id}/bots/{bot_id}/kill")
    def api_remotes_kill(node_id: str, bot_id: str) -> Dict[str, Any]:
        ctx = _hub()
        node = ctx.remotes.get(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="node not found")
        from lollms_client.apps.lollms_bot.mesh.remote import RemoteMeshClient

        try:
            return RemoteMeshClient(node).kill_bot(bot_id)
        except RemoteCoordinationError as e:
            raise HTTPException(status_code=502, detail=str(e))

    # ------------------------------------------------ remote-control API
    def _gate(request: Request) -> None:
        """Enforces a valid token on every /mesh/... call."""
        ctx = _hub()
        if not ctx.allow_remote:
            raise HTTPException(status_code=403, detail="remote control disabled")
        try:
            check_inbound_token(ctx.token or "", request.headers.get("X-Mesh-Token"))
        except RemoteCoordinationError as e:
            raise HTTPException(status_code=401, detail=str(e))

    @app.get("/mesh/{mesh_id}/info")
    def mesh_info(mesh_id: str, request: Request) -> Dict[str, Any]:
        _gate(request)
        ctx = _hub()
        return {
            "host": os.environ.get("COMPUTERNAME", "?"),
            "mesh_id": ctx.layout.root.name,
            "bots": list(ctx.registry.list_bots().values()),
        }

    @app.post("/mesh/{mesh_id}/bots")
    def mesh_register(mesh_id: str, body: Dict[str, Any], request: Request) -> Dict[str, Any]:
        _gate(request)
        ctx = _hub()
        try:
            spec = BotSpec.from_dict(body or {})
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
        ctx.registry.register(spec, status=BotStatus.PENDING)
        return {"bot": ctx.registry.get_bot(spec.bot_id)}

    @app.post("/mesh/{mesh_id}/bots/{bot_id}/start")
    def mesh_start(mesh_id: str, bot_id: str, request: Request) -> Dict[str, Any]:
        _gate(request)
        ctx = _hub()
        try:
            return ctx.supervisor.spawn(bot_id)
        except SupervisorError as e:
            code = 409 if "already running" in str(e) else 400
            raise HTTPException(status_code=code, detail=str(e))

    @app.post("/mesh/{mesh_id}/bots/{bot_id}/stop")
    def mesh_stop(mesh_id: str, bot_id: str, request: Request) -> Dict[str, Any]:
        _gate(request)
        ctx = _hub()
        try:
            return ctx.supervisor.stop(bot_id)
        except SupervisorError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/mesh/{mesh_id}/bots/{bot_id}/kill")
    def mesh_kill(mesh_id: str, bot_id: str, request: Request) -> Dict[str, Any]:
        _gate(request)
        ctx = _hub()
        try:
            return ctx.supervisor.kill(bot_id)
        except SupervisorError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/mesh/{mesh_id}/bots/{bot_id}/logs", response_class=PlainTextResponse)
    def mesh_logs(mesh_id: str, bot_id: str, request: Request, tail: int = 400) -> str:
        _gate(request)
        ctx = _hub()
        log_path = ctx.layout.bot_logs_dir(bot_id) / "stdout.log"
        if not log_path.is_file():
            return ""
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:] if tail > 0 else lines)

    @app.delete("/mesh/{mesh_id}/bots/{bot_id}")
    def mesh_delete(mesh_id: str, bot_id: str, request: Request) -> Dict[str, Any]:
        _gate(request)
        ctx = _hub()
        entry = ctx.registry.get_bot(bot_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="bot not found")
        if not BotStatus.is_terminal(entry.get("status", "")):
            raise HTTPException(status_code=409, detail="stop or kill before delete")
        ctx.registry.unregister(bot_id)
        return {"deleted": bot_id}

    # ------------------------------------------------- swarm connector
    # The swarm connector bridges external connection bindings (Discord,
    # Slack, lollms.com, ...) onto the mesh bus. The hub exposes its config
    # and lifecycle so an operator can enable it, point it at an ingress
    # topic (usually ``swarm.inbound``, consumed by the Swarm Orchestrator
    # agent), and monitor its state without touching a config file by hand.
    #
    # NOTE: the connector's *platform adapter* (the thing that actually
    # speaks Discord's API) is injected by whichever host owns the connection
    # binding. When no adapter is attached, the endpoint still accepts
    # ingest requests from remote clients that publish on the ingress topic
    # via ``POST /api/swarm/ingest`` — so an external bot can bridge without
    # the hub holding platform credentials.
    @app.get("/api/swarm/status")
    def api_swarm_status() -> Dict[str, Any]:
        ctx = _hub()
        cfg = ctx.connector_config or {}
        status = ctx.connector.status() if ctx.connector is not None else None
        payload: Dict[str, Any] = {"config": cfg, "status": status}
        if ctx.connector_adapter is not None:
            payload["adapter_platform"] = ctx.connector_adapter.platform
        return payload

    @app.get("/api/swarm/bindings")
    def api_swarm_bindings() -> Dict[str, Any]:
        """
        Lists the *live* connection bindings available to bridge. This is what
        the swarm config dialog shows so the operator picks a real platform
        (Discord, Slack, lollms.com, ...) rather than typing a name blind.
        """
        ctx = _hub()
        client = _swarm_client(ctx)
        if client is None:
            return {"bindings": [], "error": "no lollms-client configured"}
        registry = getattr(client, "connections", None)
        bindings: List[Dict[str, Any]] = []
        if isinstance(registry, dict):
            for name, binding in registry.items():
                bindings.append({
                    "name": str(name).lower(),
                    "binding_name": getattr(binding, "binding_name", name),
                    "class": type(binding).__name__,
                    "connected": bool(
                        _safe_call(getattr(binding, "is_connected", None))
                    ),
                })
        else:
            single = getattr(client, "connection", None)
            if single is not None:
                bindings.append({
                    "name": str(
                        getattr(single, "binding_name", "connection")
                    ).lower(),
                    "binding_name": getattr(single, "binding_name", "connection"),
                    "class": type(single).__name__,
                    "connected": bool(
                        _safe_call(getattr(single, "is_connected", None))
                    ),
                })
        return {"bindings": bindings}

    @app.post("/api/swarm/config")
    def api_swarm_config(body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates the connector configuration. Persists it and, when
        ``enabled`` transitions to True, starts a connector instance wired to
        the connection binding named by ``connection_name`` (if any).

        Reconciler contract:
          * disabling stops the running connector AND detaches its adapter;
          * enabling starts the connector and, if ``connection_name`` names a
            live binding, attaches a :class:`ConnectionBindingAdapter` and
            starts it so platform messages flow into the mesh;
          * every transition is idempotent — a repeated POST with the same
            config does not churn the running instance.
        """
        ctx = _hub()
        body = body or {}
        cfg = dict(ctx.connector_config or load_connector_config(ctx.layout))
        for key in (
            "enabled", "ingress_topic", "reply_timeout_seconds",
            "max_concurrent", "announce_identity", "connection_name",
        ):
            if key in body:
                cfg[key] = body[key]
        try:
            save_connector_config(ctx.layout, cfg)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"save failed: {e}")
        ctx.connector_config = cfg

        if cfg.get("enabled"):
            _enable_swarm_connector(ctx, cfg)
        else:
            _disable_swarm_connector(ctx)
        return {
            "config": cfg,
            "status": ctx.connector.status() if ctx.connector else None,
            "adapter_platform": (
                ctx.connector_adapter.platform
                if ctx.connector_adapter is not None else None
            ),
        }

    @app.post("/api/swarm/ingest")
    def api_swarm_ingest(body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts one inbound platform message and publishes it to the mesh
        ingress topic. Used by a platform bridge running OUTSIDE the hub
        (a Discord bot, a lollms.com widget, ...) that wants the hub to route
        through the swarm connector's reply correlation but does not hold the
        connector instance.

        The connector MUST be enabled (so its reply drain is running). If it
        is not, this returns 409 — a publish without a drain would orphan the
        reply.
        """
        ctx = _hub()
        if ctx.connector is None or not ctx.connector.is_running():
            raise HTTPException(
                status_code=409,
                detail="swarm connector is not enabled; enable it first via "
                       "POST /api/swarm/config {enabled: true}",
            )
        body = body or {}
        text = str(body.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=422, detail="text is required")
        from lollms_client.apps.lollms_bot.mesh.connector import InboundMessage

        inbound = InboundMessage(
            text=text,
            sender=str(body.get("sender") or "external"),
            platform=str(body.get("platform") or "external"),
            channel_id=body.get("channel_id"),
            metadata=dict(body.get("metadata") or {}),
        )
        try:
            token = ctx.connector.ingest(inbound)
        except SwarmConnectorError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"reply_token": token}

    # ---------------------------------------------------------- live stream
    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        """
        Pushes (a) status snapshots every 1s and (b) live bus messages
        whenever they appear. The browser uses these to animate the graph.

        Design: the WebSocket opens its own EPHEMERAL pub/sub subscription to
        ``*``-ish topic coverage — actually, we tap every topic discovered
        from the bus and keep re-tapping. To avoid double-delivery to real
        subscribers, we open subscriptions that are immediately abandoned
        (``from_now=True``) — the animation layer just wants "what's
        happening NOW", not replay from history; the recorder handles replay.
        """
        ctx = _hub()
        await ws.accept()
        tapped: set = set()
        watcher_name = f"hub-ws::{os.getpid()}::{time.time_ns()}"
        try:
            while True:
                # 1. Emit the current snapshot.
                snapshot = {
                    "type": "snapshot",
                    "bots": ctx.registry.list_bots(),
                    "topics": ctx.bus.list_topics(),
                }
                await ws.send_text(json.dumps(snapshot))

                # 2. Tap new topics.
                try:
                    known = {t["topic"] for t in ctx.bus.list_topics()}
                except Exception:
                    known = set()
                new_topics = known - tapped
                for topic in new_topics:
                    try:
                        ctx.bus.subscribe(watcher_name, topic, from_now=True)
                        tapped.add(topic)
                    except Exception:
                        pass

                # 3. Drain every tapped topic and forward as events.
                for topic in list(tapped):
                    try:
                        batch = ctx.bus.poll_topic(watcher_name, topic, max_batch=100)
                    except Exception:
                        continue
                    for msg in batch:
                        await ws.send_text(json.dumps({
                            "type": "message",
                            "topic": msg.topic,
                            "sender": msg.sender,
                            "payload": msg.payload,
                            "headers": msg.headers,
                            "correlation_id": msg.correlation_id,
                            "t": msg.created_at,
                        }))

                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            try:
                ctx.bus.unsubscribe(watcher_name)
            except Exception:
                pass

    return app


# ---------------------------------------------------------------------------
# CLI runners
# ---------------------------------------------------------------------------

def run_hub(
    mesh_root: Path,
    *,
    bind: str = "127.0.0.1",
    port: int = 8123,
    token: Optional[str] = None,
    allow_remote: bool = False,
    log_level: str = "info",
) -> int:
    """
    Serves the hub. Returns a process exit code suitable for the CLI.

    Prints an explicit warning when binding a non-loopback address, and a
    second warning when ``allow_remote`` is on without a token — the operator
    must be aware of both risks before continuing.
    """
    if not _HUB_DEPS_OK:  # pragma: no cover
        ASCIIColors.red(
            "The hub requires `fastapi` and `uvicorn`.\n"
            "Install them with:\n    pip install fastapi uvicorn"
        )
        return 2

    layout = MeshLayout(root=Path(mesh_root).expanduser().resolve())
    layout.materialize()

    # Fail-fast sanity checks BEFORE spinning uvicorn (avoids a confusing
    # "bind error" from an inner thread).
    if bind not in ("127.0.0.1", "localhost", "::1"):
        ASCIIColors.yellow(
            f"⚠️  Binding the hub to {bind}:{port} on a public interface. "
            "Anyone on the network can reach the UI and the /mesh/... control API."
        )
    if allow_remote and not token:
        ASCIIColors.yellow(
            "⚠️  allow_remote=True with no token: /mesh/... endpoints are OPEN "
            "to anyone with the URL. Set --token <secret>."
        )

    app = create_hub_app(layout, token=token, allow_remote=allow_remote)
    uvicorn.run(
        app,
        host=bind,
        port=int(port),
        log_level=log_level,
        ws="auto",
        # Reloader intentionally disabled: the hub holds a persistent recorder
        # thread and a sqlite connection; a reload would orphan both.
        reload=False,
    )
    return 0
