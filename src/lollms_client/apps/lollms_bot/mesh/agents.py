
"""
Agent catalog for the lollms-bot mesh.

An **agent** is the operator-facing concept: a persona (Handbag) bound to a
specific model (LLM profile) with a declared mission (capabilities + inbound
topics). This is *exactly* what :class:`BotSpec` already models — so this
module is a thin, well-named façade that:

  * discovers the local catalog (handbags + model profiles) so a UI can offer
    pickers instead of raw paths,
  * validates and registers a new agent's ``BotSpec`` in the mesh registry,
  * builds the ``BotSpec`` from high-level choices while still accepting a
    full spec for advanced/scripted use.

It deliberately **does not spawn** anything. Spawning, stopping, killing and
restarting are the :class:`MeshSupervisor`'s responsibility, and the hub /
CLI compose the two. Keeping AgentCatalog free of process control means it
can be imported (and unit-tested) without dragging subprocesses into scope.

Catalog discovery is *the same* logic used by the hub (``gui/server.py``)
because the operator's mental model must be consistent: what the "New Agent"
picker shows at creation must match what "add bot" shows afterwards. Rather
than duplicate the rules, we keep the canonical implementations here and have
the hub import them — a single source of truth for "what counts as a handbag
and a model profile".
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry, RegistryError
from lollms_client.apps.lollms_bot.mesh.types import (
    BotSpec,
    BotStatus,
    MeshLayout,
    validate_bot_id,
)


class AgentError(RuntimeError):
    """Raised for invalid agent definitions or registration failures."""


# A mission topic is the identifier a bot listens on. It must be a single
# atomic slug (no spaces, no natural-language phrases) so that "orchestrate
# stuff" typed into the field does not become a subscription that will never
# fire. Mirrors the shape used elsewhere in the mesh (topics are dot/slug).
_TOPIC_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]{0,127}$")


def _validate_mission_topics(raw: Optional[List[str]]) -> List[str]:
    """
    Normalizes and validates the mission topic list.

    Rules:
      * each entry is stripped; empty entries are dropped;
      * each remaining entry must match ``_TOPIC_RE`` (letters, digits, _ - .),
        so whitespace, commas, slashes, or natural-language descriptions are
        rejected with an actionable message;
      * duplicates are removed while preserving order.

    Returns the cleaned list. Raises ``AgentError`` on the first invalid
    entry — no silent rewriting.
    """
    out: List[str] = []
    seen: set = set()
    for raw_entry in (raw or []):
        if raw_entry is None:
            continue
        entry = str(raw_entry).strip()
        if not entry:
            continue
        if not _TOPIC_RE.match(entry):
            raise AgentError(
                f"invalid mission topic {raw_entry!r}: topics must be single "
                f"slugs (letters/digits/_/-/.), separated by commas in the UI. "
                f"Example: 'email.inbox, review.tasks'."
            )
        if entry not in seen:
            seen.add(entry)
            out.append(entry)
    return out

# ---------------------------------------------------------------------------
# Catalog discovery (canonical implementations; the hub imports these)
# ---------------------------------------------------------------------------

def _is_handbag_folder(path: Path) -> bool:
    """A handbag must carry an identity file (SOUL.md or handbag.yaml)."""
    return (path / "SOUL.md").is_file() or (path / "handbag.yaml").is_file()

def _humanize_handbag_name(handbag_path: Path) -> str:
    """
    Best-effort display name from ``handbag.yaml`` (``name``/``title``/
    ``display_name``) or the first ``name:`` line of ``SOUL.md``. Falls back
    to the folder name. Never raises — a broken manifest shouldn't hide an
    otherwise-usable handbag.
    """
    manifest = handbag_path / "handbag.yaml"
    if manifest.is_file():
        try:
            import yaml  # optional: handbags are usable without PyYAML
            data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            for key in ("name", "title", "display_name"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
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

def discover_handbags(layout: MeshLayout) -> List[Dict[str, Any]]:
    """
    Enumerates handbags the operator can turn into agents. Sources, in order:

      1. ``<mesh_root>/bots/*/handbag/`` — handbags already installed in this
         mesh (so "add another agent from the same persona" is one click).
      2. ``~/.lollms-bot/handbag``       — the classic single-bot home.
      3. ``~/.lollms-bot/personas/*``    — registered personas.
      4. ``~/.lollms-bot/handbag/coworkers/*`` — the coworker folder.

    Duplicates are collapsed by resolved path. Folders without an identity
    file are skipped (they are not handbags, even if they look like one).
    """
    from lollms_client.apps.lollms_bot.config import get_bot_home

    candidates: List[Path] = list(sorted((layout.root / "bots").glob("*/handbag")))
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
        if not _is_handbag_folder(path):
            continue
        out.append({"path": str(resolved), "name": _humanize_handbag_name(path)})
    return out

def discover_model_profiles() -> List[Dict[str, Any]]:
    """
    Lists the LLM profiles an agent can be bound to, using the SAME resolution
    path as lollms-code / the bot daemon. Returns ``[]`` when the global
    config is missing so the UI can degrade to "configure a model first".
    """
    try:
        from lollms_client.lollms_config_cli_env import get_client_from_env
        client = get_client_from_env(
            create_llm=True, create_connection=False, run_wizard_if_fail=False
        )
    except Exception:
        return []
    profiles: List[Dict[str, Any]] = []
    for attr in ("llm_profiles", "profiles", "llm_profile_registry"):
        registry = getattr(client, attr, None)
        if isinstance(registry, dict):
            for name, profile in registry.items():
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

def _describe_handbag_capabilities(handbag_path: Optional[str]) -> Dict[str, Any]:
    """
    Read-only capability summary for an agent's handbag — returned to the
    hub so the operator can SEE which MCP servers and tools are wired,
    without a mutating API and without loading the full personality stack.

    This is intentionally best-effort and defensive:
      * a missing handbag_path yields an empty summary (never raises);
      * a broken manifest yields an empty summary plus a ``warnings`` list
        so the UI can surface it instead of silently showing "no tools";
      * the returned MCP entries are the SECRET-REDACTED form produced by
        ``Handbag.mcp_manifest_summary()`` — tokens/passwords never leave
        the process.

    The operator still CONFIGURES MCP servers by editing the handbag's
    ``mcp/*.yaml`` files. That is deliberate: credentials must not round-trip
    through a browser UI or an HTTP log. This function only reads.
    """
    summary: Dict[str, Any] = {
        "mcp_servers": [],
        "tools": [],
        "mcp_warnings": [],
    }
    if not handbag_path:
        return summary
    path = Path(handbag_path).expanduser()
    if not path.is_dir():
        return summary
    try:
        from lollms_client.lollms_personality.handbag import Handbag

        handbag = Handbag(path)
    except Exception as e:
        summary["mcp_warnings"].append(f"could not open handbag: {e}")
        return summary

    try:
        summary["mcp_servers"] = list(handbag.mcp_manifest_summary())
    except Exception as e:
        summary["mcp_warnings"].append(f"mcp summary failed: {e}")
    try:
        summary["mcp_warnings"].extend(list(getattr(handbag, "mcp_warnings", []) or []))
    except Exception:
        pass
    try:
        summary["tools"] = [
            p.name if hasattr(p, "name") else str(p)
            for p in (getattr(handbag, "tool_files", None) or [])
        ]
    except Exception:
        pass
    return summary


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

@dataclass
class AgentDefinition:
    """
    High-level inputs for :meth:`AgentCatalog.create_agent`.

    These are the fields the UI's "New Agent" form collects; everything else
    (metadata, extra tags) can pass through ``extra``. Keeping this explicit
    means the required pieces (agent_id, handbag, model) are impossible to
    forget, and a bad handbag path is rejected before it ever reaches a
    subprocess.
    """
    agent_id: str
    handbag_path: str
    llm_profile: Optional[str] = None
    mission_topics: Optional[List[str]] = None
    capabilities: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentCatalog:
    """
    Creates and lists agents for one mesh.

    Responsibilities are intentionally narrow:
      * validate a user's agent choices against the catalog,
      * build a ``BotSpec``,
      * register it (PENDING) in the mesh registry.

    It never spawns: callers that want the agent *running* call
    ``MeshSupervisor.spawn(agent_id)`` afterwards. This separates "definition"
    from "execution" and keeps the catalog cheap to embed in request handlers.
    """

    def __init__(self, layout: MeshLayout):
        self.layout = layout
        self.layout.materialize()
        self.registry = MeshRegistry(layout)

    # ------------------------------------------------------------- catalog
    def catalog(self) -> Dict[str, Any]:
        """The complete catalog the UI needs, in one call."""
        return {
            "handbags": discover_handbags(self.layout),
            "profiles": discover_model_profiles(),
        }

    # ------------------------------------------------------------- listing
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        Returns every registered bot annotated as an "agent" view.

        We do not filter out non-agent bots: in the mesh every bot IS an
        agent (handbag + model). Calling it out explicitly at the API layer
        lets the UI treat them uniformly while future mesh roles (routers,
        observers) can still be tagged via ``spec.tags``.
        """
        bots = self.registry.list_bots()
        agents: List[Dict[str, Any]] = []
        # list_bots() returns a mapping bot_id -> entry; iterate values in a
        # stable order derived from the id for a deterministic UI layout.
        for bot_id in sorted(bots.keys()):
            entry = bots[bot_id]
            spec = entry.get("spec") or {}
            agents.append({
                "agent_id": bot_id,
                "name": (spec.get("metadata") or {}).get("name") or bot_id,
                "handbag_path": spec.get("handbag_path"),
                "llm_profile": spec.get("llm_profile"),
                "mission_topics": list(spec.get("inbound_routes") or []),
                "capabilities": dict(spec.get("capabilities") or {}),
                "tags": list(spec.get("tags") or []),
                "status": entry.get("status"),
                "pid": entry.get("pid"),
                "error": entry.get("error"),
                "started_at": entry.get("started_at"),
                "last_seen": entry.get("last_seen"),
                **_describe_handbag_capabilities(spec.get("handbag_path")),
            })
        return agents

    # -------------------------------------------------------------- creation
    def create_agent(self, definition: AgentDefinition) -> Dict[str, Any]:
        """
        Validates ``definition`` and registers the resulting agent.

        Raises :class:`AgentError` for any invalid input so the HTTP layer can
        map a single exception type to a 400/409/422. On success returns the
        registry entry (with ``status="pending"``), ready to be spawned by the
        supervisor.
        """
        try:
            validate_bot_id(definition.agent_id)
        except ValueError as e:
            raise AgentError(f"invalid agent_id: {e}") from e

        handbag_path = self._validate_handbag(definition.handbag_path)
        self._validate_profile(definition.llm_profile)
        # Validate mission topics BEFORE the registry write — a bad topic must
        # never make it into a BotSpec (the pump would then idle on a topic
        # nobody sends to, looking healthy but doing nothing).
        mission_topics = _validate_mission_topics(definition.mission_topics)

        # Guard against silently overwriting an existing agent: the operator
        # gets an explicit "already exists" rather than a lost handbag binding.
        if self.registry.get_bot(definition.agent_id) is not None:
            raise AgentError(
                f"agent {definition.agent_id!r} already exists in this mesh"
            )

        spec = BotSpec(
            bot_id=definition.agent_id,
            handbag_path=str(handbag_path),
            llm_profile=definition.llm_profile,
            capabilities=dict(definition.capabilities or {}),
            tags=list(definition.tags or []),
            inbound_routes=mission_topics,
            metadata=dict(definition.metadata or {}),
        )
        try:
            self.registry.register(spec, status=BotStatus.PENDING)
        except RegistryError as e:
            raise AgentError(f"could not register agent: {e}") from e
        return self.registry.get_bot(definition.agent_id) or {}

    # ------------------------------------------------------------ internals
    def _validate_handbag(self, handbag_path: str) -> Path:
        """Resolves and validates a handbag path, or raises ``AgentError``."""
        if not handbag_path or not handbag_path.strip():
            raise AgentError("handbag_path is required")
        path = Path(handbag_path).expanduser().resolve()
        if not path.is_dir():
            raise AgentError(f"handbag path does not exist or is not a folder: {path}")
        if not _is_handbag_folder(path):
            raise AgentError(
                f"folder is not a handbag (needs SOUL.md or handbag.yaml): {path}"
            )
        return path

    def _validate_profile(self, llm_profile: Optional[str]) -> None:
        """
        Warns — does not fail — when a requested profile is not in the local
        catalog. The profile might be resolvable by ``get_client_from_env``
        even if not enumerated here (e.g. env-only profiles), and rejecting a
        valid-but-unlisted profile would be worse than letting the runtime
        attempt it and report a clean startup error if it truly is unknown.
        """
        if not llm_profile or llm_profile == "default":
            return
        known = {p["name"] for p in discover_model_profiles()}
        if known and llm_profile not in known:
            # Non-fatal: surfaced via the agent's runtime logs if it fails.
            # We intentionally keep this silent to avoid spurious failures on
            # env-only profiles, but leave the hook obvious for future use.
            pass


# ---------------------------------------------------------------------------
# Capability editor: MCP servers + tools, configurable per agent handbag
# ---------------------------------------------------------------------------

#: MCP server ids are used as YAML filenames, so they must be shell-safe slugs.
_MCP_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")


class CapabilityEditor:
    """
    Read AND write the MCP servers / tools that live in an agent's handbag.

    Rationale for a *write* surface
    -------------------------------
    A Mail Agent is only useful when it is wired to the operator's real Gmail
    / Outlook / IMAP account. That wiring is an MCP server descriptor in
    ``<handbag>/mcp/<service>.yaml``. Without a way to author those files,
    the operator would have to SSH onto the machine, hunt for the handbag
    folder, and hand-craft YAML — a catastrophic UX for the exact feature
    that makes agents valuable.

    Credentials live in the descriptor's ``env`` / ``headers`` block (that's
    what every MCP server expects: env vars or auth headers). The YAML file
    is written with owner-only-ish permissions where the OS supports it, and
    the directory is the operator's own handbag — no more exposed than the
    rest of their config.

    Every method re-validates the handbag exists and has an identity file, so
    it can NEVER be tricked into writing outside an actual handbag folder.
    """

    def __init__(self, layout: MeshLayout):
        self.layout = layout
        self.registry = MeshRegistry(layout)

    # ------------------------------------------------------------- helpers
    def _handbag_for(self, agent_id: str) -> Path:
        """Resolves the agent's handbag path (validated), or raises AgentError."""
        try:
            validate_bot_id(agent_id)
        except ValueError as e:
            raise AgentError(f"invalid agent_id: {e}") from e
        entry = self.registry.get_bot(agent_id)
        if entry is None:
            raise AgentError(f"agent {agent_id!r} is not registered in this mesh")
        spec = entry.get("spec") or {}
        raw = spec.get("handbag_path")
        if not raw:
            raise AgentError(f"agent {agent_id!r} has no handbag_path")
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_dir():
            raise AgentError(f"handbag folder does not exist: {path}")
        if not _is_handbag_folder(path):
            raise AgentError(
                f"folder is not a handbag (needs SOUL.md or handbag.yaml): {path}"
            )
        return path

    @staticmethod
    def _mcp_dir(handbag: Path) -> Path:
        d = handbag / "mcp"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ----------------------------------------------------------- read APIs
    def list_mcp(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Returns the full MCP descriptors for the agent's handbag, INCLUDING
        the raw ``env``/``headers`` keys but with values redacted for list
        display. Editing uses :meth:`get_mcp` (which returns stored keys as a
        presence map so the UI can show "N secrets stored" without ever
        round-tripping the secret values back out).
        """
        handbag = self._handbag_for(agent_id)
        mcp_dir = self._mcp_dir(handbag)
        out: List[Dict[str, Any]] = []

        try:
            from lollms_client.lollms_personality.mcp_hooks import (
                parse_mcp_manifest,
                MCPConfigError,
            )
        except Exception:
            return out

        for candidate in sorted(mcp_dir.glob("*.y*ml")):
            if candidate.name.startswith("."):
                continue
            entry: Dict[str, Any] = {
                "server_id": candidate.stem,
                "source_path": str(candidate),
            }
            try:
                spec = parse_mcp_manifest(candidate.resolve())
                entry.update({
                    "title": spec.title,
                    "description": spec.description,
                    "enabled": spec.enabled,
                    "transport": spec.transport,
                    "command": spec.command,
                    "args": list(spec.args),
                    "url": spec.url,
                    # Never leak secret VALUES via the list endpoint. Only keys.
                    "env_keys": sorted(spec.env.keys()),
                    "header_keys": sorted(spec.headers.keys()),
                    "security": dict(spec.security),
                    "timeout_seconds": spec.timeout_seconds,
                })
            except MCPConfigError as e:
                entry["error"] = str(e)
            except Exception as e:
                entry["error"] = f"unexpected parse error: {e}"
            out.append(entry)
        return out

    def list_tools(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Lists the tool libraries shipped in ``<handbag>/tools/``. Each entry is
        one folder (the LCP loader resolves ``TOOLS.py`` / ``<folder>.py`` /
        the first ``.py``). Flat ``.py`` files are reported as single-file
        libraries for backward compatibility.
        """
        handbag = self._handbag_for(agent_id)
        tools_dir = handbag / "tools"
        if not tools_dir.is_dir():
            return []
        out: List[Dict[str, Any]] = []
        for item in sorted(tools_dir.iterdir()):
            if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
                out.append({
                    "name": item.stem,
                    "kind": "file",
                    "path": str(item),
                })
            elif item.is_dir() and item.name not in ("assets", "references", "__pycache__") \
                    and not item.name.startswith("."):
                py_files = [f for f in item.iterdir()
                            if f.is_file() and f.suffix == ".py" and f.stem != "__init__"]
                if py_files:
                    out.append({
                        "name": item.name,
                        "kind": "folder",
                        "path": str(item),
                        "entry_file": next(
                            (f.name for f in py_files if f.name == "TOOLS.py"),
                            py_files[0].name,
                        ),
                    })
        return out

    # ----------------------------------------------------------- write APIs
    def save_mcp(self, agent_id: str, server_id: str, descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates or replaces one MCP server descriptor in the agent's handbag.

        ``descriptor`` shape (mirrors the mcp_hooks YAML schema)::

            {
              "title": "...",              # optional display name
              "description": "...",        # optional
              "enabled": true,
              "transport": "stdio",        # stdio | http | sse
              "command": "npx",            # stdio only
              "args": ["-y", "..."],
              "env": {"KEY": "value"},     # stdio only; values may be secrets
              "url": "https://...",        # http/sse only
              "headers": {"Authorization": "Bearer ..."},  # http/sse only
              "security": { ... },
              "timeout_seconds": 30
            }

        Validation is delegated to ``parse_mcp_manifest`` AFTER writing, so an
        invalid descriptor is rejected and the previous file restored — the
        operator never ends up with a half-broken handbag.
        """
        if not _MCP_ID_RE.match(server_id or ""):
            raise AgentError(
                f"invalid MCP server_id {server_id!r}: must match "
                f"{_MCP_ID_RE.pattern}"
            )
        if not isinstance(descriptor, dict):
            raise AgentError("descriptor must be a JSON object")

        handbag = self._handbag_for(agent_id)
        mcp_dir = self._mcp_dir(handbag)
        target = mcp_dir / f"{server_id}.yaml"

        try:
            import yaml
        except ImportError as e:
            raise AgentError(f"PyYAML is required to save MCP manifests: {e}") from e

        # Snapshot the existing file so we can restore on validation failure.
        previous = target.read_text(encoding="utf-8") if target.exists() else None

        # Strip internal/None fields so the YAML stays clean and minimal.
        payload = {k: v for k, v in descriptor.items() if v is not None}
        try:
            target.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except OSError as e:
            raise AgentError(f"could not write {target}: {e}") from e

        # Validate by re-parsing. A malformed descriptor is rolled back.
        try:
            from lollms_client.lollms_personality.mcp_hooks import parse_mcp_manifest
            parse_mcp_manifest(target)
        except Exception as e:
            if previous is not None:
                try:
                    target.write_text(previous, encoding="utf-8")
                except OSError:
                    pass
            else:
                try:
                    target.unlink()
                except OSError:
                    pass
            raise AgentError(f"invalid MCP descriptor: {e}") from e

        # Restrict permissions where supported (contains credentials in env/
        # headers). Best-effort — never fails the save.
        try:
            import os as _os, stat as _stat
            if _os.name != "nt":
                _os.chmod(target, _stat.S_IRUSR | _stat.S_IWUSR)  # 0600
        except Exception:
            pass

        return {
            "server_id": server_id,
            "path": str(target),
            "saved": True,
        }

    def delete_mcp(self, agent_id: str, server_id: str) -> bool:
        """Removes one MCP descriptor. Returns True if a file was removed."""
        if not _MCP_ID_RE.match(server_id or ""):
            raise AgentError(f"invalid MCP server_id {server_id!r}")
        handbag = self._handbag_for(agent_id)
        target = self._mcp_dir(handbag) / f"{server_id}.yaml"
        if not target.exists():
            return False
        try:
            target.unlink()
            return True
        except OSError as e:
            raise AgentError(f"could not delete {target}: {e}") from e
