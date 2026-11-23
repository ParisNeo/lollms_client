import argparse
import asyncio
import os
import sys
from pathlib import Path

from ascii_colors import ASCIIColors


def _default_mesh_root(explicit: str | None) -> Path:
    """
    Resolves the mesh root for ``hub`` subcommands. Precedence:
      * explicit --mesh-root                (highest)
      * LOLLMS_BOT_MESH_ROOT env var
      * ``~/.lollms-bot/meshes/default``    (sensible default)

    The default lives one folder ABOVE the per-mesh layout to mirror the
    ``<root>/<mesh_id>/`` convention MeshLayout already enforces.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.getenv("LOLLMS_BOT_MESH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    from lollms_client.apps.lollms_bot.config import get_bot_home

    return (get_bot_home() / "meshes").resolve()


def _add_supervisor_subparsers(subparsers) -> None:
    """
    Installs the ``supervisor`` subcommand tree: start / stop / kill /
    restart / status / list / run. This is the CLI for ``MeshSupervisor``,
    usable without the hub UI.
    """
    sup = subparsers.add_parser(
        "supervisor",
        help="Manage bots' processes on the current machine (spawn/stop/kill).",
    )
    sup_sub = sup.add_subparsers(dest="sup_command", required=True)

    def common(p):
        p.add_argument("--mesh-root", default=None)
        p.add_argument("--mesh-id", default="default")

    p_run = sup_sub.add_parser(
        "run", help="Start the supervisor daemon (keeps bots alive)."
    )
    common(p_run)

    p_start = sup_sub.add_parser(
        "start", help="Spawn one bot (must already be registered)."
    )
    common(p_start)
    p_start.add_argument("--bot-id", required=True)

    p_stop = sup_sub.add_parser("stop", help="Gracefully stop one bot.")
    common(p_stop)
    p_stop.add_argument("--bot-id", required=True)

    p_kill = sup_sub.add_parser("kill", help="Force-kill one bot.")
    common(p_kill)
    p_kill.add_argument("--bot-id", required=True)

    p_restart = sup_sub.add_parser("restart", help="Restart one bot.")
    common(p_restart)
    p_restart.add_argument("--bot-id", required=True)

    p_status = sup_sub.add_parser("status", help="Show registry + process state.")
    common(p_status)
    p_status.add_argument("--bot-id", default=None)

    p_list = sup_sub.add_parser("list", help="List running process records.")
    common(p_list)


def _run_supervisor_command(args: argparse.Namespace) -> int:
    """Dispatches the ``supervisor`` subcommand tree. Returns an exit code."""
    import time
    from lollms_client.apps.lollms_bot.mesh.supervisor import (
        MeshSupervisor,
        SupervisorError,
    )
    from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry
    from lollms_client.apps.lollms_bot.mesh.types import MeshLayout

    mesh_root = _default_mesh_root(args.mesh_root)
    layout = MeshLayout.under(mesh_root, args.mesh_id)
    layout.materialize()

    cmd = args.sup_command

    if cmd == "run":
        sup = MeshSupervisor(layout)  # autostart_watchdog=True by default
        ASCIIColors.cyan(
            f"Supervisor running for mesh {args.mesh_id!r} at {layout.root} — "
            f"Ctrl+C to stop (bots keep last status)."
        )
        try:
            # The supervisor's watchdog runs in its own thread. This command
            # is a daemon: no bots are spawned here; the operator starts
            # individual bots via `supervisor start`. Sufficient for
            # "one supervisor per machine" deployments.
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            ASCIIColors.yellow("\nSupervisor stopped.")
            sup.stop_watchdog()
        return 0

    # All other commands are one-shot; construct a short-lived supervisor
    # (it registers to the same registry, so the daemon — if running — will
    # pick up our state changes too).
    sup = MeshSupervisor(layout, autostart_watchdog=False)
    try:
        if cmd == "start":
            out = sup.spawn(args.bot_id)
            ASCIIColors.green(
                f"Started {args.bot_id!r} → pid {out['pid']} (log: {out['log_path']})"
            )
            return 0
        if cmd == "stop":
            out = sup.stop(args.bot_id)
            ASCIIColors.green(f"Stopped {args.bot_id!r} (running was: {out.get('was_running')})")
            return 0
        if cmd == "kill":
            out = sup.kill(args.bot_id)
            ASCIIColors.green(f"Killed {args.bot_id!r} (running was: {out.get('was_running')})")
            return 0
        if cmd == "restart":
            out = sup.restart(args.bot_id)
            ASCIIColors.green(f"Restarted {args.bot_id!r} → pid {out['pid']}")
            return 0
        if cmd == "status":
            reg = MeshRegistry(layout)
            entries = (
                {args.bot_id: reg.get_bot(args.bot_id)}
                if args.bot_id
                else reg.list_bots()
            )
            for bot_id, entry in sorted(entries.items()):
                if entry is None:
                    ASCIIColors.warning(f"  {bot_id}: (not registered)")
                    continue
                status = entry.get("status", "?")
                pid = entry.get("pid")
                host = entry.get("host", "?")
                err = entry.get("error")
                line = f"  {bot_id:20} [{status}] pid={pid} host={host}"
                if err:
                    line += f"  ⚠ {err}"
                ASCIIColors.info(line)
            return 0
        if cmd == "list":
            procs = sup.list_running()
            if not procs:
                ASCIIColors.info("No tracked processes in this shell.")
                return 0
            for p in procs:
                ASCIIColors.info(
                    f"  {p['bot_id']:20} pid={p['pid']} alive={p['alive']} "
                    f"restarts={p['restarts']}"
                )
            return 0
    except SupervisorError as e:
        ASCIIColors.red(f"Supervisor error: {e}")
        return 1
    ASCIIColors.red(f"Unknown supervisor command: {cmd}")
    return 2


def _add_hub_subparsers(subparsers) -> None:
    """Installs the ``hub`` subcommand tree: launch / serve / info / nodes."""
    hub = subparsers.add_parser(
        "hub",
        help="Web Mission Control for the bot mesh (multi-instance manager).",
    )
    hub_sub = hub.add_subparsers(dest="hub_command", required=True)

    def common(p):
        p.add_argument(
            "--mesh-root",
            default=None,
            help="Mesh root directory (default: ~/.lollms-bot/meshes).",
        )
        p.add_argument(
            "--mesh-id",
            default="default",
            help="Mesh id to open (a subfolder of --mesh-root). Default: default.",
        )

    # hub launch — the full UI, loopback by default
    p_launch = hub_sub.add_parser("launch", help="Launch Mission Control (UI).")
    common(p_launch)
    p_launch.add_argument("--bind", default="127.0.0.1",
                          help="Bind address. Use 0.0.0.0 for cross-PC (default: 127.0.0.1).")
    p_launch.add_argument("--port", type=int, default=8123,
                          help="Bind port (default: 8123).")
    p_launch.add_argument("--token", default=None,
                          help="Shared token for the remote-control API. Required to enable cross-PC bot launch.")
    p_launch.add_argument("--allow-remote", action="store_true",
                          help="Enable /mesh/... remote-control endpoints (requires --token).")

    # hub serve — remote-control only, no browser UI needed
    p_serve = hub_sub.add_parser("serve", help="Serve only the remote-control API (headless node).")
    common(p_serve)
    p_serve.add_argument("--bind", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8123)
    p_serve.add_argument("--token", required=True,
                          help="Shared token (REQUIRED for a headless node).")
    p_serve.set_defaults(allow_remote=True)

    # hub info — describe the local mesh presence
    p_info = hub_sub.add_parser("info", help="Print the mesh's bots and topics.")
    common(p_info)

    # hub nodes — list configured remote nodes
    p_nodes = hub_sub.add_parser("nodes", help="List configured remote mesh nodes.")
    common(p_nodes)

    p_nodes_add = hub_sub.add_parser("add-node", help="Register a remote mesh node.")
    common(p_nodes_add)
    p_nodes_add.add_argument("--node-id", required=True)
    p_nodes_add.add_argument("--url", required=True)
    p_nodes_add.add_argument("--token", default="")
    p_nodes_add.add_argument("--label", default="")


def _run_hub_command(args: argparse.Namespace) -> int:
    """Dispatches the ``hub`` subcommand tree. Returns a process exit code."""
    from lollms_client.apps.lollms_bot.mesh.types import MeshLayout

    mesh_root = _default_mesh_root(args.mesh_root)
    layout = MeshLayout.under(mesh_root, args.mesh_id)
    layout.materialize()

    if args.hub_command in ("launch", "serve"):
        from lollms_client.apps.lollms_bot.mesh.gui.server import run_hub

        return run_hub(
            layout.root,
            bind=args.bind,
            port=args.port,
            token=args.token,
            allow_remote=bool(args.allow_remote),
        )

    if args.hub_command == "info":
        from lollms_client.apps.lollms_bot.mesh.bus import MessageBus
        from lollms_client.apps.lollms_bot.mesh.registry import MeshRegistry

        reg = MeshRegistry(layout)
        bots = reg.list_bots()
        ASCIIColors.cyan(f"Mesh: {args.mesh_id}  ({layout.root})")
        if not bots:
            ASCIIColors.info("  (no bots registered)")
        for bot_id, entry in sorted(bots.items()):
            status = entry.get("status", "?")
            tags = ", ".join((entry.get("spec") or {}).get("tags") or [])
            ASCIIColors.info(f"  • {bot_id:20} [{status}] {tags}")
        try:
            with MessageBus(layout) as bus:
                topics = bus.list_topics()
            if topics:
                ASCIIColors.cyan("Topics:")
                for t in topics:
                    ASCIIColors.info(
                        f"  • {t['topic']:30} msgs={t['messages']} subs={t['subscribers']}"
                    )
        except Exception as e:
            ASCIIColors.warning(f"  (could not read bus: {e})")
        return 0

    if args.hub_command == "nodes":
        from lollms_client.apps.lollms_bot.mesh.gui.server import _load_remotes

        remotes = _load_remotes(layout)
        if not remotes:
            ASCIIColors.info("No remote nodes configured.")
            return 0
        for node in remotes.values():
            ASCIIColors.info(
                f"  • {node.node_id:16} {node.url:32} "
                f"mesh={node.mesh_id or '?'} label={node.label or '—'}"
            )
        return 0

    if args.hub_command == "add-node":
        from lollms_client.apps.lollms_bot.mesh.gui.server import (
            _load_remotes,
            _save_remotes,
        )
        from lollms_client.apps.lollms_bot.mesh.remote import (
            RemoteCoordinationError,
            RemoteNode,
        )

        remotes = _load_remotes(layout)
        if args.node_id in remotes:
            ASCIIColors.red(f"Node {args.node_id!r} already registered.")
            return 1
        try:
            node = RemoteNode(
                node_id=args.node_id,
                url=args.url,
                token=args.token,
                label=args.label or args.node_id,
                mesh_id=args.mesh_id,
            )
        except RemoteCoordinationError as e:
            ASCIIColors.red(f"Invalid remote node: {e}")
            return 1
        remotes[args.node_id] = node
        _save_remotes(layout, remotes)
        ASCIIColors.green(f"Registered remote node {args.node_id!r} → {node.url}")
        return 0

    ASCIIColors.red(f"Unknown hub command: {args.hub_command}")
    return 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lollms-Bot: Handbag-based Multi-platform AI Bot Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Run the interactive configuration wizard",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to a specific configuration file (default: ~/.lollms-bot/config.yaml)"
    )
    parser.add_argument(
        "--home",
        type=str,
        default=None,
        help="Override the lollms-bot home directory (default: ~/.lollms-bot)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_hub_subparsers(subparsers)
    _add_supervisor_subparsers(subparsers)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # ``hub`` subcommand tree routes to the Mission Control server and does
    # NOT require the bot's LLM config — a remote-control node may run on a
    # machine that never chats on its own.
    if getattr(args, "command", None) == "hub":
        return _run_hub_command(args)
    if getattr(args, "command", None) == "supervisor":
        return _run_supervisor_command(args)

    from lollms_client.apps.lollms_bot.config import BotConfig, ensure_bot_home

    if args.home:
        os.environ["LOLLMS_BOT_HOME"] = str(Path(args.home).expanduser())

    ensure_bot_home()

    config = BotConfig(config_path=args.config_path)

    if args.config:
        config.run_wizard()
        return 0

    from lollms_client.apps.lollms_bot.daemon import BotDaemon
    daemon = BotDaemon(config)

    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        ASCIIColors.yellow("\n🛑 Bot stopped by user.")
        return 0
    except Exception as e:
        ASCIIColors.red(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())