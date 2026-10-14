"""
lollms_code/instance_manager.py
author: ParisNeo
description: Multi-instance orchestrator for lollms-code.

Lets an active agent (or a shell) spawn, track, wait on, and kill multiple
lollms-code subprocess instances. Each instance runs a non-interactive task
(`lollms-code run <task>`) and can be pointed at a distinct config file and
LLM model profile — the mechanism for fanning out work across models of
different power/speed/cost.

Two ways to drive it:
  1. As a library: ``InstanceManager().spawn(...)`` / ``.wait()`` / ``.kill()``.
  2. As a console script: ``lollms-code-instances spawn <task> ...``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LollmsCodeInstance:
    """A single running (or finished) lollms-code subprocess."""
    id: str
    task: str
    pid: Optional[int] = None
    proc: Optional[subprocess.Popen] = None
    workspace: Optional[str] = None
    output: Optional[str] = None
    config: Optional[str] = None
    llm_profile: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    returncode: Optional[int] = None

    @property
    def alive(self) -> bool:
        if self.proc is None:
            return False
        return self.proc.poll() is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "pid": self.pid,
            "alive": self.alive,
            "workspace": self.workspace,
            "output": self.output,
            "config": self.config,
            "llm_profile": self.llm_profile,
            "returncode": self.returncode if not self.alive else None,
            "started_at": self.started_at,
        }


class InstanceManager:
    """Tracks and controls a set of lollms-code subprocess instances."""

    def __init__(self) -> None:
        self._instances: Dict[str, LollmsCodeInstance] = {}

    @staticmethod
    def _build_command(
        task: str,
        config: Optional[str],
        output: Optional[str],
        workspace: Optional[str],
        llm_profile: Optional[str],
        extra_args: Optional[List[str]],
    ) -> List[str]:
        base: List[str]
        entry = shutil.which("lollms-code")
        if entry:
            base = [entry, "run", task]
        else:
            base = [sys.executable, "-m", "lollms_client.apps.lollms_code.cli", "run", task]

        if config:
            base += ["--config", config]
        if llm_profile:
            base += ["--llm-profile", llm_profile]
        if workspace:
            base += ["--workspace", workspace]
        if output:
            base += ["--output", output]
        if extra_args:
            base += list(extra_args)
        return base

    def spawn(
        self,
        task: str,
        config: Optional[str] = None,
        output: Optional[str] = None,
        workspace: Optional[str] = None,
        llm_profile: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        capture: bool = False,
    ) -> LollmsCodeInstance:
        """Spawns a lollms-code run instance and returns its tracked record."""
        if not task or not str(task).strip():
            raise ValueError("task must be a non-empty string.")

        cmd = self._build_command(task, config, output, workspace, llm_profile, extra_args)
        popen_kwargs: Dict[str, Any] = {"cwd": workspace or None}
        if capture:
            popen_kwargs["stdout"] = subprocess.PIPE
            popen_kwargs["stderr"] = subprocess.STDOUT

        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Failed to launch lollms-code (command not found): {' '.join(cmd[:3])}... ({e})"
            ) from e

        inst = LollmsCodeInstance(
            id=uuid.uuid4().hex[:8],
            task=str(task),
            pid=proc.pid,
            proc=proc,
            workspace=workspace,
            output=output,
            config=config,
            llm_profile=llm_profile,
        )
        self._instances[inst.id] = inst
        return inst

    def get(self, instance_id: str) -> Optional[LollmsCodeInstance]:
        return self._instances.get(instance_id)

    def wait(self, instance_id: str, timeout: Optional[float] = None) -> int:
        inst = self._get_or_raise(instance_id)
        try:
            inst.returncode = inst.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise
        return inst.returncode

    def kill(self, instance_id: str) -> bool:
        inst = self._get_or_raise(instance_id)
        if not inst.alive:
            return False
        try:
            inst.proc.terminate()
            try:
                inst.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                inst.proc.kill()
                inst.proc.wait(timeout=5)
            inst.returncode = inst.proc.returncode
            return True
        except OSError:
            return False

    def kill_all(self) -> int:
        killed = 0
        for inst in list(self._instances.values()):
            if self.kill(inst.id):
                killed += 1
        return killed

    def status(self) -> List[Dict[str, Any]]:
        return [inst.to_dict() for inst in self._instances.values()]

    @property
    def all_instances(self) -> List[LollmsCodeInstance]:
        return list(self._instances.values())

    def _get_or_raise(self, instance_id: str) -> LollmsCodeInstance:
        inst = self._instances.get(instance_id)
        if inst is None:
            raise KeyError(f"Unknown lollms-code instance: '{instance_id}'.")
        return inst


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lollms-code-instances",
        description="Spawn and manage multiple lollms-code instances.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_spawn = subparsers.add_parser("spawn", help="Spawn a new lollms-code run instance.")
    p_spawn.add_argument("task", help="The task prompt for the agent.")
    p_spawn.add_argument("--config", type=str, default=None, help="Path to a config file (.env/.json/.yaml).")
    p_spawn.add_argument("--llm-profile", type=str, default=None, help="LLM model profile alias to use.")
    p_spawn.add_argument("--workspace", type=str, default=None, help="Workspace directory.")
    p_spawn.add_argument("--output", type=str, default=None, help="File to write the final output to.")
    p_spawn.add_argument("--wait", action="store_true", help="Block until the instance finishes.")

    p_list = subparsers.add_parser("list", help="List tracked instances.")
    p_kill = subparsers.add_parser("kill", help="Kill an instance (or 'all').")
    p_kill.add_argument("id", help="Instance id, or 'all'.")
    p_wait = subparsers.add_parser("wait", help="Wait for an instance to finish.")
    p_wait.add_argument("id", nargs="?", default=None, help="Instance id (all if omitted).")

    args = parser.parse_args(argv)

    # NOTE: The console-script CLI is stateless across invocations; in-process
    # orchestration (the primary use case) uses the InstanceManager class directly.
    mgr = InstanceManager()

    if args.command == "spawn":
        inst = mgr.spawn(
            task=args.task,
            config=args.config,
            output=args.output,
            workspace=args.workspace,
            llm_profile=args.llm_profile,
        )
        print(json.dumps(inst.to_dict(), indent=2))
        if args.wait:
            return int(mgr.wait(inst.id) or 0)
        return 0
    if args.command == "list":
        print(json.dumps(mgr.status(), indent=2))
        return 0
    if args.command == "kill":
        if args.id == "all":
            print(json.dumps({"killed": mgr.kill_all()}, indent=2))
        else:
            ok = mgr.kill(args.id)
            print(json.dumps({"id": args.id, "killed": ok}, indent=2))
        return 0
    if args.command == "wait":
        targets = mgr.all_instances if not args.id else [mgr.get(args.id)]
        if not targets or all(t is None for t in targets):
            print(json.dumps({"error": "no matching instance"}, indent=2))
            return 1
        codes = {}
        for t in targets:
            if t is not None:
                codes[t.id] = mgr.wait(t.id)
        print(json.dumps(codes, indent=2))
        return 0 if all(v is not None and v == 0 for v in codes.values()) else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())