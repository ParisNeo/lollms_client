"""
PersistentShellSession — a long-lived shell worker for lollms_shell.
====================================================================

Why this exists
---------------
A "shell companion" is only useful if it behaves like *one shell*. Spawning a
fresh `subprocess.run(...)` per tool call means:

    cd ..            # changes the subprocess only — parent unaware
    pwd              # still the old directory
    export FOO=bar   # lost the moment the subprocess exits
    source venv/bin/activate

...which is exactly NOT what a shell does. This module owns a single, durable
child shell process and pipes commands through it with a sentinel-delimited
protocol (the same trick Jupyter's `%cd` and `!cmd` use). State persists
naturally because the process persists.

Protocol
--------
For each command we write:

    <user command>
    __LOLLMS_SENTINEL_<nonce>__ <exit_code_expression>

…and read stdout until we see the sentinel line, at which point the trailing
integer is the exit code. This is robust to multi-line output, prompts, and
interleaved stdout/stderr because we merge stderr into stdout.

Supported shells
----------------
* Windows cmd.exe       (via `cmd.exe /Q /K`)
* Windows PowerShell    (via `powershell -NoLogo -NoProfile -Command -`)
* WSL / Linux bash/zsh  (via `bash -i` / `zsh -i`)
* Any POSIX sh          (fallback via `sh -i`)

The session is thread-safe (single lock) and never leaks the child on __del__.

Authoritative CWD
-----------------
`session.cwd()` returns the WORKER's current working directory, NOT the parent
Python process's. The REPL prompt must reflect this — otherwise a `cd` the
agent performs stays invisible to the user.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional


_log = logging.getLogger(__name__)

# Exit code that the shell reports when it cannot even evaluate the sentinel
# expression (e.g. the shell died mid-command). Distinctive enough to spot.
_SENTINEL_FAIL_EXIT = -9999


# ── PROCESS-GLOBAL SESSION REGISTRY ────────────────────────────────────────
#
# The persistent shell session is created exactly ONCE per process by
# `cli.main()`. But the tool module (`shell_session.py`) can be imported by
# MULTIPLE loaders under DIFFERENT module names (the LCP tools loader uses
# `importlib.util.spec_from_file_location`, generating a name like
# `lollms_shell_session_tool_<hash>`; the handbag loader uses yet another
# name). Each import creates a SEPARATE module object with its OWN `_SESSION`
# global — so attaching to one copy leaves the others at `None`.
#
# `_ACTIVE_SESSION` lives HERE (a module imported exactly once by Python's
# normal import machinery) and is the single source of truth. The tool copies
# call `get_active_session()` when their local `_SESSION` is `None`.
#
# ⚠️ CRITICAL: this module can STILL be imported under more than one identity
# in edge cases (installed package vs. src/, loaders with `sys.path` games,
# circular-import defensive imports catching Exception). To be bulletproof we
# ALSO publish the session on the `sys` module itself — `sys` is a singleton
# and every module copy in the process sees the same `sys` object. Tools read
# this attribute as their ultimate fallback, in addition to this module-level
# variable.
_SYS_ATTR = "_lollms_shell_active_session"
_ACTIVE_SESSION: Optional["PersistentShellSession"] = None


def set_active_session(session: Optional["PersistentShellSession"]) -> None:
    """Registers `session` as the process-wide persistent shell.

    Publishes to TWO places for cross-module-identity safety:
      1. This module's `_ACTIVE_SESSION` (fast local read).
      2. `sys._lollms_shell_active_session` — guaranteed identical across every
         module copy in the process because `sys` is a singleton.
    """
    global _ACTIVE_SESSION
    _ACTIVE_SESSION = session
    try:
        setattr(sys, _SYS_ATTR, session)
    except Exception:  # pragma: no cover — sys is always writable
        pass


def get_active_session() -> Optional["PersistentShellSession"]:
    """Returns the process-wide persistent shell, or None if not yet created.

    Reads from `sys` first — that is the ONLY value guaranteed to be the same
    across divergent module identities (installed vs src vs loader-aliased).
    """
    sys_val = getattr(sys, _SYS_ATTR, None)
    if sys_val is not None:
        return sys_val
    return _ACTIVE_SESSION


def _detect_shell_kind() -> str:
    """
    Best-effort identification of the current *parent* shell.

    This deliberately mirrors the process-tree walk done in `cli.py` — the two
    MUST agree, otherwise the prompt shows one thing and the worker runs
    another. We import lazily to avoid a circular dependency at module import.
    """
    try:
        from .cli import _active_shell  # noqa: WPS433 (deliberate local import)
        return _active_shell()
    except Exception:  # pragma: no cover — defensive only
        pass
    # Fallback: platform default.
    if platform.system() == "Windows":
        return "cmd"
    return os.environ.get("SHELL", "sh").rsplit("/", 1)[-1] or "sh"


def _shell_argv(kind: str) -> list[str]:
    """
    Constructs the argv that launches a PERSISTENT interactive shell with
    stdin kept open and prompts suppressed (so sentinel detection stays clean).
    """
    is_win = platform.system() == "Windows"
    if kind == "powershell":
        exe = "powershell"
        # -NoLogo -NoProfile -NonInteractive -Command -: reads commands from stdin.
        return [exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "-"]
    if kind == "cmd":
        # /D  : skip AutoRun registry scripts (deterministic startup)
        # /Q  : echo off (cleaner stdout for the sentinel protocol)
        # /K  : execute-then-keep-reading-from-stdin
        #
        # IMPORTANT: do NOT pass a trailing "" argument. On some Windows 11
        # builds `cmd.exe /Q /K ""` starts a shell that does NOT treat writes
        # to stdin as subsequent commands — each `cd` runs in a throwaway
        # state, so the child never leaves the directory it was started in.
        # `/D /Q /K` is the canonical, well-documented "interactive via stdin"
        # form. If a future Windows rev regresses on it too, capture-from-prompt
        # in `run()` still keeps `cwd()` correct.
        return ["cmd.exe", "/D", "/Q", "/K"]
    if kind in ("bash", "zsh", "fish", "sh", "dash"):
        exe = "bash" if kind in ("bash", "wsl") else kind
        if is_win:
            # git-bash / MSYS bash by name
            exe = "bash"
        # -i, keep stdin open; --noprofile --norc keep the environment clean
        # (a user profile that prints banners would confuse the sentinel).
        return [exe, "--noprofile", "--norc", "-i"]
    # Unknown — fall back to system sh.
    return ["sh", "-i"]


class PersistentShellSession:
    """
    A durable child shell that keeps CWD / env / variables alive across calls.

    Usage
    -----
        shell = PersistentShellSession(shell_kind="cmd")
        out, code = shell.run("cd ..")
        out, code = shell.run("echo %CD%")          # shows the parent dir
        shell.close()

    Not-a-shell-session use cases (one-off commands) should still call this
    class and then `reset()`. The class intentionally does NOT support being
    spawned twice concurrently — one session per lollms_shell process.
    """

    def __init__(self, shell_kind: Optional[str] = None, cwd: Optional[Path] = None):
        self.shell_kind = (shell_kind or _detect_shell_kind()).lower()
        self._nonce_token = uuid.uuid4().hex[:10]
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen[str]] = None
        self._closed = False
        # Cache of the child's CURRENT working directory. Updated after every
        # `run()` from the child's own trailing prompt line and used directly
        # by `cwd()` so no echo-probe round-trip is needed. This is what makes
        # `cd ..` visible to the REPL prompt on the very next turn.
        self._last_cwd: Optional[Path] = None
        self._start(cwd or Path.cwd())

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _start(self, cwd: Path) -> None:
        argv = _shell_argv(self.shell_kind)
        # Universal newlines; merge stderr into stdout. No shell=True — argv is
        # already explicit and shell=True would re-quote the sentinel.
        try:
            self._proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # line buffered
                universal_newlines=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"Could not start persistent shell ({' '.join(argv)}): {exc}"
            ) from exc
        try:
            self._last_cwd = Path(cwd).resolve()
        except (OSError, ValueError):
            self._last_cwd = Path.cwd()
        _log.info(
            "[PersistentShell] Started %s (pid %s) in %s",
            self.shell_kind, self._proc.pid, self._last_cwd,
        )
        # Prime PowerShell/bash so it doesn't echo a prompt on first use.
        self._drain_startup_noise()

    def _drain_startup_noise(self) -> None:
        """
        Consume any banner / PS1 echo the shell emits at startup. We do this by
        executing a null command and discarding its output — deterministic and
        shell-agnostic.
        """
        try:
            self.run("" if self.shell_kind == "powershell" else "rem" if self.shell_kind == "cmd" else ":")
        except Exception:
            pass  # startup noise is best-effort

    def close(self) -> None:
        """Terminate the child process cleanly. Idempotent."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin:
                try:
                    proc.stdin.write("exit\n")
                    proc.stdin.flush()
                except (OSError, ValueError):
                    pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            pass
        for s in (proc.stdin, proc.stdout):
            try:
                if s:
                    s.close()
            except OSError:
                pass
        _log.info("[PersistentShell] Closed pid %s", proc.pid)

    def __del__(self):  # pragma: no cover — best-effort safety net
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── public API ─────────────────────────────────────────────────────────

    def cwd(self) -> Path:
        """
        Returns the WORKER's actual current working directory. This is what the
        REPL prompt must display so `cd` performed by the agent is visible.

        Primary source: `self._last_cwd`, updated after every `run()` from the
        child's own trailing prompt line. This is the most reliable signal and
        costs zero round-trips — the prompt renders instantly.

        Fallback (only on the very first call before any command has run, or
        after an external cwd change we could not observe): a real `echo`/
        `pwd` probe inside the child. The probe filters cmd's prompt-echo lines
        (`<cwd>><command>`) which are otherwise indistinguishable from output.
        """
        # Fast, correct path: cached value from the last command's prompt.
        if self._last_cwd is not None:
            try:
                if self._last_cwd.is_dir():
                    return self._last_cwd
            except OSError:
                pass

        # Slow path — probe the child once (used only at session start or if
        # something external cleared the cache).
        if self.shell_kind == "cmd":
            out, code = self.run("echo %CD%")
        elif self.shell_kind == "powershell":
            out, code = self.run("(Get-Location).Path")
        else:  # bash / zsh / fish / sh
            out, code = self.run("pwd")

        if code != 0:
            return Path.cwd()

        fallback = Path.cwd()
        for raw_line in reversed(out.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            # cmd prompt-echo artifact: `<cwd>>...` — always reject.
            if ">" in line:
                continue
            if "__LOLLMS_" in line:
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = fallback / candidate
            try:
                if candidate.exists() and candidate.is_dir():
                    self._last_cwd = candidate.resolve()
                    return self._last_cwd
            except OSError:
                continue
        return fallback

    def run(self, command: str, timeout: float = 120.0) -> tuple[str, int]:
        """
        Executes `command` in the persistent shell and returns (output, exit_code).

        Output includes stderr and all command output EXCEPT the sentinel line,
        plus any stuffed echo of the sentinel expression itself. Thread-safe.
        """
        if self._closed:
            return ("Persistent shell session is closed.", 1)

        command = command.rstrip("\r\n")
        if not command.strip():
            return ("", 0)

        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None or proc.stdout is None:
                return ("Persistent shell has terminated unexpectedly.", 1)

            exit_marker = f"{self._nonce_token}_EXIT"
            # The sentinel line must be distinctive and MUST be the last thing
            # the shell echoes. Different shells report exit codes differently.
            if self.shell_kind == "cmd":
                sentinel = f"echo __LOLLMS_{exit_marker}__ %ERRORLEVEL%"
            elif self.shell_kind == "powershell":
                sentinel = (
                    f"Write-Output \"__LOLLMS_{exit_marker}__ $("
                    f"if ($LASTEXITCODE -ne $null) {{ $LASTEXITCODE }} "
                    f"elseif ($?) {{ 0 }} else {{ 1 }})\""
                )
            else:
                sentinel = f"printf '__LOLLMS_{exit_marker}_%s__\\n' \"$?\""

            full = command + "\n" + sentinel + "\n"
            try:
                proc.stdin.write(full)
                proc.stdin.flush()
            except (OSError, ValueError) as exc:
                return (f"Failed to send command to shell: {exc}", 1)

            collected: list[str] = []
            exit_code = _SENTINEL_FAIL_EXIT
            needle = f"__LOLLMS_{exit_marker}__"
            for line in iter(proc.stdout.readline, ""):
                if needle in line:
                    tail = line.split(needle, 1)[1].strip().rstrip("__").strip()
                    try:
                        exit_code = int(tail) if tail else 0
                    except ValueError:
                        exit_code = 0
                    break
                collected.append(line.rstrip("\r"))
            else:
                return (
                    "Persistent shell terminated unexpectedly (EOF on stdout). "
                    "Run `/reset` to start a new session.",
                    1,
                )

            # ── Capture the child's new cwd from its own trailing prompt ──
            # After the command, cmd/PowerShell/bash print a prompt line
            # containing the current working directory. This is the single
            # most reliable source of cwd truth — it is emitted BY the child,
            # not inferred from the command output. Scanning BEFORE cleanup
            # so the prompt line, which we DO strip from output below, is still
            # available here.
            self._update_cwd_from_prompt(collected)

            # Scrub cmd.exe echo artifacts from the collected output:
            #  (a) lines that are the sentinel EXPRESSION echoed back verbatim
            #      (`echo __LOLLMS_..._EXIT %ERRORLEVEL%` when /Q was off)
            #  (b) lines that are `<cwd>><command>` prompts where the shell
            #      echoed the command before executing it.
            # We keep legitimate command output that happens to contain '>' by
            # only dropping lines whose FIRST '>' appears before the first
            # character of the original command text.
            cleaned: list[str] = []
            first_cmd_token = command.split()[0] if command.split() else ""
            for raw in collected:
                s = raw.rstrip("\r")
                stripped = s.strip()
                if not stripped:
                    cleaned.append(s)
                    continue
                # Reject verbatim echoes of our sentinel expression.
                if stripped.startswith("echo __LOLLMS_") or \
                   stripped.startswith("printf '__LOLLMS_") or \
                   stripped.startswith("Write-Output \"__LOLLMS_"):
                    continue
                # Reject `path>command` prompt echoes for the SAME command.
                if first_cmd_token and first_cmd_token in stripped and ">" in stripped:
                    head, _, tail_part = stripped.partition(">")
                    if tail_part.lstrip().startswith(first_cmd_token):
                        continue
                cleaned.append(s)

            while cleaned and cleaned[0].strip() == "":
                cleaned.pop(0)
            while cleaned and cleaned[-1].strip() == "":
                cleaned.pop()
            return ("\n".join(cleaned), exit_code)

    # ── ENVIRONMENT SNAPSHOT ───────────────────────────────────────────────

    def snapshot_env(self, names: tuple[str, ...] = (
        "VIRTUAL_ENV", "CONDA_PREFIX", "VIRTUAL_ENV_PROMPT",
        "PYTHONHOME", "PATH", "PWD",
    )) -> dict[str, str]:
        """Reads the given env vars **from the worker shell**, in one round-trip.

        This is what makes `activate` / `deactivate` / `set VAR=val` visible to
        the CLI's environment block on the VERY NEXT turn. `os.environ` in the
        Python REPL process is untouched by these worker-side changes — reading
        it would show the stale pre-activation env.

        Returns a dict that only contains names that were non-empty in the
        worker. Missing names are omitted (not stored as empty strings) so a
        subsequent `deactivate` shows up as "no venv" rather than
        `VIRTUAL_ENV=""`.

        Never raises: on any failure returns `{}` and the caller falls back to
        `os.environ`.
        """
        if self._closed or not names:
            return {}

        if self.shell_kind == "cmd":
            # `if defined X (echo =X=%X%)` prints `=X=value` only when X is
            # non-empty. This survives values containing spaces/special chars
            # without needing per-var two-step queries. We use `%~` to avoid
            # trailing spaces being stripped from `%PATH%`-like values.
            chunks = []
            for n in names:
                chunks.append(
                    f"if defined {n} (echo ~{n}=%~%{n}%) else (echo ~{n}=)"
                )
            # One echo per variable, each prefixed with a `~NAME=` sentinel so
            # the parser can distinguish required-empty from missing.
            script = " & ".join(chunks)
        elif self.shell_kind == "powershell":
            # Emit each value on its own line prefixed with `~NAME=|`.
            parts = []
            for n in names:
                parts.append(
                    f'Write-Output ("~{n}=|" + [string]::Concat($env:{n}))'
                )
            script = "; ".join(parts)
        else:
            # POSIX: one `printf` per var, prefix `~NAME=|`.
            parts = []
            for n in names:
                parts.append(f"printf '~{n}=|%s\\n' \"${{{n}-}}\"")
            script = "; ".join(parts)

        try:
            out, code = self.run(script, timeout=15.0)
        except Exception:
            return {}
        if code != 0 and not out:
            return {}

        result: dict[str, str] = {}
        for line in out.splitlines():
            line = line.rstrip("\r").lstrip("\ufeff")
            if not line.startswith("~"):
                continue
            try:
                lhs, _, rhs = line.partition("=")
            except ValueError:
                continue
            name = lhs[1:].strip()
            if not name:
                continue
            # Strip the optional `|` sentinel from PowerShell/POSIX branches.
            if rhs.startswith("|"):
                rhs = rhs[1:]
            # cmd `%PATH%` may hold trailing spaces; only trim pure trailing
            # whitespace if the value has no embedded newlines (it can't here).
            if rhs == "":
                continue
            result[name] = rhs
        return result

    def _update_cwd_from_prompt(self, lines: list[str]) -> None:
        """Updates `_last_cwd` from the child's trailing prompt line.

        cmd.exe            → ``C:\\Users\\…\\ai>``
        PowerShell         → ``PS C:\\Users\\…\\ai>``
        bash / zsh / sh    → ``user@host:~/ai$`` (or ``#`` for root)
        fish               → ``user@host /home/user/ai>``
        WSL bash           → ``user@host:/mnt/c/...$``

        The parser is deliberately tolerant: it accepts any line that terminates
        with a known prompt character and, after stripping prompt decorations,
        resolves to an absolute existing directory. On Windows we also try the
        substring after the last ``>`` in case the prompt echo is combined with
        a partial command line. Nothing is updated if no valid candidate is
        found — the previous value (if any) stays.
        """
        if not lines:
            return

        def _accept(candidate: str) -> bool:
            if not candidate:
                return False
            candidate = candidate.strip().strip('"').strip("'").rstrip(">").rstrip("$").rstrip("#").strip()
            if not candidate:
                return False
            # Strip common PowerShell prompt prefix "PS ".
            if candidate.upper().startswith("PS "):
                candidate = candidate[3:].strip()
            # bash/zsh prompt forms: `user@host:<cwd>$` or `#`
            if self.shell_kind in ("bash", "zsh", "sh", "wsl", "fish"):
                colon = candidate.rfind(":")
                if colon != -1:
                    candidate = candidate[colon + 1:].strip()
                # Expand `~`.
                if candidate.startswith("~"):
                    candidate = str(Path.home()) + candidate[1:]
            if not candidate:
                return False
            try:
                p = Path(candidate)
            except (OSError, ValueError):
                return False
            try:
                if p.is_absolute() and p.exists() and p.is_dir():
                    self._last_cwd = p.resolve()
                    return True
            except OSError:
                return False
            return False

        # Walk from the END — the trailing prompt is what matters. Only the
        # last few lines are the prompt; earlier lines are command output.
        for raw in reversed(lines[-4:]):
            stripped = raw.strip()
            if not stripped:
                continue
            # Direct prompt-line match: ends with '>' or '$' or '#'.
            if stripped.endswith((">", "$", "#")):
                if _accept(stripped):
                    return
            # Combined prompt + echoed command line: `<path>>command`. Take
            # the portion before the first '>' if that portion is a directory.
            if ">" in stripped:
                head = stripped.split(">", 1)[0].strip()
                if _accept(head):
                    return

    def reset(self, cwd: Optional[Path] = None) -> None:
        """Kills the worker and starts a fresh one (drops all session state)."""
        self.close()
        self._closed = False
        self._nonce_token = uuid.uuid4().hex[:10]
        self._start(cwd or Path.cwd())