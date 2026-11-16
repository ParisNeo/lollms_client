import os
import sys
import re
import time
import uuid
import signal
import threading
import subprocess
import platform
import shlex
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "System Shell"
TOOL_LIBRARY_DESC = "Executes shell commands (bash, cmd, powershell) with adjustable autonomy levels for environment management and tooling. Supports long-running background jobs with polling."
TOOL_LIBRARY_ICON = "⚙️"

AUTONOMY_LEVEL: str = "safe"
_CONFIRM_HANDLER: Optional[Any] = None

# ──────────────────────────────────────────────────────────────────────────────
# Long-running job support (background execution with polling)
#
# Problem: a shell tool call cannot distinguish a 3-second `dir` from a
# 20-minute compile or a `ffmpeg` encode. A single hard timeout inevitably
# either kills legitimate work or blocks the agent round for too long.
#
# Solution: a bounded background-job manager. The LLM is instructed (in the
# tool description) to:
#   1. Launch the task with `background=true` → receives a `job_id`.
#   2. Call `tool_poll_shell_job(job_id, wait_seconds=N)` → blocks for at
#      most N seconds, returns the current status + output tail.
#   3. Repeat polling until `status == "completed"|"failed"|"killed"`.
#
# Guardrails preserve the DoS protection the hard timeout provided:
#   - `_MAX_BACKGROUND_JOBS` : concurrent job cap.
#   - `_MAX_JOB_LIFETIME`    : absolute wall-clock cap (auto-kill).
#   - `_MAX_CAPTURE_BYTES`   : bounded per-stream capture (ring-buffer).
#   - `_MAX_POLL_WAIT`       : bounded poll wait so the agent round never hangs.
#   - Process-tree kill on request and on lifetime expiry.
#   - Every launch passes through the SAME autonomy classifier as foreground.
# ──────────────────────────────────────────────────────────────────────────────

_MAX_BACKGROUND_JOBS: int = 8
_MAX_JOB_LIFETIME: float = 1800.0            # 30 minutes hard cap per job
_MAX_CAPTURE_BYTES: int = 512 * 1024         # 512 KiB per stream
_KEEP_FINISHED_JOBS: int = 32                # retained completed/failed jobs
_MAX_POLL_WAIT: float = 60.0                 # cap on tool_poll_shell_job wait
_FOREGROUND_DEFAULT_TIMEOUT: float = 120.0
_FOREGROUND_MAX_TIMEOUT: float = 600.0       # 10 minutes absolute foreground cap


class _ShellJob:
    """In-memory record for a background shell job with a bounded output ring."""

    __slots__ = (
        "job_id", "command", "proc", "started_at", "finished_at",
        "stdout_buf", "stderr_buf", "stdout_total", "stderr_total",
        "return_code", "error_msg", "status",
        "_lock", "_done_event", "_stdout_thread", "_stderr_thread",
        "_killed_by", "_killed_at", "progress_cb",
    )

    def __init__(self, job_id: str, command: str, proc: subprocess.Popen):
        self.job_id = job_id
        self.command = command
        self.proc = proc
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        # Bounded ring buffers: keep the LAST N bytes so the tail is what the
        # LLM needs when it polls (early output of a long task is usually noise,
        # whereas the last lines are what the model has to react to).
        self.stdout_buf: deque = deque(maxlen=_MAX_CAPTURE_BYTES)
        self.stderr_buf: deque = deque(maxlen=_MAX_CAPTURE_BYTES)
        self.stdout_total: int = 0
        self.stderr_total: int = 0
        self.return_code: Optional[int] = None
        self.error_msg: Optional[str] = None
        self.status: str = "running"   # running | completed | failed | killed
        self._lock = threading.Lock()
        self._done_event = threading.Event()
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._killed_by: Optional[str] = None
        self._killed_at: Optional[float] = None
        # Optional per-line callback (line, metadata) for live host streaming.
        # Foreground calls set it; background calls leave it None.
        self.progress_cb: Optional[Any] = None


_JOBS: Dict[str, _ShellJob] = {}
_JOBS_LOCK = threading.Lock()


def _bounded_append(buf: deque, total_attr_ref: List[int], chunk: str) -> None:
    """Appends `chunk` to a bounded ring buffer, tracking bytes-in for truncation flags."""
    if not chunk:
        return
    total_attr_ref[0] += len(chunk)
    # Split into characters-safe chunks: `deque(maxlen=N)` elements, one per char,
    # would waste memory. Instead, keep short string blocks. When the buffer is
    # near the cap, drop oldest blocks until under the cap.
    buf.append(chunk)
    current = sum(len(b) for b in buf)
    while current > _MAX_CAPTURE_BYTES and len(buf) > 1:
        dropped = buf.popleft()
        current -= len(dropped)


def _join_bounded(buf: deque) -> str:
    return "".join(buf)


def _trim_jobs_locked() -> None:
    """Retains at most `_KEEP_FINISHED_JOBS` finished jobs (oldest first removal)."""
    finished = [
        (jid, job) for jid, job in _JOBS.items()
        if job.status in ("completed", "failed", "killed")
    ]
    if len(finished) <= _KEEP_FINISHED_JOBS:
        return
    finished.sort(key=lambda kv: kv[1].finished_at or kv[1].started_at)
    to_remove = len(finished) - _KEEP_FINISHED_JOBS
    for jid, _job in finished[:to_remove]:
        _JOBS.pop(jid, None)


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminates a process AND its children (cross-platform best-effort)."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return

    if platform.system() == "Windows":
        # `taskkill /F /T` kills the whole tree rooted at the PID. The command
        # is invoked DIRECTLY (not via shell) so it cannot be intercepted yet
        # again by our own classifier — that would block cleanup.
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False, timeout=10,
            )
            return
        except Exception:
            pass

    # POSIX / fallback: try process group, then terminate/kill.
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=3)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _drain_stream(stream, job: _ShellJob, buf: deque, total_ref: List[int], label: str) -> None:
    """Background reader: pumps one stream into the job's bounded buffer."""
    try:
        for raw_line in iter(stream.readline, ""):
            if not raw_line:
                break
            _bounded_append(buf, total_ref, raw_line)
            # Mirror the live byte total so polling a RUNNING job reports an
            # accurate count instead of zero until finalize.
            with job._lock:
                if label == "stdout":
                    job.stdout_total = total_ref[0]
                else:
                    job.stderr_total = total_ref[0]
            cb = job.progress_cb
            if cb is not None:
                try:
                    cb(raw_line, {"stream": label})
                except Exception:
                    pass
    except Exception as exc:
        with job._lock:
            if not job.error_msg:
                job.error_msg = f"{label} read error: {exc}"
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _finalize_job(job: _ShellJob, return_code: Optional[int], killed_by: Optional[str] = None) -> None:
    """Marks a job as finished exactly once and records its return code."""
    with job._lock:
        if job.status != "running":
            return  # already finalized by kill or lifespan thread
        job.finished_at = time.time()
        job.return_code = return_code
        if killed_by:
            job._killed_by = killed_by
            job._killed_at = job.finished_at
            job.status = "killed"
            job.error_msg = job.error_msg or f"Job terminated: {killed_by}"
        elif return_code == 0:
            job.status = "completed"
        else:
            job.status = "failed"
    job._done_event.set()

    # Trim finished-job retention outside the lock (best-effort).
    with _JOBS_LOCK:
        _trim_jobs_locked()


def _lifespan_watchdog(job: _ShellJob) -> None:
    """Auto-kills a running job that exceeds `_MAX_JOB_LIFETIME`."""
    try:
        while not job._done_event.wait(timeout=5.0):
            if time.time() - job.started_at >= _MAX_JOB_LIFETIME:
                ASCIIColors.warning(
                    f"[system_shell] Job {job.job_id} exceeded {_MAX_JOB_LIFETIME:.0f}s — auto-killing."
                )
                _kill_process_tree(job.proc)
                _finalize_job(job, None, killed_by=f"lifetime exceeded ({_MAX_JOB_LIFETIME:.0f}s)")
                return
    except Exception as exc:
        ASCIIColors.warning(f"[system_shell] Lifespan watchdog error for {job.job_id}: {exc}")


def _launch_job(
    command: str,
    env: Dict[str, str],
    progress_cb: Optional[Any] = None,
) -> Tuple[Optional[_ShellJob], Optional[Dict[str, Any]]]:
    """Launches a supervised job. Returns (job, None) or (None, error_dict).

    The single launch path shared by BOTH modes:
      * background mode ("fire and poll"), and
      * foreground mode, which waits on the same job and — on timeout — adopts
        the still-running process instead of killing it (no work is lost).

    A brief 0.5s probe reports immediate crashes synchronously; if the process
    is still alive, the lifespan watchdog is armed and the live job is returned.
    """
    with _JOBS_LOCK:
        running = sum(
            1 for j in _JOBS.values() if j is not None and j.status == "running"
        )
        if running >= _MAX_BACKGROUND_JOBS:
            return None, {
                "success": False,
                "error": (
                    f"Concurrent job limit reached ({running}/{_MAX_BACKGROUND_JOBS}). "
                    f"Poll or kill an existing job before launching another."
                ),
            }
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        # Reserve the slot NOW before releasing the lock so a concurrent
        # launch cannot race past the cap.
        _JOBS[job_id] = None  # type: ignore[assignment]

    try:
        creationflags = 0
        popen_kwargs: Dict[str, Any] = {}
        if platform.system() == "Windows":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            # New session → own process group → tree-killable via killpg.
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=os.getcwd(),
            env=env,
            bufsize=1,
            creationflags=creationflags,
            **popen_kwargs,
        )
    except Exception as exc:
        with _JOBS_LOCK:
            _JOBS.pop(job_id, None)
        return None, {"success": False, "error": f"Failed to launch command: {exc}"}

    job = _ShellJob(job_id, command, proc)
    job.progress_cb = progress_cb
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    st_ref: List[int] = [0]
    se_ref: List[int] = [0]
    t_out = threading.Thread(
        target=_drain_stream,
        args=(proc.stdout, job, job.stdout_buf, st_ref, "stdout"),
        daemon=True,
        name=f"shelljob-{job_id}-out",
    )
    t_err = threading.Thread(
        target=_drain_stream,
        args=(proc.stderr, job, job.stderr_buf, se_ref, "stderr"),
        daemon=True,
        name=f"shelljob-{job_id}-err",
    )
    job._stdout_thread = t_out
    job._stderr_thread = t_err
    t_out.start()
    t_err.start()

    # Brief probe: report an immediate crash synchronously. If the process is
    # still running after 0.5s we arm the lifespan watchdog and hand the live
    # job back — the caller decides whether to keep waiting or poll.
    try:
        rc = proc.wait(timeout=0.5)
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        with job._lock:
            job.stdout_total = st_ref[0]
            job.stderr_total = se_ref[0]
        _finalize_job(job, rc)
    except subprocess.TimeoutExpired:
        watchdog = threading.Thread(
            target=_lifespan_watchdog, args=(job,),
            daemon=True, name=f"shelljob-{job_id}-wdt",
        )
        watchdog.start()

    return job, None


def _start_background_job(command: str, env: Dict[str, str]) -> Dict[str, Any]:
    """Launches a command detached and returns the initial job response."""
    job, error = _launch_job(command, env, progress_cb=None)
    if error is not None:
        return error
    if job is None:
        return {"success": False, "error": "Internal error: job launch returned no job."}

    resp = _job_status_payload(job, include_tail=True)
    resp["success"] = True
    resp["output"] = (
        f"✅ Background job launched: {job.job_id}\n"
        f"Command: {command}\n"
        f"Status: {resp['status']}\n\n"
        f"Poll with: `tool_poll_shell_job(job_id=\"{job.job_id}\", wait_seconds=30)`"
    )
    return resp


def _job_status_payload(job: _ShellJob, include_tail: bool = True) -> Dict[str, Any]:
    """Builds a serialisable status snapshot for one job."""
    with job._lock:
        finished = job.status != "running"
        # Completed drain-join (best effort) once the process exits so we
        # capture the very last flushed bytes.
        if finished and job._stdout_thread and job._stdout_thread.is_alive():
            job._stdout_thread.join(timeout=1.0)
        if finished and job._stderr_thread and job._stderr_thread.is_alive():
            job._stderr_thread.join(timeout=1.0)

        stdout_text = _join_bounded(job.stdout_buf)
        stderr_text = _join_bounded(job.stderr_buf)
        stdout_trunc = len(stdout_text) < job.stdout_total
        stderr_trunc = len(stderr_text) < job.stderr_total

        payload: Dict[str, Any] = {
            "job_id": job.job_id,
            "command": job.command,
            "status": job.status,
            "running": not finished,
            "return_code": job.return_code,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "elapsed_seconds": round(
                (job.finished_at or time.time()) - job.started_at, 2
            ),
            "stdout_bytes_total": job.stdout_total,
            "stderr_bytes_total": job.stderr_total,
            "stdout_truncated": stdout_trunc,
            "stderr_truncated": stderr_trunc,
            "error": job.error_msg,
        }
        if include_tail:
            # Cap tail per poll so a huge single poll response cannot blow the
            # agent context window.
            tail_cap = 32 * 1024
            if len(stdout_text) > tail_cap:
                stdout_text = stdout_text[-tail_cap:]
                payload["stdout_truncated"] = True
            if len(stderr_text) > tail_cap:
                stderr_text = stderr_text[-tail_cap:]
                payload["stderr_truncated"] = True
            payload["stdout_tail"] = stdout_text
            payload["stderr_tail"] = stderr_text
        return payload


def set_confirm_handler(handler: Optional[Any]) -> None:
    """Sets a custom confirmation handler across all loaded and persistent LCP module instances."""
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler
    for mod_name, mod in list(sys.modules.items()):
        if "system_shell" in mod_name and mod is not sys.modules.get(__name__):
            try:
                mod._CONFIRM_HANDLER = handler
            except Exception:
                pass


def init_tools_library(config: dict|None = None) -> None:
    global AUTONOMY_LEVEL, _CONFIRM_HANDLER
    if config and isinstance(config, dict):
        autonomy = config.get("autonomy_level", "safe").lower().strip()
        if autonomy in ("strict", "safe", "full_access"):
            AUTONOMY_LEVEL = autonomy
            ASCIIColors.debug(f"[System Shell] Host configured autonomy level: {AUTONOMY_LEVEL}")
        else:
            ASCIIColors.warning(f"[System Shell] Invalid autonomy level '{autonomy}' received. Defaulting to 'safe'.")
            AUTONOMY_LEVEL = "safe"
        if "confirm_handler" in config:
            set_confirm_handler(config.get("confirm_handler"))
    else:
        AUTONOMY_LEVEL = "safe"


_NETWORK_DOWNLOAD_COMMANDS = {
    "curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "sftp", "ftp",
    "telnet", "certutil", "bitsadmin", "nslookup", "ping", "tracert",
    "traceroute", "nmap", "tshark", "tcpdump", "iwr", "irm",
    "invoke-webrequest", "invoke-restmethod", "start-bitstransfer"
}

_SYSTEM_DESTRUCTIVE_COMMANDS = {
    "format", "diskpart", "shutdown", "reboot", "taskkill", "kill", "pkill",
    "reg", "regedit", "net", "netsh", "sc", "chmod", "chown", "useradd",
    "usermod", "iptables", "ufw"
}

def _get_safe_commands() -> set:
    is_windows = platform.system() == "Windows"
    cmds = {
        "dir", "echo", "type", "cd", "pip", "python", "py", "git",
        "ls", "pwd", "cat", "head", "tail", "mkdir", "rmdir", "del",
        "where", "which", "set", "env"
    }
    if is_windows:
        cmds.update({
            "copy", "move", "ren", "rename", "md", "rd", "cls", "erase", "if",
            "chdir", "pushd", "popd", "tree", "find", "findstr",
            "sort", "more", "help", "ver", "vol", "label", "time", "date"
        })
    else:
        cmds.update({"rm", "cp", "mv", "touch", "grep", "clear", "export", "find", "sort"})
    return cmds

def _classify_shell_command(command: str) -> tuple[bool, Optional[str]]:
    """
    Evaluates shell command safety in safe mode.
    Returns (is_safe: bool, risky_reason: Optional[str]).
    """
    is_windows = platform.system() == "Windows"
    safe_commands = _get_safe_commands()
    stripped = command.strip()
    if not stripped:
        return True, None

    try:
        parts = shlex.split(stripped, posix=(not is_windows))
    except ValueError:
        parts = stripped.split()

    if not parts:
        return True, None

    base_cmd = os.path.basename(parts[0]).lower()
    if base_cmd.endswith(".exe"):
        base_cmd = base_cmd[:-4]

    # 1. Direct check for Network / Download utilities
    if base_cmd in _NETWORK_DOWNLOAD_COMMANDS:
        return False, f"Outbound network / download utility '{base_cmd}'"

    # 2. Direct check for System Destructive / Privileged commands
    if base_cmd in _SYSTEM_DESTRUCTIVE_COMMANDS:
        return False, f"Privileged / destructive system command '{base_cmd}'"

    # 3. Peeling shell interpreter escapes (cmd /c, powershell -command, bash -c)
    if base_cmd in ("cmd", "powershell", "pwsh", "bash", "sh", "zsh"):
        sub_cmd_parts = []
        for idx, token in enumerate(parts[1:]):
            if token.lower() in ("/c", "-c", "-command", "-enc", "-encodedcommand"):
                sub_cmd_parts = parts[idx + 2:]
                break
        if sub_cmd_parts:
            inner_sub_cmd = " ".join(sub_cmd_parts)
            return _classify_shell_command(inner_sub_cmd)
        else:
            return False, f"Interactive shell interpreter invocation '{base_cmd}'"

    # 4. Inspect Python invocations
    if base_cmd in ("python", "py", "python3"):
        if "-c" in parts:
            idx = parts.index("-c")
            if idx + 1 < len(parts):
                code_payload = parts[idx + 1].lower()
                forbidden_patterns = [
                    "subprocess", "os.system", "os.popen", "pty", "popen",
                    "spawn", "execv", "execl", "socket", "urllib", "requests", "httpx"
                ]
                for p in forbidden_patterns:
                    if p in code_payload:
                        return False, f"Python one-liner with process or network escape '{p}'"
        return True, None

    # 5. Check if command is in safe commands whitelist
    if base_cmd in safe_commands:
        cmd_lower = stripped.lower()
        for net_tool in _NETWORK_DOWNLOAD_COMMANDS:
            if re.search(r'\b' + re.escape(net_tool) + r'\b', cmd_lower):
                return False, f"Command references network tool '{net_tool}'"
        return True, None

    return False, f"Unwhitelisted shell binary '{base_cmd}'"

def _is_safe_command(command: str) -> bool:
    safe, _ = _classify_shell_command(command)
    return safe

def _can_prompt_interactive() -> bool:
    """Checks whether standard input stream is an interactive terminal."""
    if not sys.stdin:
        return False
    try:
        return sys.stdin.isatty()
    except Exception:
        return False

def _prompt_user_validation(command: str, script_label: str) -> Tuple[str, str]:
    """
    Validates command execution using:
      1. Registered host confirm_handler (GUI, async queue, headless API).
      2. Interactive terminal CLI prompt if stdin is a TTY.
      3. Non-interactive fallback: cleanly assumes refusal without blocking.
    """
    global _CONFIRM_HANDLER

    handler = _CONFIRM_HANDLER
    if not handler:
        persistent_name = "lollms_client.tools_bindings.lcp.persistent_system_shell"
        if persistent_name in sys.modules:
            handler = getattr(sys.modules[persistent_name], "_CONFIRM_HANDLER", None)

    # 1. Custom host/GUI confirmation handler
    if handler is not None and callable(handler):
        try:
            try:
                res = handler(command, script_label)
            except TypeError:
                try:
                    res = handler(command, script_label, None)
                except TypeError:
                    try:
                        res = handler({
                            "tool_name": "system_shell",
                            "action_type": "shell_command",
                            "command": command,
                            "source": command,
                            "script_label": script_label,
                            "label": script_label,
                            "content": command,
                            "metadata": {"command": command, "autonomy_level": AUTONOMY_LEVEL}
                        })
                    except TypeError:
                        res = handler(command)

            if isinstance(res, tuple):
                decision = res[0]
                reason = res[1] if len(res) > 1 else ""
            elif isinstance(res, bool):
                decision, reason = ("allow", "") if res else ("reject", "Declined by operator.")
            elif isinstance(res, str):
                decision, reason = res, ""
            else:
                decision, reason = "allow", ""
            return str(decision).lower().strip(), str(reason)
        except Exception as handler_err:
            ASCIIColors.warning(f"[system_shell] Confirm handler failed: {handler_err}")
            return "reject", f"Confirmation handler failed: {handler_err}"

    # 2. Interactive terminal prompt
    if _can_prompt_interactive():
        panel_parts = [
            f"[bold cyan]Command:[/bold cyan] [yellow]$ {command}[/yellow]",
            f"[bold cyan]Autonomy Mode:[/bold cyan] [green]{AUTONOMY_LEVEL.upper()}[/green] (Protected Workspace Sandbox)",
            "\n[bold yellow]⚠️  The LLM agent wants to execute this shell command on your system.[/bold yellow]"
        ]

        ASCIIColors.panel(
            "\n".join(panel_parts),
            title=f"[bold yellow]🛡️ Shell Command Authorization ({AUTONOMY_LEVEL.upper()} Mode)[/bold yellow]",
            border_style="yellow"
        )

        while True:
            try:
                choice = input("  Authorize command? [y]es / [n]o / [a]lways for session: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return "reject", "Execution interrupted by operator (Ctrl+C / EOF)."

            if choice in ("", "y", "yes"):
                return "allow", ""
            elif choice in ("a", "always"):
                return "always", ""
            elif choice in ("n", "no"):
                try:
                    reason = input("  Reason / feedback for the LLM (optional, press Enter to skip): ").strip()
                except (EOFError, KeyboardInterrupt):
                    reason = ""
                return "reject", reason or "Operator declined execution."
            else:
                ASCIIColors.yellow("  Invalid choice. Please enter 'y', 'n', or 'a'.")

    # 3. Non-interactive fallback: assume refusal immediately without blocking
    return "reject", "Non-interactive environment: no user interaction channel available to authorize command execution in safe mode (refusal assumed without blocking)."


def tool_execute_shell_command_prompt() -> str:
    """
    Dynamically generates the detailed environment description for tool_execute_shell_command.
    Informs the LLM of the exact OS, underlying shell, CWD, allowed commands, and syntax rules.
    """
    is_windows = platform.system() == "Windows"
    os_name = platform.system()
    os_release = platform.release()
    os_arch = platform.machine()
    cwd = os.getcwd()
    py_exec = sys.executable

    safe_cmds_str = ", ".join(sorted(_get_safe_commands()))

    if is_windows:
        shell_info = f"""CURRENT SHELL ENVIRONMENT:
- Operating System: {os_name} {os_release} ({os_arch})
- Underlying Shell: cmd.exe (Windows Command Prompt, invoked via %COMSPEC% with shell=True).
  ⚠️ NOT bash, NOT zsh, NOT PowerShell!
- Working Directory (CWD): {cwd}
- Python Executable: {py_exec} (call as 'python', not 'python3')
- Autonomy Mode: {AUTONOMY_LEVEL.upper()}

MANDATORY WINDOWS CMD.EXE SYNTAX RULES:
1. DELETING FILES: Use `del <filename>` or `erase <filename>` (e.g. `del helloworld.py`).
   NEVER use `rm` or `rm -f` (they DO NOT exist in cmd.exe and are blocked).
2. DELETING FOLDERS: Use `rmdir /s /q <folder>` or `rd /s /q <folder>`. NEVER use `rm -rf`.
3. LISTING FILES: Use `dir` or `dir /b` (bare format). NEVER use `ls`.
4. VIEWING FILE CONTENT: Use `type <filename>` (or prefer <unlock_file>). NEVER use `cat`.
5. COPY / MOVE / RENAME: Use `copy`, `move`, `ren` / `rename`.
6. VERIFYING DELETION OR FILE EXISTENCE:
   - Preferred cmd.exe check: `if exist <filename> (echo EXISTS) else (echo NOT_FOUND)`
   - Or Python check: `python -c "import os; print('DELETED' if not os.path.exists('<filename>') else 'EXISTS')"`
   - ⚠️ DO NOT use `dir <filename>` to check if a file was deleted! In cmd.exe, running `dir` on a non-existent file exits with code 1 ('File Not Found' / 'Fichier introuvable'), which marks the tool execution as FAILED.
7. PATH SEPARATORS: Use backslashes '\\' or forward slashes '/' (backslashes are safest for cmd.exe built-in commands like `del`, `dir`, `type`)."""
    else:
        shell_info = f"""CURRENT SHELL ENVIRONMENT:
- Operating System: {os_name} {os_release} ({os_arch})
- Underlying Shell: /bin/sh or bash (POSIX shell with shell=True).
- Working Directory (CWD): {cwd}
- Python Executable: {py_exec} (call as 'python3' or 'python')
- Autonomy Mode: {AUTONOMY_LEVEL.upper()}

POSIX SHELL SYNTAX RULES:
1. DELETING FILES: Use `rm <filename>` or `rm -f <filename>`.
2. DELETING FOLDERS: Use `rm -rf <folder>`.
3. LISTING FILES: Use `ls` or `ls -la`.
4. VIEWING FILE CONTENT: Use `cat <filename>` (or prefer <unlock_file>).
5. VERIFYING EXISTENCE: `test -f <filename> && echo EXISTS || echo NOT_FOUND`.
6. PATH SEPARATORS: Use forward slash '/'."""

    if AUTONOMY_LEVEL == "full_access":
        autonomy_desc = (
            "AUTONOMY: FULL ACCESS mode is active. You can run all shell commands, tests, and environment tools."
        )
    elif AUTONOMY_LEVEL == "strict":
        autonomy_desc = (
            "AUTONOMY: STRICT MODE is active. All shell commands require human operator approval."
        )
    else:
        autonomy_desc = (
            f"AUTONOMY: SAFE MODE is active (Default):\n"
            f"• Auto-approved local commands: {safe_cmds_str}\n"
            f"• Network & diagnostic utilities (ping, curl, wget, ssh, nslookup, etc.) are NOT auto-approved and will prompt the user for authorization before running."
        )

    return f"""Executes a shell command in the current workspace directory.
Use this for environment management (pip install), running tests, or interacting with the OS.

{shell_info}

{autonomy_desc}

LONG-RUNNING TASKS (COMPILATION, ENCODING, DEEP SCANS, VIDEO GENERATION):
- A foreground call blocks the agent round and is capped at a wait window
  (default {_FOREGROUND_DEFAULT_TIMEOUT:.0f}s).
- AUTO-PROMOTION SAFETY NET: if a foreground command exceeds its wait window it
  is NOT killed. The still-running process is promoted to a background job and
  you receive its `job_id` (with `auto_promoted: true`). No work is lost — just
  poll that job_id. You should STILL use background mode up-front for known
  long tasks, to avoid a wasted foreground wait.
- For any task likely to exceed ~{_FOREGROUND_DEFAULT_TIMEOUT/2:.0f}s (compiling, video encoding,
  large scans, downloads, model training, package installs, media renders), use BACKGROUND MODE:
    1. Launch: `tool_execute_shell_command(command="...", background=true)`
       → returns a `job_id` immediately.
    2. Poll periodically: `tool_poll_shell_job(job_id="...", wait_seconds=30)`
       → blocks at most `wait_seconds` (cap {_MAX_POLL_WAIT:.0f}s), returns
       status + output tail. Repeat until status is `completed`/`failed`/`killed`.
    3. Retrieve result: `tool_get_shell_job_result(job_id="...")` (final tail).
    4. Terminate early : `tool_kill_shell_job(job_id="...")`.
    5. Inspect all: `tool_list_shell_jobs()`.
    6. Free memory: `tool_cleanup_shell_jobs()`.
- Background jobs are auto-killed after {_MAX_JOB_LIFETIME/60:.0f} minutes (lifetime cap) and
  are capped at {_MAX_BACKGROUND_JOBS} concurrent jobs. Output is truncated to the last
  {_MAX_CAPTURE_BYTES//1024} KiB per stream; the tail is what you receive."""


def tool_execute_shell_command(
    command: str,
    env: Optional[Any] = None,
    background: bool = False,
    wait_timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes a shell command in the current workspace directory.

    Use this for environment management (e.g., pip install), running tests, or
    interacting with the OS.

    Args:
        command (str): The shell command to execute.
        env: Host-injected unified runtime context; its tool_context (when present)
            streams output lines live to the terminal/UI.
        background (bool): When True, the command is launched detached and a
            `job_id` is returned immediately. Use `tool_poll_shell_job` to
            monitor it. Recommended for compilation, encoding, downloads, and
            any long-running task.
        wait_timeout_seconds (float, optional): In foreground mode, overrides
            the default {_FOREGROUND_DEFAULT_TIMEOUT:.0f}s wait window for this call (capped
            at {_FOREGROUND_MAX_TIMEOUT:.0f}s). If the command exceeds it, the process is
            auto-promoted to a background job (its job_id is returned). Ignored
            when background=True — background mode returns immediately.
    """
    global AUTONOMY_LEVEL
    tool_ctx = getattr(env, "tool_context", None) if env else None
    is_windows = platform.system() == "Windows"
    autonomy_level = AUTONOMY_LEVEL

    # ── Foreground timeout resolution (bounded) ──────────────────────────────
    fg_timeout = _FOREGROUND_DEFAULT_TIMEOUT
    if wait_timeout_seconds is not None:
        try:
            fg_timeout = float(wait_timeout_seconds)
        except (TypeError, ValueError):
            fg_timeout = _FOREGROUND_DEFAULT_TIMEOUT
        if fg_timeout <= 0:
            fg_timeout = _FOREGROUND_DEFAULT_TIMEOUT
        if fg_timeout > _FOREGROUND_MAX_TIMEOUT:
            fg_timeout = _FOREGROUND_MAX_TIMEOUT

    base_env = os.environ.copy()
    base_env["PYTHONUTF8"] = "1"
    base_env["PYTHONIOENCODING"] = "utf-8"
    base_env["PYTHONUNBUFFERED"] = "1"

    # ── Autonomy gate (identical for foreground and background) ──────────────
    try:
        gate_command = command
        if is_windows:
            stripped_cmd = command.strip()
            rmdir_match = re.match(r'^(rmdir|rd)\s+(/[sS]\s+/[qQ]|/[qQ]\s+/[sS])\s+(.+)$', stripped_cmd)
            if rmdir_match:
                cmd_name = rmdir_match.group(1)
                cmd_flags = rmdir_match.group(2)
                raw_dirs = rmdir_match.group(3).strip()
                parts = shlex.split(raw_dirs, posix=False)
                if len(parts) > 1:
                    gate_command = " & ".join(f'{cmd_name} {cmd_flags} "{d}"' for d in parts)

        if autonomy_level != "full_access":
            is_safe, risky_reason = _classify_shell_command(gate_command)
            if not is_safe or autonomy_level == "strict":
                prompt_label = f"shell: {command}"
                if autonomy_level == "strict":
                    prompt_label = f"[STRICT] shell: {command}"
                elif risky_reason:
                    prompt_label = f"[RISKY: {risky_reason}] shell: {command}"

                decision, rejection_reason = _prompt_user_validation(command, prompt_label)
                if decision in ("allow", "always"):
                    if decision == "always":
                        AUTONOMY_LEVEL = "full_access"
                        ASCIIColors.success("[system_shell] 🔓 Shell autonomy upgraded to full_access for this session.")
                else:
                    reason_msg = rejection_reason or f"Restricted shell operation ({risky_reason or 'unwhitelisted command'}) denied by operator."
                    allowed_list = ", ".join(sorted(_get_safe_commands()))
                    hint = ""
                    stripped_l = command.strip().lower()
                    if is_windows:
                        if stripped_l.startswith("rm ") or stripped_l == "rm":
                            hint = "\n\n💡 HINT: You are on Windows (cmd.exe). Use 'del <file>' to delete files or 'rmdir /s /q <dir>' to delete directories. 'rm' does not exist."
                        elif stripped_l.startswith("cat ") or stripped_l == "cat":
                            hint = "\n\n💡 HINT: You are on Windows (cmd.exe). Use 'type <file>' to view files. 'cat' does not exist."
                        elif stripped_l.startswith("ls ") or stripped_l == "ls":
                            hint = "\n\n💡 HINT: You are on Windows (cmd.exe). Use 'dir' or 'dir /b' to list files. 'ls' does not exist."
                        elif stripped_l.startswith("touch "):
                            hint = "\n\n💡 HINT: You are on Windows (cmd.exe). Use 'type nul > <file>' or create it using an <artifact> tag. 'touch' does not exist."

                    return {
                        "success": False,
                        "output": (
                            f"🛑 BLOCKED BY SANDBOX: The command '{command}' was not authorized.\n\n"
                            f"Reason: {reason_msg}\n"
                            f"The system shell is currently in '{autonomy_level}' mode on {platform.system()} ({'cmd.exe' if is_windows else 'sh/bash'}).\n"
                            f"Allowed safe commands include: {allowed_list}.{hint}\n\n"
                            f"⚠️ **ACTION REQUIRED FROM THE OPERATOR**: If this task requires elevated privileges, "
                            f"please enable 'full_access' mode by typing `/shell` or configuring the host application."
                        ),
                        "error": f"Blocked by sandbox ({autonomy_level} mode): {reason_msg}{hint}"
                    }
    except Exception as gate_err:
        return {"success": False, "error": f"Autonomy gate error: {gate_err}"}

    # ── Background branch ────────────────────────────────────────────────────
    if background:
        return _start_background_job(command, base_env)

    # ── Foreground branch — runs as a job, self-promotes on timeout ───────────
    # Foreground execution uses the SAME supervised-job machinery as background
    # mode. This is what lets a timeout "adopt" the still-running process
    # instead of killing it: a legitimate long task (video render, model
    # warm-up, large install) is never destroyed — the caller simply receives a
    # pollable job_id and keeps its work.
    def _emit_progress(line: str, meta: Dict[str, Any]) -> None:
        clean = line.rstrip("\r\n")
        if clean and tool_ctx and hasattr(tool_ctx, "emit_progress"):
            try:
                tool_ctx.emit_progress(clean, meta)
            except Exception:
                pass

    progress_cb = _emit_progress if tool_ctx is not None else None

    job, launch_err = _launch_job(command, base_env, progress_cb=progress_cb)
    if launch_err is not None:
        return launch_err
    if job is None:
        return {"success": False, "error": "Internal error: job launch returned no job."}

    finished = job._done_event.wait(timeout=fg_timeout)

    if not finished:
        # Timed out → PROMOTE the still-running process to a background job.
        # Do NOT kill it: the caller polls job_id and recovers the result.
        payload = _job_status_payload(job, include_tail=True)
        payload["success"] = True
        payload["mode"] = "background"
        payload["auto_promoted"] = True
        payload["adopted_from"] = "foreground_timeout"
        payload["timeout_seconds"] = fg_timeout
        payload["output"] = (
            f"⏳ Foreground wait exceeded {fg_timeout:.0f}s — the command is still "
            f"running and has been promoted to background job '{job.job_id}'. "
            f"Its work is NOT lost.\n"
            f"Command: {command}\n\n"
            f"Poll with: `tool_poll_shell_job(job_id=\"{job.job_id}\", wait_seconds=30)` "
            f"until status is 'completed'/'failed'/'killed'."
        )
        return payload

    # Completed within the foreground window — format the classic result shape.
    payload = _job_status_payload(job, include_tail=True)
    retcode = job.return_code if job.return_code is not None else -1
    out_raw = payload.get("stdout_tail", "") or ""
    err_raw = payload.get("stderr_tail", "") or ""

    error_msg = None
    if retcode != 0:
        if err_raw and err_raw.strip():
            error_msg = err_raw
        elif "2>nul" in command or "2>/dev/null" in command:
            error_msg = (
                f"Command failed with exit code {retcode} (stderr was suppressed "
                f"by 2>nul redirection; target path or file likely does not exist)."
            )
        else:
            error_msg = f"Command failed with exit code {retcode}"

    cmd_banner = f"$ {command}\n"
    raw_stdout = out_raw or (
        "(Command executed successfully with no stdout output)" if retcode == 0 else ""
    )
    return {
        "success": retcode == 0,
        "command": command,
        "output": f"{cmd_banner}{raw_stdout}",
        "stderr": err_raw,
        "error": error_msg,
        "return_code": retcode,
        "mode": "foreground",
        "timeout_seconds": fg_timeout,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Background job control tools
# ──────────────────────────────────────────────────────────────────────────────

def tool_poll_shell_job(job_id: str, wait_seconds: float = 30.0) -> Dict[str, Any]:
    """
    Polls a background shell job. Blocks up to `wait_seconds` (capped at
    `_MAX_POLL_WAIT`) waiting for the job to complete, then returns the current
    status with output tail. Call this repeatedly until the status is
    `completed`, `failed`, or `killed`.

    Args:
        job_id (str): The ID returned by `tool_execute_shell_command(background=true)`.
        wait_seconds (float): Maximum blocking seconds for this poll (default 30,
            capped at 60). Use 0 for a non-blocking status check.
    """
    try:
        wait_s = float(wait_seconds) if wait_seconds is not None else 30.0
    except (TypeError, ValueError):
        wait_s = 30.0
    if wait_s < 0:
        wait_s = 0.0
    if wait_s > _MAX_POLL_WAIT:
        wait_s = _MAX_POLL_WAIT

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return {"success": False, "error": f"Unknown job_id '{job_id}'. Use tool_list_shell_jobs() to see active jobs."}

    if wait_s > 0 and job.status == "running":
        job._done_event.wait(timeout=wait_s)

    payload = _job_status_payload(job, include_tail=True)
    payload["success"] = True
    payload["mode"] = "background"
    return payload


def tool_get_shell_job_result(job_id: str) -> Dict[str, Any]:
    """
    Retrieves the final result of a background shell job (or the current
    snapshot when still running). Equivalent to `tool_poll_shell_job` with
    zero wait; handy for a terminal read once the model has already polled
    until completion.

    Args:
        job_id (str): The job ID.
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return {"success": False, "error": f"Unknown job_id '{job_id}'."}
    payload = _job_status_payload(job, include_tail=True)
    payload["success"] = True
    payload["mode"] = "background"
    return payload


def tool_list_shell_jobs() -> Dict[str, Any]:
    """
    Lists all tracked background shell jobs (running, completed, failed, killed)
    with their status, elapsed time, and exit code. Use this when you need to
    decide which job to poll or kill.
    """
    with _JOBS_LOCK:
        jobs_snapshot = list(_JOBS.values())

    entries: List[Dict[str, Any]] = []
    for job in jobs_snapshot:
        if job is None:
            continue
        with job._lock:
            entries.append({
                "job_id": job.job_id,
                "command": job.command,
                "status": job.status,
                "return_code": job.return_code,
                "started_at": job.started_at,
                "elapsed_seconds": round(
                    (job.finished_at or time.time()) - job.started_at, 2
                ),
                "stdout_bytes_total": job.stdout_total,
                "stderr_bytes_total": job.stderr_total,
            })
    entries.sort(key=lambda e: e["started_at"])
    return {
        "success": True,
        "count": len(entries),
        "running_count": sum(1 for e in entries if e["status"] == "running"),
        "max_concurrent": _MAX_BACKGROUND_JOBS,
        "jobs": entries,
    }


def tool_kill_shell_job(job_id: str) -> Dict[str, Any]:
    """
    Terminates a background shell job AND its process tree. Safe to call on
    an already-finished job (no-op).

    Args:
        job_id (str): The job ID to terminate.
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return {"success": False, "error": f"Unknown job_id '{job_id}'."}

    if job.status != "running":
        payload = _job_status_payload(job, include_tail=True)
        payload["success"] = True
        payload["output"] = f"Job '{job_id}' is already {job.status}; nothing to kill."
        return payload

    _kill_process_tree(job.proc)
    _finalize_job(job, None, killed_by="operator requested")

    payload = _job_status_payload(job, include_tail=True)
    payload["success"] = True
    payload["output"] = f"🛑 Job '{job_id}' terminated."
    return payload


def tool_cleanup_shell_jobs(keep_last: int = 8) -> Dict[str, Any]:
    """
    Removes FINISHED background jobs from memory (running jobs are never
    touched). Keeps the most recent `keep_last` finished jobs when they exceed
    the retention cap.

    Args:
        keep_last (int): Number of most recent finished jobs to retain (>=0).
    """
    try:
        keep_last_int = int(keep_last) if keep_last is not None else 8
    except (TypeError, ValueError):
        keep_last_int = 8
    if keep_last_int < 0:
        keep_last_int = 0

    with _JOBS_LOCK:
        finished = [
            (jid, job) for jid, job in _JOBS.items()
            if job is not None and job.status in ("completed", "failed", "killed")
        ]
        finished.sort(key=lambda kv: kv[1].finished_at or kv[1].started_at)
        remove_count = max(0, len(finished) - keep_last_int)
        removed_ids = [jid for jid, _ in finished[:remove_count]]
        for jid in removed_ids:
            _JOBS.pop(jid, None)

    return {
        "success": True,
        "removed_count": len(removed_ids),
        "removed_job_ids": removed_ids,
        "remaining_jobs": len(_JOBS),
    }