import os
import sys
import re
import subprocess
import platform
import shlex
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "System Shell"
TOOL_LIBRARY_DESC = "Executes shell commands (bash, cmd, powershell) with adjustable autonomy levels for environment management and tooling."
TOOL_LIBRARY_ICON = "⚙️"

AUTONOMY_LEVEL: str = "safe"
_CONFIRM_HANDLER: Optional[Any] = None


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
            ASCIIColors.info(f"[System Shell] Host configured autonomy level: {AUTONOMY_LEVEL}")
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

{autonomy_desc}"""


def tool_execute_shell_command(
    command: str
) -> Dict[str, Any]:
    """
    Executes a shell command in the current workspace directory.
    Use this for environment management (e.g., pip install), running tests, or interacting with the OS.

    Args:
        command (str): The shell command to execute.
    """
    global AUTONOMY_LEVEL
    is_windows = platform.system() == "Windows"
    autonomy_level = AUTONOMY_LEVEL

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        if autonomy_level == "full_access":
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=os.getcwd(),
                env=env,
                timeout=120
            )
        else:
            if is_windows:
                stripped_cmd = command.strip()
                rmdir_match = re.match(r'^(rmdir|rd)\s+(/[sS]\s+/[qQ]|/[qQ]\s+/[sS])\s+(.+)$', stripped_cmd)
                if rmdir_match:
                    cmd_name = rmdir_match.group(1)
                    cmd_flags = rmdir_match.group(2)
                    raw_dirs = rmdir_match.group(3).strip()
                    parts = shlex.split(raw_dirs, posix=False)
                    if len(parts) > 1:
                        command = " & ".join(f'{cmd_name} {cmd_flags} "{d}"' for d in parts)

            is_safe, risky_reason = _classify_shell_command(command)

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
                    stripped_cmd = command.strip().lower()
                    if is_windows:
                        if stripped_cmd.startswith("rm ") or stripped_cmd == "rm":
                            hint = "\n\n💡 HINT: You are on Windows (cmd.exe). Use 'del <file>' to delete files or 'rmdir /s /q <dir>' to delete directories. 'rm' does not exist."
                        elif stripped_cmd.startswith("cat ") or stripped_cmd == "cat":
                            hint = "\n\n💡 HINT: You are on Windows (cmd.exe). Use 'type <file>' to view files. 'cat' does not exist."
                        elif stripped_cmd.startswith("ls ") or stripped_cmd == "ls":
                            hint = "\n\n💡 HINT: You are on Windows (cmd.exe). Use 'dir' or 'dir /b' to list files. 'ls' does not exist."
                        elif stripped_cmd.startswith("touch "):
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

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=os.getcwd(),
                env=env,
                timeout=60
            )

        error_msg = None
        if result.returncode != 0:
            if result.stderr and result.stderr.strip():
                error_msg = result.stderr
            else:
                if "2>nul" in command or "2>/dev/null" in command:
                    error_msg = f"Command failed with exit code {result.returncode} (stderr was suppressed by 2>nul redirection; target path or file likely does not exist)."
                else:
                    error_msg = f"Command failed with exit code {result.returncode}"

        cmd_banner = f"$ {command}\n"
        raw_stdout = result.stdout or ("(Command executed successfully with no stdout output)" if result.returncode == 0 else "")
        formatted_output = f"{cmd_banner}{raw_stdout}"

        return {
            "success": result.returncode == 0,
            "command": command,
            "output": formatted_output,
            "stderr": result.stderr,
            "error": error_msg,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command execution timed out."
        }
    except UnicodeDecodeError as ude:
        return {
            "success": False,
            "error": f"Unicode decoding error while reading command output: {ude}. Try setting PYTHONIOENCODING=utf-8 or filtering binary output."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }