import os
import sys
import re
import subprocess
import platform
import shlex
from typing import Any, Dict
from pathlib import Path
from typing import Optional
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "System Shell"
TOOL_LIBRARY_DESC = "Executes shell commands (bash, cmd, powershell) with adjustable autonomy levels for environment management and tooling."
TOOL_LIBRARY_ICON = "⚙️"

AUTONOMY_LEVEL: str = "safe"
_CONFIRM_HANDLER: Optional[Any] = None


def set_confirm_handler(handler: Optional[Any]) -> None:
    """Sets a custom confirmation handler across both current and persistent LCP module instances."""
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler
    for mod_name in (
        "lollms_client.tools_bindings.lcp.persistent_system_shell",
        "lollms_client.tools_bindings.lcp.default_tools.system_shell.system_shell"
    ):
        if mod_name in sys.modules and sys.modules[mod_name] is not sys.modules.get(__name__):
            try:
                sys.modules[mod_name]._CONFIRM_HANDLER = handler
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


def _get_safe_commands() -> set:
    is_windows = platform.system() == "Windows"
    cmds = {
        "dir", "echo", "type", "cd", "pip", "python", "py", "git",
        "ls", "pwd", "cat", "head", "tail", "mkdir", "rmdir", "del",
        "powershell", "pwsh", "cmd", "node", "npm", "npx",
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

def _is_safe_command(command: str) -> bool:
    is_windows = platform.system() == "Windows"
    safe_commands = _get_safe_commands()
    try:
        stripped = command.strip()
        parts = shlex.split(stripped, posix=(not is_windows))
        if parts:
            base_cmd = os.path.basename(parts[0]).lower()
            if base_cmd.endswith(".exe"):
                base_cmd = base_cmd[:-4]

            # Block python -c one-liners that execute system processes or shell commands in safe mode
            if base_cmd in ("python", "py"):
                if "-c" in parts:
                    idx = parts.index("-c")
                    if idx + 1 < len(parts):
                        code_payload = parts[idx + 1].lower()
                        forbidden_patterns = [
                            "subprocess", "os.system", "os.popen", "pty", "popen",
                            "spawn", "execv", "execl"
                        ]
                        if any(p in code_payload for p in forbidden_patterns):
                            return False
                        # If running a script file under python in safe mode, verify via Python authorization if interactive
                        if AUTONOMY_LEVEL == "safe":
                            from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as _ep
                            if not getattr(_ep, "_AUTO_APPROVE_PYTHON", False) and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                                script_target = next((p for p in parts[1:] if p.endswith(".py") and not p.startswith("-")), None)
                                if script_target and os.path.exists(script_target):
                                    try:
                                        src = Path(script_target).read_text(encoding="utf-8", errors="ignore")
                                        dec, reason = _ep._prompt_user_validation(src, f"shell: {command}", parts)
                                        if dec == "reject":
                                            return False
                                        elif dec == "always":
                                            _ep._AUTO_APPROVE_PYTHON = True
                                    except Exception:
                                        pass

            if base_cmd in safe_commands:
                return True                
            for safe in safe_commands:
                sl = safe.lower()
                if (stripped.lower() == sl or 
                    stripped.lower().startswith(sl + " ") or 
                    stripped.lower().startswith(sl + '"') or 
                    stripped.lower().startswith(sl + '(')):
                    return True
            return False

        
    except ValueError:
        pass
    stripped_lower = command.strip().lower()
    for safe in safe_commands:
        sl = safe.lower()
        if (stripped_lower == sl or 
            stripped_lower.startswith(sl + " ") or 
            stripped_lower.startswith(sl + '"') or 
            stripped_lower.startswith(sl + '(')):
            return True
    return False

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
            f"AUTONOMY: SAFE MODE is active. Whitelisted commands run automatically:\n"
            f"Allowed safe commands: {safe_cmds_str}\n"
            "Commands outside this whitelist will prompt the user for authorization before execution."
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
            # On Windows cmd.exe, rmdir /s /q only accepts a single directory. Expand multiple targets.
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

            if not _is_safe_command(command) or autonomy_level == "strict":
                handler = _CONFIRM_HANDLER
                if not handler:
                    persistent_name = "lollms_client.tools_bindings.lcp.persistent_system_shell"
                    if persistent_name in sys.modules:
                        handler = getattr(sys.modules[persistent_name], "_CONFIRM_HANDLER", None)

                user_authorized = False
                if handler is not None and callable(handler):
                    try:
                        res = handler(command, f"shell: {command}")
                        decision = res[0] if isinstance(res, tuple) else (res if isinstance(res, str) else ("allow" if res else "reject"))
                        if str(decision).lower().strip() in ("allow", "always"):
                            user_authorized = True
                    except Exception as handler_err:
                        ASCIIColors.warning(f"[system_shell] Confirm handler failed: {handler_err}")

                if not user_authorized:
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
                            f"🛑 BLOCKED BY SANDBOX: The command '{command}' is not in the safe whitelist.\n\n"
                            f"The system shell is currently in '{autonomy_level}' mode on {platform.system()} ({'cmd.exe' if is_windows else 'sh/bash'}).\n"
                            f"Allowed safe commands include: {allowed_list}.{hint}\n\n"
                            f"⚠️ **ACTION REQUIRED FROM THE USER**: If this task requires elevated privileges, "
                            f"please ask the user to enable 'full_access' mode by typing `/shell` in the CLI or settings."
                        ),
                        "error": f"Blocked by sandbox ({autonomy_level} mode). The command '{command}' is not whitelisted.{hint}"
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