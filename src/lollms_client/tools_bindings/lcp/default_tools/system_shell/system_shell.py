import os
import sys
import subprocess
import platform
import shlex
from typing import Any, Dict
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "System Shell"
TOOL_LIBRARY_DESC = "Executes shell commands (bash, cmd, powershell) with adjustable autonomy levels for environment management and tooling."
TOOL_LIBRARY_ICON = "⚙️"

AUTONOMY_LEVEL: str = "safe"

def init_tools_library(config: dict = None) -> None:
    global AUTONOMY_LEVEL
    if config and isinstance(config, dict):
        autonomy = config.get("autonomy_level", "safe").lower()
        if autonomy in ("safe", "full_access"):
            AUTONOMY_LEVEL = autonomy
            ASCIIColors.info(f"[System Shell] Host configured autonomy level: {AUTONOMY_LEVEL}")
        else:
            ASCIIColors.warning(f"[System Shell] Invalid autonomy level '{autonomy}' received. Defaulting to 'safe'.")
            AUTONOMY_LEVEL = "safe"
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
    else:
        autonomy_desc = (
            f"AUTONOMY: SAFE MODE is active. Only whitelisted safe commands are permitted:\n"
            f"Allowed safe commands: {safe_cmds_str}\n"
            "Any command outside this list is blocked by the sandbox. "
            "If you need elevated privileges, ask the user to enable full_access mode."
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
            if not _is_safe_command(command):
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
                        f"The system shell is currently in 'safe' mode on {platform.system()} ({'cmd.exe' if is_windows else 'sh/bash'}).\n"
                        f"Allowed safe commands include: {allowed_list}.{hint}\n\n"
                        f"⚠️ **ACTION REQUIRED FROM THE USER**: If this task requires elevated privileges (e.g., system configuration, complex shell scripts), "
                        f"please ask the user to enable 'full_access' mode by typing `/shell` in the CLI, or by pressing `Ctrl+C` and restarting with the `--shell-autonomy full_access` flag."
                    ),
                    "error": f"Blocked by sandbox (safe mode). The command '{command}' is not whitelisted.{hint}"
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
            error_msg = result.stderr if result.stderr else f"Command failed with exit code {result.returncode}"

        return {
            "success": result.returncode == 0,
            "output": result.stdout or ("Command executed successfully (no stdout)." if result.returncode == 0 else ""),
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