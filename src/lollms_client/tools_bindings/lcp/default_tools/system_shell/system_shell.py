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

def _is_safe_command(command: str) -> bool:
    is_windows = platform.system() == "Windows"
    safe_commands = {
        "dir", "echo", "type", "cd", "pip", "python", "py", "git",
        "ls", "pwd", "cat", "head", "tail", "mkdir", "rmdir", "del",
        "powershell", "pwsh", "cmd", "node", "npm", "npx",
        "where", "which", "set", "env"
    }
    if is_windows:
        safe_commands.update({
            "copy", "move", "ren", "rename", "md", "rd", "cls",
            "chdir", "pushd", "popd", "tree", "find", "findstr",
            "sort", "more", "help", "ver", "vol", "label", "time", "date"
        })
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
                if stripped.lower() == sl or stripped.lower().startswith(sl + " ") or stripped.lower().startswith(sl + '"'):
                    return True
            return False
    except ValueError:
        pass
    stripped_lower = command.strip().lower()
    for safe in safe_commands:
        sl = safe.lower()
        if stripped_lower == sl or stripped_lower.startswith(sl + " ") or stripped_lower.startswith(sl + '"'):
            return True
    return False

def tool_execute_shell_command_prompt() -> str:
    """
    Dynamically generates the description for tool_execute_shell_command.
    """
    is_windows = platform.system() == "Windows"
    os_name = "Windows (cmd/powershell)" if is_windows else "Linux/Unix (bash/sh)"
    
    if AUTONOMY_LEVEL == "full_access":
        return f"""Executes a shell command in the current workspace directory with FULL ACCESS.
Use this for environment management (e.g., pip install), running tests, or interacting with the OS.
You are operating in 'full_access' mode, meaning you can run destructive or system-level commands.
Operating System: {os_name}."""
    else:
        return f"""Executes a shell command in the current workspace directory in SAFE MODE.
Use this for environment management (e.g., pip install), running tests, or interacting with the OS.
You are operating in 'safe' mode. Only read-only or non-destructive commands are permitted.
If you need to execute a command outside this list, ask the user to enable 'full_access' mode.
Operating System: {os_name}."""

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
                allowed_list = ", ".join(sorted([
                    "dir", "echo", "type", "cd", "pip", "python", "py", "git",
                    "ls", "pwd", "cat", "head", "tail", "mkdir", "rmdir", "del",
                    "powershell", "pwsh", "cmd", "node", "npm", "npx",
                    "where", "which", "set", "env", "copy", "move", "ren", "rename",
                    "md", "rd", "cls", "chdir", "pushd", "popd", "tree", "find",
                    "findstr", "sort", "more", "help", "ver", "vol", "label",
                    "time", "date"
                ]))
                return {
                    "success": False,
                    "output": (
                        f"🛑 BLOCKED BY SANDBOX: The command '{command}' is not in the safe whitelist.\n\n"
                        f"The system shell is currently in 'safe' mode and only permits read-only or non-destructive operations.\n"
                        f"Allowed safe commands include: {allowed_list}.\n\n"
                        f"⚠️ **ACTION REQUIRED FROM THE USER**: If this task requires elevated privileges (e.g., system configuration, complex shell scripts), "
                        f"please ask the user to enable 'full_access' mode by typing `/shell` in the CLI, or by pressing `Ctrl+C` and restarting with the `--shell-autonomy full_access` flag."
                    ),
                    "error": f"Blocked by sandbox (safe mode). The command '{command}' is not whitelisted. Use the git_manager toolset for git operations."
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