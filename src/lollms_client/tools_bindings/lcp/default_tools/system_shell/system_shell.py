import os
import sys
import subprocess
import platform
from typing import Any, Dict
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "System Shell"
TOOL_LIBRARY_DESC = "Executes shell commands (bash, cmd, powershell) with adjustable autonomy levels for environment management and tooling."
TOOL_LIBRARY_ICON = "⚙️"

# ── HOST-CONFIGURABLE MODULE VARIABLES ──
# These are invisible to the LLM and strictly controlled by the host application.
AUTONOMY_LEVEL: str = "safe"

def init_tools_library(config: dict = None) -> None:
    """
    Initializes the tool library with host-provided configurations.
    This function is called by the LCP Binding during lazy initialization.
    """
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

    try:
        if is_windows:
            if autonomy_level == "full_access":
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=os.getcwd(),
                    timeout=120
                )
            else:
                safe_commands = ["dir", "echo", "type", "cd", "pip", "python", "git", "ls", "pwd", "cat", "head", "tail"]
                if not any(command.lower().strip().startswith(cmd) for cmd in safe_commands):
                    return {
                        "success": False,
                        "error": f"Command '{command}' requires 'full_access' autonomy level. Ask the user to enable it."
                    }
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=os.getcwd(),
                    timeout=60
                )
        else:
            if autonomy_level == "full_access":
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=os.getcwd(),
                    timeout=120
                )
            else:
                safe_commands = ["ls", "cat", "echo", "cd", "pip", "python", "git", "pwd", "head", "tail"]
                if not any(command.lower().strip().startswith(cmd) for cmd in safe_commands):
                    return {
                        "success": False,
                        "error": f"Command '{command}' requires 'full_access' autonomy level. Ask the user to enable it."
                    }
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=os.getcwd(),
                    timeout=60
                )

        return {
            "success": result.returncode == 0,
            "output": result.stdout or "Command executed successfully (no stdout).",
            "stderr": result.stderr,
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