"""
execute_multi_file_service.py
=============================
An LCP tool designed to concurrently launch, monitor, and manage multi-file local services,
such as a Python FastAPI backend server and a companion HTML/JS front-end.
"""

import os
import sys
import subprocess
import socket
import time
import signal
import json
from pathlib import Path
from typing import Optional, Any, List

TOOL_LIBRARY_NAME = "MULTI_FILE_SERVICE_RUNNER"
TOOL_LIBRARY_DESC = "Concurrently launches and manages a background Python backend (FastAPI/Flask) and a frontend web UI, handling port allocation and cleanup."
TOOL_LIBRARY_ICON = "🚀"

# Central registry to track active background processes across executions
_ACTIVE_PROCESSES: List[subprocess.Popen] = []

def init_tools_library() -> None:
    """Ensure required packages are available."""
    import pipmaster as pm
    pm.ensure_packages({"uvicorn": ">=0.15.0", "fastapi": ">=0.68.0"})


def _find_free_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Finds an open TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except socket.error:
                continue
    raise IOError("Could not find an open TCP port in the specified range.")


def _terminate_active_services():
    """Cleanly terminates any previously spawned background processes to prevent port locks."""
    global _ACTIVE_PROCESSES
    terminated_count = 0
    for proc in list(_ACTIVE_PROCESSES):
        try:
            if proc.poll() is None:  # Still running
                # On Windows, use taskkill or CTRL_BREAK_EVENT; on Unix use SIGTERM
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                terminated_count += 1
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                pass
    _ACTIVE_PROCESSES.clear()
    return terminated_count


def tool_execute_multi_file_service(
    backend_file: str = "main.py",
    frontend_file: str = "index.html",
    backend_port: int = 8000,
    frontend_port: int = 8080,
    uvicorn_app_import: str = "main:app"
) -> dict:
    """
    Launches a Python web server (FastAPI/Uvicorn) in the background,
    and serves the frontend HTML page concurrently, returning access URLs and log files.

    Args:
        backend_file (str): Filename of the Python backend (default "main.py").
        frontend_file (str): Filename of the HTML frontend (default "index.html").
        backend_port (str): Preferred port for the backend (default 8000).
        frontend_port (str): Preferred port for the frontend (default 8080).
        uvicorn_app_import (str): Uvicorn app import string (default "main:app").
    """
    global _ACTIVE_PROCESSES

    # 1. Terminate any stale services first to free ports
    old_terminated = _terminate_active_services()

    # 🛑 TOOLS ARE AGNOSTIC: Rely on CWD set by the orchestrator.
    # The orchestrator guarantees CWD is the isolated discussion workspace.
    workspace_dir = Path(".")

    backend_path = workspace_dir / backend_file
    frontend_path = workspace_dir / frontend_file

    if not backend_path.exists():
        return {
            "success": False,
            "error": f"Backend entry file '{backend_file}' not found in current workspace directory."
        }

    # 2. Allocate Ports
    try:
        actual_backend_port = _find_free_port(backend_port)
        actual_frontend_port = _find_free_port(frontend_port)
    except Exception as port_err:
        return {
            "success": False,
            "error": f"Port allocation failed: {port_err}"
        }

    # 3. Launch Backend Subprocess (FastAPI via Uvicorn)
    backend_log_path = workspace_dir / "backend_server.log"
    backend_log_file = open(backend_log_path, "w", encoding="utf-8")

    # Command: uvicorn main:app --host 127.0.0.1 --port <port>
    cmd = [
        sys.executable, "-m", "uvicorn",
        uvicorn_app_import,
        "--host", "127.0.0.1",
        "--port", str(actual_backend_port),
        "--log-level", "info"
    ]

    try:
        # Start in a new process group so we can terminate it and all child processes cleanly
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        preexec_fn = None if sys.platform == "win32" else os.setsid

        proc_backend = subprocess.Popen(
            cmd,
            cwd=str(workspace_dir.resolve()),
            stdout=backend_log_file,
            stderr=backend_log_file,
            creationflags=creation_flags,
            preexec_fn=preexec_fn
        )
        _ACTIVE_PROCESSES.append(proc_backend)
    except Exception as start_err:
        backend_log_file.close()
        return {
            "success": False,
            "error": f"Failed to launch backend server: {start_err}"
        }

    # 4. Launch Frontend Web Server Subprocess (Simple HTTP Server)
    frontend_log_path = workspace_dir / "frontend_server.log"
    frontend_log_file = open(frontend_log_path, "w", encoding="utf-8")

    # Command: python -m http.server <port> --bind 127.0.0.1
    cmd_frontend = [
        sys.executable, "-m", "http.server",
        str(actual_frontend_port),
        "--bind", "127.0.0.1"
    ]

    try:
        proc_frontend = subprocess.Popen(
            cmd_frontend,
            cwd=str(workspace_dir.resolve()),
            stdout=frontend_log_file,
            stderr=frontend_log_file,
            creationflags=creation_flags,
            preexec_fn=preexec_fn
        )
        _ACTIVE_PROCESSES.append(proc_frontend)
    except Exception as start_front_err:
        frontend_log_file.close()
        _terminate_active_services()  # Rollback backend too
        return {
            "success": False,
            "error": f"Failed to launch frontend server: {start_front_err}"
        }

    # 5. Wait briefly and verify health
    time.sleep(2.5)

    backend_alive = proc_backend.poll() is None
    frontend_alive = proc_frontend.poll() is None

    backend_log_file.close()
    frontend_log_file.close()

    # Read latest logs for immediate feedback
    backend_logs = backend_log_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
    frontend_logs = frontend_log_path.read_text(encoding="utf-8", errors="ignore")[-1000:]

    if not backend_alive or not frontend_alive:
        _terminate_active_services()  # Rollback
        return {
            "success": False,
            "error": "One or more servers crashed immediately upon startup.",
            "backend_alive": backend_alive,
            "frontend_alive": frontend_alive,
            "backend_logs": backend_logs,
            "frontend_logs": frontend_logs
        }

    # Return successful configuration details to the agent and user
    backend_url = f"http://127.0.0.1:{actual_backend_port}"
    frontend_url = f"http://127.0.0.1:{actual_frontend_port}/{frontend_file}"

    # Generate a lightweight LCP custom prompt injection so that the LLM is notified of the URLs
    prompt_injection = (
        f"\n\n=== 🚀 LOCAL MULTI-FILE SERVICE ACTIVE ===\n"
        f"• Backend API Service  : [{backend_url}]({backend_url}) (FastAPI/Uvicorn)\n"
        f"• Frontend Web Interface: [{frontend_url}]({frontend_url}) (HTML/JS)\n"
        f"• Log Files Generated  : `{backend_log_path.name}`, `{frontend_log_path.name}`\n"
        f"• Cleared Stale Processes: {old_terminated} process(es) terminated.\n\n"
        f"You can inform the user that the service is running successfully and give them the clickable links to view and interact with the UI live!"
    )

    return {
        "success": True,
        "message": "Concurrently launched backend and frontend servers successfully.",
        "backend_url": backend_url,
        "frontend_url": frontend_url,
        "backend_pid": proc_backend.pid,
        "frontend_pid": proc_frontend.pid,
        "cleared_previous_services": old_terminated,
        "prompt_injection": prompt_injection,
        "backend_logs_preview": backend_logs,
        "frontend_logs_preview": frontend_logs
    }


def tool_terminate_active_services() -> dict:
    """
    Terminates any currently running background servers managed by this tool.
    Useful for manual cleanup or reset.
    """
    terminated = _terminate_active_services()
    return {
        "success": True,
        "message": f"Successfully terminated {terminated} active background service(s)."
    }
