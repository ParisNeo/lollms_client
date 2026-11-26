import os
import sys
import base64
import signal
import argparse
import subprocess
import threading
import time
import json
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from ascii_colors import trace_exception, ASCIIColors

# OpenAI-Whisper requires Python >= 3.9. We use this floor to decide whether
# the host interpreter is a valid fallback when the dedicated venv is broken.
_WHISPER_MIN_PYTHON = (3, 9)

try:
    import pipmaster as pm
except ImportError:
    print("FATAL: pipmaster is not installed. Please install it using: pip install pipmaster")
    sys.exit(1)

try:
    from filelock import FileLock, Timeout
except ImportError:
    print("FATAL: The 'filelock' library is required. Please install it by running: pip install filelock")
    sys.exit(1)

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from lollms_client.lollms_stt_binding import LollmsSTTBinding

BindingName = "WhisperSTTBinding"

_PID_FILE_NAME = "whisper_server.pid"
_GRACEFUL_SHUTDOWN_TIMEOUT_S = 10
_FORCE_KILL_GRACE_S = 5


# Module-level HTTP helpers that go through the global `requests` namespace
# (not a private Session instance) so that unit tests patching
# `lollms_client.stt_bindings.whisper.requests.post` / `.get` intercept correctly.

def _module_post(url: str, **kwargs) -> "requests.Response":
    return requests.post(url, **kwargs)


def _module_get(url: str, **kwargs) -> "requests.Response":
    return requests.get(url, **kwargs)


class WhisperSTTBinding(LollmsSTTBinding):
    """
    Speech-To-Text binding for OpenAI Whisper using a resilient, multi-process
    shared model server architecture with event-driven dynamic micro-batching.
    Multiple lollms_client instances and worker processes attach to the same running daemon.
    """

    def __init__(self, **kwargs):
        # Forward kwargs so the base binding can pick up ``lollms_data_folder``
        # (injected by LollmsClient) and expose it via ``get_data_folder()``.
        super().__init__(binding_name="whisper", **kwargs)
        self.config = kwargs
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = int(kwargs.get("port", 9633))
        self.auto_start_server = kwargs.get("auto_start_server", True)
        self.wait_for_server = kwargs.get("wait_for_server", True)
        self.batch_window = float(kwargs.get("batch_window", 0.02))
        self.max_batch_size = int(kwargs.get("max_batch_size", 8))
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"
        # Resolved interpreter used to run the daemon. Populated by
        # install_server_dependencies(); may point at the host interpreter
        # when the dedicated venv cannot be created.
        self._server_python: Optional[Path] = None

        # 📂 Whisper persists its state (venv, models, tokens, pid, logs) under
        # <lollms_data_folder>/stt/whisper. This lets users relocate storage to
        # a larger disk by configuring the client-wide data folder.
        data_root = self.resolve_data_path("stt", "whisper")

        if kwargs.get("venv_path"):
            self.venv_dir = self.resolve_system_path(kwargs["venv_path"])
        else:
            self.venv_dir = (self.get_system_dir() / "venv" / "stt_whisper_venv").resolve()

        # 🗄 cache_dir hosts server state artifacts: auth token, PID, spawn lock, log.
        if kwargs.get("cache_dir"):
            self.cache_dir = self.resolve_system_path(kwargs["cache_dir"])
        else:
            self.cache_dir = data_root

        # 🗂 models_dir hosts Whisper model weights (download_root for load_model).
        if kwargs.get("models_dir"):
            self.models_dir = self.resolve_system_path(kwargs["models_dir"])
        else:
            self.models_dir = (data_root / "models").resolve()

        self.venv_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)

        self.token_file = self.cache_dir / "whisper_server.token"
        self.pid_file = self.cache_dir / _PID_FILE_NAME
        self.service_key = kwargs.get("service_key")
        if not self.service_key and self.token_file.exists():
            try:
                self.service_key = self.token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        # 🐍 Runtime interpreter the daemon will be launched with. Resolved lazily
        # during install/start; if the venv cannot be provisioned we fall back
        # to the host interpreter (see install_server_dependencies).
        self._python_executable: Optional[Path] = None
        self._using_host_python: bool = False

        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.2, status_forcelist=[502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retries))

        # 🧵 Background-start plumbing. Provisioning (pip install of torch/whisper)
        # and daemon spawn are slow (minutes on first run); they MUST NOT block
        # ``__init__``. We kick off a one-shot worker thread and let callers that
        # actually need the server (transcribe_*) join it via
        # ``ensure_server_is_running(wait=True)``.
        self._start_lock = threading.Lock()
        self._start_thread: Optional[threading.Thread] = None
        self._start_error: Optional[BaseException] = None
        self._start_done = threading.Event()

        if self.auto_start_server:
            # Never block __init__: provisioning (pip install of torch/whisper can
            # take minutes) and daemon spawn run on a background thread. The first
            # transcribe_* call transparently waits for the daemon (or re-raises the
            # startup error) via ensure_server_is_running(wait=True).
            self._kick_background_start()

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
            headers["X-Server-Token"] = self.service_key
        return headers

    def is_server_running(self) -> bool:
        """Probes the server on loopback with a fast timeout (<= 0.5s)."""
        try:
            resp = _module_get(
                f"{self.base_url}/status",
                headers=self._get_headers(),
                timeout=0.5
            )
            if resp.status_code == 200:
                data = resp.json() if callable(getattr(resp, "json", None)) else {}
                return data.get("status") == "running" if isinstance(data, dict) else True
            elif resp.status_code == 401:
                return True
        except Exception:
            return False
        return False

    def _kick_background_start(self) -> None:
        """
        Spawns the one-shot background provisioning thread if not already running.

        The thread provisions dependencies (slow ``pip install``) and launches the
        daemon. It records any exception so ``ensure_server_is_running(wait=True)``
        can re-raise it to a caller that genuinely needs the server.
        """
        with self._start_lock:
            if self._start_thread is not None and self._start_thread.is_alive():
                return
            if self._start_done.is_set():
                # Already completed (success or failure) in a prior turn.
                return
            self._start_error = None
            self._start_done.clear()
            worker = threading.Thread(
                target=self._background_start_worker,
                name="whisper-daemon-start",
                daemon=True,
            )
            self._start_thread = worker
            worker.start()

    def _spawn_under_lock(self, wait: bool, timeout_s: int) -> None:
        """
        Does the actual provision+spawn under the cross-process FileLock.

        Must never be called from the background worker thread (it would try to
        acquire the same lock twice, or attempt to join itself). The worker uses
        this directly; ``ensure_server_is_running`` uses this after its own
        cooperative-join front matter.
        """
        lock_path = self.cache_dir / "whisper_server_spawn.lock"
        lock = FileLock(lock_path, timeout=timeout_s)
        try:
            with lock:
                if self.is_server_running():
                    ASCIIColors.green(
                        f"Whisper shared daemon detected on {self.base_url}. Attached successfully."
                    )
                    return
                ASCIIColors.info(f"Spawning shared Whisper server daemon on {self.base_url}...")
                self.start_server(wait=wait, timeout_s=timeout_s)
        except Timeout:
            if self.is_server_running():
                return
            raise RuntimeError(
                f"Timed out waiting for Whisper shared daemon to start after {timeout_s}s."
            )

    def _background_start_worker(self) -> None:
        """Body of the background starter: provision + launch, capturing errors."""
        try:
            # Call the raw spawn logic directly — do NOT route through
            # ensure_server_is_running, whose cooperative-join path would attempt
            # to join this very thread.
            self._spawn_under_lock(wait=True, timeout_s=120)
        except BaseException as exc:  # noqa: BLE001 - surfaced to joining callers
            self._start_error = exc
            ASCIIColors.error(
                f"Background Whisper daemon startup failed: {exc}. "
                "The next transcribe_* call will re-raise this error."
            )
        finally:
            self._start_done.set()

    def _join_background_start(self, timeout_s: int) -> None:
        """
        Blocks until the background starter finishes (or the thread is absent).

        Re-raises the captured startup error so callers see real failures instead
        of silently querying a server that never came up.
        """
        thread = self._start_thread
        if thread is None or not thread.is_alive():
            return
        thread.join(timeout=timeout_s + 5)
        if self._start_error is not None:
            raise self._start_error

    def ensure_server_is_running(self, wait: bool = True, timeout_s: int = 120):
        """
        Ensures the shared Whisper server daemon is running without port collisions.
        Uses cross-process FileLock with double-checked probing to guarantee that
        only the first worker process spawns the daemon while all others attach to it.

        When a background starter is in flight (kicked from ``__init__`` for the
        non-blocking auto-start path), this cooperates with it: it probes quickly,
        then joins the worker. The FileLock serialises the actual spawn so only one
        process launches the daemon.
        """
        if self.is_server_running():
            return

        # Ensure a background starter exists (it may have already completed or
        # never been kicked if auto_start_server was False). This keeps the
        # non-blocking provisioning path available even for late calls.
        if not self._start_done.is_set():
            self._kick_background_start()

        # Cooperative path: if a background worker is already provisioning,
        # join it rather than duplicating the spawn under the same process.
        if self._start_thread is not None and self._start_thread.is_alive():
            if not wait:
                return
            self._join_background_start(timeout_s)
            if self.is_server_running():
                return
            if self._start_thread.is_alive():
                # The worker is still running (e.g. a long pip install) but we
                # timed out waiting. Do NOT fall through to a second spawn; report
                # the situation clearly.
                raise RuntimeError(
                    f"Whisper daemon is still provisioning after {timeout_s}s "
                    "(dependency installation in progress). Try again shortly, or "
                    "raise the timeout."
                )
            # Worker finished but the server is still down → surface its error, if any.
            if self._start_error is not None:
                raise self._start_error
        # If the background worker already finished with an error, surface it now.
        if self._start_done.is_set() and self._start_error is not None and wait:
            raise self._start_error

        self._spawn_under_lock(wait=wait, timeout_s=timeout_s)

    @staticmethod
    def _venv_python_path(venv_dir: Path) -> Path:
        """Returns the platform-correct interpreter path inside a venv."""
        if sys.platform == "win32":
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"

    def _venv_is_usable(self) -> bool:
        """True when the dedicated venv exists and its interpreter is runnable."""
        python_executable = self._venv_python_path(self.venv_dir)
        if not python_executable.exists():
            return False
        try:
            probe = subprocess.run(
                [str(python_executable), "-c", "import sys; print(sys.version_info[:2])"],
                capture_output=True, text=True, timeout=30,
            )
            return probe.returncode == 0
        except Exception:
            return False

    def _host_python_is_usable(self) -> bool:
        """
        Checks whether the host interpreter can run the server in-process.

        The host interpreter is an acceptable fallback as long as its Python
        version is >= 3.9 (openai-whisper / modern torch requirement) and it is
        not the same file as the venv interpreter.
        """
        if sys.version_info < (3, 9):
            ASCIIColors.warning(
                f"Host interpreter Python {sys.version_info.major}.{sys.version_info.minor} "
                "is too old (< 3.9) to run the Whisper server."
            )
            return False
        return True

    def _force_create_venv(self) -> bool:
        """
        Force-creates a fresh venv at self.venv_dir using the host interpreter.

        Any broken/partial venv is removed first. Returns True on success.
        """
        ASCIIColors.info(f"Force-creating Whisper virtual environment at {self.venv_dir}...")
        try:
            if self.venv_dir.exists():
                import shutil
                shutil.rmtree(self.venv_dir, ignore_errors=True)
            self.venv_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_dir)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                ASCIIColors.error(
                    f"Failed to create venv via 'python -m venv': "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
                return False
        except Exception as e:
            ASCIIColors.error(f"Failed to create venv at {self.venv_dir}: {e}")
            return False
        return self._venv_is_usable()

    def _resolve_server_python(self) -> Path:
        """
        Resolves the interpreter that will run the Whisper server.

        Order of preference:
        1. An already-working dedicated venv.
        2. A freshly force-created dedicated venv.
        3. The host interpreter, if its Python version is acceptable.
        Raises RuntimeError if none is viable.
        """
        if self._venv_is_usable():
            return self._venv_python_path(self.venv_dir)

        if self._force_create_venv():
            return self._venv_python_path(self.venv_dir)

        if self._host_python_is_usable():
            ASCIIColors.warning(
                "Dedicated Whisper venv could not be created. "
                f"Falling back to the host interpreter: {sys.executable}"
            )
            return Path(sys.executable)

        raise RuntimeError(
            f"Unable to provision a Python interpreter to run the Whisper server. "
            f"The dedicated venv at '{self.venv_dir}' could not be created and the "
            f"host interpreter (Python {sys.version_info.major}.{sys.version_info.minor}) "
            "is not supported. Install Python >= 3.9 and retry."
        )

    @staticmethod
    def _probe_imports(python_exe: Path, module_names: List[str]) -> Dict[str, str]:
        """
        Imports the given modules in ``python_exe`` and reports the exact per-module
        failure reasons. Returns an empty dict if all modules import successfully.

        Args:
            python_exe: Interpreter to probe.
            module_names: Top-level module names to import.

        Returns:
            Dict mapping module name -> error string for each failed module.
        """
        probe = (
            "import importlib, sys, json, traceback\n"
            "mods = json.loads(sys.argv[1])\n"
            "failures = {}\n"
            "for m in mods:\n"
            "    try:\n"
            "        importlib.import_module(m)\n"
            "    except BaseException as exc:\n"
            "        failures[m] = f'{type(exc).__name__}: {exc}'\n"
            "print(json.dumps(failures))\n"
        )
        try:
            result = subprocess.run(
                [str(python_exe), "-c", probe, json.dumps(module_names)],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {m: f"probe failed: {exc}" for m in module_names}

        # Parse the last stdout line as JSON (imports may print warning noise).
        for line in reversed((result.stdout or "").strip().splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        tail = (result.stderr or result.stdout or "").strip()[-800:]
        return {m: f"probe subprocess rc={result.returncode}: {tail}" for m in module_names}

    def _pip_install(self, python_executable: Path, args: List[str]) -> None:
        """
        Runs ``python -m pip <args>`` and raises a rich error on non-zero exit.

        Args:
            python_executable: Interpreter whose pip runs.
            args: Arguments passed after ``-m pip``.
        """
        cmd = [str(python_executable), "-m", "pip", *args]
        ASCIIColors.info(f"[whisper-pip] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        combined = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        if result.returncode != 0:
            raise RuntimeError(
                f"pip command failed (rc={result.returncode}) for interpreter "
                f"'{python_executable}': {' '.join(args)}\n{combined[-2000:]}"
            )
        # Surface the last lines so users can see resolver warnings even on success.
        tail = combined[-600:] if combined else "(no pip output)"
        ASCIIColors.info(f"[whisper-pip] rc=0, tail:\n{tail}")

    @staticmethod
    def _uninstall_conflicting_whisper(python_executable: Path) -> None:
        """
        Uninstalls the abandoned PyPI package named ``whisper`` if present.

        ``whisper`` (PyPI) and ``openai-whisper`` both provide a top-level
        ``whisper`` module, but only ``openai-whisper`` exposes ``load_model``.
        Leaving the abandoned package installed makes ``pip install openai-whisper``
        a silent no-op and causes ``import whisper`` to resolve to the wrong module.
        """
        probe = (
            "import importlib.metadata as m, json\n"
            "out = {}\n"
            "for dist in ('whisper', 'openai-whisper'):\n"
            "    try:\n"
            "        out[dist] = m.version(dist)\n"
            "    except m.PackageNotFoundError:\n"
            "        out[dist] = None\n"
            "print(json.dumps(out))\n"
        )
        try:
            result = subprocess.run(
                [str(python_executable), "-c", probe],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                return
            versions = json.loads(result.stdout.strip() or "{}")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return

        if versions.get("whisper") and not versions.get("openai-whisper"):
            ASCIIColors.warning(
                "Detected conflicting abandoned 'whisper' package "
                f"(version {versions['whisper']}). Uninstalling it before installing openai-whisper."
            )
            subprocess.run(
                [str(python_executable), "-m", "pip", "uninstall", "-y", "whisper"],
                capture_output=True, text=True, timeout=300,
            )

    def _install_into_interpreter(self, python_executable: Path) -> None:
        """
        Installs only the *light* packages the daemon needs to bind its port fast.

        Heavy inference dependencies (torch/torchaudio/openai-whisper) are NOT
        installed here. The daemon's own background provisioning thread owns them,
        so the HTTP server becomes reachable almost instantly and heavy installs
        never block the client. ``pipmaster`` is included because the daemon uses it
        to self-bootstrap its dependency stack.
        """
        is_host = Path(python_executable).resolve() == Path(sys.executable).resolve()

        # Sanity check pip works in this interpreter.
        self._pip_install(python_executable, ["--version"])

        lite_packages = [
            "pipmaster", "filelock", "pydantic", "numpy",
            "fastapi", "uvicorn", "ascii_colors>=0.11.10",
        ]
        ASCIIColors.info(
            f"Installing Whisper daemon lite dependencies into {python_executable} "
            "(heavy torch/whisper install is deferred to the daemon)..."
        )
        self._pip_install(python_executable, ["install", "--upgrade", *lite_packages])

        if is_host:
            ASCIIColors.green(
                "Whisper daemon lite dependencies installed into the HOST environment "
                "(fallback mode). Heavy inference deps will be provisioned by the daemon."
            )
        else:
            ASCIIColors.green(
                "Whisper daemon lite dependencies are satisfied in the dedicated venv. "
                "Heavy inference deps will be provisioned by the daemon."
            )
    def install_server_dependencies(self):
        """
        Provisions the daemon interpreter and installs its runtime dependencies.

        Only LIGHT deps (fastapi/uvicorn/numpy/filelock) are installed here so the
        daemon can bind its port quickly. Heavy inference (torch/openai-whisper) is
        installed by the daemon's own background provisioning thread, keeping this
        client-facing step fast and non-blocking on first run.
        """
        ASCIIColors.info(f"Setting up Whisper virtual environment in: {self.venv_dir}")
        python_executable = self._resolve_server_python()
        self._server_python = python_executable
        self._install_into_interpreter(python_executable)

    def run_maintenance_async(self) -> bool:
        """
        Launches a detached maintenance subprocess that ensures the daemon's
        virtual environment and lite deps are up to date, without blocking the
        caller. Safe to call repeatedly; a lock file serializes concurrent
        maintenance runs across client processes.

        Returns:
            bool: True if a maintenance subprocess was spawned or one is already
            running; False if it could not be started.
        """
        maintenance_lock = self.cache_dir / "whisper_maintenance.lock"
        try:
            lock = FileLock(maintenance_lock, timeout=0)
            lock.acquire()
        except Timeout:
            # Another process is already doing maintenance.
            return True
        except Exception:
            return False

        try:
            python_executable = self._server_python or self._resolve_server_python()
        except Exception:
            try:
                lock.release()
            except Exception:
                pass
            return False

        # Maintenance script: only touches lite deps + venv sanity; heavy deps are
        # handled by the daemon provisioning thread. Non-blocking spawn.
        script = (
            "import sys\n"
            "try:\n"
            "    import pipmaster as pm\n"
            "    pm.ensure_packages([\n"
            "        'pipmaster','filelock','pydantic','numpy',\n"
            "        'fastapi','uvicorn','ascii_colors>=0.11.10'])\n"
            "    print('[whisper-maint] lite deps OK')\n"
            "except BaseException as e:\n"
            "    print(f'[whisper-maint] ERROR: {e}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )

        log_path = self.cache_dir / "whisper_maintenance.log"
        try:
            log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            subprocess.Popen(
                [str(python_executable), "-c", script],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                start_new_session=(sys.platform != "win32"),
            )
            log_handle.close()
            ASCIIColors.info(
                f"Whisper maintenance subprocess launched (log: {log_path})."
            )
            return True
        except Exception as exc:
            try:
                lock.release()
            except Exception:
                pass
            ASCIIColors.warning(f"Could not launch whisper maintenance subprocess: {exc}")
            return False
        # Note: the FileLock is intentionally released by the subprocess exiting —
        # but since we cannot easily tie the lock lifetime to the child, we release
        # it here after spawn. The lock is a co-editing guard, not a hard mutex.
        finally:
            try:
                lock.release()
            except Exception:
                pass

    @staticmethod
    def _missing_packages(python_exe: Path, modules: List[str]) -> List[str]:
        """Legacy helper: returns the subset of module names that fail to import."""
        failures = WhisperSTTBinding._probe_imports(python_exe, modules)
        return [m for m in modules if m in failures]

    def start_server(self, wait: bool = True, timeout_s: int = 120):
        server_script = self.server_dir / "main.py"
        if not server_script.exists():
            raise FileNotFoundError(f"Whisper server script not found at {server_script}")

        # Re-provision only when the interpreter is unknown or the venv it points
        # at has since disappeared. In host-fallback mode `_venv_is_usable()` is
        # always False, so we key the decision off `_server_python` alone.
        needs_provisioning = (
            self._server_python is None
            or not Path(self._server_python).exists()
        )
        if needs_provisioning:
            self.install_server_dependencies()

        python_executable = self._server_python
        if python_executable is None or not Path(python_executable).exists():
            raise RuntimeError(
                "Whisper server interpreter could not be resolved. "
                f"Expected venv interpreter at '{self._venv_python_path(self.venv_dir)}' "
                f"or a usable host interpreter. Check installed Python version (>= 3.9)."
            )

        log_path = self.cache_dir / "whisper_server.log"
        log_handle = None
        try:
            # Line-buffered append so logs land on disk immediately.
            log_handle = open(log_path, "a", encoding="utf-8", buffering=1)

            cmd = [
                str(python_executable),
                str(server_script),
                "--host", str(self.host),
                "--port", str(self.port),
                "--cache-dir", str(self.cache_dir),
                "--models-dir", str(self.models_dir),
                "--batch-window", str(self.batch_window),
                "--max-batch-size", str(self.max_batch_size),
                "--pid-file", str(self.pid_file),
            ]
            if self.service_key:
                cmd += ["--token", self.service_key]

            ASCIIColors.info(f"Launching Whisper server: {' '.join(cmd)}")

            creationflags = 0
            if sys.platform == "win32":
                # DETACHED_PROCESS so the daemon outlives this client; CREATE_NO_WINDOW
                # to suppress a console window when launched from a GUI context.
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

            self.server_process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                start_new_session=(sys.platform != "win32"),
            )
            self._write_pid_file(self.server_process.pid)
        except Exception:
            if log_handle is not None:
                log_handle.close()
            raise
        else:
            if log_handle is not None:
                log_handle.close()

        if not wait:
            return

        # Poll the health/status endpoint until the daemon answers or times out.
        deadline = time.time() + timeout_s
        last_error: Optional[str] = None
        while time.time() < deadline:
            if self.is_server_running():
                ASCIIColors.green(f"Whisper server is up on {self.base_url}.")
                return
            if self.server_process is not None and self.server_process.poll() is not None:
                # Process exited early → surface the tail of the log.
                last_error = self._read_log_tail(log_path)
                break
            time.sleep(0.5)

        # Timed out or exited. Capture diagnostics and clean up.
        tail = last_error or self._read_log_tail(log_path)
        self._terminate_spawned_process()
        raise RuntimeError(
            f"Whisper server failed to become responsive within {timeout_s}s at {self.base_url}.\n"
            f"Server log tail ({log_path}):\n{tail}"
        )

    def _terminate_spawned_process(self) -> None:
        """Best-effort cleanup of a Popen process this binding just spawned."""
        proc = self.server_process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=_FORCE_KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=_FORCE_KILL_GRACE_S)
        except Exception:
            pass
        finally:
            self.server_process = None

    @staticmethod
    def _read_log_tail(log_path: Path, max_chars: int = 4000) -> str:
        """Returns the last ``max_chars`` of the server log, or a diagnostic string."""
        try:
            if not log_path.exists():
                return "(log file not found)"
            data = log_path.read_bytes()
            if len(data) > max_chars:
                data = data[-max_chars:]
            return data.decode("utf-8", errors="replace")
        except Exception as exc:
            return f"(could not read log: {exc})"

    def _post_json_request(self, endpoint: str, data: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.post(
                url,
                json=data,
                headers=self._get_headers(),
                timeout=3600
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to communicate with Whisper server at {url}. Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_detail = e.response.json().get('detail', e.response.text)
                except Exception:
                    err_detail = e.response.text
                raise RuntimeError(f"Whisper server error: {err_detail}") from e
            raise RuntimeError(f"Communication with Whisper server failed: {e}") from e

    def _get_request(self, endpoint: str, params: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=15
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Communication with Whisper server failed: {e}") from e

    def transcribe_audio(self, audio_source: Union[str, Path, bytes], model: Optional[str] = None, **kwargs) -> str:
        self.ensure_server_is_running(wait=True)

        if isinstance(audio_source, (str, Path)):
            audio_file = Path(audio_source)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found at: {audio_source}")
            audio_bytes = audio_file.read_bytes()
            filename_hint = audio_file.name
        elif isinstance(audio_source, bytes):
            audio_bytes = audio_source
            filename_hint = kwargs.get("filename")
        else:
            raise ValueError("audio_source must be str, Path, or bytes")

        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        payload = {
            "audio_b64": audio_b64,
            "model_name": model or self.config.get("model_name", "base"),
            "language": kwargs.get("language"),
            "task": kwargs.get("task", "transcribe"),
            "fp16": kwargs.get("fp16"),
            "device": kwargs.get("device"),
            "filename": filename_hint,
        }

        response = self._post_json_request("/transcribe", data=payload)
        result = response.json()
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text

    def transcribe_audio_with_diarization(
        self,
        audio_source: Union[str, Path, bytes],
        participants: Optional[List[str]] = None,
        voice_samples: Optional[Dict[str, Union[str, Path, bytes]]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transcribes audio with speaker diarization and voice matching.

        Segments speech, extracts acoustic embeddings using Whisper's audio encoder,
        and clusters them into distinct speakers. If voice_samples are provided,
        clusters are matched to reference voices. Unmatched clusters are mapped
        to participants in order of appearance, or 'Speaker 1', 'Speaker 2', etc.

        Returns:
            List[Dict[str, Any]]: List of dialogue turns with 'speaker', 'start', 'end', and 'text'.
        """
        self.ensure_server_is_running(wait=True)

        if isinstance(audio_source, (str, Path)):
            audio_file = Path(audio_source)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found at: {audio_source}")
            audio_bytes = audio_file.read_bytes()
            filename_hint = audio_file.name
        elif isinstance(audio_source, bytes):
            audio_bytes = audio_source
            filename_hint = kwargs.get("filename")
        else:
            raise ValueError("audio_source must be str, Path, or bytes")

        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Encode reference voice samples to base64 if provided
        encoded_samples: Optional[Dict[str, str]] = None
        if voice_samples:
            encoded_samples = {}
            for spk_name, sample_src in voice_samples.items():
                if isinstance(sample_src, (str, Path)):
                    sp = Path(sample_src)
                    if sp.exists() and sp.is_file():
                        encoded_samples[spk_name] = base64.b64encode(sp.read_bytes()).decode('utf-8')
                elif isinstance(sample_src, bytes):
                    encoded_samples[spk_name] = base64.b64encode(sample_src).decode('utf-8')

        payload = {
            "audio_b64": audio_b64,
            "participants": participants,
            "voice_samples": encoded_samples,
            "model_name": model or self.config.get("model_name", "base"),
            "language": kwargs.get("language"),
            "task": kwargs.get("task", "transcribe"),
            "fp16": kwargs.get("fp16"),
            "device": kwargs.get("device"),
            "filename": filename_hint
        }

        response = self._post_json_request("/transcribe_diarize", data=payload)
        res_json = response.json()
        turns = res_json.get("turns", [])
        return turns

    def shutdown_server(self) -> bool:
        """Sends an authenticated shutdown command to terminate the background server daemon."""
        if not self.is_server_running():
            return True
        try:
            resp = self._session.post(
                f"{self.base_url}/shutdown",
                headers=self._get_headers(),
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _write_pid_file(self, pid: int) -> None:
        try:
            self.pid_file.write_text(str(pid), encoding="utf-8")
        except Exception as e:
            ASCIIColors.warning(f"Could not write PID file {self.pid_file}: {e}")

    def _read_pid_file(self) -> Optional[int]:
        try:
            if self.pid_file.exists():
                return int(self.pid_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass
        return None

    def _delete_pid_file(self) -> None:
        try:
            self.pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    def _is_our_server_process(self, pid: int) -> bool:
        """Verifies the PID belongs to our Whisper server before any kill attempt."""
        try:
            import psutil
            proc = psutil.Process(pid)
            cmdline = " ".join(proc.cmdline()).lower()
            return "whisper" in cmdline and "server" in cmdline and "main.py" in cmdline
        except ImportError:
            return True
        except Exception:
            return False

    def _terminate_pid(self, pid: int) -> bool:
        """Cross-platform process termination with escalation to force kill."""
        try:
            import psutil
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=_FORCE_KILL_GRACE_S)
                return True
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=_FORCE_KILL_GRACE_S)
                return True
        except ImportError:
            pass
        except Exception:
            return False

        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, check=True
                )
            else:
                os.kill(pid, signal.SIGKILL)
            return True
        except Exception:
            return False

    def kill(self, force: bool = False) -> dict:
        """
        Terminates the Whisper server daemon.

        Graceful mode sends the authenticated HTTP /shutdown command and waits
        for the port to be released. Force mode (or an unresponsive daemon)
        escalates to OS-level process termination using the persisted PID file.

        Returns:
            dict: {"status": bool, "message": str}
        """
        was_running = self.is_server_running()
        pid = self._read_pid_file()

        if was_running and not force:
            if self.shutdown_server():
                start_time = time.time()
                while time.time() - start_time < _GRACEFUL_SHUTDOWN_TIMEOUT_S:
                    if not self.is_server_running():
                        self._delete_pid_file()
                        ASCIIColors.success("Whisper server shut down gracefully.")
                        return {"status": True, "message": "Whisper server shut down gracefully."}
                    time.sleep(0.3)
                ASCIIColors.warning("Graceful shutdown timed out. Escalating to force kill.")
            else:
                ASCIIColors.warning("HTTP shutdown failed. Escalating to force kill.")

        if pid is not None and self._is_our_server_process(pid):
            if self._terminate_pid(pid):
                self._delete_pid_file()
                ASCIIColors.success(f"Whisper server process (PID {pid}) terminated.")
                return {"status": True, "message": f"Whisper server process (PID {pid}) terminated."}
            ASCIIColors.error(f"Failed to terminate Whisper server process (PID {pid}).")
            return {"status": False, "message": f"Failed to terminate Whisper server process (PID {pid})."}

        if was_running:
            msg = (
                "Server is running but no valid PID file was found. "
                "Kill it manually or restart with a fresh cache directory."
            )
            ASCIIColors.error(msg)
            return {"status": False, "message": msg}

        self._delete_pid_file()
        return {"status": True, "message": "Whisper server was not running."}

    def restart(self) -> dict:
        """
        Force-kills the daemon and starts a fresh instance with current code.

        Returns:
            dict: {"status": bool, "message": str}
        """
        kill_result = self.kill(force=True)
        if not kill_result.get("status", False):
            return kill_result

        try:
            self.ensure_server_is_running(wait=True)
            if self.is_server_running():
                return {"status": True, "message": "Whisper server restarted and operational."}
            return {"status": False, "message": "Whisper server failed to become responsive after restart."}
        except Exception as e:
            trace_exception(e)
            return {"status": False, "message": f"Whisper server restart failed: {e}"}

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        return ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]

    def ps(self) -> List[dict]:
        try:
            resp = _module_get(
                f"{self.base_url}/ps",
                headers=self._get_headers(),
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, list):
                return result
            return [result]
        except Exception:
            return [{"error": "Could not connect to server to get process status."}]

    def status(self) -> dict:
        """
        Returns the daemon status and connection information.

        Returns:
            dict: {"status": bool, "message": str, "data": dict}
        """
        if self.is_server_running():
            return {
                "status": True,
                "message": f"Whisper server is running on {self.base_url}.",
                "data": {"running": True, "base_url": self.base_url},
            }
        return {
            "status": False,
            "message": "Whisper server is not running.",
            "data": {"running": False, "base_url": self.base_url},
        }


def _build_cli_binding(args: argparse.Namespace) -> WhisperSTTBinding:
    return WhisperSTTBinding(
        host=args.host,
        port=args.port,
        auto_start_server=False,
        wait_for_server=False,
        **({"cache_dir": args.cache_dir} if args.cache_dir else {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lollms-whisper",
        description="Management CLI for the shared Whisper STT daemon.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=9633, help="Server port")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override the models cache directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show daemon status")
    subparsers.add_parser("ps", help="List loaded models and queue state")
    subparsers.add_parser("kill", help="Terminate the daemon (graceful, then force)")
    subparsers.add_parser("restart", help="Force-kill and relaunch the daemon")

    args = parser.parse_args()

    binding = _build_cli_binding(args)

    if args.command == "status":
        result = binding.status()
        print(json.dumps(result.get("data", result), indent=2))
        sys.exit(0 if result.get("status") else 1)
    elif args.command == "ps":
        print(json.dumps(binding.ps(), indent=2))
    elif args.command == "kill":
        result = binding.kill(force=False)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("status") else 1)
    elif args.command == "restart":
        result = binding.restart()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("status") else 1)


if __name__ == "__main__":
    main()