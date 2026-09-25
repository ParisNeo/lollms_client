import os
import sys
import base64
import subprocess
import threading
import time
import json
import secrets
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from ascii_colors import trace_exception, ASCIIColors

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


class WhisperSTTBinding(LollmsSTTBinding):
    """
    Speech-To-Text binding for OpenAI Whisper using a resilient, multi-process
    shared model server architecture with event-driven dynamic micro-batching.
    Multiple lollms_client instances and worker processes attach to the same running daemon.
    """

    def __init__(self, **kwargs):
        super().__init__(binding_name="whisper")
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

        self.venv_dir = Path(kwargs.get("venv_path", "./venv/stt_whisper_venv")).resolve()
        self.cache_dir = Path(kwargs.get("cache_dir", "./data/stt_models/whisper")).resolve()

        self.venv_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.token_file = self.cache_dir / "whisper_server.token"
        self.service_key = kwargs.get("service_key")
        if not self.service_key and self.token_file.exists():
            try:
                self.service_key = self.token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.2, status_forcelist=[502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retries))

        if self.auto_start_server:
            self.ensure_server_is_running(self.wait_for_server)

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
            headers["X-Server-Token"] = self.service_key
        return headers

    def is_server_running(self) -> bool:
        """Probes the server on loopback with a fast timeout (<= 1.5s)."""
        try:
            resp = requests.get(
                f"{self.base_url}/status",
                headers=self._get_headers(),
                timeout=1.5
            )
            if resp.status_code == 200:
                data = resp.json() if callable(getattr(resp, "json", None)) else {}
                if isinstance(data, dict) and data.get("status") == "running":
                    return True
            elif resp.status_code == 401:
                if self.token_file.exists():
                    try:
                        self.service_key = self.token_file.read_text(encoding="utf-8").strip()
                        retry_resp = requests.get(
                            f"{self.base_url}/status",
                            headers=self._get_headers(),
                            timeout=1.5
                        )
                        return retry_resp.status_code == 200
                    except Exception:
                        pass
                return True
        except requests.exceptions.RequestException:
            return False
        return False

    def ensure_server_is_running(self, wait: bool = True, timeout_s: int = 120):
        """
        Ensures the shared Whisper server daemon is running without port collisions.
        Uses cross-process FileLock with double-checked probing to guarantee that
        only the first worker process spawns the daemon while all others attach to it.
        """
        if self.is_server_running():
            return

        lock_path = self.cache_dir / "whisper_server_spawn.lock"
        lock = FileLock(lock_path, timeout=timeout_s)

        try:
            with lock:
                if self.is_server_running():
                    ASCIIColors.green(f"Whisper shared daemon detected on {self.base_url}. Attached successfully.")
                    return

                ASCIIColors.info(f"Spawning shared Whisper server daemon on {self.base_url}...")
                self.start_server(wait=wait, timeout_s=timeout_s)
        except Timeout:
            if self.is_server_running():
                return
            raise RuntimeError(f"Timed out waiting for Whisper shared daemon to start after {timeout_s}s.")

    def install_server_dependencies(self):
        ASCIIColors.info(f"Setting up Whisper virtual environment in: {self.venv_dir}")
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir), create_if_not_exist=True)

        ASCIIColors.info("Installing Whisper server dependencies...")
        pm_v.ensure_packages(["requests", "uvicorn", "fastapi", "python-multipart", "filelock"])
        pm_v.ensure_packages(["ascii_colors>=0.11.10", "pipmaster", "tqdm", "numpy", "pillow", "pydantic"])

        torch_index_url = None
        if sys.platform == "win32":
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
                ASCIIColors.green("NVIDIA GPU detected. Installing CUDA-enabled PyTorch.")
                torch_index_url = "https://download.pytorch.org/whl/cu126"
            except (FileNotFoundError, subprocess.CalledProcessError):
                ASCIIColors.yellow("No GPU detected or nvidia-smi failed. Installing standard PyTorch.")

        pm_v.ensure_packages(["torch", "torchaudio"], index_url=torch_index_url)
        pm_v.ensure_packages(["openai-whisper"])
        ASCIIColors.green("Whisper server dependencies are satisfied.")

    def start_server(self, wait: bool = True, timeout_s: int = 120):
        server_script = self.server_dir / "main.py"
        venv_cfg = self.venv_dir / "pyvenv.cfg"

        if not venv_cfg.exists():
            self.install_server_dependencies()

        if sys.platform == "win32":
            python_executable = self.venv_dir / "Scripts" / "python.exe"
        else:
            python_executable = self.venv_dir / "bin" / "python"

        if not python_executable.exists():
            raise RuntimeError(f"Python executable not found in venv: {python_executable}.")

        if not self.service_key:
            if self.token_file.exists():
                try:
                    self.service_key = self.token_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            if not self.service_key:
                self.service_key = secrets.token_hex(16)
                try:
                    fd = os.open(str(self.token_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        f.write(self.service_key)
                except Exception:
                    self.token_file.write_text(self.service_key, encoding="utf-8")

        command = [
            str(python_executable),
            str(server_script),
            "--host", str(self.host),
            "--port", str(self.port),
            "--cache-dir", str(self.cache_dir.resolve()),
            "--token", str(self.service_key),
            "--batch-window", str(self.batch_window),
            "--max-batch-size", str(self.max_batch_size),
        ]

        log_file_path = self.cache_dir / "whisper_server.log"
        log_f = open(log_file_path, "w", encoding="utf-8")

        try:
            popen_kwargs: Dict[str, Any] = {
                "stdout": log_f,
                "stderr": subprocess.STDOUT,
            }
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                popen_kwargs["start_new_session"] = True

            self.server_process = subprocess.Popen(command, **popen_kwargs)
        finally:
            log_f.close()

        ASCIIColors.info(f"Whisper server process launched on http://{self.host}:{self.port} (PID: {self.server_process.pid})")

        if wait:
            start_time = time.time()
            while time.time() - start_time < timeout_s:
                if self.server_process.poll() is not None:
                    error_tail = "Log file is empty."
                    try:
                        if log_file_path.exists():
                            lines = log_file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                            error_tail = "\n".join(lines[-30:])
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Whisper server process terminated unexpectedly with code {self.server_process.returncode}.\n"
                        f"Log tail:\n{error_tail}"
                    )

                if self.is_server_running():
                    ASCIIColors.success(f"Whisper shared daemon is operational on {self.base_url}.")
                    return

                time.sleep(0.5)

            raise TimeoutError(f"Whisper server failed to become responsive within {timeout_s} seconds.")

    def _post_json_request(self, endpoint: str, data: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(
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
            response = requests.get(
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
            "filename": filename_hint
        }

        response = self._post_json_request("/transcribe", data=payload)
        return response.json().get("text", "")

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

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        return ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]

    def ps(self) -> List[dict]:
        try:
            return self._get_request("/ps").json()
        except Exception:
            return [{"error": "Could not connect to server to get process status."}]