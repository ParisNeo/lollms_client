import os
import sys
import time
import secrets
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from filelock import FileLock, Timeout
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from ascii_colors import ASCIIColors

from lollms_client.lollms_tts_binding import LollmsTTSBinding

BindingName = "XTTSClientBinding"


class XTTSClientBinding(LollmsTTSBinding):
    """
    Client binding for the shared, process-safe XTTS v2 server daemon.
    Guarantees a single model footprint in VRAM across all processes and workers.
    """

    def __init__(self, **kwargs):
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name="xtts", **kwargs)

        self.config = kwargs
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = int(kwargs.get("port", 9634))
        self.auto_start_server = kwargs.get("auto_start_server", True)
        self.wait_for_server = kwargs.get("wait_for_server", True)
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"
        self.venv_dir = Path(kwargs.get("venv_path", "./venv/tts_xtts_venv")).resolve()
        self.cache_dir = Path(kwargs.get("cache_dir", "./data/tts_models/xtts")).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_file = self.cache_dir / "xtts_server.token"

        self.service_key = kwargs.get("service_key")
        if not self.service_key and self.token_file.exists():
            try:
                self.service_key = self.token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.2, status_forcelist=[502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retries))

        self.target_python_version = "3.10"
        self.server_process = None

        if self.auto_start_server:
            self.ensure_server_is_running(self.wait_for_server)

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
            headers["X-Server-Token"] = self.service_key
        return headers

    def is_server_running(self) -> bool:
        try:
            resp = self._session.get(
                f"{self.base_url}/status",
                headers=self._get_headers(),
                timeout=1.5
            )
            if resp.status_code == 200:
                return resp.json().get("status") == "running"
            elif resp.status_code == 401:
                if self.token_file.exists():
                    try:
                        self.service_key = self.token_file.read_text(encoding="utf-8").strip()
                        retry = self._session.get(f"{self.base_url}/status", headers=self._get_headers(), timeout=1.5)
                        return retry.status_code == 200
                    except Exception:
                        pass
                return True
        except requests.exceptions.RequestException:
            return False
        return False

    def ensure_server_is_running(self, wait: bool = True, timeout_s: int = 120):
        if self.is_server_running():
            return

        lock_path = self.cache_dir / "xtts_server_spawn.lock"
        lock = FileLock(lock_path, timeout=timeout_s)

        try:
            with lock:
                if self.is_server_running():
                    ASCIIColors.green(f"XTTS shared daemon detected on {self.base_url}. Attached successfully.")
                    return
                ASCIIColors.info(f"Spawning shared XTTS server daemon on {self.base_url}...")
                self.start_server(wait=wait, timeout_s=timeout_s)
        except Timeout:
            if self.is_server_running():
                return
            raise RuntimeError(f"Timed out waiting for XTTS shared daemon on {self.base_url}.")

    def install_server_dependencies(self):
        ASCIIColors.info(f"Setting up Python {self.target_python_version} virtual environment in: {self.venv_dir}")
        import pipmaster as pm

        pm_instance = pm.get_pip_manager_for_version(
            self.target_python_version,
            str(self.venv_dir)
        )
        requirements_file = self.server_dir / "requirements.txt"
        success = pm_instance.ensure_requirements(str(requirements_file), verbose=True)
        if not success:
            raise RuntimeError("XTTS server dependency installation failed.")
        self._python_executable = pm_instance.target_python_executable

    def start_server(self, wait: bool = True, timeout_s: int = 120):
        server_script = self.server_dir / "main.py"
        if not self.venv_dir.exists():
            self.install_server_dependencies()
        else:
            if sys.platform == "win32":
                self._python_executable = str(self.venv_dir / "Scripts" / "python.exe")
            else:
                self._python_executable = str(self.venv_dir / "bin" / "python")

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

        voices_dir = self.server_dir / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)

        command = [
            str(self._python_executable),
            str(server_script),
            "--host", str(self.host),
            "--port", str(self.port),
            "--voices-dir", str(voices_dir),
            "--token", str(self.service_key),
        ]

        log_file_path = self.cache_dir / "xtts_server.log"
        log_f = open(log_file_path, "w", encoding="utf-8")
        try:
            popen_kwargs: Dict[str, Any] = {"stdout": log_f, "stderr": subprocess.STDOUT}
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                popen_kwargs["start_new_session"] = True
            self.server_process = subprocess.Popen(command, **popen_kwargs)
        finally:
            log_f.close()

        if wait:
            start_time = time.time()
            while time.time() - start_time < timeout_s:
                if self.is_server_running():
                    ASCIIColors.green("XTTS server is ready.")
                    return
                time.sleep(1)
            raise TimeoutError(f"XTTS server failed to start within {timeout_s}s.")

    def __del__(self):
        # Do not kill the server on object destruction as it is a shared daemon
        pass

    def shutdown_server(self) -> bool:
        if not self.is_server_running():
            return True
        try:
            resp = self._session.post(f"{self.base_url}/shutdown", headers=self._get_headers(), timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate_audio(self, text: str, voice: Optional[str] = None, language: str = "en", **kwargs) -> bytes:
        self.ensure_server_is_running(True)
        payload = {"text": text, "voice": voice, "language": language}
        payload.update(kwargs)
        response = self._session.post(f"{self.base_url}/generate_audio", json=payload, headers=self._get_headers(), timeout=300)
        response.raise_for_status()
        return response.content

    def list_voices(self, **kwargs) -> List[str]:
        self.ensure_server_is_running(True)
        response = self._session.get(f"{self.base_url}/list_voices", headers=self._get_headers(), timeout=15)
        response.raise_for_status()
        return response.json().get("voices", [])

    def list_models(self, **kwargs) -> list:
        return ["tts_models/multilingual/multi-dataset/xtts_v2"]

    def upload_voice(self, voice_path: str, voice_name: Optional[str] = None) -> dict:
        self.ensure_server_is_running(True)
        voice_file = Path(voice_path)
        if not voice_file.exists():
            return {"success": False, "voice_name": None, "message": f"Voice file not found: {voice_path}"}

        with open(voice_file, "rb") as f:
            files = {"voice_file": (voice_file.name, f, f"audio/{voice_file.suffix.lstrip('.')}")}
            data = {"voice_name": voice_name} if voice_name else {}
            response = self._session.post(
                f"{self.base_url}/upload_voice",
                files=files,
                data=data,
                headers=self._get_headers(),
                timeout=60
            )
            response.raise_for_status()
            return response.json()