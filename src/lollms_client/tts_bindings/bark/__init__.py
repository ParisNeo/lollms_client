# File: lollms_client/tts_bindings/bark/__init__.py
from __future__ import annotations

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
import pipmaster as pm
from ascii_colors import ASCIIColors

from lollms_client.lollms_tts_binding import LollmsTTSBinding

BindingName = "BarkClientBinding"


class BarkClientBinding(LollmsTTSBinding):
    """
    Client binding for the shared Suno Bark generative TTS daemon server.
    Implements self-spawning shared singleton architecture with multi-process mutualization,
    cross-process FileLock, and persistent loopback daemon attachment.
    """

    def __init__(self, **kwargs):
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name="bark", **kwargs)

        self.host = kwargs.get("host", "127.0.0.1")
        self.port = int(kwargs.get("port", 9636))
        self.auto_start_server = kwargs.get("auto_start_server", True)
        self.wait_for_server = kwargs.get("wait_for_server", True)
        self.batch_window = float(kwargs.get("batch_window", 0.02))
        self.max_batch_size = int(kwargs.get("max_batch_size", 4))
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"

        self.cache_dir = Path(kwargs.get("cache_dir", "./data/tts_models/bark")).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_file = self.cache_dir / "bark_server.token"

        self.service_key = kwargs.get("service_key")
        if not self.service_key and self.token_file.exists():
            try:
                self.service_key = self.token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.2, status_forcelist=[502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retries))

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

        lock_path = self.cache_dir / "bark_server_spawn.lock"
        lock = FileLock(lock_path, timeout=timeout_s)

        try:
            with lock:
                if self.is_server_running():
                    ASCIIColors.green(f"Bark shared daemon detected on {self.base_url}. Attached successfully.")
                    return
                ASCIIColors.info(f"Spawning shared Bark TTS server daemon on {self.base_url}...")
                self.start_server(wait=wait, timeout_s=timeout_s)
        except Timeout:
            if self.is_server_running():
                return
            raise RuntimeError(f"Timed out waiting for Bark shared daemon on {self.base_url}.")

    def start_server(self, wait: bool = True, timeout_s: int = 120):
        server_dir = self.binding_root / "server"
        requirements_file = server_dir / "requirements.txt"
        server_script = server_dir / "main.py"

        venv_path = server_dir / "venv"
        pm_v = pm.PackageManager(venv_path=venv_path)
        pm_v.ensure_requirements(str(requirements_file), verbose=True)

        if sys.platform == "win32":
            python_executable = venv_path / "Scripts" / "python.exe"
        else:
            python_executable = venv_path / "bin" / "python"

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
            "--cache-dir", str(self.cache_dir),
            "--token", str(self.service_key),
            "--batch-window", str(self.batch_window),
            "--max-batch-size", str(self.max_batch_size),
        ]

        log_file_path = self.cache_dir / "bark_server.log"
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
                    ASCIIColors.success("Bark server is operational.")
                    return
                time.sleep(0.5)
            raise TimeoutError(f"Bark server failed to respond within {timeout_s}s.")

    def __del__(self):
        # Shared singleton daemon remains alive across worker process garbage collection
        pass

    def shutdown_server(self) -> bool:
        if not self.is_server_running():
            return True
        try:
            resp = self._session.post(f"{self.base_url}/shutdown", headers=self._get_headers(), timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate_audio(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        self.ensure_server_is_running(True)
        payload = {"text": text, "voice": voice, **kwargs}
        response = self._session.post(f"{self.base_url}/generate_audio", json=payload, headers=self._get_headers(), timeout=300)
        response.raise_for_status()
        return response.content

    def list_voices(self, **kwargs) -> List[str]:
        self.ensure_server_is_running(True)
        response = self._session.get(f"{self.base_url}/list_voices", headers=self._get_headers(), timeout=15)
        response.raise_for_status()
        return response.json().get("voices", [])

    def list_models(self, **kwargs) -> List[str]:
        return ["bark"]

    def set_voice(self, voice: str):
        self.ensure_server_is_running(True)
        response = self._session.post(f"{self.base_url}/set_voice", json={"voice": voice}, headers=self._get_headers(), timeout=15)
        response.raise_for_status()
        return response.json()