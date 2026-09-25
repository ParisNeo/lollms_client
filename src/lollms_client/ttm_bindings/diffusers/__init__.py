# lollms_client/ttm_bindings/diffusers/__init__.py
from __future__ import annotations

import os
import sys
import base64
import time
import secrets
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

import pipmaster as pm
from filelock import FileLock, Timeout
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_ttm_binding import LollmsTTMBinding

BindingName = "DiffusersTTMBinding"

DEFAULT_TTM_ZOO = [
    {
        "name": "MiniMax Music 3",
        "description": "High-performance full song generation (up to 5 mins) from lyrics and structured descriptions.",
        "size": "25GB",
        "type": "model",
        "link": "MiniMaxAI/MiniMax-Music3",
    },
    {
        "name": "Stable Audio Open 1.0",
        "description": "Stability AI's rectified-flow DiT model for music and sound effects (up to 47s, 44.1 kHz stereo).",
        "size": "4.8GB",
        "type": "model",
        "link": "stabilityai/stable-audio-open-1.0",
    },
    {
        "name": "AudioLDM 2 Music",
        "description": "Text-conditioned latent diffusion model specialized in music generation.",
        "size": "3.2GB",
        "type": "model",
        "link": "cvssp/audioldm2-music",
    },
    {
        "name": "AudioLDM 2 Large",
        "description": "General text-to-audio and music latent diffusion model.",
        "size": "3.5GB",
        "type": "model",
        "link": "cvssp/audioldm2-large",
    },
    {
        "name": "MusicGen Small",
        "description": "Lightweight 300M autoregressive music generation model from Meta AudioCraft.",
        "size": "1.2GB",
        "type": "model",
        "link": "facebook/musicgen-small",
    },
]


class DiffusersTTMBinding(LollmsTTMBinding):
    """
    Text-to-Music (TTM) binding using Hugging Face Diffusers and ModularPipeline.
    Features self-spawning shared singleton daemon architecture on dedicated port 9637,
    cross-process FileLock synchronization, and continuous queue execution.
    """

    def __init__(self, **kwargs):
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name="diffusers", **kwargs)

        self.config = kwargs
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = int(kwargs.get("port", 9637))
        self.model_name = kwargs.get("model_name", "MiniMaxAI/MiniMax-Music3")
        self.auto_start_server = kwargs.get("auto_start_server", True)
        self.wait_for_server = kwargs.get("wait_for_server", True)
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"

        self.venv_dir = Path(kwargs.get("venv_path", "./venv/ttm_diffusers_venv")).resolve()
        self.cache_dir = Path(kwargs.get("cache_dir", "./data/ttm_models/diffusers")).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_file = self.cache_dir / "diffusers_ttm.token"

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
                data = resp.json()
                return data.get("status") == "running"
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

        lock_path = self.cache_dir / "diffusers_ttm_spawn.lock"
        lock = FileLock(lock_path, timeout=timeout_s)

        try:
            with lock:
                if self.is_server_running():
                    ASCIIColors.green(f"Diffusers TTM shared daemon detected on {self.base_url}. Attached successfully.")
                    return
                ASCIIColors.info(f"Spawning shared Diffusers TTM server daemon on {self.base_url}...")
                self.start_server(wait=wait, timeout_s=timeout_s)
        except Timeout:
            if self.is_server_running():
                return
            raise RuntimeError(f"Timed out waiting for Diffusers TTM shared daemon on {self.base_url}.")

    def install_server_dependencies(self):
        ASCIIColors.info(f"Setting up Diffusers TTM virtual environment in: {self.venv_dir}")
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir), create_if_not_exist=True)

        pm_v.ensure_packages(["requests", "uvicorn", "fastapi", "python-multipart", "filelock"])
        pm_v.ensure_packages(["ascii_colors>=0.11.10", "pipmaster", "tqdm", "numpy", "soundfile", "scipy"])

        torch_index_url = None
        if sys.platform == "win32":
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
                ASCIIColors.green("NVIDIA GPU detected. Installing CUDA-enabled PyTorch.")
                torch_index_url = "https://download.pytorch.org/whl/cu126"
            except (FileNotFoundError, subprocess.CalledProcessError):
                ASCIIColors.yellow("No GPU detected or nvidia-smi failed. Installing standard PyTorch.")

        pm_v.ensure_packages(["torch", "torchaudio"], index_url=torch_index_url)
        pm_v.ensure_packages(["transformers", "accelerate", "diffusers"])
        ASCIIColors.green("Diffusers TTM server dependencies are satisfied.")

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
            "--cache-dir", str(self.cache_dir),
            "--token", str(self.service_key),
            "--model-name", str(self.model_name),
        ]

        log_file_path = self.cache_dir / "diffusers_ttm_server.log"
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
                    ASCIIColors.success("Diffusers TTM server is operational.")
                    return
                time.sleep(1)
            raise TimeoutError(f"Diffusers TTM server failed to respond within {timeout_s}s.")

    def __del__(self):
        # Do not kill the server on client destruction as it is a shared singleton daemon
        pass

    def shutdown_server(self) -> bool:
        if not self.is_server_running():
            return True
        try:
            resp = self._session.post(f"{self.base_url}/shutdown", headers=self._get_headers(), timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate_music(self, prompt: str, duration: int = 15, **kwargs) -> bytes:
        self.ensure_server_is_running(True)
        payload = {
            "prompt": prompt,
            "duration": duration,
            "model_name": kwargs.get("model_name", self.model_name),
            "num_inference_steps": kwargs.get("num_inference_steps", 30),
            "guidance_scale": kwargs.get("guidance_scale", 7.0),
            "seed": kwargs.get("seed", -1),
        }
        response = self._session.post(
            f"{self.base_url}/generate_music",
            json=payload,
            headers=self._get_headers(),
            timeout=600
        )
        response.raise_for_status()
        return response.content

    def generate_song(self, prompt: str, lyrics: Optional[str] = None, **kwargs) -> bytes:
        self.ensure_server_is_running(True)
        payload = {
            "prompt": prompt,
            "lyrics": lyrics or "",
            "model_name": kwargs.get("model_name", self.model_name),
            "duration": kwargs.get("duration", 60),
            "num_inference_steps": kwargs.get("num_inference_steps", 30),
            "seed": kwargs.get("seed", -1),
        }
        response = self._session.post(
            f"{self.base_url}/generate_song",
            json=payload,
            headers=self._get_headers(),
            timeout=1200
        )
        response.raise_for_status()
        return response.content

    def generate_song_from_lyrics(self, prompt: str, lyrics: str, **kwargs) -> bytes:
        return self.generate_song(prompt=prompt, lyrics=lyrics, **kwargs)

    def list_models(self, **kwargs) -> List[str]:
        self.ensure_server_is_running(True)
        try:
            resp = self._session.get(f"{self.base_url}/list_models", headers=self._get_headers(), timeout=15)
            resp.raise_for_status()
            return resp.json().get("models", [])
        except Exception:
            return [m["link"] for m in DEFAULT_TTM_ZOO]

    def get_zoo(self) -> List[Dict[str, Any]]:
        return list(DEFAULT_TTM_ZOO)

    def download_from_zoo(self, index: int, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
        zoo = self.get_zoo()
        if not (0 <= index < len(zoo)):
            return {"status": False, "message": "Index out of bounds."}
        item = zoo[index]
        return self.pull_model(item["link"], progress_callback=progress_callback)

    def pull_model(self, model_name: str, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
        self.ensure_server_is_running(True)
        try:
            if progress_callback:
                progress_callback({"status": "starting", "message": f"Downloading {model_name}..."})
            resp = self._session.post(
                f"{self.base_url}/pull_model",
                json={"model_name": model_name},
                headers=self._get_headers(),
                timeout=3600
            )
            resp.raise_for_status()
            if progress_callback:
                progress_callback({"status": "success", "message": f"Downloaded {model_name}"})
            return resp.json()
        except Exception as e:
            return {"status": False, "message": str(e)}

    def ps(self) -> List[dict]:
        try:
            resp = self._session.get(f"{self.base_url}/ps", headers=self._get_headers(), timeout=10)
            return resp.json()
        except Exception:
            return [{"error": "Could not connect to Diffusers TTM server."}]

    def get_server_logs(self) -> str:
        log_file = self.cache_dir / "diffusers_ttm_server.log"
        if not log_file.exists():
            return "No server log file found."
        try:
            lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            return "\n".join(lines[-100:])
        except Exception as e:
            return f"Failed to read logs: {e}"