# lollms_client/ttv_bindings/diffusers/__init__.py
"""
Text-to-Video (TTV) binding using Hugging Face Diffusers.

Runs a shared singleton FastAPI daemon on a dedicated port (default 9634) that
loads video pipelines (LTX-Video, CogVideoX, AnimateDiff, or any generic
DiffusionPipeline) and serializes generation through a single worker thread.
Multiple client processes attach to the same daemon via FileLock, sharing one
GPU-resident model.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pipmaster as pm
import requests
from ascii_colors import ASCIIColors, trace_exception
from filelock import FileLock, Timeout
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from lollms_client.lollms_ttv_binding import LollmsTTVBinding

pm.ensure_packages(["requests", "filelock"])

BindingName = "DiffusersTTVBinding"

DEFAULT_TTV_ZOO = [
    {
        "name": "LTX-Video",
        "description": "Lightricks LTX-Video – fast, high-quality text-to-video.",
        "size": "~9GB",
        "type": "model",
        "link": "Lightricks/LTX-Video",
    },
    {
        "name": "CogVideoX-2b",
        "description": "THUDM CogVideoX 2B – efficient text-to-video.",
        "size": "~10GB",
        "type": "model",
        "link": "THUDM/CogVideoX-2b",
    },
    {
        "name": "CogVideoX-5b",
        "description": "THUDM CogVideoX 5B – higher quality text-to-video.",
        "size": "~20GB",
        "type": "model",
        "link": "THUDM/CogVideoX-5b",
    },
    {
        "name": "AnimateDiff",
        "description": "AnimateDiff motion adapter for SD1.5-based animation.",
        "size": "~2GB",
        "type": "model",
        "link": "guoyww/animatediff-motion-adapter-v1-5-2",
    },
    {
        "name": "ModelScope T2V",
        "description": "Damo-vilab text-to-video MS 1.7B.",
        "size": "~5GB",
        "type": "model",
        "link": "damo-vilab/text-to-video-ms-1.7b",
    },
]


class DiffusersTTVBinding(LollmsTTVBinding):
    """
    Text-to-Video binding backed by a shared Diffusers daemon.

    The first instance spawns the daemon; subsequent instances (across
    processes) attach to the same running server via FileLock.
    """

    def __init__(self, **kwargs):
        if "model" in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.pop("model")
        super().__init__(binding_name="diffusers", **kwargs)

        self.config = kwargs
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = int(kwargs.get("port", 9634))
        self.model_name = kwargs.get("model_name", "Lightricks/LTX-Video")
        self.auto_start_server = kwargs.get("auto_start_server", True)
        self.wait_for_server = kwargs.get("wait_for_server", False)
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"

        system_root = self.get_system_dir()
        if kwargs.get("venv_path"):
            self.venv_dir = self.resolve_system_path(kwargs["venv_path"])
        else:
            self.venv_dir = (system_root / "venv" / "ttv_diffusers_venv").resolve()

        if kwargs.get("models_path"):
            self.models_path = self.resolve_system_path(kwargs["models_path"])
        else:
            self.models_path = (system_root / "data" / "ttv_models" / "diffusers").resolve()
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.hf_token = kwargs.get("hf_token", "")
        self.token_file = self.models_path / "diffusers_ttv.token"

        self.service_key = kwargs.get("service_key")
        if not self.service_key and self.token_file.exists():
            try:
                self.service_key = self.token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.2, status_forcelist=[502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retries))

        self.server_process: Optional[subprocess.Popen] = None

        if self.auto_start_server:
            self.ensure_server_is_running(self.wait_for_server)

    # ── HTTP helpers ────────────────────────────────────────────────────────
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
            headers["X-Server-Token"] = self.service_key
        return headers

    def is_server_running(self) -> bool:
        try:
            resp = requests.get(
                f"{self.base_url}/status",
                headers=self._get_headers(),
                timeout=0.5,
            )
            if resp.status_code == 200:
                data = resp.json() if callable(getattr(resp, "json", None)) else {}
                return data.get("status") == "running" if isinstance(data, dict) else True
            if resp.status_code == 401:
                return True
        except Exception:
            return False
        return False

    # ── Daemon lifecycle ────────────────────────────────────────────────────
    def ensure_server_is_running(self, wait: bool = False, timeout_s: int = 120) -> None:
        if not self.auto_start_server:
            return
        if self.is_server_running():
            return

        lock_path = self.models_path / "diffusers_ttv_spawn.lock"
        try:
            lock = FileLock(lock_path, timeout=timeout_s if wait else 1.0)
            with lock:
                if self.is_server_running():
                    ASCIIColors.green(f"Diffusers TTV daemon detected on {self.base_url}.")
                    return
                ASCIIColors.info(f"Spawning shared Diffusers TTV daemon on {self.base_url}...")
                self.start_server(wait=wait, timeout_s=timeout_s)
        except (Timeout, Exception) as e:
            if self.is_server_running():
                return
            if wait:
                raise RuntimeError(
                    f"Timed out waiting for Diffusers TTV daemon on {self.base_url}: {e}"
                )

    def install_server_dependencies(self) -> None:
        ASCIIColors.info(f"Setting up Diffusers TTV virtual environment in: {self.venv_dir}")
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir), create_if_not_exist=True)

        pm_v.ensure_packages(["requests", "uvicorn", "fastapi", "python-multipart", "filelock"])
        pm_v.ensure_packages(["ascii_colors>=0.11.10", "pipmaster", "tqdm", "numpy", "pillow", "imageio", "imageio-ffmpeg"])

        torch_index_url = None
        if sys.platform == "win32":
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
                ASCIIColors.green("NVIDIA GPU detected. Installing CUDA-enabled PyTorch.")
                torch_index_url = "https://download.pytorch.org/whl/cu126"
            except (FileNotFoundError, subprocess.CalledProcessError):
                ASCIIColors.yellow("No GPU detected. Installing standard PyTorch.")

        pm_v.ensure_packages(["torch", "torchvision"], index_url=torch_index_url)
        pm_v.ensure_packages(["transformers", "accelerate", "diffusers"])
        ASCIIColors.green("Diffusers TTV server dependencies are satisfied.")

    def start_server(self, wait: bool = False, timeout_s: int = 120) -> None:
        def _launch():
            try:
                server_script = self.server_dir / "main.py"
                venv_cfg = self.venv_dir / "pyvenv.cfg"

                if not venv_cfg.exists():
                    self.install_server_dependencies()

                if sys.platform == "win32":
                    python_executable = self.venv_dir / "Scripts" / "python.exe"
                else:
                    python_executable = self.venv_dir / "bin" / "python"

                if not python_executable.exists():
                    ASCIIColors.error(f"Python executable not found: {python_executable}.")
                    return

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
                            with os.fdopen(fd, "w", encoding="utf-8") as f:
                                f.write(self.service_key)
                        except Exception:
                            self.token_file.write_text(self.service_key, encoding="utf-8")

                command = [
                    str(python_executable),
                    "-u",
                    str(server_script),
                    "--host", str(self.host),
                    "--port", str(self.port),
                    "--cache-dir", str(self.models_path),
                    "--token", str(self.service_key),
                    "--model-name", str(self.model_name),
                ]

                log_file_path = self.models_path / "diffusers_ttv_server.log"
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
            except Exception as ex:
                ASCIIColors.error(f"Diffusers TTV daemon launch failed: {ex}")

        if wait:
            _launch()
            start_time = time.time()
            while time.time() - start_time < timeout_s:
                if self.is_server_running():
                    ASCIIColors.success("Diffusers TTV server is operational.")
                    return
                time.sleep(1)
            raise TimeoutError(f"Diffusers TTV server failed to respond within {timeout_s}s.")
        else:
            threading.Thread(target=_launch, daemon=True).start()

    def shutdown_server(self) -> bool:
        if not self.is_server_running():
            return True
        try:
            resp = self._session.post(
                f"{self.base_url}/shutdown",
                headers=self._get_headers(),
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ── Generation ──────────────────────────────────────────────────────────
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 512,
        num_frames: int = 24,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = -1,
        fps: int = 8,
        **kwargs,
    ) -> bytes:
        self.ensure_server_is_running(True)
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "fps": fps,
            "model_name": kwargs.get("model_name", self.model_name),
        }
        response = self._session.post(
            f"{self.base_url}/generate_video",
            json=payload,
            headers=self._get_headers(),
            timeout=1800,
        )
        response.raise_for_status()
        return response.content

    # ── Model management ────────────────────────────────────────────────────
    def list_models(self, **kwargs) -> List[str]:
        self.ensure_server_is_running(True)
        try:
            resp = self._session.get(
                f"{self.base_url}/list_models",
                headers=self._get_headers(),
                timeout=15,
            )
            resp.raise_for_status()
            models = resp.json().get("models", [])
            if models:
                return models
        except Exception:
            pass
        discovered: List[str] = []
        if self.models_path.exists():
            for d in self.models_path.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    discovered.append(d.name.replace("__", "/"))
        return list(dict.fromkeys([m["link"] for m in DEFAULT_TTV_ZOO] + discovered))

    def get_zoo(self) -> List[Dict[str, Any]]:
        return list(DEFAULT_TTV_ZOO)

    def install_model(self, model_name: str, **kwargs) -> dict:
        return self.pull_model(model_name=model_name, **kwargs)

    def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        self.ensure_server_is_running(True)
        try:
            if progress_callback:
                progress_callback({"status": "starting", "message": f"Downloading {model_name}..."})
            resp = self._session.post(
                f"{self.base_url}/pull_model",
                json={"model_name": model_name},
                headers=self._get_headers(),
                timeout=3600,
            )
            resp.raise_for_status()
            if progress_callback:
                progress_callback({"status": "success", "message": f"Downloaded {model_name}"})
            return resp.json()
        except Exception as e:
            return {"status": False, "message": str(e)}

    def download_from_zoo(
        self,
        index: int,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        zoo = self.get_zoo()
        if not (0 <= index < len(zoo)):
            return {"status": False, "message": "Index out of bounds."}
        return self.pull_model(zoo[index]["link"], progress_callback=progress_callback)

    def ps(self) -> List[dict]:
        try:
            resp = self._session.get(
                f"{self.base_url}/ps",
                headers=self._get_headers(),
                timeout=10,
            )
            return resp.json()
        except Exception:
            return [{"error": "Could not connect to Diffusers TTV server."}]

    def get_server_logs(self) -> str:
        log_file = self.models_path / "diffusers_ttv_server.log"
        if not log_file.exists():
            return "No server log file found."
        try:
            lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            return "\n".join(lines[-100:])
        except Exception as e:
            return f"Failed to read logs: {e}"

    # ── Resource management contract ────────────────────────────────────────
    def is_local(self) -> bool:
        return True

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        return self.is_server_running()

    def has_active_resources(self) -> bool:
        return self.is_server_running()

    def get_loaded_models(self) -> List[str]:
        return [self.model_name] if self.is_server_running() else []

    def unload_model(self, model_name: Optional[str] = None) -> bool:
        try:
            resp = self._session.post(
                f"{self.base_url}/shutdown",
                headers=self._get_headers(),
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False