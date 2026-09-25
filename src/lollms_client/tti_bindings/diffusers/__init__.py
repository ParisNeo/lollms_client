import os
import sys
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import subprocess
import time
import json
import secrets
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

import pipmaster as pm
from filelock import FileLock, Timeout
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_tti_binding import LollmsTTIBinding

BindingName = "DiffusersTTIBinding"


class DiffusersTTIBinding(LollmsTTIBinding):
    """
    Client binding for the shared Diffusers TTI server daemon.
    Operates in process-safe Shared Singleton mode across all worker processes
    without port drifting or VRAM duplication.
    """

    def __init__(self, **kwargs):
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name=BindingName, config=kwargs)

        self.config = kwargs
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = int(kwargs.get("port", 9632))
        self.auto_start_server = kwargs.get("auto_start_server", True)
        self.wait_for_server = kwargs.get("wait_for_server", True)
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"

        self.venv_dir = Path(kwargs.get("venv_path", "./venv/tti_diffusers_venv")).resolve()
        self.models_path = Path(kwargs.get("models_path", "./data/tti_models/diffusers")).resolve()
        self.extra_models_path = kwargs.get("extra_models_path")
        self.hf_token = kwargs.get("hf_token", "")
        self.server_log_depth = int(kwargs.get("server_log_depth", 500))
        self.models_path.mkdir(exist_ok=True, parents=True)

        self.token_file = self.models_path / "diffusers_server.token"
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

        if self.config.get("model_name"):
            try:
                self.set_settings(self.config)
            except Exception as e:
                ASCIIColors.warning(f"Could not sync initial settings to server: {e}")

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
            headers["X-Server-Token"] = self.service_key
        return headers

    def is_server_running(self) -> bool:
        try:
            response = self._session.get(
                f"{self.base_url}/status",
                headers=self._get_headers(),
                timeout=1.5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "running"
            elif response.status_code == 401:
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

        lock_path = self.models_path / "diffusers_server_spawn.lock"
        lock = FileLock(lock_path, timeout=timeout_s)

        try:
            with lock:
                if self.is_server_running():
                    ASCIIColors.green("Diffusers Server is already running and responsive (Shared Singleton).")
                    return
                ASCIIColors.info(f"Spawning shared Diffusers server daemon on {self.base_url}...")
                self.start_server(wait=wait, timeout_s=timeout_s)
        except Timeout:
            if self.is_server_running():
                return
            raise RuntimeError(f"Timed out waiting for Diffusers server on {self.base_url}.")

    def install_server_dependencies(self):
        ASCIIColors.info(f"Setting up Diffusers virtual environment in: {self.venv_dir}")
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir), create_if_not_exist=True)

        pm_v.ensure_packages(["requests", "uvicorn", "fastapi", "python-multipart", "filelock"])
        pm_v.ensure_packages(["ascii_colors>=0.11.10", "pipmaster", "tqdm", "numpy", "pillow"])

        torch_index_url = None
        if sys.platform == "win32":
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
                ASCIIColors.green("NVIDIA GPU detected. Installing CUDA-enabled PyTorch.")
                torch_index_url = "https://download.pytorch.org/whl/cu128"
            except (FileNotFoundError, subprocess.CalledProcessError):
                ASCIIColors.yellow("Installing standard PyTorch.")

        pm_v.ensure_packages(["torch", "torchvision", "torchaudio"], index_url=torch_index_url)
        pm_v.ensure_packages(["transformers", "safetensors", "accelerate", "diffusers"])
        ASCIIColors.green("Diffusers server dependencies are satisfied.")

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
            "-u",
            str(server_script),
            "--host", str(self.host),
            "--port", str(self.port),
            "--models-path", str(self.models_path.resolve()),
            "--token", str(self.service_key),
        ]
        if self.extra_models_path:
            command.extend(["--extra-models-path", str(Path(self.extra_models_path).resolve())])
        if self.hf_token:
            command.extend(["--hf-token", self.hf_token])

        log_file_path = self.models_path / "diffusers_server.log"
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
                    ASCIIColors.success("Diffusers server is ready.")
                    return
                time.sleep(1)
            raise TimeoutError(f"Diffusers server failed to start within {timeout_s}s.")

    def __del__(self):
        # Shared singleton daemon remains running for all application worker processes
        pass

    def shutdown_server(self) -> bool:
        if not self.is_server_running():
            return True
        try:
            resp = self._session.post(f"{self.base_url}/shutdown", headers=self._get_headers(), timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _post_json_request(self, endpoint: str, data: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.post(url, json=data, headers=self._get_headers(), timeout=3600)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed request to Diffusers server at {url}: {e}")
            raise RuntimeError(f"Diffusers server error: {e}") from e

    def _get_request(self, endpoint: str, params: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.get(url, params=params, headers=self._get_headers(), timeout=60)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Communication with Diffusers server failed: {e}") from e

    def unload_model(self):
        try:
            self._post_json_request("/unload_model")
        except Exception as e:
            ASCIIColors.warning(f"Could not send unload request to server: {e}")

    def install_model(self, model_name: str, **kwargs) -> dict:
        """
        Installs a model from Hugging Face into the local models directory so it becomes searchable.
        """
        return self.pull_model(model_name=model_name, **kwargs)

    def pull_model(self, model_name: str, **kwargs) -> dict:
        """
        Downloads a model from Hugging Face into the local models directory
        so it becomes searchable and selectable.
        """
        self.ensure_server_is_running(True)
        payload = {
            "model_name": model_name,
            "hf_id": model_name,
            **kwargs
        }
        response = self._post_json_request("/pull_model", data=payload)
        return response.json()

    def generate_image(self, prompt: str, negative_prompt: str = "", **kwargs) -> bytes:
        self.ensure_server_is_running(True)
        params = kwargs.copy()
        if "model_name" not in params and self.config.get("model_name"):
            params["model_name"] = self.config["model_name"]

        response = self._post_json_request("/generate_image", data={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "params": params
        })
        return response.content

    def edit_image(self, images: Union[str, List[str]], prompt: str, **kwargs) -> bytes:
        self.ensure_server_is_running(True)
        images_b64 = []
        if not isinstance(images, list):
            images = [images]

        for img in images:
            if hasattr(img, 'save'):
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                b64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                images_b64.append(b64_string)
            elif isinstance(img, str):
                try:
                    b64_string = img.split(";base64,")[1] if ";base64," in img else img
                    base64.b64decode(b64_string)
                    images_b64.append(b64_string)
                except Exception:
                    p = Path(img)
                    if p.exists():
                        b64_string = base64.b64encode(p.read_bytes()).decode('utf-8')
                        images_b64.append(b64_string)

        params = kwargs.copy()
        if "model_name" not in params and self.config.get("model_name"):
            params["model_name"] = self.config["model_name"]
        if "mask" in params and params["mask"]:
            params["mask_image"] = params.pop("mask")

        response = self._post_json_request("/edit_image", data={
            "prompt": prompt,
            "images_b64": images_b64,
            "params": params
        })
        return response.content

    def list_models(self) -> list:
        self.ensure_server_is_running(True)
        try:
            return self._get_request("/list_models").json()
        except Exception:
            return []

    def set_settings(self, settings: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> bool:
        self.ensure_server_is_running(True)
        parsed = settings if isinstance(settings, dict) else {s["name"]: s["value"] for s in settings if "name" in s and "value" in s}
        response = self._post_json_request("/set_settings", data=parsed)
        return response.json().get("success", False)

    def ps(self) -> List[dict]:
        try:
            return self._get_request("/ps").json()
        except Exception:
            return [{"error": "Could not connect to server to get process status."}]