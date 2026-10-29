"""
vLLM-Omni / MiniMax-H3 Self-Managed TTV binding for lollms_client.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_ttv_binding import LollmsTTVBinding

BindingName = "VllmOmniTTVBinding"

_TTV_ZOO: List[Dict[str, Any]] = [
    {
        "name": "MiniMax-H3-FL2VA-FP8",
        "hf_id": "ParisNeo/MiniMax-H3-FL2VA-FP8",
        "description": "Optimized FP8 pre-quantized MiniMax-H3 video model for single GPU (A100 40G)",
        "size": "~35-40 GB",
        "type": "t2va/fl2va",
        "link": "https://huggingface.co/ParisNeo/MiniMax-H3-FL2VA-FP8",
    },
    {
        "name": "MiniMax-H3-Original",
        "hf_id": "MiniMaxAI/MiniMax-H3",
        "description": "Original unquantized MiniMax-H3 multimodal base model",
        "size": "~100+ GB",
        "type": "t2va/fl2va",
        "link": "https://huggingface.co/MiniMaxAI/MiniMax-H3",
    },
]


class VllmOmniTTVBinding(LollmsTTVBinding):
    def __init__(
        self,
        server_mode: str = "remote",
        base_url: str = "http://localhost:8002",
        model: str = "ParisNeo/MiniMax-H3-FL2VA-FP8",
        api_key: Optional[str] = None,
        venv_path: str = "./venv_vllm_omni",
        models_dir: str = "./models/vllm_omni",
        gpus: str = "0",
        gpu_memory_utilization: float = 0.90,
        port: int = 8002,
        auto_start_server: bool = True,
        default_aspect_ratio: str = "16:9",
        default_duration: float = 4.0,
        default_fps: int = 24,
        timeout: int = 600,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(binding_name="vllm_omni_ttv", debug=debug, **kwargs)

        self.server_mode = server_mode.lower().strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        
        self.venv_path = Path(venv_path).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.gpus = str(gpus)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.port = int(port)
        self.auto_start_server = auto_start_server

        self.default_aspect_ratio = default_aspect_ratio
        self.default_duration = default_duration
        self.default_fps = default_fps
        self.timeout = timeout

        self.process: Optional[subprocess.Popen] = None

        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Environment & Server Management Methods
    # ------------------------------------------------------------------

    def get_python_executable(self) -> Path:
        """Returns the Python executable path within the configured venv."""
        if os.name == "nt":
            return self.venv_path / "Scripts" / "python.exe"
        return self.venv_path / "bin" / "python"

    def get_vllm_executable(self) -> Path:
        """Returns the vllm CLI executable path within the configured venv."""
        if os.name == "nt":
            return self.venv_path / "Scripts" / "vllm.exe"
        return self.venv_path / "bin" / "vllm"

    def install_environment(self, **kwargs) -> Dict[str, Any]:
        """Creates the virtualenv and installs vllm and vllm-omni."""
        try:
            ASCIIColors.info(f"Creating virtual environment at {self.venv_path}...")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)

            py_bin = self.get_python_executable()
            ASCIIColors.info("Upgrading pip and installing wheel/setuptools...")
            subprocess.run([str(py_bin), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)

            ASCIIColors.info("Installing vllm and vllm-omni packages...")
            subprocess.run([str(py_bin), "-m", "pip", "install", "vllm>=0.32.0", "vllm-omni>=0.32.0", "huggingface_hub"], check=True)

            return {"status": True, "message": f"Environment successfully initialized at {self.venv_path}"}
        except Exception as e:
            trace_exception(e)
            return {"status": False, "message": f"Environment installation failed: {str(e)}"}

    def start_server(self, **kwargs) -> Dict[str, Any]:
        """Launches the vLLM-Omni server process in managed mode."""
        if self.health_check():
            return {"status": True, "message": "Server is already running and healthy."}

        vllm_bin = self.get_vllm_executable()
        if not vllm_bin.exists():
            return {"status": False, "message": f"vLLM executable not found at {vllm_bin}. Run install_environment() first."}

        # Select model source: local directory if downloaded, or HF repo ID
        local_model_path = self.models_dir / self.model.replace("/", "--")
        target_model = str(local_model_path) if local_model_path.exists() else self.model

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpus
        env["HF_HOME"] = str(self.models_dir)

        cmd = [
            str(vllm_bin), "serve", target_model,
            "--host", "0.0.0.1" if os.name != "nt" else "127.0.0.1",
            "--port", str(self.port),
            "--tensor-parallel-size", "1",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--init-timeout", "1200",
            "--stage-init-timeout", "1200",
            "--omni",
            "--task-type", "fl2va",
            "--trust-remote-code",
            "--vae-use-slicing",
            "--vae-use-tiling",
            "--enable-layerwise-offload",
            "--cache-backend", "cache_dit",
        ]

        ASCIIColors.info(f"Starting managed vLLM-Omni server on GPUs [{self.gpus}] port {self.port}...")
        try:
            self.process = subprocess.Popen(cmd, env=env)
            
            # Wait for server readiness
            start_time = time.time()
            while time.time() - start_time < 300:
                if self.health_check():
                    ASCIIColors.success("vLLM-Omni server successfully started and ready!")
                    return {"status": True, "message": "Server started successfully."}
                if self.process.poll() is not None:
                    return {"status": False, "message": f"Server process exited unexpectedly with code {self.process.returncode}."}
                time.sleep(3)

            return {"status": False, "message": "Timed out waiting for server to start."}
        except Exception as e:
            trace_exception(e)
            return {"status": False, "message": f"Failed to launch server: {e}"}

    def stop_server(self, **kwargs) -> Dict[str, Any]:
        """Stops the managed server process to free VRAM."""
        if self.process and self.process.poll() is None:
            ASCIIColors.info("Terminating managed vLLM-Omni server process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            return {"status": True, "message": "Server process stopped."}
        return {"status": True, "message": "No active managed server process was running."}

    # ------------------------------------------------------------------
    # Model Pulling & Zoo
    # ------------------------------------------------------------------

    def pull_model(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """Downloads model weights from Hugging Face into the configured models_dir."""
        try:
            from huggingface_hub import snapshot_download

            target_dir = self.models_dir / model_name.replace("/", "--")
            ASCIIColors.info(f"Downloading {model_name} into {target_dir}...")
            
            download_path = snapshot_download(
                repo_id=model_name,
                local_dir=target_dir,
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
            )
            return {"status": True, "path": download_path, "message": f"Successfully pulled {model_name}"}
        except Exception as e:
            trace_exception(e)
            return {"status": False, "message": f"Failed to pull model: {str(e)}"}

    def install_model(self, model_name: str, **kwargs) -> Dict[str, Any]:
        return self.pull_model(model_name, **kwargs)

    def get_zoo(self) -> List[Dict[str, Any]]:
        return _TTV_ZOO

    def download_from_zoo(self, index: int, progress_callback: Optional[Callable[[dict], None]] = None) -> Dict[str, Any]:
        if index < 0 or index >= len(_TTV_ZOO):
            return {"status": False, "message": f"Index {index} out of range"}
        entry = _TTV_ZOO[index]
        return self.pull_model(entry["hf_id"])

    # ------------------------------------------------------------------
    # Generation Logic
    # ------------------------------------------------------------------

    def _ensure_server_ready(self):
        if self.server_mode == "managed" and self.auto_start_server and not self.health_check():
            res = self.start_server()
            if not res["status"]:
                raise RuntimeError(f"Managed server auto-start failed: {res['message']}")

    def _prepare_image_bytes(self, image: Union[bytes, str, Path]) -> bytes:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                return path.read_bytes()
            return base64.b64decode(image)
        return image

    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        task: str = "t2va",
        **kwargs,
    ) -> bytes:
        self._ensure_server_ready()

        data: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio or self.default_aspect_ratio,
            "fps": fps or self.default_fps,
            "num_inference_steps": num_inference_steps,
        }
        if self.model:
            data["model"] = self.model
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        if seed is not None and seed != -1:
            data["seed"] = seed

        extra_params = {
            "task": task,
            "duration": duration if duration is not None else self.default_duration,
        }
        extra_params.update(kwargs.get("extra_params", {}))
        data["extra_params"] = json.dumps(extra_params)

        try:
            resp = self._session.post(
                f"{self.base_url}/v1/videos/sync",
                data=data,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.content
        except requests.HTTPError as e:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"MiniMax-H3 T2V generation failed [{resp.status_code}]: {detail}") from e

    def generate_video_from_image(
        self,
        prompt: str,
        image: Union[bytes, str, Path],
        negative_prompt: Optional[str] = None,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        self._ensure_server_ready()

        image_bytes = self._prepare_image_bytes(image)
        data: Dict[str, Any] = {
            "prompt": prompt,
            "fps": fps or self.default_fps,
            "num_inference_steps": num_inference_steps,
        }
        if self.model:
            data["model"] = self.model
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        if seed is not None and seed != -1:
            data["seed"] = seed

        extra_params = {
            "task": "fl2va",
            "duration": duration if duration is not None else self.default_duration,
        }
        data["extra_params"] = json.dumps(extra_params)

        files = {
            "input_reference": ("start_frame.png", io.BytesIO(image_bytes), "image/png")
        }

        try:
            resp = self._session.post(
                f"{self.base_url}/v1/videos/sync",
                data=data,
                files=files,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.content
        except requests.HTTPError as e:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"MiniMax-H3 I2V generation failed [{resp.status_code}]: {detail}") from e

    def generate_video_first_and_last_frame(
        self,
        prompt: str,
        start_image: Union[bytes, str, Path],
        end_image: Union[bytes, str, Path],
        negative_prompt: Optional[str] = None,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        self._ensure_server_ready()

        start_bytes = self._prepare_image_bytes(start_image)
        end_bytes = self._prepare_image_bytes(end_image)

        data: Dict[str, Any] = {
            "prompt": prompt,
            "fps": fps or self.default_fps,
            "num_inference_steps": num_inference_steps,
        }
        if self.model:
            data["model"] = self.model
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        if seed is not None and seed != -1:
            data["seed"] = seed

        extra_params = {
            "task": "fl2va",
            "duration": duration if duration is not None else self.default_duration,
            "frame_indices": [0, -1],
        }
        data["extra_params"] = json.dumps(extra_params)

        files = [
            ("input_references", ("start_frame.png", io.BytesIO(start_bytes), "image/png")),
            ("input_references", ("end_frame.png", io.BytesIO(end_bytes), "image/png")),
        ]

        try:
            resp = self._session.post(
                f"{self.base_url}/v1/videos/sync",
                data=data,
                files=files,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.content
        except requests.HTTPError as e:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"MiniMax-H3 First/Last frame generation failed [{resp.status_code}]: {detail}") from e

    def health_check(self) -> bool:
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self, **kwargs) -> List[str]:
        try:
            resp = self._session.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return [entry["name"] for entry in _TTV_ZOO]