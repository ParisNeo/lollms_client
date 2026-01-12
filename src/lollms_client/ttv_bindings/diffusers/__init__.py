import os
import sys
import requests
import subprocess
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

# Ensure pipmaster is available.
try:
    import pipmaster as pm
except ImportError:
    print("FATAL: pipmaster is not installed. Please install it using: pip install pipmaster")
    sys.exit(1)

# Ensure filelock is available for process-safe server startup.
try:
    from filelock import FileLock, Timeout
except ImportError:
    print("FATAL: The 'filelock' library is required. Please install it by running: pip install filelock")
    sys.exit(1)

from lollms_client.lollms_ttv_binding import LollmsTTVBinding
from ascii_colors import ASCIIColors

BindingName = "DiffusersTTVBinding"

class DiffusersTTVBinding(LollmsTTVBinding):
    def __init__(self, **kwargs):
        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name=BindingName, config=kwargs)

        self.config = kwargs
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 9634) # Different port from TTI (9632) and TTS (9633)
        self.auto_start_server = kwargs.get("auto_start_server", True)
        self.wait_for_server = kwargs.get("wait_for_server", False)
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"
        self.venv_dir = Path("./venv/ttv_diffusers_venv")
        self.models_path = Path(kwargs.get("models_path", "./data/ttv_models/diffusers")).resolve()
        self.hf_token = kwargs.get("hf_token", "")
        
        self.models_path.mkdir(exist_ok=True, parents=True)
        
        if self.auto_start_server:
            self.ensure_server_is_running(self.wait_for_server)

    def is_server_running(self) -> bool:
        """Checks if the server is already running and responsive."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=4)
            if response.status_code == 200 and response.json().get("status") == "running":
                return True
        except requests.exceptions.RequestException:
            return False
        return False

    def ensure_server_is_running(self, wait=False):
        """
        Ensures the Diffusers TTV server is running.
        """
        self.server_dir.mkdir(exist_ok=True)
        ASCIIColors.info("Attempting to start or connect to the Diffusers TTV server...")

        if self.is_server_running():
            ASCIIColors.green("Diffusers TTV Server is already running and responsive.")
            return
        else:
            self.start_server(wait)

    def install_server_dependencies(self):
        """
        Installs the server's dependencies into a dedicated virtual environment.
        """
        ASCIIColors.info(f"Setting up virtual environment in: {self.venv_dir}")
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir))

        # Core dependencies
        ASCIIColors.info(f"Installing server dependencies")
        pm_v.ensure_packages([
            "requests", "uvicorn", "fastapi", "python-multipart", "filelock", "ascii_colors", "pipmaster"
        ])
        
        ASCIIColors.info(f"Installing torch and diffusers ecosystem")
        torch_index_url = "https://download.pytorch.org/whl/cu121" if sys.platform == "win32" else None
        
        pm_v.ensure_packages(["torch", "torchvision"], index_url=torch_index_url)
        pm_v.ensure_packages([
            "transformers", "accelerate", "safetensors", "sentencepiece", "protobuf", "moviepy", "imageio[ffmpeg]"
        ])
        
        # Install latest diffusers from git for LTX support
        ASCIIColors.info(f"Installing diffusers library from github")
        pm_v.ensure_packages([
            {
                "name": "diffusers",
                "vcs": "git+https://github.com/huggingface/diffusers.git"
            }
        ])

        ASCIIColors.green("Server dependencies are satisfied.")

    def start_server(self, wait=True):
        """
        Launches the FastAPI server in a background thread.
        """
        import threading
        
        def _start_server_background():
            lock_path = self.server_dir / "diffusers_ttv_server.lock"
            lock = FileLock(lock_path)
            with lock.acquire(timeout=0):
                try:
                    server_script = self.server_dir / "main.py"
                    if not self.venv_dir.exists():
                        self.install_server_dependencies()

                    if sys.platform == "win32":
                        python_executable = self.venv_dir / "Scripts" / "python.exe"
                    else:
                        python_executable = self.venv_dir / "bin" / "python"

                    command = [
                        str(python_executable),
                        str(server_script),
                        "--host", self.host,
                        "--port", str(self.port),
                        "--models-path", str(self.models_path.resolve())
                    ]

                    if self.hf_token:
                        command.extend(["--hf-token", self.hf_token])

                    creationflags = subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
                    self.server_process = subprocess.Popen(command, creationflags=creationflags)
                    ASCIIColors.info("Diffusers TTV server process launched in the background.")
                    
                    while not self.is_server_running():
                        time.sleep(1)
                    
                except Exception as e:
                    ASCIIColors.error(f"Failed to start Diffusers TTV server: {e}")
                    raise

        thread = threading.Thread(target=_start_server_background, daemon=True)
        thread.start()
        if wait:
            thread.join()

    def _post_json_request(self, endpoint: str, data: Optional[dict] = None) -> requests.Response:
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, json=data, timeout=3600) # Long timeout for video generation
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to communicate with TTV server at {url}: {e}")
            raise RuntimeError("Communication with the TTV server failed.") from e

    def _get_request(self, endpoint: str, params: Optional[dict] = None) -> requests.Response:
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to communicate with TTV server at {url}: {e}")
            raise RuntimeError("Communication with the TTV server failed.") from e

    def generate_video(self, prompt: str, **kwargs) -> bytes:
        """
        Generates video data from the provided text prompt.
        """
        self.ensure_server_is_running(True)
        
        params = kwargs.copy()
        if "model_name" not in params and self.config.get("model_name"):
            params["model_name"] = self.config["model_name"]
            
        payload = {
            "prompt": prompt,
            "negative_prompt": params.get("negative_prompt", ""),
            "params": params
        }
        
        response = self._post_json_request("/generate_video", data=payload)
        return response.content

    def list_models(self, **kwargs) -> List[str]:
        """
        Lists available TTV models.
        """
        self.ensure_server_is_running(True)
        try:
            return self._get_request("/list_models").json()
        except Exception:
            return []

    def get_zoo(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "LTX-Video (LTX-2)",
                "description": "Lightricks LTX-Video model (2B). High quality, realistic video generation.",
                "size": "5GB",
                "type": "safetensors",
                "link": "Lightricks/LTX-Video"
            },
            {
                "name": "CogVideoX-2b",
                "description": "THUDM CogVideoX 2B model.",
                "size": "4GB",
                "type": "safetensors",
                "link": "THUDM/CogVideoX-2b"
            },
            {
                "name": "CogVideoX-5b",
                "description": "THUDM CogVideoX 5B model (requires significant VRAM).",
                "size": "10GB",
                "type": "safetensors",
                "link": "THUDM/CogVideoX-5b"
            }
        ]
        
    def download_from_zoo(self, index: int, progress_callback: Callable[[dict], None] = None) -> dict:
        zoo = self.get_zoo()
        if index < 0 or index >= len(zoo):
            return {"status": False, "message": "Index out of bounds"}
        
        item = zoo[index]
        model_name = item["link"]
        
        payload = {"model_name": model_name}
        try:
            if progress_callback:
                progress_callback({"status": "starting", "message": f"Pulling {model_name}..."})
            
            # Simple pull endpoint
            # For TTV, models are often downloaded on-demand by the pipeline from HF, 
            # but we can pre-fetch them.
            # We'll rely on the server's cache or a dedicated pull endpoint if we implement one.
            # Here we just trigger a load check or simple success message as diffusers handles caching.
            
            return {"status": True, "message": f"Model {model_name} selected. It will be downloaded on first use if not cached."}
            
        except Exception as e:
            return {"status": False, "message": str(e)}

    def __del__(self):
        pass
