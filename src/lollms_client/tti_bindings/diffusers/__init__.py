import os
import sys
import base64
import requests
import subprocess
import time
import json
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

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

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import ASCIIColors

BindingName = "DiffusersBinding"

class DiffusersBinding(LollmsTTIBinding):
    """
    Client binding for a dedicated, managed Diffusers server.
    This architecture prevents multiple models from being loaded into memory
    in a multi-worker environment, solving OOM errors.
    """
    def __init__(self,
                 **kwargs):

        super().__init__(binding_name=BindingName)

        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')

        self.config = kwargs
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 9632)
        self.auto_start_server = kwargs.get("auto_start_server", True)
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"
        self.venv_dir = Path("./venv/tti_diffusers_venv")
        self.models_path = Path(kwargs.get("models_path", "./data/models/diffusers_models")).resolve()
        self.extra_models_path = kwargs.get("extra_models_path")
        self.models_path.mkdir(exist_ok=True, parents=True)
        if self.auto_start_server:
            self.ensure_server_is_running()

    def is_server_running(self) -> bool:
        """Checks if the server is already running and responsive."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=4)
            if response.status_code == 200 and response.json().get("status") == "running":
                return True
        except requests.exceptions.RequestException:
            return False
        return False


    def ensure_server_is_running(self):
        """
        Ensures the Diffusers server is running. If not, it attempts to start it
        in a process-safe manner using a file lock. This method is designed to
        prevent race conditions in multi-worker environments.
        """
        self.server_dir.mkdir(exist_ok=True)
        # Use a lock file in the binding's server directory for consistency across instances
        lock_path = self.server_dir / "diffusers_server.lock"
        lock = FileLock(lock_path)

        ASCIIColors.info("Attempting to start or connect to the Diffusers server...")

        # First, perform a quick check without the lock to avoid unnecessary waiting.
        if self.is_server_running():
            ASCIIColors.green("Diffusers Server is already running and responsive.")
            return

        try:
            # Try to acquire the lock with a timeout. If another process is starting
            # the server, this will wait until it's finished.
            with lock.acquire(timeout=3):
                # After acquiring the lock, we MUST re-check if the server is running.
                # Another process might have started it and released the lock while we were waiting.
                if not self.is_server_running():
                    ASCIIColors.yellow("Lock acquired. Starting dedicated Diffusers server...")
                    self.start_server()
                    # The process that starts the server is responsible for waiting for it to be ready
                    # BEFORE releasing the lock. This is the key to preventing race conditions.
                    self._wait_for_server()
                else:
                    ASCIIColors.green("Server was started by another process while we waited. Connected successfully.")
        except Timeout:
            # This happens if the process holding the lock takes more than 60 seconds to start the server.
            # We don't try to start another one. We just wait for the existing one to be ready.
            ASCIIColors.yellow("Could not acquire lock, another process is taking a long time to start the server. Waiting...")
            self._wait_for_server(timeout=60) # Give it a longer timeout here just in case.

        # A final verification to ensure we are connected.
        if not self.is_server_running():
            raise RuntimeError("Failed to start or connect to the Diffusers server after all attempts.")

    def install_server_dependencies(self):
        """
        Installs the server's dependencies into a dedicated virtual environment
        using pipmaster, which handles complex packages like PyTorch.
        """
        ASCIIColors.info(f"Setting up virtual environment in: {self.venv_dir}")
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir))

        # --- PyTorch Installation ---
        ASCIIColors.info(f"Installing server dependencies")
        pm_v.ensure_packages([
            "requests", "uvicorn", "fastapi", "python-multipart", "filelock"
        ])
        ASCIIColors.info(f"Installing parisneo libraries")
        pm_v.ensure_packages([
            "ascii_colors","pipmaster"
        ])
        ASCIIColors.info(f"Installing misc libraries (numpy, tqdm...)")
        pm_v.ensure_packages([
            "tqdm", "numpy"
        ])
        ASCIIColors.info(f"Installing Pillow")
        pm_v.ensure_packages([
            "pillow"
        ])
        
        ASCIIColors.info(f"Installing pytorch")
        torch_index_url = None
        if sys.platform == "win32":
            try:
                # Use nvidia-smi to detect CUDA
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
                ASCIIColors.green("NVIDIA GPU detected. Installing CUDA-enabled PyTorch.")
                # Using a common and stable CUDA version. Adjust if needed.
                torch_index_url = "https://download.pytorch.org/whl/cu128"
            except (FileNotFoundError, subprocess.CalledProcessError):
                ASCIIColors.yellow("`nvidia-smi` not found or failed. Installing standard PyTorch. If you have an NVIDIA GPU, please ensure drivers are installed and in PATH.")

        # Base packages including torch. pm.ensure_packages handles verbose output.
        pm_v.ensure_packages(["torch", "torchvision"], index_url=torch_index_url)

        # Standard dependencies
        ASCIIColors.info(f"Installing transformers dependencies")
        pm_v.ensure_packages([
            "transformers", "safetensors", "accelerate"
        ])
        ASCIIColors.info(f"[Optional] Installing xformers")
        try:
            pm_v.ensure_packages([
                "xformers"
            ])
        except:
            pass
        # Git-based diffusers to get the latest version
        ASCIIColors.info(f"Installing diffusers library from github")
        pm_v.ensure_packages([
            {
                "name": "diffusers",
                "vcs": "git+https://github.com/huggingface/diffusers.git",
                "condition": ">=0.35.1"
            }
        ])

        ASCIIColors.green("Server dependencies are satisfied.")

    def start_server(self):
        """
        Installs dependencies and launches the FastAPI server as a background subprocess.
        This method should only be called from within a file lock.
        """
        server_script = self.server_dir / "main.py"
        if not server_script.exists():
            # Fallback for old structure
            server_script = self.binding_root / "server.py"
            if not server_script.exists():
                raise FileNotFoundError(f"Server script not found at {server_script}. Make sure it's in a 'server' subdirectory.")
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
            "--models-path", str(self.models_path.resolve()) # Pass models_path to server
        ]

        if self.extra_models_path:
            resolved_extra_path = Path(self.extra_models_path).resolve()
            command.extend(["--extra-models-path", str(resolved_extra_path)])

        # Use DETACHED_PROCESS on Windows to allow the server to run independently of the parent process.
        # On Linux/macOS, the process will be daemonized enough to not be killed with the worker.
        creationflags = subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0

        self.server_process = subprocess.Popen(command, creationflags=creationflags)
        ASCIIColors.info("Diffusers server process launched in the background.")

    def _wait_for_server(self, timeout=300):
        """Waits for the server to become responsive."""
        ASCIIColors.info("Waiting for Diffusers server to become available...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                ASCIIColors.green("Diffusers Server is up and running.")
                # Set initial settings from the binding's config, but only if a model is specified.
                if self.config.get("model_name"):
                    try:
                        ASCIIColors.info(f"Syncing initial client settings to server (model: {self.config['model_name']})...")
                        self.set_settings(self.config)
                    except Exception as e:
                        ASCIIColors.warning(f"Could not sync initial settings to server: {e}")
                else:
                    ASCIIColors.warning("Client has no model_name configured, skipping initial settings sync.")
                return
            time.sleep(2)
        raise RuntimeError("Failed to connect to the Diffusers server within the specified timeout.")

    def _post_json_request(self, endpoint: str, data: Optional[dict] = None) -> requests.Response:
        """Helper to make POST requests with a JSON body."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, json=data, timeout=3600) # Long timeout for generation
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to communicate with Diffusers server at {url}.")
            ASCIIColors.error(f"Error details: {e}")
            if hasattr(e, 'response') and e.response:
                try:
                    ASCIIColors.error(f"Server response: {e.response.json().get('detail', e.response.text)}")
                except json.JSONDecodeError:
                    ASCIIColors.error(f"Server raw response: {e.response.text}")
            raise RuntimeError("Communication with the Diffusers server failed.") from e

    def _post_multipart_request(self, endpoint: str, data: Optional[dict] = None, files: Optional[list] = None) -> requests.Response:
        """Helper to make multipart/form-data POST requests for file uploads."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, data=data, files=files, timeout=3600)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            # (Error handling is the same as above)
            ASCIIColors.error(f"Failed to communicate with Diffusers server at {url}.")
            ASCIIColors.error(f"Error details: {e}")
            if hasattr(e, 'response') and e.response:
                try:
                    ASCIIColors.error(f"Server response: {e.response.json().get('detail', e.response.text)}")
                except json.JSONDecodeError:
                    ASCIIColors.error(f"Server raw response: {e.response.text}")
            raise RuntimeError("Communication with the Diffusers server failed.") from e
        
    def _get_request(self, endpoint: str, params: Optional[dict] = None) -> requests.Response:
        """Helper to make GET requests to the server."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to communicate with Diffusers server at {url}.")
            raise RuntimeError("Communication with the Diffusers server failed.") from e

    def unload_model(self):
        ASCIIColors.info("Requesting server to unload the current model...")
        try:
            self._post_json_request("/unload_model")
        except Exception as e:
            ASCIIColors.warning(f"Could not send unload request to server: {e}")
        pass

    def generate_image(self, prompt: str, negative_prompt: str = "", **kwargs) -> bytes:
        params = kwargs.copy()
        if "model_name" not in params and self.config.get("model_name"):
            params["model_name"] = self.config["model_name"]
            
        response = self._post_json_request("/generate_image", data={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "params": params
        })
        return response.content

    def edit_image(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], prompt: str, **kwargs) -> bytes:
        images_b64 = []
        if not isinstance(images, list):
            images = [images]


        for img in images:
            # Case 1: Input is a PIL Image object
            if hasattr(img, 'save'):
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                b64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                images_b64.append(b64_string)
            
            # Case 2: Input is a string (could be path or already base64)
            elif isinstance(img, str):
                try:
                    b64_string = img.split(";base64,")[1] if ";base64," in img else img
                    base64.b64decode(b64_string) # Validate
                    images_b64.append(b64_string)
                except Exception:
                    ASCIIColors.warning(f"Warning: A string input was not a valid file path or base64. Skipping.")
            else:
                raise ValueError(f"Unsupported image type in edit_image: {type(img)}")
        if not images_b64:
            raise ValueError("No valid images were provided to the edit_image function.")
        
        params = kwargs.copy()
        if "model_name" not in params and self.config.get("model_name"):
            params["model_name"] = self.config["model_name"]

        # Translate "mask" to "mask_image" for server compatibility
        if "mask" in params and params["mask"]:
            params["mask_image"] = params.pop("mask")

        json_payload = {
            "prompt": prompt,
            "images_b64": images_b64,
            "params": params
        }
        response = self._post_json_request("/edit_image", data=json_payload)
        return response.content

    def list_models(self) -> List[Dict[str, Any]]:
        return self._get_request("/list_models").json()

    def list_local_models(self) -> List[str]:
        return self._get_request("/list_local_models").json()

    def list_available_models(self) -> List[str]:
        return self._get_request("/list_available_models").json()

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        return self._get_request("/list_models").json()

    def get_settings(self, **kwargs) -> List[Dict[str, Any]]:
        # The server holds the state, so we fetch it.
        return self._get_request("/get_settings").json()

    def set_settings(self, settings: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> bool:
        # Normalize settings from list of dicts to a single dict if needed
        parsed_settings = settings if isinstance(settings, dict) else {s["name"]: s["value"] for s in settings if "name" in s and "value" in s}
        response = self._post_json_request("/set_settings", data=parsed_settings)
        return response.json().get("success", False)

    def ps(self) -> List[dict]:
        try:
            return self._get_request("/ps").json()
        except Exception:
            return [{"error": "Could not connect to server to get process status."}]

    def __del__(self):
        # The client destructor does not stop the server,
        # as it is a shared resource for all worker processes.
        pass
