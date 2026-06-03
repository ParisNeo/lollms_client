import os
import sys
import base64
import requests
import subprocess
import time
import json
from io import BytesIO
from pathlib import Path
from ascii_colors import trace_exception
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

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import ASCIIColors

BindingName = "DiffusersTTIBinding"

class DiffusersTTIBinding(LollmsTTIBinding):
    def __init__(self, **kwargs):
        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name=BindingName, config=kwargs)

        self.config = kwargs
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 9632)
        self.auto_start_server = kwargs.get("auto_start_server", False)
        self.wait_for_server = kwargs.get("wait_for_server", False)
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"
        
        self.venv_dir =  Path(kwargs.get("venv_path", "./venv/tti_diffusers_venv"))
        self.models_path = Path(kwargs.get("models_path", "./data/tti_models/diffusers")).resolve()
        self.extra_models_path = kwargs.get("extra_models_path")
        self.hf_token = kwargs.get("hf_token", "")  # NEW
        self.server_log_depth = int(kwargs.get("server_log_depth", 500))
        self.models_path.mkdir(exist_ok=True, parents=True)
        if self.auto_start_server:
            self.ensure_server_is_running(self.wait_for_server)

        # Always sync current configuration to the server on startup
        if self.config.get("model_name"):
            try:
                self.set_settings(self.config)
            except Exception as e:
                ASCIIColors.warning(f"Could not sync initial settings to server: {e}")


    def is_server_running(self) -> bool:
        """Checks if the server is already running and responsive."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=4)
            if response.status_code == 200 and response.json().get("status") == "running":
                return True
        except requests.exceptions.RequestException:
            return False
        return False

    def _is_port_available(self, host: str, port: int) -> bool:
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return True
        except OSError:
            return False

    def ensure_server_is_running(self, wait= False):
        """
        Ensures the Diffusers server is running. If not, it attempts to start it
        in a process-safe manner using a file lock. This method is designed to
        prevent race conditions in multi-worker environments.
        """
        self.server_dir.mkdir(exist_ok=True)
        ASCIIColors.info("Attempting to start or connect to the Diffusers server...")

        # First, perform a quick check without the lock to avoid unnecessary waiting.
        if self.is_server_running():
            ASCIIColors.green("Diffusers Server is already running and responsive.")
            return

        # If the port is busy by another application, find a free port
        original_port = self.port
        while not self._is_port_available(self.host, self.port):
            ASCIIColors.warning(f"Port {self.port} is busy. Trying next port...")
            self.port += 1
            self.base_url = f"http://{self.host}:{self.port}"

        if self.port != original_port:
            ASCIIColors.info(f"Selected new port {self.port} for Diffusers server.")

        self.start_server(wait)

    def install_server_dependencies(self):
        """
        Installs the server's dependencies into a dedicated virtual environment
        using pipmaster, which handles complex packages like PyTorch.
        """
        ASCIIColors.info(f"Setting up virtual environment in: {self.venv_dir}")
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir), create_if_not_exist=True)

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
        pm_v.ensure_packages(["torch", "torchvision", "torchaudio"], index_url=torch_index_url)

        # Standard dependencies
        ASCIIColors.info(f"Installing transformers dependencies")
        pm_v.ensure_packages([
            "transformers", "safetensors", "accelerate"
        ])
        ASCIIColors.info(f"Installing hugging face dependencies")
        pm_v.ensure_packages([
            "hf_xet"
        ])
        ASCIIColors.info(f"Installing bits and bytes for quantized models")
        pm_v.ensure_packages([
            "bitsandbytes"
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
        pm_v.ensure_packages("diffusers")

        ASCIIColors.info(f"Installing GGUF support libraries")
        pm_v.ensure_packages({"gguf":">=0.13.0"})

        ASCIIColors.green("Server dependencies are satisfied.")

    def start_server(self, wait: bool = True, timeout_s: int = 40):
        """
        Launches the FastAPI server in a background thread.
        """
        import threading
        import subprocess
        import sys
        import time

        def _start_server_background():
            lock_path = self.venv_dir / "diffusers_server.lock"
            lock = FileLock(lock_path)

            try:
                with lock.acquire(timeout=0):
                    server_script = self.server_dir / "main.py"
                    venv_cfg = self.venv_dir / "pyvenv.cfg"

                    if not venv_cfg.exists():
                        ASCIIColors.warning(
                            "Invalid or missing virtual environment. Reinstalling..."
                        )

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

                    if self.extra_models_path:
                        command.extend([
                            "--extra-models-path",
                            str(Path(self.extra_models_path).resolve())
                        ])

                    if self.hf_token:
                        command.extend([
                            "--hf-token",
                            self.hf_token
                        ])

                    log_file_path = self.models_path / "diffusers_server.log"
                    log_f = open(log_file_path, "w", encoding="utf-8")

                    self.server_process = subprocess.Popen(
                        command,
                        stdout=log_f,
                        stderr=subprocess.STDOUT
                    )
                    log_f.close()

                    ASCIIColors.info(
                        f"Diffusers server launched on "
                        f"http://{self.host}:{self.port}"
                    )

                    if wait:
                        start_time = time.time()

                        while True:
                            if self.server_process.poll() is not None:
                                raise RuntimeError(
                                    "Diffusers server process terminated unexpectedly."
                                )

                            if self.is_server_running():
                                ASCIIColors.success(
                                    "Diffusers server is ready."
                                )
                                return

                            elapsed = time.time() - start_time

                            if elapsed >= timeout_s:
                                raise TimeoutError(
                                    f"Server failed to start within "
                                    f"{timeout_s} seconds."
                                )

                            time.sleep(1)

            except Exception as ex:
                ASCIIColors.error(
                    f"Failed to start Diffusers server: {ex}"
                )
                raise

        thread = threading.Thread(
            target=_start_server_background,
            daemon=True
        )

        thread.start()

        if wait:
            thread.join()

    def _wait_for_server(self, timeout=30):
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

    def edit_image(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], prompt: str, **kwargs) -> bytes:
        self.ensure_server_is_running(True)
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

    def list_models(self) -> list:
        """
        Lists only models that are available locally on disk.

        The Diffusers server scans `models_path` and `extra_models_path` for:
          - Diffusers pipeline folders (with model_index.json, etc.)
          - .safetensors checkpoints and associated configs.

        Returns list of dicts: {"model_name": str, "display_name": str, "description": str}
        """
        self.ensure_server_is_running(True)
        try:
            response = self._get_request("/list_models")
            data = response.json()
            if not isinstance(data, list):
                return []
            return data
        except Exception as e:
            ASCIIColors.warning(f"Failed to list local Diffusers models: {e}")
            return []


    def list_local_models(self) -> List[str]:
        self.ensure_server_is_running(True)
        return self._get_request("/list_local_models").json()

    def list_available_models(self) -> List[str]:
        self.ensure_server_is_running(True)
        return self._get_request("/list_available_models").json()

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        self.ensure_server_is_running(True)
        return self._get_request("/list_models").json()

    def get_settings(self, **kwargs) -> List[Dict[str, Any]]:
        self.ensure_server_is_running(True)
        # The server holds the state, so we fetch it.
        return self._get_request("/get_settings").json()
    
    def get_zoo(self):
        return [
            {"name": "Stable Diffusion 1.5", "description": "The classic and versatile SD1.5 base model.", "size": "4.27GB", "type": "checkpoint", "link": "runwayml/stable-diffusion-v1-5"},
            {"name": "Stable Diffusion 2.1", "description": "The 768x768 base model from the SD2.x series.", "size": "5GB", "type": "checkpoint", "link": "stabilityai/stable-diffusion-2-1"},
            {"name": "Stable Diffusion XL 1.0", "description": "Official 1024x1024 text-to-image model from Stability AI.", "size": "7GB", "type": "checkpoint", "link": "stabilityai/stable-diffusion-xl-base-1.0"},
            {"name": "SDXL Turbo", "description": "A fast, real-time text-to-image model based on SDXL.", "size": "7GB", "type": "checkpoint", "link": "stabilityai/sdxl-turbo"},
            {"name": "Kandinsky 3", "description": "A powerful multilingual model with strong prompt understanding.", "size": "Unknown", "type": "checkpoint", "link": "kandinsky-community/kandinsky-3"},
            {"name": "Playground v2.5", "description": "A high-quality model focused on aesthetic outputs.", "size": "Unknown", "type": "checkpoint", "link": "playgroundai/playground-v2.5-1024px-aesthetic"},
            {"name": "epiCRealism", "description": "A popular community model for generating photorealistic images.", "size": "2GB", "type": "checkpoint", "link": "emilianJR/epiCRealism"},
            {"name": "Realistic Vision 5.1", "description": "One of the most popular realistic models, great for portraits.", "size": "2GB", "type": "checkpoint", "link": "SG161222/Realistic_Vision_V5.1_noVAE"},
            {"name": "Photon", "description": "A model known for high-quality, realistic images with good lighting.", "size": "2GB", "type": "checkpoint", "link": "Photon-v1"},
            {"name": "Waifu Diffusion 1.4", "description": "A widely-used model for generating high-quality anime-style images.", "size": "2GB", "type": "checkpoint", "link": "hakurei/waifu-diffusion"},
            {"name": "Counterfeit V3.0", "description": "A strong model for illustrative and 2.5D anime styles.", "size": "2GB", "type": "checkpoint", "link": "gsdf/Counterfeit-V3.0"},
            {"name": "Animagine XL 3.0", "description": "A state-of-the-art anime model on the SDXL architecture.", "size": "7GB", "type": "checkpoint", "link": "cagliostrolab/animagine-xl-3.0"},
            {"name": "DreamShaper 8", "description": "Versatile SD1.5 style model (CivitAI).", "size": "2GB", "type": "checkpoint", "link": "https://civitai.com/api/download/models/128713"},
            {"name": "Juggernaut XL", "description": "Artistic SDXL (CivitAI).", "size": "7GB", "type": "checkpoint", "link": "https://civitai.com/api/download/models/133005"},
            {"name": "Stable Diffusion 3 Medium", "description": "SOTA model with advanced prompt understanding (Gated).", "size": "Unknown", "type": "checkpoint", "link": "stabilityai/stable-diffusion-3-medium-diffusers"},
            {"name": "Instruct-Pix2Pix", "description": "Instruction-based image editing (SD1.5).", "size": "4.3GB", "type": "checkpoint", "link": "timbrooks/instruct-pix2pix"},
            {"name": "FLUX.1 Schnell", "description": "Powerful and fast next-gen model (Gated).", "size": "Unknown", "type": "checkpoint", "link": "black-forest-labs/FLUX.1-schnell"},
            {"name": "FLUX.1 Dev", "description": "Larger developer version of FLUX.1 (Gated).", "size": "Unknown", "type": "checkpoint", "link": "black-forest-labs/FLUX.1-dev"},
            {"name": "FLUX.2 Klein 4B", "description": "Distilled, fast, commercial-friendly 4B parameter model.", "size": "16GB", "type": "checkpoint", "link": "black-forest-labs/FLUX.2-klein-4B"},
            {"name": "FLUX.2 Klein 4B FP8", "description": "Pre-quantized 8-bit compact variant of FLUX.2 Klein.", "size": "13GB", "type": "checkpoint", "link": "black-forest-labs/FLUX.2-klein-4b-fp8"},
            {"name": "FLUX.2 Klein 4B GGUF", "description": "Ultra-low memory GGUF quantized 4B Klein model.", "size": "8GB", "type": "checkpoint", "link": "unsloth/FLUX.2-klein-4B-GGUF"},
            {"name": "FLUX.2 Klein 9B", "description": "9B parameter distilled model (Non-commercial).", "size": "36GB", "type": "checkpoint", "link": "black-forest-labs/FLUX.2-klein-9B"},
            {"name": "FLUX.2 Dev 12B", "description": "Next-generation 12B guidance model (Gated).", "size": "Unknown", "type": "checkpoint", "link": "black-forest-labs/FLUX.2-dev"},
            {"name": "Qwen2.5-VL 7B Instruct", "description": "Advanced Qwen vision-language and image editing model.", "size": "15GB", "type": "checkpoint", "link": "Qwen/Qwen2.5-VL-7B-Instruct"},
            {"name": "FLUX.1 Dev GGUF (Unsloth)", "description": "Optimized 12B FLUX.1 Developer GGUF model.", "size": "Variable", "type": "checkpoint", "link": "unsloth/FLUX.1-dev-GGUF"},
            {"name": "FLUX.1 Schnell GGUF (Unsloth)", "description": "Optimized 4-step FLUX.1 Schnell GGUF model.", "size": "Variable", "type": "checkpoint", "link": "unsloth/FLUX.1-schnell-GGUF"},
            {"name": "FLUX.2 Dev GGUF (Unsloth)", "description": "Optimized 32B FLUX.2 Developer GGUF model.", "size": "Variable", "type": "checkpoint", "link": "unsloth/FLUX.2-dev-GGUF"},
            {"name": "FLUX.2 Klein 9B GGUF (Unsloth)", "description": "Optimized 9B FLUX.2 Klein GGUF model.", "size": "Variable", "type": "checkpoint", "link": "unsloth/FLUX.2-klein-9B-GGUF"},
            {"name": "FLUX.1 Kontext Dev GGUF (Unsloth)", "description": "12B in-context image editing GGUF model.", "size": "Variable", "type": "checkpoint", "link": "unsloth/FLUX.1-Kontext-dev-GGUF"},
            {"name": "Qwen Image Edit 2511 GGUF (Unsloth)", "description": "Qwen 20B image editing and instruction GGUF model.", "size": "Variable", "type": "checkpoint", "link": "unsloth/Qwen-Image-Edit-2511-GGUF"},
            ]

    def get_zoo_model_config(self, index_or_name: Union[int, str]) -> Dict[str, Any]:
        """
        Returns the optimized configuration dictionary for a given model from the zoo.
        Accepts either the integer index or the string key/name.
        """
        zoo = self.get_zoo()
        item = None
        if isinstance(index_or_name, int):
            if 0 <= index_or_name < len(zoo):
                item = zoo[index_or_name]
        else:
            name_lower = index_or_name.lower()
            item = next((x for x in zoo if x["name"].lower() == name_lower or x["link"].lower() == name_lower), None)

        if not item:
            raise ValueError(f"Model '{index_or_name}' not found in the Diffusers TTI Zoo.")

        link = item["link"]
        config = {
            "model_name": link,
            "width": 1024,
            "height": 1024,
            "seed": -1,
        }

        # ── Apply optimized defaults based on model family ──
        if "flux.2" in link.lower() or "flux2" in link.lower() or "flux.1" in link.lower() or "flux1" in link.lower():
            config.update({
                "num_inference_steps": 4 if "schnell" in link.lower() or "klein" in link.lower() else 50,
                "guidance_scale": 1.0 if "schnell" in link.lower() or "klein" in link.lower() else 3.5,
                "torch_dtype_str": "bfloat16",
            })
            if "gguf" in link.lower():
                config.update({
                    "quant_backend": "gguf",
                    "quant_level": "Q4_K_M",
                    "allow_patterns": ["*Q4_K_M.gguf", "*.json", "*.txt"]
                })
        elif "qwen" in link.lower():
            config.update({
                "num_inference_steps": 40,
                "guidance_scale": 1.0,
                "torch_dtype_str": "bfloat16",
            })
            if "gguf" in link.lower():
                config.update({
                    "quant_backend": "gguf",
                    "quant_level": "Q4_K_M",
                    "allow_patterns": ["*Q4_K_M.gguf", "*.json", "*.txt"]
                })
        elif "stable-diffusion-xl" in link.lower() or "sdxl" in link.lower():
            config.update({
                "num_inference_steps": 4 if "turbo" in link.lower() else 30,
                "guidance_scale": 0.0 if "turbo" in link.lower() else 7.5,
                "torch_dtype_str": "float16",
            })
        elif "stable-diffusion-v1-5" in link.lower() or "pix2pix" in link.lower():
            config.update({
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "torch_dtype_str": "float16",
            })

        return config

    def download_from_zoo(self, index: int, progress_callback: Callable[[dict], None] = None) -> dict:
        zoo = self.get_zoo()
        if index < 0 or index >= len(zoo):
            msg = "Index out of bounds"
            ASCIIColors.error(msg)
            return {"status": False, "message": msg}
        item = zoo[index]
        return self.pull_model(item["link"], progress_callback=progress_callback)

    def set_settings(self, settings: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> bool:
        self.ensure_server_is_running(True)
            # Normalize settings from list of dicts to a single dict if needed
        parsed_settings = settings if isinstance(settings, dict) else {s["name"]: s["value"] for s in settings if "name" in s and "value" in s}
        response = self._post_json_request("/set_settings", data=parsed_settings)
        return response.json().get("success", False)

    def ps(self) -> List[dict]:
        try:
            return self._get_request("/ps").json()
        except Exception:
            return [{"error": "Could not connect to server to get process status."}]

    def pull_model(self, model_name: str, local_name: Optional[str] = None, allow_patterns: Optional[List[str]] = None, progress_callback: Callable[[dict], None] = None) -> dict:
        """
        Pulls a model from Hugging Face or URL via the server.
        """
        payload = {}
        if model_name.startswith("http") and "huggingface.co" not in model_name:
             # Assume direct file URL if not huggingface repo url (roughly)
             if model_name.endswith(".safetensors"):
                payload["safetensors_url"] = model_name
             else:
                payload["hf_id"] = model_name 
        else:
             # Clean up URL if provided as https://huggingface.co/publisher/model
             if "huggingface.co/" in model_name:
                 model_name = model_name.split("huggingface.co/")[-1]
             payload["hf_id"] = model_name

        if local_name:
            payload["local_name"] = local_name
        if allow_patterns:
            payload["allow_patterns"] = allow_patterns

        try:
            if progress_callback:
                progress_callback({"status": "starting", "message": f"Sending pull request for {model_name}..."})

            ASCIIColors.info(f"Sending pull request for {model_name}...")
            # Use a very long timeout as downloads can be huge (GBs)
            response = requests.post(f"{self.base_url}/pull_model", json=payload, timeout=7200) 
            response.raise_for_status()
            
            msg = "Model pulled successfully."
            ASCIIColors.success(msg)
            if progress_callback:
                progress_callback({"status": "success", "message": msg, "completed": 100, "total": 100})
            return {"status": True, "message": msg}
        except Exception as e:
            error_msg = f"Failed to pull model: {e}"
            if hasattr(e, 'response') and e.response:
                 error_msg += f" Server response: {e.response.text}"
            ASCIIColors.error(error_msg)
            if progress_callback:
                progress_callback({"status": "error", "message": error_msg})
            return {"status": False, "message": error_msg}

    def upgrade_diffusers(self, progress_callback: Callable[[dict], None] = None) -> dict:
        """
        Upgrades the diffusers library in the virtual environment.
        """
        try:
            if progress_callback:
                progress_callback({"status": "starting", "message": "Upgrading diffusers..."})

            ASCIIColors.info("Upgrading diffusers from GitHub...")
            if sys.platform == "win32":
                python_executable = self.venv_dir / "Scripts" / "python.exe"
            else:
                python_executable = self.venv_dir / "bin" / "python"

            subprocess.check_call([
                str(python_executable), "-m", "pip", "install", "--upgrade", 
                "git+https://github.com/huggingface/diffusers.git"
            ])
            msg = "Diffusers upgraded successfully."
            ASCIIColors.success(msg)
            ASCIIColors.info("Please restart the application/server to apply changes.")
            
            if progress_callback:
                progress_callback({"status": "success", "message": msg})
            return {"status": True, "message": msg}
        except Exception as e:
            error_msg = f"Failed to upgrade diffusers: {e}"
            ASCIIColors.error(error_msg)
            if progress_callback:
                progress_callback({"status": "error", "message": error_msg})
            return {"status": False, "message": error_msg}


    def get_server_logs(self) -> str:
        """
        Returns the last N lines of the server log, where N is defined by server_log_depth.
        """
        log_file_path = self.models_path / "diffusers_server.log"
        if not log_file_path.exists():
            return "No diffusers server log file found."

        try:
            with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                depth = getattr(self, "server_log_depth", 500)
                tail_lines = lines[-depth:] if len(lines) > depth else lines
                return "\n".join(line.rstrip() for line in tail_lines)
        except Exception as e:
            return f"Failed to read diffusers server logs: {e}"

    def reinstall_dependencies(self):
        """
        Re‑install the Python packages required by the Diffusers server.

        This method looks for a ``requirements.txt`` file located in the
        same directory as this ``__init__.py``.  It then runs:

            ``python -m pip install -r requirements.txt``

        using the **same interpreter** that runs the current process,
        ensuring that the correct virtual environment is targeted.

        Returns
        -------
        dict
            ``{'status': bool, 'message': str}`` – ``status`` is ``True`` on
            success, ``False`` otherwise.  ``message`` contains a short
            description or the error that occurred.
        """
        try:
            self.install_server_dependencies()
            return {
                "status": True,
                "message": "Dependencies reinstalled successfully.",
            }

        except Exception as e:
            trace_exception(e)
            return {"status": False, "message": str(e)}


    def __del__(self):
        # The client destructor does not stop the server,
        # as it is a shared resource for all worker processes.
        pass
