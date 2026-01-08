import time
import sys
import subprocess
from pathlib import Path
import requests
import pipmaster as pm
from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import ASCIIColors
from filelock import FileLock

BindingName = "GGUFDiffusion"

class GGUFDiffusion(LollmsTTIBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name=BindingName, **kwargs)
        self.server_port = 8182
        self.base_url = f"http://localhost:{self.server_port}"
        self.server_process = None
        self.ensure_server_running()

    # --- Abstract Method Implementations ---

    def get_settings(self):
        """Returns the current configuration settings."""
        return self.config

    def set_settings(self, settings):
        """Updates the configuration settings."""
        self.config.update(settings)
        return True

    def list_models(self):
        """
        Lists available models.
        Since we rely on a file path in settings, we return the configured path/ID.
        """
        gguf_path = self.config.get('gguf_path', '')
        base_id = self.config.get('base_model_id', 'black-forest-labs/FLUX.1-dev')
        model_name = Path(gguf_path).name if gguf_path else "No GGUF Selected"
        return [f"{base_id} :: {model_name}"]

    def list_services(self):
        """Lists available services."""
        return ["local_server"]

    def edit_image(self, image, prompt, negative_prompt="", width=1024, height=1024, **kwargs):
        """
        Edit image / Image-to-Image generation.
        """
        ASCIIColors.warning("Image editing (Img2Img) is not yet supported in GGUF Diffusion binding.")
        return None

    # --- Core Logic ---

    def ensure_server_running(self):
        """Ensures the inference server is running."""
        try:
            requests.get(f"{self.base_url}/health", timeout=1)
        except (requests.ConnectionError, requests.Timeout):
            ASCIIColors.info("Starting GGUF Diffusion server...")
            self.start_server()

    def start_server(self):
        """Starts the separate Python process for the server."""
        server_dir = Path(__file__).parent / "server"
        venv_dir = server_dir / "venv"
        
        # Use pipmaster to handle dependencies in a dedicated venv
        if not pm.is_installed("fastapi"):
            ASCIIColors.info("Installing dependencies...")
            pkg = pm.PackageManager(venv_path=str(venv_dir))
            pkg.install_requirements(str(server_dir / "requirements.txt"))

        python_executable = str(venv_dir / "bin" / "python") if sys.platform != "win32" else str(venv_dir / "Scripts" / "python.exe")
        script_path = str(server_dir / "main.py")

        with FileLock(str(server_dir / "server.lock")):
            self.server_process = subprocess.Popen(
                [python_executable, script_path],
                cwd=str(server_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Wait for startup
            for _ in range(30):
                try:
                    requests.get(f"{self.base_url}/health", timeout=1)
                    ASCIIColors.success("Server started successfully.")
                    return
                except:
                    time.sleep(1)
            ASCIIColors.error("Failed to start GGUF Diffusion server.")

    def install_dependencies(self):
        server_dir = Path(__file__).parent / "server"
        venv_dir = server_dir / "venv"
        pkg = pm.PackageManager(venv_path=str(venv_dir))
        pkg.install_requirements(str(server_dir / "requirements.txt"))
        return {"status": True, "message": "Dependencies installed."}

    def _download_model(self, url, filename):
        """Helper to download models with progress indication."""
        try:
            # Use self.app.lollms_paths if available, otherwise fallback
            if hasattr(self, 'app') and hasattr(self.app, 'lollms_paths'):
                models_dir = self.app.lollms_paths.personal_models_path / "gguf_diffusion"
            else:
                models_dir = Path.home() / "Documents" / "lollms" / "models" / "gguf_diffusion"
            
            models_dir.mkdir(parents=True, exist_ok=True)
            output_path = models_dir / filename

            if output_path.exists():
                stat = output_path.stat()
                if stat.st_size > 1024 * 1024 * 100: # Simple check: > 100MB likely valid
                    return {"status": True, "message": f"Model {filename} already exists at {output_path}."}
                else:
                    ASCIIColors.warning(f"File exists but seems too small. Re-downloading: {filename}")

            ASCIIColors.info(f"Downloading {filename} from {url}...")
            ASCIIColors.info(f"Target path: {output_path}")

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192 * 4):
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Minimal progress logging to avoid spamming console
                        if total_size > 0 and downloaded % (1024 * 1024 * 100) < 8192 * 4: # Log every ~100MB
                             progress = (downloaded / total_size) * 100
                             ASCIIColors.info(f"Downloading {filename}: {progress:.1f}%")

            ASCIIColors.success(f"Successfully downloaded {filename}")
            return {"status": True, "message": f"Download complete. Path: {output_path}"}
        except Exception as e:
            ASCIIColors.error(f"Failed to download {filename}: {e}")
            return {"status": False, "message": str(e)}

    def install_flux(self):
        """Downloads Flux.1-Dev Q4_0."""
        url = "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_0.gguf"
        return self._download_model(url, "flux1-dev-Q4_0.gguf")

    def install_qwen_image(self):
        """Downloads Qwen Image Q4_K_S."""
        url = "https://huggingface.co/city96/Qwen-Image-gguf/resolve/main/qwen-image-Q4_K_S.gguf"
        return self._download_model(url, "qwen-image-Q4_K_S.gguf")

    def load_flux(self):
        """Pre-loads the model via the server."""
        device_cfg = self.config.get("device", "auto")
        payload = {
            "base_model_id": self.config.get("base_model_id", "black-forest-labs/FLUX.1-dev"),
            "gguf_path": self.config.get("gguf_path", ""),
            "device": device_cfg,
            "vram_policy": self.config.get("vram_policy", "regular")
        }
        
        try:
            # High timeout because loading/patching takes time
            response = requests.post(f"{self.base_url}/load", json=payload, timeout=300)
            if response.status_code == 200:
                ASCIIColors.success("Flux model loaded successfully.")
                return {"status": True, "message": "Model loaded successfully."}
            else:
                return {"status": False, "message": f"Load failed: {response.text}"}
        except Exception as e:
            return {"status": False, "message": f"Error: {e}"}

    def generate_image(self, prompt, negative_prompt="", width=1024, height=1024, **kwargs):
        """Generates an image using the local server."""
        
        # Resolve Device
        device_cfg = self.config.get("device", "auto")
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": kwargs.get("steps", 20),
            "guidance_scale": kwargs.get("guidance_scale", 3.5),
            "base_model_id": self.config.get("base_model_id", "black-forest-labs/FLUX.1-dev"),
            "gguf_path": self.config.get("gguf_path", ""),
            "device": device_cfg,
            "vram_policy": self.config.get("vram_policy", "regular")
        }

        try:
            response = requests.post(f"{self.base_url}/generate", json=payload, stream=True)
            if response.status_code == 200:
                # Assuming server returns raw image bytes
                return self.process_image(response.content, **kwargs)
            else:
                ASCIIColors.error(f"Generation failed: {response.text}")
                return None
        except Exception as e:
            ASCIIColors.error(f"Error calling generation server: {e}")
            return None

    def __del__(self):
        if self.server_process:
            self.server_process.terminate()
