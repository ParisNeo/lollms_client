import os
import sys
import requests
import subprocess
import time
import base64
from pathlib import Path
from typing import Optional, List

try:
    from filelock import FileLock, Timeout
except ImportError:
    print("FATAL: The 'filelock' library is required. Please install it: pip install filelock")
    sys.exit(1)

from lollms_client.lollms_tts_binding import LollmsTTSBinding
from ascii_colors import ASCIIColors

BindingName = "FishSpeechClientBinding"

class FishSpeechClientBinding(LollmsTTSBinding):
    """
    Client binding for Fish Speech (OpenAudio S1) TTS server.
    Provides state-of-the-art multilingual voice synthesis with zero-shot cloning.
    """
    def __init__(self, **kwargs):
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')

        self.config = kwargs
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 8080)
        self.auto_start_server = kwargs.get("auto_start_server", False)
        self.compile = kwargs.get("compile", True)
        self.device = kwargs.get("device", "auto")
        self.model_name = kwargs.get("model_name", "fishaudio/openaudio-s1-mini")
        
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"
        self.venv_dir = Path("./venv/tts_fish_speech_venv")
        
        # Python version requirement
        self.target_python_version = "3.12"
        
        # Model paths
        self.checkpoints_dir = self.server_dir / "checkpoints"
        self.references_dir = self.server_dir / "references"

        if self.auto_start_server:
            self.ensure_server_is_running()

    def is_server_running(self) -> bool:
        """Check if the Fish Speech server is running and responsive."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            return False
        return False

    def ensure_server_is_running(self):
        """
        Ensure the Fish Speech server is running using file lock for process safety.
        """
        self.server_dir.mkdir(exist_ok=True)
        lock_path = self.server_dir / "fish_speech_server.lock"
        lock = FileLock(lock_path)

        ASCIIColors.info("Attempting to start or connect to Fish Speech server...")
        
        if self.is_server_running():
            ASCIIColors.green("Fish Speech server is already running.")
            return

        try:
            with lock.acquire(timeout=10):
                if not self.is_server_running():
                    ASCIIColors.yellow("Lock acquired. Starting Fish Speech server...")
                    self.start_server()
                    self._wait_for_server(timeout=60)
                else:
                    ASCIIColors.green("Server started by another process.")
        except Timeout:
            ASCIIColors.yellow("Waiting for another process to start the server...")
            self._wait_for_server(timeout=90)

        if not self.is_server_running():
            raise RuntimeError("Failed to start or connect to Fish Speech server.")

    def install_server_dependencies(self):
        """
        Install Fish Speech dependencies into a dedicated Python 3.10 virtual environment.
        """
        ASCIIColors.info(f"Setting up Python {self.target_python_version} environment in: {self.venv_dir}")
        
        try:
            import pipmaster as pm
        except ImportError:
            print("FATAL: pipmaster is required. Install with: pip install pipmaster")
            raise Exception("pipmaster not found")
        
        try:
            ASCIIColors.info(f"Bootstrapping portable Python {self.target_python_version}...")
            pm_instance = pm.get_pip_manager_for_version(
                self.target_python_version,
                str(self.venv_dir)
            )
            
            ASCIIColors.green(f"Portable Python {self.target_python_version} ready.")
            ASCIIColors.info(f"Using interpreter: {pm_instance.target_python_executable}")
            
        except RuntimeError as e:
            ASCIIColors.error(f"Failed to bootstrap Python {self.target_python_version}: {e}")
            raise Exception(f"Fish Speech requires Python {self.target_python_version}")
        
        # Install requirements
        requirements_file = self.server_dir / "requirements.txt"
        ASCIIColors.info("Installing Fish Speech dependencies...")
        
        success = pm_instance.ensure_requirements(str(requirements_file), verbose=True)
        if not success:
            ASCIIColors.error("Failed to install dependencies.")
            raise RuntimeError("Fish Speech dependency installation failed.")

        ASCIIColors.green("Dependencies installed successfully.")
        self._python_executable = pm_instance.target_python_executable
        
        # Download model weights
        self._download_model_weights(pm_instance)

    def _download_model_weights(self, pm_instance):
        """Download Fish Speech model weights if not present."""
        model_path = self.checkpoints_dir / self.model_name.split('/')[-1]
        
        if model_path.exists():
            ASCIIColors.info(f"Model weights found at {model_path}")
            return
        
        ASCIIColors.yellow(f"Downloading model weights for {self.model_name}...")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use huggingface-cli to download
            result = subprocess.run([
                str(self._python_executable),
                "-m", "huggingface_hub.commands.huggingface_cli",
                "download",
                self.model_name,
                "--local-dir", str(model_path)
            ], check=True, capture_output=True, text=True)
            
            ASCIIColors.green(f"Model downloaded to {model_path}")
        except subprocess.CalledProcessError as e:
            ASCIIColors.error(f"Failed to download model: {e.stderr}")
            raise RuntimeError("Model download failed.")

    def start_server(self):
        """Launch the Fish Speech API server as a background process."""
        server_script = self.server_dir / "main.py"
        if not server_script.exists():
            raise FileNotFoundError(f"Server script not found at {server_script}")

        if not self.venv_dir.exists():
            self.install_server_dependencies()
        else:
            try:
                import pipmaster as pm
                pm_instance = pm.get_pip_manager_for_version(
                    self.target_python_version,
                    str(self.venv_dir)
                )
                self._python_executable = pm_instance.target_python_executable
            except Exception as e:
                ASCIIColors.warning(f"Could not verify Python version: {e}")
                # Fallback
                if sys.platform == "win32":
                    self._python_executable = str(self.venv_dir / "Scripts" / "python.exe")
                else:
                    self._python_executable = str(self.venv_dir / "bin" / "python")

        # Prepare model path
        model_short_name = self.model_name.split('/')[-1]
        model_path = self.checkpoints_dir / model_short_name

        command = [
            str(self._python_executable),
            str(server_script),
            "--host", self.host,
            "--port", str(self.port),
            "--model-path", str(model_path),
            "--device", self.device
        ]
        
        if self.compile:
            command.append("--compile")
        
        creationflags = subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
        self.server_process = subprocess.Popen(command, creationflags=creationflags)
        ASCIIColors.info("Fish Speech server launched.")

    def _wait_for_server(self, timeout=60):
        """Wait for the server to become responsive."""
        ASCIIColors.info("Waiting for Fish Speech server...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                ASCIIColors.green("Fish Speech server is ready.")
                return
            time.sleep(3)
        raise RuntimeError("Fish Speech server failed to start within timeout.")

    def generate_audio(self, text: str, voice: Optional[str] = None, 
                      reference_text: Optional[str] = None, **kwargs) -> bytes:
        """
        Generate audio from text using Fish Speech.
        
        Args:
            text: Text to synthesize (supports emotion markers like (happy), (sad))
            voice: Path to reference audio file for voice cloning (WAV/MP3, 10-30s)
            reference_text: Transcript of reference audio (improves accuracy)
            **kwargs: Additional parameters (format, top_p, temperature, etc.)
        """
        self.ensure_server_is_running()
        
        payload = {
            "text": text,
            "reference_text": reference_text,
            "format": kwargs.get("format", "wav"),
            "top_p": kwargs.get("top_p", 0.9),
            "temperature": kwargs.get("temperature", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
            "normalize": kwargs.get("normalize", True),
            "chunk_length": kwargs.get("chunk_length", 200)
        }
        
        # Handle reference audio
        if voice:
            voice_path = Path(voice)
            if not voice_path.exists():
                # Try references directory
                voice_path = self.references_dir / voice
                if not voice_path.exists():
                    raise FileNotFoundError(f"Reference audio not found: {voice}")
            
            # Encode audio as base64
            with open(voice_path, 'rb') as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            payload["reference_audio"] = audio_base64
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/tts",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to communicate with Fish Speech server: {e}")
            raise RuntimeError("Fish Speech server communication failed.") from e

    def list_voices(self, **kwargs) -> List[str]:
        """Get available reference voices."""
        self.ensure_server_is_running()
        try:
            response = requests.get(f"{self.base_url}/list_voices")
            response.raise_for_status()
            return response.json().get("voices", [])
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to get voices: {e}")
            return []

    def list_models(self, **kwargs) -> List[str]:
        """List available Fish Speech models."""
        return [
            "fishaudio/openaudio-s1-mini",
            "fishaudio/fish-speech-1.5"
        ]