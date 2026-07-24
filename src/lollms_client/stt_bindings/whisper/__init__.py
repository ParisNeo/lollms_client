import os
import sys
import base64
import subprocess
import threading
import time
import json
import socket
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from ascii_colors import trace_exception, ASCIIColors

try:
    import pipmaster as pm
except ImportError:
    print("FATAL: pipmaster is not installed. Please install it using: pip install pipmaster")
    sys.exit(1)

try:
    from filelock import FileLock, Timeout
except ImportError:
    print("FATAL: The 'filelock' library is required. Please install it by running: pip install filelock")
    sys.exit(1)

import requests
from lollms_client.lollms_stt_binding import LollmsSTTBinding

BindingName = "WhisperSTTBinding"

class WhisperSTTBinding(LollmsSTTBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name="whisper")
        self.config = kwargs
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 9633)
        self.auto_start_server = kwargs.get("auto_start_server", False)
        self.wait_for_server = kwargs.get("wait_for_server", False)
        self.server_process = None
        self.base_url = f"http://{self.host}:{self.port}"
        self.binding_root = Path(__file__).parent
        self.server_dir = self.binding_root / "server"
        
        self.venv_dir = Path(kwargs.get("venv_path", "./venv/stt_whisper_venv")).resolve()
        self.cache_dir = Path(kwargs.get("cache_dir", "./data/stt_models/whisper")).resolve()
        
        self.venv_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        if self.auto_start_server:
            self.ensure_server_is_running(self.wait_for_server)

    def is_server_running(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/status", timeout=4)
            if response.status_code == 200 and response.json().get("status") == "running":
                return True
        except requests.exceptions.RequestException:
            return False
        return False

    def _is_port_available(self, host: str, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return True
        except OSError:
            return False

    def ensure_server_is_running(self, wait=False):
        ASCIIColors.info("Attempting to start or connect to the Whisper server...")
        if self.is_server_running():
            ASCIIColors.green("Whisper Server is already running and responsive.")
            return

        original_port = self.port
        while not self._is_port_available(self.host, self.port):
            ASCIIColors.warning(f"Port {self.port} is busy. Trying next port...")
            self.port += 1
            self.base_url = f"http://{self.host}:{self.port}"

        if self.port != original_port:
            ASCIIColors.info(f"Selected new port {self.port} for Whisper server.")

        self.start_server(wait)

    def install_server_dependencies(self):
        ASCIIColors.info(f"Setting up virtual environment in: {self.venv_dir}")
        pm_v = pm.PackageManager(venv_path=str(self.venv_dir), create_if_not_exist=True)

        ASCIIColors.info(f"Installing server dependencies")
        pm_v.ensure_packages(["requests", "uvicorn", "fastapi", "python-multipart", "filelock"])
        pm_v.ensure_packages(["ascii_colors", "pipmaster"])
        pm_v.ensure_packages(["tqdm", "numpy", "pillow"])
        
        ASCIIColors.info(f"Installing pytorch")
        torch_index_url = None
        if sys.platform == "win32":
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
                ASCIIColors.green("NVIDIA GPU detected. Installing CUDA-enabled PyTorch.")
                torch_index_url = "https://download.pytorch.org/whl/cu126"
            except (FileNotFoundError, subprocess.CalledProcessError):
                ASCIIColors.yellow("`nvidia-smi` not found or failed. Installing standard PyTorch.")

        pm_v.ensure_packages(["torch", "torchaudio"], index_url=torch_index_url)
        pm_v.ensure_packages(["openai-whisper"])
        ASCIIColors.green("Server dependencies are satisfied.")

    def start_server(self, wait: bool = True, timeout_s: int = 40):
        def _start_server_background():
            lock_path = self.venv_dir / "whisper_server.lock"
            lock = FileLock(lock_path)

            try:
                with lock.acquire(timeout=0):
                    server_script = self.server_dir / "main.py"
                    venv_cfg = self.venv_dir / "pyvenv.cfg"

                    if not venv_cfg.exists():
                        ASCIIColors.warning("Invalid or missing virtual environment. Reinstalling...")
                        self.install_server_dependencies()

                    if sys.platform == "win32":
                        python_executable = self.venv_dir / "Scripts" / "python.exe"
                    else:
                        python_executable = self.venv_dir / "bin" / "python"

                    if not python_executable.exists():
                        raise RuntimeError(f"Python executable not found in venv: {python_executable}.")

                    command = [
                        str(python_executable),
                        str(server_script),
                        "--host", self.host,
                        "--port", str(self.port),
                        "--cache-dir", str(self.cache_dir.resolve())
                    ]

                    log_file_path = self.cache_dir / "whisper_server.log"
                    log_f = open(log_file_path, "w", encoding="utf-8")

                    try:
                        creationflags = 0
                        if sys.platform == "win32":
                            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                        
                        self.server_process = subprocess.Popen(
                            command,
                            stdout=log_f,
                            stderr=subprocess.STDOUT,
                            creationflags=creationflags
                        )
                    except Exception as popen_err:
                        log_f.close()
                        raise RuntimeError(f"Failed to execute subprocess: {popen_err}")
                    finally:
                        log_f.close()

                    ASCIIColors.info(f"Whisper server launched on http://{self.host}:{self.port}")

                    if wait:
                        start_time = time.time()
                        while True:
                            if self.server_process.poll() is not None:
                                error_tail = "No log data available."
                                try:
                                    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as err_log:
                                        lines = err_log.readlines()
                                        error_tail = "".join(lines[-30:]) if lines else "Log file is empty."
                                except Exception:
                                    pass
                                raise RuntimeError(
                                    f"Whisper server process terminated unexpectedly with code {self.server_process.returncode}.\n"
                                    f"--- Server Log Tail ---\n{error_tail}"
                                )

                            if self.is_server_running():
                                ASCIIColors.success("Whisper server is ready.")
                                return

                            elapsed = time.time() - start_time
                            if elapsed >= timeout_s:
                                raise TimeoutError(f"Server failed to start within {timeout_s} seconds.")
                            time.sleep(1)
            except Exception as ex:
                ASCIIColors.error(f"Failed to start Whisper server: {ex}")
                raise

        thread = threading.Thread(target=_start_server_background, daemon=True)
        thread.start()
        if wait:
            thread.join()

    def _post_json_request(self, endpoint: str, data: Optional[dict] = None) -> requests.Response:
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, json=data, timeout=3600)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to communicate with Whisper server at {url}. Error: {e}")
            if hasattr(e, 'response') and e.response:
                try:
                    err_detail = e.response.json().get('detail', e.response.text)
                except json.JSONDecodeError:
                    err_detail = e.response.text
                ASCIIColors.error(f"Server response: {err_detail}")
                raise RuntimeError(f"Whisper server error: {err_detail}") from e
            raise RuntimeError("Communication with the Whisper server failed.") from e

    def _get_request(self, endpoint: str, params: Optional[dict] = None) -> requests.Response:
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            ASCIIColors.error(f"Failed to communicate with Whisper server at {url}.")
            raise RuntimeError("Communication with the Whisper server failed.") from e

    def transcribe_audio(self, audio_source: Union[str, Path, bytes], model: Optional[str] = None, **kwargs) -> str:
        if self.auto_start_server:
            self.ensure_server_is_running(True)
            
        if not self.is_server_running() and not self.auto_start_server:
             raise RuntimeError("Whisper server is not running and auto_start_server is False.")

        try:
            if isinstance(audio_source, (str, Path)):
                audio_file = Path(audio_source)
                if not audio_file.exists():
                    raise FileNotFoundError(f"Audio file not found at: {audio_source}")
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
            elif isinstance(audio_source, bytes):
                audio_bytes = audio_source
            else:
                raise ValueError("audio_source must be str, Path, or bytes")

            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            payload = {
                "audio_b64": audio_b64,
                "model_name": model or self.config.get("model_name", "base"),
                "language": kwargs.get("language"),
                "task": kwargs.get("task", "transcribe"),
                "fp16": kwargs.get("fp16")
            }
            
            response = self._post_json_request("/transcribe", data=payload)
            return response.json().get("text", "")
            
        except Exception as e:
            ASCIIColors.error(f"Whisper transcription failed: {e}")
            trace_exception(e)
            raise Exception(f"Whisper transcription error: {e}") from e

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        return ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]

    def ps(self) -> List[dict]:
        try:
            return self._get_request("/ps").json()
        except Exception:
            return [{"error": "Could not connect to server to get process status."}]

    def __del__(self):
        pass
