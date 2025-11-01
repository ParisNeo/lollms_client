# bindings/llamacpp_server/binding.py
import json
import os
import pprint
import re
import socket
import subprocess
import sys
import threading
import time
import tempfile
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict, Any, Set
import base64
from lollms_client.lollms_discussion import LollmsDiscussion
import requests # For HTTP client
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT

from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm
import platform

# --- Multi-process locking for registry ---
# On Windows, we need msvcrt, on POSIX, fcntl
try:
    if platform.system() == "Windows":
        import msvcrt
    else:
        import fcntl
except ImportError:
    # This might happen in some restricted environments.
    # The binding will fall back to thread-safety only.
    msvcrt = fcntl = None


class FileLock:
    def __init__(self, lock_file_path):
        self.lock_file_path = lock_file_path
        self.lock_file = None
        self._is_windows = platform.system() == "Windows"

    def __enter__(self):
        self.lock_file = open(self.lock_file_path, 'w')
        if self._is_windows and msvcrt:
            msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_LOCK, 1)
        elif not self._is_windows and fcntl:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_file:
            if self._is_windows and msvcrt:
                self.lock_file.seek(0)
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            elif not self._is_windows and fcntl:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()
            self.lock_file = None

# --- End multi-process locking ---


# Ensure llama-cpp-binaries, requests, pillow, and psutil are installed
pm.ensure_packages(["requests", "pillow", "psutil"]) # pillow for dummy image in test, psutil for multi-process management
if not pm.is_installed("llama-cpp-binaries"):
    def install_llama_cpp():
        system = platform.system()
        python_version_simple = f"py{sys.version_info.major}{sys.version_info.minor}"  # e.g. py310 for 3.10

        version_tag = "v0.56.0"
        cuda_suffix = "+cu124"

        if system == "Windows":
            # Try version-specific URL first
            url = f"https://github.com/oobabooga/llama-cpp-binaries/releases/download/{version_tag}/llama_cpp_binaries-{version_tag.lstrip('v')}{cuda_suffix}-{python_version_simple}-none-win_amd64.whl"
            # Fallback to generic py3 if version-specific doesn't exist
            fallback_url = f"https://github.com/oobabooga/llama-cpp-binaries/releases/download/{version_tag}/llama_cpp_binaries-{version_tag.lstrip('v')}{cuda_suffix}-py3-none-win_amd64.whl"
        elif system == "Linux":
            # Try version-specific URL first
            url = f"https://github.com/oobabooga/llama-cpp-binaries/releases/download/{version_tag}/llama_cpp_binaries-{version_tag.lstrip('v')}{cuda_suffix}-{python_version_simple}-none-linux_x86_64.whl"
            # Fallback to generic py3 if version-specific doesn't exist
            fallback_url = f"https://github.com/oobabooga/llama-cpp-binaries/releases/download/{version_tag}/llama_cpp_binaries-{version_tag.lstrip('v')}{cuda_suffix}-py3-none-linux_x86_64.whl"
        else:
            ASCIIColors.error(f"Unsupported OS for precompiled llama-cpp-binaries: {system}. "
                            "You might need to set 'llama_server_binary_path' in the binding config "
                            "to point to a manually compiled llama.cpp server binary.")
            return False


        ASCIIColors.info(f"Attempting to install llama-cpp-binaries from: {url}")
        try:
            pm.install(url)
        except Exception as e:
            ASCIIColors.warning(f"Failed to install specific version from {url}: {e}")
            ASCIIColors.info(f"Attempting fallback URL: {fallback_url}")
            try:
                pm.install(fallback_url)
            except Exception as e_fallback:
                ASCIIColors.error(f"Failed to install from fallback URL {fallback_url}: {e_fallback}")
                ASCIIColors.error("Please try installing llama-cpp-binaries manually, e.g., 'pip install llama-cpp-python[server]' or from a wheel.")

    install_llama_cpp()

try:
    import llama_cpp_binaries
    import psutil
except ImportError:
    ASCIIColors.error("llama-cpp-binaries or psutil package not found. Please ensure they are installed.")
    ASCIIColors.error("You can try: pip install llama-cpp-python[server] psutil")
    llama_cpp_binaries = None
    psutil = None


# --- Predefined patterns ---
_QUANT_COMPONENTS_SET: Set[str] = {
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q2_K_S", "Q3_K_S", "Q4_K_S", "Q5_K_S",
    "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q3_K_L", "Q2_K_XS", "Q3_K_XS", "Q4_K_XS", "Q5_K_XS", "Q6_K_XS",
    "Q2_K_XXS", "Q3_K_XXS", "Q4_K_XXS", "Q5_K_XXS", "Q6_K_XXS", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "F16", "FP16", "F32", "FP32", "BF16", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS", "IQ3_M_K", "IQ3_S_K", "IQ4_XS_K", "IQ4_NL_K",
    "I8", "I16", "I32", "ALL_F32", "MOSTLY_F16", "MOSTLY_Q4_0", "MOSTLY_Q4_1", "MOSTLY_Q5_0", "MOSTLY_Q5_1",
    "MOSTLY_Q8_0", "MOSTLY_Q2_K", "MOSTLY_Q3_K_S", "MOSTLY_Q3_K_M", "MOSTLY_Q3_K_L",
    "MOSTLY_Q4_K_S", "MOSTLY_Q4_K_M", "MOSTLY_Q5_K_S", "MOSTLY_Q5_K_M", "MOSTLY_Q6_K",
    "MOSTLY_IQ1_S", "MOSTLY_IQ1_M", "MOSTLY_IQ2_XXS", "MOSTLY_IQ2_XS", "MOSTLY_IQ2_S", "MOSTLY_IQ2_M",
    "MOSTLY_IQ3_XXS", "MOSTLY_IQ3_S", "MOSTLY_IQ3_M", "MOSTLY_IQ4_NL", "MOSTLY_IQ4_XS"
}
_MODEL_NAME_SUFFIX_COMPONENTS_SET: Set[str] = {
    "instruct", "chat", "GGUF", "HF", "ggml", "pytorch", "AWQ", "GPTQ", "EXL2",
    "base", "cont", "continue", "ft", "v0.1", "v0.2", "v1.0", "v1.1", "v1.5", "v1.6", "v2.0"
}
_ALL_REMOVABLE_COMPONENTS: List[str] = sorted(
    list(_QUANT_COMPONENTS_SET.union(_MODEL_NAME_SUFFIX_COMPONENTS_SET)), key=len, reverse=True
)

def get_gguf_model_base_name(file_path_or_name: Union[str, Path]) -> str:
    if isinstance(file_path_or_name, str): p = Path(file_path_or_name)
    elif isinstance(file_path_or_name, Path): p = file_path_or_name
    else: raise TypeError(f"Input must be a string or Path object. Got: {type(file_path_or_name)}")
    name_part = p.stem if p.suffix.lower() == ".gguf" else p.name
    if name_part.lower().endswith(".gguf"): name_part = name_part[:-5]
    while True:
        original_name_part_len = len(name_part)
        stripped_in_this_iteration = False
        for component in _ALL_REMOVABLE_COMPONENTS:
            component_lower = component.lower()
            for separator in [".", "-", "_"]:
                pattern_to_check = f"{separator}{component_lower}"
                if name_part.lower().endswith(pattern_to_check):
                    name_part = name_part[:-(len(pattern_to_check))]
                    stripped_in_this_iteration = True; break
            if stripped_in_this_iteration: break
        if not stripped_in_this_iteration or not name_part: break
    while name_part and name_part[-1] in ['.', '-', '_']: name_part = name_part[:-1]
    return name_part

# --- Global Server Registry (File-based for multi-process support) ---

class ServerRegistry:
    def __init__(self):
        self.registry_dir = Path(tempfile.gettempdir()) / "lollms_llamacpp_servers"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self.lock_file = self.registry_dir / "registry.lock"
        self.my_pid = os.getpid()

    def _is_pid_running(self, pid: int) -> bool:
        if psutil is None: return True # Conservative default if psutil is missing
        return psutil.pid_exists(pid)

    def _read_registry(self) -> Dict[str, Any]:
        if not self.registry_file.exists():
            return {}
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write_registry(self, data: Dict[str, Any]):
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _clean_stale_entries(self, registry_data: Dict[str, Any]) -> bool:
        """Cleans stale servers and clients. Returns True if changes were made."""
        changed = False
        # Clean dead servers
        dead_servers = [k for k, v in registry_data.items() if not self._is_pid_running(v['pid'])]
        for key in dead_servers:
            ASCIIColors.warning(f"Registry Cleaner: Found dead server process (PID: {registry_data[key]['pid']}). Removing entry {key}.")
            del registry_data[key]
            changed = True

        # Clean dead clients from living servers
        for key, server_info in list(registry_data.items()):
            dead_clients = [pid for pid in server_info.get('client_pids', []) if not self._is_pid_running(pid)]
            if dead_clients:
                ASCIIColors.warning(f"Registry Cleaner: Found dead client PIDs {dead_clients} for server {key}. Cleaning up.")
                server_info['client_pids'] = [pid for pid in server_info['client_pids'] if pid not in dead_clients]
                server_info['ref_count'] = len(server_info['client_pids'])
                changed = True

            # If a server has no clients left after cleanup, it's an orphan. Remove it.
            if server_info['ref_count'] <= 0:
                ASCIIColors.warning(f"Registry Cleaner: Server {key} (PID: {server_info['pid']}) has no clients left. Shutting it down.")
                try:
                    p = psutil.Process(server_info['pid'])
                    p.terminate()
                    p.wait(timeout=5)
                except psutil.NoSuchProcess: pass
                except Exception as e: ASCIIColors.error(f"Error terminating orphaned server PID {server_info['pid']}: {e}")
                del registry_data[key]
                changed = True
        
        return changed

    def get_server(self, server_key: str) -> Optional[Dict[str, Any]]:
        with FileLock(self.lock_file):
            registry = self._read_registry()
            self._clean_stale_entries(registry) # Always clean before read
            server_info = registry.get(server_key)
            if server_info:
                self._write_registry(registry) # Write back changes from cleaning
            return server_info

    def register_new_server(self, server_key: str, pid: int, port: int):
        with FileLock(self.lock_file):
            registry = self._read_registry()
            # Clean just in case something happened between server start and registration
            self._clean_stale_entries(registry)
            
            registry[server_key] = {
                "pid": pid, "port": port,
                "ref_count": 1, "client_pids": [self.my_pid]
            }
            self._write_registry(registry)
            ASCIIColors.info(f"Process {self.my_pid} registered new server {server_key} (PID: {pid}, Port: {port})")

    def increment_ref_count(self, server_key: str):
        with FileLock(self.lock_file):
            registry = self._read_registry()
            self._clean_stale_entries(registry)
            
            server_info = registry.get(server_key)
            if server_info:
                if self.my_pid not in server_info['client_pids']:
                    server_info['client_pids'].append(self.my_pid)
                    server_info['ref_count'] = len(server_info['client_pids'])
                    self._write_registry(registry)
                    ASCIIColors.info(f"Process {self.my_pid} attached to server {server_key}. New ref_count: {server_info['ref_count']}")
            else:
                ASCIIColors.warning(f"Process {self.my_pid} tried to attach to non-existent server {server_key}.")

    def decrement_ref_count(self, server_key: str):
        with FileLock(self.lock_file):
            registry = self._read_registry()
            made_changes = self._clean_stale_entries(registry)
            
            server_info = registry.get(server_key)
            if server_info:
                if self.my_pid in server_info['client_pids']:
                    server_info['client_pids'].remove(self.my_pid)
                    server_info['ref_count'] = len(server_info['client_pids'])
                    made_changes = True
                    ASCIIColors.info(f"Process {self.my_pid} detached from server {server_key}. New ref_count: {server_info['ref_count']}")

                if server_info['ref_count'] <= 0:
                    ASCIIColors.info(f"Last client (PID: {self.my_pid}) detached. Shutting down server {server_key} (PID: {server_info['pid']}).")
                    try:
                        p = psutil.Process(server_info['pid'])
                        p.terminate()
                        p.wait(timeout=10)
                    except psutil.NoSuchProcess:
                        ASCIIColors.warning(f"Server process {server_info['pid']} was already gone.")
                    except Exception as e:
                        ASCIIColors.error(f"Error terminating server process {server_info['pid']}: {e}")
                    del registry[server_key]
            
            if made_changes:
                self._write_registry(registry)

BindingName = "LlamaCppServerBinding"
DEFAULT_LLAMACPP_SERVER_HOST = "127.0.0.1"

class LlamaCppServerProcess:
    def __init__(self, 
                 model_path: Union[str, Path], 
                 clip_model_path: Optional[Union[str, Path]] = None, 
                 server_binary_path: Optional[Union[str, Path]]=None, 
                 server_args: Dict[str, Any]={},
                 process_pid: Optional[int]=None, # PID if we are attaching to existing process
                 port: Optional[int]=None,
                 ):
        """Initialize the Llama.cpp server process wrapper.
           Can either start a new process or wrap an existing one.
        """
        self.model_path = Path(model_path)
        self.clip_model_path = Path(clip_model_path) if clip_model_path else None
        
        if server_binary_path:
            self.server_binary_path = Path(server_binary_path)
        elif llama_cpp_binaries:
            self.server_binary_path = Path(llama_cpp_binaries.get_binary_path())
        else:
            raise FileNotFoundError("llama_cpp_binaries not found and no server_binary_path provided.")

        self.port: Optional[int] = port
        self.pid: Optional[int] = process_pid
        self.server_args = server_args
        # The actual subprocess.Popen object. Will be None if this instance is just a client to a server started by another process.
        self.process: Optional[subprocess.Popen] = None 
        self.session = requests.Session()
        self.host = self.server_args.get("host",DEFAULT_LLAMACPP_SERVER_HOST)
        self.base_url: Optional[str] = f"http://{self.host}:{self.port}" if self.port else None
        self.is_healthy = False
        self._stderr_lines: List[str] = []
        self._stderr_thread: Optional[threading.Thread] = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if self.clip_model_path and not self.clip_model_path.exists():
            ASCIIColors.warning(f"Clip model file '{self.clip_model_path}' not found. Vision features may not work or may use a different auto-detected clip model.")
        if not self.server_binary_path.exists():
            raise FileNotFoundError(f"Llama.cpp server binary not found: {self.server_binary_path}")

    def attach(self):
        """Attaches to an already running process by checking its health."""
        if not self.pid or not self.port:
            raise ValueError("Cannot attach without PID and port.")
        self.base_url = f"http://{self.host}:{self.port}"
        health_url = f"{self.base_url}/health"
        try:
            response = self.session.get(health_url, timeout=5)
            if response.status_code == 200 and response.json().get("status") == "ok":
                self.is_healthy = True
                ASCIIColors.green(f"Successfully attached to Llama.cpp server on port {self.port} (PID: {self.pid}).")
                return
        except requests.exceptions.RequestException as e:
            ASCIIColors.warning(f"Failed to attach to server on port {self.port}: {e}")
            self.is_healthy = False
            raise ConnectionError(f"Could not connect to existing server at {health_url}")

    def _filter_stderr(self, stderr_pipe):
        try:
            for line in iter(stderr_pipe.readline, ''):
                if line:
                    self._stderr_lines.append(line.strip())
                    if len(self._stderr_lines) > 50: self._stderr_lines.pop(0)
                    if "llama_model_loaded" in line or "error" in line.lower() or "failed" in line.lower():
                        ASCIIColors.debug(f"[LLAMA_SERVER_STDERR:{self.port}] {line.strip()}")
                    elif "running on port" in line: # Server startup message
                        ASCIIColors.info(f"[LLAMA_SERVER_STDERR:{self.port}] {line.strip()}")
        except ValueError: pass
        except Exception as e: ASCIIColors.warning(f"Exception in stderr filter thread for port {self.port}: {e}")

    def start(self, port_to_use: int):
        self.port = port_to_use
        self.base_url = f"http://{self.host}:{self.port}"
        
        cmd = [
            str(self.server_binary_path),
            "--model", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
        ]

        arg_map = {
            "n_ctx": "--ctx-size", "n_gpu_layers": "--gpu-layers", "main_gpu": "--main-gpu",
            "tensor_split": "--tensor-split", "use_mmap": (lambda v: ["--no-mmap"] if not v else []),
            "use_mlock": (lambda v: ["--mlock"] if v else []), "seed": "--seed",
            "n_batch": "--batch-size", "n_threads": "--threads", "n_threads_batch": "--threads-batch",
            "rope_scaling_type": "--rope-scaling", "rope_freq_base": "--rope-freq-base",
            "rope_freq_scale": "--rope-freq-scale",
            "embedding": (lambda v: ["--embedding"] if v else []),
            "verbose": (lambda v: ["--verbose"] if v else []),
            "chat_template": "--chat-template",
            "parallel_slots": "--parallel", # Number of parallel processing slots
        }
        
        if self.clip_model_path: # This should be the actual path resolved by the binding
            cmd.extend(["--mmproj", str(self.clip_model_path)])

        for key, cli_arg in arg_map.items():
            val = self.server_args.get(key)
            if val is not None:
                if callable(cli_arg): cmd.extend(cli_arg(val))
                else: cmd.extend([cli_arg, str(val)])
        
        extra_cli_flags = self.server_args.get("extra_cli_flags", [])
        if isinstance(extra_cli_flags, str): extra_cli_flags = extra_cli_flags.split()
        cmd.extend(extra_cli_flags)

        ASCIIColors.info(f"Starting Llama.cpp server ({' '.join(cmd)})")
        
        env = os.environ.copy()
        if os.name == 'posix' and self.server_binary_path.parent != Path('.'):
            lib_path_str = str(self.server_binary_path.parent.resolve())
            current_ld_path = env.get('LD_LIBRARY_PATH', '')
            env['LD_LIBRARY_PATH'] = f"{lib_path_str}:{current_ld_path}" if current_ld_path else lib_path_str

        try:
            self.process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
            self.pid = self.process.pid
        except Exception as e:
            ASCIIColors.error(f"Failed to start llama.cpp server process on port {self.port}: {e}"); trace_exception(e); raise

        self._stderr_thread = threading.Thread(target=self._filter_stderr, args=(self.process.stderr,), daemon=True)
        self._stderr_thread.start()

        health_url = f"{self.base_url}/health"
        max_wait_time = self.server_args.get("server_startup_timeout", 60)
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            if self.process.poll() is not None:
                stderr_output = "\n".join(self._stderr_lines[-10:])
                raise RuntimeError(f"Llama.cpp server (port {self.port}) terminated unexpectedly (exit code {self.process.poll()}) during startup. Stderr:\n{stderr_output}")
            try:
                response = self.session.get(health_url, timeout=2)
                if response.status_code == 200 and response.json().get("status") == "ok":
                    self.is_healthy = True
                    ASCIIColors.green(f"Llama.cpp server started successfully on port {self.port} (PID: {self.pid}).")
                    return
            except requests.exceptions.ConnectionError: time.sleep(1)
            except Exception as e: ASCIIColors.warning(f"Health check for port {self.port} failed: {e}"); time.sleep(1)
        
        self.is_healthy = False
        self.shutdown() 
        stderr_output = "\n".join(self._stderr_lines[-10:])
        raise TimeoutError(f"Llama.cpp server failed to become healthy on port {self.port} within {max_wait_time}s. Stderr:\n{stderr_output}")

    def shutdown(self):
        """ This method only shuts down a server if this instance owns the Popen object.
            The actual termination for multi-process is handled by the ServerRegistry. """
        self.is_healthy = False
        if self.process:
            ASCIIColors.info(f"Shutting down owned Llama.cpp server process (PID: {self.process.pid} on port {self.port})...")
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                ASCIIColors.warning(f"Llama.cpp server (port {self.port}) did not terminate gracefully, killing...")
                self.process.kill()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired: ASCIIColors.error(f"Failed to kill llama.cpp server process (port {self.port}).")
            except Exception as e: ASCIIColors.error(f"Error during server shutdown (port {self.port}): {e}")
            finally:
                self.process = None
                if self._stderr_thread and self._stderr_thread.is_alive(): self._stderr_thread.join(timeout=1)
                ASCIIColors.info(f"Llama.cpp server on port {self.port} shut down.")


class LlamaCppServerBinding(LollmsLLMBinding):
    DEFAULT_SERVER_ARGS = {
        "n_gpu_layers": 0, "n_ctx": 128000, "n_batch": 512,
        "embedding": False, "verbose": False, "server_startup_timeout": 120,
        "parallel_slots": 4, # Default parallel slots for server
        "stop_sequences": ["<|im_start|>"], # Default stop sequences
    }

    def __init__(self, **kwargs):
        super().__init__(BindingName, **kwargs)
        if llama_cpp_binaries is None or psutil is None: 
            raise ImportError("llama-cpp-binaries and psutil packages are required.")

        self.registry = ServerRegistry()
        models_path = kwargs.get("models_path", Path(__file__).parent/"models")
        self.models_path = Path(models_path)
        self.initial_model_name_preference: Optional[str] = kwargs.get("model_name")
        self.user_provided_model_name: Optional[str] = kwargs.get("model_name")
        self.initial_clip_model_name_preference: Optional[str] = kwargs.get("clip_model_name") 
        self._model_path_map: Dict[str, Path] = {}
        self._scan_models()
        self.default_completion_format =  kwargs.get("default_completion_format", ELF_COMPLETION_FORMAT.Chat)
        self.server_args = {**self.DEFAULT_SERVER_ARGS, **(kwargs.get("config") or {}), **kwargs}
        self.server_binary_path = self._get_server_binary_path()
        
        self.current_model_path: Optional[Path] = None 
        self.clip_model_path: Optional[Path] = None
        self.server_process: Optional[LlamaCppServerProcess] = None
        self.port: Optional[int] = None
        self.server_key: Optional[str] = None

        ASCIIColors.info("LlamaCppServerBinding initialized. Server will start on-demand with first generation call.")

    def _get_server_binary_path(self) -> Path:
        custom_path_str = self.server_args.get("llama_server_binary_path")
        if custom_path_str:
            custom_path = Path(custom_path_str)
            if custom_path.exists() and custom_path.is_file():
                ASCIIColors.info(f"Using custom llama.cpp server binary: {custom_path}"); return custom_path
            else: ASCIIColors.warning(f"Custom binary '{custom_path_str}' not found. Falling back.")
        if llama_cpp_binaries:
            bin_path_str = llama_cpp_binaries.get_binary_path()
            if bin_path_str:
                bin_path = Path(bin_path_str)
                if bin_path.exists() and bin_path.is_file():
                    ASCIIColors.info(f"Using binary from llama-cpp-binaries: {bin_path}"); return bin_path
        raise FileNotFoundError("Llama.cpp server binary not found.")

    def _resolve_model_path(self, model_name_or_path: str) -> Path:
        """
        Resolves a model name or path to a full Path object.
        It prioritizes the internal map, then checks for absolute/relative paths,
        and rescans the models directory as a fallback.
        """
        if model_name_or_path in self._model_path_map:
            return self._model_path_map[model_name_or_path]
        model_p = Path(model_name_or_path)
        if model_p.is_absolute() and model_p.exists(): return model_p
        path_in_models_dir = self.models_path / model_name_or_path
        if path_in_models_dir.exists(): return path_in_models_dir
        self._scan_models()
        if model_name_or_path in self._model_path_map:
            return self._model_path_map[model_name_or_path]
        raise FileNotFoundError(f"Model '{model_name_or_path}' not found.")

    def _find_available_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0)); return s.getsockname()[1]

    def _release_server_instance(self):
        if self.server_process and self.server_key:
            self.registry.decrement_ref_count(self.server_key)
        self.server_process = None
        self.port = None
        self.server_key = None
        self.current_model_path = None
        self.clip_model_path = None

    def load_model(self, model_name_or_path: str) -> bool:
        self.user_provided_model_name = model_name_or_path
        try:
            resolved_model_path = self._resolve_model_path(model_name_or_path)
        except Exception as ex:
            trace_exception(ex); return False

        final_clip_model_path: Optional[Path] = None
        if self.initial_clip_model_name_preference:
            p_clip_pref = Path(self.initial_clip_model_name_preference)
            if p_clip_pref.is_absolute() and p_clip_pref.exists(): final_clip_model_path = p_clip_pref
            elif (self.models_path / p_clip_pref).exists(): final_clip_model_path = self.models_path / p_clip_pref
            else: ASCIIColors.warning(f"Specified clip model '{self.initial_clip_model_name_preference}' not found.")

        if not final_clip_model_path:
            base_name = get_gguf_model_base_name(resolved_model_path.stem)
            potential_paths = [
                resolved_model_path.parent / f"{base_name}.mmproj",
                resolved_model_path.parent / f"mmproj-{base_name}.gguf",
                self.models_path / f"{base_name}.mmproj",
                self.models_path / f"mmproj-{base_name}.gguf",
            ]
            for p_clip in potential_paths:
                if p_clip.exists(): final_clip_model_path = p_clip; break
        
        final_clip_model_path_str = str(final_clip_model_path) if final_clip_model_path else "None"
        new_server_key = f"{resolved_model_path}|{final_clip_model_path_str}"

        if self.server_process and self.server_key == new_server_key and self.server_process.is_healthy:
            ASCIIColors.info(f"Model '{model_name_or_path}' is already loaded. No change.")
            return True

        if self.server_process and self.server_key != new_server_key:
            self._release_server_instance()
        
        # Check registry for an existing server
        existing_server_info = self.registry.get_server(new_server_key)
        if existing_server_info:
            ASCIIColors.info(f"Found existing server for {new_server_key} in registry (PID: {existing_server_info['pid']}, Port: {existing_server_info['port']}). Attaching...")
            try:
                self.server_process = LlamaCppServerProcess(
                    model_path=resolved_model_path, clip_model_path=final_clip_model_path,
                    process_pid=existing_server_info['pid'], port=existing_server_info['port'],
                    server_args=self.server_args
                )
                self.server_process.attach() # This verifies health
                self.port = self.server_process.port
                self.current_model_path = resolved_model_path
                self.clip_model_path = final_clip_model_path
                self.server_key = new_server_key
                self.registry.increment_ref_count(new_server_key)
                return True
            except Exception as e:
                ASCIIColors.error(f"Failed to attach to existing server: {e}. It might be stale. Will attempt to start a new one.")
                self.registry.decrement_ref_count(new_server_key) # Clean up failed attach
        
        # Start a new server
        ASCIIColors.info(f"No existing server found for {new_server_key}. Starting a new one.")
        self.current_model_path = resolved_model_path
        self.clip_model_path = final_clip_model_path
        self.server_key = new_server_key

        try:
            new_port = self._find_available_port()
            current_server_args = self.server_args.copy()
            if "parallel_slots" not in current_server_args or current_server_args["parallel_slots"] <=0:
                current_server_args["parallel_slots"] = self.DEFAULT_SERVER_ARGS["parallel_slots"]

            new_server = LlamaCppServerProcess(
                model_path=self.current_model_path, clip_model_path=self.clip_model_path,
                server_binary_path=self.server_binary_path, server_args=current_server_args
            )
            new_server.start(port_to_use=new_port)

            if new_server.is_healthy:
                self.server_process = new_server
                self.port = new_port
                self.registry.register_new_server(self.server_key, new_server.pid, new_port)
                ASCIIColors.green(f"New server {self.server_key} started and registered.")
                return True
            else:
                return False
        except Exception as e:
            ASCIIColors.error(f"Failed to start new server for '{model_name_or_path}': {e}"); trace_exception(e)
            self._release_server_instance()
            return False

    def unload_model(self):
        if self.server_process:
            self._release_server_instance()
        else:
            ASCIIColors.info("Unload called, but no server was active for this binding instance.")

    def _ensure_server_is_running(self) -> bool:
        """
        Checks if the server is healthy. If not, it attempts to load the configured model.
        Returns True if the server is healthy and ready, False otherwise.
        """
        if self.server_process and self.server_process.is_healthy:
            return True

        ASCIIColors.info("Server is not running. Attempting to start on-demand...")
        
        model_to_load = self.user_provided_model_name or self.initial_model_name_preference
        
        if not model_to_load:
            self._scan_models()
            available_models = self.list_models()
            if not available_models:
                ASCIIColors.error("No model specified and no GGUF models found in models path.")
                return False
            
            model_to_load = available_models[0]['name']
            ASCIIColors.info(f"No model was specified. Automatically selecting the first available model: '{model_to_load}'")

        if self.load_model(model_to_load):
            return True
        else:
            ASCIIColors.error(f"Automatic model load for '{model_to_load}' failed.")
            return False

    def _get_request_url(self, endpoint: str) -> str:
        return f"{self.server_process.base_url}{endpoint}"

    def _prepare_generation_payload(self, prompt: str, system_prompt: str = "", n_predict: Optional[int] = None,
                                   temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9,
                                   repeat_penalty: float = 1.1, repeat_last_n: Optional[int] = 64,
                                   seed: Optional[int] = None, stream: bool = False, use_chat_format: bool = True,
                                   images: Optional[List[str]] = None,
                                   stop_sequences: Optional[List[str]] = None,
                                   split:Optional[bool]=False,
                                   user_keyword:Optional[str]="!@>user:",
                                   ai_keyword:Optional[str]="!@>assistant:", 
                                   **extra_params) -> Dict:
        payload_params = {
            "temperature": self.server_args.get("temperature", 0.7), "top_k": self.server_args.get("top_k", 40),
            "top_p": self.server_args.get("top_p", 0.9), "repeat_penalty": self.server_args.get("repeat_penalty", 1.1),
            "repeat_last_n": self.server_args.get("repeat_last_n", 64), "mirostat": self.server_args.get("mirostat_mode", 0),
            "mirostat_tau": self.server_args.get("mirostat_tau", 5.0), "mirostat_eta": self.server_args.get("mirostat_eta", 0.1),
        }
        if "grammar_string" in self.server_args and self.server_args["grammar_string"]:
             payload_params["grammar"] = self.server_args["grammar_string"]

        payload_params.update({"temperature": temperature, "top_k": top_k, "top_p": top_p, "repeat_penalty": repeat_penalty, "repeat_last_n": repeat_last_n})
        if n_predict is not None: payload_params['n_predict'] = n_predict
        if seed is not None: payload_params['seed'] = seed

        # --- Handle stop sequences ---
        all_stop_sequences = set(self.server_args.get("stop_sequences", []))
        if stop_sequences:
            all_stop_sequences.update(stop_sequences)
        if all_stop_sequences:
            payload_params['stop'] = list(all_stop_sequences)
        # --- End stop sequences ---

        payload_params = {k: v for k, v in payload_params.items() if v is not None}
        payload_params.update(extra_params)

        if use_chat_format and self.default_completion_format == ELF_COMPLETION_FORMAT.Chat:
            messages = []
            if system_prompt and system_prompt.strip(): messages.append({"role": "system", "content": system_prompt})
            user_content: Union[str, List[Dict[str, Any]]] = prompt
            if split:
                messages += self.split_discussion(user_content,user_keyword=user_keyword, ai_keyword=ai_keyword)
            else:
                messages.append({"role": "user", "content": user_content})
            if images and self.clip_model_path:
                image_parts = []
                for img_path in images:
                    try:
                        with open(img_path, "rb") as image_file: encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        image_type = Path(img_path).suffix[1:].lower() or "png"; image_type = "jpeg" if image_type == "jpg" else image_type
                        image_parts.append({"type": "image_url", "image_url": {"url": f"data:image/{image_type};base64,{encoded_string}"}})
                    except Exception as ex: trace_exception(ex)
                messages[-1]["content"] =[{"type": "text", "text": messages[-1]["content"]}] +  image_parts # type: ignore
            final_payload = {"messages": messages, "stream": stream, **payload_params}
            if 'n_predict' in final_payload: final_payload['max_tokens'] = final_payload.pop('n_predict')
            return final_payload
        else:
            full_prompt = f"{system_prompt}\n\nUSER: {prompt}\nASSISTANT:" if system_prompt and system_prompt.strip() else prompt
            final_payload = {"prompt": full_prompt, "stream": stream, **payload_params}
            if images and self.clip_model_path:
                image_data_list = []
                for i, img_path in enumerate(images):
                    try:
                        with open(img_path, "rb") as image_file: encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        image_data_list.append({"data": encoded_string, "id": i + 10})
                    except Exception as e_img: ASCIIColors.error(f"Could not encode image {img_path}: {e_img}")
                if image_data_list: final_payload["image_data"] = image_data_list
            return final_payload


    def generate_text(self,
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
                     n_predict: Optional[int] = None,
                     stream: Optional[bool] = None,
                     temperature: float = 0.7,
                     top_k: int = 40,
                     top_p: float = 0.9,
                     repeat_penalty: float = 1.1,
                     repeat_last_n: int = 64,
                     seed: Optional[int] = None,
                     n_threads: Optional[int] = None,
                     ctx_size: int | None = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     stop_sequences: Optional[List[str]] = None,
                     split:Optional[bool]=False,
                     user_keyword:Optional[str]="!@>user:",
                     ai_keyword:Optional[str]="!@>assistant:", 
                     **generation_kwargs
                     ) -> Union[str, dict]:
        
        if not self._ensure_server_is_running():
            return {"status": False, "error": "Llama.cpp server could not be started. Please check model configuration and logs."}

        _use_chat_format = True
        payload = self._prepare_generation_payload(
            prompt=prompt, system_prompt=system_prompt, n_predict=n_predict,
            temperature=temperature if temperature is not None else self.server_args.get("temperature",0.7),
            top_k=top_k if top_k is not None else self.server_args.get("top_k",40),
            top_p=top_p if top_p is not None else self.server_args.get("top_p",0.9),
            repeat_penalty=repeat_penalty if repeat_penalty is not None else self.server_args.get("repeat_penalty",1.1),
            repeat_last_n=repeat_last_n if repeat_last_n is not None else self.server_args.get("repeat_last_n",64),
            seed=seed if seed is not None else self.server_args.get("seed", -1), stream=stream,
            use_chat_format=_use_chat_format, images=images,
            stop_sequences=stop_sequences,
            split= split, user_keyword=user_keyword, ai_keyword=ai_keyword, **generation_kwargs
        )
        endpoint = "/v1/chat/completions" if _use_chat_format else "/completion"
        request_url = self._get_request_url(endpoint)
        
        full_response_text = ""
        try:
            response = self.server_process.session.post(request_url, json=payload, stream=stream, timeout=self.server_args.get("generation_timeout", 300))
            response.raise_for_status()
            if stream:
                for line in response.iter_lines():
                    if not line: continue
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '): line_str = line_str[6:]
                    if line_str == '[DONE]': break
                    try:
                        chunk_data = json.loads(line_str)
                        chunk_content = (chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '') if _use_chat_format
                                         else chunk_data.get('content', ''))
                        if chunk_content:
                            full_response_text += chunk_content
                            if streaming_callback and not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                ASCIIColors.info("Streaming callback requested stop."); response.close(); break
                        if chunk_data.get('stop', False) or chunk_data.get('stopped_eos',False) or chunk_data.get('stopped_limit',False): break
                    except json.JSONDecodeError: ASCIIColors.warning(f"Failed to decode JSON stream chunk: {line_str}"); continue
                return full_response_text
            else:
                response_data = response.json()
                return response_data.get('choices', [{}])[0].get('message', {}).get('content', '') if _use_chat_format \
                       else response_data.get('content','')
        except requests.exceptions.RequestException as e:
            error_message = f"Llama.cpp server request error: {e}"
            if e.response is not None:
                try: error_details = e.response.json(); error_message += f" - Details: {error_details.get('error', e.response.text)}"
                except json.JSONDecodeError: error_message += f" - Response: {e.response.text[:200]}"
            ASCIIColors.error(error_message)
            return {"status": False, "error": error_message, "details": str(e.response.text if e.response else "No response text")}
        except Exception as ex:
            error_message = f"Llama.cpp generation error: {str(ex)}"; trace_exception(ex)
            return {"status": False, "error": error_message}

    def chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = None,
             stream: Optional[bool] = None,
             temperature: float = 0.7,
             top_k: int = 40,
             top_p: float = 0.9,
             repeat_penalty: float = 1.1,
             repeat_last_n: int = 64,
             seed: Optional[int] = None,
             n_threads: Optional[int] = None,
             ctx_size: Optional[int] = None,
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
             stop_sequences: Optional[List[str]] = None,
             **generation_kwargs
             ) -> Union[str, dict]:

        if not self._ensure_server_is_running():
            return {"status": "error", "message": "Llama.cpp server could not be started. Please check model configuration and logs."}

        messages = discussion.export("openai_chat", branch_tip_id)
        payload = {
            "messages": messages, "max_tokens": n_predict, "temperature": temperature,
            "top_k": top_k, "top_p": top_p, "repeat_penalty": repeat_penalty,
            "seed": seed, "stream": stream, **generation_kwargs
        }
        
        all_stop_sequences = set(self.server_args.get("stop_sequences", []))
        if stop_sequences:
            all_stop_sequences.update(stop_sequences)
        if all_stop_sequences:
            payload['stop'] = list(all_stop_sequences)

        payload = {k: v for k, v in payload.items() if v is not None}
        
        endpoint = "/v1/chat/completions"
        request_url = self._get_request_url(endpoint)
        full_response_text = ""

        try:
            response = self.server_process.session.post(request_url, json=payload, stream=stream, timeout=self.server_args.get("generation_timeout", 300))
            response.raise_for_status()

            if stream:
                for line in response.iter_lines():
                    if not line: continue
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '): line_str = line_str[6:]
                    if line_str == '[DONE]': break
                    try:
                        chunk_data = json.loads(line_str)
                        choices = chunk_data.get('choices', [{}])
                        if choices and len(choices)>0:
                            chunk_content = choices[0].get('delta', {}).get('content', '')
                            if chunk_content:
                                full_response_text += chunk_content
                                if streaming_callback and not streaming_callback(chunk_content, MSG_TYPE.MSG_TYPE_CHUNK):
                                    ASCIIColors.info("Streaming callback requested stop.")
                                    response.close()
                                    break
                    except json.JSONDecodeError:
                        ASCIIColors.warning(f"Failed to decode JSON stream chunk: {line_str}")
                        continue
                return full_response_text
            else:
                response_data = response.json()
                return response_data.get('choices', [{}])[0].get('message', {}).get('content', '')

        except requests.exceptions.RequestException as e:
            error_message = f"Llama.cpp server request error: {e}"
            if e.response is not None:
                try:
                    error_details = e.response.json()
                    error_message += f" - Details: {error_details.get('error', e.response.text)}"
                except json.JSONDecodeError:
                    error_message += f" - Response: {e.response.text[:200]}"
            ASCIIColors.error(error_message)
            return {"status": "error", "message": error_message}
        except Exception as ex:
            error_message = f"Llama.cpp generation error: {str(ex)}"
            trace_exception(ex)
            return {"status": "error", "message": error_message}
        
    def tokenize(self, text: str) -> List[int]:
        if not self._ensure_server_is_running(): return []
        try:
            response = self.server_process.session.post(self._get_request_url("/tokenize"), json={"content": text})
            response.raise_for_status(); return response.json().get("tokens", [])
        except Exception as e: ASCIIColors.error(f"Tokenization error: {e}"); trace_exception(e); return []

    def detokenize(self, tokens: List[int]) -> str:
        if not self._ensure_server_is_running(): return ""
        try:
            response = self.server_process.session.post(self._get_request_url("/detokenize"), json={"tokens": tokens})
            response.raise_for_status(); return response.json().get("content", "")
        except Exception as e: ASCIIColors.error(f"Detokenization error: {e}"); trace_exception(e); return ""

    def count_tokens(self, text: str) -> int: return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        if not self._ensure_server_is_running(): return []
        if not self.server_args.get("embedding"): 
            ASCIIColors.warning("Embedding not enabled in server_args. Please set 'embedding' to True in config."); return []
        try:
            payload = {"input": text}; request_url = self._get_request_url("/v1/embeddings")
            response = self.server_process.session.post(request_url, json=payload)
            if response.status_code == 404: # Fallback
                request_url = self._get_request_url("/embedding")
                response = self.server_process.session.post(request_url, json={"content": text})
            response.raise_for_status(); data = response.json()
            if "data" in data and isinstance(data["data"], list) and "embedding" in data["data"][0]: return data["data"][0]["embedding"]
            elif "embedding" in data and isinstance(data["embedding"], list): return data["embedding"]
            else: raise ValueError(f"Unexpected embedding response: {data}")
        except requests.exceptions.RequestException as e:
            err_msg = f"Embedding request error: {e}"; 
            if e.response: err_msg += f" - {e.response.text[:200]}"
            ASCIIColors.error(err_msg)
            return []
        except Exception as ex: 
            trace_exception(ex); ASCIIColors.error(f"Embedding failed: {str(ex)}")
            return []
        
    def get_model_info(self) -> dict:
        is_loaded = self.server_process is not None and self.server_process.is_healthy
        info = {
            "name": self.binding_name,
            "user_provided_model_name": self.user_provided_model_name,
            "model_path": str(self.current_model_path) if self.current_model_path else "Not loaded",
            "clip_model_path": str(self.clip_model_path) if self.clip_model_path else "N/A",
            "loaded": is_loaded,
            "server_args": self.server_args, "port": self.port if self.port else "N/A",
            "server_key": str(self.server_key) if self.server_key else "N/A",
        }
        if is_loaded:
            try:
                props_resp = self.server_process.session.get(self._get_request_url("/props"), timeout=5).json()
                info.update({
                    "server_n_ctx": props_resp.get("default_generation_settings",{}).get("n_ctx"),
                    "server_chat_format": props_resp.get("chat_format"),
                    "server_clip_model_from_props": props_resp.get("mmproj"),
                })
            except Exception: pass 
            
            is_llava = self.clip_model_path is not None or \
                       (info.get("server_clip_model_from_props") is not None) or \
                       ("llava" in self.current_model_path.name.lower() if self.current_model_path else False)
            info["supports_vision"] = is_llava
            info["supports_structured_output"] = self.server_args.get("grammar_string") is not None
        return info

    def _scan_models(self):
        self._model_path_map = {}
        if not self.models_path.exists() or not self.models_path.is_dir():
            ASCIIColors.warning(f"Models path does not exist or is not a directory: {self.models_path}")
            return

        all_paths = list(self.models_path.rglob("*.gguf"))
        filenames_count = {}
        for path in all_paths:
            if path.is_file():
                filenames_count[path.name] = filenames_count.get(path.name, 0) + 1

        for model_file in all_paths:
            if model_file.is_file():
                relative_path_str = str(model_file.relative_to(self.models_path).as_posix())
                if filenames_count[model_file.name] > 1:
                    unique_name = relative_path_str
                else:
                    unique_name = model_file.name
                self._model_path_map[unique_name] = model_file
        
        ASCIIColors.info(f"Scanned {len(self._model_path_map)} models from {self.models_path}.")

    def list_models(self) -> List[Dict[str, Any]]:
        self._scan_models()
        models_found = []
        for unique_name, model_path in self._model_path_map.items():
            models_found.append({
                'name': unique_name, 'model_name': model_path.name,
                'path': str(model_path), 'size': model_path.stat().st_size
            })
        return sorted(models_found, key=lambda x: x['name'])
    
    def __del__(self):
        self.unload_model()

    def get_ctx_size(self, model_name: Optional[str] = None) -> Optional[int]:
        if model_name is None:
            model_name = self.user_provided_model_name or self.initial_model_name_preference
            if not model_name and self.current_model_path:
                model_name = self.current_model_path.name

        if model_name is None: 
            ASCIIColors.warning("Cannot determine context size without a model name.")
            return None

        known_contexts = {
            'llama3.1': 131072, 'llama3.2': 131072, 'llama3.3': 131072, 'llama3': 8192,
            'llama2': 4096, 'mixtral8x22b': 65536, 'mixtral': 32768, 'mistral': 32768,
            'gemma3': 131072, 'gemma2': 8192, 'gemma': 8192, 'phi3': 131072, 'phi2': 2048,
            'phi': 2048, 'qwen2.5': 131072, 'qwen2': 32768, 'qwen': 8192,
            'codellama': 16384, 'codegemma': 8192, 'deepseek-coder-v2': 131072,
            'deepseek-coder': 16384, 'deepseek-v2': 131072, 'deepseek-llm': 4096,
            'yi1.5': 32768, 'yi': 4096, 'command-r': 131072, 'wizardlm2': 32768,
            'wizardlm': 16384, 'zephyr': 65536, 'vicuna': 2048, 'falcon': 2048,
            'starcoder': 8192, 'stablelm': 4096, 'orca2': 4096, 'orca': 4096,
            'dolphin': 32768, 'openhermes': 8192,
        }
        normalized_model_name = model_name.lower().strip()
        sorted_base_models = sorted(known_contexts.keys(), key=len, reverse=True)

        for base_name in sorted_base_models:
            if base_name in normalized_model_name:
                context_size = known_contexts[base_name]
                ASCIIColors.info(f"Using hardcoded context size for '{model_name}' based on '{base_name}': {context_size}")
                return context_size

        ASCIIColors.warning(f"Context size not found for model '{model_name}' in the hardcoded list.")
        return None

if __name__ == '__main__':
    # NOTE: This test block is designed for a single-process scenario to verify basic functionality.
    # Testing the multi-process capabilities requires a separate script that launches multiple
    # instances of a test program using this binding. The logic here, however, will now use the
    # new file-based registry system.
    full_streamed_text = ""
    ASCIIColors.yellow("Testing LlamaCppServerBinding...")

    try:
        models_path_str = os.environ.get("LOLLMS_MODELS_PATH", str(Path(__file__).parent / "test_models"))
        model_name_str = os.environ.get("LOLLMS_TEST_MODEL_GGUF", "tinyllama-1.1b-chat-v1.0.Q2_K.gguf")
        
        models_path = Path(models_path_str)
        models_path.mkdir(parents=True, exist_ok=True)
        test_model_path = models_path / model_name_str
        
        primary_model_available = test_model_path.exists()
        if not primary_model_available:
            ASCIIColors.warning(f"Test model {test_model_path} not found. Please place a GGUF model there or set env vars.")
            ASCIIColors.warning("Some tests will be skipped.")

    except Exception as e:
        ASCIIColors.error(f"Error setting up test paths: {e}"); trace_exception(e)
        sys.exit(1)

    binding_config = {
        "n_gpu_layers": 0, "n_ctx": 512, "embedding": True,
        "verbose": False, "server_startup_timeout": 180, "parallel_slots": 2,
        "stop_sequences": ["<|user|>", "\nUSER:"], # Example default stop sequences
    }

    active_binding1: Optional[LlamaCppServerBinding] = None
    active_binding2: Optional[LlamaCppServerBinding] = None
    
    try:
        if primary_model_available:
            # --- Test 1: Auto-start server on first generation call ---
            ASCIIColors.cyan("\n--- Test 1: Auto-start server with specified model name ---")
            active_binding1 = LlamaCppServerBinding(
                model_name=model_name_str, models_path=str(models_path), config=binding_config
            )
            ASCIIColors.info("Binding1 initialized. No server should be running yet.")
            ASCIIColors.info(f"Initial model info: {json.dumps(active_binding1.get_model_info(), indent=2)}")

            prompt_text = "What is the capital of France?"
            generated_text = active_binding1.generate_text(
                prompt_text, 
                system_prompt="Concise expert.", 
                n_predict=20, 
                stream=False,
                stop_sequences=["Paris"] # Test per-call stop sequence
            )
            
            if isinstance(generated_text, str) and "Paris" not in generated_text: # Should stop *before* generating Paris
                ASCIIColors.green(f"SUCCESS: Auto-start generation with stop sequence successful. Response: '{generated_text}'")
            else:
                ASCIIColors.error(f"FAILURE: Auto-start generation failed or stop sequence ignored. Response: {generated_text}")

            ASCIIColors.info(f"Model info after auto-start: {json.dumps(active_binding1.get_model_info(), indent=2)}")
            if not active_binding1.server_process or not active_binding1.server_process.is_healthy:
                 raise RuntimeError("Server for binding1 did not seem to start correctly.")

            # --- Test 2: Server reuse with a second binding ---
            ASCIIColors.cyan("\n--- Test 2: Server reuse with a second binding ---")
            active_binding2 = LlamaCppServerBinding(
                model_name=model_name_str, models_path=str(models_path), config=binding_config
            )
            generated_text_b2 = active_binding2.generate_text("Ping", n_predict=5, stream=False)
            if isinstance(generated_text_b2, str):
                ASCIIColors.green(f"SUCCESS: Binding2 generation successful. Response: {generated_text_b2}")
            else:
                 ASCIIColors.error(f"FAILURE: Binding2 generation failed. Response: {generated_text_b2}")

            if active_binding1.port != active_binding2.port:
                ASCIIColors.error("FAILURE: Bindings for the same model are using different ports! Server sharing failed.")
            else:
                ASCIIColors.green("SUCCESS: Both bindings use the same server port. Server sharing works.")

            # --- Test 3: Unload and auto-reload ---
            ASCIIColors.cyan("\n--- Test 3: Unload and auto-reload ---")
            active_binding1.unload_model()
            ASCIIColors.info("Binding1 unloaded. Ref count should be 1, server still up for binding2.")
            
            generated_text_reloaded = active_binding1.generate_text("Test reload", n_predict=5, stream=False)
            if isinstance(generated_text_reloaded, str):
                 ASCIIColors.green(f"SUCCESS: Generation after reload successful. Response: {generated_text_reloaded}")
            else:
                 ASCIIColors.error(f"FAILURE: Generation after reload failed. Response: {generated_text_reloaded}")

            if active_binding1.port != active_binding2.port:
                ASCIIColors.error("FAILURE: Port mismatch after reload.")
            else:
                 ASCIIColors.green("SUCCESS: Correctly re-used same server after reload.")

        else:
            ASCIIColors.warning("\n--- Primary model not available, skipping most tests ---")

        # --- Test 4: Initialize with model_name=None and auto-find ---
        ASCIIColors.cyan("\n--- Test 4: Initialize with model_name=None and auto-find ---")
        unspecified_binding = LlamaCppServerBinding(model_name=None, models_path=str(models_path), config=binding_config)
        gen_unspec = unspecified_binding.generate_text("Ping", n_predict=5, stream=False)
        if primary_model_available:
            if isinstance(gen_unspec, str):
                ASCIIColors.green(f"SUCCESS: Auto-find generation successful. Response: {gen_unspec}")
                ASCIIColors.info(f"Model auto-selected: {unspecified_binding.user_provided_model_name}")
            else:
                ASCIIColors.error(f"FAILURE: Auto-find generation failed. Response: {gen_unspec}")
        else: # If no models, this should fail gracefully
            if isinstance(gen_unspec, dict) and 'error' in gen_unspec:
                ASCIIColors.green("SUCCESS: Correctly failed to generate when no models are available.")
            else:
                ASCIIColors.error(f"FAILURE: Incorrect behavior when no models are available. Response: {gen_unspec}")
        
    except Exception as e_main:
        ASCIIColors.error(f"An unexpected error occurred during testing: {e_main}")
        trace_exception(e_main)
    finally:
        ASCIIColors.cyan("\n--- Unloading Models and Stopping Servers ---")
        if active_binding1: active_binding1.unload_model(); ASCIIColors.info("Binding1 unloaded.")
        if active_binding2: active_binding2.unload_model(); ASCIIColors.info("Binding2 unloaded.")
        # Any other bindings will be cleaned up by __del__ on exit
        
        registry = ServerRegistry()
        with FileLock(registry.lock_file):
            final_state = registry._read_registry()
            if not final_state or not any(c for s in final_state.values() for c in s.get('client_pids',[])):
                ASCIIColors.green("All servers shut down correctly and registry is empty or has no clients.")
                if final_state: registry._write_registry({}) # Clean up for next run
            else:
                ASCIIColors.warning(f"Warning: Registry is not empty after tests: {final_state}")
                registry._clean_stale_entries(final_state)
                registry._write_registry(final_state)
                ASCIIColors.info("Forced a final registry cleanup.")

    ASCIIColors.yellow("\nLlamaCppServerBinding test finished.")