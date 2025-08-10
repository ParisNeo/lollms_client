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

# Ensure llama-cpp-binaries and requests are installed
pm.ensure_packages(["requests", "pillow"]) # pillow for dummy image in test
if not pm.is_installed("llama-cpp-binaries"):
    def install_llama_cpp():
        system = platform.system()
        python_version_simple = f"py{sys.version_info.major}" # e.g. py310 for 3.10

        # Determine CUDA suffix based on common recent versions. Adjust if needed.
        # For simplicity, we'll target a common recent CUDA version.
        # Users with specific needs might need to install manually.
        # As of late 2023/early 2024, cu121 or cu118 are common.
        # The oobabooga binaries often use +cu124 for recent builds. Let's try that.
        cuda_suffix = "+cu124" 


        if system == "Windows":
            # llama_cpp_binaries-0.14.0+cu124-py3-none-win_amd64.whl
            url = f"https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.12.0/llama_cpp_binaries-0.14.0{cuda_suffix}-{python_version_simple}-none-win_amd64.whl"
            fallback_url = "https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.14.0/llama_cpp_binaries-0.14.0+cu124-py3-none-win_amd64.whl" # Generic py3
        elif system == "Linux":
            # llama_cpp_binaries-0.14.0+cu124-py3-none-linux_x86_64.whl
            url = f"https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.14.0/llama_cpp_binaries-0.14.0{cuda_suffix}-{python_version_simple}-none-linux_x86_64.whl"
            fallback_url = "https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.14.0/llama_cpp_binaries-0.14.0+cu124-py3-none-linux_x86_64.whl" # Generic py3
        else:
            ASCIIColors.warning(f"Unsupported OS for prebuilt llama-cpp-binaries: {system}. Please install manually.")
            return

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
except ImportError:
    ASCIIColors.error("llama-cpp-binaries package not found. Please install it.")
    ASCIIColors.error("You can try: pip install llama-cpp-python[server] (for server support)")
    ASCIIColors.error("Or download a wheel from: https://github.com/oobabooga/llama-cpp-binaries/releases or https://pypi.org/project/llama-cpp-python/#files")
    llama_cpp_binaries = None


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

# --- Global Server Registry ---
_active_servers: Dict[tuple, 'LlamaCppServerProcess'] = {}
_server_ref_counts: Dict[tuple, int] = {}
_server_registry_lock = threading.Lock()

BindingName = "LlamaCppServerBinding"
DEFAULT_LLAMACPP_SERVER_HOST = "127.0.0.1"
# Port is now dynamic, this constant is less critical for direct use but good for reference.
# DEFAULT_LLAMACPP_SERVER_PORT = 9641

class LlamaCppServerProcess:
    def __init__(self, model_path: Union[str, Path], clip_model_path: Optional[Union[str, Path]] = None, server_binary_path: Optional[Union[str, Path]]=None, server_args: Dict[str, Any]={}):
        self.model_path = Path(model_path)
        self.clip_model_path = Path(clip_model_path) if clip_model_path else None
        
        if server_binary_path:
            self.server_binary_path = Path(server_binary_path)
        elif llama_cpp_binaries:
            self.server_binary_path = Path(llama_cpp_binaries.get_binary_path())
        else:
            raise FileNotFoundError("llama_cpp_binaries not found and no server_binary_path provided.")

        self.port: Optional[int] = None # Set by start() method
        self.server_args = server_args
        self.process: Optional[subprocess.Popen] = None
        self.session = requests.Session()
        self.host = self.server_args.get("host",DEFAULT_LLAMACPP_SERVER_HOST)
        self.base_url: Optional[str] = None # Set by start() method
        self.is_healthy = False
        self._stderr_lines: List[str] = []
        self._stderr_thread: Optional[threading.Thread] = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if self.clip_model_path and not self.clip_model_path.exists():
            ASCIIColors.warning(f"Clip model file '{self.clip_model_path}' not found. Vision features may not work or may use a different auto-detected clip model.")
        if not self.server_binary_path.exists():
            raise FileNotFoundError(f"Llama.cpp server binary not found: {self.server_binary_path}")

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
                    ASCIIColors.green(f"Llama.cpp server started successfully on port {self.port}.")
                    return
            except requests.exceptions.ConnectionError: time.sleep(1)
            except Exception as e: ASCIIColors.warning(f"Health check for port {self.port} failed: {e}"); time.sleep(1)
        
        self.is_healthy = False
        self.shutdown() 
        stderr_output = "\n".join(self._stderr_lines[-10:])
        raise TimeoutError(f"Llama.cpp server failed to become healthy on port {self.port} within {max_wait_time}s. Stderr:\n{stderr_output}")

    def shutdown(self):
        self.is_healthy = False
        if self.process:
            ASCIIColors.info(f"Shutting down Llama.cpp server (PID: {self.process.pid} on port {self.port})...")
            try:
                if os.name == 'nt': self.process.terminate()
                else: self.process.terminate()
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
    }

    def __init__(self, model_name: str, models_path: str, clip_model_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None, default_completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Chat, **kwargs):
        super().__init__(binding_name=BindingName)
        if llama_cpp_binaries is None: raise ImportError("llama-cpp-binaries package is required but not found.")

        self.models_path = Path(models_path)
        self.user_provided_model_name = model_name  # Store the name/path user gave
        self._model_path_map: Dict[str, Path] = {}  # Maps unique name to full Path

        # Initial scan for available models
        self._scan_models()

        # Determine the model to load
        effective_model_to_load = model_name
        if not effective_model_to_load and self._model_path_map:
            # If no model was specified and we have models, pick the first one
            # Sorting ensures a deterministic choice
            first_model_name = sorted(self._model_path_map.keys())[0]
            effective_model_to_load = first_model_name
            ASCIIColors.info(f"No model was specified. Automatically selecting the first available model: '{effective_model_to_load}'")
            self.user_provided_model_name = effective_model_to_load  # Update for get_model_info etc.

        # Initial hint for clip_model_path, resolved fully in load_model
        self.clip_model_path: Optional[Path] = None
        if clip_model_name:
            p_clip = Path(clip_model_name)
            if p_clip.is_absolute() and p_clip.exists():
                self.clip_model_path = p_clip
            elif (self.models_path / clip_model_name).exists(): # Relative to models_path
                self.clip_model_path = self.models_path / clip_model_name
            else:
                ASCIIColors.warning(f"Specified clip_model_name '{clip_model_name}' not found. Will rely on auto-detection if applicable.")
        
        self.default_completion_format = default_completion_format
        self.server_args = {**self.DEFAULT_SERVER_ARGS, **(config or {}), **kwargs}
        self.server_binary_path = self._get_server_binary_path()
        
        self.current_model_path: Optional[Path] = None # Actual resolved path of loaded model
        self.server_process: Optional[LlamaCppServerProcess] = None
        self.port: Optional[int] = None
        self.server_key: Optional[tuple] = None

        # Now, attempt to load the selected model
        if effective_model_to_load:
            if not self.load_model(effective_model_to_load):
                ASCIIColors.error(f"Initial model load for '{effective_model_to_load}' failed. Binding may not be functional.")
        else:
            ASCIIColors.warning("No models found in the models path. The binding will be idle until a model is loaded.")

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
        raise FileNotFoundError("Llama.cpp server binary not found. Ensure 'llama-cpp-binaries' or 'llama-cpp-python[server]' is installed or provide 'llama_server_binary_path'.")

    def _resolve_model_path(self, model_name_or_path: str) -> Path:
        """
        Resolves a model name or path to a full Path object.
        It prioritizes the internal map, then checks for absolute/relative paths,
        and rescans the models directory as a fallback.
        """
        # 1. Check if the provided name is a key in our map
        if model_name_or_path in self._model_path_map:
            resolved_path = self._model_path_map[model_name_or_path]
            ASCIIColors.info(f"Resolved model name '{model_name_or_path}' to path: {resolved_path}")
            return resolved_path

        # 2. If not in map, treat it as a potential path (absolute or relative to models_path)
        model_p = Path(model_name_or_path)
        if model_p.is_absolute():
            if model_p.exists() and model_p.is_file():
                return model_p

        path_in_models_dir = self.models_path / model_name_or_path
        if path_in_models_dir.exists() and path_in_models_dir.is_file():
            ASCIIColors.info(f"Found model at relative path: {path_in_models_dir}")
            return path_in_models_dir

        # 3. As a fallback, rescan the models directory in case the file was just added
        ASCIIColors.info("Model not found in cache, rescanning directory...")
        self._scan_models()
        if model_name_or_path in self._model_path_map:
            resolved_path = self._model_path_map[model_name_or_path]
            ASCIIColors.info(f"Found model '{model_name_or_path}' after rescan: {resolved_path}")
            return resolved_path

        # Final check for absolute path after rescan
        if model_p.is_absolute() and model_p.exists() and model_p.is_file():
            return model_p

        raise FileNotFoundError(f"Model '{model_name_or_path}' not found in the map, as an absolute path, or within '{self.models_path}'.")

    def _find_available_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0)); return s.getsockname()[1]

    def _release_server_instance(self):
        if self.server_process and self.server_key:
            with _server_registry_lock:
                if self.server_key in _server_ref_counts:
                    _server_ref_counts[self.server_key] -= 1
                    ASCIIColors.info(f"Decremented ref count for server {self.server_key}. New count: {_server_ref_counts[self.server_key]}")
                    if _server_ref_counts[self.server_key] <= 0:
                        ASCIIColors.info(f"Ref count for server {self.server_key} is zero. Shutting it down.")
                        server_to_stop = _active_servers.pop(self.server_key, None)
                        _server_ref_counts.pop(self.server_key, None)
                        if server_to_stop:
                            try: server_to_stop.shutdown()
                            except Exception as e: ASCIIColors.error(f"Error shutting down server {self.server_key}: {e}")
                        # else: ASCIIColors.warning(f"Attempted to stop server {self.server_key} but it was not in _active_servers.") # Can be noisy
                else:
                     ASCIIColors.warning(f"Server key {self.server_key} not in ref counts during release. Might have been shut down already.")
                     _active_servers.pop(self.server_key, None) # Ensure removal

        self.server_process = None
        self.port = None
        self.server_key = None


    def load_model(self, model_name_or_path: str) -> bool:
        self.user_provided_model_name = model_name_or_path # Keep track of the selected model name
        try:
            resolved_model_path = self._resolve_model_path(model_name_or_path)
        except Exception as ex:
            trace_exception(ex)
            return False
        # Determine the clip_model_path for this server instance
        # Priority: 1. Explicit `clip_model_path` from init (if exists) 2. Auto-detection
        final_clip_model_path: Optional[Path] = None
        if self.clip_model_path and self.clip_model_path.exists(): # From __init__
            final_clip_model_path = self.clip_model_path
            ASCIIColors.info(f"Using explicitly configured LLaVA clip model: {final_clip_model_path}")
        elif not self.clip_model_path or (self.clip_model_path and not self.clip_model_path.exists()): # if init path was bad or not given
            if self.clip_model_path and not self.clip_model_path.exists():
                ASCIIColors.warning(f"Initial clip model path '{self.clip_model_path}' not found. Attempting auto-detection.")
            base_name = get_gguf_model_base_name(resolved_model_path.stem)
            potential_paths = [
                resolved_model_path.parent / f"{base_name}.mmproj",
                resolved_model_path.parent / f"mmproj-{base_name}.gguf",
                resolved_model_path.with_suffix(".mmproj"),
                self.models_path / f"{base_name}.mmproj", # Check in general models dir too
                self.models_path / f"mmproj-{base_name}.gguf",
            ]
            for p_clip in potential_paths:
                if p_clip.exists():
                    final_clip_model_path = p_clip
                    ASCIIColors.info(f"Auto-detected LLaVA clip model: {final_clip_model_path}")
                    break
        
        final_clip_model_path_str = str(final_clip_model_path) if final_clip_model_path else None
        
        # Server key based on model and essential server configurations (like clip model)
        # More server_args could be added to the key if they necessitate separate server instances
        # For example, different n_gpu_layers might require a server restart.
        # For now, model and clip model are the main differentiators for distinct servers.
        new_server_key = (str(resolved_model_path), final_clip_model_path_str)

        with _server_registry_lock:
            # If this binding instance is already using the exact same server, do nothing
            if self.server_process and self.server_key == new_server_key and self.server_process.is_healthy:
                ASCIIColors.info(f"Model '{model_name_or_path}' with clip '{final_clip_model_path_str}' is already loaded and server is healthy on port {self.port}. No change.")
                return True

            # If this binding was using a *different* server, release it first
            if self.server_process and self.server_key != new_server_key:
                ASCIIColors.info(f"Switching models. Releasing previous server: {self.server_key}")
                self._release_server_instance() # This clears self.server_process, self.port, self.server_key

            # Check if a suitable server already exists in the global registry
            if new_server_key in _active_servers:
                existing_server = _active_servers[new_server_key]
                if existing_server.is_healthy:
                    ASCIIColors.info(f"Reusing existing healthy server for {new_server_key} on port {existing_server.port}.")
                    self.server_process = existing_server
                    self.port = existing_server.port
                    _server_ref_counts[new_server_key] += 1
                    self.current_model_path = resolved_model_path
                    self.clip_model_path = final_clip_model_path # Update binding's clip path
                    self.server_key = new_server_key
                    return True
                else: # Found existing but unhealthy server
                    ASCIIColors.warning(f"Found unhealthy server for {new_server_key}. Attempting to remove and restart.")
                    try: existing_server.shutdown()
                    except Exception as e: ASCIIColors.error(f"Error shutting down unhealthy server {new_server_key}: {e}")
                    _active_servers.pop(new_server_key, None)
                    _server_ref_counts.pop(new_server_key, None)
            
            # No suitable server found or existing was unhealthy: start a new one
            ASCIIColors.info(f"Starting new server for {new_server_key}.")
            self.current_model_path = resolved_model_path
            self.clip_model_path = final_clip_model_path # Update binding's clip path for the new server
            self.server_key = new_server_key # Set before potential failure to allow cleanup by _release_server_instance

            new_port_for_server = self._find_available_port()
            
            current_server_args_for_new_server = self.server_args.copy()
            # Ensure parallel_slots is set; it's crucial for shared servers
            if "parallel_slots" not in current_server_args_for_new_server or not isinstance(current_server_args_for_new_server["parallel_slots"], int) or current_server_args_for_new_server["parallel_slots"] <=0:
                current_server_args_for_new_server["parallel_slots"] = self.DEFAULT_SERVER_ARGS["parallel_slots"]
            
            ASCIIColors.info(f"New Llama.cpp server: model={self.current_model_path}, clip={self.clip_model_path}, port={new_port_for_server}, slots={current_server_args_for_new_server['parallel_slots']}")

            try:
                new_server = LlamaCppServerProcess(
                    model_path=str(self.current_model_path),
                    clip_model_path=str(self.clip_model_path) if self.clip_model_path else None,
                    server_binary_path=str(self.server_binary_path),
                    server_args=current_server_args_for_new_server,
                )
                new_server.start(port_to_use=new_port_for_server) # Actual server start

                if new_server.is_healthy:
                    self.server_process = new_server
                    self.port = new_port_for_server
                    _active_servers[self.server_key] = new_server
                    _server_ref_counts[self.server_key] = 1
                    ASCIIColors.green(f"New server {self.server_key} started on port {self.port}.")
                    return True
                else: # Should have been caught by new_server.start() raising an error
                    ASCIIColors.error(f"New server {self.server_key} failed to become healthy (this state should be rare).")
                    self._release_server_instance() # Clean up registry if something went very wrong
                    return False
            except Exception as e:
                ASCIIColors.error(f"Failed to load model '{model_name_or_path}' and start server: {e}")
                trace_exception(e)
                self._release_server_instance() # Ensure cleanup if start failed
                return False


    def unload_model(self):
        if self.server_process:
            ASCIIColors.info(f"Unloading model for binding. Current server: {self.server_key}, port: {self.port}")
            self._release_server_instance() # Handles ref counting and actual shutdown if needed
        else:
            ASCIIColors.info("Unload_model called, but no server process was active for this binding instance.")
        self.current_model_path = None 
        self.clip_model_path = None # Also clear the instance's clip path idea
        # self.port and self.server_key are cleared by _release_server_instance

    def _get_request_url(self, endpoint: str) -> str:
        if not self.server_process or not self.server_process.is_healthy:
            raise ConnectionError("Llama.cpp server is not running or not healthy.")
        return f"{self.server_process.base_url}{endpoint}"

    def _prepare_generation_payload(self, prompt: str, system_prompt: str = "", n_predict: Optional[int] = None,
                                   temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9,
                                   repeat_penalty: float = 1.1, repeat_last_n: Optional[int] = 64,
                                   seed: Optional[int] = None, stream: bool = False, use_chat_format: bool = True,
                                   images: Optional[List[str]] = None,
                                    split:Optional[bool]=False, # put to true if the prompt is a discussion
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
            if images and self.clip_model_path: # Use the binding's current clip_model_path
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
            if images and self.clip_model_path: # Use binding's clip_model_path
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
                     temperature: float = 0.7, # Ollama default is 0.8, common default 0.7
                     top_k: int = 40,          # Ollama default is 40
                     top_p: float = 0.9,       # Ollama default is 0.9
                     repeat_penalty: float = 1.1, # Ollama default is 1.1
                     repeat_last_n: int = 64,  # Ollama default is 64
                     seed: Optional[int] = None,
                     n_threads: Optional[int] = None,
                     ctx_size: int | None = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     split:Optional[bool]=False, # put to true if the prompt is a discussion
                     user_keyword:Optional[str]="!@>user:",
                     ai_keyword:Optional[str]="!@>assistant:", 
                     **generation_kwargs
                     ) -> Union[str, dict]:
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths for multimodal generation.
            n_predict (Optional[int]): Maximum number of tokens to generate. Uses instance default if None.
            stream (Optional[bool]): Whether to stream the output. Uses instance default if None.
            temperature (Optional[float]): Sampling temperature. Uses instance default if None.
            top_k (Optional[int]): Top-k sampling parameter. Uses instance default if None.
            top_p (Optional[float]): Top-p sampling parameter. Uses instance default if None.
            repeat_penalty (Optional[float]): Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty. Uses instance default if None.
            seed (Optional[int]): Random seed for generation. Uses instance default if None.
            n_threads (Optional[int]): Number of threads to use. Uses instance default if None.
            ctx_size (int | None): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback function for streaming output.
                - First parameter (str): The chunk of text received.
                - Second parameter (str): The message type (e.g., MSG_TYPE.MSG_TYPE_CHUNK).
            split:Optional[bool]: put to true if the prompt is a discussion
            user_keyword:Optional[str]: when splitting we use this to extract user prompt 
            ai_keyword:Optional[str]": when splitting we use this to extract ai prompt

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
        """
        if not self.server_process or not self.server_process.is_healthy:
             return {"status": False, "error": "Llama.cpp server is not running or not healthy."}

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
            split= split, user_keyword=user_keyword, ai_keyword=ai_keyword, **generation_kwargs
        )
        endpoint = "/v1/chat/completions" if _use_chat_format else "/completion"
        request_url = self._get_request_url(endpoint)
        
        # Debug payload (simplified)
        # debug_payload = {k:v for k,v in payload.items() if k not in ["image_data","messages"] or (k=="messages" and not any("image_url" in part for item in v for part in (item.get("content") if isinstance(item.get("content"),list) else [])))} # Complex filter for brevity
        # ASCIIColors.debug(f"Request to {request_url} with payload (simplified): {json.dumps(debug_payload, indent=2)[:500]}...")


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
                       else response_data.get('content','') # /completion has 'content' at top level for non-stream
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
             **generation_kwargs
             ) -> Union[str, dict]:
        """
        Conduct a chat session with the llama.cpp server using a LollmsDiscussion object.

        Args:
            discussion (LollmsDiscussion): The discussion object containing the conversation history.
            branch_tip_id (Optional[str]): The ID of the message to use as the tip of the conversation branch. Defaults to the active branch.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (Optional[bool]): Whether to stream the output.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p sampling parameter.
            repeat_penalty (float): Penalty for repeated tokens.
            repeat_last_n (int): Number of previous tokens to consider for repeat penalty.
            seed (Optional[int]): Random seed for generation.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.

        Returns:
            Union[str, dict]: The generated text or an error dictionary.
        """
        if not self.server_process or not self.server_process.is_healthy:
            return {"status": "error", "message": "Llama.cpp server is not running or not healthy."}

        # 1. Export the discussion to the OpenAI chat format, which llama.cpp server understands.
        # This handles system prompts, user/assistant roles, and multi-modal content.
        messages = discussion.export("openai_chat", branch_tip_id)

        # 2. Build the generation payload for the server
        payload = {
            "messages": messages,
            "max_tokens": n_predict,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "seed": seed,
            "stream": stream,
            **generation_kwargs # Pass any extra parameters
        }
        # Remove None values, as the API expects them to be absent
        payload = {k: v for k, v in payload.items() if v is not None}
        
        endpoint = "/v1/chat/completions"
        request_url = self._get_request_url(endpoint)
        full_response_text = ""

        try:
            # 3. Make the request to the server
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
                        chunk_content = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
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
            else: # Not streaming
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
        if not self.server_process or not self.server_process.is_healthy: raise ConnectionError("Server not running.")
        try:
            response = self.server_process.session.post(self._get_request_url("/tokenize"), json={"content": text})
            response.raise_for_status(); return response.json().get("tokens", [])
        except Exception as e: ASCIIColors.error(f"Tokenization error: {e}"); trace_exception(e); return []

    def detokenize(self, tokens: List[int]) -> str:
        if not self.server_process or not self.server_process.is_healthy: raise ConnectionError("Server not running.")
        try:
            response = self.server_process.session.post(self._get_request_url("/detokenize"), json={"tokens": tokens})
            response.raise_for_status(); return response.json().get("content", "")
        except Exception as e: ASCIIColors.error(f"Detokenization error: {e}"); trace_exception(e); return ""

    def count_tokens(self, text: str) -> int: return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        if not self.server_process or not self.server_process.is_healthy: raise Exception("Server not running.")
        if not self.server_args.get("embedding"): raise Exception("Embedding not enabled in server_args.")
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
            raise Exception(err_msg) from e
        except Exception as ex: trace_exception(ex); raise Exception(f"Embedding failed: {str(ex)}") from ex
        
    def get_model_info(self) -> dict:
        info = {
            "name": self.binding_name,
            "user_provided_model_name": self.user_provided_model_name,
            "model_path": str(self.current_model_path) if self.current_model_path else "Not loaded",
            "clip_model_path": str(self.clip_model_path) if self.clip_model_path else "N/A",
            "loaded": self.server_process is not None and self.server_process.is_healthy,
            "server_args": self.server_args, "port": self.port if self.port else "N/A",
            "server_key": str(self.server_key) if self.server_key else "N/A",
        }
        if info["loaded"] and self.server_process:
            try:
                props_resp = self.server_process.session.get(self._get_request_url("/props"), timeout=5).json()
                info.update({
                    "server_n_ctx": props_resp.get("default_generation_settings",{}).get("n_ctx"),
                    "server_chat_format": props_resp.get("chat_format"),
                    "server_clip_model_from_props": props_resp.get("mmproj"), # Server's view of clip model
                })
            except Exception: pass 
            
            is_llava = self.clip_model_path is not None or \
                       (info.get("server_clip_model_from_props") is not None) or \
                       ("llava" in self.current_model_path.name.lower() if self.current_model_path else False)
            info["supports_vision"] = is_llava
            info["supports_structured_output"] = self.server_args.get("grammar_string") is not None
        return info

    def _scan_models(self):
        """
        Scans the models_path for GGUF files and populates the model map.
        Handles duplicate filenames by prefixing them with their parent directory path.
        """
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
                # On Windows, path separators can be tricky. Convert to generic format.
                relative_path_str = str(model_file.relative_to(self.models_path).as_posix())
                if filenames_count[model_file.name] > 1:
                    # Duplicate filename, use relative path as the unique name
                    unique_name = relative_path_str
                else:
                    # Unique filename, use the name itself
                    unique_name = model_file.name
                
                self._model_path_map[unique_name] = model_file
        
        ASCIIColors.info(f"Scanned {len(self._model_path_map)} models from {self.models_path}.")

    def listModels(self) -> List[Dict[str, Any]]:
        """
        Lists all available GGUF models, rescanning the directory first.
        """
        self._scan_models()  # Always rescan when asked for the list

        models_found = []
        for unique_name, model_path in self._model_path_map.items():
            models_found.append({
                'name': unique_name,  # The unique name for selection
                'model_name': model_path.name, # The original filename for display
                'path': str(model_path),  # The full path
                'size': model_path.stat().st_size
            })
        
        # Sort the list alphabetically by the unique name for consistent ordering
        return sorted(models_found, key=lambda x: x['name'])
    
    def __del__(self):
        self.unload_model()

    def get_ctx_size(self, model_name: Optional[str] = None) -> Optional[int]:
        """
        Retrieves context size for a model from a hardcoded list.

        This method checks if the model name contains a known base model identifier
        (e.g., 'llama3.1', 'gemma2') to determine its context length. It's intended
        as a failsafe when the context size cannot be retrieved directly from the
        Ollama API.
        """
        if model_name is None:
            model_name = self.model_name

        # Hardcoded context sizes for popular models. More specific names (e.g., 'llama3.1')
        # should appear, as they will be checked first due to the sorting logic below.
        known_contexts = {
            'llama3.1': 131072,   # Llama 3.1 extended context
            'llama3.2': 131072,   # Llama 3.2 extended context
            'llama3.3': 131072,   # Assuming similar to 3.1/3.2
            'llama3': 8192,       # Llama 3 default
            'llama2': 4096,       # Llama 2 default
            'mixtral8x22b': 65536, # Mixtral 8x22B default
            'mixtral': 32768,     # Mixtral 8x7B default
            'mistral': 32768,     # Mistral 7B v0.2+ default
            'gemma3': 131072,     # Gemma 3 with 128K context
            'gemma2': 8192,       # Gemma 2 default
            'gemma': 8192,        # Gemma default
            'phi3': 131072,       # Phi-3 variants often use 128K (mini/medium extended)
            'phi2': 2048,         # Phi-2 default
            'phi': 2048,          # Phi default (older)
            'qwen2.5': 131072,    # Qwen2.5 with 128K
            'qwen2': 32768,       # Qwen2 default for 7B
            'qwen': 8192,         # Qwen default
            'codellama': 16384,   # CodeLlama extended
            'codegemma': 8192,    # CodeGemma default
            'deepseek-coder-v2': 131072,  # DeepSeek-Coder V2 with 128K
            'deepseek-coder': 16384,  # DeepSeek-Coder V1 default
            'deepseek-v2': 131072,    # DeepSeek-V2 with 128K
            'deepseek-llm': 4096,     # DeepSeek-LLM default
            'yi1.5': 32768,       # Yi-1.5 with 32K
            'yi': 4096,           # Yi base default
            'command-r': 131072,  # Command-R with 128K
            'wizardlm2': 32768,   # WizardLM2 (Mistral-based)
            'wizardlm': 16384,    # WizardLM default
            'zephyr': 65536,      # Zephyr beta (Mistral-based extended)
            'vicuna': 2048,       # Vicuna default (up to 16K in some variants)
            'falcon': 2048,       # Falcon default
            'starcoder': 8192,    # StarCoder default
            'stablelm': 4096,     # StableLM default
            'orca2': 4096,        # Orca 2 default
            'orca': 4096,         # Orca default
            'dolphin': 32768,     # Dolphin (often Mistral-based)
            'openhermes': 8192,   # OpenHermes default
        }

        normalized_model_name = model_name.lower().strip()

        # Sort keys by length in descending order. This ensures that a more specific
        # name like 'llama3.1' is checked before a less specific name like 'llama3'.
        sorted_base_models = sorted(known_contexts.keys(), key=len, reverse=True)

        for base_name in sorted_base_models:
            if base_name in normalized_model_name:
                context_size = known_contexts[base_name]
                ASCIIColors.warning(
                    f"Using hardcoded context size for model '{model_name}' "
                    f"based on base name '{base_name}': {context_size}"
                )
                return context_size

        ASCIIColors.warning(f"Context size not found for model '{model_name}' in the hardcoded list.")
        return None

if __name__ == '__main__':
    global full_streamed_text # Define for the callback
    full_streamed_text = ""
    ASCIIColors.yellow("Testing LlamaCppServerBinding...")

    # --- Configuration ---
    # This should be the NAME of your GGUF model file.
    # Ensure this model is placed in your models_path directory.
    # Example: models_path = "E:\\lollms\\models\\gguf" (Windows)
    #          model_name = "Mistral-Nemo-Instruct-2407-Q2_K.gguf"
    
    # For CI/local testing without specific paths, you might download a tiny model
    # or require user to set environment variables for these.
    # For this example, replace with your actual paths/model.
    try:
        models_path_str = os.environ.get("LOLLMS_MODELS_PATH", str(Path(__file__).parent / "test_models"))
        model_name_str = os.environ.get("LOLLMS_TEST_MODEL_GGUF", "tinyllama-1.1b-chat-v1.0.Q2_K.gguf") # A small model
        llava_model_name_str = os.environ.get("LOLLMS_TEST_LLAVA_MODEL_GGUF", "llava-v1.5-7b.Q2_K.gguf") # Placeholder
        llava_clip_name_str = os.environ.get("LOLLMS_TEST_LLAVA_CLIP", "mmproj-model2-q4_0.gguf") # Placeholder

        models_path = Path(models_path_str)
        models_path.mkdir(parents=True, exist_ok=True) # Ensure test_models dir exists
        
        # Verify model exists, or skip tests gracefully
        test_model_path = models_path / model_name_str
        if not test_model_path.exists():
            ASCIIColors.warning(f"Test model {test_model_path} not found. Please place a GGUF model there or set LOLLMS_TEST_MODEL_GGUF and LOLLMS_MODELS_PATH env vars.")
            ASCIIColors.warning("Some tests will be skipped.")
            # sys.exit(1) # Or allow to continue with skips
            primary_model_available = False
        else:
            primary_model_available = True

    except Exception as e:
        ASCIIColors.error(f"Error setting up test paths: {e}"); trace_exception(e)
        sys.exit(1)

    binding_config = {
        "n_gpu_layers": 0, "n_ctx": 512, "embedding": True,
        "verbose": False, "server_startup_timeout": 180, "parallel_slots": 2,
    }

    active_binding1: Optional[LlamaCppServerBinding] = None
    active_binding2: Optional[LlamaCppServerBinding] = None
    active_binding_llava: Optional[LlamaCppServerBinding] = None

    try:
        if primary_model_available:
            ASCIIColors.cyan("\n--- Initializing First LlamaCppServerBinding Instance ---")
            # Test default model selection by passing model_name=None
            ASCIIColors.info("Testing default model selection (model_name=None)")
            active_binding1 = LlamaCppServerBinding(
                model_name=None, models_path=str(models_path), config=binding_config
            )
            if not active_binding1.server_process or not active_binding1.server_process.is_healthy:
                raise RuntimeError("Server for binding1 failed to start or become healthy.")
            ASCIIColors.green(f"Binding1 initialized with default model. Server for '{active_binding1.current_model_path.name}' running on port {active_binding1.port}.")
            ASCIIColors.info(f"Binding1 Model Info: {json.dumps(active_binding1.get_model_info(), indent=2)}")

            ASCIIColors.cyan("\n--- Initializing Second LlamaCppServerBinding Instance (Same Model, explicit name) ---")
            # Load the same model explicitly now
            model_to_load_explicitly = active_binding1.user_provided_model_name
            active_binding2 = LlamaCppServerBinding(
                model_name=model_to_load_explicitly, models_path=str(models_path), config=binding_config
            )
            if not active_binding2.server_process or not active_binding2.server_process.is_healthy:
                raise RuntimeError("Server for binding2 failed to start or become healthy (should reuse).")
            ASCIIColors.green(f"Binding2 initialized. Server for '{active_binding2.current_model_path.name}' running on port {active_binding2.port}.")
            ASCIIColors.info(f"Binding2 Model Info: {json.dumps(active_binding2.get_model_info(), indent=2)}")

            if active_binding1.port != active_binding2.port:
                ASCIIColors.error("ERROR: Bindings for the same model are using different ports! Server sharing failed.")
            else:
                ASCIIColors.green("SUCCESS: Both bindings use the same server port. Server sharing appears to work.")
            
            # --- List Models (scans configured directories) ---
            ASCIIColors.cyan("\n--- Listing Models (from search paths, using binding1) ---")
            # Create a dummy duplicate model to test unique naming
            duplicate_folder = models_path / "subdir"
            duplicate_folder.mkdir(exist_ok=True)
            duplicate_model_path = duplicate_folder / test_model_path.name
            import shutil
            shutil.copy(test_model_path, duplicate_model_path)
            ASCIIColors.info(f"Created a duplicate model for testing: {duplicate_model_path}")
            
            listed_models = active_binding1.listModels()
            if listed_models: 
                ASCIIColors.green(f"Found {len(listed_models)} GGUF files.")
                pprint.pprint(listed_models)
                # Check if the duplicate was handled
                names = [m['name'] for m in listed_models]
                if test_model_path.name in names and f"subdir/{test_model_path.name}" in names:
                    ASCIIColors.green("SUCCESS: Duplicate model names were correctly handled.")
                else:
                    ASCIIColors.error("FAILURE: Duplicate model names were not handled correctly.")
            else: ASCIIColors.warning("No GGUF models found in search paths.")
            
            # Clean up dummy duplicate
            duplicate_model_path.unlink()
            duplicate_folder.rmdir()


            # --- Tokenize/Detokenize ---
            ASCIIColors.cyan("\n--- Tokenize/Detokenize (using binding1) ---")
            sample_text = "Hello, Llama.cpp server world!"
            tokens = active_binding1.tokenize(sample_text)
            ASCIIColors.green(f"Tokens for '{sample_text}': {tokens[:10]}...")
            if tokens:
                detokenized_text = active_binding1.detokenize(tokens)
                ASCIIColors.green(f"Detokenized text: {detokenized_text}")
            else: ASCIIColors.warning("Tokenization returned empty list.")

            # --- Text Generation (Non-Streaming, Chat API, binding1) ---
            ASCIIColors.cyan("\n--- Text Generation (Non-Streaming, Chat API, binding1) ---")
            prompt_text = "What is the capital of Germany?"
            generated_text = active_binding1.generate_text(prompt_text, system_prompt="Concise expert.", n_predict=20, stream=False)
            if isinstance(generated_text, str): ASCIIColors.green(f"Generated text (binding1): {generated_text}")
            else: ASCIIColors.error(f"Generation failed (binding1): {generated_text}")

            # --- Text Generation (Streaming, Completion API, binding2) ---
            ASCIIColors.cyan("\n--- Text Generation (Streaming, Chat API, binding2) ---")
            full_streamed_text = "" # Reset global
            def stream_callback(chunk: str, msg_type: int): global full_streamed_text; ASCIIColors.green(f"{chunk}", end="", flush=True); full_streamed_text += chunk; return True
            
            result_b2 = active_binding2.generate_text(prompt_text, system_prompt="Concise expert.", n_predict=30, stream=True, streaming_callback=stream_callback)
            print("\n--- End of Stream (binding2) ---")
            if isinstance(result_b2, str): ASCIIColors.green(f"Full streamed text (binding2): {result_b2}")
            else: ASCIIColors.error(f"Streaming generation failed (binding2): {result_b2}")

            # --- Embeddings (binding1) ---
            if binding_config.get("embedding"):
                ASCIIColors.cyan("\n--- Embeddings (binding1) ---")
                try:
                    embedding_vector = active_binding1.embed("Test embedding.")
                    ASCIIColors.green(f"Embedding (first 3 dims): {embedding_vector[:3]}... Dim: {len(embedding_vector)}")
                except Exception as e_emb: ASCIIColors.warning(f"Could not get embedding: {e_emb}")
            else: ASCIIColors.yellow("\n--- Embeddings Skipped (embedding: false) ---")

        else: # primary_model_available is False
            ASCIIColors.warning("Primary test model not available. Skipping most tests.")


        # --- LLaVA Test (Conceptual - requires a LLaVA model and mmproj) ---
        ASCIIColors.cyan("\n--- LLaVA Vision Test (if model available) ---")
        llava_model_path = models_path / llava_model_name_str
        llava_clip_path_actual = models_path / llava_clip_name_str # Assuming clip is in models_path too

        if llava_model_path.exists() and llava_clip_path_actual.exists():
            dummy_image_path = models_path / "dummy_llava_image.png"
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (150, 70), color = ('magenta')); d = ImageDraw.Draw(img); d.text((10,10), "LLaVA Test", fill=('white')); img.save(dummy_image_path)
                ASCIIColors.info(f"Created dummy image for LLaVA: {dummy_image_path}")

                llava_binding_config = binding_config.copy()
                # LLaVA might need specific chat template if server doesn't auto-detect well.
                # llava_binding_config["chat_template"] = "llava-1.5" 
                
                active_binding_llava = LlamaCppServerBinding(
                    model_name=str(llava_model_path.name), # Pass filename, let it resolve
                    models_path=str(models_path), 
                    clip_model_name=str(llava_clip_path_actual.name), # Pass filename for clip
                    config=llava_binding_config
                )
                if not active_binding_llava.server_process or not active_binding_llava.server_process.is_healthy:
                     raise RuntimeError("LLaVA server failed to start or become healthy.")
                ASCIIColors.green(f"LLaVA Binding initialized. Server for '{active_binding_llava.current_model_path.name}' running on port {active_binding_llava.port}.")
                ASCIIColors.info(f"LLaVA Binding Model Info: {json.dumps(active_binding_llava.get_model_info(), indent=2)}")


                llava_prompt = "Describe this image."
                llava_response = active_binding_llava.generate_text(
                    prompt=llava_prompt, images=[str(dummy_image_path)], n_predict=40, stream=False
                )
                if isinstance(llava_response, str): ASCIIColors.green(f"LLaVA response: {llava_response}")
                else: ASCIIColors.error(f"LLaVA generation failed: {llava_response}")

            except ImportError: ASCIIColors.warning("Pillow not found. Cannot create dummy image for LLaVA.")
            except Exception as e_llava: ASCIIColors.error(f"LLaVA test error: {e_llava}"); trace_exception(e_llava)
            finally:
                if dummy_image_path.exists(): dummy_image_path.unlink()
        else:
            ASCIIColors.warning(f"LLaVA model '{llava_model_path.name}' or clip model '{llava_clip_path_actual.name}' not found in '{models_path}'. Skipping LLaVA test.")
        
        if primary_model_available and active_binding1:
            # --- Test changing model (using binding1 to load a different or same model) ---
            ASCIIColors.cyan("\n--- Testing Model Change (binding1 reloads its model) ---")
            # For a real change, use a different model name if available. Here, we reload the same.
            reload_success = active_binding1.load_model(active_binding1.user_provided_model_name) # Reload original model
            if reload_success and active_binding1.server_process and active_binding1.server_process.is_healthy:
                ASCIIColors.green(f"Model reloaded/re-confirmed successfully by binding1. Server on port {active_binding1.port}.")
                reloaded_gen = active_binding1.generate_text("Ping", n_predict=5, stream=False)
                if isinstance(reloaded_gen, str): ASCIIColors.green(f"Post-reload ping (binding1): {reloaded_gen.strip()}")
                else: ASCIIColors.error(f"Post-reload generation failed (binding1): {reloaded_gen}")
            else:
                ASCIIColors.error("Failed to reload model or server not healthy after reload attempt by binding1.")

    except ImportError as e_imp: ASCIIColors.error(f"Import error: {e_imp}.")
    except FileNotFoundError as e_fnf: ASCIIColors.error(f"File not found error: {e_fnf}.")
    except ConnectionError as e_conn: ASCIIColors.error(f"Connection error: {e_conn}")
    except RuntimeError as e_rt:
        ASCIIColors.error(f"Runtime error: {e_rt}")
        if active_binding1 and active_binding1.server_process: ASCIIColors.error(f"Binding1 stderr:\n{active_binding1.server_process._stderr_lines[-20:]}")
        if active_binding2 and active_binding2.server_process: ASCIIColors.error(f"Binding2 stderr:\n{active_binding2.server_process._stderr_lines[-20:]}")
        if active_binding_llava and active_binding_llava.server_process: ASCIIColors.error(f"LLaVA Binding stderr:\n{active_binding_llava.server_process._stderr_lines[-20:]}")
    except Exception as e_main: ASCIIColors.error(f"An unexpected error occurred: {e_main}"); trace_exception(e_main)
    finally:
        ASCIIColors.cyan("\n--- Unloading Models and Stopping Servers ---")
        if active_binding1: active_binding1.unload_model(); ASCIIColors.info("Binding1 unloaded.")
        if active_binding2: active_binding2.unload_model(); ASCIIColors.info("Binding2 unloaded.")
        if active_binding_llava: active_binding_llava.unload_model(); ASCIIColors.info("LLaVA Binding unloaded.")
        
        # Check if any servers remain (should be none if all bindings unloaded)
        with _server_registry_lock:
            if _active_servers:
                ASCIIColors.warning(f"Warning: {_active_servers.keys()} servers still in registry after all known bindings unloaded.")
                for key, server_proc in list(_active_servers.items()): # list() for safe iteration if modifying
                    ASCIIColors.info(f"Force shutting down stray server: {key}")
                    try: server_proc.shutdown()
                    except Exception as e_shutdown: ASCIIColors.error(f"Error shutting down stray server {key}: {e_shutdown}")
                    _active_servers.pop(key,None)
                    _server_ref_counts.pop(key,None)
            else:
                ASCIIColors.green("All servers shut down correctly.")

    ASCIIColors.yellow("\nLlamaCppServerBinding test finished.")