import subprocess
import sys
import os
import time
import requests
import socket
import re
import platform
import zipfile
import tarfile
import json
import yaml
import atexit
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

import pipmaster as pm
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion

# Ensure dependencies
pm.ensure_packages(["openai", "huggingface_hub", "filelock", "requests", "tqdm", "psutil", "pyyaml"])
import openai
from huggingface_hub import hf_hub_download
from filelock import FileLock
from tqdm import tqdm
import psutil

BindingName = "LlamaCppServerBinding"

def get_free_port(start_port=9624, max_port=10000):
    """
    Finds a free port on localhost. 
    Race-condition safe-ish: We bind to it to check, but release it immediately.
    Real safety comes from the FileLock around this call.
    """
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('localhost', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports available.")

class LlamaCppServerBinding(LollmsLLMBinding):
    def __init__(self, **kwargs):
        super().__init__(BindingName, **kwargs)
        self.config = kwargs
        
        # Configuration
        self.host = kwargs.get("host", "localhost")
        self.model_name = kwargs.get("model_name", "")
        self.n_ctx = kwargs.get("ctx_size", 4096)
        self.n_gpu_layers = kwargs.get("n_gpu_layers", -1) 
        self.n_threads = kwargs.get("n_threads", None)
        self.n_parallel = kwargs.get("n_parallel", 1)
        self.batch_size = kwargs.get("batch_size", 512)
        
        # Server Management
        self.max_active_models = int(kwargs.get("max_active_models", 1))
        self.idle_timeout = float(kwargs.get("idle_timeout", -1))
        
        # Paths
        self.binding_dir = Path(__file__).parent
        self.bin_dir = self.binding_dir / "bin"
        self.models_dir = Path(kwargs.get("models_path", "models/llama_cpp_models")).resolve()
        
        # Multimodal Registry
        self.mm_registry_path = self.models_dir / "multimodal_bindings.yaml"
        
        # Registry directory for inter-process coordination
        self.servers_dir = self.models_dir / "servers"
        self.servers_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir.mkdir(exist_ok=True)
        
        # Global lock file for all operations on the registry
        self.global_lock_path = self.models_dir / "global_server_manager.lock"
        
        # Installation check
        if not self._get_server_executable().exists():
            ASCIIColors.warning("Llama.cpp binary not found. Attempting installation...")
            self.install_llama_cpp()
            
        # Register cleanup for this process
        atexit.register(self.cleanup_orphans_if_needed)

    def _get_server_executable(self) -> Path:
        if platform.system() == "Windows":
            return self.bin_dir / "llama-server.exe"
        else:
            return self.bin_dir / "llama-server"

    def detect_hardware(self) -> str:
        sys_plat = platform.system()
        if sys_plat == "Darwin":
            return "macos"
        try:
            subprocess.check_call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "cuda"
        except:
            pass
        return "cpu"

    def install_llama_cpp(self):
        try:
            ASCIIColors.info("Checking latest llama.cpp release...")
            releases_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
            response = requests.get(releases_url)
            response.raise_for_status()
            release_data = response.json()
            assets = release_data.get("assets", [])
            
            hardware = self.detect_hardware()
            sys_plat = platform.system()
            
            target_asset = None
            search_terms = []
            
            if sys_plat == "Windows":
                search_terms.append("win")
                search_terms.append("cuda" if hardware == "cuda" else "avx2")
                search_terms.append("x64")
            elif sys_plat == "Linux":
                search_terms.append("ubuntu")
                search_terms.append("x64")
            elif sys_plat == "Darwin":
                search_terms.append("macos")
                search_terms.append("arm64" if platform.machine() == "arm64" else "x64")

            for asset in assets:
                name = asset["name"].lower()
                if "cudart" in name: continue 
                if all(term in name for term in search_terms):
                    if "cuda" in name and "cu11" in name and hardware == "cuda": continue 
                    target_asset = asset
                    break
            
            # Windows CPU fallback
            if not target_asset and sys_plat == "Windows" and hardware == "cpu":
                 for asset in assets:
                    if "cudart" in asset["name"].lower(): continue
                    if "win" in asset["name"].lower() and "x64" in asset["name"].lower() and "cuda" not in asset["name"].lower():
                        target_asset = asset
                        break

            if not target_asset:
                raise RuntimeError(f"No suitable binary found for {sys_plat} / {hardware}")

            download_url = target_asset["browser_download_url"]
            filename = target_asset["name"]
            dest_file = self.bin_dir / filename
            
            ASCIIColors.info(f"Downloading {filename}...")
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(dest_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            ASCIIColors.info("Extracting...")
            if filename.endswith(".zip"):
                with zipfile.ZipFile(dest_file, 'r') as z: z.extractall(self.bin_dir)
            elif filename.endswith(".tar.gz"):
                with tarfile.open(dest_file, "r:gz") as t: t.extractall(self.bin_dir)
            
            dest_file.unlink()
            
            # Normalize binary name
            exe_name = "llama-server.exe" if sys_plat == "Windows" else "llama-server"
            legacy_name = "server.exe" if sys_plat == "Windows" else "server"
            if not (self.bin_dir / exe_name).exists() and (self.bin_dir / legacy_name).exists():
                shutil.move(str(self.bin_dir / legacy_name), str(self.bin_dir / exe_name))
            
            if sys_plat != "Windows":
                exe_path = self.bin_dir / exe_name
                if exe_path.exists(): os.chmod(exe_path, 0o755)

            ASCIIColors.success("Llama.cpp installed successfully.")
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"Failed to install llama.cpp: {e}")

    # --- Server Management Logic ---

    def _get_registry_file(self, model_name: str) -> Path:
        # Sanitize filename
        safe_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_', '.'))
        return self.servers_dir / f"{safe_name}.json"

    def _get_server_info(self, model_name: str) -> Optional[Dict]:
        """Reads registry file for a model, returns dict or None if invalid."""
        reg_file = self._get_registry_file(model_name)
        if not reg_file.exists():
            return None
        
        try:
            with open(reg_file, 'r') as f:
                info = json.load(f)
            
            # Verify process is alive
            if psutil.pid_exists(info['pid']):
                # Verify it's actually llama-server (optional but safe)
                try:
                    p = psutil.Process(info['pid'])
                    if "llama" in p.name().lower() or "server" in p.name().lower():
                        return info
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # If we get here, process is dead or invalid
            ASCIIColors.warning(f"Found stale registry file for {model_name} (PID {info['pid']}). Cleaning up.")
            reg_file.unlink()
            return None
        except Exception:
            # Corrupt file
            if reg_file.exists(): reg_file.unlink()
            return None

    def _kill_server(self, model_name: str, info: Dict):
        """Kills a server process and removes its registry file."""
        ASCIIColors.info(f"Stopping server for {model_name} (PID {info['pid']})...")
        try:
            p = psutil.Process(info['pid'])
            p.terminate()
            p.wait(timeout=5)
        except psutil.NoSuchProcess:
            pass # Already gone
        except psutil.TimeoutExpired:
            p.kill()
        except Exception as e:
            ASCIIColors.error(f"Error killing process: {e}")
        
        # Remove registry file
        reg_file = self._get_registry_file(model_name)
        if reg_file.exists():
            reg_file.unlink()

    def _ensure_capacity_locked(self):
        """
        Called while holding the lock. Ensures we have space for a new model.
        """
        registry_files = list(self.servers_dir.glob("*.json"))
        
        # 1. Clean up stale entries first
        valid_servers = []
        for rf in registry_files:
            try:
                with open(rf, 'r') as f:
                    data = json.load(f)
                if psutil.pid_exists(data['pid']):
                    valid_servers.append((rf, data))
                else:
                    rf.unlink() # Clean stale
            except:
                if rf.exists(): rf.unlink()

        # 2. Check capacity
        if len(valid_servers) >= self.max_active_models:
            # Sort by file modification time (mtime), which acts as our "last used" heartbeat
            # Oldest mtime = Least Recently Used
            valid_servers.sort(key=lambda x: x[0].stat().st_mtime)
            
            # Kill the oldest
            oldest_file, oldest_info = valid_servers[0]
            model_to_kill = oldest_info.get("model_name", "unknown")
            ASCIIColors.warning(f"Max active models ({self.max_active_models}) reached. Unloading LRU model: {model_to_kill}")
            self._kill_server(model_to_kill, oldest_info)

    def _load_mm_registry(self) -> Dict[str, str]:
        if not self.mm_registry_path.exists():
            return {}
        try:
            with open(self.mm_registry_path, 'r') as f:
                registry = yaml.safe_load(f) or {}
            
            # Self-healing: remove missing files
            updated = False
            to_remove = []
            for m, p in registry.items():
                if not (self.models_dir / m).exists() or not (self.models_dir / p).exists():
                    to_remove.append(m)
                    updated = True
            
            for m in to_remove:
                del registry[m]
            
            if updated:
                self._save_mm_registry(registry)
            return registry
        except Exception as e:
            ASCIIColors.error(f"Failed to load multimodal registry: {e}")
            return {}

    def _save_mm_registry(self, registry: Dict[str, str]):
        try:
            with open(self.mm_registry_path, 'w') as f:
                yaml.dump(registry, f)
        except Exception as e:
            ASCIIColors.error(f"Failed to save multimodal registry: {e}")

    def bind_multimodal_model(self, model_name: str, mmproj_name: str) -> dict:
        """Explicitly binds a model to an mmproj file."""
        if not (self.models_dir / model_name).exists():
            return {"status": False, "error": f"Model {model_name} not found."}
        if not (self.models_dir / mmproj_name).exists():
            return {"status": False, "error": f"Projector {mmproj_name} not found."}
        
        registry = self._load_mm_registry()
        registry[model_name] = mmproj_name
        self._save_mm_registry(registry)
        
        ASCIIColors.success(f"Bound {model_name} with {mmproj_name}")
        return {"status": True, "message": f"Bound {model_name} with {mmproj_name}"}

    def _find_mmproj(self, model_path: Path) -> Optional[Path]:
        """Finds a corresponding mmproj file for a given model path."""
        # 1. Check registry first
        registry = self._load_mm_registry()
        if model_path.name in registry:
            proj_path = self.models_dir / registry[model_path.name]
            if proj_path.exists():
                return proj_path

        # 2. Automatic detection patterns
        stem = model_path.stem
        clean_stem = re.sub(r'\.(Q\d_.*|f16|f32)$', '', stem)
        patterns = [
            f"{stem}.mmproj", f"{stem}-mmproj.gguf", f"{stem}.mmproj.gguf",
            f"{clean_stem}.mmproj", f"{clean_stem}-mmproj.gguf",
            f"mmproj-{stem}.gguf", "mmproj.gguf"
        ]
        
        for p in patterns:
            pot = model_path.parent / p
            if pot.exists():
                return pot
        
        # 3. Last resort: simple scan
        try:
            for f in model_path.parent.iterdir():
                if f.is_file() and "mmproj" in f.name.lower() and f.name != model_path.name:
                    if f.suffix in [".gguf", ".mmproj", ".bin"]:
                        return f
        except:
            pass
            
        return None

    def _spawn_server_detached(self, model_name: str):
        """Spawns the server process detached so it survives if this python script ends."""
        exe_path = self._get_server_executable()
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
            
        port = get_free_port()
        
        cmd = [
            str(exe_path),
            "--model", str(model_path),
            "--host", self.host,
            "--port", str(port),
            "--ctx-size", str(self.n_ctx),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--parallel", str(self.n_parallel),
            "--batch-size", str(self.batch_size),
            "--embedding"
        ]

        # Automatic detection or Registry-based mmproj
        mmproj_path = self._find_mmproj(model_path)
        if mmproj_path:
            ASCIIColors.info(f"Detected multimodal projector: {mmproj_path}")
            cmd.extend(["--mmproj", str(mmproj_path)])
        
        if self.n_threads:
            cmd.extend(["--threads", str(self.n_threads)])

        ASCIIColors.info(f"Spawning server for {model_name} on port {port}...")
        
        # Process creation flags for detachment
        kwargs = {}
        if platform.system() == "Windows":
            kwargs['creationflags'] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['start_new_session'] = True

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs
        )
        
        # Wait for health check
        url = f"http://{self.host}:{port}/v1"
        start_time = time.time()
        while time.time() - start_time < 120:
            try:
                res = requests.get(f"{url}/models", timeout=1)
                if res.status_code == 200:
                    return proc.pid, port, url
            except:
                pass
            
            if proc.poll() is not None:
                raise RuntimeError(f"Server process exited immediately with code {proc.returncode}")
            time.sleep(0.5)
                
        proc.terminate()
        raise TimeoutError(f"Server for {model_name} failed to become responsive.")

    def load_model(self, model_name: str) -> bool:
        """Thread-safe and Process-safe model loading."""
        if not self.global_lock_path.parent.exists():
            self.global_lock_path.parent.mkdir(parents=True)
            
        lock = FileLock(str(self.global_lock_path))
        try:
            with lock.acquire(timeout=60): 
                info = self._get_server_info(model_name)
                if info:
                    try:
                        self._get_registry_file(model_name).touch()
                    except:
                        pass
                    self.model_name = model_name
                    return True
                
                self._ensure_capacity_locked()
                pid, port, url = self._spawn_server_detached(model_name)
                
                reg_file = self._get_registry_file(model_name)
                with open(reg_file, 'w') as f:
                    json.dump({
                        "model_name": model_name, "pid": pid, "port": port, "url": url, "started_at": time.time()
                    }, f)
                
                self.model_name = model_name
                return True
        except Exception as e:
            ASCIIColors.error(f"Error loading model {model_name}: {e}")
            trace_exception(e)
            return False

    def _get_client(self, model_name: str = None) -> openai.OpenAI:
        target_model = model_name or self.model_name
        if not target_model:
            raise ValueError("No model specified.")
        info = self._get_server_info(target_model)
        if not info:
            if self.load_model(target_model):
                info = self._get_server_info(target_model)
            else:
                raise RuntimeError(f"Could not load model {target_model}")
        else:
            try:
                self._get_registry_file(target_model).touch()
            except:
                pass
        if not info:
             raise RuntimeError(f"Model {target_model} failed to load.")
        return openai.OpenAI(base_url=info['url'], api_key="sk-no-key-required")

    def _execute_with_retry(self, func: Callable, *args, **kwargs):
        retries = 60
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except openai.InternalServerError as e:
                if e.status_code == 503:
                    if i % 10 == 0:
                        ASCIIColors.warning(f"Model is loading (503). Waiting... ({i+1}/{retries})")
                    time.sleep(2)
                    continue
                raise e
            except openai.APIConnectionError:
                if i % 10 == 0:
                    ASCIIColors.warning(f"Connection error. Waiting... ({i+1}/{retries})")
                time.sleep(2)
                continue
        return func(*args, **kwargs)

    def generate_text(self, prompt: str, n_predict: int = None, stream: bool = False, **kwargs) -> Union[str, Dict]:
        try:
            client = self._get_client()
            def do_gen():
                return client.completions.create(
                    model=self.model_name, prompt=prompt,
                    max_tokens=n_predict if n_predict else 1024,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9), stream=stream,
                    extra_body={"top_k": kwargs.get("top_k", 40), "repeat_penalty": kwargs.get("repeat_penalty", 1.1), "n_predict": n_predict}
                )
            completion = self._execute_with_retry(do_gen)
            if stream:
                full_text = ""
                for chunk in completion:
                    content = chunk.choices[0].text
                    full_text += content
                    if kwargs.get("streaming_callback"):
                        if not kwargs["streaming_callback"](content, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
                return full_text
            else:
                return completion.choices[0].text
        except Exception as e:
            trace_exception(e)
            return {"status": False, "error": str(e)}

    def _chat(self, discussion: LollmsDiscussion, **kwargs) -> Union[str, Dict]:
        try:
            client = self._get_client()
            messages = discussion.export("openai_chat")
            def do_chat():
                return client.chat.completions.create(
                    model=self.model_name, messages=messages,
                    max_tokens=kwargs.get("n_predict", 1024),
                    temperature=kwargs.get("temperature", 0.7),
                    stream=kwargs.get("stream", False),
                    extra_body={"top_k": kwargs.get("top_k", 40), "repeat_penalty": kwargs.get("repeat_penalty", 1.1)}
                )
            response = self._execute_with_retry(do_chat)
            if kwargs.get("stream", True if kwargs.get("streaming_callback") else False):
                full_text = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    full_text += content
                    if kwargs.get("streaming_callback"):
                        if not kwargs["streaming_callback"](content, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
                return full_text
            else:
                return response.choices[0].message.content
        except Exception as e:
            trace_exception(e)
            return {"status": False, "error": str(e)}

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        if self.models_dir.exists():
            for f in self.models_dir.glob("*.gguf"):
                # Hide files explicitly containing 'mmproj' as they are not standalone models
                if "mmproj" in f.name.lower():
                    continue
                    
                if re.search(r'-\d{5}-of-\d{5}\.gguf$', f.name):
                    if "00001-of-" not in f.name: continue 
                models.append({"model_name": f.name, "owned_by": "local", "created": time.ctime(f.stat().st_ctime), "size": f.stat().st_size})
        return models

    def get_model_info(self) -> dict:
        info = {"name": BindingName, "version": "source-wrapper", "active_model": self.model_name}
        reg = self._get_server_info(self.model_name)
        if reg: info["host_address"] = reg['url']
        return info

    def tokenize(self, text: str) -> list:
        try:
            client = self._get_client() 
            url = client.base_url
            def do_tokenize():
                ep = f"{url}tokenize"
                res = requests.post(ep, json={"content": text})
                if res.status_code == 404:
                     res = requests.post(str(url).replace("/v1/", "/tokenize"), json={"content": text})
                if res.status_code == 503:
                    raise openai.InternalServerError("Loading model", response=res, body=None)
                return res
            res = self._execute_with_retry(do_tokenize)
            if res.status_code == 200: return res.json().get("tokens", [])
        except: pass
        return list(text)

    def detokenize(self, tokens: list) -> str:
        try:
            client = self._get_client()
            url = client.base_url
            def do_detokenize():
                ep = f"{url}detokenize"
                res = requests.post(ep, json={"tokens": tokens})
                if res.status_code == 404:
                     res = requests.post(str(url).replace("/v1/", "/detokenize"), json={"tokens": tokens})
                if res.status_code == 503:
                    raise openai.InternalServerError("Loading model", response=res, body=None)
                return res
            res = self._execute_with_retry(do_detokenize)
            if res.status_code == 200: return res.json().get("content", "")
        except: pass
        return "".join(map(str, tokens))

    def count_tokens(self, text: str) -> int: return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> List[float]:
        client = self._get_client()
        def do_embed():
            return client.embeddings.create(input=text, model=self.model_name)
        res = self._execute_with_retry(do_embed)
        return res.data[0].embedding
            
    def get_zoo(self) -> List[Dict[str, Any]]:
        return [
            # Ministral 3: High-performance edge model (3B)
            {
                "name": "Ministral-3-3B-Instruct-2512-GGUF",
                "description": "Mistral AI Ministral 3 3B Instruct (Bartowski Quant) - Efficient Edge Model",
                "size": "2.2 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF",
                "filename": "mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
            },
            # Devstral 2 Mini: Agentic coding specialist (24B)
            {
                "name": "Devstral-Small-2-24B-Instruct-GGUF",
                "description": "Mistral AI Devstral Small 2 24B Instruct (Bartowski Quant) - Coding Specialist",
                "size": "14.8 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF",
                "filename": "mistralai_Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf"
            },
            # Llama 4 Scout: Meta's efficient MoE (17B)
            {
                "name": "Llama-4-Scout-17B-Instruct-GGUF",
                "description": "Meta Llama 4 Scout 17B Instruct (Bartowski Quant) - 16-Expert MoE",
                "size": "11.2 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/meta-llama_Llama-4-Scout-17B-16E-Instruct-old-GGUF",
                "filename": "meta-llama_Llama-4-Scout-17B-16E-Instruct-Q4_K_M.gguf"
            },
            # Qwen 3 VL: Vision-Language with "Thinking" (32B)
            {
                "name": "Qwen3-VL-32B-Thinking-GGUF",
                "description": "Qwen 3 VL 32B Thinking (Bartowski Quant) - Vision CoT Reasoning",
                "size": "19.5 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/Qwen_Qwen3-VL-32B-Thinking-GGUF",
                "filename": "Qwen_Qwen3-VL-32B-Thinking-Q4_K_M.gguf"
            },
            # Qwen 3: Dense reasoning powerhouse (72B)
            {
                "name": "Qwen3-72B-Embiggened-GGUF",
                "description": "Qwen 3 72B Embiggened (Bartowski Quant) - Enhanced Reasoning Dense Model",
                "size": "43.1 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/cognitivecomputations_Qwen3-72B-Embiggened-GGUF",
                "filename": "Qwen3-72B-Embiggened-Q4_K_M.gguf"
            },
            # Devstral 2: Massive coding architecture (123B)
            {
                "name": "Devstral-2-123B-Instruct-GGUF",
                "description": "Mistral AI Devstral 2 123B Instruct (Bartowski Quant) - Heavy Duty Coding",
                "size": "71.4 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/mistralai_Devstral-2-123B-Instruct-2512-GGUF",
                "filename": "Devstral-2-123B-Instruct-2512-Q4_K_M.gguf"
            },
            # ChatGPT OSS: Open weights rival (120B)
            {
                "name": "ChatGPT-OSS-120B-GGUF",
                "description": "OpenAI GPT-OSS 120B (Bartowski Quant) - Open Weight Research Model",
                "size": "69.8 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/openai_gpt-oss-120b-GGUF",
                "filename": "gpt-oss-120b-Q4_K_M.gguf"
            },
            # DeepSeek V3: The MoE Giant (671B Base / 37B Active)
            {
                "name": "DeepSeek-V3-0324-GGUF",
                "description": "DeepSeek V3 0324 (Bartowski Quant) - 671B MoE",
                "size": "365 GB (Q4_K_M)",
                "type": "gguf",
                "link": "bartowski/deepseek-ai_DeepSeek-V3-0324-GGUF",
                "filename": "DeepSeek-V3-0324-Q4_K_M.gguf"
            }
        ]


    def download_from_zoo(self, index: int, progress_callback: Callable[[dict], None] = None) -> dict:
        zoo = self.get_zoo(); 
        if index < 0 or index >= len(zoo): return {"status": False, "message": "Index out of bounds"}
        item = zoo[index]
        return self.pull_model(item["link"], item.get("filename"), progress_callback=progress_callback)

    def pull_model(self, repo_id: str, filename: str, mmproj_repo_id: str = None, mmproj_filename: str = None, progress_callback: Callable[[dict], None] = None) -> dict:
        try:
            match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", filename)
            files = []
            if match:
                base, total = match.group(1), int(match.group(3))
                ASCIIColors.info(f"Detected multi-file model with {total} parts.")
                for i in range(1, total + 1): files.append(f"{base}-{i:05d}-of-{total:05d}.gguf")
            else:
                files.append(filename)
            paths = []
            for f in files:
                ASCIIColors.info(f"Downloading {f} from {repo_id}...")
                if progress_callback: progress_callback({"status": "downloading", "message": f"Downloading {f}", "completed": 0, "total": 100})
                p = hf_hub_download(repo_id=repo_id, filename=f, local_dir=self.models_dir, local_dir_use_symlinks=False, resume_download=True)
                paths.append(p)
                ASCIIColors.success(f"Downloaded {f}")
            
            if mmproj_filename:
                proj_repo = mmproj_repo_id if mmproj_repo_id else repo_id
                ASCIIColors.info(f"Downloading mmproj {mmproj_filename} from {proj_repo}...")
                hf_hub_download(repo_id=proj_repo, filename=mmproj_filename, local_dir=self.models_dir, local_dir_use_symlinks=False, resume_download=True)
                ASCIIColors.success(f"Downloaded mmproj {mmproj_filename}")
                # Automatically bind the model with its projector
                self.bind_multimodal_model(filename, mmproj_filename)
                
            msg = f"Successfully downloaded model: {filename}"
            if mmproj_filename: msg += f" and bound with projector: {mmproj_filename}"
            if progress_callback: progress_callback({"status": "success", "message": msg, "completed": 100, "total": 100})
            return {"status": True, "message": msg, "path": paths[0]}
        except Exception as e:
            trace_exception(e)
            return {"status": False, "error": str(e)}

    def cleanup_orphans_if_needed(self):
        pass

    def __del__(self):
        pass
