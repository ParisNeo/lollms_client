# lollms_client/llm_bindings/vllm/__init__.py

import os
import shutil
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict, Any, Tuple
import json
import threading
import gc
import importlib
import platform

def detect_os():
    system = platform.system()
    if system == "Windows":
        return "Windows"
    elif system == "Linux":
        return "Linux"
    elif system == "Darwin":
        return "macOS"
    else:
        return "Unknown OS"

if detect_os()=="Windows":
    raise Exception("Windows is not supported by vllm, use wsl")

# --- Package Management and Conditional Imports ---
try:
    # Pipmaster is assumed to be installed by the parent lollms_client.
    # We ensure specific packages for this binding.
    
    # Check if vllm is already importable to avoid re-running ensure_packages unnecessarily
    # on subsequent imports within the same session if it was successful once.
    _vllm_already_imported = 'vllm' in globals() or importlib.util.find_spec('vllm') is not None

    if not _vllm_already_imported:
        import pipmaster as pm # Assuming pipmaster is available
        pm.ensure_packages([
            "tensorrt_llm",
            "torch", 
            "transformers>=4.37.0",
            "huggingface_hub>=0.20.0",
            "pillow"
        ])
    
    from tensorrt_llm import LLM, SamplingParams
    from PIL import Image 
    import torch
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download, HfFileSystem, snapshot_download
    import vllm # To get __version__

    _vllm_deps_installed = True
    _vllm_installation_error = None
except Exception as e:
    _vllm_deps_installed = False
    _vllm_installation_error = e
    # Define placeholders if imports fail
    LLM, SamplingParams, Image, vllm_multimodal_utils = None, None, None, None
    torch, AutoTokenizer, hf_hub_download, HfFileSystem, snapshot_download, vllm = None, None, None, None, None, None


# --- LOLLMS Client Imports ---
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT # Assuming ELF_COMPLETION_FORMAT is in lollms_types
from ascii_colors import ASCIIColors, trace_exception


# --- Constants ---
BindingName = "VLLMBinding"
DEFAULT_models_folder = Path.home() / ".lollms" / "bindings_models" / "vllm_models"


# --- VLLM Engine Manager ---
class VLLMEngineManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not _vllm_deps_installed:
            raise RuntimeError(f"vLLM or its dependencies not installed. Cannot create VLLMEngineManager. Error: {_vllm_installation_error}")
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return
            # Stores: key -> (LLM_engine, tokenizer, ref_count, engine_kwargs_tuple_key)
            self._engines: Dict[Tuple, Tuple[Optional[LLM], Optional[Any], int, Tuple]] = {}
            self._engine_locks: Dict[Tuple, threading.Lock] = {} # Per-engine initialization lock
            self._initialized = True
            ASCIIColors.green("VLLMEngineManager initialized.")

    def _get_engine_config_key(self, resolved_model_path: Path, engine_params: Dict[str, Any]) -> Tuple:
        critical_params = [
            'tensor_parallel_size', 'quantization', 'dtype', 'max_model_len',
            'trust_remote_code', 'enforce_eager', 'gpu_memory_utilization',
            'swap_space', 'max_num_seqs', 'max_num_batched_tokens', 'tokenizer', 'tokenizer_mode',
            'image_input_type', 'image_token_id', 'image_feature_size', 'image_input_shape' # Common vision params
        ]
        key_parts = [str(resolved_model_path)]
        for param_name in sorted(critical_params):
            if param_name in engine_params:
                value = engine_params[param_name]
                # Make common mutable types hashable for the key
                if isinstance(value, list): value = tuple(value)
                elif isinstance(value, dict): value = tuple(sorted(value.items()))
                key_parts.append((param_name, value))
        return tuple(key_parts)

    def get_engine(self, 
                   resolved_model_path: Path, 
                   is_gguf: bool,
                   engine_params: Dict[str, Any]
                   ) -> Tuple[Optional[LLM], Optional[Any]]:
        
        engine_key = self._get_engine_config_key(resolved_model_path, engine_params)

        with self._lock:
            if engine_key not in self._engine_locks:
                self._engine_locks[engine_key] = threading.Lock()
        
        with self._engine_locks[engine_key]:
            with self._lock:
                if engine_key in self._engines:
                    llm_engine, tokenizer, ref_count, _ = self._engines[engine_key]
                    self._engines[engine_key] = (llm_engine, tokenizer, ref_count + 1, engine_key)
                    ASCIIColors.info(f"Reusing vLLM engine for {resolved_model_path.name}. Key: {engine_key}. Ref count: {ref_count + 1}")
                    return llm_engine, tokenizer

            ASCIIColors.info(f"Creating new vLLM engine for {resolved_model_path.name} with key: {engine_key}")
            try:
                llm_args = {"model": str(resolved_model_path), **engine_params}
                if is_gguf and "quantization" not in llm_args: # Only set if not overridden by user
                    llm_args["quantization"] = "gguf"
                
                new_llm_engine = LLM(**llm_args)
                new_tokenizer = None
                try:
                    if hasattr(new_llm_engine, 'get_tokenizer'):
                        new_tokenizer = new_llm_engine.get_tokenizer()
                    else: raise AttributeError("get_tokenizer not on LLM object.")
                except Exception as e_vllm_tok:
                    ASCIIColors.warning(f"vLLM engine tokenizer error ({e_vllm_tok}). Loading with AutoTokenizer.")
                    tok_path_hint = engine_params.get('tokenizer', str(resolved_model_path.parent if is_gguf else resolved_model_path))
                    if not Path(tok_path_hint).exists() and "/" not in tok_path_hint:
                         tok_path_hint = str(resolved_model_path.parent if is_gguf else resolved_model_path)
                    try:
                        new_tokenizer = AutoTokenizer.from_pretrained(
                            tok_path_hint, trust_remote_code=engine_params.get("trust_remote_code", False)
                        )
                    except Exception as e_hf_tok:
                        ASCIIColors.error(f"AutoTokenizer failed for {tok_path_hint}: {e_hf_tok}")
                
                with self._lock:
                    self._engines[engine_key] = (new_llm_engine, new_tokenizer, 1, engine_key)
                ASCIIColors.green(f"New vLLM engine for {resolved_model_path.name} created. Ref count: 1")
                return new_llm_engine, new_tokenizer
            
            except Exception as e:
                trace_exception(e)
                ASCIIColors.error(f"Failed to create vLLM engine for {resolved_model_path.name}: {e}")
                return None, None

    def release_engine(self, resolved_model_path: Path, engine_params: Dict[str, Any]):
        engine_key = self._get_engine_config_key(resolved_model_path, engine_params)
        with self._lock:
            if engine_key in self._engines:
                llm_engine, tokenizer, ref_count, _ = self._engines[engine_key]
                if ref_count <= 1:
                    ASCIIColors.info(f"Releasing vLLM engine for {resolved_model_path.name} (key: {engine_key}). Final reference.")
                    del self._engines[engine_key]
                    if engine_key in self._engine_locks: del self._engine_locks[engine_key]
                    del llm_engine
                    del tokenizer
                    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
                    gc.collect()
                    ASCIIColors.green(f"Engine for {resolved_model_path.name} removed.")
                else:
                    self._engines[engine_key] = (llm_engine, tokenizer, ref_count - 1, engine_key)
                    ASCIIColors.info(f"Decremented ref count for {resolved_model_path.name}. New: {ref_count - 1}")
            else:
                ASCIIColors.warning(f"Release called for non-managed engine key: {engine_key}")

if _vllm_deps_installed:
    engine_manager = VLLMEngineManager()
else:
    engine_manager = None


# --- Helper Functions ---
def is_hf_model_id(model_name: str) -> bool:
    return "/" in model_name and not Path(model_name).exists() and not model_name.endswith(".gguf")

def is_hf_gguf_model_id(model_name: str) -> bool:
    if "/" in model_name and model_name.endswith(".gguf"):
        return len(model_name.split("/")) > 1
    return False

def resolve_hf_model_path(model_id_or_gguf_id: str, models_base_path: Path) -> Path:
    if not _vllm_deps_installed: raise RuntimeError("Hugging Face utilities not available.")
    
    is_single_gguf = is_hf_gguf_model_id(model_id_or_gguf_id)
    
    if is_single_gguf:
        parts = model_id_or_gguf_id.split("/")
        repo_id, gguf_filename = "/".join(parts[:-1]), parts[-1]
        local_repo_name = repo_id.replace("/", "__")
        local_gguf_dir = models_base_path / local_repo_name
        local_gguf_path = local_gguf_dir / gguf_filename
        
        if not local_gguf_path.exists():
            ASCIIColors.info(f"Downloading GGUF {model_id_or_gguf_id} to {local_gguf_dir}...")
            local_gguf_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id=repo_id, filename=gguf_filename, local_dir=local_gguf_dir, local_dir_use_symlinks=False, resume_download=True)
        return local_gguf_path
    else:
        local_model_dir_name = model_id_or_gguf_id.replace("/", "__")
        local_model_path = models_base_path / local_model_dir_name
        if not local_model_path.exists() or not any(local_model_path.iterdir()):
            ASCIIColors.info(f"Downloading model repo {model_id_or_gguf_id} to {local_model_path}...")
            snapshot_download(repo_id=model_id_or_gguf_id, local_dir=local_model_path, local_dir_use_symlinks=False, resume_download=True)
        return local_model_path


# --- VLLM Binding Class ---
class VLLMBinding(LollmsLLMBinding):
    def __init__(self, 
                 **kwargs 
        ):
        if not _vllm_deps_installed:
            raise ImportError(f"vLLM or its dependencies not installed. Binding unusable. Error: {_vllm_installation_error}")
        if engine_manager is None:
             raise RuntimeError("VLLMEngineManager failed to initialize. Binding unusable.")
        models_folder = kwargs.get("models_folder")
        _models_folder = Path(models_folder) if models_folder is not None else DEFAULT_models_folder
        _models_folder.mkdir(parents=True, exist_ok=True)

        super().__init__(BindingName, **kwargs)
        self.models_folder= _models_folder
        self.model_name=kwargs.get("model_name", "")
        self.default_completion_format=kwargs.get("default_completion_format", ELF_COMPLETION_FORMAT.Chat)

        
        self.models_folder: Path = _models_folder
        self.llm_engine: Optional[LLM] = None
        self.tokenizer = None
        self.current_model_name_or_id: Optional[str] = None
        self.current_resolved_model_path: Optional[Path] = None
        self.current_engine_params: Optional[Dict[str, Any]] = None
        self.vllm_engine_kwargs_config = kwargs.copy()

        if self.model_name:
            try:
                self.load_model(self.model_name)
            except Exception as e:
                ASCIIColors.error(f"Auto-load model '{self.model_name}' failed: {e}")
                trace_exception(e)

    def _get_vllm_engine_params_for_load(self) -> Dict[str, Any]:
        params = self.vllm_engine_kwargs_config.copy()
        if torch and torch.cuda.is_available():
            params.setdefault('tensor_parallel_size', torch.cuda.device_count())
            params.setdefault('gpu_memory_utilization', 0.90)
            params.setdefault('dtype', 'auto')
        else:
            params.setdefault('tensor_parallel_size', 1)
            params.setdefault('gpu_memory_utilization', 0)
            params.setdefault('enforce_eager', True)
            if not (torch and torch.cuda.is_available()): ASCIIColors.warning("No CUDA GPU by PyTorch, vLLM on CPU or may fail.")
        params.setdefault('trust_remote_code', False) # Important default
        return params

    def load_model(self, model_name_or_id: str) -> bool:
        ASCIIColors.info(f"Binding {id(self)} loading model: {model_name_or_id}")
        self.close() # Release any existing model held by this instance

        resolved_model_path: Path
        is_gguf_model = False
        effective_engine_params = self._get_vllm_engine_params_for_load()
        
        potential_local_path = Path(model_name_or_id)
        if potential_local_path.is_absolute():
            if not potential_local_path.exists():
                ASCIIColors.error(f"Absolute path not found: {potential_local_path}")
                return False
            resolved_model_path = potential_local_path
        else:
            path_in_models_dir = self.models_folder / model_name_or_id
            if path_in_models_dir.exists():
                resolved_model_path = path_in_models_dir
            elif is_hf_model_id(model_name_or_id) or is_hf_gguf_model_id(model_name_or_id):
                try:
                    resolved_model_path = resolve_hf_model_path(model_name_or_id, self.models_folder)
                except Exception as e:
                    ASCIIColors.error(f"HF model resolve/download failed for {model_name_or_id}: {e}"); return False
            else:
                ASCIIColors.error(f"Model '{model_name_or_id}' not found locally or as HF ID."); return False

        if resolved_model_path.is_file() and resolved_model_path.suffix.lower() == ".gguf":
            is_gguf_model = True
        elif not resolved_model_path.is_dir():
            ASCIIColors.error(f"Resolved path {resolved_model_path} not valid model."); return False

        self.llm_engine, self.tokenizer = engine_manager.get_engine(resolved_model_path, is_gguf_model, effective_engine_params)

        if self.llm_engine:
            self.current_model_name_or_id = model_name_or_id
            self.current_resolved_model_path = resolved_model_path
            self.current_engine_params = effective_engine_params
            self.model_name = model_name_or_id # Update superclass
            ASCIIColors.green(f"Binding {id(self)} obtained engine for: {model_name_or_id}")
            if not self.tokenizer: ASCIIColors.warning("Tokenizer unavailable for current model.")
            return True
        else:
            ASCIIColors.error(f"Binding {id(self)} failed to get engine for: {model_name_or_id}")
            self.close() # Clear any partial state
            return False

    def generate_text(self, 
                     prompt: str,
                     images: Optional[List[str]] = None, 
                     system_prompt: str = "",
                     n_predict: Optional[int] = 1024, 
                     stream: bool = False, # vLLM's generate is blocking, stream is pseudo
                     temperature: float = 0.7,
                     top_k: int = 50,
                     top_p: float = 0.95,
                     repeat_penalty: float = 1.1,
                     repeat_last_n: int = 64, # Note: vLLM applies penalty to full context
                     seed: Optional[int] = None,
                     n_threads: int = 8, # Note: vLLM manages its own threading/parallelism
                     streaming_callback: Optional[Callable[[str, int], bool]] = None,
                     split:Optional[bool]=False, # put to true if the prompt is a discussion
                     user_keyword:Optional[str]="!@>user:",
                     ai_keyword:Optional[str]="!@>assistant:",
                     ) -> Union[str, Dict[str, any]]:
        if not self.llm_engine: return {"status": False, "error": "Engine not loaded."}

        sampling_dict = {
            "temperature": float(temperature) if float(temperature) > 0.001 else 0.001, # Temp > 0
            "top_p": float(top_p), "top_k": int(top_k) if top_k > 0 else -1,
            "max_tokens": int(n_predict) if n_predict is not None else 1024,
            "repetition_penalty": float(repeat_penalty),
        }
        if sampling_dict["temperature"] <= 0.001 and sampling_dict["top_k"] !=1 : # Greedy like
            sampling_dict["top_k"] = 1
            sampling_dict["temperature"] = 1.0 # Valid combination for greedy

        if seed is not None: sampling_dict["seed"] = int(seed)
        
        sampling_params = SamplingParams(**sampling_dict)
        gen_kwargs = {}
        
        if images:
            if not self.tokenizer: return {"status": False, "error": "Tokenizer needed for multimodal."}
            # Vision model image processing is complex and model-specific.
            # This is a simplified placeholder for LLaVA-like models.
            # Requires vLLM >= 0.4.0 and appropriate model/engine_params.
            try:
                pil_images = [Image.open(img_path).convert('RGB') for img_path in images]
                
                # The prompt might need an image token, e.g. <image>. This should be part of `self.current_engine_params`
                image_token_str = self.current_engine_params.get("image_token_str", "<image>") 
                if image_token_str not in prompt and images:
                     prompt = f"{image_token_str}\n{prompt}"
                
                # This is a simplified view. `process_multimodal_inputs` in vLLM is more robust.
                # The structure of multi_modal_data can vary.
                if len(pil_images) == 1: mm_data_content = pil_images[0]
                else: mm_data_content = pil_images
                
                # For vLLM, prompts can be text or token IDs.
                # If providing multi_modal_data, usually prompt_token_ids are also needed.
                # This can get complex as it depends on how the model expects images to be interleaved.
                # For a simple case where image comes first:
                encoded_prompt_ids = self.tokenizer.encode(system_prompt+"\n"+prompt if system_prompt else prompt)
                gen_kwargs["prompt_token_ids"] = [encoded_prompt_ids] # List of lists
                gen_kwargs["multi_modal_data"] = [{"image": mm_data_content}] # List of dicts
                gen_kwargs["prompts"] = None # Don't use prompts if prompt_token_ids is used
                ASCIIColors.info("Prepared basic multimodal inputs.")
            except Exception as e_mm:
                return {"status": False, "error": f"Multimodal prep error: {e_mm}"}
        else:
            gen_kwargs["prompts"] = [system_prompt+"\n"+prompt if system_prompt else prompt]

        try:
            outputs = self.llm_engine.generate(**gen_kwargs, sampling_params=sampling_params)
            full_response_text = outputs[0].outputs[0].text
            if stream and streaming_callback:
                if not streaming_callback(full_response_text, MSG_TYPE.MSG_TYPE_CHUNK):
                    ASCIIColors.info("Streaming callback stopped (pseudo-stream).")
            return full_response_text
        except Exception as e:
            trace_exception(e); return {"status": False, "error": f"vLLM generation error: {e}"}

    def tokenize(self, text: str) -> List[int]:
        if not self.tokenizer: ASCIIColors.warning("Tokenizer unavailable."); return [ord(c) for c in text]
        try:
            encoded = self.tokenizer.encode(text)
            return encoded.ids if hasattr(encoded, 'ids') else encoded
        except Exception as e: trace_exception(e); return []

    def detokenize(self, tokens: List[int]) -> str:
        if not self.tokenizer: ASCIIColors.warning("Tokenizer unavailable."); return "".join(map(chr, tokens)) # Crude fallback
        try: return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e: trace_exception(e); return ""
        
    def count_tokens(self, text: str) -> int:
        if not self.tokenizer: return len(text)
        return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> list:
        raise NotImplementedError("VLLMBinding does not provide generic text embedding.")

    def get_model_info(self) -> dict:
        info = {
            "binding_name": self.binding_name,
            "vllm_version": vllm.__version__ if vllm else "N/A",
            "models_folder": str(self.models_folder),
            "loaded_model_name_or_id": self.current_model_name_or_id,
            "resolved_model_path": str(self.current_resolved_model_path) if self.current_resolved_model_path else None,
            "engine_parameters_used": self.current_engine_params,
            "supports_structured_output": False, # Can be True with outlines, not basic
            "supports_vision": "multi_modal_data" in LLM.generate.__annotations__ if LLM else False
        }
        if self.llm_engine and hasattr(self.llm_engine, 'llm_engine') and hasattr(self.llm_engine.llm_engine, 'model_config'):
            cfg = self.llm_engine.llm_engine.model_config
            hf_cfg = getattr(cfg, 'hf_config', None)
            info["loaded_model_config_details"] = {
                "model_type": getattr(hf_cfg, 'model_type', getattr(cfg, 'model_type', "N/A")),
                "vocab_size": getattr(hf_cfg, 'vocab_size', getattr(cfg, 'vocab_size', "N/A")),
                "max_model_len": getattr(cfg, 'max_model_len', "N/A"),
                "quantization": getattr(self.llm_engine.llm_engine, 'quantization_method', "N/A"),
                "dtype": str(getattr(cfg, 'dtype', "N/A")),
            }
        return info

    def listModels(self) -> List[Dict[str, Any]]:
        local_models = []
        if not self.models_folder.exists(): return []
        for item_path in self.models_folder.rglob('*'):
            try:
                model_info = {"model_name": None, "path": str(item_path), "type": None, "size_gb": None}
                if item_path.is_dir() and ((item_path / "config.json").exists() or list(item_path.glob("*.safetensors"))):
                    is_sub_dir = any(Path(m["path"]) == item_path.parent for m in local_models if m["type"] == "HuggingFace Directory")
                    if is_sub_dir: continue
                    model_info.update({
                        "model_name": item_path.name, "type": "HuggingFace Directory",
                        "size_gb": round(sum(f.stat().st_size for f in item_path.glob('**/*') if f.is_file()) / (1024**3), 2)
                    })
                    local_models.append(model_info)
                elif item_path.is_file() and item_path.suffix.lower() == ".gguf":
                    model_info.update({
                        "model_name": str(item_path.relative_to(self.models_folder)), "type": "GGUF File",
                        "size_gb": round(item_path.stat().st_size / (1024**3), 2)
                    })
                    local_models.append(model_info)
            except Exception as e: ASCIIColors.warning(f"Error processing {item_path}: {e}")
        return local_models

    def __del__(self):
        self.close()

    def close(self):
        if self.llm_engine and self.current_resolved_model_path and self.current_engine_params:
            ASCIIColors.info(f"Binding {id(self)} close(). Releasing engine for: {self.current_resolved_model_path.name}")
            engine_manager.release_engine(self.current_resolved_model_path, self.current_engine_params)
        self.llm_engine = None
        self.tokenizer = None
        self.current_model_name_or_id = None
        self.current_resolved_model_path = None
        self.current_engine_params = None
        self.model_name = ""


# --- Exports for LOLLMS ---
__all__ = ["VLLMBinding", "BindingName"]


# --- Main Test Block (Example Usage) ---
if __name__ == '__main__':
    if not _vllm_deps_installed:
        print(f"{ASCIIColors.RED}VLLM dependencies not met. Skipping tests. Error: {_vllm_installation_error}{ASCIIColors.RESET}")
        exit()

    ASCIIColors.yellow("--- VLLMBinding Test ---")
    test_models_dir = DEFAULT_models_folder / "test_run_vllm_binding"
    test_models_dir.mkdir(parents=True, exist_ok=True)
    ASCIIColors.info(f"Using test models directory: {test_models_dir}")

    # Choose small models for testing to save time/resources
    # test_hf_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    test_hf_id = "microsoft/phi-2" # Needs trust_remote_code=True
    # test_gguf_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    test_gguf_id = "bartowski/Phi-2-GGUF/phi-2.Q2_K.gguf" # May need tokenizer="microsoft/phi-2"

    common_engine_args = {"trust_remote_code": True, "gpu_memory_utilization": 0.5} # Lower util for testing

    def test_binding_instance(name: str, model_id_to_load: str, specific_engine_args: Dict = {}):
        ASCIIColors.cyan(f"\n--- Testing Instance: {name} with Model: {model_id_to_load} ---")
        final_engine_args = {**common_engine_args, **specific_engine_args}
        binding = None
        try:
            binding = VLLMBinding(models_folder=test_models_dir, **final_engine_args)
            if binding.load_model(model_id_to_load):
                ASCIIColors.green(f"Model {binding.current_model_name_or_id} loaded by {name}.")
                info = binding.get_model_info()
                ASCIIColors.magenta(f"Model Info for {name}: {json.dumps(info['loaded_model_config_details'] if 'loaded_model_config_details' in info else 'N/A', indent=2, default=str)}")
                
                test_prompt = "What is the main purpose of a CPU in a computer?"
                if "phi-2" in model_id_to_load.lower(): # Phi-2 uses a specific prompt format
                    test_prompt = f"Instruct: {test_prompt}\nOutput:"

                ASCIIColors.info(f"Prompt for {name}: {test_prompt}")
                response = binding.generate_text(test_prompt, n_predict=50, temperature=0.1)
                if isinstance(response, str): ASCIIColors.green(f"Response from {name}: {response}")
                else: ASCIIColors.error(f"Generation failed for {name}: {response}")
                
                tokens = binding.tokenize("Test tokenization.")
                ASCIIColors.info(f"Token count for {name} ('Test tokenization.'): {len(tokens)}")

            else:
                ASCIIColors.error(f"Failed to load model {model_id_to_load} for {name}.")
        except Exception as e:
            ASCIIColors.error(f"Error during test for {name} with {model_id_to_load}: {e}")
            trace_exception(e)
        finally:
            if binding:
                binding.close()
                ASCIIColors.info(f"Closed binding for {name}.")
            # After closing a binding, the engine_manager ref count should decrease.
            # If it was the last reference, the engine should be removed from manager.
            # This can be verified by checking engine_manager._engines (for debugging)
            # print(f"DEBUG: Engines in manager after closing {name}: {engine_manager._engines.keys()}")

    # Test different models
    test_binding_instance("HF_Phi2_Instance1", test_hf_id)
    test_binding_instance("GGUF_Phi2_Instance", test_gguf_id, specific_engine_args={"tokenizer": "microsoft/phi-2"})

    # Test sharing: Two instances requesting the same model config
    ASCIIColors.cyan("\n--- Testing Model Sharing (Two instances, same HF model) ---")
    args_for_shared = {**common_engine_args, "max_model_len": 2048} # Add a param to make key specific
    binding_A = VLLMBinding(models_folder=test_models_dir, **args_for_shared)
    binding_B = VLLMBinding(models_folder=test_models_dir, **args_for_shared)
    
    loaded_A = binding_A.load_model(test_hf_id)
    if loaded_A: ASCIIColors.green(f"Binding A loaded {test_hf_id}. Manager should have 1 ref.")
    # print(f"DEBUG: Engines after A loads: {engine_manager._engines.keys()}") # For debug
    
    loaded_B = binding_B.load_model(test_hf_id) # Should reuse the engine loaded by A
    if loaded_B: ASCIIColors.green(f"Binding B loaded {test_hf_id}. Manager should have 2 refs for this engine.")
    # print(f"DEBUG: Engines after B loads: {engine_manager._engines.keys()}") # For debug

    if loaded_A:
        resp_A = binding_A.generate_text(f"Instruct: Hello from A!\nOutput:", n_predict=10)
        ASCIIColors.info(f"Response from A (shared model): {resp_A}")
    if loaded_B:
        resp_B = binding_B.generate_text(f"Instruct: Hello from B!\nOutput:", n_predict=10)
        ASCIIColors.info(f"Response from B (shared model): {resp_B}")

    binding_A.close()
    ASCIIColors.info("Binding A closed. Manager should have 1 ref left for this engine.")
    # print(f"DEBUG: Engines after A closes: {engine_manager._engines.keys()}") # For debug
    binding_B.close()
    ASCIIColors.info("Binding B closed. Manager should have 0 refs, engine should be removed.")
    # print(f"DEBUG: Engines after B closes: {engine_manager._engines.keys()}") # For debug

    # Vision Test (Conceptual - requires a real vision model and setup)
    ASCIIColors.cyan("\n--- Conceptual Vision Test ---")
    # test_vision_model_id = "llava-hf/llava-1.5-7b-hf" # Example LLaVA model
    # vision_args = {**common_engine_args, "image_input_type": "pixel_values", "image_token_id": 32000, "image_feature_size":576}
    # try:
    #     # Create a dummy image
    #     dummy_img_path = "dummy_vision_test.png"
    #     img = Image.new('RGB', (224, 224), color = 'blue')
    #     img.save(dummy_img_path)
    #     binding_vision = VLLMBinding(models_folder=test_models_dir, **vision_args)
    #     if binding_vision.load_model(test_vision_model_id):
    #          # Prompt for LLaVA often includes <image>
    #          vision_prompt = "USER: <image>\nWhat is in this image?\nASSISTANT:"
    #          response = binding_vision.generate_text(vision_prompt, images=[dummy_img_path], n_predict=30)
    #          ASCIIColors.green(f"Vision response: {response}")
    #     else:
    #          ASCIIColors.warning(f"Could not load vision model {test_vision_model_id}")
    #     if Path(dummy_img_path).exists(): Path(dummy_img_path).unlink()
    # except Exception as e_vis:
    #     ASCIIColors.warning(f"Vision test block skipped or failed: {e_vis}. This often requires specific model and VRAM.")


    ASCIIColors.yellow("\n--- VLLMBinding Test Finished ---")
    # Optional: Clean up test directory
    # import shutil
    # if input(f"Clean up {test_models_dir}? (y/N): ").lower() == 'y':
    #    shutil.rmtree(test_models_dir)
    #    ASCIIColors.info(f"Cleaned up {test_models_dir}")