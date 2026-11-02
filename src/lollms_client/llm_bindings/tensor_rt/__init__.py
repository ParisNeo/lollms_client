# lollms_client/llm_bindings/tensor_rt/__init__.py
"""
TensorRT-LLM binding for lollms_client.

Provides local inference via NVIDIA TensorRT-LLM. Requires Linux (Windows is
not supported by TensorRT-LLM). Models are loaded from a local folder or
downloaded from Hugging Face.
"""

import gc
import importlib
import json
import os
import platform
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT, MSG_TYPE

BindingName = "TensorRTLLMBinding"
DEFAULT_MODELS_FOLDER = Path.home() / ".lollms" / "bindings_models" / "tensorrt_llm_models"


def _detect_os() -> str:
    system = platform.system()
    if system == "Windows":
        return "Windows"
    if system == "Linux":
        return "Linux"
    if system == "Darwin":
        return "macOS"
    return "Unknown OS"


if _detect_os() == "Windows":
    raise Exception("Windows is not supported by TensorRT-LLM. Please use WSL2 or Linux.")


# ── Conditional imports ──────────────────────────────────────────────────────
try:
    _trt_already_imported = importlib.util.find_spec("tensorrt_llm") is not None
    if not _trt_already_imported:
        import pipmaster as pm
        pm.ensure_packages([
            "tensorrt_llm",
            "torch",
            "transformers>=4.37.0",
            "huggingface_hub>=0.20.0",
            "pillow",
        ])

    from tensorrt_llm import LLM, SamplingParams
    from PIL import Image
    import torch
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download, snapshot_download
    import tensorrt_llm as _trt_module

    _trt_deps_installed = True
    _trt_installation_error: Optional[Exception] = None
except Exception as e:
    _trt_deps_installed = False
    _trt_installation_error = e
    LLM = SamplingParams = Image = None
    torch = AutoTokenizer = hf_hub_download = snapshot_download = None
    _trt_module = None


# ── Engine manager (shared across binding instances) ────────────────────────
class TensorRTEngineManager:
    _instance: Optional["TensorRTEngineManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not _trt_deps_installed:
            raise RuntimeError(
                f"TensorRT-LLM or its dependencies are not installed. "
                f"Error: {_trt_installation_error}"
            )
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        with self._lock:
            if getattr(self, "_initialized", False):
                return
            self._engines: Dict[Tuple, Tuple[Optional[Any], Optional[Any], int, Tuple]] = {}
            self._engine_locks: Dict[Tuple, threading.Lock] = {}
            self._initialized = True
            ASCIIColors.green("TensorRTEngineManager initialized.")

    def _get_engine_config_key(self, resolved_model_path: Path, engine_params: Dict[str, Any]) -> Tuple:
        critical_params = [
            "tensor_parallel_size", "quantization", "dtype", "max_model_len",
            "trust_remote_code", "enforce_eager", "gpu_memory_utilization",
            "swap_space", "max_num_seqs", "max_num_batched_tokens",
            "tokenizer", "tokenizer_mode",
        ]
        key_parts: List[Any] = [str(resolved_model_path)]
        for param_name in sorted(critical_params):
            if param_name in engine_params:
                value = engine_params[param_name]
                if isinstance(value, list):
                    value = tuple(value)
                elif isinstance(value, dict):
                    value = tuple(sorted(value.items()))
                key_parts.append((param_name, value))
        return tuple(key_parts)

    def get_engine(
        self,
        resolved_model_path: Path,
        is_gguf: bool,
        engine_params: Dict[str, Any],
    ) -> Tuple[Optional[Any], Optional[Any]]:
        engine_key = self._get_engine_config_key(resolved_model_path, engine_params)

        with self._lock:
            if engine_key not in self._engine_locks:
                self._engine_locks[engine_key] = threading.Lock()

        with self._engine_locks[engine_key]:
            with self._lock:
                if engine_key in self._engines:
                    llm_engine, tokenizer, ref_count, _ = self._engines[engine_key]
                    self._engines[engine_key] = (llm_engine, tokenizer, ref_count + 1, engine_key)
                    ASCIIColors.info(
                        f"Reusing TensorRT-LLM engine for {resolved_model_path.name}. "
                        f"Ref count: {ref_count + 1}"
                    )
                    return llm_engine, tokenizer

            ASCIIColors.info(f"Creating new TensorRT-LLM engine for {resolved_model_path.name}...")
            try:
                llm_args = {"model": str(resolved_model_path), **engine_params}
                if is_gguf and "quantization" not in llm_args:
                    llm_args["quantization"] = "gguf"

                new_llm_engine = LLM(**llm_args)
                new_tokenizer = None
                try:
                    if hasattr(new_llm_engine, "get_tokenizer"):
                        new_tokenizer = new_llm_engine.get_tokenizer()
                    else:
                        raise AttributeError("get_tokenizer not on LLM object.")
                except Exception as e_tok:
                    ASCIIColors.warning(
                        f"TensorRT-LLM tokenizer error ({e_tok}). Loading with AutoTokenizer."
                    )
                    tok_path_hint = engine_params.get(
                        "tokenizer",
                        str(resolved_model_path.parent if is_gguf else resolved_model_path),
                    )
                    if not Path(tok_path_hint).exists() and "/" not in tok_path_hint:
                        tok_path_hint = str(
                            resolved_model_path.parent if is_gguf else resolved_model_path
                        )
                    try:
                        new_tokenizer = AutoTokenizer.from_pretrained(
                            tok_path_hint,
                            trust_remote_code=engine_params.get("trust_remote_code", False),
                        )
                    except Exception as e_hf_tok:
                        ASCIIColors.error(f"AutoTokenizer failed for {tok_path_hint}: {e_hf_tok}")

                with self._lock:
                    self._engines[engine_key] = (new_llm_engine, new_tokenizer, 1, engine_key)
                ASCIIColors.green(
                    f"New TensorRT-LLM engine for {resolved_model_path.name} created."
                )
                return new_llm_engine, new_tokenizer
            except Exception as e:
                trace_exception(e)
                ASCIIColors.error(
                    f"Failed to create TensorRT-LLM engine for {resolved_model_path.name}: {e}"
                )
                return None, None

    def release_engine(self, resolved_model_path: Path, engine_params: Dict[str, Any]):
        engine_key = self._get_engine_config_key(resolved_model_path, engine_params)
        with self._lock:
            if engine_key in self._engines:
                llm_engine, tokenizer, ref_count, _ = self._engines[engine_key]
                if ref_count <= 1:
                    ASCIIColors.info(
                        f"Releasing TensorRT-LLM engine for {resolved_model_path.name}."
                    )
                    del self._engines[engine_key]
                    if engine_key in self._engine_locks:
                        del self._engine_locks[engine_key]
                    del llm_engine
                    del tokenizer
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    self._engines[engine_key] = (llm_engine, tokenizer, ref_count - 1, engine_key)
                    ASCIIColors.info(
                        f"Decremented ref count for {resolved_model_path.name}. "
                        f"New: {ref_count - 1}"
                    )
            else:
                ASCIIColors.warning(f"Release called for non-managed engine key: {engine_key}")


if _trt_deps_installed:
    engine_manager: Optional[TensorRTEngineManager] = TensorRTEngineManager()
else:
    engine_manager = None


# ── Helpers ─────────────────────────────────────────────────────────────────
def is_hf_model_id(model_name: str) -> bool:
    return "/" in model_name and not Path(model_name).exists() and not model_name.endswith(".gguf")


def is_hf_gguf_model_id(model_name: str) -> bool:
    if "/" in model_name and model_name.endswith(".gguf"):
        return len(model_name.split("/")) > 1
    return False


def resolve_hf_model_path(model_id_or_gguf_id: str, models_base_path: Path) -> Path:
    if not _trt_deps_installed:
        raise RuntimeError("Hugging Face utilities not available.")

    if is_hf_gguf_model_id(model_id_or_gguf_id):
        parts = model_id_or_gguf_id.split("/")
        repo_id, gguf_filename = "/".join(parts[:-1]), parts[-1]
        local_repo_name = repo_id.replace("/", "__")
        local_gguf_dir = models_base_path / local_repo_name
        local_gguf_path = local_gguf_dir / gguf_filename
        if not local_gguf_path.exists():
            ASCIIColors.info(f"Downloading GGUF {model_id_or_gguf_id} to {local_gguf_dir}...")
            local_gguf_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=gguf_filename,
                local_dir=local_gguf_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        return local_gguf_path

    local_model_dir_name = model_id_or_gguf_id.replace("/", "__")
    local_model_path = models_base_path / local_model_dir_name
    if not local_model_path.exists() or not any(local_model_path.iterdir()):
        ASCIIColors.info(f"Downloading model repo {model_id_or_gguf_id} to {local_model_path}...")
        snapshot_download(
            repo_id=model_id_or_gguf_id,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    return local_model_path


# ── Binding class ───────────────────────────────────────────────────────────
class TensorRTLLMBinding(LollmsLLMBinding):
    def __init__(self, **kwargs):
        if not _trt_deps_installed:
            raise ImportError(
                f"TensorRT-LLM or its dependencies are not installed. "
                f"Error: {_trt_installation_error}"
            )
        if engine_manager is None:
            raise RuntimeError("TensorRTEngineManager failed to initialize.")

        models_folder = kwargs.get("models_folder")
        _models_folder = Path(models_folder) if models_folder is not None else DEFAULT_MODELS_FOLDER
        _models_folder.mkdir(parents=True, exist_ok=True)

        super().__init__(BindingName, **kwargs)
        self.models_folder: Path = _models_folder
        self.model_name: str = kwargs.get("model_name", "")
        self.default_completion_format = kwargs.get(
            "default_completion_format", ELF_COMPLETION_FORMAT.Chat
        )

        self.llm_engine: Optional[Any] = None
        self.tokenizer = None
        self.current_model_name_or_id: Optional[str] = None
        self.current_resolved_model_path: Optional[Path] = None
        self.current_engine_params: Optional[Dict[str, Any]] = None
        self.trt_engine_kwargs_config = kwargs.copy()

        if self.model_name:
            try:
                self.load_model(self.model_name)
            except Exception as e:
                ASCIIColors.error(f"Auto-load model '{self.model_name}' failed: {e}")
                trace_exception(e)

    def _get_engine_params_for_load(self) -> Dict[str, Any]:
        params = self.trt_engine_kwargs_config.copy()
        if torch and torch.cuda.is_available():
            params.setdefault("tensor_parallel_size", torch.cuda.device_count())
            params.setdefault("gpu_memory_utilization", 0.90)
            params.setdefault("dtype", "auto")
        else:
            params.setdefault("tensor_parallel_size", 1)
            params.setdefault("gpu_memory_utilization", 0)
            params.setdefault("enforce_eager", True)
            ASCIIColors.warning("No CUDA GPU detected by PyTorch; TensorRT-LLM may fail.")
        params.setdefault("trust_remote_code", False)
        return params

    def load_model(self, model_name_or_id: str) -> bool:
        ASCIIColors.info(f"Binding {id(self)} loading model: {model_name_or_id}")
        self.close()

        resolved_model_path: Path
        is_gguf_model = False
        effective_engine_params = self._get_engine_params_for_load()

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
                    ASCIIColors.error(f"HF model resolve/download failed for {model_name_or_id}: {e}")
                    return False
            else:
                ASCIIColors.error(f"Model '{model_name_or_id}' not found locally or as HF ID.")
                return False

        if resolved_model_path.is_file() and resolved_model_path.suffix.lower() == ".gguf":
            is_gguf_model = True
        elif not resolved_model_path.is_dir():
            ASCIIColors.error(f"Resolved path {resolved_model_path} is not a valid model.")
            return False

        self.llm_engine, self.tokenizer = engine_manager.get_engine(
            resolved_model_path, is_gguf_model, effective_engine_params
        )

        if self.llm_engine:
            self.current_model_name_or_id = model_name_or_id
            self.current_resolved_model_path = resolved_model_path
            self.current_engine_params = effective_engine_params
            self.model_name = model_name_or_id
            ASCIIColors.green(f"Binding {id(self)} obtained engine for: {model_name_or_id}")
            if not self.tokenizer:
                ASCIIColors.warning("Tokenizer unavailable for current model.")
            return True

        ASCIIColors.error(f"Binding {id(self)} failed to get engine for: {model_name_or_id}")
        self.close()
        return False

    def generate_text(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: str = "",
        n_predict: Optional[int] = 1024,
        stream: bool = False,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
        seed: Optional[int] = None,
        n_threads: int = 8,
        streaming_callback: Optional[Callable[[str, int], bool]] = None,
        split: Optional[bool] = False,
        user_keyword: Optional[str] = "!@>user:",
        ai_keyword: Optional[str] = "!@>assistant:",
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        if not self.llm_engine:
            return {"status": False, "error": "Engine not loaded."}

        sampling_dict: Dict[str, Any] = {
            "temperature": float(temperature) if float(temperature) > 0.001 else 0.001,
            "top_p": float(top_p),
            "top_k": int(top_k) if top_k > 0 else -1,
            "max_tokens": int(n_predict) if n_predict is not None else 1024,
            "repetition_penalty": float(repeat_penalty),
        }
        if sampling_dict["temperature"] <= 0.001 and sampling_dict["top_k"] != 1:
            sampling_dict["top_k"] = 1
            sampling_dict["temperature"] = 1.0
        if seed is not None:
            sampling_dict["seed"] = int(seed)

        sampling_params = SamplingParams(**sampling_dict)
        gen_kwargs: Dict[str, Any] = {}

        if images:
            if not self.tokenizer:
                return {"status": False, "error": "Tokenizer needed for multimodal."}
            try:
                pil_images = [Image.open(img_path).convert("RGB") for img_path in images]
                image_token_str = (self.current_engine_params or {}).get("image_token_str", "<image>")
                if image_token_str not in prompt and images:
                    prompt = f"{image_token_str}\n{prompt}"
                mm_data_content = pil_images[0] if len(pil_images) == 1 else pil_images
                encoded_prompt_ids = self.tokenizer.encode(
                    system_prompt + "\n" + prompt if system_prompt else prompt
                )
                gen_kwargs["prompt_token_ids"] = [encoded_prompt_ids]
                gen_kwargs["multi_modal_data"] = [{"image": mm_data_content}]
                gen_kwargs["prompts"] = None
            except Exception as e_mm:
                return {"status": False, "error": f"Multimodal prep error: {e_mm}"}
        else:
            gen_kwargs["prompts"] = [system_prompt + "\n" + prompt if system_prompt else prompt]

        try:
            outputs = self.llm_engine.generate(**gen_kwargs, sampling_params=sampling_params)
            full_response_text = outputs[0].outputs[0].text
            if stream and streaming_callback:
                if not streaming_callback(full_response_text, MSG_TYPE.MSG_TYPE_CHUNK):
                    ASCIIColors.info("Streaming callback stopped (pseudo-stream).")
            return full_response_text
        except Exception as e:
            trace_exception(e)
            return {"status": False, "error": f"TensorRT-LLM generation error: {e}"}

    def tokenize(self, text: str) -> List[int]:
        if not self.tokenizer:
            ASCIIColors.warning("Tokenizer unavailable.")
            return [ord(c) for c in text]
        try:
            encoded = self.tokenizer.encode(text)
            return encoded.ids if hasattr(encoded, "ids") else encoded
        except Exception as e:
            trace_exception(e)
            return []

    def detokenize(self, tokens: List[int]) -> str:
        if not self.tokenizer:
            ASCIIColors.warning("Tokenizer unavailable.")
            return "".join(map(chr, tokens))
        try:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e:
            trace_exception(e)
            return ""

    def count_tokens(self, text: str) -> int:
        if not self.tokenizer:
            return len(text)
        return len(self.tokenize(text))

    def embed(self, text: str, **kwargs) -> list:
        raise NotImplementedError("TensorRTLLMBinding does not provide generic text embedding.")

    def get_model_info(self) -> dict:
        info: Dict[str, Any] = {
            "binding_name": self.binding_name,
            "tensorrt_llm_version": getattr(_trt_module, "__version__", "N/A") if _trt_module else "N/A",
            "models_folder": str(self.models_folder),
            "loaded_model_name_or_id": self.current_model_name_or_id,
            "resolved_model_path": str(self.current_resolved_model_path) if self.current_resolved_model_path else None,
            "engine_parameters_used": self.current_engine_params,
            "supports_structured_output": False,
            "supports_vision": bool(self.current_engine_params and "image_token_str" in self.current_engine_params),
        }
        return info

    def list_models(self) -> List[Dict[str, Any]]:
        local_models: List[Dict[str, Any]] = []
        if not self.models_folder.exists():
            return []
        for item_path in self.models_folder.rglob("*"):
            try:
                model_info: Dict[str, Any] = {
                    "model_name": None, "path": str(item_path), "type": None, "size_gb": None,
                }
                if item_path.is_dir() and (
                    (item_path / "config.json").exists() or list(item_path.glob("*.safetensors"))
                ):
                    is_sub_dir = any(
                        Path(m["path"]) == item_path.parent
                        for m in local_models
                        if m["type"] == "HuggingFace Directory"
                    )
                    if is_sub_dir:
                        continue
                    model_info.update({
                        "model_name": item_path.name,
                        "type": "HuggingFace Directory",
                        "size_gb": round(
                            sum(f.stat().st_size for f in item_path.glob("**/*") if f.is_file())
                            / (1024 ** 3),
                            2,
                        ),
                    })
                    local_models.append(model_info)
                elif item_path.is_file() and item_path.suffix.lower() == ".gguf":
                    model_info.update({
                        "model_name": str(item_path.relative_to(self.models_folder)),
                        "type": "GGUF File",
                        "size_gb": round(item_path.stat().st_size / (1024 ** 3), 2),
                    })
                    local_models.append(model_info)
            except Exception as e:
                ASCIIColors.warning(f"Error processing {item_path}: {e}")
        return local_models

    # ── Resource management contract ────────────────────────────────────────
    def is_local(self) -> bool:
        return True

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        return self.llm_engine is not None

    def has_active_resources(self) -> bool:
        return self.llm_engine is not None

    def get_loaded_models(self) -> List[str]:
        return [self.current_model_name_or_id] if self.current_model_name_or_id else []

    def unload_model(self, model_name: Optional[str] = None) -> bool:
        self.close()
        return True

    def install_model(self, model_name: str, **kwargs) -> dict:
        try:
            path = resolve_hf_model_path(model_name, self.models_folder)
            return {"status": True, "message": f"Model installed at {path}", "path": str(path)}
        except Exception as e:
            trace_exception(e)
            return {"status": False, "message": str(e)}

    def pull_model(self, model_name: str, **kwargs) -> dict:
        return self.install_model(model_name, **kwargs)

    def ps(self) -> List[Dict[str, Any]]:
        if not self.llm_engine:
            return []
        return [{
            "model_name": self.current_model_name_or_id,
            "is_loaded": True,
            "device": "cuda",
            "loader": "TensorRT-LLM",
        }]

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        if self.llm_engine and self.current_resolved_model_path and self.current_engine_params:
            ASCIIColors.info(
                f"Binding {id(self)} close(). Releasing engine for: "
                f"{self.current_resolved_model_path.name}"
            )
            engine_manager.release_engine(self.current_resolved_model_path, self.current_engine_params)
        self.llm_engine = None
        self.tokenizer = None
        self.current_model_name_or_id = None
        self.current_resolved_model_path = None
        self.current_engine_params = None
        self.model_name = ""


__all__ = ["TensorRTLLMBinding", "BindingName"]