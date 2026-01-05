import os
import importlib
from io import BytesIO
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import base64
import threading
import queue
from concurrent.futures import Future
import time
import hashlib
import requests
from tqdm import tqdm
import json
import shutil
import yaml
import numpy as np
import gc
import argparse
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, Form
from fastapi import Request, Response
from fastapi.responses import Response
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
import sys
import platform
import inspect
import logging
from logging.handlers import RotatingFileHandler

__author__ = "ParisNeo"
__version__ = "1.1.2"

# --- Constants ---
# Files containing these substrings (case-insensitive) will be hidden from the model list
AUXILIARY_KEYWORDS = ['mmproj', 'vae', 'adapter', 'lora', 'encoder', 'clip', 'controlnet']

# --- Logging Configuration ---
LOG_FILENAME = "server.log"
logger = logging.getLogger("DiffusersServer")
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = RotatingFileHandler(LOG_FILENAME, maxBytes=5*1024*1024, backupCount=5)
c_handler.setLevel(logging.WARNING) # Only warn/error on standard stderr (ASCIIColors handles info)
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

def log_info(message: str):
    """Dual logging: ASCIIColors for console, Logger for file."""
    ASCIIColors.info(message)
    logger.info(message)

def log_error(message: str, exc: Exception = None):
    """Dual logging: ASCIIColors for console, Logger for file."""
    ASCIIColors.error(message)
    if exc:
        logger.error(message, exc_info=exc)
    else:
        logger.error(message)

def log_warning(message: str):
    """Dual logging: ASCIIColors for console, Logger for file."""
    ASCIIColors.warning(message)
    logger.warning(message)

# --- Pydantic Models ---
class PullModelRequest(BaseModel):
    hf_id: Optional[str] = Field(default=None, description="Hugging Face repo id or URL")
    safetensors_url: Optional[str] = Field(default=None, description="Direct URL to a file")
    local_name: Optional[str] = Field(default=None, description="Optional name/folder under models/")
    filename: Optional[str] = Field(default=None, description="Specific filename to download from repo")

class BindComponentsRequest(BaseModel):
    model_name: str
    base_model: str
    vae_path: Optional[str] = None
    other_component_path: Optional[str] = None

# Add binding root to sys.path
binding_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(binding_root))

# --- Dependency Check and Imports ---
try:
    import torch
    from diffusers import (
        AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting,
        DiffusionPipeline, StableDiffusionPipeline, QwenImageEditPipeline, QwenImageEditPlusPipeline,
        FluxTransformer2DModel, GGUFQuantizationConfig
    )
    from diffusers.utils import load_image
    from PIL import Image
    from ascii_colors import trace_exception, ASCIIColors
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    print(f"FATAL: A required package is missing from the server's venv: {e}.")
    DIFFUSERS_AVAILABLE = False
    class Dummy: 
        def from_pretrained(self, *args, **kwargs): pass
        def from_single_file(self, *args, **kwargs): pass
    torch = Dummy()
    torch.cuda = Dummy()
    torch.cuda.is_available = lambda: False
    torch.backends = Dummy()
    torch.backends.mps = Dummy()
    torch.backends.mps.is_available = lambda: False
    AutoPipelineForText2Image = AutoPipelineForImage2Image = AutoPipelineForInpainting = DiffusionPipeline = StableDiffusionPipeline = QwenImageEditPipeline = QwenImageEditPlusPipeline = FluxTransformer2DModel = GGUFQuantizationConfig = Image = load_image = ASCIIColors = trace_exception = Dummy

# --- Server Setup ---
app = FastAPI(title="Diffusers TTI Server")
router = APIRouter()

# --- Core Logic ---
CIVITAI_MODELS = {
    "DreamShaper-8": {
        "display_name": "DreamShaper 8", "url": "https://civitai.com/api/download/models/128713",
        "filename": "dreamshaper_8.safetensors", "description": "Versatile SD1.5 style model.", "owned_by": "civitai"
    },
    "Juggernaut-xl": {
        "display_name": "Juggernaut XL", "url": "https://civitai.com/api/download/models/133005",
        "filename": "juggernautXL_version6Rundiffusion.safetensors", "description": "Artistic SDXL.", "owned_by": "civitai"
    },
}

TORCH_DTYPE_MAP_STR_TO_OBJ = {
    "float16": getattr(torch, 'float16', 'float16'), "bfloat16": getattr(torch, 'bfloat16', 'bfloat16'),
    "float32": getattr(torch, 'float32', 'float32'), "auto": "auto"
}

SCHEDULER_MAPPING = {
    "default": None, "ddim": "DDIMScheduler", "ddpm": "DDPMScheduler", "deis_multistep": "DEISMultistepScheduler",
    "dpm_multistep": "DPMSolverMultistepScheduler", "dpm_multistep_karras": "DPMSolverMultistepScheduler", "dpm_single": "DPMSolverSinglestepScheduler",
    "dpm_adaptive": "DPMSolverPlusPlusScheduler", "dpm++_2m": "DPMSolverMultistepScheduler", "dpm++_2m_karras": "DPMSolverMultistepScheduler",
    "dpm++_2s_ancestral": "DPMSolverAncestralDiscreteScheduler", "dpm++_2s_ancestral_karras": "DPMSolverAncestralDiscreteScheduler", "dpm++_sde": "DPMSolverSDEScheduler",
    "dpm++_sde_karras": "DPMSolverSDEScheduler", "euler_ancestral_discrete": "EulerAncestralDiscreteScheduler", "euler_discrete": "EulerDiscreteScheduler",
    "heun_discrete": "HeunDiscreteScheduler", "heun_karras": "HeunDiscreteScheduler", "lms_discrete": "LMSDiscreteScheduler",
    "lms_karras": "LMSDiscreteScheduler", "pndm": "PNDMScheduler", "unipc_multistep": "UniPCMultistepScheduler",
    "dpm++_2m_sde": "DPMSolverMultistepScheduler", "dpm++_2m_sde_karras": "DPMSolverMultistepScheduler", "dpm2": "KDPM2DiscreteScheduler",
    "dpm2_karras": "KDPM2DiscreteScheduler", "dpm2_a": "KDPM2AncestralDiscreteScheduler", "dpm2_a_karras": "KDPM2AncestralDiscreteScheduler",
    "euler": "EulerDiscreteScheduler", "euler_a": "EulerAncestralDiscreteScheduler", "heun": "HeunDiscreteScheduler", "lms": "LMSDiscreteScheduler"
}

SCHEDULER_USES_KARRAS_SIGMAS = [
    "dpm_multistep_karras","dpm++_2m_karras","dpm++_2s_ancestral_karras", "dpm++_sde_karras","heun_karras","lms_karras",
    "dpm++_2m_sde_karras","dpm2_karras","dpm2_a_karras"
]

def get_gguf_registry_path() -> Path:
    if state and state.models_path:
        return state.models_path / "gguf_bindings.yaml"
    return Path("./models/gguf_bindings.yaml")

def load_gguf_bindings():
    path = get_gguf_registry_path()
    if path.exists():
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load gguf bindings: {e}")
            return {}
    return {}

def save_gguf_bindings(data):
    path = get_gguf_registry_path()
    try:
        with open(path, 'w') as f:
            yaml.dump(data, f)
    except Exception as e:
        log_error(f"Failed to save gguf bindings: {e}")

def get_binding_info_with_key(model_filename: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Returns (binding_key, binding_data) or (None, None).
    """
    bindings = load_gguf_bindings()
    if not bindings:
        return None, None
    
    # 1. Exact match
    if model_filename in bindings:
        return model_filename, bindings[model_filename]
    
    # 2. Case insensitive match
    keys_lower = {k.lower(): k for k in bindings.keys()}
    if model_filename.lower() in keys_lower:
        real_key = keys_lower[model_filename.lower()]
        return real_key, bindings[real_key]
        
    # 3. Stem match
    for key in bindings:
        if key.lower() in model_filename.lower():
            return key, bindings[key]
            
    return None, None

def get_binding_info(model_filename: str):
    _, info = get_binding_info_with_key(model_filename)
    return info

class ModelManager:
    def __init__(self, config: Dict[str, Any], models_path: Path, registry: 'PipelineRegistry'):
        self.config = config
        self.models_path = models_path
        self.registry = registry
        self.pipeline: Optional[DiffusionPipeline] = None
        self.current_task: Optional[str] = None
        self.ref_count = 0
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.is_loaded = False
        self.last_used_time = time.time()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.worker_thread.start()
        self._stop_monitor_event = threading.Event()
        self._unload_monitor_thread = None
        self._start_unload_monitor()
        self.supported_args: Optional[set] = None

    def acquire(self):
        with self.lock:
            self.ref_count += 1
            return self

    def release(self):
        with self.lock:
            self.ref_count -= 1
            return self.ref_count

    def stop(self):
        self._stop_event.set()
        if self._unload_monitor_thread:
            self._stop_monitor_event.set()
            self._unload_monitor_thread.join(timeout=2)
        self.queue.put(None)
        self.worker_thread.join(timeout=5)

    def _start_unload_monitor(self):
        unload_after = self.config.get("unload_inactive_model_after", 0)
        if unload_after > 0 and self._unload_monitor_thread is None:
            self._stop_monitor_event.clear()
            self._unload_monitor_thread = threading.Thread(target=self._unload_monitor, daemon=True)
            self._unload_monitor_thread.start()

    def _unload_monitor(self):
        unload_after = self.config.get("unload_inactive_model_after", 0)
        if unload_after <= 0:
            return
        log_info(f"Starting inactivity monitor for '{self.config['model_name']}' (timeout: {unload_after}s).")
        while not self._stop_monitor_event.wait(timeout=5.0):
            with self.lock:
                if not self.is_loaded:
                    continue
                if time.time() - self.last_used_time > unload_after:
                    log_info(f"Model '{self.config['model_name']}' has been inactive. Unloading.")
                    self._unload_pipeline()

    def _resolve_model_path(self, model_name: str) -> Union[str, Path]:
        # 1. Check if model_name is a binding key (Reverse Lookup)
        bindings = load_gguf_bindings()
        if model_name in bindings:
            # Check if key itself is a file
            direct_key_path = self.models_path / model_name
            if direct_key_path.exists():
                return direct_key_path
            
            if state.extra_models_path:
                extra_key_path = state.extra_models_path / model_name
                if extra_key_path.exists():
                    return extra_key_path

            # Heuristic: Scan for files containing the key
            candidates = list(self.models_path.rglob(f"*{model_name}*"))
            if not candidates and state.extra_models_path:
                candidates = list(state.extra_models_path.rglob(f"*{model_name}*"))
            
            # Filter for .gguf
            candidates = [c for c in candidates if c.suffix == '.gguf']
            
            if candidates:
                best_match = min(candidates, key=lambda x: len(str(x)))
                log_info(f"Resolved binding key '{model_name}' to file: {best_match.name}")
                return best_match

        # 2. CivitAI shortcuts
        if model_name in CIVITAI_MODELS:
            filename = CIVITAI_MODELS[model_name]["filename"]
            local_path = self.models_path / filename
            if not local_path.exists():
                self._download_civitai_model(model_name)
            return local_path

        # 3. Primary models path (Relative)
        direct_path = self.models_path / model_name
        if direct_path.exists():
            return direct_path

        # 4. Extra models path (Relative)
        if state.extra_models_path:
            extra_direct_path = state.extra_models_path / model_name
            if extra_direct_path.exists():
                return extra_direct_path

        # 5. Absolute path
        path_obj = Path(model_name)
        if path_obj.is_absolute() and path_obj.exists():
            return model_name

        # 6. Recursive search
        if len(path_obj.parts) == 1:
            if state.extra_models_path and state.extra_models_path.exists():
                found_paths = list(state.extra_models_path.rglob(model_name))
                if found_paths:
                    return found_paths[0]

            found_paths = list(self.models_path.rglob(model_name))
            if found_paths:
                return found_paths[0]

        # 7. Fallback (HF Repo)
        return model_name

    def _download_civitai_model(self, model_key: str):
        model_info = CIVITAI_MODELS[model_key]
        url = model_info["url"]
        filename = model_info["filename"]
        dest_path = self.models_path / filename
        temp_path = dest_path.with_suffix(".temp")
        log_info(f"Downloading '{filename}' from Civitai... to {dest_path}")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(temp_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {filename}") as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            shutil.move(temp_path, dest_path)
            log_info(f"Model '{filename}' downloaded successfully.")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise Exception(f"Failed to download model {filename}: {e}")

    def _set_scheduler(self):
        if not self.pipeline:
            return
        if "Qwen" in self.config.get("model_name", "") or "FLUX" in self.config.get("model_name", ""):
            log_info("Special model detected, skipping custom scheduler setup.")
            return
        scheduler_name_key = self.config["scheduler_name"].lower()
        if scheduler_name_key == "default":
            return
        scheduler_class_name = SCHEDULER_MAPPING.get(scheduler_name_key)
        if scheduler_class_name:
            try:
                SchedulerClass = getattr(importlib.import_module("diffusers.schedulers"), scheduler_class_name)
                scheduler_config = self.pipeline.scheduler.config
                scheduler_config["use_karras_sigmas"] = scheduler_name_key in SCHEDULER_USES_KARRAS_SIGMAS
                self.pipeline.scheduler = SchedulerClass.from_config(scheduler_config)
                log_info(f"Switched scheduler to {scheduler_class_name}")
            except Exception as e:
                log_warning(f"Could not switch scheduler to {scheduler_name_key}: {e}. Using current default.")

    def _execute_load_pipeline(self, task: str, model_path: Union[str, Path], torch_dtype: Any):
        if platform.system() == "Windows":
            os.environ["HF_HUB_ENABLE_SYMLINKS"] = "0"
        
        model_name_from_config = self.config.get("model_name", "")
        use_device_map = False

        try:
            load_params = {}
            if self.config.get("hf_cache_path"):
                load_params["cache_dir"] = str(self.config["hf_cache_path"])
            load_params["torch_dtype"] = torch_dtype

            # Logic to detect model type based on resolved path or config name
            is_qwen_model = "Qwen" in model_name_from_config or "Qwen" in str(model_path)
            is_flux_model = "FLUX" in model_name_from_config or "flux" in str(model_path).lower()
            is_gguf = str(model_path).endswith(".gguf")

            if is_gguf:
                log_info(f"Detected GGUF model: {model_path}")
                model_filename = Path(model_path).name
                binding_info = get_binding_info(model_filename)
                
                # If no binding, try intelligent defaults for well-known models
                if not binding_info:
                    if "qwen" in model_filename.lower():
                        log_warning(f"No explicit binding found for {model_filename}, auto-detecting base as Qwen/Qwen-Image-Edit")
                        binding_info = {"base_model": "Qwen/Qwen-Image-Edit"}
                    else:
                        raise ValueError(f"GGUF model '{model_filename}' requires a binding configuration (base_model) via bind_model_components.")

                base_model = binding_info["base_model"]
                
                # Base model validation & Auto-correction
                # Check if base_model looks like a local file (e.g. .gguf, .safetensors) which is invalid for base_model
                base_path_obj = Path(base_model)
                is_invalid_file = False
                if base_path_obj.exists() and base_path_obj.is_file():
                    is_invalid_file = True
                elif str(base_model).lower().endswith(('.gguf', '.safetensors', '.ckpt', '.bin')):
                    is_invalid_file = True
                
                if is_invalid_file:
                     if "qwen" in model_filename.lower():
                         log_warning(f"Invalid binding configuration! Base model points to a file ({base_model}). Auto-correcting to 'Qwen/Qwen-Image-Edit'.")
                         log_warning("NOTE: Diffusers needs a 'base model' (architecture configuration) to run GGUF weights. This base model downloads CONFIG FILES from HF, but the heavy weights come from your local GGUF file.")
                         base_model = "Qwen/Qwen-Image-Edit"
                     else:
                         log_error(f"Invalid binding configuration! Base model points to a file ({base_model}). Please update the binding with a valid HF Repo ID or Pipeline directory.")
                         # We proceed, but it will likely fail below if it tries to load a pipeline from a weights file

                # Optimization: Prevent downloading large weights if they are going to be replaced or ignored
                if "ignore_patterns" not in load_params:
                    load_params["ignore_patterns"] = []
                
                # Always ignore the main heavy weights from the base model
                load_params["ignore_patterns"].extend([
                    "*.safetensors", "*.bin", "*.ckpt", "*.pth"  # Ignore root level weights
                ])
                # Specifically ignore transformer/unet weights as we replace them with GGUF
                load_params["ignore_patterns"].extend([
                    "transformer/*.safetensors", "transformer/*.bin", 
                    "unet/*.safetensors", "unet/*.bin"
                ])

                # Handle local overrides to prevent downloading components we have
                if binding_info.get("vae_path"):
                    log_info(f"VAE provided locally ({binding_info['vae_path']}). Skipping VAE download.")
                    load_params["ignore_patterns"].extend(["vae/*", "vae_encoder/*", "vae_decoder/*"])
                    load_params["vae"] = None # Don't load VAE in from_pretrained

                if binding_info.get("other_component_path"):
                    # Heuristic: if mmproj is provided, it usually replaces visual encoders or projections
                    if "mmproj" in binding_info["other_component_path"].lower():
                        # For Qwen, this might replace vision_encoder or image_proj
                        # We can't easily skip downloading vision_encoder unless we know for sure the pipeline supports loading it from file later
                        pass

                log_info(f"Loading base pipeline configuration from {base_model}...")
                log_info(f"Note: This step downloads configuration files (tokenizer, scheduler, etc). Weights will be loaded from {model_filename}.")
                
                load_params.update({"use_safetensors": True, "token": self.config["hf_token"]})
                if self.config["hf_variant"]: load_params["variant"] = self.config["hf_variant"]
                
                if "Qwen-Image-Edit-2509" in base_model:
                    pipeline_cls = QwenImageEditPlusPipeline
                elif "Qwen-Image-Edit" in base_model:
                    pipeline_cls = QwenImageEditPipeline
                elif "flux" in base_model.lower():
                    pipeline_cls = AutoPipelineForText2Image
                else:
                    pipeline_cls = DiffusionPipeline

                self.pipeline = pipeline_cls.from_pretrained(base_model, **load_params)
                
                # --- Inject Local Components ---
                
                # 1. VAE
                if binding_info.get("vae_path"):
                    vae_p = self._resolve_model_path(binding_info["vae_path"])
                    if Path(vae_p).exists():
                        from diffusers import AutoencoderKL
                        self.pipeline.vae = AutoencoderKL.from_single_file(vae_p, torch_dtype=torch_dtype)
                        log_info(f"Loaded local VAE from {vae_p}")
                    else:
                        log_warning(f"Local VAE path {vae_p} not found. Pipeline might be missing VAE.")

                # 2. Main GGUF Transformer/UNet
                log_info(f"Loading GGUF quantized weights from {model_path}...")
                quantization_config = GGUFQuantizationConfig(compute_dtype=torch_dtype)
                
                if "flux" in base_model.lower():
                    transformer = FluxTransformer2DModel.from_single_file(
                        model_path, quantization_config=quantization_config, torch_dtype=torch_dtype
                    )
                    self.pipeline.transformer = transformer
                elif "qwen" in base_model.lower():
                    try:
                        TransformerCls = self.pipeline.transformer.__class__
                        if hasattr(TransformerCls, 'from_single_file'):
                            transformer = TransformerCls.from_single_file(
                                model_path, quantization_config=quantization_config, torch_dtype=torch_dtype
                            )
                            self.pipeline.transformer = transformer
                            log_info("Successfully swapped GGUF transformer.")
                        else:
                            log_warning(f"Transformer class {TransformerCls.__name__} does not support from_single_file.")
                    except Exception as e:
                        log_error(f"Failed to swap GGUF transformer: {e}")
                
                # 3. Other Components (mmproj, etc.) - Placeholder for future expansion
                # Currently diffusers doesn't have a standard 'load_mmproj'
                if binding_info.get("other_component_path"):
                    log_info(f"Note: 'other_component_path' ({binding_info['other_component_path']}) is defined but manual loading for this component type is not yet implemented for this pipeline.")

            elif is_qwen_model or is_flux_model:
                log_info(f"Special model '{model_name_from_config}' detected. Using dedicated pipeline loader.")
                load_params.update({
                    "use_safetensors": self.config["use_safetensors"],
                    "token": self.config["hf_token"],
                    "local_files_only": self.config["local_files_only"]
                })
                if self.config["hf_variant"]: load_params["variant"] = self.config["hf_variant"]
                if not self.config["safety_checker_on"]: load_params["safety_checker"] = None
                
                should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                if should_offload:
                    log_info(f"Offload enabled. Forcing device_map='auto' for {model_name_from_config}.")
                    use_device_map = True
                    load_params["device_map"] = "auto"
                
                if is_flux_model:
                    self.pipeline = AutoPipelineForText2Image.from_pretrained(model_name_from_config, **load_params)
                elif "Qwen-Image-Edit-2509" in model_name_from_config:
                    self.pipeline = QwenImageEditPlusPipeline.from_pretrained(model_name_from_config, **load_params)
                elif "Qwen-Image-Edit" in model_name_from_config:
                    self.pipeline = QwenImageEditPipeline.from_pretrained(model_name_from_config, **load_params)
                elif "Qwen/Qwen-Image" in model_name_from_config:
                    self.pipeline = DiffusionPipeline.from_pretrained(model_name_from_config, **load_params)
            
            else:
                is_safetensors_file = str(model_path).endswith(".safetensors")
                if is_safetensors_file:
                    log_info(f"Loading standard model from local .safetensors file: {model_path}")
                    try:
                        self.pipeline = AutoPipelineForText2Image.from_single_file(model_path, **load_params)
                    except Exception as e:
                        log_warning(f"Failed to load with AutoPipeline, falling back to StableDiffusionPipeline: {e}")
                        self.pipeline = StableDiffusionPipeline.from_single_file(model_path, **load_params)
                else:
                    log_info(f"Loading standard model from Hub: {model_path}")
                    load_params.update({
                        "use_safetensors": self.config["use_safetensors"],
                        "token": self.config["hf_token"],
                        "local_files_only": self.config["local_files_only"]
                    })
                    if self.config["hf_variant"]: load_params["variant"] = self.config["hf_variant"]
                    if not self.config["safety_checker_on"]: load_params["safety_checker"] = None
                    
                    is_large_model = "stable-diffusion-3" in str(model_path)
                    should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                    if is_large_model and should_offload:
                        log_info(f"Large model '{model_path}' detected with offload enabled. Using device_map='auto'.")
                        use_device_map = True
                        load_params["device_map"] = "auto"

                    if task == "text2image":
                        self.pipeline = AutoPipelineForText2Image.from_pretrained(model_path, **load_params)
                    elif task == "image2image":
                        self.pipeline = AutoPipelineForImage2Image.from_pretrained(model_path, **load_params)
                    elif task == "inpainting":
                        self.pipeline = AutoPipelineForInpainting.from_pretrained(model_path, **load_params)
        
        except Exception as e:
            error_str = str(e).lower()
            if "401" in error_str or "gated" in error_str or "authorization" in error_str:
                msg = (f"AUTHENTICATION FAILED for model '{model_name_from_config}'. Please ensure you accepted the model license and provided a valid HF token.")
                raise RuntimeError(msg) 
            raise e

        # VAE fix for float16 artifacts
        if self.pipeline and hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'dtype'):
             if self.pipeline.vae.dtype == torch.float16:
                log_info("Upcasting VAE to float32 to prevent artifacts.")
                self.pipeline.vae = self.pipeline.vae.to(dtype=torch.float32)

        self._set_scheduler()

        if not use_device_map:
            self.pipeline.to(self.config["device"])
            if self.config["enable_xformers"]:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    log_warning(f"Could not enable xFormers: {e}.")
            
            if self.config["enable_cpu_offload"] and self.config["device"] != "cpu":
                self.pipeline.enable_model_cpu_offload()
            elif self.config["enable_sequential_cpu_offload"] and self.config["device"] != "cpu":
                self.pipeline.enable_sequential_cpu_offload()
        else:
             log_info("Device map handled device placement.")

        if self.pipeline:
            sig = inspect.signature(self.pipeline.__call__)
            self.supported_args = {p.name for p in sig.parameters.values()}
            log_info(f"Pipeline supported arguments detected: {self.supported_args}")

        self.is_loaded = True
        self.current_task = task
        self.last_used_time = time.time()
        log_info(f"Model '{model_name_from_config}' loaded successfully using '{'device_map' if use_device_map else 'standard'}' mode for task '{task}'.")

    def _load_pipeline_for_task(self, task: str):
        if self.pipeline and self.current_task == task:
            return
        if self.pipeline:
            self._unload_pipeline()
        
        model_name = self.config.get("model_name", "")
        if not model_name:
            raise ValueError("Model name cannot be empty for loading.")
        
        log_info(f"Loading Diffusers model: {model_name} for task: {task}")
        model_path = self._resolve_model_path(model_name)
        torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower())
        
        try:
            self._execute_load_pipeline(task, model_path, torch_dtype)
            return
        except Exception as e:
            is_oom = "out of memory" in str(e).lower()
            if not is_oom or not hasattr(self, 'registry'):
                raise e
        
        log_warning(f"Failed to load '{model_name}' due to OOM. Attempting to unload other models to free VRAM.")
        
        candidates_to_unload = [m for m in self.registry.get_all_managers() if m is not self and m.is_loaded]
        candidates_to_unload.sort(key=lambda m: m.last_used_time)

        if not candidates_to_unload:
            log_error("OOM error, but no other models are available to unload.")
            raise Exception("OOM error, but no other models are available to unload.")

        for victim in candidates_to_unload:
            log_info(f"Unloading '{victim.config['model_name']}' to free VRAM.")
            victim._unload_pipeline()
            
            try:
                log_info(f"Retrying to load '{model_name}'...")
                self._execute_load_pipeline(task, model_path, torch_dtype)
                log_info(f"Successfully loaded '{model_name}' after freeing VRAM.")
                return
            except Exception as retry_e:
                is_oom_retry = "out of memory" in str(retry_e).lower()
                if not is_oom_retry:
                    raise retry_e 
        
        log_error(f"Could not load '{model_name}' even after unloading all other models.")
        raise e

    def _unload_pipeline(self):
        if self.pipeline:
            model_name = self.config.get('model_name', 'Unknown')
            del self.pipeline
            self.pipeline = None
            self.supported_args = None
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.is_loaded = False
            self.current_task = None
            log_info(f"Model '{model_name}' unloaded and VRAM cleared.")

    def _generation_worker(self):
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=1)
                if job is None:
                    break
                future, task, pipeline_args = job
                output = None
                try:
                    with self.lock:
                        self.last_used_time = time.time()
                        if not self.is_loaded or self.current_task != task:
                            self._load_pipeline_for_task(task)
                    
                    if self.supported_args:
                        filtered_args = {k: v for k, v in pipeline_args.items() if k in self.supported_args}
                    else:
                        log_warning("Supported argument set not found. Using unfiltered arguments.")
                        filtered_args = pipeline_args

                    with torch.no_grad():
                        output = self.pipeline(**filtered_args)
                    
                    pil = output.images[0]
                    buf = BytesIO()
                    pil.save(buf, format="PNG")
                    future.set_result(buf.getvalue())
                except Exception as e:
                    trace_exception(e)
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
                    if output is not None:
                        del output
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except queue.Empty:
                continue

class PipelineRegistry:
    _instance = None
    _lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._managers = {}
                cls._instance._registry_lock = threading.Lock()
        return cls._instance
    @staticmethod
    def _get_critical_keys():
        return [
            "model_name","device","torch_dtype_str","use_safetensors",
            "safety_checker_on","hf_variant","enable_cpu_offload",
            "enable_sequential_cpu_offload","enable_xformers",
            "local_files_only","hf_cache_path","unload_inactive_model_after"
        ]
    def _get_config_key(self, config: Dict[str, Any]) -> str:
        key_data = tuple(sorted((k, config.get(k)) for k in self._get_critical_keys()))
        return hashlib.sha256(str(key_data).encode('utf-8')).hexdigest()
    def get_manager(self, config: Dict[str, Any], models_path: Path) -> ModelManager:
        key = self._get_config_key(config)
        with self._registry_lock:
            if key not in self._managers:
                self._managers[key] = ModelManager(config.copy(), models_path, self)
            return self._managers[key].acquire()
    def release_manager(self, config: Dict[str, Any]):
        key = self._get_config_key(config)
        with self._registry_lock:
            if key in self._managers:
                manager = self._managers[key]
                ref_count = manager.release()
                if ref_count == 0:
                    log_info(f"Reference count for model '{config.get('model_name')}' is zero. Cleaning up manager.")
                    manager.stop()
                    with manager.lock:
                        manager._unload_pipeline()
                    del self._managers[key]
    def get_active_managers(self) -> List[ModelManager]:
        with self._registry_lock:
            return [m for m in self._managers.values() if m.is_loaded]
    def get_all_managers(self) -> List[ModelManager]:
        with self._registry_lock:
            return list(self._managers.values())

class ServerState:
    def __init__(self, models_path: Path, extra_models_path: Optional[Path] = None):
        self.models_path = models_path
        self.extra_models_path = extra_models_path
        self.models_path.mkdir(parents=True, exist_ok=True)
        if self.extra_models_path:
            self.extra_models_path.mkdir(parents=True, exist_ok=True)
        self.config_path = self.models_path.parent / "diffusers_server_config.json"
        self.registry = PipelineRegistry()
        self.manager: Optional[ModelManager] = None
        self.config = {}
        self.load_config()
        self._resolve_device_and_dtype()
        if self.config.get("model_name"):
            try:
                log_info(f"Acquiring initial model manager for '{self.config['model_name']}' on startup.")
                self.manager = self.registry.get_manager(self.config, self.models_path)
            except Exception as e:
                log_error(f"Failed to acquire model manager on startup: {e}")
                self.manager = None

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": "", "device": "auto", "torch_dtype_str": "auto", "use_safetensors": True,
            "scheduler_name": "default", "safety_checker_on": True, "num_inference_steps": 25,
            "guidance_scale": 7.0, "width": 1024, "height": 1024, "seed": -1,
            "enable_cpu_offload": False, "enable_sequential_cpu_offload": False, "enable_xformers": False,
            "hf_variant": None, "hf_token": None, "hf_cache_path": None, "local_files_only": False,
            "unload_inactive_model_after": 0
        }

    def save_config(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            log_info(f"Server config saved to {self.config_path}")
        except Exception as e:
            log_error(f"Failed to save server config: {e}")

    def load_config(self):
        default_config = self.get_default_config()
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
                self.config = default_config
                log_info(f"Loaded server configuration from {self.config_path}")
            except (json.JSONDecodeError, IOError) as e:
                log_warning(f"Could not load config file, using defaults. Error: {e}")
                self.config = default_config
        else:
            self.config = default_config
        self.save_config()

    def _resolve_device_and_dtype(self):
        if self.config.get("device", "auto").lower() == "auto":
            self.config["device"] = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        if ("Qwen" in self.config.get("model_name", "") or "FLUX" in self.config.get("model_name", "")) and self.config["device"] == "cuda":
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                self.config["torch_dtype_str"] = "bfloat16"
                log_info("Special model detected. Forcing dtype to bfloat16 for stability.")
                return

        if self.config["torch_dtype_str"].lower() == "auto":
            self.config["torch_dtype_str"] = "float16" if self.config["device"] != "cpu" else "float32"

    def update_settings(self, new_settings: Dict[str, Any]):
        if 'model' in new_settings and 'model_name' not in new_settings:
            new_settings['model_name'] = new_settings.pop('model')

        if self.config.get("model_name") and not new_settings.get("model_name"):
            log_info("Incoming settings have no model_name. Preserving existing model.")
            new_settings["model_name"] = self.config["model_name"]

        if self.manager:
            self.registry.release_manager(self.manager.config)
            self.manager = None

        self.config.update(new_settings)
        log_info(f"Server config updated. Current model_name: {self.config.get('model_name')}")
        
        self._resolve_device_and_dtype()

        if self.config.get("model_name"):
            log_info("Acquiring model manager with updated configuration...")
            self.manager = self.registry.get_manager(self.config, self.models_path)
        else:
            log_warning("No model_name in config after update, manager not acquired.")
        
        self.save_config()
        return True

    def get_active_manager(self) -> ModelManager:
        if self.manager:
            return self.manager
        raise HTTPException(status_code=400, detail="No model is configured. Please set a model using /set_settings.")

state: Optional[ServerState] = None

# --- API Models & Sanitization ---
class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)

class EditRequestJSON(BaseModel):
    prompt: str
    images_b64: List[str] = Field(description="A list of Base64 encoded image strings.")
    params: Dict[str, Any] = Field(default_factory=dict)

def get_sanitized_request_for_logging(request_data: Any) -> Dict[str, Any]:
    """Sanitizes request data for logging, truncating large base64 strings."""
    import copy
    try:
        if hasattr(request_data, 'model_dump'):
            data = request_data.model_dump()
        elif isinstance(request_data, dict):
            data = copy.deepcopy(request_data)
        else:
            return {"error": "Unsupported data type"}

        if 'images_b64' in data and isinstance(data['images_b64'], list):
            count = len(data['images_b64'])
            data['images_b64'] = f"[<{count} base64 image(s) truncated>]"

        if 'params' in data and isinstance(data.get('params'), dict):
            if 'mask_image' in data['params'] and isinstance(data['params']['mask_image'], str):
                data['params']['mask_image'] = "[<base64 mask truncated>]"
        
        return data
    except Exception:
        return {"error": "Failed to sanitize request data."}

# --- API Endpoints ---
@router.post("/bind_model_components")
def bind_model_components_endpoint(payload: BindComponentsRequest):
    try:
        bindings = load_gguf_bindings()
        key = payload.model_name
        if os.sep in key or "/" in key:
            key = Path(key).name
            
        bindings[key] = {
            "base_model": payload.base_model,
            "vae_path": payload.vae_path,
            "other_component_path": payload.other_component_path
        }
        save_gguf_bindings(bindings)
        return {"status": True, "message": f"Bound {key} to base {payload.base_model}"}
    except Exception as e:
        log_error(f"Bind error: {e}", exc=e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_image")
def generate_image(request: T2IRequest):
    manager = None
    temp_config = None
    try:
        if "model_name" in request.params and request.params["model_name"]:
            temp_config = state.config.copy()
            temp_config["model_name"] = request.params.pop("model_name")
            manager = state.registry.get_manager(temp_config, state.models_path)
            log_info(f"Using per-request model: {temp_config['model_name']}")
        else:
            manager = state.get_active_manager()
            log_info(f"Using session-configured model: {manager.config.get('model_name')}")

        pipeline_args = manager.config.copy()
        pipeline_args.update(request.params)
        pipeline_args["prompt"] = request.prompt
        pipeline_args["negative_prompt"] = request.negative_prompt
        
        # Safe float/int conversions
        pipeline_args["width"] = int(pipeline_args.get("width", 1024))
        pipeline_args["height"] = int(pipeline_args.get("height", 1024))
        pipeline_args["num_inference_steps"] = int(pipeline_args.get("num_inference_steps", 25))
        pipeline_args["guidance_scale"] = float(pipeline_args.get("guidance_scale", 7.0))

        seed = int(pipeline_args.get("seed", -1))
        pipeline_args["generator"] = None
        if seed != -1:
            pipeline_args["generator"] = torch.Generator(device=manager.config["device"]).manual_seed(seed)
        
        model_name = manager.config.get("model_name", "")
        task = "text2image"
        
        if "Qwen-Image-Edit" in model_name:
            rng_seed = seed if seed != -1 else None
            rng = np.random.default_rng(seed=rng_seed)
            random_pixels = rng.integers(0, 256, size=(pipeline_args["height"], pipeline_args["width"], 3), dtype=np.uint8)
            pipeline_args["image"] = Image.fromarray(random_pixels, 'RGB')
            pipeline_args["strength"] = float(pipeline_args.get("strength", 1.0))
            task = "image2image" 
        
        log_args = {k: v for k, v in pipeline_args.items() if k not in ['generator', 'image']}
        log_info(f"Generating Image. Params: {json.dumps(log_args, default=str)}")
        
        future = Future()
        manager.queue.put((future, task, pipeline_args))
        result_bytes = future.result() 
        return Response(content=result_bytes, media_type="image/png")
    except Exception as e:
        log_error(f"Generation error: {e}", exc=e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_config and manager:
            state.registry.release_manager(temp_config)

@router.post("/edit_image")
def edit_image(request: EditRequestJSON):
    manager = None
    temp_config = None
    try:
        if "model_name" in request.params and request.params["model_name"]:
            temp_config = state.config.copy()
            temp_config["model_name"] = request.params.pop("model_name")
            manager = state.registry.get_manager(temp_config, state.models_path)
            log_info(f"Using per-request model: {temp_config['model_name']}")
        else:
            manager = state.get_active_manager()
            log_info(f"Using session-configured model: {manager.config.get('model_name')}")

        pipeline_args = manager.config.copy()
        pipeline_args.update(request.params)
        pipeline_args["prompt"] = request.prompt
        model_name = manager.config.get("model_name", "")

        pil_images = []
        for b64_string in request.images_b64:
            b64_data = b64_string.split(";base64,")[1] if ";base64," in b64_string else b64_string
            image_bytes = base64.b64decode(b64_data)
            pil_images.append(Image.open(BytesIO(image_bytes)).convert("RGB"))

        if not pil_images: raise HTTPException(status_code=400, detail="No valid images provided.")

        seed = int(pipeline_args.get("seed", -1))
        pipeline_args["generator"] = None
        if seed != -1: pipeline_args["generator"] = torch.Generator(device=manager.config["device"]).manual_seed(seed)
        
        if "mask_image" in pipeline_args and pipeline_args["mask_image"]:
            b64_mask = pipeline_args["mask_image"]
            b64_data = b64_mask.split(";base64,")[1] if ";base64," in b64_mask else b64_mask
            mask_bytes = base64.b64decode(b64_data)
            pipeline_args["mask_image"] = Image.open(BytesIO(mask_bytes)).convert("L")
        
        task = "inpainting" if "mask_image" in pipeline_args and pipeline_args["mask_image"] else "image2image"
        
        if "Qwen-Image-Edit-2509" in model_name:
            task = "image2image"
            pipeline_args.update({"true_cfg_scale": 4.0, "guidance_scale": 1.0, "num_inference_steps": 40, "negative_prompt": " "})
            edit_mode = pipeline_args.get("edit_mode", "fusion")
            if edit_mode == "fusion": pipeline_args["image"] = pil_images
        else:
            pipeline_args.update({"image": pil_images[0]})

        log_args = {k: v for k, v in pipeline_args.items() if k not in ['generator', 'image', 'mask_image']}
        log_info(f"Editing Image. Params: {json.dumps(log_args, default=str)}")
        
        future = Future()
        manager.queue.put((future, task, pipeline_args))
        return Response(content=future.result(), media_type="image/png")
    except Exception as e:
        sanitized_payload = get_sanitized_request_for_logging(request)
        log_error(f"Edit error. Payload: {json.dumps(sanitized_payload, indent=2)}", exc=e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_config and manager:
            state.registry.release_manager(temp_config)

@router.post("/pull_model")
def pull_model_endpoint(payload: PullModelRequest):
    if not payload.hf_id and not payload.safetensors_url:
        raise HTTPException(status_code=400, detail="Provide either 'hf_id' or 'safetensors_url'.")
    
    if payload.hf_id:
        model_id = payload.hf_id.strip()
        if payload.filename:
            filename = payload.filename
            dest_dir = state.models_path
            try:
                log_info(f"Downloading single file '{filename}' from '{model_id}'")
                hf_hub_download(
                    repo_id=model_id, filename=filename, local_dir=dest_dir,
                    local_dir_use_symlinks=False, token=state.config.get("hf_token"),
                    resume_download=True
                )
                log_info(f"File '{filename}' downloaded.")
                return {"status": "ok", "model_name": filename}
            except Exception as e:
                log_error(f"Download failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to download file from HF: {e}")

        folder_name = payload.local_name or model_id.replace("/", "__")
        dest_dir = state.models_path / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            log_info(f"Pulling HF model '{model_id}' into {dest_dir}")
            load_params: Dict[str, Any] = {}
            if state.config.get("hf_cache_path"): load_params["cache_dir"] = str(state.config["hf_cache_path"])
            if state.config.get("hf_token"): load_params["token"] = state.config["hf_token"]
            load_params["local_files_only"] = False

            pipe = DiffusionPipeline.from_pretrained(model_id, **load_params)
            pipe.save_pretrained(dest_dir)
            del pipe; gc.collect(); 
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            log_info(f"Model '{model_id}' pulled successfully.")
            return {"status": "ok", "model_name": folder_name}
        except Exception as e:
            log_error(f"Pull failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to pull HF model: {e}")

    if payload.safetensors_url:
        url = payload.safetensors_url.strip()
        default_name = url.split("/")[-1] or "model.safetensors"
        if not default_name.endswith(".safetensors") and not default_name.endswith(".gguf"):
            default_name += ".safetensors"
        filename = payload.local_name or default_name
        dest_path = state.models_path / filename
        temp_path = dest_path.with_suffix(".temp")

        log_info(f"Downloading file from {url} to {dest_path}")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(temp_path, "wb") as f, tqdm(total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {filename}") as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if not chunk: continue
                        f.write(chunk)
                        bar.update(len(chunk))
            shutil.move(temp_path, dest_path)
            log_info(f"File downloaded to {dest_path}")
            return {"status": "ok", "model_name": filename}
        except Exception as e:
            if temp_path.exists(): temp_path.unlink()
            log_error(f"Download failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download file: {e}")

@router.get("/list_local_models")
def list_local_models_endpoint():
    local_models = set()
    models_root = Path(args.models_path)
    extra_root = Path(args.extra_models_path) if args.extra_models_path else None

    def scan_root(root: Path):
        if not root or not root.exists(): return
        for model_index in root.rglob("model_index.json"): local_models.add(model_index.parent.name)
        for safepath in root.rglob("*.safetensors"):
            if (safepath.parent / "model_index.json").exists(): continue
            local_models.add(safepath.name)
        for gguf_path in root.rglob("*.gguf"):
            local_models.add(gguf_path.name)
            
    scan_root(models_root)
    scan_root(extra_root)
    return sorted(list(local_models))

def get_clean_model_name(path: Path, roots: List[Path]) -> str:
    try:
        for root in roots:
            if root and path.is_relative_to(root):
                return str(path.relative_to(root))
        return path.name
    except ValueError:
        return path.name

@app.get("/list_models")
def list_models() -> list[dict]:
    if not state: return []
    models_root = state.models_path
    extra_root = state.extra_models_path
    result = []
    seen_paths = set()

    def scan_root(root: Path):
        if not root or not root.exists(): return
        
        # 1. Diffusers folders
        for model_index in root.rglob("model_index.json"):
             folder = model_index.parent
             resolved_path = str(folder.resolve())
             if resolved_path in seen_paths: continue
             seen_paths.add(resolved_path)
             display_name = get_clean_model_name(folder, [models_root, extra_root])
             result.append({"model_name": display_name, "display_name": display_name, "description": "Local Diffusers pipeline"})

        # 2. Safetensors files
        for safepath in root.rglob("*.safetensors"):
            if (safepath.parent / "model_index.json").exists(): continue
            resolved_path = str(safepath.resolve())
            if resolved_path in seen_paths: continue
            
            # Filter auxiliary files
            fname_lower = safepath.name.lower()
            if any(x in fname_lower for x in AUXILIARY_KEYWORDS):
                continue

            seen_paths.add(resolved_path)
            display_name = get_clean_model_name(safepath, [models_root, extra_root])
            result.append({"model_name": display_name, "display_name": display_name, "description": "Local .safetensors checkpoint"})
            
        # 3. GGUF files
        for gguf_path in root.rglob("*.gguf"):
            resolved_path = str(gguf_path.resolve())
            if resolved_path in seen_paths: continue
            
            # Filter auxiliary files
            fname_lower = gguf_path.name.lower()
            if any(x in fname_lower for x in AUXILIARY_KEYWORDS):
                continue

            seen_paths.add(resolved_path)
            filename = gguf_path.name
            
            # Default model_name is the filename
            model_name = get_clean_model_name(gguf_path, [models_root, extra_root])
            display_name = model_name
            desc = "Local GGUF checkpoint (Unbound)"
            
            # Check for binding key
            key, binding_info = get_binding_info_with_key(filename)
            if binding_info:
                 base = binding_info.get("base_model", "?")
                 desc = f"Bound GGUF [Base: {base}]"
                 # IMPORTANT: Use the binding key as the model_name if available.
                 # This allows the client to select the user-friendly name from YAML.
                 # The server must then resolve this key back to the file in _resolve_model_path.
                 if key:
                     model_name = key
                     display_name = f"{key} ({filename})"
                 
            result.append({"model_name": model_name, "display_name": display_name, "description": desc})

    scan_root(models_root)
    scan_root(extra_root)
    return result

@router.get("/list_available_models")
def list_available_models_endpoint():
    models_dicts = list_models()
    discoverable = [m['model_name'] for m in models_dicts]
    return sorted(list(set(discoverable)))

@router.get("/get_settings")
def get_settings_endpoint():
    settings_list = []
    available_models = list_available_models_endpoint()
    schedulers = list(SCHEDULER_MAPPING.keys())
    config_to_display = state.config or state.get_default_config()
    for name, value in config_to_display.items():
        setting = {"name": name, "type": str(type(value).__name__), "value": value}
        if name == "model_name": setting["options"] = available_models
        if name == "scheduler_name": setting["options"] = schedulers
        settings_list.append(setting)
    return settings_list

@router.post("/set_settings")
def set_settings_endpoint(settings: Dict[str, Any]):
    try:
        success = state.update_settings(settings)
        return {"success": success}
    except Exception as e:
        log_error(f"Settings update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
def status_endpoint():
    return {"status": "running", "diffusers_available": DIFFUSERS_AVAILABLE, "model_loaded": state.manager.is_loaded if state.manager else False}
    
@router.post("/unload_model")
def unload_model_endpoint():
    if state.manager:
        state.manager._unload_pipeline()
        state.registry.release_manager(state.manager.config)
        state.manager = None
    return {"status": "unloaded"}

@router.post("/shutdown")
def shutdown_endpoint():
    def kill_server():
        time.sleep(0.5)
        log_warning("Shutting down server as requested.")
        os._exit(0)
    threading.Thread(target=kill_server, daemon=True).start()
    return {"status": "shutdown_initiated"}

@router.get("/ps")
def ps_endpoint():
    managers = state.registry.get_all_managers()
    return [{
            "model_name": m.config.get("model_name"), "is_loaded": m.is_loaded,
            "task": m.current_task, "device": m.config.get("device"), "ref_count": m.ref_count,
            "queue_size": m.queue.qsize(), "last_used": time.ctime(m.last_used_time)
        } for m in managers]

app.include_router(router)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusers TTI Server")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9630)
    parser.add_argument("--models-path", type=str, required=True)
    parser.add_argument("--extra-models-path", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    ASCIIColors.red("=" * 80)
    ASCIIColors.red(f"   Diffusers TTI Server v{__version__}")
    ASCIIColors.red("=" * 80)
    
    log_info(f"Starting server on http://{args.host}:{args.port}")
    MODELS_PATH = Path(args.models_path)
    EXTRA_MODELS_PATH = Path(args.extra_models_path) if args.extra_models_path else None
    state = ServerState(MODELS_PATH, EXTRA_MODELS_PATH)
    if args.hf_token: state.config["hf_token"] = args.hf_token
    
    if state and not DIFFUSERS_AVAILABLE:
        log_error("Diffusers or dependencies missing.")
    elif state:
        log_info(f"Detected device: {state.config['device']}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
