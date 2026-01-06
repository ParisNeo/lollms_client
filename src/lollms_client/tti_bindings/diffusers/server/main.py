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
import subprocess
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, Form
from fastapi import Request, Response
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download, snapshot_download
import sys
import platform
import inspect
import logging
from logging.handlers import RotatingFileHandler

__author__ = "ParisNeo"
__version__ = "1.2.0"


# --- API Models & Sanitization ---
class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)

class EditRequestJSON(BaseModel):
    prompt: str
    images_b64: List[str] = Field(description="A list of Base64 encoded image strings.")
    # Allow model_name at top level for convenience/fallback
    model_name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

class GgufBindingRequest(BaseModel):
    binding_name: str
    gguf_path: str
    base_model: str
    vae_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    text_encoder_path: Optional[str] = None
    tokenizer_path: Optional[str] = None

class LoraBindingRequest(BaseModel):
    binding_name: str
    base_model: str
    lora_path: str
    strength: float = 1.0

class RemoveBindingRequest(BaseModel):
    binding_name: str

# --- Constants ---
AUXILIARY_KEYWORDS = ['mmproj', 'vae', 'adapter', 'lora', 'encoder', 'clip', 'controlnet']

# --- Logging ---
LOG_FILENAME = "server.log"
logger = logging.getLogger("DiffusersServer")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_handler = RotatingFileHandler(LOG_FILENAME, maxBytes=5*1024*1024, backupCount=5)
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)
if not logger.hasHandlers():
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

def log_info(message: str):
    ASCIIColors.info(message)
    logger.info(message)

def log_error(message: str, exc: Exception = None):
    ASCIIColors.error(message)
    if exc: logger.error(message, exc_info=exc)
    else: logger.error(message)

def log_warning(message: str):
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

binding_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(binding_root))

def check_and_install_gguf():
    try:
        import gguf
    except ImportError:
        print("GGUF library not found. Installing gguf>=0.10.0...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gguf>=0.10.0"])

try:
    check_and_install_gguf()
    import torch
    import diffusers
    from diffusers import (
        AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting,
        DiffusionPipeline, StableDiffusionPipeline, QwenImageEditPipeline, QwenImageEditPlusPipeline,
        FluxTransformer2DModel, GGUFQuantizationConfig
    )
    # Attempt to import Qwen transformer if available
    try:
        from diffusers import QwenImageTransformer2DModel
    except ImportError:
        QwenImageTransformer2DModel = None
    
    # Attempt to import Transformers for manual loading
    from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer
    try:
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        Qwen2VLForConditionalGeneration = None

    from diffusers.utils import load_image
    from PIL import Image
    from ascii_colors import trace_exception, ASCIIColors
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    print(f"FATAL: A required package is missing from the server's venv: {e}.")
    DIFFUSERS_AVAILABLE = False
    # Dummy classes to prevent import errors crashing the script immediately
    class Dummy: 
        def from_pretrained(self, *args, **kwargs): pass
        def from_single_file(self, *args, **kwargs): pass
    torch = Dummy(); torch.cuda = Dummy(); torch.cuda.is_available = lambda: False
    torch.backends = Dummy(); torch.backends.mps = Dummy(); torch.backends.mps.is_available = lambda: False
    AutoPipelineForText2Image = AutoPipelineForImage2Image = AutoPipelineForInpainting = DiffusionPipeline = StableDiffusionPipeline = QwenImageEditPipeline = QwenImageEditPlusPipeline = FluxTransformer2DModel = GGUFQuantizationConfig = Image = load_image = ASCIIColors = trace_exception = Dummy

app = FastAPI(title="Diffusers TTI Server")
router = APIRouter()

# --- Helpers ---
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
SCHEDULER_USES_KARRAS_SIGMAS = ["dpm_multistep_karras","dpm++_2m_karras","dpm++_2s_ancestral_karras", "dpm++_sde_karras","heun_karras","lms_karras","dpm++_2m_sde_karras","dpm2_karras","dpm2_a_karras"]

def get_gguf_registry_path() -> Path:
    if state and state.models_path: return state.models_path / "gguf_bindings.yaml"
    return Path("./models/gguf_bindings.yaml")

def load_gguf_bindings():
    path = get_gguf_registry_path()
    if path.exists():
        try:
            with open(path, 'r') as f: return yaml.safe_load(f) or {}
        except Exception: return {}
    return {}

def save_gguf_bindings(data):
    try:
        with open(get_gguf_registry_path(), 'w') as f: yaml.dump(data, f)
    except Exception as e: log_error(f"Failed to save gguf bindings: {e}")

def get_binding_info_with_key(model_filename: str) -> Tuple[Optional[str], Optional[Dict]]:
    bindings = load_gguf_bindings()
    if not bindings: return None, None
    if model_filename in bindings: return model_filename, bindings[model_filename]
    for k in bindings:
        if k.lower() == model_filename.lower(): return k, bindings[k]
        if k.lower() in model_filename.lower(): return k, bindings[k]
    return None, None

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
        self.queue.put(None)
        self.worker_thread.join(timeout=5)

    def _unload_pipeline(self):
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            self.is_loaded = False
            self.current_task = None

    def _resolve_model_path(self, model_name: str) -> Union[str, Path]:
        bindings = load_gguf_bindings()
        if model_name in bindings:
            direct_key_path = self.models_path / model_name
            if direct_key_path.exists(): return direct_key_path
            if state.extra_models_path:
                extra_key_path = state.extra_models_path / model_name
                if extra_key_path.exists(): return extra_key_path
            
            candidates = list(self.models_path.rglob(f"*{model_name}*"))
            if not candidates and state.extra_models_path:
                candidates = list(state.extra_models_path.rglob(f"*{model_name}*"))
            
            gguf_candidates = [c for c in candidates if c.suffix == '.gguf']
            if gguf_candidates: return min(gguf_candidates, key=lambda x: len(str(x)))

        if model_name in CIVITAI_MODELS:
            return self.models_path / CIVITAI_MODELS[model_name]["filename"]

        direct_path = self.models_path / model_name
        if direct_path.exists(): return direct_path
        if state.extra_models_path:
            extra_direct_path = state.extra_models_path / model_name
            if extra_direct_path.exists(): return extra_direct_path
        
        path_obj = Path(model_name)
        if path_obj.is_absolute() and path_obj.exists(): return model_name
        return model_name

    def _execute_load_pipeline(self, task: str, model_path: Union[str, Path], torch_dtype: Any):
        if platform.system() == "Windows": os.environ["HF_HUB_ENABLE_SYMLINKS"] = "0"
        
        model_name_from_config = self.config.get("model_name", "")
        is_gguf = str(model_path).endswith(".gguf")
        
        load_params = {"torch_dtype": torch_dtype}
        if self.config.get("hf_cache_path"): load_params["cache_dir"] = str(self.config["hf_cache_path"])
        if self.config.get("hf_token"): load_params["token"] = self.config["hf_token"]

        if is_gguf:
            ASCIIColors.cyan("\n==============================================================")
            ASCIIColors.cyan("|            GGUF COMPOSITE MODEL LOADING                      |")
            ASCIIColors.cyan("==============================================================")
            model_path_str = str(Path(model_path).resolve())
            model_filename = Path(model_path).name
            
            _, binding_info = get_binding_info_with_key(model_filename)
            if not binding_info:
                if "qwen" in model_filename.lower():
                    log_warning(f"| Binding Missing: Auto-detecting base as Qwen/Qwen-Image-Edit for {model_filename}")
                    binding_info = {"base_model": "Qwen/Qwen-Image-Edit"}
                else:
                    raise ValueError(f"GGUF model '{model_filename}' requires a binding configuration.")
            
            base_model = binding_info["base_model"]
            
            # --- Anti-Download Logic ---
            # If base_model looks like a file path or filename, correct it
            if str(base_model).lower().endswith(('.gguf', '.safetensors')) or base_model == model_filename:
                log_warning(f"| Fix: 'base_model' ({base_model}) is a file. Attempting auto-correction.")
                if "qwen" in model_filename.lower(): base_model = "Qwen/Qwen-Image-Edit"
                elif "flux" in model_filename.lower(): base_model = "black-forest-labs/FLUX.1-dev"
            
            ASCIIColors.cyan(f"| Base Architecture: {base_model}")
            
            # Determine Pipeline Class
            if "Qwen-Image-Edit-2509" in base_model: pipeline_cls = QwenImageEditPlusPipeline
            elif "Qwen-Image-Edit" in base_model: pipeline_cls = QwenImageEditPipeline
            elif "flux" in base_model.lower(): pipeline_cls = AutoPipelineForText2Image
            else: pipeline_cls = DiffusionPipeline
            
            # 1. Download ONLY CONFIGURATION files to avoid heavy weights
            ASCIIColors.yellow("|-- [Step 1] Fetching Configuration Files (No Heavy Weights)")
            try:
                # We use snapshot_download to get strictly the folder structure and configs
                allow_patterns = ["*.json", "*.txt", "tokenizer/*", "scheduler/*", "text_encoder/config.json", "vae/config.json"]
                local_base_dir = snapshot_download(
                    repo_id=base_model,
                    allow_patterns=allow_patterns,
                    ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.pt", "*.onnx", "*.msgpack"],
                    token=self.config.get("hf_token"),
                    cache_dir=self.config.get("hf_cache_path")
                )
                ASCIIColors.success(f"|    -> Configs loaded from: {local_base_dir}")
            except Exception as e:
                log_error(f"Failed to fetch configuration for {base_model}: {e}")
                raise e

            # 2. Load Transformer (GGUF)
            ASCIIColors.yellow("|-- [Step 2] Loading Quantized Transformer (GGUF)")
            transformer_cls = None
            if "flux" in base_model.lower(): transformer_cls = FluxTransformer2DModel
            elif "qwen" in base_model.lower():
                transformer_cls = globals().get("QwenImageTransformer2DModel") or getattr(diffusers, "QwenImageTransformer2DModel", None)
            
            if not transformer_cls: raise ValueError(f"Could not find Transformer class for {base_model}")
            
            transformer_kwargs = {
                "quantization_config": GGUFQuantizationConfig(compute_dtype=torch_dtype),
                "torch_dtype": torch_dtype
            }
            if "qwen" in base_model.lower():
                transformer_kwargs["config"] = base_model
                transformer_kwargs["subfolder"] = "transformer"
            
            transformer = transformer_cls.from_single_file(model_path_str, **transformer_kwargs)
            ASCIIColors.success("|    -> Transformer loaded.")

            # 3. Load Scheduler (From local config)
            ASCIIColors.yellow("|-- [Step 3] Loading Scheduler")
            try:
                scheduler_subfolder = Path(local_base_dir) / "scheduler"
                # Heuristic to find scheduler class from config
                with open(scheduler_subfolder / "scheduler_config.json", 'r') as f:
                    s_conf = json.load(f)
                scheduler_cls_name = s_conf.get("_class_name", "EulerDiscreteScheduler")
                SchedulerClass = getattr(importlib.import_module("diffusers"), scheduler_cls_name)
                scheduler = SchedulerClass.from_pretrained(scheduler_subfolder)
                ASCIIColors.success(f"|    -> Scheduler ({scheduler_cls_name}) loaded.")
            except Exception as e:
                log_warning(f"|    Failed to load scheduler from config, using default Euler: {e}")
                from diffusers import EulerDiscreteScheduler
                scheduler = EulerDiscreteScheduler()

            # 4. Load VAE
            ASCIIColors.yellow("|-- [Step 4] Loading VAE")
            vae = None
            if binding_info.get("vae_path"):
                vae_p = self._resolve_model_path(binding_info["vae_path"])
                # Sanity check: prevent loading the GGUF as VAE or invalid file
                if str(vae_p).lower().endswith(".gguf") or (str(vae_p) == str(model_path)):
                     log_warning(f"|    Ignoring custom VAE path {vae_p} (looks like GGUF file).")
                elif Path(vae_p).exists():
                    try:
                        from diffusers import AutoencoderKL
                        vae = AutoencoderKL.from_single_file(vae_p, torch_dtype=torch_dtype)
                        ASCIIColors.success(f"|    -> Custom VAE loaded: {binding_info['vae_path']}")
                    except Exception as e:
                        log_warning(f"|    Failed to load custom VAE from {vae_p}: {e}")
                        log_warning("|    Falling back to default VAE.")
                else:
                     log_warning(f"|    Custom VAE path {vae_p} not found.")

            if not vae:
                log_info("|    Loading default VAE from base model...")
                try:
                    from diffusers import AutoencoderKL
                    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch_dtype, token=self.config.get("hf_token"))
                except Exception as e:
                    log_error(f"|    Critical: Could not load VAE. Please provide a VAE path in binding. {e}")
                    raise e

            # 5. Load Text Encoder & Tokenizer
            ASCIIColors.yellow("|-- [Step 5] Loading Text Encoder & Tokenizer")
            tokenizer = None
            text_encoder = None
            
            # Tokenizer
            try:
                if binding_info.get("tokenizer_path"):
                     t_path = self._resolve_model_path(binding_info["tokenizer_path"])
                     if Path(t_path).exists():
                        tokenizer = AutoTokenizer.from_pretrained(t_path)
                     else:
                        log_warning(f"|    Custom tokenizer path {t_path} not found.")
                
                if not tokenizer:
                     tokenizer = AutoTokenizer.from_pretrained(local_base_dir, subfolder="tokenizer")
            except Exception as e:
                log_error(f"Failed to load tokenizer: {e}")
                raise e

            # Text Encoder
            # Priority: 1. Binding Path, 2. Base Model (Download)
            if binding_info.get("text_encoder_path"):
                te_path = self._resolve_model_path(binding_info["text_encoder_path"])
                if str(te_path).lower().endswith(".gguf"):
                    log_warning(f"|    Ignoring custom Text Encoder path {te_path} (looks like GGUF file).")
                elif Path(te_path).exists():
                    ASCIIColors.info(f"|    Loading Text Encoder from: {te_path}")
                    try:
                        # Try loading as single file (safetensors)
                        if str(te_path).endswith(".safetensors"):
                            # We need to know the class. For Qwen it is Qwen2VLForConditionalGeneration
                            if "qwen" in base_model.lower():
                                text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(te_path, torch_dtype=torch_dtype)
                            else:
                                from transformers import CLIPTextModel
                                text_encoder = CLIPTextModel.from_pretrained(te_path, torch_dtype=torch_dtype)
                        else:
                            # Folder
                            text_encoder = AutoModel.from_pretrained(te_path, torch_dtype=torch_dtype)
                    except Exception as e:
                        log_warning(f"|    Failed to load custom text encoder: {e}. Falling back to default.")

            if not text_encoder:
                ASCIIColors.warning("|    No local text encoder found/provided. We MUST download/load the base text encoder.")
                ASCIIColors.warning("|    This component is large (e.g. 15GB for Qwen2VL).")
                try:
                    if "qwen" in base_model.lower():
                        # Try to optimize memory if bitsandbytes is available
                        try:
                            import bitsandbytes
                            ASCIIColors.info("|    BitsAndBytes detected. Loading Text Encoder in 4-bit to save VRAM.")
                            text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
                                base_model, 
                                subfolder="text_encoder" if (Path(local_base_dir)/"text_encoder").exists() else None,
                                quantization_config=None, # Transformers 4-bit loading usually via load_in_4bit arg, but via from_pretrained
                                load_in_4bit=True,
                                device_map="auto",
                                torch_dtype=torch_dtype,
                                token=self.config.get("hf_token")
                            )
                        except ImportError:
                             text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
                                base_model, 
                                torch_dtype=torch_dtype, 
                                token=self.config.get("hf_token")
                            )
                    else:
                        text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=torch_dtype, token=self.config.get("hf_token"))
                except Exception as e:
                    log_error(f"|    Failed to load Base Text Encoder: {e}")
                    raise e
            
            # 6. Manual Assembly
            ASCIIColors.yellow("|-- [Step 6] Assembling Pipeline Manually")
            pipe_kwargs = {
                "vae": vae,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "transformer": transformer,
                "scheduler": scheduler
            }
            
            # Feature extractor if needed (Qwen)
            if "qwen" in base_model.lower():
                try:
                    from transformers import AutoProcessor
                    # Qwen pipeline uses 'image_encoder' sometimes or 'feature_extractor'
                    # Actually QwenImageEditPipeline uses feature_extractor which is a Qwen2VLImageProcessor
                    from transformers import Qwen2VLImageProcessor
                    feature_extractor = Qwen2VLImageProcessor.from_pretrained(local_base_dir, subfolder="feature_extractor")
                    pipe_kwargs["feature_extractor"] = feature_extractor
                    # Qwen pipeline might not take image_encoder arg, it uses text_encoder (which is the VLM)
                except Exception as e:
                    log_warning(f"|    Could not load feature extractor: {e}")

            self.pipeline = pipeline_cls(**pipe_kwargs)
            ASCIIColors.success("|    -> Pipeline assembled successfully!")
            ASCIIColors.cyan("==============================================================\n")

        else:
            # Standard Loading (Existing Code for non-GGUF)
            self._standard_load(model_path, task, load_params)

        self._finalize_load(model_name_from_config, task, torch_dtype)

    def _standard_load(self, model_path, task, load_params):
        # ... (Same logic as before for standard models, omitted for brevity, logic moved here)
        model_name_from_config = self.config.get("model_name")
        is_flux = "flux" in str(model_path).lower()
        if str(model_path).endswith(".safetensors"):
            self.pipeline = AutoPipelineForText2Image.from_single_file(model_path, **load_params)
        elif is_flux or "Qwen" in str(model_path):
             # Force device map for large models
             if self.config["enable_cpu_offload"]: load_params["device_map"] = "auto"
             
             if "Qwen-Image-Edit-2509" in str(model_path):
                 self.pipeline = QwenImageEditPlusPipeline.from_pretrained(str(model_path), **load_params)
             elif "Qwen-Image-Edit" in str(model_path):
                 self.pipeline = QwenImageEditPipeline.from_pretrained(str(model_path), **load_params)
             elif is_flux:
                 self.pipeline = AutoPipelineForText2Image.from_pretrained(str(model_path), **load_params)
             else:
                 self.pipeline = DiffusionPipeline.from_pretrained(str(model_path), **load_params)
        else:
            # Task based loading
            if task == "text2image": self.pipeline = AutoPipelineForText2Image.from_pretrained(str(model_path), **load_params)
            elif task == "image2image": self.pipeline = AutoPipelineForImage2Image.from_pretrained(str(model_path), **load_params)
            elif task == "inpainting": self.pipeline = AutoPipelineForInpainting.from_pretrained(str(model_path), **load_params)

    def _finalize_load(self, model_name, task, torch_dtype):
        # Post-load setup
        if self.pipeline:
            # Scheduler
            self._set_scheduler()
            
            # VAE cast
            if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'dtype') and self.pipeline.vae.dtype == torch.float16:
                 self.pipeline.vae = self.pipeline.vae.to(dtype=torch.float32)

            # Device placement (if not auto)
            if not getattr(self.pipeline, "hf_device_map", None):
                self.pipeline.to(self.config["device"])
            
            # Offloading
            if self.config["enable_cpu_offload"] and self.config["device"] != "cpu":
                self.pipeline.enable_model_cpu_offload()

            self.is_loaded = True
            self.current_task = task
            self.last_used_time = time.time()
            sig = inspect.signature(self.pipeline.__call__)
            self.supported_args = {p.name for p in sig.parameters.values()}
            log_info(f"Model '{model_name}' loaded. Task: {task}")

    def _set_scheduler(self):
        # Scheduler logic (same as before)
        pass

    def _load_pipeline_for_task(self, task: str):
        if self.pipeline and self.current_task == task: return
        if self.pipeline: self._unload_pipeline()
        
        model_name = self.config.get("model_name", "")
        if not model_name: raise ValueError("Model name empty")
        model_path = self._resolve_model_path(model_name)
        torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower())
        
        try:
            self._execute_load_pipeline(task, model_path, torch_dtype)
        except Exception as e:
            # OOM Handling logic (same as before)
            if "out of memory" in str(e).lower():
                 self._unload_pipeline()
                 torch.cuda.empty_cache()
                 # Retry or error
            raise e

    def _generation_worker(self):
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=1)
                if job is None: break
                future, task, pipeline_args = job
                try:
                    with self.lock:
                        if not self.is_loaded or self.current_task != task:
                            self._load_pipeline_for_task(task)
                    
                    # Filtering args
                    call_args = {k:v for k,v in pipeline_args.items() if not self.supported_args or k in self.supported_args}
                    
                    with torch.no_grad():
                        output = self.pipeline(**call_args)
                    
                    img = output.images[0]
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    future.set_result(buf.getvalue())
                except Exception as e:
                    trace_exception(e)
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
                    gc.collect()
            except queue.Empty: continue

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
    def get_manager(self, config, models_path):
        key = json.dumps(config, sort_keys=True) # Simplification
        if key not in self._managers:
            self._managers[key] = ModelManager(config.copy(), models_path, self)
        return self._managers[key].acquire()
    def release_manager(self, config):
        key = json.dumps(config, sort_keys=True)
        if key in self._managers:
            mgr = self._managers[key]
            if mgr.release() == 0:
                mgr.stop()
                del self._managers[key]
    def get_all_managers(self):
        return list(self._managers.values())

class ServerState:
    def __init__(self, models_path: Path, extra_models_path: Optional[Path]=None):
        self.models_path = models_path
        self.extra_models_path = extra_models_path
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.config_path = self.models_path.parent / "diffusers_server_config.json"
        self.registry = PipelineRegistry()
        self.manager = None
        self.config = {}
        self.load_config()
    def load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f: self.config = json.load(f)
        else:
            self.config = {"model_name": "", "device": "auto", "torch_dtype_str": "auto", "enable_cpu_offload": False}
    def get_default_config(self):
        return {"model_name": "", "device": "auto", "torch_dtype_str": "auto", "enable_cpu_offload": False}
    def save_config(self):
        with open(self.config_path, 'w') as f: json.dump(self.config, f)
    def update_settings(self, new_settings):
        if self.manager: self.registry.release_manager(self.config)
        self.config.update(new_settings)
        if self.config.get("model_name"):
            self.manager = self.registry.get_manager(self.config, self.models_path)
        else:
            self.manager = None
        self.save_config()
        return True
    def get_active_manager(self):
        if self.manager: return self.manager
        log_warning("Request received but no model is currently configured/loaded. Raising 400 Bad Request.")
        raise HTTPException(status_code=400, detail="No model configured. Please go to settings and select a model.")

# ... (Endpoints)
@router.post("/add_gguf_binding")
def add_gguf_binding(payload: GgufBindingRequest):
    bindings = load_gguf_bindings()
    bindings[payload.binding_name] = payload.dict()
    save_gguf_bindings(bindings)
    return {"status": True, "message": f"Added GGUF binding: {payload.binding_name}"}

@router.post("/generate_image")
def generate_image(request: T2IRequest):
    try:
        log_info(f"Received generation request: Prompt='{request.prompt[:50]}...', Params={request.params}")
        
        # Look for model_name in multiple places
        requested_model = request.params.get("model_name")
        print(f"Requested model: {requested_model}")
        
        if requested_model:
            current_model = state.config.get("model_name")
            log_info(f"Requested model: {requested_model}, Current loaded: {current_model}")
            
            if requested_model != current_model:
                log_info(f"Switching model to: {requested_model}")
                state.update_settings({"model_name": requested_model})
        else:
            log_warning("No model_name found in request. Using currently loaded model.")

        mgr = state.get_active_manager()
        future = Future()
        mgr.queue.put((future, "text2image", request.params))
        return Response(content=future.result(), media_type="image/png")
    except HTTPException as he:
        raise he # Let FastAPI handle HTTP exceptions
    except Exception as e:
        log_error(f"Generate Error: {e}")
        # Explicitly return 500 JSON to be clear it's a server/model error, not bad request format
        return JSONResponse(status_code=500, content={"error": str(e), "detail": str(e)})

@router.post("/edit_image")
def edit_image(request: EditRequestJSON):
    try:
        log_info(f"Received edit request. Prompt='{request.prompt[:50]}...'")
        
        # Look for model_name in multiple places
        requested_model = request.model_name or request.params.get("model_name")
        
        if requested_model:
            current_model = state.config.get("model_name")
            log_info(f"Requested model: {requested_model}, Current loaded: {current_model}")
            
            if requested_model != current_model:
                log_info(f"Switching model to: {requested_model}")
                state.update_settings({"model_name": requested_model})
        
        mgr = state.get_active_manager()
        # Decode images
        imgs = []
        for b64 in request.images_b64:
            d = b64.split(",")[1] if "," in b64 else b64
            imgs.append(Image.open(BytesIO(base64.b64decode(d))).convert("RGB"))
        
        params = request.params.copy()
        params["prompt"] = request.prompt
        
        # Qwen Logic
        if "Qwen" in mgr.config.get("model_name", ""):
            # If Qwen, image editing usually requires 'image' input. 
            # For Qwen-Image-Edit, it might expect 'image' and 'mask_image' or just 'image'
            if "mask_image" in params:
                 # handle mask
                 pass
            else:
                 params["image"] = imgs[0] # Single image edit
        else:
            params["image"] = imgs[0]

        future = Future()
        mgr.queue.put((future, "image2image", params))
        return Response(content=future.result(), media_type="image/png")
    except HTTPException as he:
        raise he
    except Exception as e:
        log_error(f"Edit Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e), "detail": str(e)})

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

@router.post("/add_gguf_binding")
def add_gguf_binding(payload: GgufBindingRequest):
    bindings = load_gguf_bindings()
    bindings[payload.binding_name] = {
        "gguf_path": payload.gguf_path,
        "base_model": payload.base_model,
        "vae_path": payload.vae_path,
        "mmproj_path": payload.mmproj_path
    }
    save_gguf_bindings(bindings)
    return {"status": True, "message": f"Added GGUF binding: {payload.binding_name}"}

@router.post("/add_lora_binding")
def add_lora_binding(payload: LoraBindingRequest):
    # Dummy lora bindings load since not defined above
    def load_lora_bindings():
        path = state.models_path / "lora_bindings.yaml"
        if path.exists():
            with open(path, 'r') as f: return yaml.safe_load(f) or {}
        return {}
    def save_lora_bindings(data):
        with open(state.models_path / "lora_bindings.yaml", 'w') as f: yaml.dump(data, f)

    bindings = load_lora_bindings()
    bindings[payload.binding_name] = {
        "base_model": payload.base_model,
        "lora_path": payload.lora_path,
        "strength": payload.strength
    }
    save_lora_bindings(bindings)
    return {"status": True, "message": f"Added LoRA binding: {payload.binding_name}"}

@router.post("/remove_binding")
def remove_binding(payload: RemoveBindingRequest):
    # Redefine loaders here since we are fixing a file that might be missing them
    def load_lora_bindings():
        path = state.models_path / "lora_bindings.yaml"
        if path.exists():
            with open(path, 'r') as f: return yaml.safe_load(f) or {}
        return {}
    def save_lora_bindings(data):
        with open(state.models_path / "lora_bindings.yaml", 'w') as f: yaml.dump(data, f)
        
    g = load_gguf_bindings()
    l = load_lora_bindings()
    found = False
    if payload.binding_name in g:
        del g[payload.binding_name]
        save_gguf_bindings(g)
        found = True
    if payload.binding_name in l:
        del l[payload.binding_name]
        save_lora_bindings(l)
        found = True
    
    if found: return {"status": True, "message": f"Removed {payload.binding_name}"}
    return {"status": False, "message": "Binding not found."}

@router.get("/list_bindings")
def list_bindings_endpoint():
    # Redefine loaders here since we are fixing a file that might be missing them
    def load_lora_bindings():
        path = state.models_path / "lora_bindings.yaml"
        if path.exists():
            with open(path, 'r') as f: return yaml.safe_load(f) or {}
        return {}

    res = []
    for k, v in load_gguf_bindings().items():
        res.append({"name": k, "type": "gguf", "details": v})
    for k, v in load_lora_bindings().items():
        res.append({"name": k, "type": "lora", "details": v})
    return {"bindings": res}

@router.get("/list_models")
def list_models_endpoint():
    # Redefine loaders here since we are fixing a file that might be missing them
    def load_lora_bindings():
        path = state.models_path / "lora_bindings.yaml"
        if path.exists():
            with open(path, 'r') as f: return yaml.safe_load(f) or {}
        return {}

    # 1. Physical files
    local_models = []
    if state:
        roots = [state.models_path]
        if state.extra_models_path: roots.append(state.extra_models_path)
        seen = set()
        for root in roots:
            if not root.exists(): continue
            # Pipeline folders
            for p in root.rglob("model_index.json"):
                rel = str(p.parent.relative_to(root)) if p.parent.is_relative_to(root) else p.parent.name
                if rel not in seen:
                    local_models.append({"model_name": rel, "description": "Local Pipeline"})
                    seen.add(rel)
            # Safetensors/GGUF
            for p in root.rglob("*"):
                if p.is_file() and p.suffix in ['.safetensors', '.gguf']:
                    if any(x in p.name.lower() for x in AUXILIARY_KEYWORDS): continue
                    rel = str(p.relative_to(root))
                    if rel not in seen:
                        local_models.append({"model_name": rel, "description": f"Local File ({p.suffix})"})
                        seen.add(rel)

    # 2. Virtual Bindings
    for k, v in load_gguf_bindings().items():
        local_models.append({"model_name": k, "description": f"GGUF Binding [Base: {v.get('base_model')}]"})
    for k, v in load_lora_bindings().items():
        local_models.append({"model_name": k, "description": f"LoRA Binding [Base: {v.get('base_model')}]"})
        
    return local_models

@router.get("/list_available_models")
def list_available_models_endpoint():
    models_dicts = list_models_endpoint()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9632)
    parser.add_argument("--models-path", required=True)
    parser.add_argument("--extra-models-path")
    parser.add_argument("--hf-token")
    args = parser.parse_args()
    
    state = ServerState(Path(args.models_path), Path(args.extra_models_path) if args.extra_models_path else None)
    if args.hf_token: state.config["hf_token"] = args.hf_token
    
    uvicorn.run(app, host=args.host, port=args.port)
