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
import pipmaster as pm
pm.ensure_packages("requests")

import requests
from huggingface_hub import snapshot_download
from tqdm import tqdm
import json
import shutil
import numpy as np
import gc
import argparse
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, Form
from fastapi import Request, Response
from fastapi.responses import Response
from pydantic import BaseModel, Field
import sys
import platform
import inspect

pm.ensure_packages(["ascii_colors>=0.11.10", "torch", "pillow", "diffusers"])

from ascii_colors import trace_exception, ASCIIColors

class SafeDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._popped_shapes = {}

    def pop(self, key, *args):
        if key in self:
            val = super().pop(key)
            if hasattr(val, "shape"):
                self._popped_shapes[key] = val.shape
            return val
        if args:
            return args[0]
            
        import torch
        ASCIIColors.warning(f"[SafeDict] GGUF conversion key '{key}' was missing. Supplying fallback tensor.")
        
        # Resolve shape dynamically from peer keys if possible
        base_key = key.rsplit(".", 1)[0]
        suffix = key.rsplit(".", 1)[-1]
        
        if suffix == "bias":
            weight_key = f"{base_key}.weight"
            if weight_key in self._popped_shapes:
                out_features = self._popped_shapes[weight_key][0]
                return torch.zeros(out_features, dtype=torch.bfloat16)
            if "time_in" in key or "vector_in" in key:
                return torch.zeros(3072, dtype=torch.bfloat16)
            return torch.zeros(256, dtype=torch.bfloat16)
            
        elif suffix == "weight":
            if "time_in" in key or "vector_in" in key:
                shape = (3072, 256)
            else:
                shape = (256, 256)
            self._popped_shapes[key] = shape
            return torch.zeros(shape, dtype=torch.bfloat16)
            
        return torch.zeros((256, 256), dtype=torch.bfloat16)

class PullModelRequest(BaseModel):
    hf_id: Optional[str] = Field(default=None, description="Hugging Face repo id or URL, e.g. 'stabilityai/sdxl-turbo'")
    safetensors_url: Optional[str] = Field(default=None, description="Direct URL to a .safetensors file")
    local_name: Optional[str] = Field(default=None, description="Optional name/folder under models/")
    allow_patterns: Optional[List[str]] = Field(default=None, description="Selective file list/patterns to download.")

# Add binding root to sys.path to ensure local modules can be imported if structured that way.
binding_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(binding_root))


# --- Dependency Check and Imports ---

try:
    import torch
    from diffusers import (
        AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting,
        DiffusionPipeline, StableDiffusionPipeline,
        QwenImageEditPipeline, QwenImageEditPlusPipeline,
    )
except Exception as ex:
    trace_exception(ex)    
    input("Press a button to continue")
    sys.exit(-1)
from PIL import Image
DIFFUSERS_AVAILABLE = True

# ── New-generation pipelines (require git main of diffusers) ─────────────────
# Each is imported individually with a graceful fallback so that older installs
# still work – the model just won't be available at runtime.
def _try_import(module, name, fallback=None):
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except (ImportError, AttributeError):
        return fallback
FluxKontextPipeline      = _try_import("diffusers", "FluxKontextPipeline")
Flux2KleinPipeline       = _try_import("diffusers", "Flux2KleinPipeline")
Flux2Pipeline            = _try_import("diffusers", "Flux2Pipeline")
FluxFillPipeline         = _try_import("diffusers", "FluxFillPipeline")
FluxPriorReduxPipeline   = _try_import("diffusers", "FluxPriorReduxPipeline")
FluxPipeline             = _try_import("diffusers", "FluxPipeline")

# ── Quantization support ─────────────────────────────────────────────────────
# PipelineQuantizationConfig is the unified entry-point for all quant backends.
# Individual config classes are imported separately for fine-grained mapping.
PipelineQuantizationConfig = _try_import("diffusers.quantizers", "PipelineQuantizationConfig")
DiffusersBnBConfig         = _try_import("diffusers", "BitsAndBytesConfig")      # transformer/vae
DiffusersQuantoConfig      = _try_import("diffusers.quantizers.quantization_config", "QuantoConfig")
DiffusersTorchAoConfig     = _try_import("diffusers", "TorchAoConfig")
DiffusersGGUFConfig        = _try_import("diffusers", "GGUFQuantizationConfig")
# Text encoders that come from Transformers need their own BnB config class.
try:
    from transformers import BitsAndBytesConfig as TransformersBnBConfig
except ImportError:
    TransformersBnBConfig = None

_QUANT_AVAILABILITY = {
    "PipelineQuantizationConfig": PipelineQuantizationConfig is not None,
    "BitsAndBytesConfig (diffusers)": DiffusersBnBConfig is not None,
    "QuantoConfig":                   DiffusersQuantoConfig  is not None,
    "TorchAoConfig":                  DiffusersTorchAoConfig is not None,
    "GGUFQuantizationConfig":         DiffusersGGUFConfig    is not None,
    "BitsAndBytesConfig (transformers)": TransformersBnBConfig is not None,
}
_MISSING_QUANT = [k for k, v in _QUANT_AVAILABILITY.items() if not v]
if _MISSING_QUANT:
    ASCIIColors.warning(f"Some quantization backends unavailable: {_MISSING_QUANT}")

_PIPELINE_AVAILABILITY = {
    "FluxKontextPipeline":    FluxKontextPipeline    is not None,
    "Flux2KleinPipeline":     Flux2KleinPipeline     is not None,
    "Flux2Pipeline":          Flux2Pipeline          is not None,
    "FluxFillPipeline":       FluxFillPipeline       is not None,
    "FluxPriorReduxPipeline": FluxPriorReduxPipeline is not None,
    "FluxPipeline":           FluxPipeline           is not None,
}
_MISSING_PIPELINES = [k for k, v in _PIPELINE_AVAILABILITY.items() if not v]
if _MISSING_PIPELINES:
    ASCIIColors.warning(
        f"The following next-gen pipeline classes are not available in the current "
        f"diffusers install (upgrade to git main to enable them): {_MISSING_PIPELINES}"
    )

# --- Server Setup ---
app = FastAPI(title="Diffusers TTI Server")
router = APIRouter()
MODELS_PATH = Path("./models")

# ---------------------------------------------------------------------------
# Model catalogues
# ---------------------------------------------------------------------------
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

HF_PUBLIC_MODELS = {
    "General Purpose & SDXL": [
        {"model_name": "stabilityai/stable-diffusion-xl-base-1.0", "display_name": "Stable Diffusion XL 1.0", "desc": "Official 1024x1024 text-to-image model from Stability AI."},
        {"model_name": "stabilityai/sdxl-turbo", "display_name": "SDXL Turbo", "desc": "A fast, real-time text-to-image model based on SDXL."},
        {"model_name": "kandinsky-community/kandinsky-3", "display_name": "Kandinsky 3", "desc": "A powerful multilingual model with strong prompt understanding and aesthetic quality."},
        {"model_name": "playgroundai/playground-v2.5-1024px-aesthetic", "display_name": "Playground v2.5", "desc": "A high-quality model focused on aesthetic outputs."},
    ],
    "Photorealistic": [
        {"model_name": "emilianJR/epiCRealism", "display_name": "epiCRealism", "desc": "A popular community model for generating photorealistic images."},
        {"model_name": "SG161222/Realistic_Vision_V5.1_noVAE", "display_name": "Realistic Vision 5.1", "desc": "One of the most popular realistic models, great for portraits and scenes."},
        {"model_name": "Photon-v1", "display_name": "Photon", "desc": "A model known for high-quality, realistic images with good lighting and detail."},
    ],
    "Anime & Illustration": [
        {"model_name": "hakurei/waifu-diffusion", "display_name": "Waifu Diffusion 1.4", "desc": "A widely-used model for generating high-quality anime-style images."},
        {"model_name": "gsdf/Counterfeit-V3.0", "display_name": "Counterfeit V3.0", "desc": "A strong model for illustrative and 2.5D anime styles."},
        {"model_name": "cagliostrolab/animagine-xl-3.0", "display_name": "Animagine XL 3.0", "desc": "A state-of-the-art anime model on the SDXL architecture."},
    ],
    "Artistic & Stylized": [
        {"model_name": "wavymulder/Analog-Diffusion", "display_name": "Analog Diffusion", "desc": "Creates images with a vintage, analog film aesthetic."},
        {"model_name": "dreamlike-art/dreamlike-photoreal-2.0", "display_name": "Dreamlike Photoreal 2.0", "desc": "Produces stunning, artistic, and photorealistic images."},
    ],
    "Image Editing Tools": [
        {"model_name": "stabilityai/stable-diffusion-xl-refiner-1.0", "display_name": "SDXL Refiner 1.0", "desc": "A dedicated refiner model to improve details in SDXL generations."},
        {"model_name": "timbrooks/instruct-pix2pix", "display_name": "Instruct-Pix2Pix", "desc": "The original instruction-based image editing model (SD 1.5)."},
        {"model_name": "kandinsky-community/kandinsky-2-2-instruct-pix2pix", "display_name": "Kandinsky 2.2 Instruct", "desc": "An instruction-based model with strong prompt adherence, based on Kandinsky 2.2."},
        {"model_name": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "display_name": "SDXL Inpainting", "desc": "A dedicated inpainting model based on SDXL 1.0 for filling in masked areas."},
        {"model_name": "Qwen/Qwen-Image-Edit", "display_name": "Qwen Image Edit", "desc": "An instruction-based model for various image editing tasks. (Review License)."},
        {"model_name": "Qwen/Qwen-Image-Edit-2509", "display_name": "Qwen Image Edit Plus", "desc": "Advanced multi-image editing and fusion. (Review License)."},
    ],
    "Legacy & Base Models": [
        {"model_name": "runwayml/stable-diffusion-v1-5", "display_name": "Stable Diffusion 1.5", "desc": "The classic and versatile SD1.5 base model."},
        {"model_name": "stabilityai/stable-diffusion-2-1", "display_name": "Stable Diffusion 2.1", "desc": "The 768x768 base model from the SD2.x series."},
    ],
    # ── NEW: Black Forest Labs next-generation models ──────────────────────────
    "FLUX.1 (Black Forest Labs)": [
        {
            "model_name": "black-forest-labs/FLUX.1-schnell",
            "display_name": "FLUX.1 Schnell",
            "desc": "Extremely fast distilled FLUX model. Best-in-class quality in just 4 steps. Apache 2.0. (gated – needs HF token)",
        },
        {
            "model_name": "black-forest-labs/FLUX.1-dev",
            "display_name": "FLUX.1 Dev",
            "desc": "Full guidance-distilled FLUX 12B model. Requires access request on HF.",
        },
        {
            "model_name": "black-forest-labs/FLUX.1-Kontext-dev",
            "display_name": "FLUX.1 Kontext [dev]",
            "desc": "12B image-editing model with in-context control: iterative edits, character consistency, style transfer. Non-commercial.",
        },
        {
            "model_name": "black-forest-labs/FLUX.1-Fill-dev",
            "display_name": "FLUX.1 Fill [dev]",
            "desc": "Dedicated inpainting / outpainting model based on FLUX.1. Non-commercial.",
        },
        {
            "model_name": "black-forest-labs/FLUX.1-Redux-dev",
            "display_name": "FLUX.1 Redux [dev]",
            "desc": "Image-variation model. Feed a reference image to generate stylistically similar outputs. Non-commercial.",
        },
    ],
    "FLUX.2 Klein (Black Forest Labs)": [
        {
            "model_name": "black-forest-labs/FLUX.2-klein-4B",
            "display_name": "FLUX.2 Klein 4B (distilled)",
            "desc": "4B distilled model. Fastest quality generation/editing in <1s. Apache 2.0 – fully commercial.",
        },
        {
            "model_name": "black-forest-labs/FLUX.2-klein-base-4B",
            "display_name": "FLUX.2 Klein 4B Base",
            "desc": "4B base (non-distilled) model. Slower but higher fidelity than the distilled variant. Apache 2.0.",
        },
        {
            "model_name": "black-forest-labs/FLUX.2-klein-9B",
            "display_name": "FLUX.2 Klein 9B (distilled)",
            "desc": "9B distilled model. Best quality/speed balance. Non-commercial license.",
        },
        {
            "model_name": "black-forest-labs/FLUX.2-klein-base-9B",
            "display_name": "FLUX.2 Klein 9B Base",
            "desc": "9B base (non-distilled) model. Highest fidelity in the Klein family. Non-commercial license.",
        },
    ],
    "FLUX.2 Dev (Black Forest Labs)": [
        {
            "model_name": "black-forest-labs/FLUX.2-dev",
            "display_name": "FLUX.2 Dev",
            "desc": "Next-generation 12B guidance model. Uses Mistral Small 3.1 text encoder. Non-commercial.",
        },
    ],
}

HF_GATED_MODELS = {
    "Next-Generation (Gated Access Required)": [
        {"model_name": "stabilityai/stable-diffusion-3-medium-diffusers", "display_name": "Stable Diffusion 3 Medium", "desc": "State-of-the-art model with advanced prompt understanding. Requires free registration."},
        {"model_name": "black-forest-labs/FLUX.1-schnell", "display_name": "FLUX.1 Schnell", "desc": "A powerful and extremely fast next-generation model. Requires access request."},
        {"model_name": "black-forest-labs/FLUX.1-dev", "display_name": "FLUX.1 Dev", "desc": "The larger developer version of the FLUX.1 model. Requires access request."},
        {"model_name": "black-forest-labs/FLUX.1-Kontext-dev", "display_name": "FLUX.1 Kontext [dev]", "desc": "Open-weight image editing model. Requires HF access request."},
        {"model_name": "black-forest-labs/FLUX.2-dev", "display_name": "FLUX.2 Dev", "desc": "Next-gen 12B model. Requires HF access request."},
        {"model_name": "black-forest-labs/FLUX.2-klein-9B", "display_name": "FLUX.2 Klein 9B", "desc": "Requires HF access request (non-commercial)."},
        {"model_name": "black-forest-labs/FLUX.2-klein-base-9B", "display_name": "FLUX.2 Klein 9B Base", "desc": "Requires HF access request (non-commercial)."},
    ]
}

# ---------------------------------------------------------------------------
# Helper: classify a model by its name
# ---------------------------------------------------------------------------

def _model_family(model_name: str) -> str:
    """Return a short family tag used for routing in load/generate logic."""
    mn = model_name.lower()
    if "flux.1-kontext" in mn or "flux1-kontext" in mn:
        return "flux_kontext"
    if "flux.2-klein" in mn or "flux2-klein" in mn:
        return "flux2_klein"
    if "flux.2-dev" in mn or "flux2-dev" in mn or "flux.2/dev" in mn:
        return "flux2_dev"
    if "flux.1-fill" in mn or "flux1-fill" in mn:
        return "flux_fill"
    if "flux.1-redux" in mn or "flux1-redux" in mn:
        return "flux_redux"
    if "flux" in mn:
        return "flux"
    if "qwen" in mn:
        return "qwen"
    return "standard"


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

# ---------------------------------------------------------------------------
# Families that skip the custom scheduler setup (they use FlowMatch or own
# scheduler and don't support all diffusers scheduler classes).
# ---------------------------------------------------------------------------
_SKIP_SCHEDULER_FAMILIES = {"flux", "flux_kontext", "flux2_klein", "flux2_dev", "flux_fill", "flux_redux", "qwen"}


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
        ASCIIColors.info(f"Starting inactivity monitor for '{self.config['model_name']}' (timeout: {unload_after}s).")
        while not self._stop_monitor_event.wait(timeout=5.0):
            with self.lock:
                if not self.is_loaded:
                    continue
                if time.time() - self.last_used_time > unload_after:
                    ASCIIColors.info(f"Model '{self.config['model_name']}' has been inactive. Unloading.")
                    self._unload_pipeline()

    def _resolve_model_path(self, model_name: str) -> Union[str, Path]:
        path_obj = Path(model_name)
        if path_obj.is_absolute() and path_obj.exists():
            return model_name
        if model_name in CIVITAI_MODELS:
            filename = CIVITAI_MODELS[model_name]["filename"]
            local_path = self.models_path / filename
            if not local_path.exists():
                self._download_civitai_model(model_name)
            return local_path

        flat_name = model_name.replace("/", "__")

        for root in [self.models_path, getattr(state, "extra_models_path", None)]:
            if not root or not root.exists():
                continue
            p = root / flat_name
            if p.exists():
                return p
            p = root / model_name
            if p.exists():
                return p
            found = list(root.rglob(flat_name)) or list(root.rglob(model_name))
            if found:
                return found[0]

        return model_name

    def _download_civitai_model(self, model_key: str):
        model_info = CIVITAI_MODELS[model_key]
        url = model_info["url"]
        filename = model_info["filename"]
        dest_path = self.models_path / filename
        temp_path = dest_path.with_suffix(".temp")
        ASCIIColors.cyan(f"Downloading '{filename}' from Civitai... to {dest_path}")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(temp_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {filename}") as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            shutil.move(temp_path, dest_path)
            ASCIIColors.green(f"Model '{filename}' downloaded successfully.")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise Exception(f"Failed to download model {filename}: {e}")

    def _set_scheduler(self):
        if not self.pipeline:
            return
        family = _model_family(self.config.get("model_name", ""))
        if family in _SKIP_SCHEDULER_FAMILIES:
            ASCIIColors.info(f"Skipping custom scheduler setup for '{family}' family model.")
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
                ASCIIColors.info(f"Switched scheduler to {scheduler_class_name}")
            except Exception as e:
                ASCIIColors.warning(f"Could not switch scheduler to {scheduler_name_key}: {e}. Using current default.")

    # ------------------------------------------------------------------
    # Quantization config builder
    # ------------------------------------------------------------------
    def _build_quant_config(self) -> "Optional[Any]":
        """
        Build a PipelineQuantizationConfig from the simple quant_backend /
        quant_level / quant_components settings stored in self.config.

        Returns None if quantization is disabled or the required libraries
        are not installed, so callers can safely pass it as
        ``quantization_config=self._build_quant_config()`` and it will be
        silently ignored when None.

        Backend reference
        -----------------
        bitsandbytes_4bit  – NF4/FP4, ~50 % VRAM reduction, requires bitsandbytes
        bitsandbytes_8bit  – INT8,     ~30 % VRAM reduction, requires bitsandbytes
        quanto             – INT8/INT4/FP8, CPU+MPS friendly, requires optimum-quanto
        torchao            – INT8/INT4/FP8, torch.compile-friendly, requires torchao
        gguf               – load pre-quantised .gguf transformer files (Q8_0 … Q2_K)

        Per-family smart defaults (when quant_components is None)
        ---------------------------------------------------------
        FLUX/FLUX2 models: quantize ["transformer", "text_encoder_2"]  (T5 is huge)
        Qwen models      : quantize ["transformer", "text_encoder"]
        Everything else  : quantize ["transformer"]
        """
        backend = self.config.get("quant_backend")
        if not backend:
            return None

        if PipelineQuantizationConfig is None:
            ASCIIColors.warning(
                "quantization requested but PipelineQuantizationConfig is not available. "
                "Upgrade diffusers: pip install git+https://github.com/huggingface/diffusers.git"
            )
            return None

        level      = self.config.get("quant_level")
        components = self.config.get("quant_components")  # may be None → auto
        model_name = self.config.get("model_name", "")
        family = _model_family(model_name)
        torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config.get("torch_dtype_str", "bfloat16"), torch.bfloat16)

        # ── smart component defaults ────────────────────────────────────────────
        if components is None:
            if family in {"flux", "flux_kontext", "flux2_klein", "flux2_dev", "flux_fill", "flux_redux"}:
                components = ["transformer", "text_encoder_2"]
            elif family == "qwen":
                components = ["transformer", "text_encoder"]
            else:
                components = ["transformer"]

        # ── Text-encoder components usually come from Transformers, not Diffusers,
        #    so they need TransformersBnBConfig rather than DiffusersBnBConfig.
        #    We build a quant_mapping so we can mix backends when needed.
        # ──────────────────────────────────────────────────────────────────────────

        try:
            # ── bitsandbytes ──────────────────────────────────────────────────────
            if backend in ("bitsandbytes_4bit", "bitsandbytes_8bit"):
                if DiffusersBnBConfig is None:
                    raise ImportError("bitsandbytes not installed. Run: pip install bitsandbytes")

                if backend == "bitsandbytes_4bit":
                    bnb_type = level or "nf4"
                    diffusers_cfg = DiffusersBnBConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=bnb_type,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                    )
                    transformer_cfg = diffusers_cfg
                    # Text encoders must use the Transformers BnB config
                    te_cfg = None
                    if TransformersBnBConfig is not None:
                        te_cfg = TransformersBnBConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type=bnb_type,
                            bnb_4bit_compute_dtype=torch_dtype,
                            bnb_4bit_use_double_quant=True,
                        )
                else:  # bitsandbytes_8bit
                    diffusers_cfg = DiffusersBnBConfig(load_in_8bit=True)
                    transformer_cfg = diffusers_cfg
                    te_cfg = TransformersBnBConfig(load_in_8bit=True) if TransformersBnBConfig else None

                # Build per-component mapping so text-encoders use the right class
                te_component_names = {"text_encoder", "text_encoder_2", "text_encoder_3"}
                quant_mapping = {}
                for comp in components:
                    if comp in te_component_names and te_cfg is not None:
                        quant_mapping[comp] = te_cfg
                    else:
                        quant_mapping[comp] = transformer_cfg

                ASCIIColors.info(f"Quantization: {backend} ({bnb_type if backend=='bitsandbytes_4bit' else 'int8'}) on {list(quant_mapping.keys())}")
                return PipelineQuantizationConfig(quant_mapping=quant_mapping)

            # ── quanto ────────────────────────────────────────────────────────────
            elif backend == "quanto":
                if DiffusersQuantoConfig is None:
                    raise ImportError("optimum-quanto not installed. Run: pip install optimum-quanto")
                dtype = level or "int8"
                quant_mapping = {comp: DiffusersQuantoConfig(weights_dtype=dtype) for comp in components}
                ASCIIColors.info(f"Quantization: quanto {dtype} on {components}")
                return PipelineQuantizationConfig(quant_mapping=quant_mapping)

            # ── torchao ───────────────────────────────────────────────────────────
            elif backend == "torchao":
                if DiffusersTorchAoConfig is None:
                    raise ImportError("torchao not installed. Run: pip install torchao")
                dtype_str = level or "int8wo"
                quant_mapping = {comp: DiffusersTorchAoConfig(dtype_str) for comp in components}
                ASCIIColors.info(f"Quantization: torchao {dtype_str} on {components}")
                return PipelineQuantizationConfig(quant_mapping=quant_mapping)

            # ── gguf ──────────────────────────────────────────────────────────────
            elif backend == "gguf":
                # GGUF quantizes only the transformer; it must be loaded via
                # from_single_file with a GGUFQuantizationConfig on the model
                # component, not via PipelineQuantizationConfig.
                # We return a sentinel dict instead so _execute_load_pipeline
                # can handle it in the GGUF-specific branch.
                if DiffusersGGUFConfig is None:
                    raise ImportError("gguf not installed. Run: pip install gguf")
                gguf_dtype_str = level or "Q8_0"
                ASCIIColors.info(f"Quantization: GGUF {gguf_dtype_str} (handled in loader)")
                return {"_gguf": True, "compute_dtype": torch_dtype, "gguf_level": gguf_dtype_str}

            else:
                ASCIIColors.warning(f"Unknown quant_backend '{backend}', ignoring.")
                return None

        except ImportError as e:
            ASCIIColors.error(f"Quantization backend '{backend}' unavailable: {e}")
            return None

    # ------------------------------------------------------------------
    # Core loader — branched by model family
    # ------------------------------------------------------------------
    def _load_gguf_pipeline(self, model_name: str, model_path: Union[str, Path], quant_cfg: Dict[str, Any]):
        """
        Loads a GGUF model (such as a GGUF transformer) and integrates it into a standard pipeline.
        """
        import torch
        from diffusers import GGUFQuantizationConfig

        # ── Monkeypatch the Diffusers conversion function to prevent KeyErrors on pruned models (like FLUX.2 Klein) ──
        import diffusers.loaders.single_file_utils as sfu
        import diffusers.loaders.single_file_model as sfm

        # Get original reference
        original_convert_fn = sfu.convert_flux_transformer_checkpoint_to_diffusers

        def patched_convert_fn(checkpoint, *args, **kwargs):
            ASCIIColors.info("[SafeDict] Intercepting GGUF checkpoint conversion with SafeDict wrapper.")
            
            # --- Dynamic Dimension Adaptor on torch.split ---
            original_torch_split = torch.split

            def patched_torch_split(tensor, split_size_or_sections, dim=0, *args_split, **kwargs_split):
                if isinstance(split_size_or_sections, (list, tuple)):
                    actual_sum = sum(split_size_or_sections)
                    tensor_dim_size = tensor.shape[dim]
                    if actual_sum != tensor_dim_size:
                        new_sizes = list(split_size_or_sections)
                        new_sizes[-1] = tensor_dim_size - sum(new_sizes[:-1])
                        ASCIIColors.warning(
                            f"[SafeDict] Adjusting torch.split sizes from {split_size_or_sections} "
                            f"to {new_sizes} to match tensor dim {tensor_dim_size}"
                        )
                        split_size_or_sections = tuple(new_sizes)
                return original_torch_split(tensor, split_size_or_sections, dim=dim, *args_split, **kwargs_split)

            # Apply localized hook to python torch module
            safe_checkpoint = SafeDict(checkpoint)
            original_split_ref = torch.split
            torch.split = patched_torch_split
            if hasattr(sfu, "torch"):
                sfu.torch.split = patched_torch_split
                
            try:
                return original_convert_fn(safe_checkpoint, *args, **kwargs)
            finally:
                # Always restore original methods
                torch.split = original_split_ref
                if hasattr(sfu, "torch"):
                    sfu.torch.split = original_split_ref

        # 1. Patch single_file_utils namespace
        sfu.convert_flux_transformer_checkpoint_to_diffusers = patched_convert_fn

        # 2. Patch single_file_model namespace
        if hasattr(sfm, "convert_flux_transformer_checkpoint_to_diffusers"):
            sfm.convert_flux_transformer_checkpoint_to_diffusers = patched_convert_fn

        # 3. Patch the SINGLE_FILE_LOADABLE_CLASSES dictionary by name or partial name match
        patched_count = 0
        if hasattr(sfm, "SINGLE_FILE_LOADABLE_CLASSES") and isinstance(sfm.SINGLE_FILE_LOADABLE_CLASSES, dict):
            for key, val in sfm.SINGLE_FILE_LOADABLE_CLASSES.items():
                if isinstance(val, dict):
                    fn = val.get("checkpoint_mapping_fn")
                    if fn and (
                        fn == original_convert_fn or 
                        getattr(fn, "__name__", "") == "convert_flux_transformer_checkpoint_to_diffusers" or
                        "convert_flux_transformer" in getattr(fn, "__name__", "")
                    ):
                        val["checkpoint_mapping_fn"] = patched_convert_fn
                        patched_count += 1
                        ASCIIColors.info(f"[SafeDict] Successfully monkeypatched SINGLE_FILE_LOADABLE_CLASSES['{key}']")
        
        if patched_count == 0:
            ASCIIColors.warning("[SafeDict] Warning: No matching loader class was found in SINGLE_FILE_LOADABLE_CLASSES to patch.")

        # Resolve local GGUF file path
        gguf_file = Path(model_path)
        if gguf_file.exists() and gguf_file.is_dir():
            gguf_files = list(gguf_file.glob("*.gguf"))
            if gguf_files:
                # Prioritize GGUF file matching our selected quantization level
                gguf_level = quant_cfg.get("gguf_level", "Q4_K_M")
                level_match = [f for f in gguf_files if gguf_level.lower() in f.name.lower()]
                gguf_file = level_match[0] if level_match else gguf_files[0]
            else:
                raise FileNotFoundError(f"No .gguf file found in folder: {model_path}")
        elif not gguf_file.exists():
            raise FileNotFoundError(f"GGUF file path does not exist: {model_path}")

        ASCIIColors.info(f"Loading GGUF transformer from single file: {gguf_file}")

        compute_dtype = quant_cfg.get("compute_dtype", torch.bfloat16)
        family = _model_family(model_name)

        # ── Determine Base Shell & Config Repository Early ──
        base_shell = self.config.get("gguf_base_shell")
        if not base_shell:
            if "klein" in model_name.lower():
                base_shell = "black-forest-labs/FLUX.2-klein-4B"
            elif "flux.2" in model_name.lower() or "flux2" in model_name.lower():
                base_shell = "black-forest-labs/FLUX.2-dev"
            elif "kontext" in model_name.lower():
                base_shell = "black-forest-labs/FLUX.1-Kontext-dev"
            elif "fill" in model_name.lower():
                base_shell = "black-forest-labs/FLUX.1-Fill-dev"
            elif "redux" in model_name.lower():
                base_shell = "black-forest-labs/FLUX.1-Redux-dev"
            elif family == "qwen":
                base_shell = "Qwen/Qwen-Image-Edit-2509"
            else:
                base_shell = "black-forest-labs/FLUX.1-dev"

        # Determine config repository dynamically to allow custom gated-bypass shells or fallback to base_shell
        gguf_config_shell = self.config.get("gguf_config_shell") or base_shell
        gguf_subfolder = self.config.get("gguf_subfolder") or "transformer"

        load_params_single = {
            "quantization_config": GGUFQuantizationConfig(compute_dtype=compute_dtype),
            "torch_dtype": compute_dtype,
            "config": gguf_config_shell,
            "subfolder": gguf_subfolder,
        }
        
        hf_token = self.config.get("hf_token") or os.environ.get("HF_TOKEN")
        if hf_token:
            load_params_single["token"] = hf_token

        # Resolve the correct Model Class for GGUF loading
        if family == "qwen":
            try:
                from diffusers import QwenImageTransformer2DModel
                model_class = QwenImageTransformer2DModel
            except ImportError:
                from diffusers import FluxTransformer2DModel
                model_class = FluxTransformer2DModel
        else:
            from diffusers import FluxTransformer2DModel
            model_class = FluxTransformer2DModel

        # Load GGUF Transformer natively using the correct class
        transformer = model_class.from_single_file(
            str(gguf_file),
            **load_params_single
        )

        ASCIIColors.info(f"Assembling Pipeline using base shell: {base_shell}")

        load_params = {
            "transformer": transformer,
            "torch_dtype": compute_dtype,
        }
        if hf_token:
            load_params["token"] = hf_token
        if self.config.get("hf_cache_path"):
            load_params["cache_dir"] = str(self.config["hf_cache_path"])
        if not self.config["safety_checker_on"]:
            load_params["safety_checker"] = None

        # Resolve Pipeline Class
        if family == "qwen":
            try:
                from diffusers import QwenImageEditPlusPipeline
                pipeline_class = QwenImageEditPlusPipeline
            except ImportError:
                from diffusers import FluxPipeline
                pipeline_class = FluxPipeline
        else:
            from diffusers import FluxPipeline
            pipeline_class = FluxPipeline

        self.pipeline = pipeline_class.from_pretrained(base_shell, **load_params)

    def _execute_load_pipeline(self, task: str, model_path: Union[str, Path], torch_dtype: Any):
        if platform.system() == "Windows":
            os.environ["HF_HUB_ENABLE_SYMLINKS"] = "0"

        model_name = self.config.get("model_name", "")
        family = _model_family(model_name)
        use_device_map = False

        # ── Shared HF load parameters ───────────────────────────────────────────
        base_params: Dict[str, Any] = {"torch_dtype": torch_dtype}
        if self.config.get("hf_cache_path"):
            base_params["cache_dir"] = str(self.config["hf_cache_path"])

        hf_params = {
            **base_params,
            "use_safetensors": self.config["use_safetensors"],
            "token": self.config.get("hf_token") or os.environ.get("HF_TOKEN"),
            "local_files_only": self.config["local_files_only"],
        }
        if self.config.get("hf_variant"):
            hf_params["variant"] = self.config["hf_variant"]
        if not self.config["safety_checker_on"]:
            hf_params["safety_checker"] = None

        # ── Quantization ──────────────────────────────────────────────────────
        quant_cfg = self._build_quant_config()
        _is_gguf_quant = isinstance(quant_cfg, dict) and quant_cfg.get("_gguf")
        if quant_cfg is not None and not _is_gguf_quant:
            hf_params["quantization_config"] = quant_cfg
            # quantized models must use device_map; disable manual .to() later
            use_device_map = True
            hf_params.setdefault("device_map", "auto")

        try:
            # ── GGUF Quantization override (Must happen first!) ──
            if _is_gguf_quant:
                self._load_gguf_pipeline(model_name, model_path, quant_cfg)
                use_device_map = False

            # ── FLUX.1 Kontext ────────────────────────────────────────────────────
            elif family == "flux_kontext":
                if FluxKontextPipeline is None:
                    raise ImportError("FluxKontextPipeline not available. Upgrade diffusers: pip install git+https://github.com/huggingface/diffusers.git")
                ASCIIColors.info(f"Loading FLUX.1 Kontext pipeline for '{model_name}'.")
                should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                if should_offload:
                    use_device_map = True
                    hf_params["device_map"] = "balanced"
                self.pipeline = FluxKontextPipeline.from_pretrained(model_name, **hf_params)

            # ── FLUX.2 Klein ──────────────────────────────────────────────────────
            elif family == "flux2_klein":
                if Flux2KleinPipeline is None:
                    raise ImportError("Flux2KleinPipeline not available. Upgrade diffusers: pip install git+https://github.com/huggingface/diffusers.git")
                ASCIIColors.info(f"Loading FLUX.2 Klein pipeline for '{model_name}'.")
                should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                if should_offload:
                    use_device_map = True
                    hf_params["device_map"] = "balanced"
                self.pipeline = Flux2KleinPipeline.from_pretrained(model_name, **hf_params)

            # ── FLUX.2 Dev ────────────────────────────────────────────────────────
            elif family == "flux2_dev":
                if Flux2Pipeline is None:
                    raise ImportError("Flux2Pipeline not available. Upgrade diffusers: pip install git+https://github.com/huggingface/diffusers.git")
                ASCIIColors.info(f"Loading FLUX.2 Dev pipeline for '{model_name}'.")
                should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                if should_offload:
                    use_device_map = True
                    hf_params["device_map"] = "balanced"
                self.pipeline = Flux2Pipeline.from_pretrained(model_name, **hf_params)

            # ── FLUX.1 Fill (dedicated inpainting) ────────────────────────────────
            elif family == "flux_fill":
                if FluxFillPipeline is None:
                    raise ImportError("FluxFillPipeline not available. Upgrade diffusers: pip install git+https://github.com/huggingface/diffusers.git")
                ASCIIColors.info(f"Loading FLUX.1 Fill pipeline for '{model_name}'.")
                should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                if should_offload:
                    use_device_map = True
                    hf_params["device_map"] = "balanced"
                self.pipeline = FluxFillPipeline.from_pretrained(model_name, **hf_params)

            # ── FLUX.1 Redux (image variation) ────────────────────────────────────
            elif family == "flux_redux":
                if FluxPriorReduxPipeline is None:
                    raise ImportError("FluxPriorReduxPipeline not available. Upgrade diffusers: pip install git+https://github.com/huggingface/diffusers.git")
                ASCIIColors.info(f"Loading FLUX.1 Redux pipeline for '{model_name}'.")
                should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                if should_offload:
                    use_device_map = True
                    hf_params["device_map"] = "balanced"
                self.pipeline = FluxPriorReduxPipeline.from_pretrained(model_name, **hf_params)

            # ── Generic FLUX (Schnell / Dev via AutoPipeline) ─────────────────────
            elif family == "flux":
                ASCIIColors.info(f"Loading generic FLUX pipeline for '{model_name}'.")
                should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                if should_offload:
                    use_device_map = True
                    hf_params["device_map"] = "balanced"
                self.pipeline = AutoPipelineForText2Image.from_pretrained(model_name, **hf_params)

            # ── Qwen Image Edit ───────────────────────────────────────────────────
            elif family == "qwen":
                ASCIIColors.info(f"Loading Qwen Image Edit pipeline for '{model_name}'.")
                should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                if should_offload:
                    use_device_map = True
                    hf_params["device_map"] = "balanced"
                if "Qwen-Image-Edit" in model_name:
                    try:
                        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(model_name, **hf_params)
                    except Exception:
                        self.pipeline = QwenImageEditPipeline.from_pretrained(model_name, **hf_params)
                else:
                    self.pipeline = DiffusionPipeline.from_pretrained(model_name, **hf_params)

            # ── Standard (SDXL, SD1.5, etc.) ─────────────────────────────────────
            else:
                # ── GGUF path: load pre-quantised transformer .gguf + pipeline shell
                if _is_gguf_quant:
                    self._load_gguf_pipeline(model_name, model_path, quant_cfg)
                    use_device_map = False
                else:
                    ASCIIColors.info(f"Loading standard model from Hub: {model_path}")
                    is_large = "stable-diffusion-3" in str(model_path)
                    should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                    if is_large and should_offload:
                        use_device_map = True
                        hf_params["device_map"] = "auto"
                    if task == "text2image":
                        self.pipeline = AutoPipelineForText2Image.from_pretrained(model_path, **hf_params)
                    elif task == "image2image":
                        self.pipeline = AutoPipelineForImage2Image.from_pretrained(model_path, **hf_params)
                    elif task == "inpainting":
                        self.pipeline = AutoPipelineForInpainting.from_pretrained(model_path, **hf_params)

        except Exception as e:
            err = str(e).lower()
            if "401" in err or "gated" in err or "authorization" in err:
                raise RuntimeError(
                    f"AUTHENTICATION FAILED for model '{model_name}'. "
                    "Please accept the model license on Hugging Face and provide a valid HF token."
                )
            raise e

        # Fix: upcast VAE to float32 to prevent black/chunky artifacts
        if self.pipeline and hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'dtype'):
            if self.pipeline.vae.dtype == torch.float16:
                ASCIIColors.info("Upcasting VAE to float32 to prevent artifacts.")
                self.pipeline.vae = self.pipeline.vae.to(dtype=torch.float32)

        self._set_scheduler()

        if not use_device_map:
            self.pipeline.to(self.config["device"])
            if self.config["enable_xformers"]:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    ASCIIColors.warning(f"Could not enable xFormers: {e}.")
            if self.config["enable_cpu_offload"] and self.config["device"] != "cpu":
                self.pipeline.enable_model_cpu_offload()
            elif self.config["enable_sequential_cpu_offload"] and self.config["device"] != "cpu":
                self.pipeline.enable_sequential_cpu_offload()
        else:
            ASCIIColors.info("Device map handled device placement. Skipping manual pipeline.to() and offload calls.")

        if self.pipeline:
            sig = inspect.signature(self.pipeline.__call__)
            self.supported_args = {p.name for p in sig.parameters.values()}
            ASCIIColors.info(f"Pipeline supported arguments detected: {self.supported_args}")

        self.is_loaded = True
        self.current_task = task
        self.last_used_time = time.time()
        ASCIIColors.green(
            f"Model '{model_name}' (family: {family}) loaded successfully "
            f"using '{'device_map' if use_device_map else 'standard'}' mode for task '{task}'."
        )

    def _load_pipeline_for_task(self, task: str):
        if self.pipeline and self.current_task == task:
            return
        if self.pipeline:
            self._unload_pipeline()

        model_name = self.config.get("model_name", "")
        if not model_name:
            raise ValueError("Model name cannot be empty for loading.")

        # Ensure device and dtype are resolved before loading
        if self.config.get("device", "auto").lower() == "auto":
            if torch.cuda.is_available():
                self.config["device"] = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.config["device"] = "mps"
            else:
                self.config["device"] = "cpu"

        # Resolve dtype if auto
        if self.config.get("torch_dtype_str", "auto").lower() == "auto":
            self.config["torch_dtype_str"] = "float16" if self.config["device"] != "cpu" else "float32"

        ASCIIColors.info(f"Loading Diffusers model: {model_name} for task: {task}")
        model_path = self._resolve_model_path(model_name)
        torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower())

        try:
            self._execute_load_pipeline(task, model_path, torch_dtype)
            return
        except Exception as e:
            if "out of memory" not in str(e).lower() or not hasattr(self, 'registry'):
                raise e

        ASCIIColors.warning(f"Failed to load '{model_name}' due to OOM. Attempting to unload other models to free VRAM.")
        candidates = sorted(
            [m for m in self.registry.get_all_managers() if m is not self and m.is_loaded],
            key=lambda m: m.last_used_time
        )
        if not candidates:
            raise Exception("OOM error, but no other models are available to unload.")

        last_exc = e
        for victim in candidates:
            ASCIIColors.info(f"Unloading '{victim.config['model_name']}' to free VRAM.")
            victim._unload_pipeline()
            try:
                self._execute_load_pipeline(task, model_path, torch_dtype)
                ASCIIColors.green(f"Successfully loaded '{model_name}' after freeing VRAM.")
                return
            except Exception as retry_e:
                last_exc = retry_e
                if "out of memory" not in str(retry_e).lower():
                    raise retry_e

        raise last_exc

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
            ASCIIColors.info(f"Model '{model_name}' unloaded and VRAM cleared.")

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
                        ASCIIColors.warning("Supported argument set not found. Using unfiltered arguments.")
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
            "local_files_only","hf_cache_path","unload_inactive_model_after",
            "quant_backend","quant_level","quant_components"
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
                    ASCIIColors.info(f"Reference count for model '{config.get('model_name')}' is zero. Cleaning up manager.")
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
                ASCIIColors.info(f"Acquiring initial model manager for '{self.config['model_name']}' on startup.")
                self.manager = self.registry.get_manager(self.config, self.models_path)
            except Exception as e:
                ASCIIColors.error(f"Failed to acquire model manager on startup: {e}")
                self.manager = None

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": "", "device": "auto", "torch_dtype_str": "auto", "use_safetensors": True,
            "scheduler_name": "default", "safety_checker_on": True, "num_inference_steps": 25,
            "guidance_scale": 7.0, "width": 1024, "height": 1024, "seed": -1,
            "enable_cpu_offload": False, "enable_sequential_cpu_offload": False, "enable_xformers": False,
            "hf_variant": None, "hf_token": None, "hf_cache_path": None, "local_files_only": False,
            "unload_inactive_model_after": 0,
            # ── Quantization ──────────────────────────────────────────────────────
            # quant_backend : null | "bitsandbytes_4bit" | "bitsandbytes_8bit" | "quanto" | "torchao" | "gguf"
            # quant_level   : nf4/fp4 (bnb4) | int8/int4/float8 (quanto) | int8wo/int4wo/fp8wo (torchao) | Q8_0/Q4_K_M/... (gguf)
            # quant_components: list of pipeline component names, e.g. ["transformer","text_encoder_2"]
            #                   null = smart per-family defaults
            "quant_backend": None,
            "quant_level": None,
            "quant_components": None,
        }

    def save_config(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            ASCIIColors.info(f"Server config saved to {self.config_path}")
        except Exception as e:
            ASCIIColors.error(f"Failed to save server config: {e}")

    def load_config(self):
        default_config = self.get_default_config()
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
                self.config = default_config
                ASCIIColors.info(f"Loaded server configuration from {self.config_path}")
            except (json.JSONDecodeError, IOError) as e:
                ASCIIColors.warning(f"Could not load config file, using defaults. Error: {e}")
                self.config = default_config
        else:
            self.config = default_config
        self.save_config()

    def _resolve_device_and_dtype(self):
        if self.config.get("device", "auto").lower() == "auto":
            self.config["device"] = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        family = _model_family(self.config.get("model_name", ""))
        is_bfloat_preferred = family in {"flux", "flux_kontext", "flux2_klein", "flux2_dev", "flux_fill", "flux_redux", "qwen"}
        if is_bfloat_preferred and self.config["device"] == "cuda":
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                self.config["torch_dtype_str"] = "bfloat16"
                ASCIIColors.info(f"'{family}' model on compatible hardware — forcing dtype to bfloat16 for stability.")
                return

        if self.config["torch_dtype_str"].lower() == "auto":
            self.config["torch_dtype_str"] = "float16" if self.config["device"] != "cpu" else "float32"

    def update_settings(self, new_settings: Dict[str, Any]):
        if 'model' in new_settings and 'model_name' not in new_settings:
            new_settings['model_name'] = new_settings.pop('model')

        if self.config.get("model_name") and not new_settings.get("model_name"):
            ASCIIColors.info("Incoming settings have no model_name. Preserving existing model.")
            new_settings["model_name"] = self.config["model_name"]

        if self.manager:
            self.registry.release_manager(self.manager.config)
            self.manager = None

        self.config.update(new_settings)
        ASCIIColors.info(f"Server config updated. Current model_name: {self.config.get('model_name')}")
        self._resolve_device_and_dtype()

        if self.config.get("model_name"):
            ASCIIColors.info("Acquiring model manager with updated configuration...")
            self.manager = self.registry.get_manager(self.config, self.models_path)
        else:
            ASCIIColors.warning("No model_name in config after update, manager not acquired.")

        self.save_config()
        return True

    def get_active_manager(self) -> ModelManager:
        if self.manager:
            return self.manager
        raise HTTPException(status_code=400, detail="No model is configured or manager is not active. Please set a model using the /set_settings endpoint.")


state: Optional[ServerState] = None

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)

class EditRequestPayload(BaseModel):
    prompt: str
    image_paths: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)

class EditRequestJSON(BaseModel):
    prompt: str
    images_b64: List[str] = Field(description="A list of Base64 encoded image strings.")
    params: Dict[str, Any] = Field(default_factory=dict)


def get_sanitized_request_for_logging(request_data: Any) -> Dict[str, Any]:
    import copy
    try:
        if hasattr(request_data, 'model_dump'):
            data = request_data.model_dump()
        elif isinstance(request_data, dict):
            data = copy.deepcopy(request_data)
        else:
            return {"error": "Unsupported data type for sanitization"}
        if 'images_b64' in data and isinstance(data['images_b64'], list):
            count = len(data['images_b64'])
            data['images_b64'] = f"[<{count} base64 image(s) truncated>]"
        if 'params' in data and isinstance(data.get('params'), dict):
            if 'mask_image' in data['params'] and isinstance(data['params']['mask_image'], str):
                original_len = len(data['params']['mask_image'])
                data['params']['mask_image'] = f"[<base64 mask truncated, len={original_len}>]"
        return data
    except Exception:
        return {"error": "Failed to sanitize request data."}


# ---------------------------------------------------------------------------
# Helper: build pipeline_args for new FLUX family models
# ---------------------------------------------------------------------------

def _enrich_args_for_family(family: str, pipeline_args: Dict[str, Any], pil_images: List, seed: int, manager: ModelManager) -> str:
    """
    Mutate pipeline_args in-place for new-gen FLUX models and return the task string.
    Returns the task name to use when submitting to the worker queue.
    """
    task = "text2image"

    if family == "flux_kontext":
        # Kontext works as an image-to-image editor. An input image is mandatory.
        if pil_images:
            pipeline_args["image"] = pil_images[0]
        else:
            # Pure text-to-image not supported natively; generate a blank placeholder
            rng = np.random.default_rng(seed=seed if seed != -1 else None)
            h, w = pipeline_args.get("height", 1024), pipeline_args.get("width", 1024)
            pipeline_args["image"] = Image.fromarray(
                rng.integers(0, 256, (h, w, 3), dtype=np.uint8), 'RGB'
            )
        # Kontext recommends guidance_scale=2.5–4, fewer steps for distilled variant
        pipeline_args.setdefault("guidance_scale", 2.5)
        pipeline_args.setdefault("num_inference_steps", 28)
        task = "image2image"

    elif family == "flux2_klein":
        # Klein 4B distilled is very fast: 4 steps with guidance_scale≈1 (CFG-free)
        # Klein 9B / base variants need more steps and higher guidance.
        mn = manager.config.get("model_name", "")
        if "base" in mn.lower():
            pipeline_args.setdefault("guidance_scale", 4.0)
            pipeline_args.setdefault("num_inference_steps", 50)
        else:
            # distilled
            pipeline_args.setdefault("guidance_scale", 1.0)
            pipeline_args.setdefault("num_inference_steps", 4)
        # Klein also supports editing when an image is supplied
        if pil_images:
            pipeline_args["image"] = pil_images[0]
            task = "image2image"
        else:
            task = "text2image"

    elif family == "flux2_dev":
        pipeline_args.setdefault("guidance_scale", 2.5)
        pipeline_args.setdefault("num_inference_steps", 50)
        if pil_images:
            pipeline_args["image"] = pil_images[0]
            task = "image2image"

    elif family == "flux_fill":
        # Fill expects image + mask_image; task = inpainting
        if pil_images:
            pipeline_args["image"] = pil_images[0]
        task = "inpainting"

    elif family == "flux_redux":
        # Redux is a prior model – it takes an image and returns prompt embeds that
        # are fed back to the same pipeline (unusual single-pipeline pattern here).
        if pil_images:
            pipeline_args["image"] = pil_images[0]
        pipeline_args.setdefault("guidance_scale", 2.5)
        pipeline_args.setdefault("num_inference_steps", 50)
        task = "image2image"

    elif family == "flux":
        # Generic FLUX.1 (Schnell / Dev)
        mn = manager.config.get("model_name", "")
        if "schnell" in mn.lower():
            pipeline_args.setdefault("guidance_scale", 0.0)
            pipeline_args.setdefault("num_inference_steps", 4)
        else:
            pipeline_args.setdefault("guidance_scale", 3.5)
            pipeline_args.setdefault("num_inference_steps", 50)
        if pil_images:
            pipeline_args["image"] = pil_images[0]
            task = "image2image"

    return task


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate_image")
async def generate_image(request: T2IRequest):
    manager = None
    temp_config = None
    try:
        if "model_name" in request.params and request.params["model_name"]:
            temp_config = state.config.copy()
            temp_config["model_name"] = request.params.pop("model_name")
            manager = state.registry.get_manager(temp_config, state.models_path)
            ASCIIColors.info(f"Using per-request model: {temp_config['model_name']}")
        else:
            manager = state.get_active_manager()
            ASCIIColors.info(f"Using session-configured model: {manager.config.get('model_name')}")

        pipeline_args = manager.config.copy()
        pipeline_args.update(request.params)

        pipeline_args["prompt"] = request.prompt
        pipeline_args["negative_prompt"] = request.negative_prompt

        width             = pipeline_args.get("width", 1024)
        height            = pipeline_args.get("height", 1024)
        num_steps         = pipeline_args.get("num_inference_steps", 25)
        seed              = pipeline_args.get("seed", -1)
        guidance_scale    = pipeline_args.get("guidance_scale", 7.0)
        pipeline_args["width"]               = int(width   or 1024)
        pipeline_args["height"]              = int(height  or 1024)
        pipeline_args["num_inference_steps"] = int(num_steps or 25)
        pipeline_args["guidance_scale"]      = float(guidance_scale or 7.0)
        seed = int(seed if seed is not None else -1)

        pipeline_args["generator"] = None
        if seed != -1:
            pipeline_args["generator"] = torch.Generator(device=manager.config["device"]).manual_seed(seed)

        model_name = manager.config.get("model_name", "")
        family = _model_family(model_name)

        # Route to new-gen enricher or legacy Qwen special-casing
        if family in {"flux_kontext", "flux2_klein", "flux2_dev", "flux_fill", "flux_redux", "flux"}:
            task = _enrich_args_for_family(family, pipeline_args, [], seed, manager)
        elif "Qwen-Image-Edit" in model_name:
            rng = np.random.default_rng(seed=seed if seed != -1 else None)
            h, w = pipeline_args["height"], pipeline_args["width"]
            pipeline_args["image"] = Image.fromarray(
                rng.integers(0, 256, (h, w, 3), dtype=np.uint8), 'RGB'
            )
            pipeline_args["strength"] = float(pipeline_args.get("strength", 1.0))
            task = "image2image"
        else:
            task = "text2image"

        log_args = {k: v for k, v in pipeline_args.items() if k not in ['generator', 'image']}
        if pipeline_args.get("generator"):  log_args['generator'] = f"<torch.Generator(seed={seed})>"
        if pipeline_args.get("image"):      log_args['image']     = "<PIL Image object>"
        ASCIIColors.cyan("--- Generating Image with Settings ---")
        try:
            print(json.dumps(log_args, indent=2, default=str))
        except Exception:
            print(log_args)
        ASCIIColors.cyan("------------------------------------")

        future = Future()
        manager.queue.put((future, task, pipeline_args))
        return Response(content=future.result(), media_type="image/png")

    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_config and manager:
            state.registry.release_manager(temp_config)
            ASCIIColors.info(f"Released per-request model: {temp_config['model_name']}")


@router.post("/edit_image")
async def edit_image(request: EditRequestJSON):
    manager = None
    temp_config = None
    ASCIIColors.info(f"Received /edit_image request with {len(request.images_b64)} image(s).")
    ASCIIColors.info(request.params)
    try:
        if "model_name" in request.params and request.params["model_name"]:
            temp_config = state.config.copy()
            temp_config["model_name"] = request.params.pop("model_name")
            manager = state.registry.get_manager(temp_config, state.models_path)
            ASCIIColors.info(f"Using per-request model: {temp_config['model_name']}")
        else:
            manager = state.get_active_manager()
            ASCIIColors.info(f"Using session-configured model: {manager.config.get('model_name')}")

        pipeline_args = manager.config.copy()
        pipeline_args.update(request.params)
        pipeline_args["prompt"] = request.prompt

        model_name = manager.config.get("model_name", "")
        family = _model_family(model_name)

        pil_images = []
        for b64_string in request.images_b64:
            b64_data = b64_string.split(";base64,")[1] if ";base64," in b64_string else b64_string
            pil_images.append(Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB"))

        if not pil_images:
            raise HTTPException(status_code=400, detail="No valid images provided.")

        seed = int(pipeline_args.get("seed", -1))
        pipeline_args["generator"] = None
        if seed != -1:
            pipeline_args["generator"] = torch.Generator(device=manager.config["device"]).manual_seed(seed)

        # Decode mask if present
        if "mask_image" in pipeline_args and pipeline_args["mask_image"]:
            b64_mask = pipeline_args["mask_image"]
            b64_data = b64_mask.split(";base64,")[1] if ";base64," in b64_mask else b64_mask
            pipeline_args["mask_image"] = Image.open(BytesIO(base64.b64decode(b64_data))).convert("L")

        # Route by family
        if family in {"flux_kontext", "flux2_klein", "flux2_dev", "flux_fill", "flux_redux", "flux"}:
            task = _enrich_args_for_family(family, pipeline_args, pil_images, seed, manager)

        elif "Qwen-Image-Edit-2509" in model_name:
            task = "image2image"
            pipeline_args.update({
                "true_cfg_scale":       pipeline_args.get("true_cfg_scale", 4.0),
                "guidance_scale":       pipeline_args.get("guidance_scale", 1.0),
                "num_inference_steps":  pipeline_args.get("num_inference_steps", 40),
                "negative_prompt":      pipeline_args.get("negative_prompt", ""),
            })
            edit_mode = pipeline_args.get("edit_mode", "fusion")
            if edit_mode == "fusion":
                pipeline_args["image"] = pil_images
        else:
            pipeline_args["image"] = pil_images[0]
            task = "inpainting" if ("mask_image" in pipeline_args and pipeline_args["mask_image"]) else "image2image"

        log_args = {k: v for k, v in pipeline_args.items() if k not in ['generator', 'image', 'mask_image']}
        if pipeline_args.get("generator"):   log_args['generator']  = f"<torch.Generator(seed={seed})>"
        if 'image' in pipeline_args:         log_args['image']      = f"[<{len(pil_images)} PIL Image(s)>]"
        if pipeline_args.get("mask_image"):  log_args['mask_image'] = "<PIL Mask Image>"
        ASCIIColors.cyan("--- Editing Image with Settings ---")
        try:
            print(json.dumps(log_args, indent=2, default=str))
        except Exception:
            print(log_args)
        ASCIIColors.cyan("---------------------------------")

        future = Future()
        manager.queue.put((future, task, pipeline_args))
        return Response(content=future.result(), media_type="image/png")

    except Exception as e:
        ASCIIColors.error(f"Exception in /edit_image. Sanitized Payload: {json.dumps(get_sanitized_request_for_logging(request), indent=2)}")
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_config and manager:
            state.registry.release_manager(temp_config)
            ASCIIColors.info(f"Released per-request model: {temp_config['model_name']}")


@router.post("/pull_model")
def pull_model_endpoint(payload: PullModelRequest):
    if not payload.hf_id and not payload.safetensors_url:
        raise HTTPException(status_code=400, detail="Provide either 'hf_id' or 'safetensors_url'.")

    if payload.hf_id:
        model_id = payload.hf_id.strip()
        folder_name = payload.local_name or model_id.replace("/", "__")
        dest_dir = state.models_path / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            ASCIIColors.cyan(f"Pulling HF model '{model_id}' into {dest_dir}")

            # Download directly to destination, bypassing default HF cache
            token = state.config.get("hf_token") or os.environ.get("HF_TOKEN")
            download_params = {
                "repo_id": model_id,
                "local_dir": dest_dir,
                "local_dir_use_symlinks": False,
                "token": token,
            }
            if payload.allow_patterns:
                download_params["allow_patterns"] = payload.allow_patterns

            snapshot_download(**download_params)

            # If no model_index.json, load and re-save to ensure proper diffusers format
            if not (dest_dir / "model_index.json").exists():
                ASCIIColors.info("No model_index.json found. Attempting to convert to Diffusers format...")
                token = state.config.get("hf_token")
                load_params = {}
                if token:
                    load_params["token"] = token
                try:
                    pipe = DiffusionPipeline.from_pretrained(str(dest_dir), **load_params)
                    pipe.save_pretrained(dest_dir)
                    del pipe
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as ex:
                    ASCIIColors.warning(f"Could not convert directory to a standard Diffusers pipeline: {ex}. This folder might contain raw checkpoints/single-file weights.")

            ASCIIColors.green(f"Model '{model_id}' pulled to {dest_dir}")
            return {"status": "ok", "model_name": folder_name}
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to pull HF model: {e}")

    if payload.safetensors_url:
        url = payload.safetensors_url.strip()
        default_name = url.split("/")[-1] or "model.safetensors"
        if not default_name.endswith(".safetensors"):
            default_name += ".safetensors"
        filename = payload.local_name or default_name
        dest_path = state.models_path / filename
        temp_path = dest_path.with_suffix(".temp")
        ASCIIColors.cyan(f"Downloading safetensors from {url} to {dest_path}")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(temp_path, "wb") as f, tqdm(total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {filename}") as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bar.update(len(chunk))
            shutil.move(temp_path, dest_path)
            ASCIIColors.green(f"Safetensors file downloaded to {dest_path}")
            return {"status": "ok", "model_name": filename}
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to download safetensors: {e}")


@router.get("/list_local_models")
def list_local_models_endpoint():
    local_models = set()

    def scan_root(root: Path):
        if not root or not root.exists():
            return
        for model_index in root.rglob("model_index.json"):
            local_models.add(model_index.parent.name)
        for safepath in root.rglob("*.safetensors"):
            if (safepath.parent / "model_index.json").exists():
                continue
            local_models.add(safepath.name)

    scan_root(Path(args.models_path))
    if args.extra_models_path:
        scan_root(Path(args.extra_models_path))
    return sorted(list(local_models))


@app.get("/list_models")
def list_models() -> list[dict]:
    result = []
    seen_paths = set()

    def scan_root(root: Path):
        if not root or not root.exists():
            return
        for model_index in root.rglob("model_index.json"):
            folder = model_index.parent
            resolved = str(folder.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            result.append({"model_name": resolved, "display_name": folder.name, "description": "Local Diffusers pipeline"})
        for safepath in root.rglob("*.safetensors"):
            if (safepath.parent / "model_index.json").exists():
                continue
            resolved = str(safepath.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            result.append({"model_name": resolved, "display_name": safepath.stem, "description": "Local .safetensors checkpoint"})

    scan_root(Path(args.models_path))
    if args.extra_models_path:
        scan_root(Path(args.extra_models_path))
    return result


@router.get("/list_available_models")
def list_available_models_endpoint():
    return sorted({m['model_name'] for m in list_models()})


@router.get("/list_catalog")
def list_catalog_endpoint():
    """Return the full curated model catalogue (public + gated) as a structured list."""
    catalog = []
    for category, models in {**HF_PUBLIC_MODELS, **HF_GATED_MODELS}.items():
        for m in models:
            catalog.append({
                "category":     category,
                "model_name":   m["model_name"],
                "display_name": m.get("display_name", m["model_name"]),
                "description":  m.get("desc", ""),
                "family":       _model_family(m["model_name"]),
            })
    return catalog


@router.get("/get_settings")
def get_settings_endpoint():
    settings_list = []
    available_models = list_available_models_endpoint()
    schedulers = list(SCHEDULER_MAPPING.keys())
    config_to_display = state.config or state.get_default_config()
    for name, value in config_to_display.items():
        setting = {"name": name, "type": str(type(value).__name__), "value": value}
        if name == "model_name":    setting["options"] = available_models
        if name == "scheduler_name": setting["options"] = schedulers
        settings_list.append(setting)
    return settings_list


@router.post("/set_settings")
def set_settings_endpoint(settings: Dict[str, Any]):
    try:
        return {"success": state.update_settings(settings)}
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def status_endpoint():
    return {
        "status": "running",
        "diffusers_available": DIFFUSERS_AVAILABLE,
        "model_loaded": state.manager.is_loaded if state.manager else False,
        "missing_pipeline_classes": _MISSING_PIPELINES,
    }


@router.post("/unload_model")
def unload_model_endpoint():
    if state.manager:
        state.manager._unload_pipeline()
        state.registry.release_manager(state.manager.config)
        state.manager = None
    return {"status": "unloaded"}


@router.get("/ps")
def ps_endpoint():
    return [{
        "model_name": m.config.get("model_name"), "is_loaded": m.is_loaded,
        "task": m.current_task, "device": m.config.get("device"), "ref_count": m.ref_count,
        "queue_size": m.queue.qsize(), "last_used": time.ctime(m.last_used_time),
        "family": _model_family(m.config.get("model_name", "")),
    } for m in state.registry.get_all_managers()]


app.include_router(router)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Diffusers TTI Server")
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=int, default=9630)
        parser.add_argument("--models-path", type=str, required=True)
        parser.add_argument("--extra-models-path", type=str, default=None)
        parser.add_argument("--hf-token", type=str, default=None,
                            help="Hugging Face access token for gated/private repos.")

        args = parser.parse_args()
        MODELS_PATH      = Path(args.models_path)
        EXTRA_MODELS_PATH = Path(args.extra_models_path) if args.extra_models_path else None
        state = ServerState(MODELS_PATH, EXTRA_MODELS_PATH)

        if args.hf_token:
            state.config["hf_token"] = args.hf_token
            ASCIIColors.info("Hugging Face token received via CLI and stored in server config.")

        ASCIIColors.cyan("─── Diffusers TTI Server ───────────────────────────────────────")
        ASCIIColors.green(f"Starting on http://{args.host}:{args.port}")
        ASCIIColors.green(f"Models path : {MODELS_PATH.resolve()}")
        if EXTRA_MODELS_PATH:
            ASCIIColors.green(f"Extra models: {EXTRA_MODELS_PATH.resolve()}")
        if not DIFFUSERS_AVAILABLE:
            ASCIIColors.error("Diffusers or its dependencies are not installed correctly!")
        else:
            ASCIIColors.info(f"Device      : {state.config['device']}")
            if _MISSING_PIPELINES:
                ASCIIColors.warning(
                    f"Some next-gen pipeline classes are unavailable "
                    f"({', '.join(_MISSING_PIPELINES)}). "
                    "Run: pip install git+https://github.com/huggingface/diffusers.git"
                )
            else:
                ASCIIColors.green("All next-gen pipeline classes detected ✓")
        ASCIIColors.cyan("────────────────────────────────────────────────────────────────")

        uvicorn.run(app, host=args.host, port=args.port, reload=False)
    except Exception as ex:
        trace_exception(ex)
