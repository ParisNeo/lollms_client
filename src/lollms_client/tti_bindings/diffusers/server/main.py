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
from fastapi import Body

class PullModelRequest(BaseModel):
    hf_id: Optional[str] = Field(default=None, description="Hugging Face repo id or URL, e.g. 'stabilityai/sdxl-turbo'")
    safetensors_url: Optional[str] = Field(default=None, description="Direct URL to a .safetensors file")
    local_name: Optional[str] = Field(default=None, description="Optional name/folder under models/")

# Add binding root to sys.path to ensure local modules can be imported if structured that way.
binding_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(binding_root))

# --- Dependency Check and Imports ---
try:
    import torch
    from diffusers import (
        AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting,
        DiffusionPipeline, StableDiffusionPipeline, QwenImageEditPipeline, QwenImageEditPlusPipeline
    )
    from diffusers.utils import load_image
    from PIL import Image
    from ascii_colors import trace_exception, ASCIIColors
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    print(f"FATAL: A required package is missing from the server's venv: {e}.")
    DIFFUSERS_AVAILABLE = False
    # Define dummy classes to allow server to start and report error via API
    class Dummy: pass
    torch = Dummy()
    torch.cuda = Dummy()
    torch.cuda.is_available = lambda: False
    torch.backends = Dummy()
    torch.backends.mps = Dummy()
    torch.backends.mps.is_available = lambda: False
    AutoPipelineForText2Image = AutoPipelineForImage2Image = AutoPipelineForInpainting = DiffusionPipeline = StableDiffusionPipeline = QwenImageEditPipeline = QwenImageEditPlusPipeline = Image = load_image = ASCIIColors = trace_exception = Dummy

# --- Server Setup ---
app = FastAPI(title="Diffusers TTI Server")
router = APIRouter()
MODELS_PATH = Path("./models")

# --- START: Core Logic (Complete and Unabridged) ---
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
    ]
}

HF_GATED_MODELS = {
    "Next-Generation (Gated Access Required)": [
        {"model_name": "stabilityai/stable-diffusion-3-medium-diffusers", "display_name": "Stable Diffusion 3 Medium", "desc": "State-of-the-art model with advanced prompt understanding. Requires free registration."},
        {"model_name": "black-forest-labs/FLUX.1-schnell", "display_name": "FLUX.1 Schnell", "desc": "A powerful and extremely fast next-generation model. Requires access request."},
        {"model_name": "black-forest-labs/FLUX.1-dev", "display_name": "FLUX.1 Dev", "desc": "The larger developer version of the FLUX.1 model. Requires access request."},
    ]
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

        # Search in extra models path
        if state.extra_models_path and state.extra_models_path.exists():
            found_paths = list(state.extra_models_path.rglob(model_name))
            if found_paths:
                ASCIIColors.info(f"Found model in extra path: {found_paths[0]}")
                return found_paths[0]

        # Search in primary models path
        found_paths = list(self.models_path.rglob(model_name))
        if found_paths:
            ASCIIColors.info(f"Found model in primary path: {found_paths[0]}")
            return found_paths[0]

        # Fallback for HF hub models that are folders, not single files.
        local_path = self.models_path / model_name
        if local_path.exists():
            return local_path

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
        if "Qwen" in self.config.get("model_name", "") or "FLUX" in self.config.get("model_name", ""):
            ASCIIColors.info("Special model detected, skipping custom scheduler setup.")
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

            is_qwen_model = "Qwen" in model_name_from_config
            is_flux_model = "FLUX" in model_name_from_config

            if is_qwen_model or is_flux_model:
                ASCIIColors.info(f"Special model '{model_name_from_config}' detected. Using dedicated pipeline loader.")
                load_params.update({
                    "use_safetensors": self.config["use_safetensors"],
                    "token": self.config["hf_token"],
                    "local_files_only": self.config["local_files_only"]
                })
                if self.config["hf_variant"]:
                    load_params["variant"] = self.config["hf_variant"]
                if not self.config["safety_checker_on"]:
                    load_params["safety_checker"] = None
                
                should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                if should_offload:
                    ASCIIColors.info(f"Offload enabled. Forcing device_map='auto' for {model_name_from_config}.")
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
                    ASCIIColors.info(f"Loading standard model from local .safetensors file: {model_path}")
                    try:
                        self.pipeline = AutoPipelineForText2Image.from_single_file(model_path, **load_params)
                    except Exception as e:
                        ASCIIColors.warning(f"Failed to load with AutoPipeline, falling back to StableDiffusionPipeline: {e}")
                        self.pipeline = StableDiffusionPipeline.from_single_file(model_path, **load_params)
                else:
                    ASCIIColors.info(f"Loading standard model from Hub: {model_path}")
                    load_params.update({
                        "use_safetensors": self.config["use_safetensors"],
                        "token": self.config["hf_token"],
                        "local_files_only": self.config["local_files_only"]
                    })
                    if self.config["hf_variant"]:
                        load_params["variant"] = self.config["hf_variant"]
                    if not self.config["safety_checker_on"]:
                        load_params["safety_checker"] = None
                    
                    is_large_model = "stable-diffusion-3" in str(model_path)
                    should_offload = self.config["enable_cpu_offload"] or self.config["enable_sequential_cpu_offload"]
                    if is_large_model and should_offload:
                        ASCIIColors.info(f"Large model '{model_path}' detected with offload enabled. Using device_map='auto'.")
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
        ASCIIColors.green(f"Model '{model_name_from_config}' loaded successfully using '{'device_map' if use_device_map else 'standard'}' mode for task '{task}'.")

    def _load_pipeline_for_task(self, task: str):
        if self.pipeline and self.current_task == task:
            return
        if self.pipeline:
            self._unload_pipeline()
        
        model_name = self.config.get("model_name", "")
        if not model_name:
            raise ValueError("Model name cannot be empty for loading.")
        
        ASCIIColors.info(f"Loading Diffusers model: {model_name} for task: {task}")
        model_path = self._resolve_model_path(model_name)
        torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower())
        
        try:
            self._execute_load_pipeline(task, model_path, torch_dtype)
            return
        except Exception as e:
            is_oom = "out of memory" in str(e).lower()
            if not is_oom or not hasattr(self, 'registry'):
                raise e
        
        ASCIIColors.warning(f"Failed to load '{model_name}' due to OOM. Attempting to unload other models to free VRAM.")
        
        candidates_to_unload = [m for m in self.registry.get_all_managers() if m is not self and m.is_loaded]
        candidates_to_unload.sort(key=lambda m: m.last_used_time)

        if not candidates_to_unload:
            ASCIIColors.error("OOM error, but no other models are available to unload.")
            raise Exception("OOM error, but no other models are available to unload.")

        for victim in candidates_to_unload:
            ASCIIColors.info(f"Unloading '{victim.config['model_name']}' (last used: {time.ctime(victim.last_used_time)}) to free VRAM.")
            victim._unload_pipeline()
            
            try:
                ASCIIColors.info(f"Retrying to load '{model_name}'...")
                self._execute_load_pipeline(task, model_path, torch_dtype)
                ASCIIColors.green(f"Successfully loaded '{model_name}' after freeing VRAM.")
                return
            except Exception as retry_e:
                is_oom_retry = "out of memory" in str(retry_e).lower()
                if not is_oom_retry:
                    raise retry_e 
        
        ASCIIColors.error(f"Could not load '{model_name}' even after unloading all other models.")
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
            "unload_inactive_model_after": 0
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
        
        if ("Qwen" in self.config.get("model_name", "") or "FLUX" in self.config.get("model_name", "")) and self.config["device"] == "cuda":
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                self.config["torch_dtype_str"] = "bfloat16"
                ASCIIColors.info("Special model detected on compatible hardware. Forcing dtype to bfloat16 for stability.")
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

# --- Pydantic Models for API ---
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
    """
    Takes a request object (Pydantic model or dict) and returns a 'safe' dictionary
    for logging, with long base64 strings replaced by placeholders.
    """
    import copy

    try:
        if hasattr(request_data, 'model_dump'):
            data = request_data.model_dump()
        elif isinstance(request_data, dict):
            data = copy.deepcopy(request_data)
        else:
            return {"error": "Unsupported data type for sanitization"}

        # Sanitize the main list of images
        if 'images_b64' in data and isinstance(data['images_b64'], list):
            count = len(data['images_b64'])
            data['images_b64'] = f"[<{count} base64 image(s) truncated>]"

        # Sanitize a potential mask in the 'params' dictionary
        if 'params' in data and isinstance(data.get('params'), dict):
            if 'mask_image' in data['params'] and isinstance(data['params']['mask_image'], str):
                original_len = len(data['params']['mask_image'])
                data['params']['mask_image'] = f"[<base64 mask truncated, len={original_len}>]"
        
        return data
    except Exception:
        return {"error": "Failed to sanitize request data."}

# --- API Endpoints ---
@router.post("/generate_image")
async def generate_image(request: T2IRequest):
    manager = None
    temp_config = None
    try:
        # Determine which model manager to use for this specific request
        if "model_name" in request.params and request.params["model_name"]:
            temp_config = state.config.copy()
            temp_config["model_name"] = request.params.pop("model_name") # Remove from params to avoid being passed to pipeline
            manager = state.registry.get_manager(temp_config, state.models_path)
            ASCIIColors.info(f"Using per-request model: {temp_config['model_name']}")
        else:
            manager = state.get_active_manager()
            ASCIIColors.info(f"Using session-configured model: {manager.config.get('model_name')}")

        # Start with the manager's config (base settings)
        pipeline_args = manager.config.copy()
        # Override with per-request parameters
        pipeline_args.update(request.params)

        # Add prompts and ensure types for specific args
        pipeline_args["prompt"] = request.prompt
        pipeline_args["negative_prompt"] = request.negative_prompt
        width = pipeline_args.get("width", 1024)
        height = pipeline_args.get("height", 1024)
        num_inference_steps = pipeline_args.get("num_inference_steps", 25)
        seed = pipeline_args.get("seed", -1)
        guidance_scale = pipeline_args.get("guidance_scale", 7.0)
        pipeline_args["width"] = int(width if width else 1024)
        pipeline_args["height"] = int(height if height else 1024)
        pipeline_args["num_inference_steps"] = int(num_inference_steps if num_inference_steps else 25)
        pipeline_args["guidance_scale"] = float(guidance_scale if guidance_scale else 7.0)

        seed = int(seed if seed is not None else -1)
        pipeline_args["generator"] = None
        if seed != -1:
            pipeline_args["generator"] = torch.Generator(device=manager.config["device"]).manual_seed(seed)
        
        model_name = manager.config.get("model_name", "")
        task = "text2image"
        
        if "Qwen-Image-Edit" in model_name:
            rng_seed = seed if seed != -1 else None
            rng = np.random.default_rng(seed=rng_seed)
            random_pixels = rng.integers(0, 256, size=(pipeline_args["height"], pipeline_args["width"], 3), dtype=np.uint8)
            placeholder_image = Image.fromarray(random_pixels, 'RGB')
            pipeline_args["image"] = placeholder_image
            pipeline_args["strength"] = float(pipeline_args.get("strength", 1.0))
            task = "image2image" 
        
        log_args = {k: v for k, v in pipeline_args.items() if k not in ['generator', 'image']}
        if pipeline_args.get("generator"): log_args['generator'] = f"<torch.Generator(seed={seed})>"
        if pipeline_args.get("image"): log_args['image'] = "<PIL Image object>"
        
        ASCIIColors.cyan("--- Generating Image with Settings ---")
        try:
            print(json.dumps(log_args, indent=2, default=str))
        except Exception as e:
            ASCIIColors.warning(f"Could not print all settings: {e}")
            print(log_args)
        ASCIIColors.cyan("------------------------------------")
        
        future = Future()
        manager.queue.put((future, task, pipeline_args))
        result_bytes = future.result()
        return Response(content=result_bytes, media_type="image/png")
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

        # Start with manager's config, then override with request params
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
        if pipeline_args.get("generator"): log_args['generator'] = f"<torch.Generator(seed={seed})>"
        if 'image' in pipeline_args: log_args['image'] = f"[<{len(pil_images)} PIL Image(s)>]"
        if 'mask_image' in pipeline_args and pipeline_args['mask_image']: log_args['mask_image'] = "<PIL Mask Image>"

        ASCIIColors.cyan("--- Editing Image with Settings ---")
        try:
            print(json.dumps(log_args, indent=2, default=str))
        except Exception as e:
            ASCIIColors.warning(f"Could not print all settings: {e}")
            print(log_args)
        ASCIIColors.cyan("---------------------------------")
        
        future = Future(); manager.queue.put((future, task, pipeline_args))
        return Response(content=future.result(), media_type="image/png")
    except Exception as e:
        sanitized_payload = get_sanitized_request_for_logging(request)
        ASCIIColors.error(f"Exception in /edit_image. Sanitized Payload: {json.dumps(sanitized_payload, indent=2)}")
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

    # 1) Pull Hugging Face model into a folder
    if payload.hf_id:
        model_id = payload.hf_id.strip()
        folder_name = payload.local_name or model_id.replace("/", "__")
        dest_dir = state.models_path / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            ASCIIColors.cyan(f"Pulling HF model '{model_id}' into {dest_dir}")
            # Reuse config options for HF access
            load_params: Dict[str, Any] = {}
            if state.config.get("hf_cache_path"):
                load_params["cache_dir"] = str(state.config["hf_cache_path"])
            if state.config.get("hf_token"):
                load_params["token"] = state.config["hf_token"]
            # For a pull, we want to actually download:
            load_params["local_files_only"] = False

            # Use DiffusionPipeline (or AutoPipelineForText2Image) to download, then save_pretrained
            pipe = DiffusionPipeline.from_pretrained(model_id, **load_params)
            pipe.save_pretrained(dest_dir)
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ASCIIColors.green(f"Model '{model_id}' pulled to {dest_dir}")
            return {"status": "ok", "model_name": folder_name}
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to pull HF model: {e}")

    # 2) Pull raw .safetensors from URL
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

    # 1) Single-file models: top-level *.safetensors only
    for f in state.models_path.glob("*.safetensors"):
        local_models.add(f.name)

    if state.extra_models_path and state.extra_models_path.exists():
        for f in state.extra_models_path.glob("*.safetensors"):
            local_models.add(f.name)

    # 2) Folder-based HF/diffusers models: treat folder name as the model
    def add_folder_models(base: Path):
        if not base or not base.exists():
            return
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            has_index = (entry / "model_index.json").exists()
            has_safetensors = any(entry.glob("*.safetensors"))
            if has_index or has_safetensors:
                local_models.add(entry.name)

    add_folder_models(state.models_path)
    add_folder_models(state.extra_models_path)

    return sorted(list(local_models))

@router.get("/list_models")
def list_models_endpoint():
    models = []

    # 1) Add local models (single-file and folder)
    local_files = list_local_models_endpoint()
    for model_name in local_files:
        display_name = model_name
        description = "(Local) Folder model" if not model_name.endswith(".safetensors") else "(Local) Local safetensors file"
        models.append({
            "model_name": model_name,
            "display_name": display_name,
            "description": description,
            "owned_by": "local_user"
        })

    # 2) Add public Hugging Face models organized by category
    for category, hf_models in HF_PUBLIC_MODELS.items():
        for model_info in hf_models:
            models.append({
                "model_name": model_info["model_name"],
                "display_name": model_info["display_name"],
                "description": f"({category}) {model_info['desc']}",
                "owned_by": "huggingface"
            })

    # 3) Add gated Hugging Face models if HF token is available
    if state.config.get("hf_token"):
        for category, gated_models in HF_GATED_MODELS.items():
            for model_info in gated_models:
                models.append({
                    "model_name": model_info["model_name"],
                    "display_name": model_info["display_name"],
                    "description": f"({category}) {model_info['desc']}",
                    "owned_by": "huggingface"
                })

    # 4) Add Civitai models
    for key, info in CIVITAI_MODELS.items():
        models.append({
            "model_name": key,
            "display_name": info["display_name"],
            "description": f"(Civitai) {info['description']}",
            "owned_by": info["owned_by"]
        })

    return models


@router.get("/list_available_models")
def list_available_models_endpoint():
    discoverable = [m['model_name'] for m in list_models_endpoint()]
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
        trace_exception(e)
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
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=9630, help="Port to bind to.")
    parser.add_argument("--models-path", type=str, required=True, help="Path to the models directory.")
    parser.add_argument("--extra-models-path", type=str, default=None, help="Path to an extra models directory.")
    args = parser.parse_args()

    MODELS_PATH = Path(args.models_path)
    EXTRA_MODELS_PATH = Path(args.extra_models_path) if args.extra_models_path else None
    state = ServerState(MODELS_PATH, EXTRA_MODELS_PATH)
    
    ASCIIColors.cyan(f"--- Diffusers TTI Server ---")
    ASCIIColors.green(f"Starting server on http://{args.host}:{args.port}")
    ASCIIColors.green(f"Serving models from: {MODELS_PATH.resolve()}")
    if EXTRA_MODELS_PATH:
        ASCIIColors.green(f"Serving extra models from: {EXTRA_MODELS_PATH.resolve()}")
    if not DIFFUSERS_AVAILABLE:
        ASCIIColors.error("Diffusers or its dependencies are not installed correctly in the server's environment!")
    else:
        ASCIIColors.info(f"Detected device: {state.config['device']}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
