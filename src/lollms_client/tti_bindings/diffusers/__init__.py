# lollms_client/tti_bindings/diffusers/__init__.py
import os
import importlib
from io import BytesIO
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import base64
import pipmaster as pm
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
import atexit
import multiprocessing
import httpx
import uvicorn
from fastapi import FastAPI, Response, APIRouter
from pydantic import BaseModel

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors

# --- Dependency Management ---
pm.ensure_packages(["torch","torchvision"],index_url="https://download.pytorch.org/whl/cu126")
pm.ensure_packages(["pillow","transformers","safetensors","requests","tqdm", "fastapi", "uvicorn", "httpx"])

try:
    import torch
    from diffusers import (
        AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting,
        DiffusionPipeline, StableDiffusionPipeline, QwenImageEditPipeline, QwenImageEditPlusPipeline
    )
    from diffusers.utils import load_image
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    class DiffusionPipeline: pass
    class Image: pass


BindingName = "DiffusersTTIBinding_Impl"
# =================================================================================================
# == Internal Classes for the Server Process (The "Engine")
# =================================================================================================

CIVITAI_MODELS = {
    "realistic-vision-v6": { "display_name": "Realistic Vision V6.0", "url": "https://civitai.com/api/download/models/501240?type=Model&format=SafeTensor&size=pruned&fp=fp16", "filename": "realisticVisionV60_v60B1.safetensors", "description": "Photorealistic SD1.5 checkpoint.", "owned_by": "civitai" },
    "absolute-reality": { "display_name": "Absolute Reality", "url": "https://civitai.com/api/download/models/132760?type=Model&format=SafeTensor&size=pruned&fp=fp16", "filename": "absolutereality_v181.safetensors", "description": "General realistic SD1.5.", "owned_by": "civitai" },
    "dreamshaper-8": { "display_name": "DreamShaper 8", "url": "https://civitai.com/api/download/models/128713", "filename": "dreamshaper_8.safetensors", "description": "Versatile SD1.5 style model.", "owned_by": "civitai" },
    "juggernaut-xl": { "display_name": "Juggernaut XL", "url": "https://civitai.com/api/download/models/133005", "filename": "juggernautXL_version6Rundiffusion.safetensors", "description": "Artistic SDXL.", "owned_by": "civitai" },
    
   "lyriel-v1.6": {
        "display_name": "Lyriel v1.6",
        "url": "https://civitai.com/api/download/models/72396?type=Model&format=SafeTensor&size=full&fp=fp16",
        "filename": "lyriel_v16.safetensors",
        "description": "Fantasy/stylized SD1.5.",
        "owned_by": "civitai"
    },
    "ui_icons": {
        "display_name": "UI Icons",
        "url": "https://civitai.com/api/download/models/367044?type=Model&format=SafeTensor&size=full&fp=fp16",
        "filename": "uiIcons_v10.safetensors",
        "description": "A model for generating UI icons.",
        "owned_by": "civitai"
    },
    "meinamix": {
        "display_name": "MeinaMix",
        "url": "https://civitai.com/api/download/models/948574?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "meinamix_meinaV11.safetensors",
        "description": "Anime/illustration SD1.5.",
        "owned_by": "civitai"
    },
    "rpg-v5": {
        "display_name": "RPG v5",
        "url": "https://civitai.com/api/download/models/124626?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "rpg_v5.safetensors",
        "description": "RPG assets SD1.5.",
        "owned_by": "civitai"
    },
    "pixel-art-xl": {
        "display_name": "Pixel Art XL",
        "url": "https://civitai.com/api/download/models/135931?type=Model&format=SafeTensor",
        "filename": "pixelartxl_v11.safetensors",
        "description": "Pixel art SDXL.",
        "owned_by": "civitai"
    },
    "lowpoly-world": {
        "display_name": "Lowpoly World",
        "url": "https://civitai.com/api/download/models/146502?type=Model&format=SafeTensor",
        "filename": "LowpolySDXL.safetensors",
        "description": "Lowpoly style SD1.5.",
        "owned_by": "civitai"
    },
    "toonyou": {
        "display_name": "ToonYou",
        "url": "https://civitai.com/api/download/models/125771?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "toonyou_beta6.safetensors",
        "description": "Cartoon/Disney SD1.5.",
        "owned_by": "civitai"
    },
    "papercut": {
        "display_name": "Papercut",
        "url": "https://civitai.com/api/download/models/133503?type=Model&format=SafeTensor",
        "filename": "papercut.safetensors",
        "description": "Paper cutout SD1.5.",
        "owned_by": "civitai"
    },
    "fantassifiedIcons": {
        "display_name": "Fantassified Icons",
        "url": "https://civitai.com/api/download/models/67584?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "fantassifiedIcons_fantassifiedIconsV20.safetensors",
        "description": "Flat, modern Icons.",
        "owned_by": "civitai"
    },
    "game_icon_institute": {
        "display_name": "Game icon institute",
        "url": "https://civitai.com/api/download/models/158776?type=Model&format=SafeTensor&size=full&fp=fp16",
        "filename": "gameIconInstituteV10_v10.safetensors",
        "description": "Flat, modern game Icons.",
        "owned_by": "civitai"
    },
    "M4RV3LS_DUNGEONS": {
        "display_name": "M4RV3LS & DUNGEONS",
        "url": "https://civitai.com/api/download/models/139417?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "filename": "M4RV3LSDUNGEONSNEWV40COMICS_mD40.safetensors",
        "description": "comics.",
        "owned_by": "civitai"
    },
}
TORCH_DTYPE_MAP_STR_TO_OBJ = { "float16": getattr(torch, 'float16', 'float16'), "bfloat16": getattr(torch, 'bfloat16', 'bfloat16'), "float32": getattr(torch, 'float32', 'float32'), "auto": "auto" }
SCHEDULER_MAPPING = { "default": None, "ddim": "DDIMScheduler", "ddpm": "DDPMScheduler", "euler_discrete": "EulerDiscreteScheduler", "lms_discrete": "LMSDiscreteScheduler" }
SCHEDULER_USES_KARRAS_SIGMAS = []

class ModelManager:
    # This is the full ModelManager from our previous working version
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

    def acquire(self):
        with self.lock: self.ref_count += 1
        return self

    def release(self):
        with self.lock: self.ref_count -= 1
        return self.ref_count
        
    def stop(self):
        self._stop_event.set()
        self.queue.put(None)
        self.worker_thread.join(timeout=5)

    def _resolve_model_path(self, model_name: str) -> Union[str, Path]:
        if model_name in CIVITAI_MODELS:
            pass
        return model_name

    def _execute_load_pipeline(self, task: str, model_path: Union[str, Path], torch_dtype: Any):
        common_args = {"torch_dtype": torch_dtype}
        if "Qwen-Image-Edit-2509" in str(model_path): self.pipeline = QwenImageEditPlusPipeline.from_pretrained(model_path, **common_args)
        elif "Qwen-Image-Edit" in str(model_path): self.pipeline = QwenImageEditPipeline.from_pretrained(model_path, **common_args)
        elif "Qwen/Qwen-Image" in str(model_path): self.pipeline = DiffusionPipeline.from_pretrained(model_path, **common_args)
        else: self.pipeline = AutoPipelineForText2Image.from_pretrained(model_path, **common_args)
        self.pipeline.to(self.config["device"])
        self.is_loaded = True

    def _load_pipeline_for_task(self, task: str):
        if self.pipeline and self.current_task == task: return
        if self.pipeline: self._unload_pipeline()
        model_name = self.config.get("model_name", "")
        model_path = self._resolve_model_path(model_name)
        torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower())
        try:
            self._execute_load_pipeline(task, model_path, torch_dtype)
        except Exception as e:
            if "out of memory" in str(e).lower():
                ASCIIColors.warning("OOM detected. Trying to free memory...")
                self._execute_load_pipeline(task, model_path, torch_dtype)
            else:
                raise e

    def _unload_pipeline(self):
        if self.pipeline:
            del self.pipeline; self.pipeline = None; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            self.is_loaded = False
            ASCIIColors.info(f"Model unloaded and VRAM cleared.")

    def _generation_worker(self):
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=1)
                if job is None: break
                future, task, pipeline_args = job
                output = None
                try:
                    with self.lock:
                        self.last_used_time = time.time()
                        if not self.is_loaded or self.current_task != task: self._load_pipeline_for_task(task)
                    with torch.no_grad(): output = self.pipeline(**pipeline_args)
                    pil = output.images[0]; buf = BytesIO(); pil.save(buf, format="PNG"); future.set_result(buf.getvalue())
                except Exception as e: future.set_exception(e)
                finally:
                    self.queue.task_done()
                    if output: del output
                    if self.config.get("force_reload_between_generations"): self._unload_pipeline()
                    else: gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except queue.Empty: continue

class PipelineRegistry:
    _instance = None; _lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls); cls._instance._managers = {}; cls._instance._registry_lock = threading.Lock()
        return cls._instance
    def get_manager(self, config: Dict[str, Any], models_path: Path) -> ModelManager:
        key = hashlib.sha256(str(sorted(config.items())).encode()).hexdigest()
        with self._registry_lock:
            if key not in self._managers: self._managers[key] = ModelManager(config.copy(), models_path, self)
            return self._managers[key].acquire()
    def release_manager(self, config: Dict[str, Any]): pass

class _DiffusersServerBindingImpl(LollmsTTIBinding):
    QWEN_ASPECT_RATIOS = { "1:1": (1328, 1328), "16:9": (1664, 928), "9:16": (928, 1664), "4:3": (1472, 1104), "3:4": (1104, 1472) }
    QWEN_POSITIVE_MAGIC = ", Ultra HD, 4K, cinematic composition."
    DEFAULT_CONFIG = { "model_name": "", "device": "auto", "torch_dtype_str": "auto", "force_reload_between_generations": False, "server_port": 28374 }
    HF_DEFAULT_MODELS = [
        {"family": "SDXL", "model_name": "stabilityai/stable-diffusion-xl-base-1.0"},
        {"family": "Qwen", "model_name": "Qwen/Qwen-Image"},
        {"family": "Editors", "model_name": "Qwen/Qwen-Image-Edit"},
        {"family": "Editors", "model_name": "Qwen/Qwen-Image-Edit-2509"}
    ]
    
    def __init__(self, **kwargs):
        super().__init__(binding_name="diffusers_server_impl")
        if not DIFFUSERS_AVAILABLE: raise RuntimeError("Diffusers library not found.")
        self.config = self.DEFAULT_CONFIG.copy(); self.config.update(kwargs)
        self.model_name = self.config.get("model_name", "")
        self.models_path = Path(self.config.get("models_path", str(Path(__file__).parent / "models")))
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.registry = PipelineRegistry()
        self._resolve_device_and_dtype()
        self.manager: Optional[ModelManager] = None
        if self.model_name: self._acquire_manager()

    def _snap_to_qwen_aspect_ratio(self, width, height):
        target_ratio = width / height; best_match = (width, height); min_diff = float('inf')
        for dims in self.QWEN_ASPECT_RATIOS.values():
            ratio = dims[0] / dims[1]; diff = abs(target_ratio - ratio)
            if diff < min_diff: min_diff = diff; best_match = dims
        return best_match

    def _acquire_manager(self):
        if self.manager: self.registry.release_manager(self.manager.config)
        self.manager = self.registry.get_manager(self.config, self.models_path)

    def _resolve_device_and_dtype(self):
        if self.config["device"].lower() == "auto": self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if "Qwen" in self.config.get("model_name", "") and self.config["device"] == "cuda":
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                self.config["torch_dtype_str"] = "bfloat16"; return
        if self.config["torch_dtype_str"].lower() == "auto": self.config["torch_dtype_str"] = "float16" if self.config["device"] != "cpu" else "float32"

    def _prepare_seed(self, kwargs: Dict[str, Any]) -> Optional[torch.Generator]:
        seed = kwargs.pop("seed", self.config.get("seed",-1)); 
        if seed == -1: return None
        return torch.Generator(device=self.config["device"]).manual_seed(seed)

    def generate_image(self, prompt: str, negative_prompt: str = "", width: int|None = None, height: int|None = None, **kwargs) -> bytes:
        kwargs = kwargs.copy()
        if not self.model_name: raise RuntimeError("No model name configured.")
        if not self.manager: self._acquire_manager()
        w = width or self.config.get("width", 512); h = height or self.config.get("height", 512)
        is_qwen = "Qwen" in self.model_name
        if is_qwen: w, h = self._snap_to_qwen_aspect_ratio(w, h)
        if "Qwen-Image-Edit" in self.model_name:
            noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            return self.edit_image(images=[Image.fromarray(noise, 'RGB')], prompt=prompt, negative_prompt=negative_prompt, width=w, height=h, **kwargs)
        if "Qwen/Qwen-Image" in self.model_name:
            pipeline_args = { "prompt": prompt + self.QWEN_POSITIVE_MAGIC, "negative_prompt": negative_prompt or " ", "width": w, "height": h, "num_inference_steps": kwargs.get("num_inference_steps", 50), "true_cfg_scale": kwargs.get("guidance_scale", 4.0), "generator": self._prepare_seed(kwargs) }
        else:
            pipeline_args = { "prompt": prompt, "negative_prompt": negative_prompt or "", "width": w, "height": h, "generator": self._prepare_seed(kwargs) }; pipeline_args.update(kwargs)
        future = Future(); self.manager.queue.put((future, "text2image", pipeline_args)); return future.result()

    def edit_image(self, images, prompt, negative_prompt="", mask=None, width=None, height=None, **kwargs) -> bytes:
        kwargs = kwargs.copy()
        if not self.model_name: raise RuntimeError("No model name configured.")
        if not self.manager: self._acquire_manager()
        w = width or self.config.get("width", 512); h = height or self.config.get("height", 512)
        is_qwen = "Qwen" in self.model_name; is_qwen_2509 = "Qwen-Image-Edit-2509" in self.model_name
        if is_qwen: w, h = self._snap_to_qwen_aspect_ratio(w, h)
        base_args = { "prompt": prompt, "negative_prompt": negative_prompt, "width": w, "height": h, "num_inference_steps": kwargs.get("num_inference_steps", 50), "generator": self._prepare_seed(kwargs) }
        if is_qwen:
            base_args["true_cfg_scale"] = kwargs.get("guidance_scale", 4.0)
            if is_qwen_2509: base_args["guidance_scale"] = kwargs.get("guidance_scale_plus", 1.0)
        else: base_args["guidance_scale"] = kwargs.get("guidance_scale", 7.5)
        if mask: pipeline_args = {**base_args, "image": images[0], "mask_image": mask}; task="inpainting"
        else: pipeline_args = {**base_args, "image": images[0] if len(images)==1 else images}; task="image2image"
        future = Future(); self.manager.queue.put((future, task, pipeline_args)); return future.result()

    def list_models(self) -> list:
        return self.HF_DEFAULT_MODELS + list(CIVITAI_MODELS.keys())

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        return [{"name": m["model_name"], "caption": m["model_name"]} for m in self.HF_DEFAULT_MODELS]

    def set_settings(self, settings: Dict[str, Any]) -> bool:
        self.config.update(settings)
        if self.manager: self.manager.config.update(settings)
        return True

    def get_settings(self) -> Dict[str, Any]: return self.config

# =================================================================================================
# == Server Process Logic ==
# =================================================================================================
_server_binding_instance: Optional[_DiffusersServerBindingImpl] = None

class ServerGenerateRequest(BaseModel): prompt: str; negative_prompt: str; width: Optional[int]; height: Optional[int]; kwargs: dict
class ServerEditRequest(BaseModel): images: List[str]; prompt: str; negative_prompt: str; mask: Optional[str]; width: Optional[int]; height: Optional[int]; kwargs: dict
class ServerSettingsRequest(BaseModel): settings: dict

def _run_server_process(port: int, **model_kwargs):
    global _server_binding_instance; _server_binding_instance = _DiffusersServerBindingImpl(**model_kwargs)
    app = FastAPI()
    @app.post("/generate")
    async def generate(req: ServerGenerateRequest):
        try: return Response(content=_server_binding_instance.generate_image(req.prompt, req.negative_prompt, req.width, req.height, **req.kwargs), media_type="image/png")
        except Exception as e: return Response(content=json.dumps({"error": str(e)}), status_code=500)
    @app.post("/edit")
    async def edit(req: ServerEditRequest):
        images = [Image.open(BytesIO(base64.b64decode(i))) for i in req.images]
        mask = Image.open(BytesIO(base64.b64decode(req.mask))) if req.mask else None
        try: return Response(content=_server_binding_instance.edit_image(images, req.prompt, req.negative_prompt, mask, req.width, req.height, **req.kwargs), media_type="image/png")
        except Exception as e: return Response(content=json.dumps({"error": str(e)}), status_code=500)
    @app.post("/set_settings")
    async def set_settings(req: ServerSettingsRequest): return {"success": _server_binding_instance.set_settings(req.settings)}
    @app.get("/get_settings")
    async def get_settings(): return _server_binding_instance.get_settings()
    @app.get("/list_models")
    async def list_models(): return _server_binding_instance.list_models()
    @app.get("/list_services")
    async def list_services(): return _server_binding_instance.list_services()
    @app.get("/health")
    async def health(): return {"status": "ok"}
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

# =================================================================================================
# == Public Facing Proxy Class ==
# =================================================================================================
_server_process = None
_initialization_lock = multiprocessing.Lock()
def _cleanup_server():
    global _server_process
    if _server_process and _server_process.is_alive(): _server_process.terminate(); _server_process.join()
atexit.register(_cleanup_server)

class DiffusersTTIBinding_Impl(LollmsTTIBinding):
    DEFAULT_CONFIG = _DiffusersServerBindingImpl.DEFAULT_CONFIG

    def __init__(self, **kwargs):
        super().__init__(binding_name="diffusers")
        self.config = kwargs
        self.server_port = self.config.get("server_port", 28374)
        self.server_url = f"http://127.0.0.1:{self.server_port}"
        with _initialization_lock: self._check_and_start_server()
        self.client = httpx.Client(timeout=600.0)

    def _check_and_start_server(self):
        global _server_process
        try:
            if httpx.get(f"{self.server_url}/health", timeout=1.0).status_code == 200: return
        except httpx.ConnectError: pass
        ASCIIColors.green(f"WORKER {os.getpid()}: Is leader. Starting dedicated model server...")
        ctx = multiprocessing.get_context('spawn')
        _server_process = ctx.Process(target=_run_server_process, args=(self.server_port,), kwargs=self.config, daemon=True)
        _server_process.start()
        self._wait_for_server()

    def _wait_for_server(self):
        for _ in range(60):
            try:
                if httpx.get(self.server_url + "/health", timeout=1.0).status_code == 200:
                    ASCIIColors.green("Server is up!"); return
            except httpx.ConnectError: time.sleep(1)
        raise RuntimeError("Model server failed to start in time.")

    def generate_image(self, prompt: str, negative_prompt: str = "", width: int|None = None, height: int|None = None, **kwargs) -> bytes:
        payload = {"prompt": prompt, "negative_prompt": negative_prompt, "width": width, "height": height, "kwargs": kwargs}
        response = self.client.post(f"{self.server_url}/generate", json=payload)
        if response.status_code != 200: raise Exception(f"Error from model server: {response.text}")
        return response.content

    def edit_image(self, images: Union[str, List[str], Image.Image, List[Image.Image]], prompt: str, negative_prompt: Optional[str] = "", mask: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None, **kwargs) -> bytes:
        def to_base64(img):
            if isinstance(img, str) and Path(img).exists():
                with open(img, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')
            elif isinstance(img, Image.Image):
                buffered = BytesIO(); img.save(buffered, format="PNG"); return base64.b64encode(buffered.getvalue()).decode('utf-8')
            elif isinstance(img, str): return img
            return None
        images_b64 = [to_base64(i) for i in (images if isinstance(images, list) else [images])]
        mask_b64 = to_base64(mask) if mask else None
        payload = {"images": images_b64, "prompt": prompt, "negative_prompt": negative_prompt, "mask": mask_b64, "width": width, "height": height, "kwargs": kwargs}
        response = self.client.post(f"{self.server_url}/edit", json=payload)
        if response.status_code != 200: raise Exception(f"Error from model server: {response.text}")
        return response.content
        
    def set_settings(self, settings: Dict[str, Any]) -> bool:
        response = self.client.post(f"{self.server_url}/set_settings", json={"settings": settings})
        return response.status_code == 200 and response.json().get("success", False)

    def get_settings(self) -> Dict[str, Any]:
        response = self.client.get(f"{self.server_url}/get_settings")
        return response.json() if response.status_code == 200 else {}
        
    def list_models(self) -> list:
        response = self.client.get(f"{self.server_url}/list_models")
        return response.json() if response.status_code == 200 else []

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        response = self.client.get(f"{self.server_url}/list_services")
        return response.json() if response.status_code == 200 else []