# lollms_client/tti_bindings/diffusers/__init__.py
import os
import importlib
from io import BytesIO
from typing import Optional, List, Dict, Any, Union
from pathlib import Path


try:
    import pipmaster as pm
    import platform # For OS detection for torch index

    # Determine initial device preference to guide torch installation
    preferred_torch_device_for_install = "cpu" # Default assumption
    
    # Tentatively set preference based on OS, assuming user might want GPU if available
    if platform.system() == "Linux" or platform.system() == "Windows":
        # On Linux/Windows, CUDA is the primary GPU acceleration for PyTorch.
        # We will try to install a CUDA version of PyTorch.
        preferred_torch_device_for_install = "cuda"
    elif platform.system() == "Darwin":
        # On macOS, MPS is the acceleration. Standard torch install usually handles this.
        preferred_torch_device_for_install = "mps" # or keep cpu if mps detection is later

    torch_pkgs = ["torch", "torchaudio", "torchvision", "xformers"]
    diffusers_core_pkgs = ["diffusers", "Pillow", "transformers", "safetensors"]

    torch_index_url = None
    if preferred_torch_device_for_install == "cuda":
        # Specify a common CUDA version index. Pip should resolve the correct torch version.
        # As of late 2023/early 2024, cu118 or cu121 are common. Let's use cu126.
        # Users with different CUDA setups might need to pre-install torch manually.
        torch_index_url = "https://download.pytorch.org/whl/cu126"
        ASCIIColors.info(f"Attempting to ensure PyTorch with CUDA support (target index: {torch_index_url})")
        # Install torch and torchaudio first from the specific index
        pm.ensure_packages(torch_pkgs, index_url=torch_index_url)
        # Then install audiocraft and other dependencies; pip should use the already installed torch
        pm.ensure_packages(diffusers_core_pkgs)
    else:
        # For CPU, MPS, or if no specific CUDA preference was determined for install
        ASCIIColors.info("Ensuring PyTorch, AudioCraft, and dependencies using default PyPI index.")
        pm.ensure_packages(torch_pkgs + diffusers_core_pkgs)

    import whisper
    import torch
    _whisper_installed = True
except Exception as e:
    _whisper_installation_error = str(e)
    whisper = None
    torch = None


try:
    import torch
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline
    from diffusers.utils import load_image # Potentially for future img2img etc.
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    torch = None
    AutoPipelineForText2Image = None
    DiffusionPipeline = None
    Image = None
    load_image = None
    DIFFUSERS_AVAILABLE = False
    # Detailed error will be raised in __init__ if user tries to use it

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors
import json # For potential JSONDecodeError and settings

# Defines the binding name for the manager
BindingName = "DiffusersTTIBinding_Impl"

# Helper for torch.dtype string conversion
TORCH_DTYPE_MAP_STR_TO_OBJ = {
    "float16": torch.float16 if torch else "float16", # Keep string if torch not loaded
    "bfloat16": torch.bfloat16 if torch else "bfloat16",
    "float32": torch.float32 if torch else "float32",
    "auto": "auto"
}
TORCH_DTYPE_MAP_OBJ_TO_STR = {v: k for k, v in TORCH_DTYPE_MAP_STR_TO_OBJ.items()}
if torch: # Add None mapping if torch is loaded
    TORCH_DTYPE_MAP_OBJ_TO_STR[None] = "None"


# Common Schedulers mapping (User-friendly name to Class name)
SCHEDULER_MAPPING = {
    "default": None,  # Use model's default
    "ddim": "DDIMScheduler",
    "ddpm": "DDPMScheduler",
    "deis_multistep": "DEISMultistepScheduler",
    "dpm_multistep": "DPMSolverMultistepScheduler", # Alias
    "dpm_multistep_karras": "DPMSolverMultistepScheduler", # Configured with use_karras_sigmas=True
    "dpm_single": "DPMSolverSinglestepScheduler",
    "dpm_adaptive": "DP soluzioniPlusPlusScheduler", # DPM++ 2M Karras in A1111
    "dpm++_2m": "DPMSolverMultistepScheduler", 
    "dpm++_2m_karras": "DPMSolverMultistepScheduler", # Configured with use_karras_sigmas=True
    "dpm++_2s_ancestral": "DPMSolverAncestralDiscreteScheduler",
    "dpm++_2s_ancestral_karras": "DPMSolverAncestralDiscreteScheduler", # Configured with use_karras_sigmas=True
    "dpm++_sde": "DPMSolverSDEScheduler",
    "dpm++_sde_karras": "DPMSolverSDEScheduler", # Configured with use_karras_sigmas=True
    "euler_ancestral_discrete": "EulerAncestralDiscreteScheduler",
    "euler_discrete": "EulerDiscreteScheduler",
    "heun_discrete": "HeunDiscreteScheduler",
    "heun_karras": "HeunDiscreteScheduler", # Configured with use_karras_sigmas=True
    "lms_discrete": "LMSDiscreteScheduler",
    "lms_karras": "LMSDiscreteScheduler", # Configured with use_karras_sigmas=True
    "pndm": "PNDMScheduler",
    "unipc_multistep": "UniPCMultistepScheduler",
}
SCHEDULER_USES_KARRAS_SIGMAS = [
    "dpm_multistep_karras", "dpm++_2m_karras", "dpm++_2s_ancestral_karras",
    "dpm++_sde_karras", "heun_karras", "lms_karras"
]


class DiffusersTTIBinding_Impl(LollmsTTIBinding):
    """
    Concrete implementation of LollmsTTIBinding for Hugging Face Diffusers library.
    Allows running various text-to-image models locally.
    """
    DEFAULT_CONFIG = {
        "model_id_or_path": "stabilityai/stable-diffusion-2-1-base",
        "device": "auto",  # "auto", "cuda", "mps", "cpu"
        "torch_dtype_str": "auto",  # "auto", "float16", "bfloat16", "float32"
        "use_safetensors": True,
        "scheduler_name": "default",
        "safety_checker_on": True, # Note: Diffusers default is ON
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "default_width": 768, # Default for SD 2.1 base
        "default_height": 768, # Default for SD 2.1 base
        "seed": -1,  # -1 for random on each call
        "enable_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "enable_xformers": False, # Explicit opt-in for xformers
        "hf_variant": None,  # e.g., "fp16"
        "hf_token": None,
        "local_files_only": False,
    }


    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 lollms_paths: Optional[Dict[str, Union[str, Path]]] = None,
                 **kwargs # Catches other potential parameters like 'service_key' or 'client_id'
                 ):
        """
        Initialize the Diffusers TTI binding.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary for the binding.
                                               Overrides DEFAULT_CONFIG.
            lollms_paths (Optional[Dict[str, Union[str, Path]]]): Dictionary of LOLLMS paths.
                                                                  Used for model/cache directories.
            **kwargs: Catches other parameters (e.g. service_key).
        """
        super().__init__(binding_name="diffusers")
        
        if not DIFFUSERS_AVAILABLE:
            ASCIIColors.error("Diffusers library or its dependencies (torch, Pillow, transformers) are not installed or failed to import.")
            ASCIIColors.info("Attempting to install/verify packages...")
            pm.ensure_packages(["torch", "diffusers", "Pillow", "transformers", "safetensors"])
            try:
                import torch as _torch
                from diffusers import AutoPipelineForText2Image as _AutoPipelineForText2Image
                from diffusers import DiffusionPipeline as _DiffusionPipeline
                from PIL import Image as _Image
                globals()['torch'] = _torch
                globals()['AutoPipelineForText2Image'] = _AutoPipelineForText2Image
                globals()['DiffusionPipeline'] = _DiffusionPipeline
                globals()['Image'] = _Image
                
                # Re-populate torch dtype maps if torch was just loaded
                global TORCH_DTYPE_MAP_STR_TO_OBJ, TORCH_DTYPE_MAP_OBJ_TO_STR
                TORCH_DTYPE_MAP_STR_TO_OBJ = {
                    "float16": _torch.float16, "bfloat16": _torch.bfloat16,
                    "float32": _torch.float32, "auto": "auto"
                }
                TORCH_DTYPE_MAP_OBJ_TO_STR = {v: k for k, v in TORCH_DTYPE_MAP_STR_TO_OBJ.items()}
                TORCH_DTYPE_MAP_OBJ_TO_STR[None] = "None"
                ASCIIColors.green("Dependencies seem to be available now.")
            except ImportError as e:
                trace_exception(e)
                raise ImportError(
                    "Diffusers binding dependencies are still not met after trying to ensure them. "
                    "Please install torch, diffusers, Pillow, and transformers manually. "
                    f"Error: {e}"
                ) from e

        self.config = {**self.DEFAULT_CONFIG, **(config or {}), **kwargs}
        self.lollms_paths = {k: Path(v) for k, v in lollms_paths.items()} if lollms_paths else {}
        
        self.pipeline: Optional[DiffusionPipeline] = None
        self.current_model_id_or_path = None # To track if model needs reload

        # Resolve auto settings for device and dtype
        if self.config["device"].lower() == "auto":
            if torch.cuda.is_available(): self.config["device"] = "cuda"
            elif torch.backends.mps.is_available(): self.config["device"] = "mps"
            else: self.config["device"] = "cpu"
        
        if self.config["torch_dtype_str"].lower() == "auto":
            if self.config["device"] == "cpu": self.config["torch_dtype_str"] = "float32" # CPU usually float32
            else: self.config["torch_dtype_str"] = "float16" # Common default for GPU

        self.torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower(), torch.float32)
        if self.torch_dtype == "auto": # Should have been resolved above
             self.torch_dtype = torch.float16 if self.config["device"] != "cpu" else torch.float32
             self.config["torch_dtype_str"] = TORCH_DTYPE_MAP_OBJ_TO_STR.get(self.torch_dtype, "float32")


        # For potential lollms client specific features
        self.client_id = kwargs.get("service_key", kwargs.get("client_id", "diffusers_client_user"))

        self.load_model()


    def _resolve_model_path(self, model_id_or_path: str) -> str:
        """Resolves a model name/path against lollms_paths if not absolute."""
        if os.path.isabs(model_id_or_path):
            return model_id_or_path
        
        # Check personal_models_path/diffusers_models/<name>
        if self.lollms_paths.get('personal_models_path'):
            personal_diffusers_path = self.lollms_paths['personal_models_path'] / "diffusers_models" / model_id_or_path
            if personal_diffusers_path.exists() and personal_diffusers_path.is_dir():
                ASCIIColors.info(f"Found local model in personal_models_path: {personal_diffusers_path}")
                return str(personal_diffusers_path)

        # Check models_zoo_path/diffusers_models/<name> (if different from personal)
        if self.lollms_paths.get('models_zoo_path') and \
           self.lollms_paths.get('models_zoo_path') != self.lollms_paths.get('personal_models_path'):
            zoo_diffusers_path = self.lollms_paths['models_zoo_path'] / "diffusers_models" / model_id_or_path
            if zoo_diffusers_path.exists() and zoo_diffusers_path.is_dir():
                ASCIIColors.info(f"Found local model in models_zoo_path: {zoo_diffusers_path}")
                return str(zoo_diffusers_path)
        
        ASCIIColors.info(f"Assuming '{model_id_or_path}' is a Hugging Face Hub ID or already fully qualified.")
        return model_id_or_path

    def load_model(self):
        """Loads the Diffusers pipeline based on current configuration."""
        ASCIIColors.info("Loading Diffusers model...")
        if self.pipeline is not None:
            self.unload_model() # Ensure old model is cleared

        try:
            model_path = self._resolve_model_path(self.config["model_id_or_path"])
            self.current_model_id_or_path = model_path # Store what's actually loaded

            load_args = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": self.config["use_safetensors"],
                "token": self.config["hf_token"],
                "local_files_only": self.config["local_files_only"],
            }
            if self.config["hf_variant"]:
                load_args["variant"] = self.config["hf_variant"]
            
            if not self.config["safety_checker_on"]:
                load_args["safety_checker"] = None
            
            if self.lollms_paths.get("shared_cache_path"):
                load_args["cache_dir"] = str(self.lollms_paths["shared_cache_path"] / "huggingface_diffusers")


            # Use AutoPipelineForText2Image for flexibility
            pipeline_class_to_load = AutoPipelineForText2Image
            custom_pipeline_class_name = self.config.get("pipeline_class_name")

            if custom_pipeline_class_name:
                try:
                    diffusers_module = importlib.import_module("diffusers")
                    pipeline_class_to_load = getattr(diffusers_module, custom_pipeline_class_name)
                    ASCIIColors.info(f"Using specified pipeline class: {custom_pipeline_class_name}")
                except (ImportError, AttributeError) as e:
                    ASCIIColors.warning(f"Could not load custom pipeline class {custom_pipeline_class_name}: {e}. Falling back to AutoPipelineForText2Image.")
                    pipeline_class_to_load = AutoPipelineForText2Image
            
            self.pipeline = pipeline_class_to_load.from_pretrained(model_path, **load_args)
            
            # Scheduler
            self._set_scheduler()

            self.pipeline.to(self.config["device"])

            if self.config["enable_xformers"]:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    ASCIIColors.info("xFormers memory efficient attention enabled.")
                except Exception as e:
                    ASCIIColors.warning(f"Could not enable xFormers: {e}. Proceeding without it.")
            
            if self.config["enable_cpu_offload"] and self.config["device"] != "cpu":
                self.pipeline.enable_model_cpu_offload()
                ASCIIColors.info("Model CPU offload enabled.")
            elif self.config["enable_sequential_cpu_offload"] and self.config["device"] != "cpu":
                self.pipeline.enable_sequential_cpu_offload() # More aggressive
                ASCIIColors.info("Sequential CPU offload enabled.")


            ASCIIColors.green(f"Diffusers model '{model_path}' loaded successfully on device '{self.config['device']}' with dtype '{self.config['torch_dtype_str']}'.")

        except Exception as e:
            trace_exception(e)
            self.pipeline = None
            raise RuntimeError(f"Failed to load Diffusers model '{self.config['model_id_or_path']}': {e}") from e

    def _set_scheduler(self):
        if not self.pipeline: return
        
        scheduler_name_key = self.config["scheduler_name"].lower()
        if scheduler_name_key == "default":
            ASCIIColors.info(f"Using model's default scheduler: {self.pipeline.scheduler.__class__.__name__}")
            return

        scheduler_class_name = SCHEDULER_MAPPING.get(scheduler_name_key)
        if scheduler_class_name:
            try:
                scheduler_module = importlib.import_module("diffusers.schedulers")
                SchedulerClass = getattr(scheduler_module, scheduler_class_name)
                
                scheduler_config = self.pipeline.scheduler.config
                if scheduler_name_key in SCHEDULER_USES_KARRAS_SIGMAS:
                    scheduler_config["use_karras_sigmas"] = True
                else: # Ensure it's False if not a karras variant for this scheduler
                    if "use_karras_sigmas" in scheduler_config:
                         scheduler_config["use_karras_sigmas"] = False


                self.pipeline.scheduler = SchedulerClass.from_config(scheduler_config)
                ASCIIColors.info(f"Switched scheduler to {scheduler_name_key} ({scheduler_class_name}).")
            except Exception as e:
                trace_exception(e)
                ASCIIColors.warning(f"Could not switch scheduler to {scheduler_name_key}: {e}. Using current default.")
        else:
            ASCIIColors.warning(f"Unknown scheduler name: {self.config['scheduler_name']}. Using model default.")


    def unload_model(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            ASCIIColors.info("Diffusers pipeline unloaded.")
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_image(self,
                       prompt: str,
                       negative_prompt: Optional[str] = "",
                       width: Optional[int] = None, # Uses default from config if None
                       height: Optional[int] = None, # Uses default from config if None
                       **kwargs) -> bytes:
        """
        Generates image data using the Diffusers pipeline.

        Args:
            prompt (str): The positive text prompt.
            negative_prompt (Optional[str]): The negative prompt.
            width (int): Image width. Overrides default.
            height (int): Image height. Overrides default.
            **kwargs: Additional parameters for the pipeline:
                      - num_inference_steps (int)
                      - guidance_scale (float)
                      - seed (int)
                      - eta (float, for DDIM)
                      - num_images_per_prompt (int, though this binding returns one)
                      - clip_skip (int, if supported by pipeline - advanced)
        Returns:
            bytes: The generated image data (PNG format).
        Raises:
            Exception: If the request fails or image generation fails.
        """
        if not self.pipeline:
            raise RuntimeError("Diffusers pipeline is not loaded. Cannot generate image.")

        # Use call-specific or configured defaults
        _width = width if width is not None else self.config["default_width"]
        _height = height if height is not None else self.config["default_height"]
        _num_inference_steps = kwargs.get("num_inference_steps", self.config["num_inference_steps"])
        _guidance_scale = kwargs.get("guidance_scale", self.config["guidance_scale"])
        _seed = kwargs.get("seed", self.config["seed"])

        generator = None
        if _seed != -1: # -1 means random seed
            generator = torch.Generator(device=self.config["device"]).manual_seed(_seed)
        
        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt else None,
            "width": _width,
            "height": _height,
            "num_inference_steps": _num_inference_steps,
            "guidance_scale": _guidance_scale,
            "generator": generator,
            "num_images_per_prompt": kwargs.get("num_images_per_prompt", 1)
        }
        if "eta" in kwargs: pipeline_args["eta"] = kwargs["eta"]
        if "clip_skip" in kwargs and hasattr(self.pipeline, "clip_skip"): # Handle clip_skip if supported
            pipeline_args["clip_skip"] = kwargs["clip_skip"]


        ASCIIColors.info(f"Generating image with prompt: '{prompt[:100]}...'")
        ASCIIColors.debug(f"Pipeline args: {pipeline_args}")

        try:
            with torch.no_grad(): # Important for inference
                 pipeline_output = self.pipeline(**pipeline_args)
            
            pil_image: Image.Image = pipeline_output.images[0]

            # Convert PIL Image to bytes (PNG)
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()
            
            ASCIIColors.green("Image generated successfully.")
            return img_bytes

        except Exception as e:
            trace_exception(e)
            raise Exception(f"Diffusers image generation failed: {e}") from e

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        """
        Lists the currently loaded model as the available service.
        Future: Could scan local model directories or list known HF models.
        """
        if self.pipeline and self.current_model_id_or_path:
            return [{
                "name": os.path.basename(self.current_model_id_or_path),
                "caption": f"Diffusers: {os.path.basename(self.current_model_id_or_path)}",
                "help": (f"Currently loaded model. Path/ID: {self.current_model_id_or_path}. "
                         f"Device: {self.config['device']}. DType: {self.config['torch_dtype_str']}. "
                         f"Scheduler: {self.pipeline.scheduler.__class__.__name__}.")
            }]
        return [{"name": "diffusers_unloaded", "caption": "No Diffusers model loaded", "help": "Configure a model in settings."}]

    def get_settings(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieves the current configurable settings for the Diffusers binding.
        """
        # Actual device and dtype after auto-resolution
        resolved_device = self.config['device']
        resolved_dtype_str = self.config['torch_dtype_str']
        
        # For display, show the original 'auto' if it was set that way, plus the resolved value
        display_device = self.config['device'] if self.config['device'].lower() != 'auto' else f"auto ({resolved_device})"
        display_dtype = self.config['torch_dtype_str'] if self.config['torch_dtype_str'].lower() != 'auto' else f"auto ({resolved_dtype_str})"

        settings = [
            {"name": "model_id_or_path", "type": "str", "value": self.config["model_id_or_path"], "description": "Hugging Face model ID or local path to Diffusers model directory."},
            {"name": "device", "type": "str", "value": self.config["device"], "description": f"Device for inference. Current resolved: {resolved_device}", "options": ["auto", "cuda", "mps", "cpu"]},
            {"name": "torch_dtype_str", "type": "str", "value": self.config["torch_dtype_str"], "description": f"Torch dtype for model. Current resolved: {resolved_dtype_str}", "options": ["auto", "float16", "bfloat16", "float32"]},
            {"name": "hf_variant", "type": "str", "value": self.config["hf_variant"], "description": "Model variant (e.g., 'fp16', 'bf16'). Optional."},
            {"name": "use_safetensors", "type": "bool", "value": self.config["use_safetensors"], "description": "Prefer loading models from .safetensors files."},
            {"name": "scheduler_name", "type": "str", "value": self.config["scheduler_name"], "description": "Scheduler to use for diffusion.", "options": list(SCHEDULER_MAPPING.keys())},
            {"name": "safety_checker_on", "type": "bool", "value": self.config["safety_checker_on"], "description": "Enable the safety checker (if model has one)."},
            {"name": "enable_cpu_offload", "type": "bool", "value": self.config["enable_cpu_offload"], "description": "Enable model CPU offload (saves VRAM, slower)."},
            {"name": "enable_sequential_cpu_offload", "type": "bool", "value": self.config["enable_sequential_cpu_offload"], "description": "Enable sequential CPU offload (more VRAM savings, much slower)."},
            {"name": "enable_xformers", "type": "bool", "value": self.config["enable_xformers"], "description": "Enable xFormers memory efficient attention (if available)."},
            {"name": "default_width", "type": "int", "value": self.config["default_width"], "description": "Default width for generated images."},
            {"name": "default_height", "type": "int", "value": self.config["default_height"], "description": "Default height for generated images."},
            {"name": "num_inference_steps", "type": "int", "value": self.config["num_inference_steps"], "description": "Default number of inference steps."},
            {"name": "guidance_scale", "type": "float", "value": self.config["guidance_scale"], "description": "Default guidance scale (CFG)."},
            {"name": "seed", "type": "int", "value": self.config["seed"], "description": "Default seed for generation (-1 for random)."},
            {"name": "hf_token", "type": "str", "value": self.config["hf_token"], "description": "Hugging Face API token (for private/gated models). Set to 'None' or empty if not needed. Store securely.", "is_secret": True},
            {"name": "local_files_only", "type": "bool", "value": self.config["local_files_only"], "description": "Only use local files, do not try to download."},
        ]
        return settings

    def set_settings(self, settings: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> bool:
        """
        Applies new settings to the Diffusers binding. Some settings may trigger a model reload.
        """
        if isinstance(settings, list): # Convert from ConfigTemplate list format
            parsed_settings = {item["name"]: item["value"] for item in settings if "name" in item and "value" in item}
        elif isinstance(settings, dict):
            parsed_settings = settings
        else:
            ASCIIColors.error("Invalid settings format. Expected a dictionary or list of dictionaries.")
            return False

        old_config = self.config.copy()
        needs_reload = False

        for key, value in parsed_settings.items():
            if key in self.config:
                if self.config[key] != value:
                    self.config[key] = value
                    ASCIIColors.info(f"Setting '{key}' changed to: {value}")
                    if key in ["model_id_or_path", "device", "torch_dtype_str", 
                               "use_safetensors", "safety_checker_on", "hf_variant",
                               "enable_cpu_offload", "enable_sequential_cpu_offload", "enable_xformers",
                               "hf_token", "local_files_only"]:
                        needs_reload = True
                    elif key == "scheduler_name" and self.pipeline: # Scheduler can be changed on loaded pipeline
                        self._set_scheduler() # Attempt to apply immediately
            else:
                ASCIIColors.warning(f"Unknown setting '{key}' ignored.")

        if needs_reload:
            ASCIIColors.info("Reloading model due to settings changes...")
            try:
                # Resolve auto device/dtype again if they were part of the change
                if "device" in parsed_settings and self.config["device"].lower() == "auto":
                    if torch.cuda.is_available(): self.config["device"] = "cuda"
                    elif torch.backends.mps.is_available(): self.config["device"] = "mps"
                    else: self.config["device"] = "cpu"
                
                if "torch_dtype_str" in parsed_settings and self.config["torch_dtype_str"].lower() == "auto":
                    self.config["torch_dtype_str"] = "float16" if self.config["device"] != "cpu" else "float32"
                
                # Update torch_dtype object from string
                self.torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower(), torch.float32)
                if self.torch_dtype == "auto": # Should be resolved by now
                    self.torch_dtype = torch.float16 if self.config["device"] != "cpu" else torch.float32
                    self.config["torch_dtype_str"] = TORCH_DTYPE_MAP_OBJ_TO_STR.get(self.torch_dtype, "float32")


                self.load_model()
                ASCIIColors.green("Model reloaded successfully with new settings.")
            except Exception as e:
                trace_exception(e)
                ASCIIColors.error(f"Failed to reload model with new settings: {e}. Reverting critical settings.")
                # Revert critical settings and try to reload with old config
                self.config = old_config 
                self.torch_dtype = TORCH_DTYPE_MAP_STR_TO_OBJ.get(self.config["torch_dtype_str"].lower(), torch.float32)
                try:
                    self.load_model()
                    ASCIIColors.info("Reverted to previous model configuration.")
                except Exception as e_revert:
                    trace_exception(e_revert)
                    ASCIIColors.error(f"Failed to revert to previous model configuration: {e_revert}. Binding may be unstable.")
                return False
        return True

    def __del__(self):
        self.unload_model()

# Example Usage (for testing within this file)
if __name__ == '__main__':
    ASCIIColors.magenta("--- Diffusers TTI Binding Test ---")
    
    if not DIFFUSERS_AVAILABLE:
        ASCIIColors.error("Diffusers or its dependencies are not available. Cannot run test.")
        # Attempt to guide user for installation
        print("Please ensure PyTorch, Diffusers, Pillow, and Transformers are installed.")
        print("For PyTorch with CUDA: visit https://pytorch.org/get-started/locally/")
        print("Then: pip install diffusers Pillow transformers safetensors")
        exit(1)

    # --- Configuration ---
    # Small, fast model for testing. Replace with a full model for real use.
    # "CompVis/stable-diffusion-v1-4" is ~5GB
    # "google/ddpm-cat-256" is smaller, but a DDPM, not Stable Diffusion.
    # Using a tiny SD model if one exists, or a small variant.
    # For a quick test, let's try a small LCM LoRA with SD1.5 if possible or just a base model.
    # Note: "runwayml/stable-diffusion-v1-5" is a good standard test model.
    # For a *very* quick CI-like test, one might use a dummy model or a very small one.
    # Let's use a smaller SD variant if available, otherwise default to 2.1-base.
    test_model_id = "runwayml/stable-diffusion-v1-5" # ~4GB download. Use a smaller one if you have it locally.
    # test_model_id = "hf-internal-testing/tiny-stable-diffusion-pipe" # Very small, for testing structure
    
    # Create dummy lollms_paths
    temp_paths_dir = Path(__file__).parent / "temp_lollms_paths_diffusers"
    temp_paths_dir.mkdir(parents=True, exist_ok=True)
    mock_lollms_paths = {
        "personal_models_path": temp_paths_dir / "personal_models",
        "models_zoo_path": temp_paths_dir / "models_zoo",
        "shared_cache_path": temp_paths_dir / "shared_cache", # For Hugging Face cache
    }
    for p in mock_lollms_paths.values(): Path(p).mkdir(parents=True, exist_ok=True)
    (Path(mock_lollms_paths["personal_models_path"]) / "diffusers_models").mkdir(exist_ok=True)


    binding_config = {
        "model_id_or_path": test_model_id,
        "device": "auto", # Let it auto-detect
        "torch_dtype_str": "auto",
        "num_inference_steps": 10, # Faster for testing
        "default_width": 256, # Smaller for faster testing
        "default_height": 256,
        "safety_checker_on": False, # Often disabled for local use flexibility
        "hf_variant": "fp16" if test_model_id == "runwayml/stable-diffusion-v1-5" else None, # SD 1.5 has fp16 variant
    }

    try:
        ASCIIColors.cyan("\n1. Initializing DiffusersTTIBinding_Impl...")
        binding = DiffusersTTIBinding_Impl(config=binding_config, lollms_paths=mock_lollms_paths)
        ASCIIColors.green("Initialization successful.")
        ASCIIColors.info(f"Loaded model: {binding.current_model_id_or_path}")
        ASCIIColors.info(f"Device: {binding.config['device']}, DType: {binding.config['torch_dtype_str']}")
        ASCIIColors.info(f"Scheduler: {binding.pipeline.scheduler.__class__.__name__ if binding.pipeline else 'N/A'}")


        ASCIIColors.cyan("\n2. Listing services...")
        services = binding.list_services()
        ASCIIColors.info(json.dumps(services, indent=2))
        assert services and services[0]["name"] == os.path.basename(binding.current_model_id_or_path)

        ASCIIColors.cyan("\n3. Getting settings...")
        settings_list = binding.get_settings()
        ASCIIColors.info(json.dumps(settings_list, indent=2, default=str)) # default=str for Path objects if any
        # Find model_id_or_path in settings
        found_model_setting = any(s['name'] == 'model_id_or_path' and s['value'] == test_model_id for s in settings_list)
        assert found_model_setting, "Model ID not found or incorrect in get_settings"


        ASCIIColors.cyan("\n4. Generating an image...")
        test_prompt = "A vibrant cat astronaut exploring a neon galaxy"
        test_negative_prompt = "blurry, low quality, text, watermark"
        
        # Use smaller dimensions for test if default are large
        gen_width = min(binding.config["default_width"], 256)
        gen_height = min(binding.config["default_height"], 256)

        image_bytes = binding.generate_image(
            prompt=test_prompt,
            negative_prompt=test_negative_prompt,
            width=gen_width, height=gen_height,
            num_inference_steps=8 # Even fewer for speed
        )
        assert image_bytes and isinstance(image_bytes, bytes)
        ASCIIColors.green(f"Image generated successfully (size: {len(image_bytes)} bytes).")
        # Save the image for verification
        test_image_path = Path(__file__).parent / "test_diffusers_image.png"
        with open(test_image_path, "wb") as f:
            f.write(image_bytes)
        ASCIIColors.info(f"Test image saved to: {test_image_path.resolve()}")


        ASCIIColors.cyan("\n5. Setting new settings (changing scheduler and guidance_scale)...")
        new_settings_dict = {
            "scheduler_name": "ddim", # Change scheduler
            "guidance_scale": 5.0,   # Change guidance scale
            "num_inference_steps": 12 # Change inference steps
        }
        binding.set_settings(new_settings_dict)
        assert binding.config["scheduler_name"] == "ddim"
        assert binding.config["guidance_scale"] == 5.0
        assert binding.config["num_inference_steps"] == 12
        ASCIIColors.info(f"New scheduler (intended): ddim, Actual: {binding.pipeline.scheduler.__class__.__name__}")
        ASCIIColors.info(f"New guidance_scale: {binding.config['guidance_scale']}")
        
        ASCIIColors.cyan("\n6. Generating another image with new settings...")
        image_bytes_2 = binding.generate_image(
            prompt="A serene landscape with a crystal river",
            width=gen_width, height=gen_height
        )
        assert image_bytes_2 and isinstance(image_bytes_2, bytes)
        ASCIIColors.green(f"Second image generated successfully (size: {len(image_bytes_2)} bytes).")
        test_image_path_2 = Path(__file__).parent / "test_diffusers_image_2.png"
        with open(test_image_path_2, "wb") as f:
            f.write(image_bytes_2)
        ASCIIColors.info(f"Second test image saved to: {test_image_path_2.resolve()}")

        # Test model reload by changing a critical parameter (e.g. safety_checker_on)
        # This requires a different model or a config that can be easily toggled.
        # For now, assume reload on critical param change works if no error is thrown.
        ASCIIColors.cyan("\n7. Testing settings change requiring model reload (safety_checker_on)...")
        current_safety_on = binding.config["safety_checker_on"]
        binding.set_settings({"safety_checker_on": not current_safety_on})
        assert binding.config["safety_checker_on"] == (not current_safety_on)
        ASCIIColors.green("Model reload due to safety_checker_on change seems successful.")


    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Diffusers binding test failed: {e}")
    finally:
        ASCIIColors.cyan("\nCleaning up...")
        if 'binding' in locals() and binding:
            binding.unload_model()
        
        # Clean up temp_lollms_paths
        import shutil
        if temp_paths_dir.exists():
            try:
                shutil.rmtree(temp_paths_dir)
                ASCIIColors.info(f"Cleaned up temporary directory: {temp_paths_dir}")
            except Exception as e_clean:
                ASCIIColors.warning(f"Could not fully clean up {temp_paths_dir}: {e_clean}")
        if 'test_image_path' in locals() and test_image_path.exists():
             # os.remove(test_image_path) # Keep for manual check
             pass
        if 'test_image_path_2' in locals() and test_image_path_2.exists():
             # os.remove(test_image_path_2) # Keep for manual check
             pass
        ASCIIColors.magenta("--- Diffusers TTI Binding Test Finished ---")