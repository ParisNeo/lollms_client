# lollms_client/lollms_tti_binding.py (updated)
from abc import abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from ascii_colors import trace_exception, ASCIIColors
import yaml
import io
import base64
import requests
import importlib
from lollms_client.lollms_base_binding import LollmsBaseBinding

try:
    from PIL import Image, PngImagePlugin
except ImportError:
    ASCIIColors.warning("Pillow not found. Image processing (metadata/watermarking) will be limited. Install with: pip install Pillow")


class TTIGenerationResult(dict):
    """
    Unified result container for TTI/Omni generation.
    Backward compatible: behaves like a dict, but also exposes
    convenience accessors. Legacy code expecting raw bytes should
    call `.first_image_bytes()`.

    Keys:
        images: List[bytes]            -> list of raw image bytes (post-processed)
        text:   Optional[str]          -> any text returned alongside images (omni models)
        raw:    Optional[Any]          -> raw provider response (debug/advanced use)
        metadata: Optional[dict]       -> generation metadata (seed, steps, model, etc.)
    """
    def __init__(self, images=None, text=None, raw=None, metadata=None):
        super().__init__(
            images=images or [],
            text=text,
            raw=raw,
            metadata=metadata or {}
        )

    @property
    def images(self) -> List[bytes]:
        return self.get("images", [])

    @property
    def text(self) -> Optional[str]:
        return self.get("text")

    def first_image_bytes(self) -> bytes:
        imgs = self.images
        if not imgs:
            raise ValueError("No image was returned by the binding.")
        return imgs[0]


class LollmsTTIBinding(LollmsBaseBinding):
    """Abstract base class for all LOLLMS Text-to-Image / Omni bindings."""

    def __init__(self,
                 binding_name: str = "unknown",
                 debug: Optional[bool] = False,
                 supports_omni: bool = False,
                 supports_text_output: bool = False,
                 **kwargs):
        """
        Initialize the LollmsTTIBinding base class.

        New Args (Omni support):
        - supports_omni (bool): True if the underlying model/service can return
          text alongside images in a single call (e.g. vLLM-Omni chat-completions
          style models that narrate + generate, or edit + explain).
        - supports_text_output (bool): True if generate() may populate `text`
          in the TTIGenerationResult even outside of a full omni pipeline.

        Watermarking Settings (set via kwargs/config):
        - watermark_path, watermark_size_x/y, watermark_pos_x/y, author, system
        """
        super().__init__(binding_name=binding_name, debug=debug, **kwargs)
        self.watermark_path = kwargs.get("watermark_path", None)
        self.watermark_size = (
            kwargs.get("watermark_size_x", 100),
            kwargs.get("watermark_size_y", 100)
        )
        self.watermark_pos = (
            kwargs.get("watermark_pos_x", 10),
            kwargs.get("watermark_pos_y", 10)
        )
        self.author = kwargs.get("author", "ParisNeo")
        self.system = kwargs.get("system", "LoLLMS")

        # --- Omni capability flags ---
        self.supports_omni = supports_omni
        self.supports_text_output = supports_text_output

    # ------------------------------------------------------------------
    # Image post-processing (unchanged, reused for every image produced)
    # ------------------------------------------------------------------
    def process_image(self, image_bytes: bytes, **kwargs) -> bytes:
        try:
            img = Image.open(io.BytesIO(image_bytes))

            wm_path = kwargs.get("watermark_path", self.watermark_path)
            wm_size = (
                kwargs.get("watermark_size_x", self.watermark_size[0]),
                kwargs.get("watermark_size_y", self.watermark_size[1])
            )
            wm_pos = (
                kwargs.get("watermark_pos_x", self.watermark_pos[0]),
                kwargs.get("watermark_pos_y", self.watermark_pos[1])
            )
            author = kwargs.get("author", self.author)
            system = kwargs.get("system", self.system)

            final_metadata = {
                "Software": "LoLLMS",
                "Binding": self.binding_name,
                "Model": self.config.get("model_name", "unknown"),
                "Author": author,
                "System": system,
                "Description": f"Built using LoLLMS ({self.binding_name}) with model {self.config.get('model_name', 'unknown')}",
                "AI-Generated": "True"
            }
            if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
                final_metadata.update(kwargs["metadata"])

            if wm_path:
                try:
                    if str(wm_path).startswith("http"):
                        wm_res = requests.get(wm_path)
                        wm_img = Image.open(io.BytesIO(wm_res.content)).convert("RGBA")
                    elif ";base64," in str(wm_path):
                        wm_data = base64.b64decode(str(wm_path).split(",")[1])
                        wm_img = Image.open(io.BytesIO(wm_data)).convert("RGBA")
                    else:
                        wm_img = Image.open(wm_path).convert("RGBA")

                    wm_img = wm_img.resize(wm_size, Image.Resampling.LANCZOS)
                    if img.mode != 'RGBA':
                        img = img.convert("RGBA")
                    img.paste(wm_img, wm_pos, wm_img)
                    img = img.convert("RGB")
                except Exception as wm_err:
                    ASCIIColors.error(f"Failed to apply visible watermark: {wm_err}")

            output = io.BytesIO()
            png_info = PngImagePlugin.PngInfo()
            for k, v in final_metadata.items():
                png_info.add_text(k, str(v))

            img.save(output, format="PNG", pnginfo=png_info)
            return output.getvalue()

        except Exception as e:
            trace_exception(e)
            ASCIIColors.error("Image post-processing failed. Returning raw bytes.")
            return image_bytes

    # ------------------------------------------------------------------
    # NEW unified entrypoint — omni-capable
    # ------------------------------------------------------------------
    def generate(self,
                 prompt: str,
                 negative_prompt: Optional[str] = "",
                 width: int = 512,
                 height: int = 512,
                 images: Optional[Union[str, List[str]]] = None,
                 mask: Optional[str] = None,
                 n: int = 1,
                 modalities: Optional[List[str]] = None,
                 **kwargs) -> TTIGenerationResult:
        """
        Unified generation entrypoint supporting classic TTI models AND
        modern omni models that return text + image(s) together.

        Default implementation provides retro-compatibility: it dispatches
        to `generate_image` (or `edit_image` if `images` is provided) and
        wraps the raw bytes into a TTIGenerationResult with no text.

        Omni-capable bindings (supports_omni=True) SHOULD override this
        method directly instead of generate_image/edit_image, since they
        need to return text alongside images from a single request.

        Args:
            modalities: list like ["image"], ["text","image"] — hints to
                        the underlying provider about desired output types.
                        Non-omni bindings ignore this safely.
        """
        if images:
            raw = self.edit_image(
                images=images, prompt=prompt, negative_prompt=negative_prompt,
                mask=mask, width=width, height=height, **kwargs
            )
        else:
            raw = self.generate_image(
                prompt=prompt, negative_prompt=negative_prompt,
                width=width, height=height, **kwargs
            )
        return TTIGenerationResult(images=[raw], text=None, raw=raw)

    # ------------------------------------------------------------------
    # Legacy abstract methods (kept for retrocompatibility)
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_image(self,
                       prompt: str,
                       negative_prompt: Optional[str] = "",
                       width: int = 512,
                       height: int = 512,
                       **kwargs) -> bytes:
        """Generates image data from the provided text prompt. Returns raw bytes only."""
        pass

    @abstractmethod
    def edit_image(self,
                   images: Union[str, List[str]],
                   prompt: str,
                   negative_prompt: Optional[str] = "",
                   mask: Optional[str] = None,
                   width: Optional[int] = None,
                   height: Optional[int] = None,
                   **kwargs) -> bytes:
        """Edits an image or a set of images based on the provided prompts. Returns raw bytes only."""
        pass

    @abstractmethod
    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        pass

    @abstractmethod
    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def list_models(self) -> list:
        pass

    def get_zoo(self) -> List[Dict[str, Any]]:
        return []

    def download_from_zoo(self, index: int, progress_callback: Callable[[dict], None] = None) -> dict:
        return {"status": False, "message": "Not implemented"}

    @abstractmethod
    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        pass

class LollmsTTIBindingManager:
    """Manages TTI binding discovery and instantiation."""

    def __init__(self, tti_bindings_dir: Union[str, Path] = Path(__file__).parent.parent / "tti_bindings"):
        self.tti_bindings_dir = Path(tti_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        binding_dir = self.tti_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.tti_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName) # Assumes BindingName is defined
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load TTI binding {binding_name}: {str(e)}")

    def create_binding(self,
                      binding_name: str,
                      **kwargs) -> Optional[LollmsTTIBinding]:
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)

        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            try:
                return binding_class(**kwargs)
            except Exception as e:
                trace_exception(e)
                print(f"Failed to instantiate TTI binding {binding_name}: {str(e)}")
                return None
        return None

    def _get_fallback_description(binding_name: str) -> Dict:
        return {
            "binding_name": binding_name,
            "title": binding_name.replace("_", " ").title(),
            "author": "Unknown",
            "creation_date": "N/A",
            "last_update_date": "N/A",
            "description": f"A binding for {binding_name}. No description.yaml file was found, so common parameters are shown as a fallback.",
            "input_parameters": [
                {
                    "name": "model_name",
                    "type": "str",
                    "description": "The model name, ID, or filename to be used.",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "host_address",
                    "type": "str",
                    "description": "The host address of the service (for API-based bindings).",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "models_path",
                    "type": "str",
                    "description": "The path to the models directory (for local bindings).",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "service_key",
                    "type": "str",
                    "description": "The API key or service key for authentication (if applicable).",
                    "mandatory": False,
                    "default": ""
                }
            ]
        }

    @staticmethod
    def get_bindings_list(llm_bindings_dir: Union[str, Path]) -> List[Dict]:
        bindings_dir = Path(llm_bindings_dir)
        if not bindings_dir.is_dir():
            return []

        bindings_list = []
        for binding_folder in bindings_dir.iterdir():
            if binding_folder.is_dir() and (binding_folder / "__init__.py").exists():
                binding_name = binding_folder.name
                description_file = binding_folder / "description.yaml"
                
                binding_info = {}
                if description_file.exists():
                    try:
                        with open(description_file, 'r', encoding='utf-8') as f:
                            binding_info = yaml.safe_load(f)
                        binding_info['binding_name'] = binding_name
                    except Exception as e:
                        print(f"Error loading description.yaml for {binding_name}: {e}")
                        binding_info = LollmsTTIBindingManager._get_fallback_description(binding_name)
                else:
                    binding_info = LollmsTTIBindingManager._get_fallback_description(binding_name)
                
                bindings_list.append(binding_info)

        return sorted(bindings_list, key=lambda b: b.get('title', b['binding_name']))

    def get_available_bindings(self) -> list[str]:
        return [binding_dir.name for binding_dir in self.tti_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]


def get_available_bindings(tti_bindings_dir: Union[str, Path] = None) -> List[Dict]:
    if tti_bindings_dir is None:
        tti_bindings_dir = Path(__file__).parent / "tti_bindings"
    return LollmsTTIBindingManager.get_bindings_list(tti_bindings_dir)

def list_binding_models(tti_binding_name: str, tti_binding_config: Optional[Dict[str, any]]|None = None, tti_bindings_dir: str|Path = Path(__file__).parent / "tti_bindings") -> List[Dict]:
    """
    Lists all available models for a specific binding.
    """
    binding = LollmsTTIBindingManager(tti_bindings_dir).create_binding(
        binding_name=tti_binding_name,
        **{
            k: v
            for k, v in (tti_binding_config or {}).items()
            if k != "binding_name"
        }
    )

    return binding.list_models() if binding else []
