# lollms_client/lollms_ttv_binding.py
from abc import abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from ascii_colors import trace_exception
from lollms_client.lollms_base_binding import LollmsBaseBinding

# A media input is either a filesystem path or the raw bytes of the file.
MediaInput = Union[str, Path, bytes]


class LollmsTTVBinding(LollmsBaseBinding):
    """Abstract base class for all LOLLMS Text-to-Video bindings."""

    def __init__(self,
                 binding_name:str="unknown",
                 debug:Optional[bool] = False,
                 **kwargs):
        """
        Initialize the LollmsTTVBinding base class.
        """
        super().__init__(binding_name=binding_name, debug=debug, **kwargs)

    @abstractmethod
    def generate_video(self, prompt: str, **kwargs) -> bytes:
        """
        Generates video data from the provided text prompt.
        """
        pass

    # ── Optional capabilities ─────────────────────────────────────────────────
    # Bindings opt in by overriding supports_*() AND the matching generate_*().
    # Callers must check supports_*() first; the default generate_*() raises.

    def supports_lipsync(self) -> bool:
        """
        True if this binding can generate a talking video from a portrait image and an
        audio track (mouth movements synchronized to the audio). Default: False.
        Implementations may return False at runtime when the current backend/hardware
        tier cannot do it.
        """
        return False

    def generate_lipsync_video(self, image: MediaInput, audio: MediaInput, prompt: str = "", **kwargs) -> bytes:
        """
        Generates a lip-synced video: `image` (path or bytes) is animated so that the
        subject speaks `audio` (path or bytes). `prompt` optionally guides the scene.
        Returns the encoded video (mp4) as bytes.

        Optional kwargs a binding may honor: negative_prompt, seed, fps, width, height,
        num_frames, num_inference_steps, guidance_scale, image_ext / audio_ext (when
        passing raw bytes). Unknown kwargs must be ignored, not rejected.

        Unimplemented by default: bindings without this ability raise NotImplementedError.
        """
        raise NotImplementedError(f"{self.binding_name} does not support lip-sync video generation")

    def get_capabilities(self) -> Dict[str, bool]:
        """Capability flags for UIs/agents. Extend as new optional abilities are added."""
        return {"text_to_video": True, "lipsync": bool(self.supports_lipsync())}

    @abstractmethod
    def list_models(self, **kwargs) -> List[str]:
        """
        Lists the available TTV models or services supported by the binding.
        """
        pass

    def get_zoo(self) -> List[Dict[str, Any]]:
        """
        Returns a list of models available for download.
        each entry is a dict with:
        name, description, size, type, link
        """
        return []

    def download_from_zoo(self, index: int, progress_callback: Callable[[dict], None] = None) -> dict:
        """
        Downloads a model from the zoo using its index.
        """
        return {"status": False, "message": "Not implemented"}

    def install_model(self, model_name: str, **kwargs) -> dict:
        """
        Installs a video model from a remote repository into the local models directory.
        """
        if hasattr(self, "pull_model") and callable(self.pull_model):
            return self.pull_model(model_name, **kwargs)
        return {"status": False, "message": f"install_model not implemented for {self.binding_name}"}

    def pull_model(self, model_name: str, **kwargs) -> dict:
        """
        Downloads a video model from a remote repository into the local models directory.
        """
        if hasattr(self, "install_model") and callable(self.install_model):
            return self.install_model(model_name, **kwargs)
        return {"status": False, "message": f"pull_model not implemented for {self.binding_name}"}

class LollmsTTVBindingManager:
    """Manages TTV binding discovery and instantiation."""

    def __init__(self, ttv_bindings_dir: Union[str, Path] = Path(__file__).parent / "ttv_bindings"):
        self.ttv_bindings_dir = Path(ttv_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        """Dynamically load a specific TTV binding implementation."""
        binding_dir = self.ttv_bindings_dir / binding_name
        init_file = binding_dir / "__init__.py"
        if binding_dir.is_dir() and init_file.exists():
            try:
                import importlib.util
                # Use file-based loading to support directory names with dots (e.g. sd.cpp)
                module_name = f"lollms_ttv_binding_{binding_name.replace('.', '_')}"
                spec = importlib.util.spec_from_file_location(module_name, init_file)
                if spec is None:
                    raise ImportError(f"Cannot find spec for {init_file}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                binding_class = getattr(module, module.BindingName)
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load TTV binding {binding_name}: {str(e)}")

    def create_binding(self,
                      binding_name: str,
                      **kwargs) -> Optional[LollmsTTVBinding]:
        """
        Create an instance of a specific TTV binding.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)

        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            try:
                return binding_class(**kwargs)
            except Exception as e:
                trace_exception(e)
                from ascii_colors import ASCIIColors
                ASCIIColors.error(f"Failed to instantiate TTV binding {binding_name}: {str(e)}")
                return None
        return None

    def get_available_bindings(self) -> list[str]:
        """
        Return list of available TTV binding names based on subdirectories.
        """
        return [binding_dir.name for binding_dir in self.ttv_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]


def list_binding_models(ttv_binding_name: str, ttv_binding_config: Optional[Dict[str, any]]|None = None, ttv_bindings_dir: str|Path = Path(__file__).parent / "ttv_bindings") -> List[Dict]:
    """
    Lists all available models for a specific binding.
    """
    binding = LollmsTTVBindingManager(ttv_bindings_dir).create_binding(
        binding_name=ttv_binding_name,
        **{
            k: v
            for k, v in (ttv_binding_config or {}).items()
            if k != "binding_name"
        }
    )

    return binding.list_models() if binding else []
