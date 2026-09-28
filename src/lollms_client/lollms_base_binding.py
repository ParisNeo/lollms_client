from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import os
import yaml
import inspect
from ascii_colors import ASCIIColors

class LollmsBaseBinding(ABC):
    """
    Base class for all LOLLMS bindings (LLM, TTI, TTS, STT, TTM, TTV, MCP).
    Enforces a unified initialization, resource lifecycle management, and
    strict separation between the system configuration directory (~/.lollms_client)
    and the project workspace.
    """
    def __init__(self, binding_name: str, debug: Optional[bool] = False, **kwargs):
        """
        Initialize the binding.
        
        Args:
            binding_name (str): The name of the binding.
            debug (Optional[bool]): Enable debug logging.
            **kwargs: Configuration parameters passed from the manager/app.
        """
        self.binding_name = binding_name
        self.debug = debug
        self.config = kwargs
        self._last_error: Optional[str] = None

        # Optional override for bindings root/cwd. If not provided, defaults to Path(".") (current directory)
        base_override = (
            kwargs.get("system_dir")
            or kwargs.get("lollms_path")
            or kwargs.get("bindings_path")
            or kwargs.get("cwd")
            or os.getenv("LOLLMS_SYSTEM_PATH")
        )
        if base_override:
            self.system_dir = Path(base_override).expanduser().resolve()
            self.system_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.system_dir = Path(".").resolve()

        self.binding_dir = self._get_binding_dir()
        self.description = self._load_description()

    @staticmethod
    def get_system_dir() -> Path:
        """
        Returns the root system directory for LOLLMS shared resources (binaries, models, caches).
        Defaults to ~/.lollms_client (or overridden via LOLLMS_SYSTEM_PATH environment variable).
        """
        env_path = os.getenv("LOLLMS_SYSTEM_PATH")
        if env_path:
            p = Path(env_path).expanduser().resolve()
        else:
            p = (Path.home() / ".lollms_client").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolve_system_path(self, relative_or_absolute_path: Union[str, Path]) -> Path:
        """
        Resolves a path relative to self.system_dir (which defaults to current working directory,
        or an explicit override when provided).
        """
        p = Path(relative_or_absolute_path).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self.system_dir / p).resolve()
        
    def _get_binding_dir(self) -> Path:
        """
        Locates the directory of the concrete binding class.
        """
        try:
            return Path(inspect.getfile(self.__class__)).parent
        except Exception:
            return Path(".")

    def _load_description(self) -> Dict:
        """
        Loads the description.yaml file from the binding directory.
        """
        desc_file = self.binding_dir / "description.yaml"
        if desc_file.exists():
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                ASCIIColors.error(f"Failed to load description.yaml for {self.binding_name}: {e}")
        return {}

    # ── Resource Management Contract ──────────────────────────────────────────

    def is_local(self) -> bool:
        """
        Indicates whether this binding runs locally on the host machine and consumes
        local RAM/VRAM resources. Remote/API bindings (e.g., OpenAI, Claude, Groq)
        return False. Local server/in-process bindings (e.g., llama_cpp_server, diffusers, whisper)
        return True.
        """
        return False

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        """
        Checks if the specified model (or active model) is currently loaded in memory/VRAM.
        For remote bindings, returns True (always available).
        For local bindings, returns True only if actively loaded and ready for inference.
        """
        return True if not self.is_local() else False

    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Loads the specified model into memory/VRAM.
        Returns True on success, False if loading failed (e.g. out of VRAM/RAM or file missing).
        """
        return True

    def unload_model(self, model_name: Optional[str] = None) -> bool:
        """
        Unloads the specified model (or all models) from memory/VRAM.
        Default base/remote implementation simulates freeing 0 bytes.
        """
        target = model_name or getattr(self, "model_name", "unknown")
        ASCIIColors.info(f"[{self.binding_name}] Simulating model unload for remote/unsupported binding. Liberated model '{target}' with 0 bytes.")
        return True

    def get_loaded_models(self) -> List[str]:
        """
        Returns the list of model identifiers currently occupying local memory/VRAM.
        Remote bindings return an empty list.
        """
        return []

    def has_active_resources(self) -> bool:
        """
        Checks if this binding is currently holding local memory/VRAM/process resources.
        """
        if not self.is_local():
            return False
        return bool(self.get_loaded_models())

    def get_last_error(self) -> Optional[str]:
        """
        Returns the last error message encountered during model loading or inference.
        """
        return self._last_error

    @abstractmethod
    def list_models(self) -> List[Any]:
        """
        List available models or resources provided by this binding.
        Must be implemented by all bindings.
        """
        pass

    def ps(self) -> List[Dict[str, Any]]:
        """
        Verify resources or processes associated with this binding.
        Returns a list of status information. Default remote/API simulation.
        """
        target = getattr(self, "model_name", "unknown")
        return [{
            "model_name": target,
            "is_loaded": True,
            "device": "cloud/api" if not self.is_local() else "local",
            "vram_size": 0,
            "gpu_usage_percent": 0.0,
            "cpu_usage_percent": 0.0,
            "ref_count": 1,
            "status": "active"
        }]

    def get_server_logs(self) -> str:
        """
        This method returns the logs for bindings that spin out a background server.
        """
        return ""