# lollms_client/lollms_mcp_binding.py
from abc import abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from ascii_colors import trace_exception, ASCIIColors
import yaml
from lollms_client.lollms_base_binding import LollmsBaseBinding

class LollmsMCPBinding(LollmsBaseBinding):
    """
    Abstract Base Class for LOLLMS Model Context Protocol (MCP) Bindings.
    """

    def __init__(self,
                 binding_name: str,
                 **kwargs
                 ):
        """
        Initialize the LollmsMCPBinding.
        """
        super().__init__(binding_name=binding_name, **kwargs)
        self.settings = kwargs


    @abstractmethod
    def discover_tools(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Discover available tools compliant with the MCP specification.
        """
        pass

    @abstractmethod
    def execute_tool(self,
                     tool_name: str,
                     params: Dict[str, Any],
                     **kwargs) -> Dict[str, Any]:
        """
        Execute a specified tool with the given parameters.
        """
        pass

    def get_binding_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the binding.
        """
        return self.settings

    def list_models(self) -> List[Any]:
        """
        For MCP, list_models returns an empty list as it primarily deals with tools.
        Or could be implemented to return available tools as models if desired.
        """
        return []
    
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

class LollmsMCPBindingManager:
    """
    Manages discovery and instantiation of MCP bindings.
    """

    def __init__(self, mcp_bindings_dir: Union[str, Path] = Path(__file__).parent / "mcp_bindings"):
        """
        Initialize the LollmsMCPBindingManager.
        """
        self.mcp_bindings_dir = Path(mcp_bindings_dir)
        if not self.mcp_bindings_dir.is_absolute():
            self.mcp_bindings_dir = (Path(__file__).parent.parent / mcp_bindings_dir).resolve()

        self.available_bindings: Dict[str, type[LollmsMCPBinding]] = {}


    def _load_binding_class(self, binding_name: str) -> Optional[type[LollmsMCPBinding]]:
        """
        Dynamically load a specific MCP binding class.
        """
        binding_dir = self.mcp_bindings_dir / binding_name
        if binding_dir.is_dir():
            init_file_path = binding_dir / "__init__.py"
            if init_file_path.exists():
                try:
                    module_spec = importlib.util.spec_from_file_location(
                        f"lollms_client.mcp_bindings.{binding_name}",
                        str(init_file_path)
                    )
                    if module_spec and module_spec.loader:
                        module = importlib.util.module_from_spec(module_spec)
                        module_spec.loader.exec_module(module)
                        
                        if not hasattr(module, 'BindingName'):
                            ASCIIColors.warning(f"Binding '{binding_name}' __init__.py does not define BindingName variable.")
                            return None
                            
                        binding_class_name = module.BindingName
                        if not hasattr(module, binding_class_name):
                            ASCIIColors.warning(f"Binding '{binding_name}' __init__.py defines BindingName='{binding_class_name}', but class not found.")
                            return None

                        binding_class = getattr(module, binding_class_name)
                        if not issubclass(binding_class, LollmsMCPBinding):
                             ASCIIColors.warning(f"Class {binding_class_name} in {binding_name} is not a subclass of LollmsMCPBinding.")
                             return None
                        return binding_class
                    else:
                        ASCIIColors.warning(f"Could not create module spec for MCP binding '{binding_name}'.")
                except Exception as e:
                    ASCIIColors.error(f"Failed to load MCP binding '{binding_name}': {e}")
                    trace_exception(e)
        return None

    def create_binding(self,
                       binding_name: str,
                       **kwargs
                       ) -> Optional[LollmsMCPBinding]:
        """
        Create an instance of a specific MCP binding.
        """
        if binding_name not in self.available_bindings:
            binding_class = self._load_binding_class(binding_name)
            if binding_class:
                self.available_bindings[binding_name] = binding_class
            else:
                ASCIIColors.error(f"MCP binding '{binding_name}' class not found or failed to load.")
                return None
        
        binding_class_to_instantiate = self.available_bindings.get(binding_name)
        if binding_class_to_instantiate:
            try:
                return binding_class_to_instantiate(
                    **kwargs
                )
            except Exception as e:
                ASCIIColors.error(f"Failed to instantiate MCP binding '{binding_name}': {e}")
                trace_exception(e)
                return None
        return None


    def get_available_bindings(self) -> List[str]:
        """
        Return list of available MCP binding names.
        """
        available = []
        if self.mcp_bindings_dir.is_dir():
            for item in self.mcp_bindings_dir.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    available.append(item.name)
        return available

    def get_binding_description(self, binding_name: str) -> Optional[Dict[str, Any]]:
        """
        Loads and returns the content of the description.yaml file for a given binding.
        """
        binding_dir = self.mcp_bindings_dir / binding_name
        description_file = binding_dir / "description.yaml"

        if not description_file.exists():
            ASCIIColors.warning(f"No description.yaml found for MCP binding '{binding_name}'.")
            return None

        try:
            with open(description_file, 'r', encoding='utf-8') as f:
                description = yaml.safe_load(f)
            return description
        except yaml.YAMLError as e:
            ASCIIColors.error(f"Error parsing description.yaml for MCP binding '{binding_name}': {e}")
            trace_exception(e)
            return None
        except Exception as e:
            ASCIIColors.error(f"Error reading description.yaml for MCP binding '{binding_name}': {e}")
            trace_exception(e)
            return None

def list_binding_models(mcp_binding_name: str, mcp_binding_config: Optional[Dict[str, any]]|None = None, mcp_bindings_dir: str|Path = Path(__file__).parent / "mcp_bindings") -> List[Dict]:
    """
    Lists all available models/tools for a specific binding.
    """
    binding = LollmsMCPBindingManager(mcp_bindings_dir).create_binding(
        binding_name=mcp_binding_name,
        **{
            k: v
            for k, v in (mcp_binding_config or {}).items()
            if k != "binding_name"
        }
    )

    return binding.discover_tools() if binding else []
