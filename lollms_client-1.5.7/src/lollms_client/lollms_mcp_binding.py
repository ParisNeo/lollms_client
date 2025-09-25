# lollms_client/lollms_mcp_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from ascii_colors import trace_exception, ASCIIColors
import yaml
class LollmsMCPBinding(ABC):
    """
    Abstract Base Class for LOLLMS Model Context Protocol (MCP) Bindings.

    MCP bindings are responsible for interacting with MCP-compliant tool servers
    or emulating MCP tool interactions locally. They handle tool discovery
    and execution based on requests, typically orchestrated by an LLM.
    """

    def __init__(self,
                 binding_name: str
                 ):
        """
        Initialize the LollmsMCPBinding.

        Args:
            binding_name (str): The unique name of this binding.
        """
        self.binding_name = binding_name


    @abstractmethod
    def discover_tools(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Discover available tools compliant with the MCP specification.

        Each tool definition should follow the MCP standard, typically including:
        - name (str): Unique name of the tool.
        - description (str): Natural language description of what the tool does.
        - input_schema (dict): JSON schema defining the tool's input parameters.
        - output_schema (dict): JSON schema defining the tool's output.
        (Other MCP fields like `prompts`, `resources` could be supported by specific bindings)

        Args:
            **kwargs: Additional arguments specific to the binding's discovery mechanism
                      (e.g., tool_server_url, specific_tool_names_to_filter).

        Returns:
            List[Dict[str, Any]]: A list of tool definitions. Each dictionary
                                  should conform to the MCP tool definition structure.
                                  Returns an empty list if no tools are found or an error occurs.
        """
        pass

    @abstractmethod
    def execute_tool(self,
                     tool_name: str,
                     params: Dict[str, Any],
                     **kwargs) -> Dict[str, Any]:
        """
        Execute a specified tool with the given parameters.

        The execution should adhere to the input and output schemas defined in the
        tool's MCP definition.

        Args:
            tool_name (str): The name of the tool to execute.
            params (Dict[str, Any]): A dictionary of parameters to pass to the tool,
                                     conforming to the tool's `input_schema`.
            **kwargs: Additional arguments specific to the binding's execution mechanism
                      (e.g., timeout, user_context).

        Returns:
            Dict[str, Any]: The result of the tool execution, conforming to the
                            tool's `output_schema`. If an error occurs during
                            execution, the dictionary should ideally include an 'error'
                            key with a descriptive message.
                            Example success: {"result": "Weather is sunny"}
                            Example error: {"error": "API call failed", "details": "..."}
        """
        pass

    def get_binding_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the binding.

        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        return self.config

class LollmsMCPBindingManager:
    """
    Manages discovery and instantiation of MCP bindings.
    """

    def __init__(self, mcp_bindings_dir: Union[str, Path] = Path(__file__).parent / "mcp_bindings"):
        """
        Initialize the LollmsMCPBindingManager.

        Args:
            mcp_bindings_dir (Union[str, Path]): Directory containing MCP binding implementations.
                                                 Defaults to "mcp_bindings" subdirectory relative to this file.
        """
        self.mcp_bindings_dir = Path(mcp_bindings_dir)
        if not self.mcp_bindings_dir.is_absolute():
             # If relative, assume it's relative to the parent of this file (lollms_client directory)
            self.mcp_bindings_dir = (Path(__file__).parent.parent / mcp_bindings_dir).resolve()

        self.available_bindings: Dict[str, type[LollmsMCPBinding]] = {}
        ASCIIColors.info(f"LollmsMCPBindingManager initialized. Bindings directory: {self.mcp_bindings_dir}")


    def _load_binding_class(self, binding_name: str) -> Optional[type[LollmsMCPBinding]]:
        """
        Dynamically load a specific MCP binding class from the mcp_bindings directory.
        Assumes each binding is in a subdirectory named after the binding_name,
        and has an __init__.py that defines a `BindingName` variable and the binding class.
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
                        
                        # Ensure BindingName is defined in the module, and it matches the class name
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

        Args:
            binding_name (str): Name of the MCP binding to create.
            config (Optional[Dict[str, Any]]): Configuration for the binding.
            lollms_paths (Optional[Dict[str, Union[str, Path]]]): LOLLMS specific paths.


        Returns:
            Optional[LollmsMCPBinding]: Binding instance or None if creation failed.
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
        Return list of available MCP binding names based on subdirectories.
        This method scans the directory structure.
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

        Args:
            binding_name (str): The name of the binding.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with the parsed YAML content, or None if the file doesn't exist or is invalid.
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