from lollms_client.lollms_core import LollmsClient, ELF_COMPLETION_FORMAT
from lollms_client.lollms_types import MSG_TYPE # Assuming ELF_GENERATION_FORMAT is not directly used by users from here
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, LollmsMessage
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_utilities import PromptReshaper # Keep general utilities
# Import new MCP binding classes
from lollms_client.lollms_mcp_binding import LollmsMCPBinding, LollmsMCPBindingManager
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager
# Import new bindings utils
from lollms_client.lollms_bindings_utils import list_bindings, get_binding_desc

__version__ = "1.11.1" # Updated version

# Optionally, you could define __all__ if you want to be explicit about exports
__all__ = [
    "LollmsClient",
    "ELF_COMPLETION_FORMAT",
    "MSG_TYPE",
    "LollmsDiscussion",
    "LollmsMessage",
    "LollmsPersonality",
    "LollmsDataManager",
    "PromptReshaper",
    "LollmsMCPBinding", # Export LollmsMCPBinding ABC
    "LollmsLLMBindingManager",
    "LollmsMCPBindingManager", # Export LollmsMCPBindingManager
    "list_bindings",
    "get_binding_desc"
]
