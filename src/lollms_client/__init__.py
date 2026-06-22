# ── Logging Initialization (MUST be first) ───────────────────────────────────
# Configure per-module file routing with rolling rotation BEFORE any other imports
import ascii_colors as logging

# Now import the rest of the library
from lollms_client.lollms_core import LollmsClient, ELF_COMPLETION_FORMAT
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, LollmsMessage
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_utilities import PromptReshaper
from lollms_client.lollms_tools_binding import LollmsToolBinding, LollmsTOOLBindingManager
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager
from lollms_client.lollms_bindings_utils import list_bindings, get_binding_desc

__version__ = "1.15.3" # Updated version

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
    "LollmsToolBinding",
    "LollmsLLMBindingManager",
    "LollmsTOOLBindingManager",
    "list_bindings",
    "get_binding_desc",
    "logger",
    "openai_logger",
    "discussion_logger",
]
