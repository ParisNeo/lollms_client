# ── Logging Initialization (MUST be first) ───────────────────────────────────
# Configure per-module file routing with rolling rotation BEFORE any other imports
import ascii_colors as logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_folder="./logs",                # Per-module file routing
    log_folder_mode="rolling",          # Rotate by size
    log_folder_maxBytes=10_000_000,     # 10MB per file before rotation
    log_folder_backupCount=5,           # Keep 5 backup files
)

# Now import the rest of the library
from lollms_client.lollms_core import LollmsClient, ELF_COMPLETION_FORMAT
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, LollmsMessage
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_utilities import PromptReshaper
from lollms_client.lollms_tools_binding import LollmsToolBinding, LollmsTOOLBindingManager
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager
from lollms_client.lollms_bindings_utils import list_bindings, get_binding_desc

__version__ = "1.14.23"

# Create module-level loggers for easy access
logger = logging.getLogger(__name__)
openai_logger = logging.getLogger("lollms_client.llm_bindings.openai")
discussion_logger = logging.getLogger("lollms_client.lollms_discussion")

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
