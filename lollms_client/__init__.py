# lollms_client/__init__.py
from lollms_client.lollms_core import LollmsClient, ELF_COMPLETION_FORMAT
from lollms_client.lollms_tasks import TasksLibrary
from lollms_client.lollms_types import MSG_TYPE # Assuming ELF_GENERATION_FORMAT is not directly used by users from here
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
from lollms_client.lollms_utilities import PromptReshaper # Keep general utilities
from lollms_client.lollms_functions import FunctionCalling_Library

__version__ = "0.15.2"

# Optionally, you could define __all__ if you want to be explicit about exports
__all__ = [
    "LollmsClient",
    "ELF_COMPLETION_FORMAT",
    "TasksLibrary",
    "MSG_TYPE",
    "LollmsDiscussion",
    "LollmsMessage",
    "PromptReshaper",
    "FunctionCalling_Library"
]
