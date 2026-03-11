# lollms_discussion/__init__.py
#
# Public surface of the lollms_discussion package.
# All symbols that existed in the original single-file module are re-exported
# here so that existing import statements keep working unchanged:
#
#   from lollms_discussion import LollmsDiscussion, LollmsDataManager, ...
#   import lollms_discussion as ld; ld.LollmsDiscussion(...)
#
# Internal structure
# ------------------
#   _db.py          EncryptedString, create_dynamic_models, LollmsDataManager
#   _message.py     LollmsMessage
#   _artefacts.py   ArtefactType, ArtefactManager
#   _mixin_core.py  CoreMixin   – __init__, factories, proxy, DB helpers, message CRUD
#   _mixin_prompt.py PromptMixin – instruction builders, _post_process_llm_response
#   _mixin_chat.py  ChatMixin   – simplified_chat(), chat()
#   _mixin_utils.py UtilsMixin  – export, prune, images, shims, serialisation

from ._db        import EncryptedString, create_dynamic_models, LollmsDataManager
from ._message   import LollmsMessage
from ._artefacts import ArtefactType, ArtefactManager

from ._mixin_core   import CoreMixin
from ._mixin_prompt import PromptMixin
from ._mixin_chat   import ChatMixin
from ._mixin_utils  import UtilsMixin


class LollmsDiscussion(CoreMixin, PromptMixin, ChatMixin, UtilsMixin):
    """
    Represents and manages a single discussion.
    
    Attributes:
        scratchpad (str): A volatile area containing full-length tool outputs 
                          and technical data for the current generation turn.

    Composed from four mixins (MRO left-to-right):
      CoreMixin    – construction, proxying, DB ops, message CRUD, data zones
      PromptMixin  – system-prompt builders, LLM response post-processing
      ChatMixin    – simplified_chat(), chat()
      UtilsMixin   – export, prune, context status, images, shims, serialisation

    All public methods from the original single-file implementation are present
    and behave identically (full retrocompatibility).
    """
    # No extra code needed — the MRO gives us everything.
    pass


__all__ = [
    # Main class
    "LollmsDiscussion",
    # Data layer
    "LollmsDataManager",
    "EncryptedString",
    "create_dynamic_models",
    # Supporting classes
    "LollmsMessage",
    "ArtefactType",
    "ArtefactManager",
]
