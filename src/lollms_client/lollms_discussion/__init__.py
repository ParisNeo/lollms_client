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
#   _db.py           EncryptedString, create_dynamic_models, LollmsDataManager
#   _message.py      LollmsMessage
#   _artefacts.py    ArtefactType, ArtefactManager
#   _mixin_core.py   CoreMixin    – __init__, factories, proxy, DB helpers, message CRUD
#   _mixin_prompt.py PromptMixin  – instruction builders, _post_process_llm_response
#   _mixin_chat.py   ChatMixin    – simplified_chat(), chat()
#   _mixin_utils.py  UtilsMixin   – export, prune, images, shims, serialisation
#   _mixin_branch.py BranchMixin  – branch discovery, navigation, forking, deletion

from ._db        import EncryptedString, create_dynamic_models, LollmsDataManager
from ._message   import LollmsMessage
from ._artefacts import ArtefactType, ArtefactManager

from ._mixin_core   import CoreMixin
from ._mixin_prompt import PromptMixin
from ._mixin_chat   import ChatMixin
from ._mixin_utils  import UtilsMixin
from ._mixin_branch import BranchMixin, BranchInfo, MessageNode
from ._mixin_memory import MemoryMixin


class LollmsDiscussion(CoreMixin, PromptMixin, ChatMixin, UtilsMixin, BranchMixin, MemoryMixin):
    """
    Represents and manages a single discussion.

    Memory system
    ─────────────
    Attach a LollmsMemoryManager to enable multi-level persistent memory:

        from lollms_client.lollms_discussion.lollms_memory import LollmsMemoryManager, MemoryConfig

        mem = LollmsMemoryManager(
            db_path="sqlite:///memories.db",
            owner_id=discussion.id,
            config=MemoryConfig(working_token_budget=800),
        )
        discussion.memory_manager = mem          # attach after creation
        # … or pass at chat time:
        discussion.chat("...", memory_manager=mem)

    Attributes:
        scratchpad (str): A volatile area containing full-length tool outputs
                          and technical data for the current generation turn.

    Composed from five mixins (MRO left-to-right):
      CoreMixin    – construction, proxying, DB ops, message CRUD, data zones
      PromptMixin  – system-prompt builders, LLM response post-processing
      ChatMixin    – simplified_chat(), chat()
      UtilsMixin   – export, prune, context status, images, shims, serialisation
      BranchMixin  – branch discovery, navigation, forking, deletion, merging

    Branch management quick-reference
    ----------------------------------
    # Discover
    disc.list_branches()                    → [BranchInfo, …]
    disc.get_tree()                         → [MessageNode, …]  (forest)
    disc.get_children(msg_id)               → [LollmsMessage, …]
    disc.get_siblings(msg_id)               → [LollmsMessage, …]
    disc.get_branch_info(leaf_id)           → BranchInfo | None
    disc.get_message_branches(msg_id)       → [BranchInfo, …]

    # Navigate
    disc.switch_branch(leaf_id)             → bool
    disc.switch_to_sibling(direction=±1)    → LollmsMessage | None

    # Create / fork
    disc.fork_from(msg_id, label=…)         → LollmsMessage   (new branch tip)

    # Delete
    disc.delete_branch(leaf_id)             → int  (messages removed)
    disc.prune_branch(msg_id)               → int  (subtree removed)

    # Merge
    disc.merge_branches(source, target)     → LollmsMessage   (new tip)

    # Label
    disc.set_branch_label(leaf_id, label)   → bool

    # Diff
    disc.branch_diff(leaf_a, leaf_b)        → dict
    """
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
    # Branch management
    "BranchInfo",
    "MessageNode",
    "MemoryMixin",
]