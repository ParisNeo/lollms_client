# ── Logging Initialization (MUST be first) ───────────────────────────────────
# Configure per-module file routing with rolling rotation BEFORE any other imports
import ascii_colors as logging

# Now import the rest of the library
from lollms_client.lollms_core import LollmsClient, ELF_COMPLETION_FORMAT, LollmsBindingProfile, LollmsModelProfile
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, LollmsMessage
from lollms_client.lollms_memory import LollmsMemoryManager, MemoryConfig, FailureMemory
from lollms_client.lollms_personality.lollms_personality import (
    LollmsPersonality,
    Agent,
    AgentRole,
    RAGDataSource,
    CapabilityFlags,
    SkillsManager,
    SubAgentSpawner,
    ModelSwitcher,
    BindingToolsBuilder,
    ToolsManager
)
from lollms_client.lollms_personality.skill import Skill
from lollms_client.lollms_personality.handbag import Handbag
from lollms_client.lollms_utilities import PromptReshaper
from lollms_client.lollms_tools_binding import LollmsToolBinding, LollmsTOOLBindingManager
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager
from lollms_client.lollms_connection_binding import (
    LollmsConnectionBinding,
    LollmsConnectionBindingManager,
    ConnectionSendResult,
    ConnectionReceiveResult,
)
from lollms_client.lollms_bindings_utils import list_bindings, get_binding_desc


# ── Lazy-loaded workflow exports (PEP 562 to prevent circular import on package bootstrap) ──
_WORKFLOW_EXPORTS = {
    "Workflow",
    "WorkflowNode",
    "WorkflowEdge",
    "NodeType",
    "WorkflowStatus",
    "GuardType",
    "WorkflowContext",
    "WorkflowEngine",
    "GuardViolationError",
    "create_file_organizer_workflow",
    "create_dual_model_review_workflow",
}


def __getattr__(name: str):
    if name in _WORKFLOW_EXPORTS:
        from lollms_client.lollms_workflow.workflow_types import (
            NodeType,
            WorkflowStatus,
            GuardType,
            WorkflowNode,
            WorkflowEdge,
            WorkflowContext,
            Workflow,
        )
        from lollms_client.lollms_workflow.workflow_engine import (
            WorkflowEngine,
            GuardViolationError,
            create_file_organizer_workflow,
            create_dual_model_review_workflow,
        )
        wf_map = {
            "NodeType": NodeType,
            "WorkflowStatus": WorkflowStatus,
            "GuardType": GuardType,
            "WorkflowNode": WorkflowNode,
            "WorkflowEdge": WorkflowEdge,
            "WorkflowContext": WorkflowContext,
            "Workflow": Workflow,
            "WorkflowEngine": WorkflowEngine,
            "GuardViolationError": GuardViolationError,
            "create_file_organizer_workflow": create_file_organizer_workflow,
            "create_dual_model_review_workflow": create_dual_model_review_workflow,
        }
        for k, v in wf_map.items():
            globals()[k] = v
        if name in wf_map:
            return wf_map[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_WORKFLOW_EXPORTS))


__version__ = "1.20.11"

__all__ = [
    "LollmsClient",
    "LollmsBindingProfile",
    "LollmsModelProfile",
    "ELF_COMPLETION_FORMAT",
    "MSG_TYPE",
    "LollmsDiscussion",
    "LollmsMessage",
    "LollmsPersonality",
    "Agent",
    "AgentRole",
    "RAGDataSource",
    "LollmsDataManager",
    "LollmsMemoryManager",
    "MemoryConfig",
    "FailureMemory",
    "PromptReshaper",
    "LollmsToolBinding",
    "LollmsConnectionBinding",
    "LollmsConnectionBindingManager",
    "ConnectionSendResult",
    "ConnectionReceiveResult",
    "LollmsLLMBindingManager",
    "LollmsTOOLBindingManager",
    "list_bindings",
    "get_binding_desc",
    "CapabilityFlags",
    "SkillsManager",
    "SubAgentSpawner",
    "ModelSwitcher",
    "BindingToolsBuilder",
    "ToolsManager",
    "Skill",
    "Handbag",
    "Workflow",
    "WorkflowNode",
    "WorkflowEdge",
    "NodeType",
    "WorkflowStatus",
    "GuardType",
    "WorkflowContext",
    "WorkflowEngine",
    "GuardViolationError",
    "create_file_organizer_workflow",
    "create_dual_model_review_workflow",
]