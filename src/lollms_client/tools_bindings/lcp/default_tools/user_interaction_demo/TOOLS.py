"""
user_interaction_demo.py — Reference LCP tool demonstrating privilege requests
(confirm handler via host config) and user interaction requests
(text / secret / select) from inside a tool's internal flow.

Use case: preparing a deployment that legitimately requires the operator to
pick a strategy, provide registry credentials, and name an image tag — all
collected mid-execution before the tool returns its unified result.
"""
from typing import Any, Dict, Optional

TOOL_LIBRARY_NAME = "User Interaction Demo"
TOOL_LIBRARY_DESC = (
    "Demonstrates human-in-the-loop input requests (text, secret, select) "
    "from inside LCP tools."
)
TOOL_LIBRARY_ICON = "🙋"

_CONFIRM_HANDLER = None
_USER_INTERACTION_HANDLER = None


def set_confirm_handler(handler: Optional[Any]) -> None:
    """Host-injected hook: receives the authorization callback."""
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler


def set_user_interaction_handler(handler: Optional[Any]) -> None:
    """Host-injected hook: receives the user-interaction callback."""
    global _USER_INTERACTION_HANDLER
    _USER_INTERACTION_HANDLER = handler


def init_tools_library(config: Optional[dict] = None) -> None:
    """Host-injected hook: captures handlers and settings from host_tool_configs."""
    global _CONFIRM_HANDLER, _USER_INTERACTION_HANDLER
    if config and isinstance(config, dict):
        _CONFIRM_HANDLER = config.get("confirm_handler", _CONFIRM_HANDLER)
        _USER_INTERACTION_HANDLER = config.get("user_interaction_handler", _USER_INTERACTION_HANDLER)


def tool_prepare_deployment(
    action: str = "deploy",
    target: str = "production",
    user_interaction_handler=None,
) -> Dict[str, Any]:
    """
    Prepares a deployment by collecting the strategy, registry credentials and
    image tag from the operator before returning a unified deployment plan.

    Args:
        action (str): The deployment action to perform (e.g. 'deploy', 'rollback').
        target (str): The target environment (e.g. 'production', 'staging').
    """
    handler = user_interaction_handler or _USER_INTERACTION_HANDLER
    if handler is None:
        return {
            "success": False,
            "error": "No user interaction handler available. This tool must run inside a host that injects one (lollms-code CLI or GUI).",
        }

    if target == "production" and _CONFIRM_HANDLER is not None:
        decision, reason = _CONFIRM_HANDLER(
            f"prepare_deployment(action={action}, target={target})",
            f"deploy:{target}",
        )
        if decision == "reject":
            return {"success": False, "error": f"Operator refused production access: {reason}"}

    strategy = handler({
        "type": "select",
        "title": f"Deployment strategy for {target}",
        "prompt": "Choose how the service should be rolled out:",
        "options": ["blue-green", "rolling", "canary"],
        "default": "canary",
    })
    if strategy is None:
        return {"success": False, "error": "Cancelled by user (strategy selection)."}

    api_key = handler({
        "type": "secret",
        "title": "Registry credentials",
        "prompt": f"Enter the API key used to push images to the '{target}' registry:",
    })
    if not api_key:
        return {"success": False, "error": "Cancelled by user (registry API key is required)."}

    image_tag = handler({
        "type": "text",
        "title": "Image tag",
        "prompt": "Which image tag should be deployed?",
        "default": "latest",
    })
    if image_tag is None:
        return {"success": False, "error": "Cancelled by user (image tag)."}

    masked_key = f"***{api_key[-4:]}" if len(api_key) >= 4 else "****"
    return {
        "success": True,
        "output": (
            f"Deployment prepared: {action} → {target} "
            f"(strategy: {strategy}, tag: {image_tag}, key: {masked_key})."
        ),
        "strategy": strategy,
        "image_tag": image_tag,
    }