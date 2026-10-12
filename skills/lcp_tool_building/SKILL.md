---
title: "Building LCP Tools: Privilege Requests and User Input Collection"
description: "Practical how-to for authoring LCP tools: tool function anatomy, docstring-driven schemas, requesting execution privileges via the confirm handler, and collecting user input (text, secrets, select-from-list) mid-execution from the CLI or GUI."
category: "tool_engineering"
tags: [lcp, tools, howto, user_interaction, confirm_handler, secrets, safe_mode]
visibility: loadable
modifiable: true
---

# Building LCP Tools: Privilege Requests & User Input Collection

A hands-on guide to **writing** LCP tools. For binding architecture, discovery internals, and mounting APIs, see the `lcp_tool_development` skill. This skill covers only what a tool author needs: function anatomy, schemas, asking for privileges, and asking the user for information.

---

## 1. Minimal Tool Anatomy

An LCP tool is a plain Python function named `tool_*` in a `.py` file inside a tools folder (e.g. `.lollms_code/tools/my_tool/my_tool.py`):

```python
from typing import Any, Dict

def tool_greet(name: str, enthusiasm: int = 1) -> Dict[str, Any]:
    """
    Greets a person with adjustable enthusiasm.

    Args:
        name (str): The name of the person to greet.
        enthusiasm (int): Number of exclamation marks (1-5).
    """
    return {"success": True, "output": f"Hello {name}" + "!" * max(1, min(5, enthusiasm))}
```

**Rules that matter to you:**
- The **function name** is the tool name (`tool_greet`). Prefix anything else with `_` to keep it private.
- The **first docstring paragraph** becomes the LLM-facing description.
- `Args:` lines in the form `name (type): desc` document parameters.
- Annotations drive the schema: `int→integer`, `float→number`, `bool→boolean`, `list→array`, `dict→object`, everything else `→string`.
- A parameter **with a default is optional**; without one, it is required.
- Always return `{"success": True, "output": ...}` or `{"success": False, "error": ...}`.
- **Coerce defensively** — LLMs often pass strings for numbers:

```python
if isinstance(enthusiasm, str):
    try:
        enthusiasm = int(enthusiasm)
    except ValueError:
        enthusiasm = 1
```

---

## 2. Requesting Privileges (Complex / Sensitive Actions)

When a tool performs risky or irreversible work, gate it behind the host's **confirmation handler**. You never prompt the user yourself — you ask the host, and the host decides how to ask (terminal prompt, GUI modal, auto-approve).

### How your tool receives the confirm handler

Declare the module hooks; the binding injects the handler at lazy load, either via `init_tools_library(config)` (host_tool_configs) or via `set_confirm_handler`:

```python
from typing import Any, Dict, Optional

_CONFIRM_HANDLER = None

def set_confirm_handler(handler: Optional[Any]) -> None:
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler

def init_tools_library(config: Optional[dict] = None) -> None:
    global _CONFIRM_HANDLER
    if config and isinstance(config, dict):
        _CONFIRM_HANDLER = config.get("confirm_handler", _CONFIRM_HANDLER)
```

### Calling it

```python
decision, reason = _CONFIRM_HANDLER(
    source,              # what is about to run (code / command / description)
    script_label,        # short human-readable label
    argv=None,           # optional argument list
)
```

| Return | Meaning | Your action |
| :--- | :--- | :--- |
| `("allow", "")` | Approved once | Proceed |
| `("always", "")` | Approved for the session | Proceed (and skip future prompts if you track it) |
| `("reject", reason)` | Refused | **Stop** and return `{"success": False, "error": reason}` — the reason is fed back to the LLM so it can adapt |

### Autonomy awareness

Hosts configure an autonomy level (`strict` / `safe` / `full_access`) for libraries like `execute_python` and `system_shell`. If your tool is autonomy-aware, read it from the host config:

```python
AUTONOMY_LEVEL = "safe"

def init_tools_library(config: Optional[dict] = None) -> None:
    global AUTONOMY_LEVEL
    if config and isinstance(config, dict):
        AUTONOMY_LEVEL = config.get("autonomy_level", "safe")
```

Honor it: auto-run only benign paths in `safe`, and never assume `full_access`.

---

## 3. Requesting User Input (text, secrets, select)

Sometimes a tool needs **information only the human has** — an API key, a filename, a choice between strategies. LCP supports this with the **user interaction handler**.

### How your tool receives it

Declare `user_interaction_handler=None` as a parameter of your tool function. The binding **hides it from the LLM schema** and injects the live host handler at execution time (fallback: capture it via `init_tools_library(config)` / `set_user_interaction_handler` like the confirm handler above).

```python
def tool_prepare_deployment(
    action: str = "deploy",
    target: str = "production",
    user_interaction_handler=None,   # hidden from the LLM, injected by the host
) -> Dict[str, Any]:
    ...
```

### The request contract

Call the handler with a dict; it returns `Optional[str]` (`None` = user cancelled or timed out):

```python
answer = user_interaction_handler({
    "type": "select",              # "text" | "secret" | "select"
    "title": "Deployment strategy",
    "prompt": "Choose how the service should be rolled out:",
    "options": ["blue-green", "rolling", "canary"],   # select only
    "default": "canary",           # optional preselected/fallback value
})
```

| `type` | Renders as | Notes |
| :--- | :--- | :--- |
| `"text"` | Free-text input box | `default` pre-fills the field |
| `"secret"` | Password field (masked, toggleable) | **Never echo the value into `output`** — mask it (`***1234`) |
| `"select"` | Dropdown of `options` | `default` preselects if present in options |

### Handling the answer

```python
if answer is None:
    return {"success": False, "error": "Cancelled by user (strategy selection)."}
```

Treat `None` as a first-class cancellation: return a clean failure immediately; never retry in a loop.

### Where it works

- **lollms-code CLI**: terminal panel + prompt (select shows numbered options; secrets use `getpass` masking; non-interactive shells fall back to `default` or cancel).
- **lollms-code GUI**: persistent modal dialog dispatched through the event queue (text box / password field / dropdown), wired into stop/cancel so pressing ⏹ never deadlocks the worker thread.

---

## 4. Complete Worked Example

A deployment-preparation tool that combines **privilege gating** (production requires confirmation) with **three user-input requests**:

```python
"""
deploy_helper.py — Collects operator input before preparing a deployment.
"""
from typing import Any, Dict, Optional

TOOL_LIBRARY_NAME = "Deploy Helper"
TOOL_LIBRARY_DESC = "Prepares deployments by collecting strategy, credentials and tags from the operator."
TOOL_LIBRARY_ICON = "🚀"

_CONFIRM_HANDLER = None
_USER_INTERACTION_HANDLER = None

def set_confirm_handler(handler: Optional[Any]) -> None:
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler

def set_user_interaction_handler(handler: Optional[Any]) -> None:
    global _USER_INTERACTION_HANDLER
    _USER_INTERACTION_HANDLER = handler

def init_tools_library(config: Optional[dict] = None) -> None:
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
        return {"success": False,
                "error": "No user interaction handler available in this host."}

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
        "prompt": f"Enter the API key for the '{target}' registry:",
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
        "output": f"Deployment prepared: {action} → {target} (strategy: {strategy}, tag: {image_tag}, key: {masked_key}).",
        "strategy": strategy,
        "image_tag": image_tag,
    }
```

A runnable copy of this example ships with the client at
`tools_bindings/lcp/default_tools/user_interaction_demo/user_interaction_demo.py` — it is auto-discovered by lollms-code, so you can ask the agent *"use tool_prepare_deployment to deploy to production"* and watch the modal prompts appear.

---

## 5. Best Practices Checklist

1. **Gate, don't ask twice**: use the confirm handler for *authorization* (yes/no/always) and the interaction handler for *information* (values). Never use one for the other's job.
2. **Never log secrets**: mask returned secret values (`***1234`) before putting them in `output`.
3. **Cancel fast**: `None` from the interaction handler or `reject` from the confirm handler means stop immediately with an actionable error — the LLM will see it and adjust.
4. **Hide injected params**: `user_interaction_handler`, `lollms_client_instance`, `discussion_instance`, `tool_context` never appear in the LLM schema — declare them with `=None` defaults.
5. **Provide `default`s** where a sensible choice exists; hosts use them in non-interactive environments.
6. **Keep helpers private** (`_resolve_...`, `_CONFIRM_HANDLER`) — no `tool_` prefix.
7. **Return the canonical shape**: `{"success": bool, "output"|"error": ...}` plus optional payload keys.