---
title: "Building LCP Tools: A Practical Creation Guide"
description: "Step-by-step guide to writing LCP tools: schema-driven docstrings, return contracts, requesting privileges via the confirmation handler, and asking the user for keys, text, or choices with the user-interaction protocol. Includes a complete worked use case."
category: "tool_engineering"
tags: [lcp, tools, howto, user_interaction, privileges, confirm_handler]
visibility: loadable
modifiable: true
---

# Building LCP Tools — A Practical Creation Guide

This skill teaches you **how to write LCP tools**, not how the engine works. For the binding internals (discovery, AST schema generation, mounting APIs), see the companion skill `lcp_tool_development`.

An LCP tool is a plain Python function named `tool_*` in a `.py` file placed in a tools folder. That's it. No classes, no registration calls, no YAML — the function name, type annotations, defaults, and docstring *are* the tool.

---

## 1. Your First Tool in 60 Seconds

Create `my_tools/greeting.py`:

```python
from typing import Dict

def tool_greet(name: str, excitement_level: int = 1) -> Dict:
    """
    Greets a person with an adjustable level of enthusiasm.

    Args:
        name (str): The name of the person to greet.
        excitement_level (int): Number of exclamation marks to add (1-5).
    """
    level = max(1, min(5, int(excitement_level)))
    return {"success": True, "output": f"Hello, {name}{'!' * level}"}
```

---

## 2. The Docstring IS the Schema

The engine reads your source with `ast` — there is no config file. Three rules:

1. **First docstring paragraph** = the description the LLM reads. Make it an instruction: what it does, when to use it.
2. **`Args:` block** = parameter documentation. Use `param (type): description` lines.
3. **Annotations and defaults** = the JSON schema:

   | Write this | LLM sees |
   | :--- | :--- |
   | `name: str` | required string |
   | `count: int = 5` | optional integer, default 5 |
   | `items: list = []` | optional array |
   | `flags: dict = {}` | optional object |

Pitfalls:
- LLMs often pass strings for numbers/booleans — coerce defensively: `count = int(count) if isinstance(count, str) else count`.
- Helpers must NOT start with `tool_` or they become tools.
- `lollms_client_instance`, `discussion_instance`, `tool_context` params are hidden from the LLM but injected at runtime (declare them with `=None` defaults).

---

## 3. The Return Contract

Always return a dict. Success and failure must be unambiguous:

```python
{"success": True, "output": "Human-readable summary the LLM will reason over", ...extra payload keys...}
{"success": False, "error": "Actionable message telling the LLM what went wrong and how to fix it"}
```

- The `output` value is what the LLM actually reads — write it for the model, not for logs.
- Extra keys (`image_filename`, `matches_count`, ...) ride along as payload.
- A special `prompt_injection` key appends guidance to the LLM's next turn (e.g. "show the artefact with `<artefact_image .../>`").
- Catch your own exceptions — a crash still returns an error to the LLM, but your message is more useful than a traceback.

---

## 4. Requesting Privileges (Confirmation Handler)

When a tool does something **sensitive, irreversible, or privileged** — network calls, subprocesses, deleting files, spending money — gate it behind the host's confirmation handler. The handler is delivered to your module automatically (via `init_tools_library` config or the `set_confirm_handler` hook):

```python
_CONFIRM_HANDLER = None

def set_confirm_handler(handler):
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler

def init_tools_library(config=None):
    global _CONFIRM_HANDLER
    if config and isinstance(config, dict):
        _CONFIRM_HANDLER = config.get("confirm_handler", _CONFIRM_HANDLER)

def tool_deploy_website(site_name: str, target_dir: str) -> Dict:
    """
    Deploys the workspace site to a production directory.

    Args:
        site_name (str): Name of the site to deploy.
        target_dir (str): Destination directory for the deployment.
    """
    if _CONFIRM_HANDLER is not None:
        source = f"Copy built assets of '{site_name}' to '{target_dir}'"
        decision, reason = _CONFIRM_HANDLER(source, f"Deploy website '{site_name}'")
        if decision not in ("allow", "always"):
            return {"success": False,
                    "error": f"Deployment denied by operator: {reason or 'no reason given'}"}
    # ... perform the deployment ...
    return {"success": True, "output": f"Site '{site_name}' deployed to {target_dir}."}
```

Contract: `confirm_handler(source, script_label, argv=None) -> (decision, reason)` where `decision` is `"allow"`, `"always"` (auto-approve for the session), or `"reject"`. If no handler is registered (`None`), decide your own fail-safe policy: skip the privileged step and explain, or fail with a clear error. Never silently proceed.

---

## 5. Asking the User for Input (User Interaction Protocol)

Some tools cannot proceed without information only the human has: **an API key, a clarification, a choice between options**. The user-interaction protocol lets a tool pause and ask.

### The handler contract

The host registers one callback. Your tool sends a request dict, gets back a string (or `None` on cancel):

```python
def user_interaction_handler(request: dict) -> Optional[str]:
    # request: {"type": "text" | "secret" | "select",
    #           "title": str,           # short headline, e.g. "OpenWeather API key"
    #           "prompt": str,          # the question to show the user
    #           "options": List[str],   # only for "select"
    #           "default": Optional[str]}
```

### Receiving the handler in your module

Identical pattern to the confirm handler — the binding injects it at lazy load and on every execution:

```python
_USER_INTERACTION_HANDLER = None

def set_user_interaction_handler(handler):
    global _USER_INTERACTION_HANDLER
    _USER_INTERACTION_HANDLER = handler

def init_tools_library(config=None):
    global _USER_INTERACTION_HANDLER
    if config and isinstance(config, dict):
        _USER_INTERACTION_HANDLER = config.get("user_interaction_handler", _USER_INTERACTION_HANDLER)

def _ask_user(request_type, title, prompt, options=None, default=None):
    if _USER_INTERACTION_HANDLER is None:
        raise RuntimeError("The host application does not support user interaction.")
    return _USER_INTERACTION_HANDLER({
        "type": request_type, "title": title, "prompt": prompt,
        "options": options or [], "default": default,
    })
```

### Request types

| Type | Use for | Notes |
| :--- | :--- | :--- |
| `"secret"` | API keys, passwords, tokens | Hosts must mask input; **never echo the value into `output`** — mask it (`abcd...wxyz`). |
| `"text"` | Clarifications, descriptions, free-form input | Provide a `default` when the answer is optional. |
| `"select"` | Choosing among known options | Pass `options` list; the handler returns the chosen string. |

**Cancellation rule**: `None` means the user refused. Always treat it as a clean abort: `{"success": False, "error": "Cancelled by user."}` — never guess a default for secrets.

---

## 6. Complete Worked Use Case: Secure Service Configuration

**Scenario**: The LLM needs to connect a user to a paid weather API. It lacks the API key (secret), doesn't know the region (choice), and must not make outbound calls without permission (privilege). One tool orchestrates all three interactions.

This is the shipped example library: `src/lollms_client/tools_bindings/lcp/default_tools/user_interactive_tools/user_interactive_tools.py`. Its two tools:

1. **`tool_configure_service_connection(service_name)`** — asks for the API key (`secret`), the region (`select` from `["eu-west", "us-east", "us-west", "ap-south"]`), and an optional description (`text`); saves the connection to `service_connections.json` in the workspace with a masked confirmation.
2. **`tool_test_service_connection(service_name)`** — requests privilege confirmation via the confirm handler before performing the outbound HTTP test, then reports the HTTP status.

A typical LLM conversation using it:

> **User**: "Set up my weather API access and make sure it works."
> **LLM**: calls `tool_configure_service_connection("weather_api")` → the GUI/CLI prompts the user for the key, region, and description → tool saves and returns a masked confirmation → **LLM**: calls `tool_test_service_connection("weather_api")` → the host asks "Allow a test network request for service 'weather_api'?" → user allows → tool reports `HTTP 200` → **LLM**: "Your weather API is configured (region eu-west) and reachable."

Note how the LLM never sees the secret: the key flows user → handler → tool → workspace file, and only a masked form reaches the model's context.

---

## 7. Host Wiring (CLI and GUI Handlers)

The host passes the handler through `tools_binding_config` (or calls `binding.set_user_interaction_handler(...)` at any time):

```python
client = LollmsClient(
    tools_binding_name="lcp",
    tools_binding_config={
        "tools_folders": ["./my_tools"],
        "user_interaction_handler": my_handler,
        "confirm_handler": my_confirm_handler,
    },
)
```

### CLI handler (terminal)

```python
import getpass

def cli_user_interaction_handler(request: dict):
    kind = request.get("type", "text")
    print("\n" + "=" * 60)
    print(f"🙋  {request.get('title', 'Input required')}")
    print("=" * 60)
    print(request.get("prompt", ""))

    if kind == "select":
        options = request.get("options", [])
        for i, opt in enumerate(options, start=1):
            print(f"  {i}. {opt}")
        while True:
            try:
                raw = input(f"Choose 1-{len(options)}: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
            if raw in options:
                return raw
            print("Invalid selection.")

    try:
        if kind == "secret":
            return getpass.getpass("> ") or None
        answer = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return answer or request.get("default") or None
```

### GUI handler (event-queue bridge)

In an event-driven UI, the tool runs on a background thread and must NOT touch the UI directly. Dispatch to the UI thread and block on a response queue — the exact same pattern as the GUI confirm handler:

```python
import queue

def make_gui_user_interaction_handler(event_queue: queue.Queue):
    def _handler(request: dict):
        response_queue = queue.Queue(maxsize=1)
        event_queue.put({
            "type": "user_interaction_request",
            "request": request,
            "response_queue": response_queue,
        })
        try:
            return response_queue.get(timeout=600)   # long timeout: humans are slow
        except Exception:
            return None
    return _handler
```

The UI thread listens on `event_queue`, renders a dialog (password field for `"secret"`, dropdown for `"select"`, text box otherwise) and puts the answer (or `None`) into `response_queue`.

---

## 8. Pre-Flight Checklist

Before shipping an LCP tool, verify:

- [ ] Function name starts with `tool_`; helpers don't.
- [ ] First docstring paragraph tells the LLM **when to use the tool**.
- [ ] Every parameter documented in `Args:` with `name (type): description`.
- [ ] Numeric/bool params coerce string inputs defensively.
- [ ] Returns `{"success": ..., "output"/"error": ...}` in every branch, including cancellation.
- [ ] Sensitive steps gated behind `_CONFIRM_HANDLER` with a clear script label.
- [ ] User questions go through `_ask_user`; `None` answers abort cleanly.
- [ ] Secrets never appear in `output`, `prompt_injection`, or logs — mask them.
- [ ] File access stays inside the workspace (resolve paths against CWD, reject escapes).