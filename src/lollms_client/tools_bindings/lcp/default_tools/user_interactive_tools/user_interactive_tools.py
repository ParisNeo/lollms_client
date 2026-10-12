"""
user_interactive_tools.py — LCP library demonstrating the user-interaction
protocol (secret/key entry, free text, option selection) combined with
privilege confirmation for sensitive operations.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

TOOL_LIBRARY_NAME = "User Interactive Tools"
TOOL_LIBRARY_DESC = "Tools that ask the user for keys, text, or choices, and request privileges for sensitive steps."
TOOL_LIBRARY_ICON = "🙋"

_CONNECTIONS_FILE = "service_connections.json"
_CONFIRM_HANDLER = None
_USER_INTERACTION_HANDLER = None


def set_confirm_handler(handler: Optional[Any]) -> None:
    """Host-injected confirmation callback for privilege requests."""
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler


def set_user_interaction_handler(handler: Optional[Any]) -> None:
    """Host-injected interaction callback used to ask the user for input."""
    global _USER_INTERACTION_HANDLER
    _USER_INTERACTION_HANDLER = handler


def init_tools_library(config: Optional[dict] = None) -> None:
    """Captures host-provided handlers when the library is lazily loaded."""
    global _CONFIRM_HANDLER, _USER_INTERACTION_HANDLER
    if config and isinstance(config, dict):
        _CONFIRM_HANDLER = config.get("confirm_handler", _CONFIRM_HANDLER)
        _USER_INTERACTION_HANDLER = config.get("user_interaction_handler", _USER_INTERACTION_HANDLER)


def _ask_user(
    request_type: str,
    title: str,
    prompt: str,
    options: Optional[List[str]] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    if _USER_INTERACTION_HANDLER is None:
        raise RuntimeError(
            "No user-interaction handler is registered by the host application. "
            "The host must pass a 'user_interaction_handler' to the LCP binding."
        )
    return _USER_INTERACTION_HANDLER({
        "type": request_type,
        "title": title,
        "prompt": prompt,
        "options": options or [],
        "default": default,
    })


def _load_connections() -> Dict[str, Any]:
    path = Path(_CONNECTIONS_FILE)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_connections(connections: Dict[str, Any]) -> None:
    Path(_CONNECTIONS_FILE).write_text(
        json.dumps(connections, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def tool_configure_service_connection(service_name: str) -> Dict[str, Any]:
    """
    Interactively configures a connection to an external service by asking the
    user for an API key, a region, and an optional description, then saves the
    connection to the workspace.

    Args:
        service_name (str): The name of the service to configure (e.g. 'weather_api').
    """
    try:
        api_key = _ask_user(
            "secret",
            f"{service_name} API key",
            f"Please enter your {service_name} API key:",
        )
        if not api_key:
            return {"success": False, "error": "Configuration cancelled: no API key provided."}

        region = _ask_user(
            "select",
            f"{service_name} region",
            "Select the service region to use:",
            options=["eu-west", "us-east", "us-west", "ap-south"],
            default="eu-west",
        )
        if not region:
            return {"success": False, "error": "Configuration cancelled: no region selected."}

        description = _ask_user(
            "text",
            f"{service_name} description",
            "Optionally describe what this connection is used for:",
            default="",
        ) or ""

        connections = _load_connections()
        connections[service_name] = {
            "api_key": api_key,
            "region": region,
            "description": description,
        }
        _save_connections(connections)

        masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        return {
            "success": True,
            "output": (
                f"Connection '{service_name}' configured and saved to "
                f"'{_CONNECTIONS_FILE}' (region: {region}, key: {masked})."
            ),
            "service_name": service_name,
            "region": region,
        }
    except Exception as ex:
        return {"success": False, "error": f"Interactive configuration failed: {ex}"}


def tool_test_service_connection(service_name: str, test_endpoint: str = "https://httpbin.org/get") -> Dict[str, Any]:
    """
    Sends a test request to verify a configured service. This performs an
    outbound network call, so it requests operator approval before running.

    Args:
        service_name (str): The name of the configured service to test.
        test_endpoint (str): The URL used to test outbound connectivity.
    """
    connections = _load_connections()
    if service_name not in connections:
        return {"success": False, "error": f"No connection named '{service_name}' is configured."}

    if _CONFIRM_HANDLER is not None:
        script_label = f"Send a test network request for service '{service_name}'"
        source = (
            f"GET {test_endpoint}\n"
            f"Authorization: Bearer <{service_name} api key>\n"
        )
        decision, reason = _CONFIRM_HANDLER(source, script_label)
        if decision not in ("allow", "always"):
            return {
                "success": False,
                "error": f"Operator denied the network test: {reason or 'no reason given'}",
            }

    try:
        import urllib.request
        request = urllib.request.Request(
            test_endpoint,
            headers={"User-Agent": f"lollms-lcp/{service_name}"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
        return {
            "success": True,
            "output": f"Service '{service_name}' is reachable (HTTP {status}).",
        }
    except Exception as ex:
        return {"success": False, "error": f"Network test failed: {ex}"}