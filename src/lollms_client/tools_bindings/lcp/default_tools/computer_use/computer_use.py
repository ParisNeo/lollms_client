"""
computer_use.py
LCP toolset for desktop automation (GUI control via the "Computer Use" pattern).

The active model must support vision: the agent takes a screenshot, decides
where to act, and verifies the result. Visual grounding is performed by the
ACTIVE vision-capable LLM binding itself — no external OCR dependency.

Workflow:
  1. tool_computer_desktop_info  → learn the screen geometry.
  2. tool_computer_screenshot    → see the current state of the screen.
  3. tool_computer_click / type / key / scroll / move_cursor → act on it.
  4. Re-screenshot to verify the effect of the action.

Coordinate system: absolute pixels with the origin at the top-left corner of
the PRIMARY monitor (unless the backend exposes a different one).
"""

import base64
import io
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Computer Use"
TOOL_LIBRARY_DESC = (
    "Desktop automation toolset. Take screenshots, move the mouse, click, "
    "type text, press keys, and scroll. Requires the active model to have "
    "vision capability: element locations are grounded visually by asking "
    "the vision model to return pixel coordinates from the screenshot."
)
TOOL_LIBRARY_ICON = "🖥️"

_MAX_SCREENSHOT_BASE64_CHARS = 6_000_000
_GROUNDING_MAX_RETRIES = 2
_VLM_N_PREDICT = 600


def init_tools_library(config: dict = None) -> None:
    return None


# ─────────────────────────── Backend abstraction ────────────────────────────

_BACKEND: Optional[Any] = None
_BACKEND_ERROR: Optional[str] = None


def _resolve_backend():
    """
    Lazily resolves the desktop automation backend.
    Prefers pyautogui (cross-platform); falls back to mss for read-only
    screenshot capability. Returns (backend_module, error_str).
    """
    global _BACKEND, _BACKEND_ERROR
    if _BACKEND is not None or _BACKEND_ERROR is not None:
        return _BACKEND, _BACKEND_ERROR

    try:
        import pyautogui  # type: ignore
        try:
            pyautogui.FAILSAFE = False
        except Exception:
            pass
        try:
            pyautogui.PAUSE = 0.05
        except Exception:
            pass
        _BACKEND = pyautogui
        return _BACKEND, None
    except ImportError as import_err:
        _BACKEND_ERROR = (
            f"pyautogui is not installed ({import_err}). Install it with "
            f"`pip install pyautogui` to enable computer use tools."
        )
        ASCIIColors.warning(f"[computer_use] Backend unavailable: {_BACKEND_ERROR}")
        return None, _BACKEND_ERROR


def _require_backend():
    """
    Returns a ready backend module or raises RuntimeError with a
    user-actionable message.
    """
    backend, error = _resolve_backend()
    if backend is None:
        raise RuntimeError(error or "No desktop automation backend available.")
    return backend


def _screen_size() -> Tuple[int, int]:
    backend = _require_backend()
    width, height = backend.size()
    return int(width), int(height)


def _clamp_coordinates(x: int, y: int) -> Tuple[int, int]:
    """
    Clamps pixel coordinates to the visible screen area. Vision models
    frequently hallucinate out-of-bounds values; clamping prevents
    off-screen clicks and keeps the feedback loop honest.
    """
    width, height = _screen_size()
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    return x, y


# ─────────────────────── Vision grounding (zero-OCR) ────────────────────────

def _resolve_vision_binding(lollms_client_instance: Any) -> Optional[Any]:
    """
    Resolves a vision-capable LLM binding from the client.
    Handles SmartRouter-style multi-binding clients by scanning children.
    """
    if not lollms_client_instance:
        return None

    active_llm = getattr(lollms_client_instance, "llm", None)
    if not active_llm:
        return None

    if getattr(active_llm, "vision_enabled", False):
        return active_llm

    child_bindings = getattr(active_llm, "child_bindings", None)
    if child_bindings:
        for binding in child_bindings.values():
            if getattr(binding, "vision_enabled", False):
                return binding

    return None


_COORDINATE_RE = re.compile(r'\{\s*"x"\s*:\s*(\d+)\s*,\s*"y"\s*:\s*(\d+)\s*\}')


def _extract_coordinates(text: str) -> Optional[Tuple[int, int]]:
    """
    Extracts the first {"x": int, "y": int} JSON object from a VLM response.
    Tolerant of code fences and surrounding prose.
    """
    if not text:
        return None
    match = _COORDINATE_RE.search(text)
    if match:
        return int(match.group(1)), int(match.group(2))
    bare = re.search(r'\((\d+)\s*,\s*(\d+)\)', text)
    if bare:
        return int(bare.group(1)), int(bare.group(2))
    return None


def _ground_element_coordinates(
    lollms_client_instance: Any,
    screenshot_b64: str,
    element_description: str,
    screen_width: int,
    screen_height: int,
) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    """
    Asks the vision model to locate an element on the screenshot and return
    its pixel coordinates. Returns ((x, y), error) — exactly one is set.
    """
    vision_binding = _resolve_vision_binding(lollms_client_instance)
    if vision_binding is None:
        return None, (
            "No vision-capable LLM binding is available. Computer use tools "
            "require a vision model to locate elements on screen."
        )

    grounding_prompt = (
        "You are a visual grounding module inside a desktop automation agent.\n"
        f"The attached screenshot is {screen_width} pixels wide and {screen_height} "
        "pixels tall. Coordinates are absolute pixels from the TOP-LEFT corner.\n"
        f"Locate: {element_description}\n\n"
        "Reply with ONLY a JSON object of the clickable center point, nothing else:\n"
        '{"x": <int>, "y": <int>}'
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"{grounding_prompt}\n\n"
                "Screenshot is attached as the image for this message."
            ),
        }
    ]

    last_error: Optional[str] = None
    for _attempt in range(_GROUNDING_MAX_RETRIES):
        try:
            response = vision_binding.generate_from_messages(
                messages=messages,
                images=[screenshot_b64],
                temperature=0.0,
                stream=False,
                n_predict=_VLM_N_PREDICT,
            )
        except Exception as gen_err:
            last_error = f"Vision grounding generation failed: {gen_err}"
            continue

        if isinstance(response, dict) and response.get("error"):
            last_error = f"Vision grounding error: {response['error']}"
            continue

        text = str(response).strip() if response is not None else ""
        coords = _extract_coordinates(text)
        if coords is None:
            last_error = (
                f"Vision model did not return parseable coordinates. Raw reply: "
                f"{text[:200] or '(empty)'}"
            )
            continue

        x, y = coords
        if not (0 <= x < screen_width and 0 <= y < screen_height):
            x, y = _clamp_coordinates(x, y)

        return (x, y), None

    return None, last_error or "Vision grounding failed for an unknown reason."


def _vision_capability_ready(lollms_client_instance: Any) -> bool:
    return _resolve_vision_binding(lollms_client_instance) is not None


# ───────────────────────────── Screenshot helpers ──────────────────────────

def _capture_screenshot_b64() -> str:
    """
    Captures the full primary screen and returns a PNG base64 string.
    """
    backend = _require_backend()
    try:
        pil_image = backend.screenshot()
    except Exception as shot_err:
        raise RuntimeError(f"Screenshot capture failed: {shot_err}")

    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _sanitize_error(text: str) -> str:
    """
    Strips absolute host paths from error strings to preserve sandbox opacity.
    """
    return re.sub(r'[A-Za-z]:\\(?:Users|home)[\\/][^\s"\']*', "<host-path>", text or "")


def _describe_screenshots_for_history(discussion_instance: Any, count: int) -> None:
    """
    No-op hook kept for interface stability: screenshot ingestion into the
    discussion artifact store is handled by the ChatMixin workspace sync when
    the screenshot is saved to the workspace directory.
    """
    return None


# ──────────────────────────────── Tools ─────────────────────────────────────

def tool_computer_desktop_info(
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Returns the geometry of the primary screen (width and height in pixels).

    Call this FIRST to learn the coordinate space before clicking or typing.
    Coordinates for all computer use tools are absolute pixels measured from
    the top-left corner of the screen.
    """
    try:
        width, height = _screen_size()
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as gen_err:
        return {"success": False, "error": _sanitize_error(str(gen_err))}

    return {
        "success": True,
        "output": (
            f"Primary screen geometry: {width}x{height} pixels. "
            "Origin (0,0) is the top-left corner. All computer use coordinates "
            "are absolute pixels within this range."
        ),
        "screen_width": width,
        "screen_height": height,
    }


def tool_computer_screenshot(
    save_to_workspace: bool = True,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Captures a screenshot of the screen and returns it as a base64 image for visual inspection.

    Use this to SEE the current state of the desktop before acting, and again
    after actions to verify their effect. The screenshot is also saved to the
    workspace as a PNG artifact when save_to_workspace is true.

    Args:
        save_to_workspace (bool): When true, persists the screenshot as a PNG file in the workspace. Default true.
    """
    try:
        screenshot_b64 = _capture_screenshot_b64()
    except RuntimeError as capture_err:
        return {"success": False, "error": _sanitize_error(str(capture_err))}
    except Exception as gen_err:
        return {"success": False, "error": _sanitize_error(str(gen_err))}

    if len(screenshot_b64) > _MAX_SCREENSHOT_BASE64_CHARS:
        return {
            "success": False,
            "error": (
                f"Screenshot payload too large ({len(screenshot_b64)} base64 chars). "
                "Reduce screen resolution or color depth."
            ),
        }

    width, height = _screen_size()

    workspace_file: Optional[str] = None
    if save_to_workspace:
        try:
            from pathlib import Path
            import uuid as _uuid

            workspace_root = Path.cwd()
            file_name = f"screenshot_{int(time.time())}_{_uuid.uuid4().hex[:6]}.png"
            target_path = workspace_root / file_name
            (target_path).write_bytes(base64.b64decode(screenshot_b64))
            workspace_file = file_name
        except Exception as persist_err:
            ASCIIColors.warning(
                f"[computer_use] Screenshot workspace persistence failed: {persist_err}"
            )

    result: Dict[str, Any] = {
        "success": True,
        "output": (
            f"Screenshot captured ({width}x{height})."
            + (f" Saved to workspace as '{workspace_file}'." if workspace_file else "")
            + " The image is attached for your inspection; describe what you see "
            "and plan your next action."
        ),
        "image_b64": screenshot_b64,
        "screen_width": width,
        "screen_height": height,
    }
    if workspace_file:
        result["screenshot_file"] = workspace_file
    return result


def tool_computer_click(
    x: int,
    y: int,
    button: str = "left",
    clicks: int = 1,
    description: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Clicks at the given pixel coordinates on the screen.

    If you do not know the exact coordinates, provide a natural language
    'description' of the target (e.g. "the Chrome address bar") and the system
    will locate it visually via the vision model. Either x/y or description
    must be provided.

    Args:
        x (int): X pixel coordinate. Ignored when description is used. Pass 0 when unused.
        y (int): Y pixel coordinate. Ignored when description is used. Pass 0 when unused.
        button (str): Mouse button: 'left', 'right', or 'middle'. Default 'left'.
        clicks (int): Number of clicks (1 = single, 2 = double). Default 1.
        description (str): Optional natural-language description of the element to click, used for visual grounding when x/y are unknown.
    """
    button = str(button or "left").strip().lower()
    if button not in ("left", "right", "middle"):
        return {"success": False, "error": f"Invalid button '{button}'. Use 'left', 'right', or 'middle'."}

    try:
        clicks = int(clicks)
    except (TypeError, ValueError):
        return {"success": False, "error": f"Invalid clicks value '{clicks}'. Provide an integer."}
    if not (1 <= clicks <= 3):
        return {"success": False, "error": f"Invalid clicks count {clicks}. Use 1, 2, or 3."}

    if description and description.strip():
        if not _vision_capability_ready(lollms_client_instance):
            return {
                "success": False,
                "error": (
                    "Visual grounding requires a vision-capable model, which is not "
                    "available. Provide explicit x/y coordinates instead."
                ),
            }
        try:
            screenshot_b64 = _capture_screenshot_b64()
            width, height = _screen_size()
        except Exception as capture_err:
            return {"success": False, "error": _sanitize_error(str(capture_err))}

        coords, grounding_error = _ground_element_coordinates(
            lollms_client_instance, screenshot_b64, description.strip(), width, height
        )
        if coords is None:
            return {
                "success": False,
                "error": f"Visual grounding failed: {grounding_error}",
            }
        x, y = coords

    try:
        if x is None or y is None:
            return {"success": False, "error": "Coordinates x and y are required when no description is given."}
        x, y = _clamp_coordinates(int(x), int(y))
    except (TypeError, ValueError):
        return {"success": False, "error": "Coordinates x and y must be integers."}

    try:
        backend = _require_backend()
        backend.click(x=x, y=y, clicks=clicks, button=button)
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as click_err:
        return {"success": False, "error": _sanitize_error(f"Click failed: {click_err}")}

    return {
        "success": True,
        "output": f"Clicked {button} button {clicks}x at ({x}, {y}).",
        "x": x,
        "y": y,
    }


def tool_computer_move_cursor(
    x: int,
    y: int,
    description: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Moves the mouse cursor to the given pixel coordinates without clicking.

    Args:
        x (int): X pixel coordinate. Pass 0 when using description.
        y (int): Y pixel coordinate. Pass 0 when using description.
        description (str): Optional natural-language description of the target element for visual grounding.
    """
    if description and description.strip():
        if not _vision_capability_ready(lollms_client_instance):
            return {
                "success": False,
                "error": (
                    "Visual grounding requires a vision-capable model, which is not "
                    "available. Provide explicit x/y coordinates instead."
                ),
            }
        try:
            screenshot_b64 = _capture_screenshot_b64()
            width, height = _screen_size()
        except Exception as capture_err:
            return {"success": False, "error": _sanitize_error(str(capture_err))}

        coords, grounding_error = _ground_element_coordinates(
            lollms_client_instance, screenshot_b64, description.strip(), width, height
        )
        if coords is None:
            return {"success": False, "error": f"Visual grounding failed: {grounding_error}"}
        x, y = coords

    try:
        if x is None or y is None:
            return {"success": False, "error": "Coordinates x and y are required when no description is given."}
        x, y = _clamp_coordinates(int(x), int(y))
    except (TypeError, ValueError):
        return {"success": False, "error": "Coordinates x and y must be integers."}

    try:
        backend = _require_backend()
        backend.moveTo(x=x, y=y)
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as move_err:
        return {"success": False, "error": _sanitize_error(f"Cursor move failed: {move_err}")}

    return {
        "success": True,
        "output": f"Moved cursor to ({x}, {y}).",
        "x": x,
        "y": y,
    }


def tool_computer_type(
    text: str,
    interval_ms: int = 20,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Types text at the current cursor position (into the currently focused element).

    Click into an input field first (tool_computer_click), then call this tool.

    Args:
        text (str): The text to type. Required.
        interval_ms (int): Delay between keystrokes in milliseconds. Default 20.
    """
    if not isinstance(text, str) or not text:
        return {"success": False, "error": "The 'text' parameter is required and must be a non-empty string."}

    try:
        interval_s = max(0.0, min(float(interval_ms) / 1000.0, 1.0))
    except (TypeError, ValueError):
        interval_s = 0.02

    try:
        backend = _require_backend()
        backend.typewrite(text, interval=interval_s)
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as type_err:
        return {"success": False, "error": _sanitize_error(f"Typing failed: {type_err}")}

    return {
        "success": True,
        "output": f"Typed {len(text)} characters.",
        "typed_chars": len(text),
    }


def tool_computer_key(
    key: str,
    presses: int = 1,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Presses a keyboard key or a combination (e.g. 'enter', 'ctrl+c', 'alt+f4', 'win', 'tab').

    Use '+' to combine modifiers. Recognized modifiers: ctrl, alt, shift, win/cmd.
    Common keys: enter, tab, escape, backspace, delete, space, up, down, left, right,
    home, end, pageup, pagedown, f1..f12, a..z, 0..9.

    Args:
        key (str): The key or key combination to press. Required.
        presses (int): Number of presses. Default 1, max 3.
    """
    if not isinstance(key, str) or not key.strip():
        return {"success": False, "error": "The 'key' parameter is required."}

    key_normalized = key.strip().lower()
    if len(key_normalized) > 20:
        return {"success": False, "error": "Key combination too long (max 20 characters)."}

    valid_token_re = re.compile(r'^[a-z0-9+]+$')
    if not valid_token_re.match(key_normalized):
        return {"success": False, "error": f"Invalid key specification '{key}'. Use tokens like 'enter', 'ctrl+c'."}

    try:
        presses = int(presses)
    except (TypeError, ValueError):
        return {"success": False, "error": f"Invalid presses value '{presses}'."}
    if not (1 <= presses <= 3):
        return {"success": False, "error": f"Invalid presses count {presses}. Use 1 to 3."}

    hotkey_parts = [part.strip() for part in key_normalized.split("+") if part.strip()]
    if not hotkey_parts:
        return {"success": False, "error": "Empty key combination."}

    try:
        backend = _require_backend()
        if len(hotkey_parts) == 1:
            for _ in range(presses):
                backend.press(hotkey_parts[0])
        else:
            for _ in range(presses):
                backend.hotkey(*hotkey_parts)
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as key_err:
        return {"success": False, "error": _sanitize_error(f"Key press failed: {key_err}")}

    return {
        "success": True,
        "output": f"Pressed '{key_normalized}' {presses} time(s).",
    }


def tool_computer_scroll(
    amount: int = 300,
    direction: str = "down",
    x: int = -1,
    y: int = -1,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Scrolls the mouse wheel at the current cursor position (or at given coordinates).

    Args:
        amount (int): Number of scroll units. PyAutoGUI units: positive values scroll. Default 300.
        direction (str): 'down' or 'up'. Default 'down'.
        x (int): Optional X coordinate to scroll at. -1 means current cursor position.
        y (int): Optional Y coordinate to scroll at. -1 means current cursor position.
    """
    direction = str(direction or "down").strip().lower()
    if direction not in ("up", "down"):
        return {"success": False, "error": f"Invalid direction '{direction}'. Use 'up' or 'down'."}

    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return {"success": False, "error": f"Invalid amount '{amount}'. Provide an integer."}
    if not (1 <= abs(amount) <= 2000):
        return {"success": False, "error": f"Scroll amount {amount} out of range (1-2000)."}

    effective_amount = amount if direction == "down" else -amount

    try:
        backend = _require_backend()
        if x is not None and x >= 0 and y is not None and y >= 0:
            cx, cy = _clamp_coordinates(int(x), int(y))
            backend.scroll(effective_amount, x=cx, y=cy)
        else:
            backend.scroll(effective_amount)
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as scroll_err:
        return {"success": False, "error": _sanitize_error(f"Scroll failed: {scroll_err}")}

    return {
        "success": True,
        "output": f"Scrolled {direction} by {amount} units.",
    }