"""
computer_use.py
LCP toolset for desktop automation (GUI control via the "Computer Use" pattern).

The active model must support vision: the agent takes a screenshot, decides
where to act, and verifies the result. Visual grounding is performed by the
ACTIVE vision-capable LLM binding itself — no external OCR dependency.

Workflow:
  1. tool_computer_desktop_info  → learn the screen geometry.
  2. tool_computer_screenshot    → see the current state of the screen.
  3. tool_computer_click / type / key / scroll / move_cursor / drag → act on it.
  4. tool_computer_wait          → allow UI animations or page transitions to settle.
  5. Re-screenshot to verify the effect of the action.

Coordinate system: absolute pixels with the origin at the top-left corner of
the PRIMARY monitor (unless the backend exposes a different one).
"""

import base64
import io
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Computer Use"
TOOL_LIBRARY_DESC = (
    "Desktop automation toolset. Take screenshots, move the mouse, click, "
    "drag, type text, press keys, and scroll. Requires the active model to have "
    "vision capability: element locations are grounded visually by asking "
    "the vision model to return pixel coordinates from the screenshot."
)
TOOL_LIBRARY_ICON = "🖥️"

_MAX_SCREENSHOT_BASE64_CHARS = 6_000_000
_GROUNDING_MAX_RETRIES = 2
_VLM_N_PREDICT = 600
_GRID_STEP = 100
_POST_ACTION_SETTLE_S = 0.4


def init_tools_library(config: Optional[dict] = None) -> None:
    """Auto-verifies that desktop automation packages are available."""
    try:
        import pipmaster as pm
        pm.ensure_packages(["pyautogui", "pillow"])
    except Exception as e:
        ASCIIColors.warning(f"[computer_use] Could not auto-install dependencies via pipmaster: {e}")


# ─────────────────────────── Backend abstraction ────────────────────────────

_BACKEND: Optional[Any] = None
_BACKEND_ERROR: Optional[str] = None


def _resolve_backend():
    """
    Lazily resolves the desktop automation backend.
    Prefers pyautogui (cross-platform). Returns (backend_module, error_str).
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
        pyautogui.size()
        _BACKEND = pyautogui
        return _BACKEND, None
    except Exception as import_err:
        _BACKEND_ERROR = (
            f"pyautogui backend failed to initialize ({type(import_err).__name__}: {import_err}). "
            f"If the message mentions NumPy/_ARRAY_API or cv2, run "
            f"`pip install --upgrade opencv-python` (or `pip install \"numpy<2\"`). "
            f"Otherwise install with `pip install pyautogui pillow`."
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
    Handles direct vision, SmartRouter-style child bindings, and fallback VLMs.
    """
    if not lollms_client_instance:
        return None

    if hasattr(lollms_client_instance, "has_vision_capability") and lollms_client_instance.has_vision_capability():
        active_llm = getattr(lollms_client_instance, "llm", None)
        if active_llm:
            return active_llm

    active_llm = getattr(lollms_client_instance, "llm", None)
    if active_llm and getattr(active_llm, "vision_enabled", False):
        return active_llm

    child_bindings = getattr(active_llm, "child_bindings", None)
    if child_bindings:
        for binding in child_bindings.values():
            if getattr(binding, "vision_enabled", False):
                return binding

    if hasattr(lollms_client_instance, "find_available_vlm"):
        vlm = lollms_client_instance.find_available_vlm()
        if vlm:
            return vlm

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


_NORMALIZED_SCALE_MARKERS = (
    "0-1000", "0 to 1000", "/1000", "normalized", "normalize", "scale 0", "1000x1000",
)

_NORMALIZED_SCALE_VALUES = {"normalized", "0-1000", "0..1000", "1000", "relative", "percent"}


def _rescale_from_1000(x: int, y: int, screen_width: int, screen_height: int) -> Tuple[int, int]:
    """Converts Qwen-style 0-1000 normalized grid coordinates to absolute pixels."""
    scaled_x = int(round((x / 1000.0) * screen_width))
    scaled_y = int(round((y / 1000.0) * screen_height))
    return _clamp_coordinates(scaled_x, scaled_y)


def _extract_coordinates_with_scale(
    text: str,
    screen_width: int,
    screen_height: int,
) -> Optional[Tuple[int, int]]:
    """
    Extracts coordinates from a VLM reply, honoring an explicit coordinate-scale
    declaration. Some vision models (Qwen-VL family) emit coordinates on a
    normalized 0-1000 grid instead of absolute pixels; a wrong interpretation
    clicks the wrong pixel. Only an EXPLICIT declaration triggers rescaling —
    either a "scale" field in the JSON reply or a normalized-scale marker in
    the raw text. Absent a declaration, coordinates are treated as absolute.
    """
    if not text:
        return None

    json_blob_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_blob_match:
        try:
            parsed = json.loads(json_blob_match.group(0))
            if isinstance(parsed, dict) and "x" in parsed and "y" in parsed:
                x = int(parsed["x"])
                y = int(parsed["y"])
                declared_scale = str(parsed.get("scale", "") or "").strip().lower()
                if declared_scale in _NORMALIZED_SCALE_VALUES:
                    return _rescale_from_1000(x, y, screen_width, screen_height)
                return x, y
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    coords = _extract_coordinates(text)
    if coords is None:
        return None

    text_lower = text.lower()
    if any(marker in text_lower for marker in _NORMALIZED_SCALE_MARKERS):
        return _rescale_from_1000(coords[0], coords[1], screen_width, screen_height)
    return coords


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
        "pixels tall. Coordinates are ABSOLUTE pixels measured from the TOP-LEFT "
        "corner of the image: (0, 0) is the top-left pixel and "
        f"({screen_width - 1}, {screen_height - 1}) is the bottom-right pixel.\n"
        "Do NOT output normalized 0-1000 grid coordinates — raw pixel values only.\n"
        f"Locate: {element_description}\n\n"
        "Reply with ONLY a JSON object of the clickable center point, nothing else:\n"
        '{"x": <int>, "y": <int>, "scale": "absolute"}\n'
        'The "scale" field must be the literal string "absolute".'
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
        coords = _extract_coordinates_with_scale(text, screen_width, screen_height)
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
    """Captures the full primary screen and returns a PNG base64 string."""
    backend = _require_backend()
    try:
        pil_image = backend.screenshot()
    except Exception as shot_err:
        raise RuntimeError(f"Screenshot capture failed: {shot_err}")

    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _sanitize_error(text: str) -> str:
    """Strips absolute host paths from error strings to preserve sandbox opacity."""
    return re.sub(r'[A-Za-z]:\\(?:Users|home)[\\/][^\s"\']*', "<host-path>", text or "")


# ──────────────────────────────── Tools ─────────────────────────────────────

def tool_computer_desktop_info(
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Returns the geometry of the primary screen (width and height in pixels).

    Call this FIRST to learn the coordinate space before clicking, dragging, or typing.
    Coordinates for all computer use tools are absolute pixels measured from
    the top-left corner of the screen.
    """
    try:
        width, height = _screen_size()
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as gen_err:
        return {"success": False, "error": _sanitize_error(str(gen_err))}

    try:
        backend = _require_backend()
        cursor_x, cursor_y = int(backend.position()[0]), int(backend.position()[1])
        cursor_text = f" Cursor is currently at ({cursor_x}, {cursor_y})."
    except Exception:
        cursor_x, cursor_y = -1, -1
        cursor_text = ""

    return {
        "success": True,
        "output": (
            f"Primary screen geometry: {width}x{height} pixels. "
            "Origin (0,0) is the top-left corner. All computer use coordinates "
            "are absolute pixels within this range. "
            f"Every screenshot you receive overlays a cyan grid with lines every "
            f"{_GRID_STEP} pixels, labeled at their (x, y) intersections. "
            f"To locate an element: find the two nearest grid lines around it, "
            f"then estimate its position as the fraction of the "
            f"{_GRID_STEP}-px gap between them (e.g. halfway between the 600 and 700 "
            f"vertical lines is x≈650)."
            + cursor_text
        ),
        "screen_width": width,
        "screen_height": height,
        **({"cursor_x": cursor_x, "cursor_y": cursor_y} if cursor_x >= 0 else {}),
    }


def _sanitize_screenshot_filename(raw_name: str) -> str:
    """
    Reduces a raw file name to a single safe PNG filename confined to the
    workspace: strips directories, traversal segments, and unsafe characters.
    """
    cleaned = (raw_name or "").strip()
    cleaned = cleaned.replace("\\", "/")
    cleaned = cleaned.split("/")[-1]
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "", cleaned)
    cleaned = cleaned.strip(" .")
    if not cleaned or cleaned in {".", ".."}:
        return ""
    if not cleaned.lower().endswith(".png"):
        cleaned += ".png"
    return cleaned


def tool_computer_screenshot(
    save_to_workspace: bool = True,
    file_name: str = "",
    show_grid: bool = True,
    tool_context: Optional[dict] = None,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Captures a screenshot of the screen and returns it as a base64 image for visual inspection.

    Use this to SEE the current state of the desktop before acting, and again
    after actions to verify their effect. The screenshot is also saved to the
    workspace as a PNG artifact when save_to_workspace is true.

    When show_grid is true (default), the returned image overlays a coordinate
    system you must use for ALL subsequent computer-use actions:

    *   **Origin**: (0, 0) is the top-left corner of the primary monitor.
    *   **Grid**: A CYAN grid is drawn with lines every 100 pixels (both
        horizontal and vertical). Each intersection is labeled with its
        (x, y) coordinate in a small black chip.
    *   **Cursor**: The current mouse pointer is marked by a RED circle with a
        BLUE center dot. Its exact (x, y) is also reported in the text output.
    *   **Reading coordinates**: To target an element, (1) find the two nearest
        vertical grid lines (X1 < X2) around it, (2) find the two nearest
        horizontal grid lines (Y1 < Y2), (3) estimate the fraction of the gap
        the element sits in (e.g. 30 % from X1=600 to X2=700 gives x ≈ 630).
        State your estimated (x, y) before clicking.
    *   **Coordinate space**: All x, y values passed to click/move/drag/type
        tools are ABSOLUTE pixels within this grid. Do NOT use normalized
        0-1000 values.

    Args:
        save_to_workspace (bool): When true, persists the screenshot as a PNG file in the workspace. Default true.
        file_name (str): Optional custom PNG file name to KEEP this specific frame (e.g. 'photo.png'). When omitted, the default 'computer_use_screenshot.png' is used and overwritten on every recapture.
        show_grid (bool): When true, overlays a cyan coordinate grid with labeled intersections on the screenshot. Default true.
    """
    try:
        backend = _require_backend()
        cursor_x, cursor_y = int(backend.position()[0]), int(backend.position()[1])
    except Exception:
        cursor_x, cursor_y = -1, -1

    try:
        screenshot_b64, width, height = _capture_annotated_b64(
            (cursor_x, cursor_y) if cursor_x >= 0 else None,
            show_grid=show_grid,
        )
    except RuntimeError as capture_err:
        return {"success": False, "error": _sanitize_error(str(capture_err))}
    except Exception as gen_err:
        return {"success": False, "error": _sanitize_error(str(gen_err))}

    workspace_file: Optional[str] = None
    saved_file_name: Optional[str] = None
    if save_to_workspace:
        try:
            from pathlib import Path

            working_dir = Path.cwd()
            if isinstance(tool_context, dict):
                context_dir = tool_context.get("working_dir")
                if context_dir:
                    try:
                        candidate = Path(context_dir)
                        if candidate.is_dir():
                            working_dir = candidate.resolve()
                    except Exception:
                        pass

            saved_file_name = _sanitize_screenshot_filename(file_name) or "computer_use_screenshot.png"
            target_path = working_dir / saved_file_name
            target_path.write_bytes(base64.b64decode(screenshot_b64))
            workspace_file = str(target_path.resolve())
        except Exception as persist_err:
            ASCIIColors.warning(
                f"[computer_use] Screenshot workspace persistence failed: {persist_err}"
            )

    cursor_clause = (
        f" The mouse pointer is at ({cursor_x}, {cursor_y}), marked by the red circle with blue center."
        if cursor_x >= 0
        else ""
    )
    result: Dict[str, Any] = {
        "success": True,
        "output": (
            f"Screenshot captured ({width}x{height})."
            + (f" Saved to workspace as '{saved_file_name}'." if workspace_file else "")
            + " The image is attached for your inspection; describe what you see "
            "and plan your next action."
            + cursor_clause
        ),
        "image_b64": screenshot_b64,
        "screenshot_b64": screenshot_b64,
        "screen_width": width,
        "screen_height": height,
    }
    if cursor_x >= 0:
        result["cursor_x"] = cursor_x
        result["cursor_y"] = cursor_y
    if workspace_file:
        result["image_filename"] = workspace_file
        result["context_files"] = [workspace_file]
        result["screenshot_file"] = workspace_file
        grid_desc = (
            f"Overlay: a CYAN grid with lines every {_GRID_STEP} pixels; each "
            f"intersection is labeled at its (x, y) coordinate in a black chip. "
            f"To estimate a target's position: (1) find the two nearest vertical "
            f"grid lines — call them X1 and X2 — (2) find the two nearest horizontal "
            f"grid lines — Y1 and Y2 — (3) estimate the fraction of the gap the "
            f"target sits between them (e.g. 30% of the way from X1=600 to X2=700 "
            f"gives x≈630). State your estimated (x, y) before clicking. "
            f"⚠️ Do NOT assume the target is at the center of the image — the "
            f"center of the image is ({width // 2}, {height // 2}), which is NOT "
            f"necessarily where the element is. Read the grid labels."
            if show_grid
            else "No coordinate grid is overlaid. Use the full-screen dimensions "
            f"({width}x{height}) and estimate coordinates proportionally from the "
            "image content, or call tool_computer_screenshot with show_grid=true "
            "for a gridded frame."
        )
        result["prompt_injection"] = (
            f"\n\n✅ **Screenshot attached to your visual context** "
            f"({saved_file_name}). Describe what you observe and act on it.\n"
            f"📐 **COORDINATE CONVENTION**: The attached screenshot is EXACTLY "
            f"{width}x{height} pixels. All coordinates are ABSOLUTE pixels: "
            f"top-left corner is (0, 0), bottom-right corner is ({width - 1}, {height - 1}). "
            f"{grid_desc}"
            + (f" {cursor_clause.strip()}." if cursor_x >= 0 else "")
        )
    return result


def tool_computer_click(
    x: int = 0,
    y: int = 0,
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
        clicks (int): Number of clicks (1 = single, 2 = double, 3 = triple). Default 1.
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

    if not (description and description.strip()):
        try:
            x_val, y_val = int(x), int(y)
        except (TypeError, ValueError):
            return {"success": False, "error": "Coordinates x and y must be integers."}
        if x_val == 0 and y_val == 0:
            return {
                "success": False,
                "error": (
                    "Refusing to click (0, 0) — the top-left corner is almost certainly "
                    "not your intended target. Either pass explicit non-zero x/y coordinates "
                    "read from your last screenshot, or pass a natural-language 'description' "
                    "(e.g. description='the Windows search box in the taskbar') so the system "
                    "can visually ground the exact click point."
                ),
            }

    try:
        if x is None or y is None:
            return {"success": False, "error": "Coordinates x and y are required when no description is given."}
        x, y = _clamp_coordinates(int(x), int(y))
    except (TypeError, ValueError):
        return {"success": False, "error": "Coordinates x and y must be integers."}

    pre_click_cursor = None
    try:
        backend = _require_backend()
        pre_click_cursor = (int(backend.position()[0]), int(backend.position()[1]))
    except Exception:
        pass

    try:
        backend = _require_backend()
        backend.click(x=x, y=y, clicks=clicks, button=button)
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as click_err:
        return {"success": False, "error": _sanitize_error(f"Click failed: {click_err}")}

    try:
        width_geo, height_geo = _screen_size()
    except Exception:
        width_geo, height_geo = 0, 0

    pre_clause = ""
    if pre_click_cursor:
        pre_clause = (
            f" (cursor was at ({pre_click_cursor[0]}, {pre_click_cursor[1]}) "
            f"before the click)"
        )

    return _post_action_result(
        f"Clicked {button} button {clicks}x at ({x}, {y}){pre_clause}. "
        f"Screen: {width_geo}x{height_geo} px. "
        f"⚠️ VERIFY on the attached screenshot: the cursor should now be at "
        f"({x}, {y}) on the grid — if the red circle is NOT on the intended "
        f"element, re-estimate coordinates from the grid labels and click again.",
        {"x": x, "y": y},
    )


def tool_computer_move_cursor(
    x: int = 0,
    y: int = 0,
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

    if not (description and description.strip()):
        try:
            x_val, y_val = int(x), int(y)
        except (TypeError, ValueError):
            return {"success": False, "error": "Coordinates x and y must be integers."}
        if x_val == 0 and y_val == 0:
            return {
                "success": False,
                "error": (
                    "Refusing to move to (0, 0) — the top-left corner is almost certainly "
                    "not your intended target. Either pass explicit non-zero x/y coordinates "
                    "read from your last screenshot, or pass a natural-language 'description' "
                    "so the system can visually ground the destination."
                ),
            }

    try:
        if x is None or y is None:
            return {"success": False, "error": "Coordinates x and y are required when no description is given."}
        x, y = _clamp_coordinates(int(x), int(y))
    except (TypeError, ValueError):
        return {"success": False, "error": "Coordinates x and y must be integers."}

    try:
        backend = _require_backend()
        pre_move_cursor = (int(backend.position()[0]), int(backend.position()[1]))
        backend.moveTo(x=x, y=y)
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as move_err:
        return {"success": False, "error": _sanitize_error(f"Cursor move failed: {move_err}")}

    pre_clause = (
        f" (was at ({pre_move_cursor[0]}, {pre_move_cursor[1]}))"
        if pre_move_cursor else ""
    )
    return _post_action_result(
        f"Moved cursor to ({x}, {y}){pre_clause}. "
        f"⚠️ VERIFY: the red circle should now be at ({x}, {y}) on the grid — "
        f"confirm the cursor is where you intended before acting.",
        {"x": x, "y": y},
    )


def tool_computer_mouse_down(
    button: str = "left",
    x: int = -1,
    y: int = -1,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Presses and holds down a mouse button without releasing it.
    Useful for starting drag-and-drop, drawing, or text selection.

    Args:
        button (str): Mouse button to hold down: 'left', 'right', or 'middle'. Default 'left'.
        x (int): Optional X coordinate to position before pressing down. -1 means current cursor position.
        y (int): Optional Y coordinate to position before pressing down. -1 means current cursor position.
    """
    button = str(button or "left").strip().lower()
    if button not in ("left", "right", "middle"):
        return {"success": False, "error": f"Invalid button '{button}'. Use 'left', 'right', or 'middle'."}

    try:
        backend = _require_backend()
        if x >= 0 and y >= 0:
            cx, cy = _clamp_coordinates(int(x), int(y))
            backend.mouseDown(x=cx, y=cy, button=button)
            pos_desc = f"at ({cx}, {cy})"
        else:
            backend.mouseDown(button=button)
            pos_desc = "at current position"
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as err:
        return {"success": False, "error": _sanitize_error(f"Mouse down failed: {err}")}

    return _post_action_result(f"Held {button} mouse button down {pos_desc}.", {})


def tool_computer_mouse_up(
    button: str = "left",
    x: int = -1,
    y: int = -1,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Releases a held mouse button. Completes drag-and-drop or text selection.

    Args:
        button (str): Mouse button to release: 'left', 'right', or 'middle'. Default 'left'.
        x (int): Optional X coordinate to move to before releasing. -1 means current cursor position.
        y (int): Optional Y coordinate to move to before releasing. -1 means current cursor position.
    """
    button = str(button or "left").strip().lower()
    if button not in ("left", "right", "middle"):
        return {"success": False, "error": f"Invalid button '{button}'. Use 'left', 'right', or 'middle'."}

    try:
        backend = _require_backend()
        if x >= 0 and y >= 0:
            cx, cy = _clamp_coordinates(int(x), int(y))
            backend.mouseUp(x=cx, y=cy, button=button)
            pos_desc = f"at ({cx}, {cy})"
        else:
            backend.mouseUp(button=button)
            pos_desc = "at current position"
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as err:
        return {"success": False, "error": _sanitize_error(f"Mouse up failed: {err}")}

    return _post_action_result(f"Released {button} mouse button {pos_desc}.", {})


def tool_computer_drag(
    to_x: int,
    to_y: int,
    from_x: int = -1,
    from_y: int = -1,
    button: str = "left",
    duration_ms: int = 500,
    from_description: str = "",
    to_description: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Performs a click-and-drag from one point to another.

    Useful for moving windows, dragging icons, sliding sliders, or selecting text.
    Coordinates can be explicit pixels or grounded visually via descriptions.

    Args:
        to_x (int): Destination X pixel coordinate.
        to_y (int): Destination Y pixel coordinate.
        from_x (int): Optional start X pixel coordinate (-1 = current cursor position).
        from_y (int): Optional start Y pixel coordinate (-1 = current cursor position).
        button (str): Mouse button used to drag: 'left', 'right', or 'middle'. Default 'left'.
        duration_ms (int): Duration of the drag movement in milliseconds. Default 500.
        from_description (str): Optional description to visually ground start coordinates.
        to_description (str): Optional description to visually ground destination coordinates.
    """
    button = str(button or "left").strip().lower()
    if button not in ("left", "right", "middle"):
        return {"success": False, "error": f"Invalid button '{button}'. Use 'left', 'right', or 'middle'."}

    try:
        duration_s = max(0.05, min(float(duration_ms) / 1000.0, 5.0))
    except (TypeError, ValueError):
        duration_s = 0.5

    # Visual grounding if descriptions provided
    if from_description or to_description:
        if not _vision_capability_ready(lollms_client_instance):
            return {
                "success": False,
                "error": "Visual grounding requires a vision-capable model. Provide explicit coordinates.",
            }
        try:
            screenshot_b64 = _capture_screenshot_b64()
            width, height = _screen_size()
        except Exception as capture_err:
            return {"success": False, "error": _sanitize_error(str(capture_err))}

        if from_description:
            f_coords, f_err = _ground_element_coordinates(
                lollms_client_instance, screenshot_b64, from_description.strip(), width, height
            )
            if f_coords is None:
                return {"success": False, "error": f"Grounding start location failed: {f_err}"}
            from_x, from_y = f_coords

        if to_description:
            t_coords, t_err = _ground_element_coordinates(
                lollms_client_instance, screenshot_b64, to_description.strip(), width, height
            )
            if t_coords is None:
                return {"success": False, "error": f"Grounding destination location failed: {t_err}"}
            to_x, to_y = t_coords

    try:
        backend = _require_backend()
        target_to_x, target_to_y = _clamp_coordinates(int(to_x), int(to_y))

        if from_x >= 0 and from_y >= 0:
            target_from_x, target_from_y = _clamp_coordinates(int(from_x), int(from_y))
            backend.moveTo(target_from_x, target_from_y)

        backend.dragTo(target_to_x, target_to_y, duration=duration_s, button=button)
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as drag_err:
        return {"success": False, "error": _sanitize_error(f"Drag failed: {drag_err}")}

    return _post_action_result(
        f"Dragged {button} button to ({target_to_x}, {target_to_y}) over {duration_s:.2f}s.",
        {"to_x": target_to_x, "to_y": target_to_y},
    )


def tool_computer_type(
    text: str,
    interval_ms: int = 20,
    press_enter: bool = False,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Types text at the current cursor position (into the currently focused element).

    Click into an input field first (tool_computer_click), then call this tool.

    Args:
        text (str): The text to type. Required.
        interval_ms (int): Delay between keystrokes in milliseconds. Default 20.
        press_enter (bool): When true, presses Enter after typing. Default false.
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
        if press_enter:
            backend.press("enter")
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as type_err:
        return {"success": False, "error": _sanitize_error(f"Typing failed: {type_err}")}

    enter_note = " Enter pressed." if press_enter else ""
    return _post_action_result(
        f"Typed {len(text)} characters.{enter_note}",
        {"typed_chars": len(text)},
    )


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
    if len(key_normalized) > 30:
        return {"success": False, "error": "Key combination too long (max 30 characters)."}

    valid_token_re = re.compile(r'^[a-z0-9+_-]+$')
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

    return _post_action_result(
        f"Pressed '{key_normalized}' {presses} time(s).",
        {},
    )


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
        amount (int): Number of scroll units. Positive integer. Default 300.
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

    effective_amount = -abs(amount) if direction == "down" else abs(amount)

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

    return _post_action_result(
        f"Scrolled {direction} by {abs(amount)} units.",
        {},
    )


def tool_computer_wait(
    seconds: float = 1.0,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Pauses execution briefly to allow an application to open, a webpage to load, or an animation to complete.

    Args:
        seconds (float): Seconds to wait (clamped between 0.1 and 10.0 seconds). Default 1.0.
    """
    try:
        dur = max(0.1, min(float(seconds), 10.0))
    except (TypeError, ValueError):
        dur = 1.0

    time.sleep(dur)
    return _post_action_result(
        f"Waited {dur:.2f} seconds for the system/UI to settle.",
        {"seconds": dur},
    )


def tool_computer_cursor_position(
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Returns the current pixel coordinates of the mouse cursor.

    Call this to verify where the pointer is resting.
    """
    try:
        backend = _require_backend()
        cur_x, cur_y = backend.position()
        return {
            "success": True,
            "x": int(cur_x),
            "y": int(cur_y),
            "output": f"Current cursor position: ({int(cur_x)}, {int(cur_y)}).",
        }
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}
    except Exception as err:
        return {"success": False, "error": _sanitize_error(f"Position lookup failed: {err}")}


def _grid_label_font(size: int = 18):
    """
    Returns the best available font for grid labels: a TrueType font (DejaVu,
    Arial, Roboto, or the first TTF PIL can find) at the requested size,
    falling back to PIL's default bitmap font.
    """
    from PIL import ImageFont

    candidate_names = (
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "arial.ttf",
        "Arial.ttf",
        "Roboto-Regular.ttf",
    )
    for font_name in candidate_names:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    for directory in (
        r"C:\Windows\Fonts",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/System/Library/Fonts",
    ):
        try:
            import os
            if os.path.isdir(directory):
                for fn in os.listdir(directory):
                    if fn.lower().endswith((".ttf", ".ttc", ".otf")):
                        try:
                            return ImageFont.truetype(os.path.join(directory, fn), size)
                        except Exception:
                            continue
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        try:
            return ImageFont.load_default()
        except Exception:
            return None
    return None


def _annotate_screenshot(
    pil_image: Any,
    cursor_x: Optional[int] = None,
    cursor_y: Optional[int] = None,
    show_grid: bool = True,
) -> Any:
    """
    Draws a labeled coordinate grid (lines every _GRID_STEP pixels) and, when
    given, a red target circle with a blue center dot at the cursor position,
    so the vision model can cross-reference element locations and see exactly
    where the pointer is resting.

    Args:
        pil_image: The PIL image to annotate.
        cursor_x: X coordinate of the cursor (if available).
        cursor_y: Y coordinate of the cursor (if available).
        show_grid: When True, draws the cyan coordinate grid and labels.
    """
    from PIL import Image, ImageDraw

    annotated = pil_image.convert("RGB")
    width, height = annotated.size

    if show_grid:
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for gx in range(_GRID_STEP, width, _GRID_STEP):
            overlay_draw.line([(gx, 0), (gx, height)], fill=(0, 180, 255, 120), width=2)
        for gy in range(_GRID_STEP, height, _GRID_STEP):
            overlay_draw.line([(0, gy), (width, gy)], fill=(0, 180, 255, 120), width=2)
        annotated = Image.alpha_composite(annotated.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(annotated)
    label_font = _grid_label_font(18)

    def _draw_label(x: int, y: int, text: str):
        try:
            bbox = draw.textbbox((x, y), text, font=label_font)
            pad = 3
            chip_x0, chip_y0 = x - pad, y - pad
            chip_x1, chip_y1 = bbox[2] + pad, bbox[3] + pad
            draw.rectangle([chip_x0, chip_y0, chip_x1, chip_y1], fill=(0, 0, 0, 200))
            draw.text((x + 1, y + 1), text, fill=(255, 255, 255), font=label_font)
        except Exception:
            draw.text((x, y), text, fill=(255, 255, 255), font=label_font)

    if show_grid:
        for gx in range(_GRID_STEP, width, _GRID_STEP):
            _draw_label(gx + 4, 4, str(gx))
        for gy in range(_GRID_STEP, height, _GRID_STEP):
            _draw_label(4, gy + 4, str(gy))

    if cursor_x is not None and cursor_y is not None:
        radius = 18
        for ring_offset, ring_color in ((5, (255, 255, 255)), (0, (255, 0, 0))):
            draw.ellipse(
                (
                    cursor_x - radius + ring_offset,
                    cursor_y - radius + ring_offset,
                    cursor_x + radius - ring_offset,
                    cursor_y + radius - ring_offset,
                ),
                outline=ring_color,
                width=4,
            )
        draw.ellipse(
            (cursor_x - 6, cursor_y - 6, cursor_x + 6, cursor_y + 6),
            fill=(0, 0, 255),
            outline=(255, 255, 255),
            width=2,
        )

    return annotated


def _capture_annotated_b64(
    cursor: Optional[Tuple[int, int]] = None,
    show_grid: bool = True,
) -> Tuple[str, int, int]:
    """
    Captures the screen and annotates it with the coordinate grid and, when
    provided, a cursor marker. Returns (base64_png, width, height).

    Args:
        cursor: Optional (x, y) tuple for the cursor position.
        show_grid: When True, draws the cyan coordinate grid and labels.
    """
    backend = _require_backend()
    try:
        pil_image = backend.screenshot()
    except Exception as shot_err:
        raise RuntimeError(f"Screenshot capture failed: {shot_err}")

    width, height = pil_image.size
    annotated = _annotate_screenshot(
        pil_image,
        cursor[0] if cursor else None,
        cursor[1] if cursor else None,
        show_grid=show_grid,
    )
    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG", optimize=True)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    if len(b64) > _MAX_SCREENSHOT_BASE64_CHARS:
        raise RuntimeError(
            f"Annotated screenshot payload too large ({len(b64)} base64 chars). "
            "Reduce screen resolution or color depth."
        )
    return b64, width, height


def _post_action_result(action_desc: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the result of a desktop action: waits briefly for the UI to settle,
    captures a grid-annotated, cursor-highlighted verification screenshot, saves
    it to the workspace, and reports the cursor position alongside the action.
    The LLM sees the immediate consequence of its action on the next turn.

    NOTE: The verification screenshot ALWAYS has the grid (show_grid=True) to
    preserve the vision-grounding contract for subsequent actions.
    """
    try:
        time.sleep(_POST_ACTION_SETTLE_S)
        backend = _require_backend()
        cursor_x, cursor_y = int(backend.position()[0]), int(backend.position()[1])
    except Exception:
        cursor_x, cursor_y = -1, -1

    result: Dict[str, Any] = {
        "success": True,
        "output": action_desc
        + (
            f" Cursor is now at ({cursor_x}, {cursor_y}). "
            "Verify on the attached screenshot that the cursor is on the "
            "intended target and the UI shows the expected change."
        ),
    }
    result.update(extra)
    if cursor_x >= 0:
        result["cursor_x"] = cursor_x
        result["cursor_y"] = cursor_y

    try:
        b64, width, height = _capture_annotated_b64(
            (cursor_x, cursor_y) if cursor_x >= 0 else None,
            show_grid=True,
        )
    except Exception as shot_err:
        ASCIIColors.warning(f"[computer_use] Post-action verification capture failed: {shot_err}")
        return result

    result["image_b64"] = b64
    result["screen_width"] = width
    result["screen_height"] = height

    saved_file_name: Optional[str] = None
    workspace_file: Optional[str] = None
    try:
        from pathlib import Path

        working_dir = Path.cwd()
        saved_file_name = "computer_use_screenshot.png"
        target_path = working_dir / saved_file_name
        target_path.write_bytes(base64.b64decode(b64))
        workspace_file = str(target_path.resolve())
    except Exception as persist_err:
        ASCIIColors.warning(f"[computer_use] Verification screenshot persistence failed: {persist_err}")

    if workspace_file:
        result["image_filename"] = workspace_file
        result["context_files"] = [workspace_file]
        result["screenshot_file"] = workspace_file
        result["prompt_injection"] = (
            f"\n\n✅ Screenshot attached ({saved_file_name}). "
            f"Cursor at ({cursor_x}, {cursor_y}). "
            f"Verify it is on the intended target; if not, re-estimate from "
            f"the grid labels and retry."
        )
    return result


def _vlm_step_verdict(
    lollms_client_instance: Any,
    screenshot_b64: str,
    step_description: str,
    last_action_desc: str,
    screen_width: int,
    screen_height: int,
) -> Tuple[bool, str]:
    """
    Asks the vision model whether the step is complete. Returns (is_done, reason).
    """
    vision_binding = _resolve_vision_binding(lollms_client_instance)
    if vision_binding is None:
        return False, "No vision model available for step verification."

    prompt = (
        "You are verifying a desktop automation step.\n"
        f"Step objective: {step_description}\n"
        f"Last action performed: {last_action_desc}\n"
        f"Screenshot is {screen_width}x{screen_height} pixels.\n\n"
        "Has the step objective been ACHIEVED in this screenshot? "
        "Reply with ONLY a JSON object: {\"done\": true|false, \"reason\": \"<one sentence>\"}\n"
        "If the UI is still loading, the target element is not yet visible, or the "
        "last action clearly missed, say done=false."
    )
    try:
        response = vision_binding.generate_from_messages(
            messages=[{"role": "user", "content": prompt}],
            images=[screenshot_b64],
            temperature=0.0,
            stream=False,
            n_predict=200,
        )
    except Exception as gen_err:
        return False, f"Verification call failed: {gen_err}"

    text = str(response).strip() if response else ""
    done_match = re.search(r'"done"\s*:\s*(true|false)', text, re.IGNORECASE)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    is_done = bool(done_match and done_match.group(1).lower() == "true")
    reason = reason_match.group(1) if reason_match else text[:150] or "(no reason given)"
    return is_done, reason


def _vlm_next_action(
    lollms_client_instance: Any,
    screenshot_b64: str,
    step_description: str,
    attempt: int,
    screen_width: int,
    screen_height: int,
) -> Optional[Dict[str, Any]]:
    """
    Asks the vision model for the next single action to advance the step.
    Returns a dict like {"tool": "click", "x": 630, "y": 330, "reason": "..."}
    or {"tool": "done", "reason": "..."} when the model believes the step is complete.
    Returns None on failure.
    """
    vision_binding = _resolve_vision_binding(lollms_client_instance)
    if vision_binding is None:
        return None

    prompt = (
        "You are a desktop automation executor. You see a screenshot and must decide "
        "the SINGLE next action to advance the step.\n"
        f"Step objective: {step_description}\n"
        f"Attempt: {attempt}\n"
        f"Screenshot: {screen_width}x{screen_height} px, (0,0) top-left. "
        f"CYAN grid every 100px with labeled intersections — use the grid to read "
        f"exact pixel coordinates. The red circle (blue center) marks the current "
        f"mouse pointer.\n\n"
        "Reply with ONLY a JSON object:\n"
        '{"tool": "click"|"type"|"key"|"scroll"|"wait"|"move_cursor"|"done", '
        '"x": <int>, "y": <int>, "text": "<string>", "key": "<string>", '
        '"reason": "<one sentence: what you see and why this action>"}\n'
        "Rules:\n"
        "- Use the grid labels to estimate x,y by interpolation (nearest lines + fraction).\n"
        "- If the step objective is already satisfied in this screenshot, use tool='done'.\n"
        "- If you need to type, click the input field FIRST (tool='click'), then type in the next attempt.\n"
        "- Do NOT repeat an action that clearly had no effect; try a different coordinate or a wait.\n"
        "- x,y are required for click/move_cursor; text for type; key for key."
    )
    try:
        response = vision_binding.generate_from_messages(
            messages=[{"role": "user", "content": prompt}],
            images=[screenshot_b64],
            temperature=0.0,
            stream=False,
            n_predict=600,
        )
    except Exception:
        return None

    text = str(response).strip() if response else ""
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not json_match:
        return None
    try:
        action = json.loads(json_match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(action, dict) or "tool" not in action:
        return None
    return action


def _execute_single_action(action: Dict[str, Any]) -> str:
    """
    Executes one action dict and returns a human-readable description.
    """
    tool_name = str(action.get("tool", "")).strip().lower()
    reason = str(action.get("reason", "")).strip()[:120]

    try:
        backend = _require_backend()
    except Exception as backend_err:
        return f"FAILED: backend unavailable ({backend_err})"

    try:
        if tool_name == "click":
            x, y = int(action.get("x", 0)), int(action.get("y", 0))
            x, y = _clamp_coordinates(x, y)
            if x == 0 and y == 0:
                return "FAILED: refused to click (0,0)."
            backend.click(x=x, y=y, button="left")
            return f"Clicked at ({x}, {y}). {reason}"
        elif tool_name == "type":
            text = str(action.get("text", ""))
            if not text:
                return "FAILED: no text to type."
            backend.typewrite(text, interval=0.02)
            return f"Typed '{text}'. {reason}"
        elif tool_name == "key":
            key = str(action.get("key", "")).strip().lower()
            if not key:
                return "FAILED: no key specified."
            parts = [p.strip() for p in key.split("+") if p.strip()]
            if len(parts) == 1:
                backend.press(parts[0])
            else:
                backend.hotkey(*parts)
            return f"Pressed '{key}'. {reason}"
        elif tool_name == "scroll":
            amount = int(action.get("amount", 300))
            direction = str(action.get("direction", "down")).strip().lower()
            sign = -1 if direction == "down" else 1
            backend.scroll(sign * min(abs(amount), 2000))
            return f"Scrolled {direction} {abs(amount)}. {reason}"
        elif tool_name == "wait":
            time.sleep(min(float(action.get("seconds", 1.0)), 3.0))
            return f"Waited. {reason}"
        elif tool_name == "move_cursor":
            x, y = int(action.get("x", 0)), int(action.get("y", 0))
            x, y = _clamp_coordinates(x, y)
            backend.moveTo(x, y)
            return f"Moved cursor to ({x}, {y}). {reason}"
        elif tool_name == "done":
            return f"Executor reports done. {reason}"
        else:
            return f"FAILED: unknown tool '{tool_name}'."
    except Exception as act_err:
        return f"FAILED: {act_err}"


def tool_computer_execute_step(
    step_description: str,
    max_attempts: int = 8,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes a single desktop automation step in an isolated child loop.

    The child loop takes a fresh screenshot, asks the vision model for the next
    action, performs it, verifies the result, and repeats until the step is
    complete or max_attempts is exhausted. The parent agent's context is NOT
    polluted with the child's screenshots — only the final verdict is returned.

    Use this for each step in a plan (see tool_computer_plan). The step must be
    self-contained and specific: "open the Camera app via the Start menu search"
    is a good step; "do something with the camera" is not.

    Args:
        step_description (str): The specific, self-contained step to execute. Must be detailed enough for a fresh agent to complete without prior context.
        max_attempts (int): Maximum action cycles before giving up. Default 8.
    """
    if not isinstance(step_description, str) or not step_description.strip():
        return {"success": False, "error": "step_description is required and must be a non-empty string."}
    if not lollms_client_instance:
        return {"success": False, "error": "This tool requires a lollms_client_instance (host-injected)."}
    if not _vision_capability_ready(lollms_client_instance):
        return {"success": False, "error": "A vision-capable model is required to execute desktop steps."}

    try:
        max_attempts = max(1, min(int(max_attempts), 20))
    except (TypeError, ValueError):
        max_attempts = 8

    width, height = _screen_size()
    last_action_desc = "initial state"
    attempts_used = 0

    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        try:
            b64, w, h = _capture_annotated_b64()
        except Exception as shot_err:
            return {
                "success": False,
                "error": f"Screenshot failed at attempt {attempt}: {shot_err}",
                "attempts_used": attempts_used,
            }

        action = _vlm_next_action(
            lollms_client_instance, b64, step_description.strip(), attempt, w, h
        )
        if action is None:
            return {
                "success": False,
                "error": f"Vision model returned no parseable action at attempt {attempt}.",
                "attempts_used": attempts_used,
            }

        if str(action.get("tool", "")).lower() == "done":
            return {
                "success": True,
                "output": f"Step complete after {attempt} attempt(s). {action.get('reason', '')}",
                "attempts_used": attempts_used,
                "final_state": str(action.get("reason", ""))[:300],
            }

        action_desc = _execute_single_action(action)
        last_action_desc = action_desc

        if action_desc.startswith("FAILED"):
            return {
                "success": False,
                "error": f"Action failed at attempt {attempt}: {action_desc}",
                "attempts_used": attempts_used,
            }

        time.sleep(0.4)
        b64_verify, vw, vh = _capture_annotated_b64()
        is_done, verdict_reason = _vlm_step_verdict(
            lollms_client_instance, b64_verify, step_description.strip(),
            last_action_desc, vw, vh
        )
        if is_done:
            return {
                "success": True,
                "output": f"Step complete after {attempt} attempt(s). {verdict_reason}",
                "attempts_used": attempts_used,
                "final_state": verdict_reason[:300],
            }

    return {
        "success": False,
        "error": (
            f"Gave up after {max_attempts} attempts. Last action: {last_action_desc}. "
            "The step could not be completed automatically."
        ),
        "attempts_used": attempts_used,
        "final_state": last_action_desc[:300],
    }


def tool_computer_plan(
    task: str,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Plans a multi-step desktop automation task into ordered, self-contained steps.

    Call this FIRST with the user's goal (e.g. "open the Camera app and take a
    photo of me"). It returns an ordered list of steps. Then execute each step
    with tool_computer_execute_step, in order, passing each step_description.
    If a step fails, re-plan the remaining work or report the failure to the user.

    Args:
        task (str): The user's high-level desktop automation goal.
    """
    if not isinstance(task, str) or not task.strip():
        return {"success": False, "error": "task is required and must be a non-empty string."}
    if not lollms_client_instance:
        return {"success": False, "error": "This tool requires a lollms_client_instance (host-injected)."}

    vision_binding = _resolve_vision_binding(lollms_client_instance)
    if vision_binding is None:
        return {"success": False, "error": "A vision-capable model is required to plan desktop tasks."}

    try:
        width, height = _screen_size()
    except Exception:
        width, height = 0, 0

    b64 = None
    try:
        b64, _, _ = _capture_annotated_b64()
    except Exception:
        pass

    plan_prompt = (
        "You are a desktop automation planner. You see the current screen state "
        "(if a screenshot is attached) and must decompose the user's task into "
        "ordered, self-contained steps.\n"
        f"Screen size: {width}x{height} pixels.\n"
        f"User task: {task.strip()}\n\n"
        "Rules for each step:\n"
        "- Self-contained: a fresh agent with NO prior context must be able to "
        "complete it just from the step text.\n"
        "- Specific: name the exact UI elements, menus, and actions (e.g. "
        "'click the Windows Start button in the taskbar', 'type the text Camera "
        "into the search box', 'click the Caméra app entry in the search results').\n"
        "- One UI interaction per step (one click, one type, one key, or one wait).\n"
        "- Include a short wait step after actions that trigger loading (app launch, page load).\n"
        "- Do NOT include steps for verification — the executor handles that.\n"
        "- Maximum 12 steps. If the task is simpler, use fewer.\n\n"
        "Reply with ONLY a JSON array of step strings, nothing else:\n"
        '["step 1", "step 2", ...]'
    )
    messages = [{"role": "user", "content": plan_prompt}]
    try:
        if b64:
            response = vision_binding.generate_from_messages(
                messages=messages, images=[b64], temperature=0.0, stream=False, n_predict=800
            )
        else:
            response = vision_binding.generate_from_messages(
                messages=messages, temperature=0.0, stream=False, n_predict=800
            )
    except Exception as gen_err:
        return {"success": False, "error": f"Planning generation failed: {gen_err}"}

    text = str(response).strip() if response else ""
    json_match = re.search(r'\[[^\]]*\]', text, re.DOTALL)
    if not json_match:
        return {"success": False, "error": f"Planner did not return a parseable step list. Raw: {text[:300]}"}
    try:
        steps = json.loads(json_match.group(0))
    except (json.JSONDecodeError, ValueError):
        return {"success": False, "error": "Planner returned invalid JSON."}
    if not isinstance(steps, list) or not steps:
        return {"success": False, "error": "Planner returned an empty step list."}
    steps = [str(s).strip() for s in steps if str(s).strip()][:12]

    return {
        "success": True,
        "output": (
            f"Plan created with {len(steps)} step(s). Execute each step in order "
            "using tool_computer_execute_step(step_description=<step text>). "
            "After each step, check the verdict before proceeding to the next."
        ),
        "steps": steps,
        "step_count": len(steps),
    }


def tool_computer_cursor_screenshot(
    file_name: str = "",
    tool_context: Optional[dict] = None,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Takes a screenshot with the current mouse cursor position highlighted: a red circle with a blue center dot marks the exact pointer location.

    Use this to CALIBRATE yourself: the highlighted screenshot shows you where
    the pointer actually is, so you can compare it with where you believed it
    was and correct your coordinate reasoning. Call it after clicks, moves, or
    drags if you are unsure where the cursor ended up.

    Args:
        file_name (str): Optional custom PNG file name to keep this calibration frame (e.g. 'cursor_check.png'). Default 'computer_use_cursor.png' is overwritten each call.
    """
    try:
        backend = _require_backend()
        cursor_x, cursor_y = int(backend.position()[0]), int(backend.position()[1])
    except Exception as pos_err:
        return {"success": False, "error": _sanitize_error(f"Cursor position lookup failed: {pos_err}")}

    return _post_action_result(
        f"Cursor calibration: the pointer is at ({cursor_x}, {cursor_y}). "
        "Compare the red-circle highlight with your intended target and recalibrate "
        "your coordinate reasoning.",
        {"cursor_x": cursor_x, "cursor_y": cursor_y},
    )


def tool_computer_screenshot_zone(
    x: int,
    y: int,
    width: int,
    height: int,
    show_grid: bool = False,
    file_name: str = "",
    save_to_workspace: bool = True,
    tool_context: Optional[dict] = None,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Captures a screenshot of a specific rectangular region (zone) of the screen.

    Use this to focus on a specific part of the screen without capturing the
    entire desktop. The region is specified by its absolute top-left corner
    (x, y) and its dimensions (width, height) in pixels.

    The returned image is cropped to the specified region. When show_grid is
    true, a coordinate grid is overlaid on the cropped image. The grid labels
    are RELATIVE to the region's top-left corner (0, 0), NOT absolute screen
    coordinates. To convert a region-relative coordinate (rx, ry) to an
    absolute screen coordinate, add the region offset: (x + rx, y + ry).

    Args:
        x (int): Absolute X coordinate of the region's top-left corner.
        y (int): Absolute Y coordinate of the region's top-left corner.
        width (int): Width of the region in pixels.
        height (int): Height of the region in pixels.
        show_grid (bool): When true, overlays a cyan coordinate grid on the region. Default false.
        file_name (str): Optional custom PNG file name for the captured region (e.g. 'zone_check.png'). Default 'computer_use_zone.png'.
        save_to_workspace (bool): When true, persists the screenshot as a PNG file in the workspace. Default true.
    """
    try:
        x, y, width, height = int(x), int(y), int(width), int(height)
    except (TypeError, ValueError):
        return {"success": False, "error": "Zone coordinates must be integers."}

    if width <= 0 or height <= 0:
        return {"success": False, "error": "Region width and height must be positive integers."}

    try:
        screen_w, screen_h = _screen_size()
    except RuntimeError as backend_err:
        return {"success": False, "error": _sanitize_error(str(backend_err))}

    # Clamp the region to the screen boundaries
    x = max(0, min(x, screen_w - 1))
    y = max(0, min(y, screen_h - 1))
    width = min(width, screen_w - x)
    height = min(height, screen_h - y)

    if width <= 0 or height <= 0:
        return {
            "success": False,
            "error": f"Region ({x}, {y}, {width}x{height}) is entirely outside the screen ({screen_w}x{screen_h}).",
        }

    try:
        backend = _require_backend()
        pil_image = backend.screenshot(region=(x, y, width, height))
    except Exception as capture_err:
        return {"success": False, "error": _sanitize_error(str(capture_err))}

    try:
        cursor_x, cursor_y = int(backend.position()[0]), int(backend.position()[1])
        # Convert cursor position to region-relative if inside the region
        if x <= cursor_x < x + width and y <= cursor_y < y + height:
            region_cursor = (cursor_x - x, cursor_y - y)
        else:
            region_cursor = None
    except Exception:
        region_cursor = None

    try:
        annotated = _annotate_screenshot(
            pil_image,
            region_cursor[0] if region_cursor else None,
            region_cursor[1] if region_cursor else None,
            show_grid=show_grid,
        )
        buffer = io.BytesIO()
        annotated.save(buffer, format="PNG", optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception as ann_err:
        return {"success": False, "error": _sanitize_error(str(ann_err))}

    if len(b64) > _MAX_SCREENSHOT_BASE64_CHARS:
        return {
            "success": False,
            "error": f"Region screenshot payload too large ({len(b64)} base64 chars).",
        }

    workspace_file: Optional[str] = None
    saved_file_name: Optional[str] = None
    if save_to_workspace:
        try:
            from pathlib import Path

            working_dir = Path.cwd()
            if isinstance(tool_context, dict):
                context_dir = tool_context.get("working_dir")
                if context_dir:
                    try:
                        candidate = Path(context_dir)
                        if candidate.is_dir():
                            working_dir = candidate.resolve()
                    except Exception:
                        pass

            saved_file_name = _sanitize_screenshot_filename(file_name) or "computer_use_zone.png"
            target_path = working_dir / saved_file_name
            target_path.write_bytes(base64.b64decode(b64))
            workspace_file = str(target_path.resolve())
        except Exception as persist_err:
            ASCIIColors.warning(
                f"[computer_use] Zone screenshot workspace persistence failed: {persist_err}"
            )

    grid_desc = (
        f"A CYAN grid with lines every {_GRID_STEP} pixels is overlaid; labels are "
        f"RELATIVE to the region's top-left corner (0, 0). To map a region-relative "
        f"coordinate (rx, ry) to absolute screen coordinates, add the region offset: "
        f"({x} + rx, {y} + ry)."
        if show_grid
        else "No coordinate grid is overlaid. The image is exactly "
        f"{width}x{height} pixels, cropped from the absolute region ({x}, {y}) to "
        f"({x + width}, {y + height})."
    )

    cursor_clause = (
        f" The mouse pointer is at region-relative ({region_cursor[0]}, {region_cursor[1]}), "
        f"marked by the red circle with blue center."
        if region_cursor
        else ""
    )

    result: Dict[str, Any] = {
        "success": True,
        "output": (
            f"Zone screenshot captured: region ({x}, {y}) to ({x + width}, {y + height}), "
            f"size {width}x{height} px."
            + (f" Saved to workspace as '{saved_file_name}'." if workspace_file else "")
            + " The image is attached for your inspection."
            + cursor_clause
        ),
        "image_b64": b64,
        "screenshot_b64": b64,
        "screen_width": width,
        "screen_height": height,
        "zone_x": x,
        "zone_y": y,
        "zone_width": width,
        "zone_height": height,
    }
    if region_cursor:
        result["cursor_x"] = region_cursor[0]
        result["cursor_y"] = region_cursor[1]
    if workspace_file:
        result["image_filename"] = workspace_file
        result["context_files"] = [workspace_file]
        result["screenshot_file"] = workspace_file
        result["prompt_injection"] = (
            f"\n\n✅ **Zone screenshot attached to your visual context** "
            f"({saved_file_name}). The image shows the absolute region "
            f"({x}, {y}) to ({x + width}, {y + height}).\n"
            f"📐 **COORDINATE CONVENTION**: {grid_desc} "
            f"All click/move/drag coordinates are ABSOLUTE screen pixels. "
            f"To target an element in this zone, estimate its position within the "
            f"image and add the region offset ({x}, {y}) to get the absolute coordinate."
            + (f" {cursor_clause.strip()}." if region_cursor else "")
        )
    return result