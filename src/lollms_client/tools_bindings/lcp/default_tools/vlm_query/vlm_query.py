import base64
from pathlib import Path
from typing import Any, Dict, Optional

TOOL_LIBRARY_NAME = "VLM Vision Query"
TOOL_LIBRARY_DESC = "Inspect, analyze, and query local workspace images using the active Vision-Language Model (VLM)."
TOOL_LIBRARY_ICON = "👁️"


def init_tools_library() -> None:
    pass


def tool_inspect_image(
    image_path: str,
    query: str = "Describe this image in detail, identify its visual subject, text, and categorize it.",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Visually inspects and analyzes a local image file from the workspace using the active Vision-Language Model (VLM).
    Use this tool to categorize ambiguous images, identify subjects (portraits, AI art, screenshots, diagrams), or read text inside images.

    Args:
        image_path (str): Relative or absolute path to the image file in the workspace (e.g. 'photo.jpg', 'media/image.png').
        query (str, optional): The visual question to ask about the image. Defaults to description and categorization.
    """
    if not lollms_client_instance:
        return {"success": False, "error": "LollmsClient instance not available."}

    p = Path(image_path)
    if not p.exists():
        ws_root = getattr(discussion_instance, "workspace_path", None) or Path.cwd()
        p = Path(ws_root) / image_path

    if not p.exists() or not p.is_file():
        return {"success": False, "error": f"Image file not found: '{image_path}'."}

    try:
        raw_bytes = p.read_bytes()
        ext = p.suffix.lower().lstrip(".") or "png"
        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        data_uri = f"data:image/{ext};base64,{b64_data}"
    except Exception as ex:
        return {"success": False, "error": f"Failed to read image file '{image_path}': {ex}"}

    vlm = None
    if hasattr(lollms_client_instance, "has_vision_capability") and lollms_client_instance.has_vision_capability():
        vlm = lollms_client_instance.llm
    elif hasattr(lollms_client_instance, "find_available_vlm"):
        vlm = lollms_client_instance.find_available_vlm()

    if not vlm:
        return {"success": False, "error": "No Vision-Language Model (VLM) is active in this session."}

    try:
        messages = [
            {"role": "system", "content": "You are an expert visual analysis assistant. Answer concisely and accurately."},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]}
        ]

        if hasattr(vlm, "generate_from_messages"):
            res = vlm.generate_from_messages(messages=messages, temperature=0.1, n_predict=512)
        elif hasattr(vlm, "generate_text"):
            res = vlm.generate_text(prompt=query, images=[b64_data], temperature=0.1, n_predict=512)
        else:
            return {"success": False, "error": "VLM binding does not support visual generation."}

        desc_text = str(res).strip()
        return {
            "success": True,
            "image_path": str(image_path),
            "output": desc_text,
            "prompt_injection": f"\n\n👁️ **Visual Inspection (`{p.name}`):**\n{desc_text}\n"
        }
    except Exception as ex:
        return {"success": False, "error": f"Visual inspection failed: {ex}"}


def tool_vlm_query(
    query: str,
    image_path: Optional[str] = None,
    image_index: int = -1,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Queries the Vision-Language Model (VLM) about an image file on disk or an image from the conversation.

    Args:
        query (str): The specific question to ask the VLM about the image.
        image_path (str, optional): Path to the image file in the workspace.
        image_index (int, optional): 0-based index of the image in the last user message if querying chat images.
    """
    if image_path:
        return tool_inspect_image(
            image_path=image_path,
            query=query,
            discussion_instance=discussion_instance,
            lollms_client_instance=lollms_client_instance
        )

    if not discussion_instance or not lollms_client_instance:
        return {"success": False, "error": "System context not available."}

    vlm = None
    if hasattr(lollms_client_instance, "has_vision_capability") and lollms_client_instance.has_vision_capability():
        vlm = lollms_client_instance.llm
    elif hasattr(lollms_client_instance, "find_available_vlm"):
        vlm = lollms_client_instance.find_available_vlm()

    if not vlm:
        return {"success": False, "error": "No Vision-Language Model (VLM) is active."}

    try:
        branch = discussion_instance.get_branch(discussion_instance.active_branch_id) if hasattr(discussion_instance, "get_branch") else None
        if not branch:
            return {"success": False, "error": "No conversation history found."}

        user_msgs = [m for m in branch if getattr(m, "sender_type", "") == "user"]
        if not user_msgs:
            return {"success": False, "error": "No user message with images found."}

        images = getattr(user_msgs[-1], "images", []) or []
        if image_index < 0 or image_index >= len(images):
            return {"success": False, "error": f"Invalid image_index. User message contains {len(images)} image(s)."}

        target_b64 = images[image_index]
        if target_b64.startswith("data:image"):
            target_b64 = target_b64.split(";base64,")[1]

        data_uri = f"data:image/jpeg;base64,{target_b64}"
        messages = [
            {"role": "system", "content": "You are a vision analysis assistant. Answer concisely."},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]}
        ]
        res = vlm.generate_from_messages(messages=messages, temperature=0.1, n_predict=512)
        return {"success": True, "output": str(res).strip()}
    except Exception as e:
        return {"success": False, "error": f"VLM query failed: {e}"}