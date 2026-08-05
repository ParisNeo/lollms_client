import base64
from typing import Any, Dict, Optional

def init_tools_library() -> None:
    pass

def tool_vlm_query(
    image_index: int,
    query: str,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Queries a Vision-Language Model (VLM) about a specific image in the conversation.
    Use this tool when you need to analyze, read text from, or understand the visual contents of an image.
    The VLM has its own context and can answer specific questions about the image.

    Args:
        image_index (int): The 0-based index of the image in the user's message.
        query (str): The specific question to ask the VLM about the image.
    """
    if not discussion_instance or not lollms_client_instance:
        return {"success": False, "error": "System context not available."}

    # Find the VLM binding. If the active binding is a SmartRouter, find the child with vision.
    vlm_binding = None
    if hasattr(lollms_client_instance.llm, "child_bindings"):
        for alias, binding in lollms_client_instance.llm.child_bindings.items():
            if getattr(binding, "vision_enabled", False):
                vlm_binding = binding
                break
    
    # Fallback to master if master supports vision
    if not vlm_binding and getattr(lollms_client_instance.llm, "vision_enabled", False):
        vlm_binding = lollms_client_instance.llm

    if not vlm_binding:
        return {"success": False, "error": "No Vision-Language Model (VLM) is mounted."}

    try:
        branch = discussion_instance.get_branch(discussion_instance.active_branch_id)
        if not branch: return {"success": False, "error": "No conversation history found."}

        user_msgs = [m for m in branch if m.sender_type == "user"]
        if not user_msgs: return {"success": False, "error": "No user message found."}

        last_user_msg = user_msgs[-1]
        images = last_user_msg.images or []
        if image_index < 0 or image_index >= len(images):
            return {"success": False, "error": f"Invalid image_index. Contains {len(images)} image(s)."}

        target_image_b64 = images[image_index]
        if target_image_b64.startswith("data:image"):
            target_image_b64 = target_image_b64.split(";base64,")[1]

        vlm_response = vlm_binding.generate_from_messages(
            messages=[
                {"role": "system", "content": "You are a vision assistant. Answer concisely."},
                {"role": "user", "content": query}
            ],
            images=[target_image_b64],
            temperature=0.1,
            stream=False,
            n_predict=1024
        )

        if isinstance(vlm_response, dict) and "error" in vlm_response:
            return {"success": False, "error": vlm_response["error"]}

        return {
            "success": True,
            "output": str(vlm_response).strip(),
            "prompt_injection": f"\n\n✅ **VLM Analysis Complete.**\nQuery: '{query}'\n**VLM Response:**\n{str(vlm_response).strip()}\n\nYou can now use this information."
        }
    except Exception as e:
        return {"success": False, "error": f"VLM query failed: {str(e)}"}