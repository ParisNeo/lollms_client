# lollms_client/lollms_history.py
# HistoryManager: Shared history export, virtual history management, and message normalization.

import re
from typing import Any, Dict, List, Optional, Union

from ascii_colors import ASCIIColors

from lollms_client.lollms_utilities import build_image_dicts


class _MessageWrapper:
    """
    Internal wrapper to normalize dictionary-based and object-based messages 
    into a consistent interface for HistoryManager.
    """
    def __init__(self, msg: Union[Dict, Any]):
        if isinstance(msg, dict):
            self.id = msg.get("id", "")
            self.sender = msg.get("sender", msg.get("role", ""))
            self.sender_type = msg.get("sender_type", msg.get("role", ""))
            self.content = msg.get("content", "")
            self.parent_id = msg.get("parent_id")
            self.metadata = msg.get("metadata", {})
            self.images = msg.get("images", [])
            self.active_images = msg.get("active_images", [])
        else:
            self.id = getattr(msg, "id", "")
            self.sender = getattr(msg, "sender", getattr(msg, "role", ""))
            self.sender_type = getattr(msg, "sender_type", getattr(msg, "role", ""))
            self.content = getattr(msg, "content", "")
            self.parent_id = getattr(msg, "parent_id", None)
            self.metadata = getattr(msg, "metadata", {})
            self.images = getattr(msg, "images", [])
            self.active_images = getattr(msg, "active_images", [])

    def get_active_images(self) -> list:
        if hasattr(self, '_original_msg') and hasattr(self._original_msg, 'get_active_images'):
            return self._original_msg.get_active_images()
        return self.active_images or []


class HistoryManager:
    """
    Handles the export of discussion history, integration of agentic virtual_history,
    and normalization of messages for specific LLM APIs (e.g., OpenAI).
    This class is shared between LollmsDiscussion and LollmsPersonality.
    """

    @staticmethod
    def _has_functional_tags(text: str) -> bool:
        """Check if the text contains functional tags like <tool> or <artifact>."""
        return bool(re.search(r'<(tool|artifact|artefact)\b[^>]*>', text or '', re.IGNORECASE))

    @staticmethod
    def _last_user_index(branch_list: list) -> int:
        """Find the index of the last user message in the branch."""
        for i in range(len(branch_list) - 1, -1, -1):
            if branch_list[i].sender_type == 'user':
                return i
        return -1

    @staticmethod
    def export(
        context: Any,
        format_type: str,
        branch: List[Any],
        branch_tip_id: Optional[str] = None,
        max_allowed_tokens: Optional[int] = None,
        suppress_system_prompt: bool = False,
        suppress_images: bool = False,
        virtual_history: Optional[list] = None,
        debug: bool = False,
        system_prompt_override: Optional[str] = None
    ) -> Any:
        """
        Exports the discussion history. 
        If virtual_history is provided, it appends the active agentic context (unstripped)
        to the sanitized historical branch.
        """
        branch = [_MessageWrapper(msg) for msg in branch]
        if virtual_history:
            virtual_history = [_MessageWrapper(vh) for vh in virtual_history]

        if not branch and format_type in ["lollms_text", "openai_chat", "ollama_chat", "markdown"]:
            return "" if format_type in ["lollms_text", "markdown"] else []

        if branch and branch[-1].sender_type == 'assistant':
            last_msg = branch[-1]
            is_empty_building_msg = not last_msg.content.strip()
            has_persisted_vh = isinstance(last_msg.metadata, dict) and bool(last_msg.metadata.get("virtual_history"))
            if is_empty_building_msg and not has_persisted_vh:
                branch = branch[:-1]

        system_prompt_part = (system_prompt_override or getattr(context, '_system_prompt', None) or "").strip()
        data_zone_part = context.get_full_data_zone()
        full_system_prompt = ""

        pruning_summary = getattr(context, 'pruning_summary', None)
        if pruning_summary:
            data_zone_part = f"--- PROJECT SYNOPSIS ---\n{pruning_summary}\n\n" + data_zone_part

        if not suppress_system_prompt:
            if system_prompt_part and data_zone_part:
                full_system_prompt = f"{system_prompt_part}\n\n{data_zone_part}"
            elif system_prompt_part:
                full_system_prompt = system_prompt_part
            else:
                full_system_prompt = data_zone_part

        _scratchpad = getattr(context, "scratchpad", "") or ""

        FUNCTIONAL_QUOTA = 2
        functional_skip_count = 0

        if format_type == "lollms_text":
            final_parts = []
            message_parts = []
            current_tokens = 0
            messages_to_render = branch
            summary_text = ""
            pruning_point_id = getattr(context, 'pruning_point_id', None)
            if pruning_summary and pruning_point_id:
                pi = next((i for i, m in enumerate(branch) if m.id == pruning_point_id), -1)
                if pi != -1:
                    messages_to_render = branch[pi:]
                    summary_text = f"!@>system:\n--- Conversation Summary ---\n{pruning_summary.strip()}\n"

            if full_system_prompt:
                sys_text = f"!@>system:\n{full_system_prompt.strip()}\n"
                sys_toks = context.lollmsClient.count_tokens(sys_text)
                if max_allowed_tokens is None or sys_toks <= max_allowed_tokens:
                    final_parts.append(sys_text)
                    current_tokens += sys_toks
            if summary_text:
                st = context.lollmsClient.count_tokens(summary_text)
                if max_allowed_tokens is None or current_tokens + st <= max_allowed_tokens:
                    final_parts.append(summary_text)
                    current_tokens += st

            last_user_idx = HistoryManager._last_user_index(messages_to_render)

            for idx, msg in enumerate(reversed(messages_to_render)):
                fwd_idx = len(messages_to_render) - 1 - idx
                sender_str = msg.sender.replace(':', '').replace('!@>', '')

                is_recent_functional = False
                if msg.sender_type == 'assistant' and HistoryManager._has_functional_tags(msg.content):
                    if functional_skip_count < FUNCTIONAL_QUOTA:
                        functional_skip_count += 1
                        is_recent_functional = True

                content = context._apply_three_view_protocol(msg, msg.content.strip(), 0 if is_recent_functional else 99)
                active_images = msg.get_active_images()
                if active_images:
                    content += f"\n({len(active_images)} image(s) attached)"
                msg_text = f"!@>{sender_str}:\n{content}\n"
                msg_toks = context.lollmsClient.count_tokens(msg_text)
                if max_allowed_tokens is not None and current_tokens + msg_toks > max_allowed_tokens:
                    break
                message_parts.insert(0, msg_text)
                current_tokens += msg_toks

                if _scratchpad and fwd_idx == last_user_idx:
                    scratch_content = f"== TOOL OUTPUT SCRATCHPAD ==\n{_scratchpad}\n== END SCRATCHPAD =="
                    scratch_text = f"!@>user:\n[SYSTEM CONTEXT]\n{scratch_content}\n[/SYSTEM CONTEXT]\n"
                    scratch_text_toks = context.lollmsClient.count_tokens(scratch_text)
                    if max_allowed_tokens is None or current_tokens + scratch_text_toks <= max_allowed_tokens:
                        message_parts.insert(1, scratch_text)
                        current_tokens += scratch_text_toks

            if virtual_history:
                for v_msg in virtual_history:
                    sender_str = "user" if v_msg.sender_type == "user" else context.lollmsClient.ai_name
                    msg_text = f"!@>{sender_str}:\n{v_msg.content}\n"
                    msg_toks = context.lollmsClient.count_tokens(msg_text)
                    if max_allowed_tokens is not None and current_tokens + msg_toks > max_allowed_tokens:
                        break
                    message_parts.append(msg_text)
                    current_tokens += msg_toks

            final_parts.extend(message_parts)
            return "".join(final_parts).strip()

        messages = []
        discussion_imgs = context.get_discussion_images()
        system_level_images = [i['data'] for i in discussion_imgs if i.get('active', True)]
        active_art_images = context.artefacts.get_context_images()
        for img in active_art_images:
            if img.get("data") and img["data"] not in system_level_images:
                system_level_images.append(img["data"])

        active_discussion_b64 = system_level_images
        if full_system_prompt or (active_discussion_b64 and format_type in ["openai_chat", "ollama_chat", "markdown"] and not suppress_images):
            discussion_level_images = build_image_dicts(active_discussion_b64) if not suppress_images else []
            if format_type == "openai_chat":
                content_parts = []
                if full_system_prompt:
                    content_parts.append({"type": "text", "text": full_system_prompt})
                if not suppress_images:
                    for img in discussion_level_images:
                        url = f"data:image/jpeg;base64,{img['data']}" if img['type'] == 'base64' else img['data']
                        content_parts.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
                if content_parts:
                    messages.append({"role": "system", "content": content_parts})
            elif format_type == "ollama_chat":
                sd = {"role": "system", "content": full_system_prompt or ""}
                if not suppress_images:
                    b64s = [i['data'] for i in discussion_level_images if i['type'] == 'base64']
                    if b64s:
                        sd["images"] = b64s
                messages.append(sd)
            elif format_type == "markdown":
                parts = []
                if full_system_prompt:
                    parts.append(f"system: {full_system_prompt}")
                if not suppress_images:
                    for img in discussion_level_images:
                        url = f"![Image](data:image/jpeg;base64,{img['data']})" if img['type'] == 'base64' else f"![Image]({img['data']})"
                        parts.append(f"\n{url}\n")
                if parts:
                    messages.append("".join(parts))
            else:
                if full_system_prompt:
                    messages.append({"role": "system", "content": full_system_prompt})

        last_user_idx = HistoryManager._last_user_index(branch)

        for idx, msg in enumerate(branch):
            role = msg.sender_type
            distance_from_end = len(branch) - 1 - idx

            is_recent_functional = False
            if msg.sender_type == 'assistant' and HistoryManager._has_functional_tags(msg.content):
                if functional_skip_count < FUNCTIONAL_QUOTA:
                    functional_skip_count += 1
                    is_recent_functional = True

            content = context._apply_three_view_protocol(msg, msg.content.strip(), 0 if is_recent_functional else 99)

            active_images_b64 = msg.get_active_images()
            images_dicts = build_image_dicts(active_images_b64)

            is_historical_assistant = (role == "assistant")
            has_persisted_vh = is_historical_assistant and isinstance(msg.metadata, dict) and bool(msg.metadata.get("virtual_history"))

            if has_persisted_vh:
                persisted_vh = msg.metadata["virtual_history"]
                for vh_entry in persisted_vh:
                    vh_role = "user" if vh_entry.get("sender_type") == "user" else "assistant"
                    vh_content = vh_entry.get("content", "")
                    if not vh_content:
                        continue

                    if format_type == "openai_chat":
                        messages.append({"role": vh_role, "content": vh_content})
                    elif format_type == "ollama_chat":
                        messages.append({"role": vh_role, "content": vh_content})
                    elif format_type == "markdown":
                        sender_str = "User" if vh_role == "user" else "Assistant"
                        messages.append(f"**{sender_str}**: {vh_content}\n")

                if images_dicts and not suppress_images and format_type == "openai_chat":
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i]["role"] == "assistant":
                            parts = [{"type": "text", "text": messages[i]["content"]}] if messages[i]["content"] else []
                            for img in images_dicts:
                                url = f"data:image/jpeg;base64,{img['data']}" if img['type'] == 'base64' else img['data']
                                parts.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
                            messages[i]["content"] = parts
                            break
            else:
                if format_type == "openai_chat":
                    if images_dicts and not suppress_images:
                        parts = [{"type": "text", "text": content}] if content else []
                        for img in images_dicts:
                            url = f"data:image/jpeg;base64,{img['data']}" if img['type'] == 'base64' else img['data']
                            parts.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
                        messages.append({"role": role, "content": parts})
                    else:
                        messages.append({"role": role, "content": content})
                elif format_type == "ollama_chat":
                    md = {"role": role, "content": content}
                    if not suppress_images:
                        b64s = [i['data'] for i in images_dicts if i['type'] == 'base64']
                        if b64s:
                            md["images"] = b64s
                    messages.append(md)
                elif format_type == "markdown":
                    line = f"**{role.capitalize()}**: {content}\n"
                    if images_dicts and not suppress_images:
                        for img in images_dicts:
                            url = f"![Image](data:image/jpeg;base64,{img['data']})" if img['type'] == 'base64' else f"![Image]({img['data']})"
                            line += f"\n{url}\n"
                    messages.append(line)

            if _scratchpad and idx == last_user_idx:
                scratch_content = f"== TOOL OUTPUT SCRATCHPAD ==\n{_scratchpad}\n== END SCRATCHPAD =="
                if format_type == "openai_chat":
                    messages.append({"role": "user", "content": f"[SYSTEM CONTEXT - TOOL OUTPUTS]\n{scratch_content}\n[/SYSTEM CONTEXT]"})
                elif format_type == "ollama_chat":
                    messages.append({"role": "user", "content": f"[SYSTEM CONTEXT - TOOL OUTPUTS]\n{scratch_content}\n[/SYSTEM CONTEXT]"})
                elif format_type == "markdown":
                    messages.append(f"**system**: {scratch_content}\n")

        if virtual_history:
            for v_msg in virtual_history:
                role = "user" if v_msg.sender_type == "user" else "assistant"
                if format_type == "openai_chat":
                    messages.append({"role": role, "content": v_msg.content})
                elif format_type == "ollama_chat":
                    messages.append({"role": role, "content": v_msg.content})
                elif format_type == "markdown":
                    sender_str = "User" if role == "user" else "Assistant"
                    messages.append(f"**{sender_str}**: {v_msg.content}\n")

        _mm = getattr(context, 'memory_manager', None)
        if _mm is not None and format_type in ("openai_chat", "ollama_chat"):
            messages = context._inject_memory_into_messages(
                messages, _mm, format_type,
                token_counter=context.lollmsClient.count_tokens,
            )

        if format_type == "openai_chat" and messages:
            messages = HistoryManager._normalize_openai_messages(messages)

        if debug:
            try:
                import os as _os
                from pathlib import Path as _Path
                import json as _json
                from datetime import datetime as _dt

                debug_dir = _Path(context.workspace_data_path) / "_debug_dumps"
                debug_dir.mkdir(parents=True, exist_ok=True)

                timestamp = _dt.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                dump_file = debug_dir / f"export_dump_{timestamp}.json"

                dump_payload = {
                    "timestamp": timestamp,
                    "discussion_id": getattr(context, 'id', 'unknown'),
                    "format_type": format_type,
                    "messages": messages
                }

                with open(dump_file, "w", encoding="utf-8") as f:
                    _json.dump(dump_payload, f, indent=2, default=str, ensure_ascii=False)

                ASCIIColors.info(f"[HistoryManager] Debug export dump saved to: {dump_file}")
            except Exception as dump_err:
                ASCIIColors.warning(f"[HistoryManager] Failed to write debug export dump: {dump_err}")

        return "\n".join(messages) if format_type == "markdown" else messages

    @staticmethod
    def _normalize_openai_messages(messages: List[Dict]) -> List[Dict]:
        """
        Normalize messages for OpenAI API compliance:
        1. Fuse all system messages into ONE at the beginning
        2. Ensure user/assistant messages alternate (merge consecutive same-role messages)
        3. Remove empty messages
        """
        if not messages:
            return messages

        normalized = []
        system_content_parts = []

        non_system_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    system_content_parts.append("\n".join(text_parts))
                else:
                    system_content_parts.append(str(content))
            else:
                non_system_messages.append(msg)

        if system_content_parts:
            fused_system_content = "\n\n".join(part for part in system_content_parts if part.strip())
            if fused_system_content.strip():
                first_sys = next((m for m in messages if m.get("role") == "system"), {})
                first_content = first_sys.get("content", "")
                if isinstance(first_content, list):
                    image_parts = [item for item in first_content if item.get("type") == "image_url"]
                    normalized.append({
                        "role": "system",
                        "content": [{"type": "text", "text": fused_system_content}] + image_parts
                    })
                else:
                    normalized.append({"role": "system", "content": fused_system_content})

        if non_system_messages:
            current_role = None
            current_content = []
            current_images = []

            for msg in non_system_messages:
                role = msg.get("role")
                content = msg.get("content", "")

                if not content and not msg.get("images"):
                    continue

                if role == current_role:
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                current_content.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                current_images.append(item)
                    else:
                        current_content.append(str(content))

                    if msg.get("images"):
                        current_images.extend(msg["images"])
                else:
                    if current_role is not None and current_content:
                        merged_content = "\n\n".join(c for c in current_content if c.strip())
                        if merged_content.strip():
                            if current_images:
                                text_part = {"type": "text", "text": merged_content}
                                normalized.append({
                                    "role": current_role,
                                    "content": [text_part] + current_images
                                })
                            else:
                                normalized.append({"role": current_role, "content": merged_content})

                    current_role = role
                    current_content = []
                    current_images = []

                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                current_content.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                current_images.append(item)
                    else:
                        current_content.append(str(content))

                    if msg.get("images"):
                        current_images.extend(msg["images"])

            if current_role is not None and current_content:
                merged_content = "\n\n".join(c for c in current_content if c.strip())
                if merged_content.strip():
                    if current_images:
                        text_part = {"type": "text", "text": merged_content}
                        normalized.append({
                            "role": current_role,
                            "content": [text_part] + current_images
                        })
                    else:
                        normalized.append({"role": current_role, "content": merged_content})

        non_sys_start = 0
        for i, msg in enumerate(normalized):
            if msg.get("role") != "system":
                non_sys_start = i
                break

        if non_sys_start < len(normalized):
            first_non_sys = normalized[non_sys_start]
            if first_non_sys.get("role") == "assistant":
                normalized.insert(non_sys_start, {
                    "role": "user",
                    "content": "Continue."
                })

        prev_role = None
        for msg in normalized:
            role = msg.get("role")
            if role == "system":
                continue
            if prev_role and prev_role != "system" and prev_role == role:
                ASCIIColors.warning(
                    f"[OpenAI Export] Consecutive {role} messages detected after normalization. "
                    "This may cause API errors."
                )
            prev_role = role

        return normalized