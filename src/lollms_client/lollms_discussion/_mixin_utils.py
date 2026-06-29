# lollms_discussion/_mixin_utils.py
# UtilsMixin: branch management, export, pruning, context status,
#             image helpers, metadata, legacy artefact shims, serialisation.

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_utilities import build_image_dicts
from lollms_client.lollms_artefact import ArtefactType
import ascii_colors as logging

# Create module-level loggers for easy access
logger = logging.getLogger(__name__)
discussion_logger = logging.getLogger("lollms_client.lollms_discussion._mixin_utils")

class UtilsMixin:
    """
    All utility methods: regenerate_branch, delete_branch, export, summarize_and_prune,
    memorize, get_context_status, image helpers, metadata helpers, legacy artefact shims,
    JSON serialisation / deserialisation, clone_without_messages, and fix_orphan_messages.
    """

    # ---- All original methods below (unchanged) ----------------------------

    def regenerate_branch(self, branch_tip_id=None, **kwargs):
        self._rebuild_message_index()
        target_id = branch_tip_id if branch_tip_id is not None else self.active_branch_id
        if not target_id or target_id not in self._message_index:
            raise ValueError("Regeneration failed: Target message ID not found.")
        target_msg = self._message_index[target_id]
        if target_msg.sender_type == 'assistant':
            user_parent_id = target_msg.parent_id
            if user_parent_id is None or user_parent_id not in self._message_index:
                raise ValueError("Regeneration failed: No valid user parent.")
            user_msg_to_regenerate_from = self._message_index[user_parent_id]
            # Delete the old AI message
            self.remove_message(target_id)
        elif target_msg.sender_type == 'user':
            user_msg_to_regenerate_from = target_msg
        else:
            raise ValueError(f"Unexpected sender type: '{target_msg.sender_type}'.")
        self.active_branch_id = user_msg_to_regenerate_from.id
        return self.chat(user_message="", add_user_message=False,
                         branch_tip_id=user_msg_to_regenerate_from.id, **kwargs)

    def delete_branch(self, message_id: str):
        """
        Compatibility shim — delegates to BranchMixin.prune_branch()
        which handles both DB-backed and in-memory discussions.

        To delete only the leaf (trimming back empty ancestors), use:
            disc.delete_branch(leaf_id, keep_ancestors=True)

        To delete a message and ALL its descendants, use:
            disc.prune_branch(message_id)
        """
        # BranchMixin.prune_branch is the canonical implementation.
        # Calling super() here would go to BranchMixin in the MRO.
        return self.prune_branch(message_id)

    def export(self, format_type, branch_tip_id=None, max_allowed_tokens=None,
               suppress_system_prompt=False, suppress_images=False):
        branch_tip_id = branch_tip_id or self.active_branch_id

        # --- Diagnostic Input Logging ---

        if not branch_tip_id and format_type in ["lollms_text","openai_chat","ollama_chat","markdown"]:
            return "" if format_type in ["lollms_text","markdown"] else []
        branch = self.get_branch(branch_tip_id)

        # Exclude the last message from the branch if it's an assistant message
        # This prevents empty assistant messages (e.g., from fast path) from being
        # included in the export, which can cause issues with some systems
        if branch and branch[-1].sender_type == 'assistant':
            branch = branch[:-1]
        # Force a refresh of the artifacts zone before export
        system_prompt_part = (self._system_prompt or "").strip()
        data_zone_part     = self.get_full_data_zone()
        full_system_prompt = ""

        # Add dynamic synopsis if available
        if self.pruning_summary:
            data_zone_part = f"--- PROJECT SYNOPSIS ---\n{self.pruning_summary}\n\n" + data_zone_part

        if not suppress_system_prompt:
            if system_prompt_part and data_zone_part:
                full_system_prompt = f"{system_prompt_part}\n\n{data_zone_part}"
            elif system_prompt_part:
                full_system_prompt = system_prompt_part
            else:
                full_system_prompt = data_zone_part
        participants = self.participants or {}

        # Scratchpad is now injected via export() to avoid template issues with
        # mid-conversation system messages in strict chat templates like llama.cpp.
        _scratchpad = getattr(self, "scratchpad", "") or ""

        def get_full_content(msg):
            content = msg.content.strip()

            # ── SECURITY & CONTEXT OPTIMIZATION: STRIP THOUGHTS BY DEFAULT ──
            # Completely strip any <think>...</think> reasoning blocks from past messages
            # before sending them to the LLM to prevent reinforcement of old reasoning paths.
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL | re.IGNORECASE)

            def _parse_attrs(attr_str: str) -> Dict[str, str]:
                return {m.group(1): m.group(2)
                        for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

            # ── SINGLE SOURCE OF TRUTH CONTEXT DOCTRINE ──
            # To prevent massive token duplication and bloat in the context window,
            # we must NEVER export the full verbatim content of artifacts/notes/skills inside past messages.
            # We strip any complete XML blocks and replace them with their lightweight placeholders during export.
            pattern = re.compile(r'<(artifact|artefact|note|skill)\b([^>]*?)>(.*?)</\1>', re.DOTALL | re.IGNORECASE)

            def _strip_tag_to_placeholder(match: re.Match) -> str:
                tag_name = match.group(1).lower()
                attrs = _parse_attrs(match.group(2))
                title = attrs.get("name") or attrs.get("title") or "artifact"
                type_label = "note" if tag_name == "note" else ("skill" if tag_name == "skill" else "artefact")
                # Export as a stable, lightweight Lollms Artifact Anchor tag so the LLM retains situational awareness
                return f'\n<lollms_artifact id="{title}" type="{type_label}" version="1" />\n'

            content = pattern.sub(_strip_tag_to_placeholder, content)

            if msg.sender_type == 'assistant':
                # Strip completed/closed processing blocks
                content = re.sub(
                    r'<processing.*?>.*?</processing>',
                    '',
                    content, flags=re.DOTALL | re.IGNORECASE
                )
                # Strip any unclosed processing tags
                content = re.sub(
                    r'<processing[^>]*>',
                    '',
                    content, flags=re.IGNORECASE
                )
                # Strip any remaining closing tags
                content = content.replace("</processing>", "")
                # Strip framework-emitted status, checklist lines, and actions
                lines = []
                for line in content.splitlines():
                    l_strip = line.strip()
                    if l_strip.startswith(("*", "✓", "🏗️", "🔧", "✅", "❌", "·ᴽЧØс·", "[BLIND_ACTION_EXECUTED]")):
                        continue
                    lines.append(line)
                content = "\n".join(lines).strip()
            return content

        # Helper: find the forward index of the last user message in a list
        def _last_user_index(branch_list):
            for i in range(len(branch_list) - 1, -1, -1):
                if branch_list[i].sender_type == 'user':
                    return i
            return -1

        def _last_user_msg(branch_list):
            idx = _last_user_index(branch_list)
            return branch_list[idx] if idx >= 0 else None

        if format_type == "lollms_text":
            final_parts = []
            message_parts = []
            current_tokens = 0
            messages_to_render = branch
            summary_text = ""
            if self.pruning_summary and self.pruning_point_id:
                pi = next((i for i,m in enumerate(branch) if m.id==self.pruning_point_id), -1)
                if pi != -1:
                    messages_to_render = branch[pi:]
                    summary_text = f"!@>system:\n--- Conversation Summary ---\n{self.pruning_summary.strip()}\n"
            if full_system_prompt:
                sys_text = f"!@>system:\n{full_system_prompt.strip()}\n"
                sys_toks = self.lollmsClient.count_tokens(sys_text)
                if max_allowed_tokens is None or sys_toks <= max_allowed_tokens:
                    final_parts.append(sys_text)
                    current_tokens += sys_toks
            if summary_text:
                st = self.lollmsClient.count_tokens(summary_text)
                if max_allowed_tokens is None or current_tokens + st <= max_allowed_tokens:
                    final_parts.append(summary_text)
                    current_tokens += st

            last_user_idx = _last_user_index(messages_to_render)

            for idx, msg in enumerate(reversed(messages_to_render)):
                # Convert reversed index back to forward index
                fwd_idx    = len(messages_to_render) - 1 - idx
                sender_str = msg.sender.replace(':','').replace('!@>','')
                
                tool_calls = msg.metadata.get("tool_calls", []) if (msg.metadata and isinstance(msg.metadata, dict)) else []
                if msg.sender_type == "assistant" and tool_calls:
                    raw_content = getattr(msg, "raw_content", "") or msg.content or ""
                    parts = re.split(r'(<(?:tool|tool_call)>.*?</(?:tool|tool_call)>)', raw_content, flags=re.DOTALL | re.IGNORECASE)
                    num_calls = min(len(parts) // 2, len(tool_calls))
                    
                    expanded_parts = []
                    for i in range(num_calls):
                        assistant_part = parts[2*i] + parts[2*i + 1]
                        tc = tool_calls[i]
                        tool_name = tc.get("name", "unknown")
                        result_obj = tc.get("result", {})
                        
                        if isinstance(result_obj, dict) and "output" in result_obj:
                            result_str = result_obj["output"]
                        else:
                            result_str = json.dumps(result_obj, indent=2, ensure_ascii=False)
                            
                        # Truncate very large results for export to avoid context bloat
                        max_result_len = 4000
                        if len(result_str) > max_result_len:
                            result_str = result_str[:max_result_len] + f"\n... [{len(result_str) - max_result_len} chars truncated]"
                        
                        user_part = f"Tool output for {tool_name}:\n{result_str}\n\nPlease continue your response based on this information."
                        
                        expanded_parts.append(f"!@>{sender_str}:\n{assistant_part}\n")
                        expanded_parts.append(f"!@>user:\n{user_part}\n")
                        
                    remaining_assistant_part = parts[2*num_calls] if 2*num_calls < len(parts) else ""
                    if remaining_assistant_part.strip():
                        clean_rem = re.sub(r'<processing.*?>.*?</processing>', '', remaining_assistant_part, flags=re.DOTALL | re.IGNORECASE)
                        clean_rem = re.sub(r'<processing[^>]*>', '', clean_rem, flags=re.IGNORECASE).replace("</processing>", "")
                        expanded_parts.append(f"!@>{sender_str}:\n{clean_rem.strip()}\n")
                        
                    # Insert in correct order
                    for part_text in reversed(expanded_parts):
                        part_toks = self.lollmsClient.count_tokens(part_text)
                        if max_allowed_tokens is not None and current_tokens + part_toks > max_allowed_tokens:
                            break
                        message_parts.insert(0, part_text)
                        current_tokens += part_toks
                else:
                    content    = get_full_content(msg)
                    active_images = msg.get_active_images()
                    if active_images:
                        content += f"\n({len(active_images)} image(s) attached)"
                    msg_text = f"!@>{sender_str}:\n{content}\n"
                    msg_toks = self.lollmsClient.count_tokens(msg_text)
                    if max_allowed_tokens is not None and current_tokens + msg_toks > max_allowed_tokens:
                        break
                    message_parts.insert(0, msg_text)
                    current_tokens += msg_toks

                # Inject scratchpad ONLY when non-empty, right after last user message
                # For OpenAI/ollama_chat formats, we inject as a user message to avoid
                # strict template issues with mid-conversation system messages
                if _scratchpad and fwd_idx == last_user_idx:
                    scratch_content = (
                        "== TOOL OUTPUT SCRATCHPAD ==\n"
                        f"{_scratchpad}\n"
                        "== END SCRATCHPAD =="
                    )
                    # Use user role with special marker to avoid template issues
                    scratch_text = f"!@>user:\n[SYSTEM CONTEXT]\n{scratch_content}\n[/SYSTEM CONTEXT]\n"
                    scratch_text_toks = self.lollmsClient.count_tokens(scratch_text)
                    if max_allowed_tokens is None or current_tokens + scratch_text_toks <= max_allowed_tokens:
                        # index 1 = immediately after the user msg prepended at [0]
                        message_parts.insert(1, scratch_text)
                        current_tokens += scratch_text_toks

            final_parts.extend(message_parts)
            return "".join(final_parts).strip()

        messages = []
        # Collect only discussion-level and active artifact images for the system message
        discussion_imgs = self.get_discussion_images()
        system_level_images = [i['data'] for i in discussion_imgs if i.get('active', True)]
        active_art_images = self.artefacts.get_context_images()
        for img in active_art_images:
            if img.get("data") and img["data"] not in system_level_images:
                system_level_images.append(img["data"])

        active_discussion_b64 = system_level_images
        if full_system_prompt or (active_discussion_b64 and format_type in ["openai_chat","ollama_chat","markdown"]):
            discussion_level_images = build_image_dicts(active_discussion_b64)
            if format_type == "openai_chat":
                content_parts = []
                if full_system_prompt:
                    content_parts.append({"type":"text","text":full_system_prompt})
                for img in discussion_level_images:
                    url = f"data:image/jpeg;base64,{img['data']}" if img['type']=='base64' else img['data']
                    content_parts.append({"type":"image_url","image_url":{"url":url,"detail":"auto"}})
                if content_parts:
                    messages.append({"role":"system","content":content_parts})
            elif format_type == "ollama_chat":
                sd = {"role":"system","content":full_system_prompt or ""}
                b64s = [i['data'] for i in discussion_level_images if i['type']=='base64']
                if b64s:
                    sd["images"] = b64s
                messages.append(sd)
            elif format_type == "markdown":
                parts = []
                if full_system_prompt:
                    parts.append(f"system: {full_system_prompt}")
                if not suppress_images:
                    for img in discussion_level_images:
                        url = f"![Image](data:image/jpeg;base64,{img['data']})" if img['type']=='base64' else f"![Image]({img['data']})"
                        parts.append(f"\n{url}\n")
                if parts:
                    messages.append("".join(parts))
            else:
                if full_system_prompt:
                    messages.append({"role":"system","content":full_system_prompt})

        last_user_idx = _last_user_index(branch)

        for idx, msg in enumerate(branch):
            role = participants.get(msg.sender, "user" if msg.sender_type=='user' else "assistant")
            if isinstance(role, dict):
                role = role.get("name","user" if msg.sender_type=='user' else "assistant")
            
            tool_calls = msg.metadata.get("tool_calls", []) if (msg.metadata and isinstance(msg.metadata, dict)) else []
            if msg.sender_type == "assistant" and tool_calls:
                raw_content = getattr(msg, "raw_content", "") or msg.content or ""
                parts = re.split(r'(<(?:tool|tool_call)>.*?</(?:tool|tool_call)>)', raw_content, flags=re.DOTALL | re.IGNORECASE)
                num_calls = min(len(parts) // 2, len(tool_calls))
                
                for i in range(num_calls):
                    assistant_part = parts[2*i] + parts[2*i + 1]
                    tc = tool_calls[i]
                    tool_name = tc.get("name", "unknown")
                    result_obj = tc.get("result", {})
                    
                    if isinstance(result_obj, dict) and "output" in result_obj:
                        result_str = result_obj["output"]
                    else:
                        result_str = json.dumps(result_obj, indent=2, ensure_ascii=False)
                        
                    # Truncate very large results for export to avoid context bloat
                    max_result_len = 4000
                    if len(result_str) > max_result_len:
                        result_str = result_str[:max_result_len] + f"\n... [{len(result_str) - max_result_len} chars truncated]"
                    
                    user_part = f"Tool output for {tool_name}:\n{result_str}\n\nPlease continue your response based on this information."
                    
                    # 1. Append assistant_part
                    if format_type == "openai_chat":
                        messages.append({"role": role, "content": assistant_part})
                    elif format_type == "ollama_chat":
                        messages.append({"role": role, "content": assistant_part})
                    elif format_type == "markdown":
                        messages.append(f"**{role.capitalize()}**: {assistant_part}\n")
                        
                    # 2. Append user_part
                    if format_type == "openai_chat":
                        messages.append({"role": "user", "content": user_part})
                    elif format_type == "ollama_chat":
                        messages.append({"role": "user", "content": user_part})
                    elif format_type == "markdown":
                        messages.append(f"**User**: {user_part}\n")
                        
                remaining_assistant_part = parts[2*num_calls] if 2*num_calls < len(parts) else ""
                if remaining_assistant_part.strip():
                    clean_rem = re.sub(r'<processing.*?>.*?</processing>', '', remaining_assistant_part, flags=re.DOTALL | re.IGNORECASE)
                    clean_rem = re.sub(r'<processing[^>]*>', '', clean_rem, flags=re.IGNORECASE).replace("</processing>", "")
                    if format_type == "openai_chat":
                        messages.append({"role": role, "content": clean_rem.strip()})
                    elif format_type == "ollama_chat":
                        messages.append({"role": role, "content": clean_rem.strip()})
                    elif format_type == "markdown":
                        messages.append(f"**{role.capitalize()}**: {clean_rem.strip()}\n")
            else:
                content = get_full_content(msg)
                active_images_b64 = msg.get_active_images()
                images_dicts = build_image_dicts(active_images_b64)
                if format_type == "openai_chat":
                    if images_dicts:
                        parts = [{"type":"text","text":content}] if content else []
                        for img in images_dicts:
                            url = f"data:image/jpeg;base64,{img['data']}" if img['type']=='base64' else img['data']
                            parts.append({"type":"image_url","image_url":{"url":url,"detail":"auto"}})
                        messages.append({"role":role,"content":parts})
                    else:
                        messages.append({"role":role,"content":content})
                elif format_type == "ollama_chat":
                    md = {"role":role,"content":content}
                    b64s = [i['data'] for i in images_dicts if i['type']=='base64']
                    if b64s:
                        md["images"] = b64s
                    messages.append(md)
                elif format_type == "markdown":
                    line = f"**{role.capitalize()}**: {content}\n"
                    if images_dicts and not suppress_images:
                        for img in images_dicts:
                            url = f"![Image](data:image/jpeg;base64,{img['data']})" if img['type']=='base64' else f"![Image]({img['data']})"
                            line += f"\n{url}\n"
                    messages.append(line)
                else:
                    raise ValueError(f"Unsupported format_type: {format_type}")

            # Inject scratchpad ONLY when non-empty, right after last user message
            # For OpenAI-compatible APIs with strict templates, use a user message
            # with special markers rather than a system message mid-conversation
            if _scratchpad and idx == last_user_idx:
                scratch_content = (
                    "== TOOL OUTPUT SCRATCHPAD ==\n"
                    f"{_scratchpad}\n"
                    "== END SCRATCHPAD =="
                )
                if format_type == "openai_chat":
                    # Use user role with marker to avoid "system role not supported here" errors
                    messages.append({
                        "role": "user",
                        "content": f"[SYSTEM CONTEXT - TOOL OUTPUTS]\n{scratch_content}\n[/SYSTEM CONTEXT]"
                    })
                elif format_type == "ollama_chat":
                    messages.append({
                        "role": "user",
                        "content": f"[SYSTEM CONTEXT - TOOL OUTPUTS]\n{scratch_content}\n[/SYSTEM CONTEXT]"
                    })
                elif format_type == "markdown":
                    messages.append(f"**system**: {scratch_content}\n")

        # ── Memory context injection ─────────────────────────────────────
        _mm = getattr(self, 'memory_manager', None)
        if _mm is not None and format_type in ("openai_chat", "ollama_chat"):
            messages = self._inject_memory_into_messages(
                messages, _mm, format_type,
                token_counter=self.lollmsClient.count_tokens,
            )

        # ── OpenAI Format Validation & Normalization ─────────────────────
        # Ensure system message is at the beginning and user/assistant alternate
        if format_type == "openai_chat" and messages:
            messages = self._normalize_openai_messages(messages)

        # --- Diagnostic Output Logging ---
        if format_type == "markdown":
            res_val = "\n".join(messages) if format_type == "markdown" else messages
            ASCIIColors.success(f"📊 [EXPORT DIAGNOSTIC END] Markdown final length: {len(res_val)} chars")
        elif format_type == "lollms_text":
            pass
        else:
            pass

        return "\n".join(messages) if format_type == "markdown" else messages

    def _normalize_openai_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Normalize messages for OpenAI API compliance:
        1. Fuse all system messages into ONE at the beginning
        2. Ensure user/assistant messages alternate (merge consecutive same-role messages)
        3. Remove empty messages

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Normalized list of messages ready for OpenAI API
        """
        if not messages:
            return messages

        # ── DEBUG LOGGING: Log message roles before normalization ───────────

        normalized = []
        system_content_parts = []

        # Step 1: Extract all system messages
        non_system_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle multimodal content - extract text parts
                    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    system_content_parts.append("\n".join(text_parts))
                else:
                    system_content_parts.append(str(content))
            else:
                non_system_messages.append(msg)

        # Step 2: Create single fused system message if any system content exists
        if system_content_parts:
            fused_system_content = "\n\n".join(part for part in system_content_parts if part.strip())
            if fused_system_content.strip():
                # Preserve images from first system message if present
                first_sys = next((m for m in messages if m.get("role") == "system"), {})
                first_content = first_sys.get("content", "")
                if isinstance(first_content, list):
                    # Keep image_url parts from first system message
                    image_parts = [item for item in first_content if item.get("type") == "image_url"]
                    normalized.append({
                        "role": "system",
                        "content": [{"type": "text", "text": fused_system_content}] + image_parts
                    })
                else:
                    normalized.append({"role": "system", "content": fused_system_content})

        # Step 3: Merge consecutive messages of the same role (user/assistant)
        if non_system_messages:
            current_role = None
            current_content = []
            current_images = []

            for msg in non_system_messages:
                role = msg.get("role")
                content = msg.get("content", "")

                # Skip empty messages
                if not content and not msg.get("images"):
                    continue

                if role == current_role:
                    # Merge with current message
                    if isinstance(content, list):
                        # Multimodal content
                        for item in content:
                            if item.get("type") == "text":
                                current_content.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                current_images.append(item)
                    else:
                        current_content.append(str(content))

                    # Merge images from consecutive messages
                    if msg.get("images"):
                        current_images.extend(msg["images"])
                else:
                    # Flush previous message if exists
                    if current_role is not None and current_content:
                        merged_content = "\n\n".join(c for c in current_content if c.strip())
                        if merged_content.strip():
                            if current_images:
                                # Reconstruct multimodal content
                                text_part = {"type": "text", "text": merged_content}
                                normalized.append({
                                    "role": current_role,
                                    "content": [text_part] + current_images
                                })
                            else:
                                normalized.append({"role": current_role, "content": merged_content})

                    # Start new message
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

            # Flush last message
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

        # Step 4: Ensure first non-system message is user (not assistant)
        # If it starts with assistant, prepend a minimal user prompt
        non_sys_start = 0
        for i, msg in enumerate(normalized):
            if msg.get("role") != "system":
                non_sys_start = i
                break

        if non_sys_start < len(normalized):
            first_non_sys = normalized[non_sys_start]
            if first_non_sys.get("role") == "assistant":
                # Insert a minimal user message to maintain alternation
                normalized.insert(non_sys_start, {
                    "role": "user",
                    "content": "Continue."
                })

        # Step 5: Validate alternation (debug check)
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

        # ── DEBUG LOGGING: Log message roles after normalization ───────────
        discussion_logger.info("AFTER normalization:")
        for i, msg in enumerate(normalized):
            role = msg.get("role", "unknown")
            content_preview = ""
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [item.get("text", "")[:50] for item in content if item.get("type") == "text"]
                content_preview = " | ".join(text_parts)[:100]
            else:
                content_preview = str(content)[:100]
            discussion_logger.info(f"  [{i}] role={role} | content_len={len(content_preview)} | preview={content_preview!r}")
        discussion_logger.info(f"[OpenAI Export] Total messages after: {len(normalized)}")

        # ── DEBUG LOGGING: Verify system message is at beginning ───────────
        if normalized and normalized[0].get("role") != "system":
            discussion_logger.error(
                f"[OpenAI Export] CRITICAL: First message role is '{normalized[0].get('role')}', "
                f"NOT 'system'. This will cause 'System message must be at the beginning' error."
            )
        else:
            discussion_logger.info("[OpenAI Export] System message correctly positioned at beginning.")

        return normalized
    

    
    def summarize_and_prune(self, max_tokens=None, preserve_last_n=4, force_technical=False):
        """
        Generates a persistent technical synopsis and prunes the context.
        If force_technical=True, it generates a state-based synopsis instead of a prose summary.
        """
        branch_tip_id = self.active_branch_id
        if not branch_tip_id:
            return

        branch = self.get_branch(branch_tip_id)

        # Calculate fingerprint to detect changes
        import hashlib
        fingerprint = hashlib.sha256("".join([f"{m.id}:{hash(m.content)}" for m in branch]).encode()).hexdigest()

        meta = dict(self.metadata or {})
        if not force_technical and max_tokens:
            current_text = self.export("lollms_text", branch_tip_id, 999999)
            if self.lollmsClient.count_tokens(current_text) <= max_tokens:
                return

        if meta.get("last_synopsis_fingerprint") == fingerprint and self.pruning_summary:
            return # Cache hit, nothing changed

        if len(branch) <= preserve_last_n and not force_technical:
            return

        # Technical Synopsis Prompt
        to_sum = branch[:-preserve_last_n] if not force_technical else branch
        text_to_sum = "\n\n".join(f"{m.sender}: {m.content}" for m in to_sum)

        prompt = (
            "You are a Technical State Auditor. Generate a 'Project State Synopsis'.\n"
            "1. List all technical decisions made.\n"
            "2. Identify the current goal and any constraints.\n"
            "3. Summarize the state of any code or document logic discussed.\n"
            "4. IGNORE all natural language greetings or prose logs.\n"
            "Return a dense, technical block. DO NOT use conversational filler.\n\n"
            f"--- LOGS ---\n{text_to_sum}\n--- SYNOPSIS:"
        )

        try:
            synopsis = self.lollmsClient.generate_text(prompt, n_predict=1024, temperature=0.1)
            self.pruning_summary = synopsis.strip()
            self.pruning_point_id = branch[-preserve_last_n].id if not force_technical else branch[-1].id
            meta["last_synopsis_fingerprint"] = fingerprint
            self.metadata = meta
            self.touch()
            ASCIIColors.success("[Tenacious Memory] Persistent Technical Synopsis updated.")
        except Exception as e:
            trace_exception(e)

    def memorize(self, branch_tip_id=None):
        try:
            ctx = self.export("markdown", branch_tip_id=branch_tip_id)
            if not ctx.strip():
                return None
            
            # Align with mock client assertions in tests by directly invoking generate_text
            prompt = (
                "Extract technical content (equations, code, solutions) for future reference "
                "from the following conversation:\n\n" + ctx
            )
            response = self.lollmsClient.generate_text(
                prompt,
                system_prompt="Extract detailed technical content. Return JSON only.",
                temperature=0.1
            )
            if not response or not isinstance(response, str):
                return None
            
            # Format to include both the text and the mandatory "Memory entry from" header expected by the tests
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"--- Memory entry from {timestamp} ---\n{response.strip()}"
            if self.memory:
                self.memory = self.memory.rstrip() + "\n\n" + entry
            else:
                self.memory = entry
            
            self.touch()
            return response
        except Exception as e:
            trace_exception(e)
            return None

    def count_discussion_tokens(self, branch_tip_id=None) -> int:
        """Reliably count all tokens currently in the active context."""
        status = self.get_context_status(branch_tip_id)
        return status["current_tokens"]

    def get_context_status(self, branch_tip_id=None) -> Dict[str, Any]:
        """
        Provides a detailed breakdown of token usage across all context zones.
        Ensures all content (zones, scratchpad, grouped artifacts, history) is contabilized.
        Optimized with persistent, change-invalidated caching to prevent network floods.
        """
        import hashlib
        max_ctx = self.max_context_size or 8192
        result = {
            "max_tokens": max_ctx,
            "current_tokens": 0,
            "percent": 0.0,
            "zones": {}
        }
        tokenizer = self.lollmsClient.count_tokens
        
        # Lazy-initialize a persistent token cache in the discussion metadata dict
        meta = dict(self.metadata or {})
        token_cache = meta.setdefault("_token_cache", {})
        cache_dirty = False

        def _get_cached_tokens(text_block: str, category_key: str) -> int:
            nonlocal cache_dirty
            if not text_block:
                return 0
            # Use MD5 hash of content as the cache invalidation key
            h = hashlib.md5(text_block.encode('utf-8', errors='ignore')).hexdigest()
            cache_entry = token_cache.get(category_key, {})
            if cache_entry.get("hash") == h:
                return cache_entry["tokens"]
            
            # Recalculate only on change
            count = tokenizer(text_block)
            token_cache[category_key] = {"hash": h, "tokens": count}
            cache_dirty = True
            return count

        # ── 1. System & Data Zones ──────────────────────────────────────────
        system_prompt_text = (self._system_prompt or "").strip()
        pruning_block = ""
        if self.pruning_summary and self.pruning_point_id:
            pruning_block = f"--- Conversation Summary ---\n{self.pruning_summary.strip()}"
        
        # Build individual zone breakdown
        zone_breakdown = {}
        zone_map = [
            ("system_prompt", system_prompt_text),
            ("memory", (self.memory or "")),
            ("user_data_zone", (self.user_data_zone or "")),
            ("discussion_data_zone", (self.discussion_data_zone or "")),
            ("personality_data_zone", (self.personality_data_zone or "")),
            ("scratchpad", (getattr(self, "scratchpad", "") or "")),
            ("pruning_summary", (pruning_block or ""))
        ]

        for key, text_val in zone_map:
            val = (text_val or "").strip()
            if val:
                zone_breakdown[key] = {"tokens": _get_cached_tokens(val, f"zone_{key}")}

        _mm = getattr(self, "memory_manager", None)
        if _mm:
            working_txt = _mm.build_working_zone(token_counter=tokenizer)
            deep_txt = _mm.build_handles_zone(token_counter=tokenizer)
            if working_txt: 
                zone_breakdown["working_memory"] = {"tokens": _get_cached_tokens(working_txt, "working_mem")}
            if deep_txt: 
                zone_breakdown["deep_memory"] = {"tokens": _get_cached_tokens(deep_txt, "deep_mem_handles")}

        # ── 2. Artefacts Grouped Breakdown ──────────────────────────────────
        active_artefacts = self.artefacts.list(active_only=True)
        active_artefacts_by_type = {}
        total_art_tokens = 0

        for art in active_artefacts:
            atype = art.get('type', 'document')
            # Extract logical .lam content for data/structured files to ensure token audits align with LLM vision
            content = self.artefacts._get_lam_content(art).strip()
            if not content and not art.get('url'):
                continue

            art_tokens = art.get("token_count")
            # Always recalculate or use cached .lam tokens for data artifacts
            if art_tokens is None or atype == ArtefactType.DATA:
                lang = art.get('language') or ''
                header = f"###[{atype.capitalize()}] {art['title']} (v{art['version']})\n"
                fence = f"```{lang}\n{content}\n```" if content else ""
                art_block = header + fence
                art_tokens = _get_cached_tokens(art_block, f"art_{art['title']}_v{art.get('version', 1)}")
                art["token_count"] = art_tokens
                cache_dirty = True

            if atype not in active_artefacts_by_type:
                active_artefacts_by_type[atype] = {"tokens": 0, "count": 0}

            active_artefacts_by_type[atype]["tokens"] += art_tokens
            active_artefacts_by_type[atype]["count"] += 1
            total_art_tokens += art_tokens

        if total_art_tokens > 0:
            zone_breakdown["artefacts"] = {
                "tokens": total_art_tokens,
                "types": active_artefacts_by_type
            }

        # Assembled System Context (including headers)
        full_data_zone = self.get_full_data_zone()

        # Append active memory context block if available so that system_context tokens include them
        mem_block = ""
        _mm = getattr(self, "memory_manager", None)
        if _mm:
            mem_block = self._build_memory_context_block(_mm, token_counter=tokenizer)

        full_sys_content = f"{system_prompt_text}\n\n{full_data_zone}".strip()
        if mem_block:
            full_sys_content += "\n\n=== ACTIVE MEMORIES ===\n" + mem_block
        if pruning_block:
            full_sys_content += "\n\n" + pruning_block

        sys_block_formatted = f"!@>system:\n{full_sys_content}\n"

        sys_tokens = _get_cached_tokens(sys_block_formatted, "full_system_context")
        result["zones"]["system_context"] = {
            "tokens": sys_tokens,
            "breakdown": zone_breakdown
        }

        # ── 3. Message History ──────────────────────────────────────────────
        branch_tip_id = branch_tip_id or self.active_branch_id
        history_tokens = 0
        history_breakdown = {"text_tokens": 0, "image_tokens": 0, "message_count": 0}
        
        if branch_tip_id:
            branch = self.get_branch(branch_tip_id)
            msgs_to_render = branch
            # Respect pruning point
            if self.pruning_summary and self.pruning_point_id:
                pi = next((i for i, m in enumerate(branch) if m.id == self.pruning_point_id), -1)
                if pi != -1:
                    msgs_to_render = branch[pi:]
            
            history_breakdown["message_count"] = len(msgs_to_render)
            
            for msg in msgs_to_render:
                sender_clean = msg.sender.replace(':', '').replace('!@>', '')
                content = msg.content.strip()
                
                # Handle Images in history
                active_imgs = msg.get_active_images()
                img_count = len(active_imgs)
                img_toks = 0
                if img_count > 0:
                    img_toks = sum(self.lollmsClient.count_image_tokens(img_data) for img_data in active_imgs)
                    history_breakdown["image_tokens"] += img_toks
                    content += f"\n({img_count} image(s) attached)"
                
                msg_text = f"!@>{sender_clean}:\n{content}\n"
                
                # Check persistent cache on the LollmsMessage object directly
                # Invalidate if message is still being modified or tokens not yet set
                if getattr(msg, "tokens", None) is not None and msg.tokens > 0:
                    msg_toks = msg.tokens
                else:
                    msg_toks = tokenizer(msg_text)
                    msg.tokens = msg_toks
                    cache_dirty = True
                
                history_breakdown["text_tokens"] += msg_toks
            
            history_tokens = history_breakdown["text_tokens"] + history_breakdown["image_tokens"]
            result["zones"]["message_history"] = {
                "tokens": history_tokens,
                "breakdown": history_breakdown
            }

        # ── 4. Global Discussion Images ─────────────────────────────────────
        disc_imgs = self.get_discussion_images()
        active_disc_imgs = [i for i in disc_imgs if i.get('active', True)]
        if active_disc_imgs:
            disc_img_tokens = sum(self.lollmsClient.count_image_tokens(i['data']) for i in active_disc_imgs)
            result["zones"]["discussion_images"] = {
                "tokens": disc_img_tokens,
                "count": len(active_disc_imgs)
            }

        # ── 5. Totals ───────────────────────────────────────────────────────
        total_tokens = sum(z.get("tokens", 0) for z in result["zones"].values())
        result["current_tokens"] = total_tokens
        result["percent"] = round((total_tokens / max_ctx) * 100, 2)

        if cache_dirty:
            self.metadata = meta
            self.touch()
            self.commit()

        return result

    def get_all_images(self, branch_tip_id=None):
        all_imgs = []
        branch = self.get_branch(branch_tip_id or self.active_branch_id)
        if not branch:
            return []
        for msg in branch:
            for i, img_info in enumerate(msg.get_all_images()):
                all_imgs.append({"message_id":msg.id,"index":i,
                                  "data":img_info["data"],"active":img_info["active"]})
        return all_imgs

    def get_active_images(self, branch_tip_id=None):
        """
        Returns all active images for the chat context:
        discussion-level images + per-message active images + active artifact images.
        """
        discussion_imgs = self.get_discussion_images()
        active = [i['data'] for i in discussion_imgs if i.get('active', True)]
        branch = self.get_branch(branch_tip_id or self.active_branch_id)
        if branch:
            for msg in branch:
                active.extend(msg.get_active_images())

        # Merge active image-type artifacts so the LLM gets the selected version's pixels
        active_art_images = self.artefacts.get_context_images()
        for img in active_art_images:
            if img.get("data") and img["data"] not in active:
                active.append(img["data"])

        return active

    def switch_to_branch(self, branch_id):
        if branch_id not in self._message_index:
            ASCIIColors.warning(f"Non-existent branch ID: {branch_id}")
            return
        # Set active branch to the exact historical message ID requested to enable forking
        self.active_branch_id = branch_id
        self.touch()

    def auto_title(self):
        try:
            if self.metadata is None:
                self.metadata = {}
            discussion = self.export("markdown", suppress_system_prompt=True, suppress_images=True)[0:1000]
            infos = self.lollmsClient.generate_structured_content(
                prompt=f"Build a title for:\n{discussion}",
                system_prompt="You are a title builder.",
                schema={"title": "Short catchy title for the discussion."}
            )
            if infos is None or "title" not in infos:
                raise ValueError("Title generation failed.")
            title = infos["title"]
            new_meta = (self.metadata or {}).copy()
            new_meta['title'] = title
            self.metadata = new_meta
            self.commit()
            return title
        except Exception as ex:
            trace_exception(ex)

    def set_metadata_item(self, itemname, item_value):
        new_meta = (self.metadata or {}).copy()
        new_meta[itemname] = item_value
        self.metadata = new_meta
        self.commit()

    # ---------------------------------------- discussion-level image methods (unchanged)

    def add_discussion_image(self, image_b64, source="user", active=True):
        current = self.get_discussion_images()
        current.append({"data":image_b64,"source":source,"active":active,
                         "created_at":datetime.utcnow().isoformat()})
        self.images = current
        self.touch()

    def get_discussion_images(self):
        if not self.images or len(self.images)==0 or type(self.images) is not list:
            return []
        if isinstance(self.images[0], str):
            ASCIIColors.yellow(f"Discussion {self.id}: Upgrading legacy image format.")
            upgraded = [{"data":s,"source":"user","active":True,
                          "created_at":datetime.utcnow().isoformat()} for s in self.images]
            self.images = upgraded
            self.touch()
        return self.images

    def toggle_discussion_image_activation(self, index, active=None):
        current = self.get_discussion_images()
        if index >= len(current):
            raise IndexError("Discussion image index out of range.")
        current[index]["active"] = not current[index].get("active",True) if active is None else bool(active)
        self.images = current
        self.touch()

    def remove_discussion_image(self, index, commit=True):
        current = self.get_discussion_images()
        if index >= len(current):
            raise IndexError("Discussion image index out of range.")
        del current[index]
        self.images = current
        self.touch()
        if commit:
            self.commit()

    def fix_orphan_messages(self):
        ASCIIColors.info(f"Checking discussion {self.id} for orphans...")
        self._rebuild_message_index()
        all_msgs = list(self._message_index.values())
        if not all_msgs:
            return
        msg_map = {m.id: m for m in all_msgs}
        root_msgs = []
        children_map = {m.id: [] for m in all_msgs}
        for m in all_msgs:
            if m.parent_id is None:
                root_msgs.append(m)
            elif m.parent_id in msg_map:
                children_map[m.parent_id].append(m.id)
        root_msgs.sort(key=lambda m: m.created_at)
        primary_root = root_msgs[0] if root_msgs else None
        reachable = set()
        queue = [r.id for r in root_msgs]
        reachable.update(queue)
        qi = 0
        while qi < len(queue):
            cur = queue[qi]; qi += 1
            for cid in children_map.get(cur,[]):
                if cid not in reachable:
                    reachable.add(cid); queue.append(cid)
        orphans = set(msg_map.keys()) - reachable
        if not orphans:
            ASCIIColors.success("No orphans found.")
            return
        orphan_tops = set()
        for oid in orphans:
            cur = oid
            while msg_map[cur].parent_id is not None and msg_map[cur].parent_id in orphans:
                cur = msg_map[cur].parent_id
            orphan_tops.add(cur)
        sorted_tops = sorted([msg_map[t] for t in orphan_tops], key=lambda m: m.created_at)
        reparented = 0
        if not primary_root:
            if sorted_tops:
                sorted_tops[0].parent_id = None
                primary_root = sorted_tops[0]
                reparented += 1
                sorted_tops = sorted_tops[1:]
        if primary_root:
            for top in sorted_tops:
                if top.id != primary_root.id:
                    top.parent_id = primary_root.id
                    reparented += 1
        if reparented > 0:
            self.touch(); self.commit()
            self._rebuild_message_index(); self._validate_and_set_active_branch()

    # ---------------------------------------- property

    @property
    def system_prompt(self):
        return self._system_prompt

    # ---------------------------------------- legacy artefact shim methods
    # These delegate to self.artefacts so existing call sites keep working.

    def list_artefacts(self):
        return self.artefacts.list_artefacts()

    def add_artefact(self, title, content="", images=None, audios=None, videos=None,
                     zip_content=None, version=1, **extra_data):
        return self.artefacts.add_artefact(title, content, images, audios, videos,
                                           zip_content, version, **extra_data)

    def get_artefact(self, title, version=None):
        return self.artefacts.get_artefact(title, version)

    def update_artefact(self, title, new_content, new_images=None, **extra_data):
        return self.artefacts.update_artefact(title, new_content, new_images, **extra_data)

    def remove_artefact(self, title, version=None):
        return self.artefacts.remove_artefact(title, version)
    
    def export_artefact(self, title: str) -> Optional[Dict[str, Any]]:
        return self.artefacts.export_artefact(title)

    def import_artefact(self, artefact_data: Dict[str, Any], activate: bool = True) -> Optional[Dict[str, Any]]:
        return self.artefacts.import_artefact(artefact_data, activate=activate)
    
    def load_artefact_into_data_zone(self, title, version=None):
        """Legacy shim: activates the artefact (new system) and also patches discussion_data_zone for compat."""
        a = self.artefacts.get(title, version)
        if not a:
            raise ValueError(f"Artefact '{title}' not found.")
        self.artefacts.activate(title, version or a['version'])
        # Also inject into discussion_data_zone for tools that read it directly
        if a.get('content'):
            section = (
                f"--- Document: {a['title']} v{a['version']} ---\n"
                f"{a['content']}\n"
                f"--- End Document: {a['title']} ---\n\n"
            )
            if section not in (self.discussion_data_zone or ""):
                self.discussion_data_zone = (self.discussion_data_zone or "").rstrip() + "\n\n" + section
        self.touch(); self.commit()

    def unload_artefact_from_data_zone(self, title, version=None):
        """Legacy shim: deactivates the artefact and removes from discussion_data_zone."""
        a = self.artefacts.get(title, version)
        if not a:
            raise ValueError(f"Artefact '{title}' not found.")
        self.artefacts.deactivate(title, version or a['version'])
        if self.discussion_data_zone and a.get('content'):
            pattern = (rf"\n*\s*--- Document: {re.escape(a['title'])} v{a['version']} ---"
                       rf".*?--- End Document: {re.escape(a['title'])} ---\s*\n*")
            self.discussion_data_zone = re.sub(pattern, "", self.discussion_data_zone, flags=re.DOTALL).strip()
        self.touch(); self.commit()

    def is_artefact_loaded(self, title, version=None):
        a = self.artefacts.get(title, version)
        if not a:
            return False
        return a.get('active', False)

    def export_as_artefact(self, title, version=1, **extra_data):
        content = (self.discussion_data_zone or "").strip()
        if not content:
            raise ValueError("Discussion data zone is empty.")
        return self.artefacts.add(title=title, artefact_type=ArtefactType.DOCUMENT,
                                  content=content, version=version, **extra_data)

    def clone_without_messages(self):
        return LollmsDiscussion.create_new(
            lollms_client=self.lollmsClient, db_manager=self.db_manager,
            system_prompt=self.system_prompt,
            user_data_zone=self.user_data_zone,
            discussion_data_zone=self.discussion_data_zone,
            personality_data_zone=self.personality_data_zone,
            memory=self.memory, participants=self.participants,
            discussion_metadata=self.metadata,
            images=[i.copy() for i in self.get_discussion_images()],
        )

    def export_to_json_str(self):
        export_data = {
            "id": self.id, "system_prompt": self.system_prompt,
            "user_data_zone": self.user_data_zone,
            "discussion_data_zone": self.discussion_data_zone,
            "personality_data_zone": self.personality_data_zone,
            "memory": self.memory, "participants": self.participants,
            "active_branch_id": self.active_branch_id,
            "discussion_metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "pruning_summary": self.pruning_summary,
            "pruning_point_id": self.pruning_point_id,
            "images": self.get_discussion_images(),
            "messages": []
        }
        for msg in self.get_all_messages_flat():
            export_data["messages"].append({
                "id": msg.id, "discussion_id": msg.discussion_id, "parent_id": msg.parent_id,
                "sender": msg.sender, "sender_type": msg.sender_type,
                "raw_content": msg.raw_content, "thoughts": msg.thoughts, "content": msg.content,
                "scratchpad": msg.scratchpad, "tokens": msg.tokens,
                "binding_name": msg.binding_name, "model_name": msg.model_name,
                "generation_speed": msg.generation_speed, "message_metadata": msg.metadata,
                "images": msg.images, "active_images": msg.active_images,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            })
        return json.dumps(export_data, indent=2)

    @classmethod
    def import_from_json_str(cls, json_str, lollms_client, db_manager=None):
        data = json.loads(json_str)
        message_data_list = data.pop("messages", [])
        data.pop("active_images", None)
        new_discussion = cls.create_new(lollms_client=lollms_client, db_manager=db_manager, **data)
        for msg_data in message_data_list:
            if 'created_at' in msg_data and msg_data['created_at']:
                msg_data['created_at'] = datetime.fromisoformat(msg_data['created_at'])
            new_discussion.add_message(**msg_data)
        new_discussion.active_branch_id = data.get('active_branch_id')
        if db_manager:
            new_discussion.commit()
        return new_discussion
