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
from ._artefacts import ArtefactType


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
        if not branch_tip_id and format_type in ["lollms_text","openai_chat","ollama_chat","markdown"]:
            return "" if format_type in ["lollms_text","markdown"] else []
        branch = self.get_branch(branch_tip_id)
        system_prompt_part = (self._system_prompt or "").strip()
        data_zone_part     = self.get_full_data_zone()
        full_system_prompt = ""
        if not suppress_system_prompt:
            if system_prompt_part and data_zone_part:
                full_system_prompt = f"{system_prompt_part}\n\n{data_zone_part}"
            elif system_prompt_part:
                full_system_prompt = system_prompt_part
            else:
                full_system_prompt = data_zone_part
        participants = self.participants or {}

        # Resolve scratchpad — empty scratchpad is suppressed entirely so no
        # blank system message is ever injected into the context.
        _scratchpad = (getattr(self, 'scratchpad', None) or "").strip()

        def get_full_content(msg):
            return msg.content.strip()

        # Helper: find the forward index of the last user message in a list
        def _last_user_index(branch_list):
            for i in range(len(branch_list) - 1, -1, -1):
                if branch_list[i].sender_type == 'user':
                    return i
            return -1

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
                if _scratchpad and fwd_idx == last_user_idx:
                    scratch_text = (
                        "!@>system:\n"
                        "== TOOL OUTPUT SCRATCHPAD ==\n"
                        f"{_scratchpad}\n"
                        "== END SCRATCHPAD ==\n"
                    )
                    scratch_toks = self.lollmsClient.count_tokens(scratch_text)
                    if max_allowed_tokens is None or current_tokens + scratch_toks <= max_allowed_tokens:
                        # index 1 = immediately after the user msg prepended at [0]
                        message_parts.insert(1, scratch_text)
                        current_tokens += scratch_toks

            final_parts.extend(message_parts)
            return "".join(final_parts).strip()

        messages = []
        active_discussion_b64 = self.get_active_images(branch_tip_id=None)
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
            if _scratchpad and idx == last_user_idx:
                scratch_content = (
                    "== TOOL OUTPUT SCRATCHPAD ==\n"
                    f"{_scratchpad}\n"
                    "== END SCRATCHPAD =="
                )
                if format_type == "openai_chat":
                    messages.append({"role": "system", "content": scratch_content})
                elif format_type == "ollama_chat":
                    messages.append({"role": "system", "content": scratch_content})
                elif format_type == "markdown":
                    messages.append(f"**system**: {scratch_content}\n")

        return "\n".join(messages) if format_type == "markdown" else messages
    

    
    def summarize_and_prune(self, max_tokens, preserve_last_n=4):
        branch_tip_id = self.active_branch_id
        if not branch_tip_id:
            return
        current_text   = self.export("lollms_text", branch_tip_id, 999999)
        current_tokens = self.lollmsClient.count_tokens(current_text)
        if current_tokens <= max_tokens:
            return
        branch = self.get_branch(branch_tip_id)
        if len(branch) <= preserve_last_n:
            return
        to_prune = branch[:-preserve_last_n]
        pruning_point = branch[-preserve_last_n]
        text = "\n\n".join(f"{m.sender}: {m.content}" for m in to_prune)
        try:
            summary = self.lollmsClient.generate_text(
                f"Summarize concisely, capturing key facts/decisions:\n---\n{text}\n---\nSUMMARY:",
                n_predict=512, temperature=0.1
            )
        except Exception as e:
            # Reveal the logic failure in context pruning
            trace_exception(e)
            print(f"[WARNING] Pruning failed: {e}")
            return
        self.pruning_summary = ((self.pruning_summary or "") + f"\n\n--- Summary ---\n{summary.strip()}").strip()
        self.pruning_point_id = pruning_point.id
        self.touch()

    def memorize(self, branch_tip_id=None):
        try:
            ctx = self.export("markdown", branch_tip_id=branch_tip_id)
            if not ctx.strip():
                return None
            memory_json = self.lollmsClient.generate_structured_content(
                "Extract technical content (equations, code, solutions) for future reference:\n\n" + ctx,
                schema={"title":"str","content":"str"},
                system_prompt="Extract detailed technical content. Return JSON only.",
                temperature=0.1
            )
            if memory_json and memory_json.get("title") and memory_json.get("content"):
                return memory_json
            return None
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
        Ensures all content (zones, scratchpad, grouped artefacts, history) is contabilized.
        Images are fixed at 256 tokens each.
        """
        max_ctx = self.max_context_size or 8192
        result = {
            "max_tokens": max_ctx,
            "current_tokens": 0,
            "percent": 0.0,
            "zones": {}
        }
        tokenizer = self.lollmsClient.count_tokens
        
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
        
        for key, text in zone_map:
            val = (text or "").strip()
            if val:
                zone_breakdown[key] = {"tokens": tokenizer(val)}

        # ── 2. Artefacts Grouped Breakdown ──────────────────────────────────
        active_artefacts = self.artefacts.list(active_only=True)
        artefacts_by_type = {}
        total_art_tokens = 0
        
        for art in active_artefacts:
            atype = art.get('type', 'document')
            content = art.get('content', '').strip()
            if not content and not art.get('url'):
                continue
            
            # Replicate assembly logic from build_artefacts_context_zone for accurate counting
            lang = art.get('language') or ''
            header = f"###[{atype.capitalize()}] {art['title']} (v{art['version']})\n"
            fence = f"```{lang}\n{content}\n```" if content else ""
            art_block = header + fence
            
            art_tokens = tokenizer(art_block)
            if atype not in artefacts_by_type:
                artefacts_by_type[atype] = {"tokens": 0, "count": 0}
            
            artefacts_by_type[atype]["tokens"] += art_tokens
            artefacts_by_type[atype]["count"] += 1
            total_art_tokens += art_tokens

        if total_art_tokens > 0:
            zone_breakdown["artefacts"] = {
                "tokens": total_art_tokens,
                "types": artefacts_by_type
            }

        # Assembled System Context (including headers)
        full_data_zone = self.get_full_data_zone()
        full_sys_content = f"{system_prompt_text}\n\n{full_data_zone}\n\n{pruning_block}".strip()
        sys_block_formatted = f"!@>system:\n{full_sys_content}\n"
        
        sys_tokens = tokenizer(sys_block_formatted)
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
                
                # Handle Images in history (Fixed at 256 tokens)
                active_imgs = msg.get_active_images()
                img_count = len(active_imgs)
                if img_count > 0:
                    img_toks = img_count * 256
                    history_breakdown["image_tokens"] += img_toks
                    content += f"\n({img_count} image(s) attached)"
                
                msg_text = f"!@>{sender_clean}:\n{content}\n"
                history_breakdown["text_tokens"] += tokenizer(msg_text)
            
            history_tokens = history_breakdown["text_tokens"] + history_breakdown["image_tokens"]
            result["zones"]["message_history"] = {
                "tokens": history_tokens,
                "breakdown": history_breakdown
            }

        # ── 4. Global Discussion Images ─────────────────────────────────────
        disc_imgs = self.get_discussion_images()
        active_disc_imgs = [i for i in disc_imgs if i.get('active', True)]
        if active_disc_imgs:
            disc_img_tokens = len(active_disc_imgs) * 256
            result["zones"]["discussion_images"] = {
                "tokens": disc_img_tokens,
                "count": len(active_disc_imgs)
            }

        # ── 5. Totals ───────────────────────────────────────────────────────
        total_tokens = sum(z.get("tokens", 0) for z in result["zones"].values())
        result["current_tokens"] = total_tokens
        result["percent"] = round((total_tokens / max_ctx) * 100, 2)
        
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
        discussion-level images + per-message active images.

        Artefact images (``ArtefactType.IMAGE``) are intentionally excluded here.
        They live in ``message.images`` once generated, and can be accessed
        separately via ``self.artefacts.get_active_images()`` for UI/export use.
        """
        discussion_imgs = self.get_discussion_images()
        active = [i['data'] for i in discussion_imgs if i.get('active', True)]
        branch = self.get_branch(branch_tip_id or self.active_branch_id)
        if not branch:
            return active
        for msg in branch:
            active.extend(msg.get_active_images())
        return active

    def switch_to_branch(self, branch_id):
        if branch_id not in self._message_index:
            ASCIIColors.warning(f"Non-existent branch ID: {branch_id}")
            return
        new_id = self._find_deepest_leaf(branch_id)
        if new_id:
            self.active_branch_id = new_id
        else:
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
