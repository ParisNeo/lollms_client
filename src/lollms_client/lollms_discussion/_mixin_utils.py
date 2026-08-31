# lollms_discussion/_mixin_utils.py
# UtilsMixin: branch management, pruning, context status,
#             image helpers, metadata, legacy artefact shims, serialisation.

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_utilities import build_image_dicts
from lollms_client.lollms_artefact import ArtefactType, sanitize_artifact_filename
import ascii_colors as logging

# Create module-level loggers for easy access
logger = logging.getLogger(__name__)
discussion_logger = logging.getLogger("lollms_client.lollms_discussion._mixin_utils")

class UtilsMixin:
    """
    All utility methods: regenerate_branch, delete_branch, summarize_and_prune,
    memorize, get_context_status, image helpers, metadata helpers, legacy artefact shims,
    JSON serialisation / deserialisation, clone_without_messages, and fix_orphan_messages.
    """

    # ---- Branch Methods --------------------------------------------

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
            self.remove_message(target_id)
        elif target_msg.sender_type == 'user':
            user_msg_to_regenerate_from = target_msg
        else:
            raise ValueError(f"Unexpected sender type: '{target_msg.sender_type}'.")
        self.active_branch_id = user_msg_to_regenerate_from.id
        return self.chat(user_message="", add_user_message=False,
                         branch_tip_id=user_msg_to_regenerate_from.id, **kwargs)

    def delete_branch(self, message_id: str):
        return self.prune_branch(message_id)

    def _sanitize_for_user_view(self, content: str) -> str:
        """
        Sanitizes assistant message content for User/Database view.
        Removes functional tags and processing blocks, leaving only the final conversational text.
        """
        import re as _re
        content = _re.sub(r'<processing[^>]*>.*?</processing>', '', content, flags=_re.DOTALL | _re.IGNORECASE)
        return content.strip()

    def _apply_three_view_protocol(self, msg, raw_content: str, distance_from_end: int = 0) -> str:
        """
        Applies the Three-View Protocol to LLM context export.

        1. Recent Assistant Messages (Original View): Preserves raw XML tags for KV-cache alignment.
        2. Older Assistant Messages (Reduced View): Replaces functional tags with opaque placeholders.
        3. User Messages: Always preserved verbatim.
        """
        from ._context_sanitizer import sanitize_context_for_llm

        if msg.sender_type != 'assistant':
            return raw_content

        # C1: KV-Cache Preservation. Keep raw tags if marked as recent functional (distance 0).
        if distance_from_end == 0:
            return raw_content

        # C2 & C3: Context Diet & Anti-Mimicry. Sanitize older messages.
        return sanitize_context_for_llm(raw_content)

    def summarize_and_prune(self, max_tokens=None, preserve_last_n=4, force_technical=False):
        """
        Generates a persistent technical synopsis and prunes the context.
        If force_technical=True, it generates a state-based synopsis instead of a prose summary.
        """
        branch_tip_id = self.active_branch_id
        if not branch_tip_id:
            return

        branch = self.get_branch(branch_tip_id)

        import hashlib
        fingerprint = hashlib.sha256("".join([f"{m.id}:{hash(m.content)}" for m in branch]).encode()).hexdigest()

        meta = dict(self.metadata or {})
        if not force_technical and max_tokens:
            current_text = self.export("lollms_text", branch_tip_id, 999999)
            if self.lollmsClient.count_tokens(current_text) <= max_tokens:
                return

        if meta.get("last_synopsis_fingerprint") == fingerprint and self.pruning_summary:
            return 

        if len(branch) <= preserve_last_n and not force_technical:
            return

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
        status = self.get_context_status(branch_tip_id)
        return status["current_tokens"]

    def get_context_status(self, branch_tip_id=None) -> Dict[str, Any]:
        """
        Provides a detailed breakdown of token usage across all context zones.
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
        
        meta = dict(self.metadata or {})
        token_cache = meta.setdefault("_token_cache", {})
        cache_dirty = False

        def _get_cached_tokens(text_block: str, category_key: str) -> int:
            nonlocal cache_dirty
            if not text_block:
                return 0
            h = hashlib.md5(text_block.encode('utf-8', errors='ignore')).hexdigest()
            cache_entry = token_cache.get(category_key, {})
            if cache_entry.get("hash") == h:
                return cache_entry["tokens"]
            
            count = tokenizer(text_block)
            token_cache[category_key] = {"hash": h, "tokens": count}
            cache_dirty = True
            return count

        system_prompt_text = (self._system_prompt or "").strip()
        pruning_block = ""
        if self.pruning_summary and self.pruning_point_id:
            pruning_block = f"--- Conversation Summary ---\n{self.pruning_summary.strip()}"
        
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

        active_artefacts = self.artefacts.list(active_only=True)
        active_artefacts_by_type = {}
        total_art_tokens = 0

        for art in active_artefacts:
            atype = art.get('type', 'document')
            content = self.artefacts._get_lam_content(art).strip()
            if not content and not art.get('url'):
                continue

            art_tokens = art.get("token_count")
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

        full_data_zone = self.get_full_data_zone()

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

        branch_tip_id = branch_tip_id or self.active_branch_id
        history_tokens = 0
        history_breakdown = {"text_tokens": 0, "image_tokens": 0, "message_count": 0}
        
        if branch_tip_id:
            branch = self.get_branch(branch_tip_id)
            msgs_to_render = branch
            if self.pruning_summary and self.pruning_point_id:
                pi = next((i for i, m in enumerate(branch) if m.id == self.pruning_point_id), -1)
                if pi != -1:
                    msgs_to_render = branch[pi:]
            
            history_breakdown["message_count"] = len(msgs_to_render)
            
            for msg in msgs_to_render:
                sender_clean = msg.sender.replace(':', '').replace('!@>', '')
                content = msg.content.strip()
                
                active_imgs = msg.get_active_images()
                img_count = len(active_imgs)
                img_toks = 0
                if img_count > 0:
                    img_toks = sum(self.lollmsClient.count_image_tokens(img_data) for img_data in active_imgs)
                    history_breakdown["image_tokens"] += img_toks
                    content += f"\n({img_count} image(s) attached)"
                
                msg_text = f"!@>{sender_clean}:\n{content}\n"
                
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

        disc_imgs = self.get_discussion_images()
        active_disc_imgs = [i for i in disc_imgs if i.get('active', True)]
        if active_disc_imgs:
            disc_img_tokens = sum(self.lollmsClient.count_image_tokens(i['data']) for i in active_disc_imgs)
            result["zones"]["discussion_images"] = {
                "tokens": disc_img_tokens,
                "count": len(active_disc_imgs)
            }

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

        active_art_images = self.artefacts.get_context_images()
        for img in active_art_images:
            if img.get("data") and img["data"] not in active:
                active.append(img["data"])

        return active

    def switch_to_branch(self, branch_id):
        if branch_id not in self._message_index:
            ASCIIColors.warning(f"Non-existent branch ID: {branch_id}")
            return
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

    @property
    def system_prompt(self):
        return self._system_prompt

    # ── Legacy Artefact Shim Methods ─────────────────────────────────────
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
        a = self.artefacts.get(title, version)
        if not a:
            raise ValueError(f"Artefact '{title}' not found.")
        self.artefacts.activate(title, version or a['version'])
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
        from . import LollmsDiscussion
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

    # ── Standalone Artefact Archive (SAA) Methods ─────────────────────────────
    def get_standalone_archive_dir(self) -> Path:
        base_workspace = Path(self.workspace_path).parent if hasattr(self, 'workspace_path') and self.workspace_path else Path("./data_workspace")
        archive_dir = base_workspace / "standalone_artefacts"
        archive_dir.mkdir(parents=True, exist_ok=True)
        return archive_dir

    def save_artefact_to_global_archive(self, title: str) -> Path:
        archive_dir = self.get_standalone_archive_dir()
        safe_title = sanitize_artifact_filename(title)
        output_path = archive_dir / f"{safe_title}.laa"
        result_path = self.artefacts.export_artefact_to_archive(title, output_path)
        ASCIIColors.info(f"[UtilsMixin] Artefact '{title}' saved to global archive at {result_path}")
        return result_path

    def load_artefact_from_global_archive(self, title: str, activate: bool = True) -> Optional[Dict[str, Any]]:
        archive_dir = self.get_standalone_archive_dir()
        safe_title = sanitize_artifact_filename(title)
        laa_path = archive_dir / f"{safe_title}.laa"
        if not laa_path.exists():
            ASCIIColors.warning(f"[UtilsMixin] Artefact '{title}' not found in global archive at {laa_path}")
            return None
        result = self.artefacts.import_artefact_from_archive(laa_path, activate=activate)
        ASCIIColors.info(f"[UtilsMixin] Artefact '{title}' loaded from global archive into discussion.")
        return result

    def list_global_archive_artefacts(self) -> List[str]:
        archive_dir = self.get_standalone_archive_dir()
        return [f.stem for f in archive_dir.glob("*.laa")]

    # ── Artefact Library and Bundle (.lab) Methods ────────────────────────────
    def save_artefact_bundle_to_global_archive(self, paths: List[Union[str, Path]], bundle_name: Optional[str] = None, include_versions: bool = False) -> Path:
        archive_dir = self.get_standalone_archive_dir()
        safe_name = sanitize_artifact_filename(bundle_name) if bundle_name else f"bundle_{uuid.uuid4().hex[:6]}"
        output_path = archive_dir / f"{safe_name}.lab"
        result_path = self.artefacts.export_artefact_bundle(paths, output_path, include_versions=include_versions)
        ASCIIColors.info(f"[UtilsMixin] Artefact bundle saved to global archive at {result_path}")
        return result_path

    def load_artefact_bundle_from_global_archive(self, bundle_name: str, activate: bool = True) -> List[Dict[str, Any]]:
        archive_dir = self.get_standalone_archive_dir()
        safe_name = sanitize_artifact_filename(bundle_name)
        lab_path = archive_dir / f"{safe_name}.lab"
        if not lab_path.exists():
            ASCIIColors.warning(f"[UtilsMixin] Bundle '{bundle_name}' not found in global archive at {lab_path}")
            return []
        result = self.artefacts.import_artefact_bundle(lab_path, activate=activate)
        ASCIIColors.info(f"[UtilsMixin] Loaded {len(result)} artefacts from bundle '{bundle_name}' into discussion.")
        return result

    def list_global_archive_bundles(self) -> List[str]:
        archive_dir = self.get_standalone_archive_dir()
        return [f.stem for f in archive_dir.glob("*.lab")]

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

    def get_artefact_content(self, artefact_title_or_path: str, version: Optional[int] = None) -> Optional[str]:
        artefact = self.artefacts.get(artefact_title_or_path, version)
        if artefact is None:
            return None

        base_ws = Path(self.workspace_path) if hasattr(self, 'workspace_path') and self.workspace_path else Path("./data_workspace")
        ws_dir = base_ws / str(self.id) / "workspace_data"

        filename = artefact.get('title', '')

        if '/' in filename or '\\' in filename:
            file_path = ws_dir / filename
        else:
            file_path = ws_dir / filename

        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()

            ASCIIColors.info(f"[UtilsMixin] Artefact '{artefact_title_or_path}' file not found on disk, attempting sync...")
            self.artefacts._sync_to_disk_workspace(
                title=artefact.get("title", filename),
                content=artefact.get("content", ""),
                version=artefact.get("version", 1),
                atype=artefact.get("type", "document"),
                language=artefact.get("language")
            )

            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()

        except Exception as e:
            ASCIIColors.warning(f"[UtilsMixin] Failed to read artefact '{artefact_title_or_path}': {e}")

        if artefact.get("content"):
            return artefact["content"]

        return None