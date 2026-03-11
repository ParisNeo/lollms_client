# lollms_discussion/_message.py
# LollmsMessage: lightweight proxy wrapper around a SQLAlchemy (or in-memory) message object.

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from lollms_discussion import LollmsDiscussion


class LollmsMessage:
    """A lightweight proxy wrapper for a message ORM object."""

    def __init__(self, discussion: 'LollmsDiscussion', db_message: Any):
        object.__setattr__(self, '_discussion', discussion)
        object.__setattr__(self, '_db_message', db_message)

    def __getattr__(self, name):
        if name == 'metadata':
            return getattr(self._db_message, 'message_metadata', {}) or {}
        if name == 'content':
            return getattr(self._db_message, 'content', "") or ""
        return getattr(self._db_message, name)

    def __setattr__(self, name, value):
        if name == 'metadata':
            setattr(self._db_message, 'message_metadata', value)
        else:
            setattr(self._db_message, name, value)
        self._discussion.touch()

    def __repr__(self):
        return f"<LollmsMessage id={self.id} sender='{self.sender}'>"

    # ------------------------------------------------------------------ images

    def get_all_images(self) -> List[Dict[str, Union[str, bool]]]:
        if not self.images:
            return []
        if self.active_images is None or not isinstance(self.active_images, list) \
                or len(self.active_images) != len(self.images):
            active_flags = [True] * len(self.images)
        else:
            active_flags = self.active_images
        return [{"data": img_data, "active": active_flags[i]}
                for i, img_data in enumerate(self.images)]

    def get_active_images(self) -> List[str]:
        if not self.images:
            return []
        if self.active_images is None or not isinstance(self.active_images, list):
            return self.images
        return [img for i, img in enumerate(self.images)
                if i < len(self.active_images) and self.active_images[i]]

    def _sync_active_images_flags(self):
        current_images = self.images or []
        if not current_images:
            self.active_images = []
            return
        metadata = self.metadata or {}
        groups = metadata.get("image_groups", []) + metadata.get("image_generation_groups", [])
        new_active_flags = [False] * len(current_images)
        grouped_indices = set()
        for group in groups:
            indices = group.get("indices", [])
            for i in indices:
                grouped_indices.add(i)
            is_group_active = group.get("is_active", True)
            if is_group_active:
                main_idx = group.get("main_image_index")
                if main_idx is None or main_idx not in indices:
                    if indices:
                        main_idx = indices[0]
                if main_idx is not None and 0 <= main_idx < len(new_active_flags):
                    new_active_flags[main_idx] = True
        for i in range(len(current_images)):
            if i not in grouped_indices:
                new_active_flags[i] = True
        self.active_images = new_active_flags

    def toggle_image_pack_activation(self, index: int, active: Optional[bool] = None):
        metadata = (self.metadata or {}).copy()
        groups = metadata.get("image_groups", []) + metadata.get("image_generation_groups", [])
        target_group = next((g for g in groups if index in g.get("indices", [])), None)
        if target_group:
            if active is not None:
                if active:
                    target_group["is_active"] = True
                    target_group["main_image_index"] = index
                else:
                    if target_group.get("main_image_index") == index:
                        target_group["is_active"] = False
            else:
                if target_group.get("main_image_index") == index:
                    target_group["is_active"] = not target_group.get("is_active", True)
                else:
                    target_group["main_image_index"] = index
                    target_group["is_active"] = True
            self.metadata = metadata
            self._sync_active_images_flags()
        else:
            new_group = {
                "id": str(uuid.uuid4()), "type": "upload", "indices": [index],
                "created_at": datetime.utcnow().isoformat(), "main_image_index": index,
                "is_active": active if active is not None else not (
                    self.active_images and self.active_images[index])
            }
            if "image_groups" not in metadata:
                metadata["image_groups"] = []
            metadata["image_groups"].append(new_group)
            self.metadata = metadata
            self._sync_active_images_flags()
        if self._discussion._is_db_backed:
            self._discussion.commit()

    def add_image_pack(self, images: List[str], group_type: str = "generated",
                       active_by_default: bool = True, title: Optional[str] = None,
                       prompt: Optional[str] = "") -> None:
        if not images:
            return
        current_images = self.images or []
        start_index = len(current_images)
        current_images.extend(images)
        self.images = current_images
        metadata = (self.metadata or {}).copy()
        groups = metadata.get("image_groups", [])
        new_indices = list(range(start_index, start_index + len(images)))
        main_image_idx = new_indices[0] if new_indices else None
        group_entry = {
            "id": str(uuid.uuid4()), "type": group_type, "indices": new_indices,
            "created_at": datetime.utcnow().isoformat(),
            "main_image_index": main_image_idx, "is_active": active_by_default
        }
        if title:
            group_entry["title"] = title
        if prompt:
            group_entry["prompt"] = prompt
        groups.append(group_entry)
        metadata["image_groups"] = groups
        self.metadata = metadata
        self._sync_active_images_flags()
        if self._discussion._is_db_backed:
            self._discussion.commit()

    def set_metadata_item(self, itemname: str, item_value, discussion):
        new_metadata = (self.metadata or {}).copy()
        new_metadata[itemname] = item_value
        self.metadata = new_metadata
        discussion.commit()
