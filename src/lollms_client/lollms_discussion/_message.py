import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from lollms_discussion import LollmsDiscussion


class LollmsMessage:
    """A lightweight proxy wrapper for a message ORM object."""

    id: str
    discussion_id: str
    parent_id: Optional[str]
    sender: str
    sender_type: str
    content: str
    raw_content: Optional[str]
    thoughts: Optional[str]
    scratchpad: Optional[str]
    tokens: Optional[int]
    binding_name: Optional[str]
    model_name: Optional[str]
    generation_speed: Optional[float]
    images: List[str]
    active_images: List[Union[str, bool]]
    created_at: datetime
    metadata: Dict[str, Any]

    def __init__(self, discussion: 'LollmsDiscussion', db_message: Any):
        object.__setattr__(self, '_discussion', discussion)
        object.__setattr__(self, '_db_message', db_message)

    @property
    def content(self) -> str:
        """The textual content of the message."""
        return getattr(self._db_message, 'content', "") or ""

    @content.setter
    def content(self, value: str) -> None:
        setattr(self._db_message, 'content', value)
        self._discussion.touch()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Dictionary of message metadata."""
        return getattr(self._db_message, 'message_metadata', {}) or {}

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        setattr(self._db_message, 'message_metadata', value)
        self._discussion.touch()

    @property
    def id(self) -> str:
        """Unique UUID identifier of the message."""
        return getattr(self._db_message, 'id', "")

    @id.setter
    def id(self, value: str) -> None:
        setattr(self._db_message, 'id', value)
        self._discussion.touch()

    @property
    def parent_id(self) -> Optional[str]:
        """UUID of the parent message in the branch DAG."""
        return getattr(self._db_message, 'parent_id', None)

    @parent_id.setter
    def parent_id(self, value: Optional[str]) -> None:
        setattr(self._db_message, 'parent_id', value)
        self._discussion.touch()

    @property
    def discussion_id(self) -> str:
        """UUID of the owning discussion."""
        return getattr(self._db_message, 'discussion_id', "")

    @discussion_id.setter
    def discussion_id(self, value: str) -> None:
        setattr(self._db_message, 'discussion_id', value)
        self._discussion.touch()

    @property
    def sender(self) -> str:
        """Display name of the sender (e.g. 'user', 'assistant')."""
        return getattr(self._db_message, 'sender', "")

    @sender.setter
    def sender(self, value: str) -> None:
        setattr(self._db_message, 'sender', value)
        self._discussion.touch()

    @property
    def sender_type(self) -> str:
        """Type of sender ('user', 'assistant', 'system')."""
        return getattr(self._db_message, 'sender_type', "")

    @sender_type.setter
    def sender_type(self, value: str) -> None:
        setattr(self._db_message, 'sender_type', value)
        self._discussion.touch()

    @property
    def thoughts(self) -> Optional[str]:
        """Reasoning or thinking trace generated alongside the response."""
        return getattr(self._db_message, 'thoughts', None)

    @thoughts.setter
    def thoughts(self, value: Optional[str]) -> None:
        setattr(self._db_message, 'thoughts', value)
        self._discussion.touch()

    @property
    def scratchpad(self) -> Optional[str]:
        """Temporary scratchpad content captured during generation."""
        return getattr(self._db_message, 'scratchpad', None)

    @scratchpad.setter
    def scratchpad(self, value: Optional[str]) -> None:
        setattr(self._db_message, 'scratchpad', value)
        self._discussion.touch()

    @property
    def tokens(self) -> Optional[int]:
        """Token count of the message content."""
        return getattr(self._db_message, 'tokens', None)

    @tokens.setter
    def tokens(self, value: Optional[int]) -> None:
        setattr(self._db_message, 'tokens', value)
        self._discussion.touch()

    @property
    def binding_name(self) -> Optional[str]:
        """Name of the LLM binding that produced this message."""
        return getattr(self._db_message, 'binding_name', None)

    @binding_name.setter
    def binding_name(self, value: Optional[str]) -> None:
        setattr(self._db_message, 'binding_name', value)
        self._discussion.touch()

    @property
    def model_name(self) -> Optional[str]:
        """Name of the model that produced this message."""
        return getattr(self._db_message, 'model_name', None)

    @model_name.setter
    def model_name(self, value: Optional[str]) -> None:
        setattr(self._db_message, 'model_name', value)
        self._discussion.touch()

    @property
    def generation_speed(self) -> Optional[float]:
        """Generation speed in tokens per second."""
        return getattr(self._db_message, 'generation_speed', None)

    @generation_speed.setter
    def generation_speed(self, value: Optional[float]) -> None:
        setattr(self._db_message, 'generation_speed', value)
        self._discussion.touch()

    @property
    def created_at(self) -> Optional[datetime]:
        """Timestamp when the message was created."""
        return getattr(self._db_message, 'created_at', None)

    @created_at.setter
    def created_at(self, value: Optional[datetime]) -> None:
        setattr(self._db_message, 'created_at', value)
        self._discussion.touch()

    def __getattr__(self, name):
        if name == 'metadata':
            return getattr(self._db_message, 'message_metadata', {}) or {}
        if name == 'content':
            return getattr(self._db_message, 'content', "") or ""
        return getattr(self._db_message, name)

    def __setattr__(self, name, value):
        prop = getattr(type(self), name, None)
        if isinstance(prop, property) and prop.fset is not None:
            prop.fset(self, value)
            return
        if name == 'metadata':
            setattr(self._db_message, 'message_metadata', value)
        else:
            setattr(self._db_message, name, value)
        self._discussion.touch()

    def __repr__(self):
        return f"<LollmsMessage id={self.id} sender='{self.sender}'>"

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

    def toggle_image_activation(self, index: int, active: Optional[bool] = None):
        """Toggles the activation state of an individual image index."""
        current_images = self.images or []
        if index < 0 or index >= len(current_images):
            return
        
        if self.active_images is None or not isinstance(self.active_images, list) or len(self.active_images) != len(current_images):
            flags = [True] * len(current_images)
        else:
            flags = list(self.active_images)
            
        if active is not None:
            flags[index] = bool(active)
        else:
            flags[index] = not flags[index]
            
        self.active_images = flags
        if self._discussion._is_db_backed:
            self._discussion.commit()

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
