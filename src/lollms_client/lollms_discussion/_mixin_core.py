# lollms_discussion/_mixin_core.py
# CoreMixin: __init__, class factories, __getattr__/__setattr__ proxy,
# internal DB helpers, message CRUD, data-zone assembly, and discussion-level image methods.

import re
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy.orm.exc import NoResultFound
from ascii_colors import ASCIIColors, trace_exception

from ._artefacts import ArtefactManager, ArtefactType
from ._message import LollmsMessage

if TYPE_CHECKING:
    from lollms_client import LollmsClient
    from ._db import LollmsDataManager


class CoreMixin:
    """
    Handles construction, ORM proxy, internal bookkeeping,
    message CRUD, and data-zone helpers.
    """

    # Class-level annotation so IDEs resolve self.artefacts without __getattr__.
    artefacts: ArtefactManager

    # ---------------------------------------------------------------- __init__

    def __init__(
        self,
        lollmsClient: 'LollmsClient',
        db_manager: Optional['LollmsDataManager'] = None,
        discussion_id: Optional[str] = None,
        db_discussion_obj: Optional[Any] = None,
        autosave: bool = False,
        max_context_size: Optional[int] = None,
    ):
        object.__setattr__(self, 'lollmsClient', lollmsClient)
        object.__setattr__(self, 'db_manager', db_manager)
        object.__setattr__(self, 'autosave', autosave)
        object.__setattr__(self, 'max_context_size', max_context_size)
        object.__setattr__(self, 'scratchpad', "")
        object.__setattr__(self, '_session', None)
        object.__setattr__(self, '_db_discussion', None)
        object.__setattr__(self, '_message_index', None)
        object.__setattr__(self, '_messages_to_delete_from_db', set())
        object.__setattr__(self, '_is_db_backed', db_manager is not None)

        if self._is_db_backed:
            if not db_discussion_obj and not discussion_id:
                raise ValueError("Either discussion_id or db_discussion_obj must be provided.")
            self._session = db_manager.get_session()
            if db_discussion_obj:
                self._db_discussion = self._session.merge(db_discussion_obj)
            else:
                try:
                    self._db_discussion = self._session.query(
                        db_manager.DiscussionModel).filter_by(id=discussion_id).one()
                except NoResultFound:
                    self._session.close()
                    raise ValueError(f"No discussion found with ID: {discussion_id}")
        else:
            self._create_in_memory_proxy(id=discussion_id)

        object.__setattr__(self, '_system_prompt',
                           getattr(self._db_discussion, 'system_prompt', None))

        metadata = getattr(self._db_discussion, 'discussion_metadata', {}) or {}
        images_data = metadata.get("discussion_images", [])
        object.__setattr__(self, 'images', images_data)

        object.__setattr__(self, 'artefacts', ArtefactManager(self))

        self._rebuild_message_index()
        self._validate_and_set_active_branch()
        self.get_discussion_images()

    # ---------------------------------------------------------------- factories

    @classmethod
    def from_messages(cls, messages, lollms_client, db_manager=None, **kwargs):
        discussion = cls.create_new(lollms_client=lollms_client, db_manager=db_manager, **kwargs)
        last_message_id = None
        for msg_data in messages:
            if isinstance(msg_data, SimpleNamespace):
                msg_kwargs = msg_data.__dict__.copy()
            elif isinstance(msg_data, dict):
                msg_kwargs = msg_data.copy()
            else:
                raise TypeError("message objects must be of type dict or SimpleNamespace")
            msg_kwargs['parent_id'] = last_message_id
            new_msg = discussion.add_message(**msg_kwargs)
            last_message_id = new_msg.id
        return discussion

    @classmethod
    def create_new(cls, lollms_client, db_manager=None, **kwargs):
        init_args = {
            'autosave': kwargs.pop('autosave', False),
            'max_context_size': kwargs.pop('max_context_size', None)
        }
        if db_manager:
            with db_manager.get_session() as session:
                valid_keys = db_manager.DiscussionModel.__table__.columns.keys()
                db_creation_args = {k: v for k, v in kwargs.items() if k in valid_keys}
                db_discussion_orm = db_manager.DiscussionModel(**db_creation_args)
                session.add(db_discussion_orm)
                session.commit()
                session.expunge(db_discussion_orm)
            return cls(lollmsClient=lollms_client, db_manager=db_manager,
                       db_discussion_obj=db_discussion_orm, **init_args)
        else:
            return cls(lollmsClient=lollms_client, discussion_id=kwargs.get('id'), **init_args)

    # ---------------------------------------------------------------- proxying

    def get_messages(self, branch_id=None):
        leaf_id = branch_id if branch_id is not None else self.active_branch_id
        return self.get_branch(leaf_id)

    def __getattr__(self, name):
        if name == 'metadata':
            return getattr(self._db_discussion, 'discussion_metadata', None)
        if name == 'messages':
            return [LollmsMessage(self, msg) for msg in self._db_discussion.messages]
        return getattr(self._db_discussion, name)

    def __setattr__(self, name, value):
        internal_attrs = [
            'lollmsClient', 'db_manager', 'autosave', 'max_context_size', 'scratchpad',
            'images', 'artefacts',
            '_session', '_db_discussion', '_message_index', '_messages_to_delete_from_db',
            '_is_db_backed', '_system_prompt',
        ]
        if name in internal_attrs:
            object.__setattr__(self, name, value)
        else:
            if name == 'system_prompt':
                object.__setattr__(self, '_system_prompt', value)
            if name == 'metadata':
                setattr(self._db_discussion, 'discussion_metadata', value)
            else:
                setattr(self._db_discussion, name, value)
            self.touch()

    # ------------------------------------------------- in-memory proxy builder

    def _create_in_memory_proxy(self, id=None):
        proxy = SimpleNamespace()
        proxy.id = id or str(uuid.uuid4())
        proxy.system_prompt = None
        proxy.user_data_zone = None
        proxy.discussion_data_zone = None
        proxy.personality_data_zone = None
        proxy.memory = None
        proxy.participants = {}
        proxy.active_branch_id = None
        proxy.discussion_metadata = {}
        proxy.created_at = datetime.utcnow()
        proxy.updated_at = datetime.utcnow()
        proxy.messages = []
        proxy.pruning_summary = None
        proxy.pruning_point_id = None
        object.__setattr__(self, '_db_discussion', proxy)

    # ------------------------------------------------- message index helpers

    def _rebuild_message_index(self):
        if self._is_db_backed and self._session and self._session.is_active \
                and self._db_discussion in self._session:
            self._session.refresh(self._db_discussion, ['messages'])
        self._message_index = {msg.id: msg for msg in self._db_discussion.messages}

    def _find_deepest_leaf(self, start_id):
        if not self._message_index:
            return None
        self._rebuild_message_index()
        children_of = {msg_id: [] for msg_id in self._message_index.keys()}
        for msg_id, msg_obj in self._message_index.items():
            if msg_obj.parent_id in children_of:
                children_of[msg_obj.parent_id].append(msg_id)

        def get_most_recent_leaf_from_list(message_list):
            if not message_list:
                return None
            leaves = [msg for msg in message_list if not children_of.get(msg.id)]
            if leaves:
                return max(leaves, key=lambda msg: msg.created_at).id
            return None

        if start_id and start_id in self._message_index:
            queue = [self._message_index[start_id]]
            visited = {start_id}
            descendants = [self._message_index[start_id]]
            head = 0
            while head < len(queue):
                cur = queue[head]; head += 1
                for child_id in children_of.get(cur.id, []):
                    if child_id not in visited:
                        visited.add(child_id)
                        child_obj = self._message_index[child_id]
                        queue.append(child_obj)
                        descendants.append(child_obj)
            result = get_most_recent_leaf_from_list(descendants)
            if result:
                return result
            if not children_of.get(start_id):
                return start_id
            return None
        else:
            return get_most_recent_leaf_from_list(list(self._message_index.values()))

    def _validate_and_set_active_branch(self):
        self._rebuild_message_index()
        if not self._message_index:
            object.__setattr__(self._db_discussion, 'active_branch_id', None)
            return
        current_active_id = self._db_discussion.active_branch_id
        if current_active_id is None or current_active_id not in self._message_index:
            new_id = self._find_deepest_leaf(None)
            if new_id:
                object.__setattr__(self._db_discussion, 'active_branch_id', new_id)
        else:
            children = [m.id for m in self._message_index.values()
                        if m.parent_id == current_active_id]
            if children:
                new_id = self._find_deepest_leaf(current_active_id)
                if new_id and new_id != current_active_id:
                    object.__setattr__(self._db_discussion, 'active_branch_id', new_id)

    # -------------------------------------------------------- DB session helpers

    def touch(self):
        metadata = (getattr(self._db_discussion, 'discussion_metadata', {}) or {}).copy()
        if self.images or "discussion_images" in metadata:
            metadata["discussion_images"] = self.images
            setattr(self._db_discussion, 'discussion_metadata', metadata)
        setattr(self._db_discussion, 'updated_at', datetime.utcnow())
        if self._is_db_backed and self.autosave:
            self.commit()

    def commit(self):
        if not self._is_db_backed or not self._session:
            return
        if self._messages_to_delete_from_db:
            for msg_id in self._messages_to_delete_from_db:
                msg_to_del = self._session.get(self.db_manager.MessageModel, msg_id)
                if msg_to_del:
                    self._session.delete(msg_to_del)
            self._messages_to_delete_from_db.clear()
        try:
            self._session.commit()
            self._rebuild_message_index()
        except Exception as e:
            # Trace the specific DB error before rolling back
            trace_exception(e)
            self._session.rollback()
            raise e

    def close(self):
        if self._session:
            self.commit()
            self._session.close()

    # -------------------------------------------------------- message CRUD

    def add_message(self, **kwargs) -> LollmsMessage:
        msg_id = kwargs.get('id', str(uuid.uuid4()))
        parent_id = kwargs.get('parent_id', self.active_branch_id)
        if 'images' in kwargs and kwargs['images'] and 'active_images' not in kwargs:
            kwargs['active_images'] = [True] * len(kwargs['images'])
        kwargs.setdefault('images', [])
        kwargs.setdefault('active_images', [])
        if 'sender_type' not in kwargs:
            kwargs['sender_type'] = 'user' if kwargs.get('sender') == 'user' else 'assistant'
        if kwargs.get('sender_type') == 'user':
            sender_name = kwargs.get('sender')
            sender_icon = kwargs.get('sender_icon')
            if sender_name:
                if self.participants is None:
                    self.participants = {}
                if sender_name not in self.participants or self.participants[sender_name].get('icon') is None:
                    self.participants[sender_name] = {"icon": sender_icon, "name": sender_name}
                    self.touch()
        message_data = {'id': msg_id, 'parent_id': parent_id, 'discussion_id': self.id,
                        'created_at': datetime.utcnow(), **kwargs}
        if 'metadata' in message_data:
            message_data['message_metadata'] = message_data.pop('metadata')
        if self._is_db_backed:
            valid_keys = {c.name for c in self.db_manager.MessageModel.__table__.columns}
            filtered_data = {k: v for k, v in message_data.items() if k in valid_keys}
            new_msg_orm = self.db_manager.MessageModel(**filtered_data)
            self._db_discussion.messages.append(new_msg_orm)
            if new_msg_orm not in self._session:
                self._session.add(new_msg_orm)
        else:
            new_msg_orm = SimpleNamespace(**message_data)
            self._db_discussion.messages.append(new_msg_orm)
        wrapped_msg = LollmsMessage(self, new_msg_orm)
        images_list = kwargs.get('images', [])
        if images_list and kwargs.get('sender_type') == 'user':
            meta = wrapped_msg.metadata or {}
            if not meta.get('image_groups'):
                groups = [{"id": str(uuid.uuid4()), "type": "upload", "indices": [i],
                           "created_at": datetime.utcnow().isoformat(),
                           "main_image_index": i, "is_active": True}
                          for i in range(len(images_list))]
                meta["image_groups"] = groups
                wrapped_msg.metadata = meta
                wrapped_msg._sync_active_images_flags()
        self.active_branch_id = msg_id
        self._message_index[msg_id] = new_msg_orm
        self.touch()
        return wrapped_msg

    def get_branch(self, leaf_id):
        if not leaf_id:
            return []
        branch_orms = []
        current_id = leaf_id
        while current_id and current_id in self._message_index:
            msg_orm = self._message_index[current_id]
            branch_orms.append(msg_orm)
            current_id = msg_orm.parent_id
        return [LollmsMessage(self, orm) for orm in reversed(branch_orms)]

    def get_message(self, message_id):
        db_message = self._message_index.get(message_id)
        if db_message:
            return LollmsMessage(self, db_message)
        return None

    def get_all_messages_flat(self):
        self._rebuild_message_index()
        return [LollmsMessage(self, msg_obj) for msg_obj in self._message_index.values()]

    def remove_message(self, message_id: str) -> bool:
        """
        Removes a single message from the discussion and index.
        Note: If this message had children, they will become orphans.
        Returns True if successful.
        """
        if self._message_index is None:
            self._rebuild_message_index()

        if message_id not in self._message_index:
            return False

        # Remove from local index
        del self._message_index[message_id]

        if self._is_db_backed:
            # Mark for deletion during next commit
            self._messages_to_delete_from_db.add(message_id)
            # Update the SQLAlchemy relationship list
            self._db_discussion.messages = [m for m in self._db_discussion.messages if m.id != message_id]
        else:
            # In-memory proxy update
            self._db_discussion.messages = [m for m in self._db_discussion.messages if getattr(m, 'id', None) != message_id]

        # Re-evaluate active branch if the tip was removed
        if self.active_branch_id == message_id:
            self._validate_and_set_active_branch()

        self.touch()
        return True

    def delete_message(self, message_id: str) -> bool:
        """Alias for remove_message."""
        return self.remove_message(message_id)

    def setMemory(self, memory: str):
        self.memory = memory

    # -------------------------------------------------------- data zone

    def get_full_data_zone(self) -> str:
        """
        Assembles all data zones + active artefacts into a single string for the system prompt.
        Order: memory → user_data → discussion_data → personality_data → scratchpad → artefacts
        """
        parts = []
        if self.memory and self.memory.strip():
            parts.append(f"-- Memory --\n{self.memory.strip()}")
        if self.user_data_zone and self.user_data_zone.strip():
            parts.append(f"-- User Data Zone --\n{self.user_data_zone.strip()}")
        if self.discussion_data_zone and self.discussion_data_zone.strip():
            parts.append(f"-- Discussion Data Zone --\n{self.discussion_data_zone.strip()}")
        if self.personality_data_zone and self.personality_data_zone.strip():
            parts.append(f"-- Personality Data Zone --\n{self.personality_data_zone.strip()}")
        
        # [NEW] Scratchpad Zone: Full length tool outputs for the LLM to analyze
        if hasattr(self, 'scratchpad') and self.scratchpad and self.scratchpad.strip():
            parts.append(f"== TOOL OUTPUT SCRATCHPAD (Full Length) ==\n{self.scratchpad.strip()}\n== END SCRATCHPAD ==")
            
        artefacts_zone = self.artefacts.build_artefacts_context_zone()
        if artefacts_zone:
            parts.append(f"-- Active Artefacts --\n{artefacts_zone}")
        return "\n\n".join(parts)

    # ------------------------------------------------ discussion-level image methods

    def add_discussion_image(self, image_b64, source="user", active=True):
        current = self.get_discussion_images()
        current.append({"data": image_b64, "source": source, "active": active,
                         "created_at": datetime.utcnow().isoformat()})
        self.images = current
        self.touch()

    def get_discussion_images(self):
        if not self.images or len(self.images) == 0 or type(self.images) is not list:
            return []
        if isinstance(self.images[0], str):
            ASCIIColors.yellow(f"Discussion {self.id}: Upgrading legacy image format.")
            upgraded = [{"data": s, "source": "user", "active": True,
                          "created_at": datetime.utcnow().isoformat()} for s in self.images]
            self.images = upgraded
            self.touch()
        return self.images

    def toggle_discussion_image_activation(self, index, active=None):
        current = self.get_discussion_images()
        if index >= len(current):
            raise IndexError("Discussion image index out of range.")
        current[index]["active"] = (not current[index].get("active", True)
                                    if active is None else bool(active))
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

    # -------------------------------------------------------- property

    @property
    def system_prompt(self):
        return self._system_prompt
