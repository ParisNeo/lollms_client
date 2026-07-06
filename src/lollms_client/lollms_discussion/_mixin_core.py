# lollms_discussion/_mixin_core.py
# CoreMixin: __init__, class factories, __getattr__/__setattr__ proxy,
# internal DB helpers, message CRUD, data-zone assembly, and discussion-level image methods.

import re
import uuid
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Union, Optional

from sqlalchemy.orm.exc import NoResultFound
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_artefact import ArtefactManager, ArtefactType, sanitize_artifact_filename
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
        memory_manager=None,
        internet_config: Optional[Dict[str, Any]] = None,
        workspace_path: Optional[str] = None,
    ):
        object.__setattr__(self, 'lollmsClient', lollmsClient)
        object.__setattr__(self, 'autosave', autosave)
        object.__setattr__(self, 'max_context_size', max_context_size)
        object.__setattr__(self, 'scratchpad', "")
        object.__setattr__(self, '_session', None)
        object.__setattr__(self, '_db_discussion', None)
        import threading
        object.__setattr__(self, '_db_lock', threading.Lock())
        object.__setattr__(self, '_message_index', None)
        object.__setattr__(self, '_messages_to_delete_from_db', set())
        object.__setattr__(self, 'internet_config', internet_config or {})

        # Resolve clean isolated directory paths
        resolved_id = discussion_id or (db_discussion_obj.id if db_discussion_obj else "viewer_session")

        # ── NATIVE AUTO-INGESTION & CLEAN ISOLATION DIRECTORIES ──
        # 🛡️ STRICT PATH SOVEREIGNTY PROTOCOL:
        # If workspace_path is provided by the host application, use it EXACTLY as-is.
        # The host application is sovereign over its filesystem structure.
        # Do NOT append 'discussions/' or the discussion ID if the user explicitly provided a path.
        if workspace_path:
            parent_dir = Path(workspace_path)
        else:
            # 🛡️ FAIL-SAFE FALLBACK: If no workspace_path is provided, auto-generate
            # a safe, isolated default directory to prevent NoneType crashes and 
            # ensure tools/viewers can always read/write artifacts.
            parent_dir = Path("./data_workspace") / "discussions" / resolved_id

        parent_dir.mkdir(parents=True, exist_ok=True)

        # 🛡️ CRITICAL FIX: PATH IDEMPOLENCE
        # Prevent path duplication (e.g., .../workspace_data/workspace_data/)
        # If the host app passes a path that ALREADY contains the subfolders, 
        # we must NOT append them again. This prevents WinError 267 during subprocess CWD resolution.
        ws_data_dir = parent_dir
        if not parent_dir.name.lower() == "workspace_data":
            ws_data_dir = parent_dir / "workspace_data"
        ws_data_dir.mkdir(parents=True, exist_ok=True)

        metadata_dir = parent_dir
        if not parent_dir.name.lower() == "artefacts_metadata":
            # Ensure we don't duplicate if parent is already named artefacts_metadata
            if ws_data_dir.parent != parent_dir or parent_dir.name.lower() != "artefacts_metadata":
                metadata_dir = parent_dir / "artefacts_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, 'workspace_path', str(parent_dir.resolve()))
        object.__setattr__(self, 'workspace_data_path', str(ws_data_dir.resolve()))
        object.__setattr__(self, 'artefacts_metadata_path', str(metadata_dir.resolve()))

        # Self-healing auto DB instantiation if db_manager is omitted
        if not db_manager:
            from ._db import LollmsDataManager
            db_file_path = parent_dir / "discussion.db"
            db_manager = LollmsDataManager(f"sqlite:///{db_file_path}")

        object.__setattr__(self, 'db_manager', db_manager)
        object.__setattr__(self, '_is_db_backed', db_manager is not None)

        if self._is_db_backed:
            if not db_discussion_obj and not resolved_id:
                raise ValueError("Either discussion_id or db_discussion_obj must be provided.")
            self._session = db_manager.get_session()
            if db_discussion_obj:
                self._db_discussion = self._session.merge(db_discussion_obj)
            else:
                try:
                    self._db_discussion = self._session.query(
                        db_manager.DiscussionModel).filter_by(id=resolved_id).one()
                except NoResultFound:
                    self._session.close()
                    raise ValueError(f"No discussion found with ID: {resolved_id}")
        else:
            self._create_in_memory_proxy(id=resolved_id)

        object.__setattr__(self, '_system_prompt',
                           getattr(self._db_discussion, 'system_prompt', None))

        metadata = getattr(self._db_discussion, 'discussion_metadata', {}) or {}
        images_data = metadata.get("discussion_images", [])
        object.__setattr__(self, 'images', images_data)

        object.__setattr__(self, 'artefacts', ArtefactManager(self))

        # ── AUTO-INGEST & SELF-HEAL WORKSPACE ──
        # Always run the bidirectional sync at startup.
        # 1. If the workspace_data directory has files, they are ingested into the DB (Disk → DB).
        # 2. If the DB has artifacts but the directory is empty/missing files, they are restored (DB → Disk).
        # The previous condition only ran sync if the directory was non-empty, which prevented
        # the heal pass from restoring orphaned DB artifacts when the physical files were missing.
        try:
            self.sync_workspace_to_artefacts()
        except Exception as sync_ex:
            ASCIIColors.warning(f"[CoreMixin] Auto-ingestion/heal of workspace files failed: {sync_ex}")

        # ── Memory system ─────────────────────────────────────────────────
        self._init_memory(memory_manager)

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
            'max_context_size': kwargs.pop('max_context_size', None),
            'internet_config': kwargs.pop('internet_config', None),
            'workspace_path': kwargs.pop('workspace_path', None)
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
        # CRITICAL GUARD: Never proxy private/internal attributes to the ORM.
        # If an underscore-prefixed attribute is missing from __dict__, raise immediately.
        # This prevents infinite recursion and confusing 'object has no attribute' errors.
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
        """
        Rebuilds the local message index from the database.
        GUARANTEE: Ensures self._message_index is always a valid dict (never None).
        """
        # Initialize as empty dict first to prevent NoneType errors
        self._message_index = {}

        if not self._is_db_backed:
            # In-memory mode: build from proxy list
            if hasattr(self._db_discussion, 'messages'):
                self._message_index = {msg.id: msg for msg in self._db_discussion.messages}
            return

        # Database mode: requires valid session
        if not (self._session and self._session.is_active):
            ASCIIColors.warning("[MessageIndex] Session is inactive. Cannot rebuild index.")
            return

        try:
            # Flush pending unsaved additions to the transaction so they survive refresh
            self._session.flush()
        except Exception:
            pass

        try:
            # Check if db_discussion is still bound to session
            if self._db_discussion not in self._session:
                ASCIIColors.warning("[MessageIndex] Discussion object detached from session. Re-merging...")
                self._db_discussion = self._session.merge(self._db_discussion)

            self._session.refresh(self._db_discussion, ['messages'])
        except Exception as e:
            # If transaction is in a pending rollback state, clear it and retry refresh
            error_str = str(e).lower()
            if "pendingrollback" in error_str or "rollback" in error_str:
                try:
                    ASCIIColors.warning("[MessageIndex] Rolling back pending transaction...")
                    self._session.rollback()
                    self._db_discussion = self._session.merge(self._db_discussion)
                    self._session.refresh(self._db_discussion, ['messages'])
                except Exception as retry_err:
                    ASCIIColors.error(f"Failed to refresh messages after rollback: {retry_err}")
            else:
                ASCIIColors.error(f"Database refresh error: {e}")

        # Safely build index from whatever messages are accessible
        try:
            msgs = getattr(self._db_discussion, 'messages', [])
            self._message_index = {msg.id: msg for msg in msgs}
        except Exception as idx_err:
            ASCIIColors.error(f"Failed to build message index from ORM objects: {idx_err}")
            self._message_index = {}  # Ensure it's never None

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
            # If the active branch has children, find the deepest leaf under it
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
        if self._is_db_backed:
            from sqlalchemy.orm.attributes import flag_modified
            try:
                flag_modified(self._db_discussion, 'discussion_metadata')
            except Exception:
                pass
        if self._is_db_backed and self.autosave:
            self.commit()

    def commit(self):
        if not self._is_db_backed or not self._session:
            return

        with self._db_lock:
            if not self._session.is_active:
                return

            # Clean up pending rollback states before attempting changes
            try:
                self._session.flush()
            except Exception as e:
                if "rollback" in str(e).lower():
                    try:
                        self._session.rollback()
                    except Exception:
                        pass

            if self._messages_to_delete_from_db:
                for msg_id in self._messages_to_delete_from_db:
                    try:
                        msg_to_del = self._session.get(self.db_manager.MessageModel, msg_id)
                        if msg_to_del:
                            self._session.delete(msg_to_del)
                    except Exception as del_err:
                        ASCIIColors.warning(f"Message deletion error deferred: {del_err}")
                self._messages_to_delete_from_db.clear()
            try:
                self._session.commit()
                self._rebuild_message_index()
            except Exception as e:
                # Trace the specific DB error before rolling back
                trace_exception(e)
                try:
                    self._session.rollback()
                except Exception:
                    pass
                # Reset indices silently so session can heal on next turn
                try:
                    self._rebuild_message_index()
                except Exception:
                    pass

    def close(self):
        if self._session:
            self.commit()
            self._session.close()

    # -------------------------------------------------------- message CRUD

    def add_message(self, **kwargs) -> LollmsMessage:
        # 🛡️ DEFENSIVE CHECK: Ensure message index is valid before proceeding
        if self._message_index is None:
            ASCIIColors.warning("[add_message] Message index is None. Rebuilding...")
            self._rebuild_message_index()

        # Fallback if rebuild failed
        if self._message_index is None:
            object.__setattr__(self, '_message_index', {})

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
            if self._session and new_msg_orm not in self._session:
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

        # Safe assignment to index
        try:
            self._message_index[msg_id] = new_msg_orm
        except Exception:
            # Rebuild index as last resort
            self._rebuild_message_index()
            if self._message_index:
                self._message_index[msg_id] = new_msg_orm

        self.touch()
        return wrapped_msg

    def get_branch(self, leaf_id):
        """
        Walks recursively from the specified leaf_id up to its absolute root ancestor (parent=None),
        returning a list of LollmsMessage objects sorted chronologically from root (oldest) to leaf (newest).
        """
        if not leaf_id:
            return []

        self._rebuild_message_index()
        current_id = leaf_id

        # Concurrency Guard: Fall back to latest committed message if the leaf_id is missing
        if current_id not in self._message_index:
            all_msgs = list(self._message_index.values())
            if all_msgs:
                all_msgs.sort(key=lambda m: m.created_at or datetime.min)
                current_id = all_msgs[-1].id
            else:
                return []

        branch_orms = []
        visited = set()

        # Traverse upwards from leaf to root
        while current_id and current_id in self._message_index:
            if current_id in visited:
                break  # Prevent infinite loop on circular parent links
            visited.add(current_id)

            msg_orm = self._message_index[current_id]
            branch_orms.append(msg_orm)
            current_id = msg_orm.parent_id

        # Reverse the list to produce a clean root-to-leaf (oldest-to-newest) chronological sequence
        return [LollmsMessage(self, orm) for orm in reversed(branch_orms)]

    def get_message(self, message_id):
        if self._message_index is None:
            self._rebuild_message_index()
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
        # Use (attr or "") to handle None datazones safely
        mem = (self.memory or "").strip()
        if mem:
            parts.append(f"-- Memory --\n{mem}")
            
        udz = (self.user_data_zone or "").strip()
        if udz:
            parts.append(f"-- User Data Zone --\n{udz}")
            
        ddz = (self.discussion_data_zone or "").strip()
        if ddz:
            parts.append(f"-- Discussion Data Zone --\n{ddz}")
            
        pdz = (self.personality_data_zone or "").strip()
        if pdz:
            parts.append(f"-- Personality Data Zone --\n{pdz}")
        
        # Scratchpad is now handled in export() to avoid mid-conversation system messages
        # that break strict chat templates (e.g., llama.cpp). See _mixin_utils.py.
            
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
    def widget_css(self) -> Optional[str]:
        meta = self.metadata or {}
        return meta.get("widget_css")

    @widget_css.setter
    def widget_css(self, value: Optional[str]):
        meta = dict(self.metadata or {})
        meta["widget_css"] = value
        self.metadata = meta
        self.touch()

    @property
    def system_prompt(self):
        return self._system_prompt

    def get_workspace_path(self) -> Optional[str]:
        """
        Returns the absolute path to the discussion's isolated workspace directory.

        This method provides backward compatibility for frontend code expecting
        a method call rather than direct attribute access.

        Returns:
            str | None: The resolved workspace path, or None if not configured.
        """
        return getattr(self, 'workspace_path', None)

    def get_workspace_data_path(self) -> Optional[str]:
        """
        Returns the absolute path to the discussion's isolated workspace_data subfolder.
        
        This is the recommended CWD (Current Working Directory) for executing 
        scripts and tools so relative paths resolve correctly.

        Returns:
            str | None: The resolved workspace_data path, or None if not configured.
        """
        return getattr(self, 'workspace_data_path', None)


    def get_workspace_path(self) -> Optional[str]:
        """
        Returns the absolute path to the discussion's isolated workspace directory.

        This method provides backward compatibility for frontend code expecting
        a method call rather than direct attribute access.

        Returns:
            str | None: The resolved workspace path, or None if not configured.
        """
        return getattr(self, 'workspace_path', None)

    def get_active_file_path(self, file_name: str, create_if_missing: bool = True) -> Optional[str]:
        """
        Safely resolves the absolute path of a file inside the discussion's workspace_data directory.
        This is the recommended way to locate a file for direct execution or viewing without relying
        on the LLM agentic loop.

        Args:
            file_name (str): The name of the file (e.g., "script.py").
            create_if_missing (bool): If True, creates the workspace directory tree if it doesn't exist.

        Returns:
            str | None: The absolute path to the file, or None if no workspace is configured.
        """
        ws_data_path = getattr(self, 'workspace_data_path', None)
        if not ws_data_path:
            return None

        file_path = Path(ws_data_path) / file_name

        if create_if_missing:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        return str(file_path.resolve())

    def sync_workspace_to_artefacts(self) -> Dict[str, int]:
        """
        Performs a comprehensive bidirectional synchronization between the physical 
        workspace_data directory and the LollmsDiscussion Artefact database.

        1. Workspace -> DB: Scans the folder. Registers new files as artefacts and 
           updates DB content if the file on disk is newer.
        2. DB -> Workspace: Ensures all active text-based artefacts in the DB exist 
           on disk, writing them out if they are missing.

        Use this after executing scripts externally via subprocess.run to guarantee
        the UI and LLM context accurately reflect all file changes.

        Returns:
            Dict[str, int]: A report containing counts of synced items:
            {"new_artefacts": int, "updated_artefacts": int, "restored_files": int}
        """
        from lollms_client.lollms_artefact import ArtefactVisibility
        import os

        ws_data_path = getattr(self, 'workspace_data_path', None)
        if not ws_data_path:
            return {"new_artefacts": 0, "updated_artefacts": 0, "restored_files": 0}

        workspace_dir = Path(ws_data_path)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        report = {"new_artefacts": 0, "updated_artefacts": 0, "restored_files": 0}

        # --- 1. WORKSPACE -> DB (Upsert) ---
        EXPLICIT_BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet", 
                                ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", 
                                ".zip", ".tar", ".gz", ".pdf", ".docx"}

        _IGNORED_SYNC_DIRS = {"__pycache__", ".venv", "venv", ".git", ".idea", ".vscode", "node_modules"}
        _IGNORED_SYNC_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".lam", ".log"}

        for f_path in workspace_dir.rglob("*"):
            if not f_path.is_file():
                continue

            # Skip files in ignored directories (e.g., ./__pycache__/file.pyc)
            if any(part in _IGNORED_SYNC_DIRS for part in f_path.parts):
                continue

            file_name = f_path.name
            file_ext = f_path.suffix.lower()

            # Skip ignored extensions and hidden files
            if file_ext in _IGNORED_SYNC_EXTS or file_name.startswith("."):
                continue

            existing_art = self.artefacts.get(file_name)
            if existing_art:
                existing_ext = Path(existing_art.get("title", "")).suffix.lower()
                if existing_ext in _IGNORED_SYNC_EXTS:
                    continue

            file_size = f_path.stat().st_size
            disk_mtime = f_path.stat().st_mtime

            # Determine type (🛑 FIX: Explicitly map binary types so they don't fall back to document/.md)
            atype = "document"
            if file_ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir", ".net", ".op"):
                atype = "code"
            elif file_ext in (".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"):
                atype = "data"
            elif file_ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
                atype = "image"

            existing_art = self.artefacts.get(file_name)
            if not existing_art:
                existing_art = self.artefacts.get(f_path.stem)

            # Check if file is binary to avoid reading it into memory as text
            is_binary = file_ext in EXPLICIT_BINARY_EXTS
            if not is_binary:
                try:
                    with open(f_path, 'rb') as f:
                        if b'\x00' in f.read(1024):
                            is_binary = True
                except Exception:
                    is_binary = True

            if is_binary:
                content_placeholder = (
                    f"### Data File: `{file_name}`\n\n"
                    f"- **Type**: {file_ext.upper()} (Binary/Structured Data)\n"
                    f"- **Size**: {file_size:,} bytes\n"
                    f"- **Location**: `./{file_name}`\n\n"
                )
                if existing_art is None:
                    self.artefacts.add(
                        title=file_name,
                        artefact_type=atype,
                        content=content_placeholder,
                        active=True,
                        visibility=ArtefactVisibility.FULL,
                        commit_message="Synced from disk (binary)"
                    )
                    report["new_artefacts"] += 1
                else:
                    # Only update if disk is newer
                    db_updated_at = existing_art.get("updated_at", "")
                    # Simple heuristic: always update placeholder if size differs to ensure metadata sync
                    if existing_art.get("content", "").find(f"Size**: {file_size:,}") == -1:
                        self.artefacts.update(
                            title=file_name,
                            new_content=content_placeholder,
                            new_type=atype,
                            active=True,
                            visibility=ArtefactVisibility.FULL,
                            commit_message="Updated from disk (binary metadata)"
                        )
                        report["updated_artefacts"] += 1
            else:
                # Text File
                try:
                    disk_content = f_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                if existing_art is None:
                    self.artefacts.add(
                        title=file_name,
                        artefact_type=atype,
                        content=disk_content,
                        active=True,
                        visibility=ArtefactVisibility.FULL,
                        commit_message="Synced from disk (text)"
                    )
                    report["new_artefacts"] += 1
                else:
                    db_content = existing_art.get("content", "")
                    if db_content != disk_content:
                        self.artefacts.update(
                            title=file_name,
                            new_content=disk_content,
                            new_type=atype,
                            active=True,
                            visibility=ArtefactVisibility.FULL,
                            commit_message="Updated from disk (text modified)"
                        )
                        report["updated_artefacts"] += 1

        # --- 2. DB -> WORKSPACE (Heal/Restore) ---
        active_arts = self.artefacts.list(active_only=True)
        for art in active_arts:
            title = art.get("title")
            atype = art.get("type", "document")
            content = art.get("content", "")

            # Skip image artifacts: we cannot reconstruct the binary file from the text placeholder.
            # For 'data' artifacts, we skip true binaries (.db, .sqlite, .xlsx) but allow text-based ones (.csv).
            if atype == "image":
                continue

            # CRITICAL: Sanitize the title before constructing the file path to prevent
            # WinError 123 when titles contain invalid filename characters (e.g., URLs like
            # "https://lollms.com/the-folding/" passed by the host app's internet import).
            safe_filename = sanitize_artifact_filename(title)
            file_path = workspace_dir / safe_filename
            file_ext = file_path.suffix.lower()

            # Determine if this is a true binary data file that we cannot reconstruct from text
            is_true_binary = atype == "data" and file_ext in (".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet")
            if is_true_binary:
                continue

            # For text-based artifacts (code, document, note, skill, and text-data like CSV), restore them to disk
            # if they are missing. This heals workspaces that lost physical files.
            if not file_path.exists() and content:
                try:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content, encoding="utf-8")
                    report["restored_files"] += 1
                except Exception as heal_ex:
                    ASCIIColors.warning(f"[Sync] Failed to restore artifact '{title}' to disk: {heal_ex}")

        self.commit()
        return report
