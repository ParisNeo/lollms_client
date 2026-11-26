# lollms_discussion/_mixin_core.py
# CoreMixin: __init__, class factories, __getattr__/__setattr__ proxy,
# internal DB helpers, message CRUD, data-zone assembly, and discussion-level image methods.

import os
import re
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Union, Optional

from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import PendingRollbackError
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_artefact import (
    ArtefactManager,
    ArtefactType,
    sanitize_artifact_filename,
    _is_ignored_path,
    _IGNORED_ARTEFACT_DIRS,
    _IGNORED_ARTEFACT_EXTS
)
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
            workspace_path: Optional[str|Path] = None,
            auto_create_db: bool = True,
            auto_sync_workspace: bool = True,
            _from_db_reload: bool = False,
        ):
            _init_t0 = time.perf_counter()
            ASCIIColors.info(
                f"[CoreMixin] __init__ START (reload={_from_db_reload}, "
                f"workspace_path={workspace_path!r})"
            )
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

            # Initialize turn-level action log eagerly so that _StreamState._dispatch_closed_tag
            # can always append without hitting the __getattr__ guard on private attributes.
            object.__setattr__(self, '_turn_actions_log', [])

            # Resolve clean isolated directory paths
            resolved_id = discussion_id or (db_discussion_obj.id if db_discussion_obj else "viewer_session")

            # ── RESTORE PERSISTED WORKSPACE PATH (early, when obj is supplied) ──
            if not workspace_path and db_discussion_obj is not None:
                saved_meta = getattr(db_discussion_obj, 'discussion_metadata', None) or {}
                if isinstance(saved_meta, dict) and saved_meta.get("workspace_path"):
                    workspace_path = saved_meta["workspace_path"]
                    ASCIIColors.info(f"[CoreMixin] Restored persisted workspace path: {workspace_path}")

            # ── NATIVE AUTO-INGESTION & CLEAN ISOLATION DIRECTORIES ──
            # When the caller supplies an explicit workspace_path, it IS the
            # workspace data root — we never inject a "workspace_data" subfolder
            # inside a user-provided directory. Only the auto-generated default
            # layout keeps the isolated workspace_data/artefacts_metadata tree.
            custom_workspace = bool(workspace_path)
            if custom_workspace:
                parent_dir = Path(workspace_path).resolve()
                # Defensive normalization: strip a trailing "workspace_data"
                # segment so legacy persisted paths still land on a clean root.
                if parent_dir.name == "workspace_data" and parent_dir.parent.name:
                    parent_dir = parent_dir.parent
                parent_dir.mkdir(parents=True, exist_ok=True)
                ws_data_dir = parent_dir
                # Keep bookkeeping OUT of the user-owned external tree: the
                # custom workspace must remain byte-for-byte the caller's own.
                # A sibling `<parent>_artefacts` folder holds the metadata so we
                # never create `.artefacts_metadata/` inside a repo the user
                # owns (which would pollute their Git status / project tree).
                metadata_dir = parent_dir.parent / f"{parent_dir.name}_artefacts"
                metadata_dir.mkdir(parents=True, exist_ok=True)
            else:
                parent_dir = Path("./data_workspace") / "discussions" / resolved_id
                parent_dir.mkdir(parents=True, exist_ok=True)

                ws_data_dir = parent_dir / "workspace_data"
                ws_data_dir.mkdir(parents=True, exist_ok=True)

                metadata_dir = parent_dir / "artefacts_metadata"
                metadata_dir.mkdir(parents=True, exist_ok=True)

            object.__setattr__(self, '_workspace_path_val', str(parent_dir.resolve()))
            object.__setattr__(self, 'workspace_data_path', str(ws_data_dir.resolve()))
            object.__setattr__(self, 'artefacts_metadata_path', str(metadata_dir.resolve()))

            # ── AGENTIC MODE: CUSTOM WORKSPACE ⇒ NO .versions, NO DEEP SCAN ──
            # A user-supplied workspace may contain hundreds of thousands of files.
            # Enumerating it at session creation (via ArtefactManager._sync_index_with_disk
            # and sync_workspace_to_artefacts) would hang the UI for minutes. In agentic
            # mode we disable artefact versioning (no .versions/ snapshots, no .lam twins)
            # and skip the initial deep scan entirely. Only files the LLM explicitly
            # writes via <artifact> (or unlocks) are tracked. The host app or external
            # Git owns versioning and disk reconciliation for a custom workspace.
            object.__setattr__(self, 'disable_artefact_versioning', custom_workspace)
            if custom_workspace:
                ASCIIColors.info(
                    f"[CoreMixin] Agentic workspace mode enabled for '{parent_dir}' "
                    f"(versioning off, deep scan skipped)."
                )

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
                        proxy_data = {'id': resolved_id}
                        self._db_discussion = db_manager.DiscussionModel(**proxy_data)
                        self._session.add(self._db_discussion)
                        self._session.commit()
                        self._session.refresh(self._db_discussion)
            else:
                self._create_in_memory_proxy(id=resolved_id)
                if auto_create_db:
                    from ._db import LollmsDataManager
                    db_file_path = parent_dir / "discussion.db"
                    auto_db_manager = LollmsDataManager(f"sqlite:///{db_file_path}")
                    object.__setattr__(self, 'db_manager', auto_db_manager)
                    object.__setattr__(self, '_is_db_backed', True)
                    self._session = auto_db_manager.get_session()
                    try:
                        self._db_discussion = self._session.query(
                            auto_db_manager.DiscussionModel).filter_by(id=resolved_id).one()
                    except NoResultFound:
                        proxy_data = {'id': resolved_id}
                        self._db_discussion = auto_db_manager.DiscussionModel(**proxy_data)
                        self._session.add(self._db_discussion)
                        self._session.commit()
                        self._session.refresh(self._db_discussion)
                else:
                    object.__setattr__(self, '_is_db_backed', False)

            # ── LATE WORKSPACE-RESTORE PASS ──
            # When the discussion is loaded by `discussion_id` (not by ORM object),
            # `db_discussion_obj` was None at the early restore point above. Now that
            # `_db_discussion` is resolved, re-read its metadata and, if a persisted
            # `workspace_path` exists, adopt it — repointing the session into agentic
            # mode (versioning off, no deep scan).
            if not custom_workspace:
                _late_meta = getattr(self._db_discussion, 'discussion_metadata', {}) or {}
                _late_ws = _late_meta.get("workspace_path") if isinstance(_late_meta, dict) else None
                if _late_ws:
                    _late_dir = Path(_late_ws).resolve()
                    if _late_dir.name == "workspace_data" and _late_dir.parent.name:
                        _late_dir = _late_dir.parent
                    _late_dir.mkdir(parents=True, exist_ok=True)
                    object.__setattr__(self, '_workspace_path_val', str(_late_dir))
                    object.__setattr__(self, 'workspace_data_path', str(_late_dir))
                    _late_meta_dir = _late_dir.parent / f"{_late_dir.name}_artefacts"
                    _late_meta_dir.mkdir(parents=True, exist_ok=True)
                    object.__setattr__(self, 'artefacts_metadata_path', str(_late_meta_dir))
                    object.__setattr__(self, 'disable_artefact_versioning', True)
                    custom_workspace = True
                    ASCIIColors.info(
                        f"[CoreMixin] Restored persisted workspace path from DB metadata: '{_late_dir}' "
                        f"(agentic mode re-enabled, deep scan skipped)."
                    )

            object.__setattr__(self, '_system_prompt',
                               getattr(self._db_discussion, 'system_prompt', None))

            metadata = getattr(self._db_discussion, 'discussion_metadata', {}) or {}
            images_data = metadata.get("discussion_images", [])
            object.__setattr__(self, 'images', images_data)

            # ── PERSIST THE CUSTOM WORKSPACE PATH FOR RELOAD ──
            # Persisting the path in discussion_metadata is what lets agentic mode
            # survive a session reload: on next construction, the path is restored
            # above and custom_workspace flips back to True.
            #
            # CRITICAL: the creation-time auto-sync below is skipped when
            # `_from_db_reload` is True, and even on a genuine create the sync
            # path can be bounded/short-circuited. So we explicitly commit the
            # metadata write here: otherwise the pending transaction is discarded
            # when `create_new`'s fresh session is closed, and the reload path
            # never sees `workspace_path`.
            if custom_workspace:
                try:
                    existing_meta = dict(getattr(self._db_discussion, 'discussion_metadata', {}) or {})
                    if existing_meta.get("workspace_path") != self.workspace_data_path:
                        existing_meta["workspace_path"] = self.workspace_data_path
                        setattr(self._db_discussion, 'discussion_metadata', existing_meta)
                        # SQLAlchemy's JSON column uses dict identity for change
                        # tracking; a plain setattr may or may not be detected.
                        # Force-flag `flag_modified` so the UPDATE always fires.
                        try:
                            from sqlalchemy.orm.attributes import flag_modified
                            flag_modified(self._db_discussion, 'discussion_metadata')
                        except Exception:
                            pass
                        # Commit immediately so the persisted path survives the
                        # session lifecycle of `create_new`/`get_discussion`.
                        if self._is_db_backed and self._session is not None:
                            try:
                                self._session.commit()
                            except Exception as commit_ex:
                                ASCIIColors.warning(
                                    f"[CoreMixin] Failed to commit workspace_path metadata: {commit_ex}"
                                )
                except Exception as persist_ex:
                    ASCIIColors.warning(f"[CoreMixin] Failed to persist workspace_path in metadata: {persist_ex}")

            object.__setattr__(self, 'artefacts', ArtefactManager(self, skip_sync_on_load=_from_db_reload))

            # Creation-time workspace ingestion. This is required in BOTH modes:
            #  - Managed mode (default layout): a full scan is cheap and desired.
            #  - Agentic mode (custom workspace): the user still expects their
            #    existing files to be registered as artefacts immediately. The
            #    old `not self.disable_artefact_versioning` guard silently
            #    disabled this in agentic mode, so a freshly created discussion
            #    against a populated workspace showed ZERO files until the host
            #    manually called sync_workspace_to_artefacts().
            #
            # sync_workspace_to_artefacts() is safe here in agentic mode:
            #   • it is bounded (agentic cap = 2000 files, no purge),
            #   • it never writes .versions/ or .lam twins (that gate lives in
            #     ArtefactManager and is independent of enumeration),
            #   • registration uses _save_all() which does not re-enter sync.
            if auto_sync_workspace and not _from_db_reload:
                try:
                    # Construction-time ingestion must ALWAYS enumerate so that a
                    # freshly created discussion over a populated workspace shows
                    # its files immediately — in BOTH managed and agentic modes.
                    # The bare default (``None``) is heal-only in agentic mode
                    # (see sync_workspace_to_artefacts), which would leave a new
                    # agentic session with zero files until the host manually
                    # synced. Pass ``True`` here to keep eager ingestion bounded
                    # by the agentic file cap.
                    self.sync_workspace_to_artefacts(enumerate_workspace=True)
                except Exception as sync_ex:
                    ASCIIColors.warning(f"[CoreMixin] Auto-ingestion/heal of workspace files failed: {sync_ex}")

            self._init_memory(memory_manager)

            _init_elapsed = time.perf_counter() - _init_t0
            _mode = "agentic" if getattr(self, "disable_artefact_versioning", False) else "managed"
            ASCIIColors.info(
                f"[CoreMixin] __init__ END (reload={_from_db_reload}, mode={_mode}) "
                f"took {_init_elapsed:.3f}s"
            )

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
            'workspace_path': kwargs.pop('workspace_path', None),
            'auto_create_db': kwargs.pop('auto_create_db', False)
        }
        if db_manager:
           with db_manager.get_session() as session:
               valid_keys = db_manager.DiscussionModel.__table__.columns.keys()
               history_bearing_keys = {"messages", "active_branch_id", "pruning_summary", "pruning_point_id"}
               db_creation_args = {
                   k: v for k, v in kwargs.items()
                   if k in valid_keys and k not in history_bearing_keys
               }
               disc_id = db_creation_args.get('id')
               if disc_id:
                   existing = session.query(db_manager.DiscussionModel).filter_by(id=disc_id).first()
                   if existing:
                       session.expunge(existing)
                       return cls(lollmsClient=lollms_client, db_manager=db_manager,
                                  db_discussion_obj=existing, _from_db_reload=True, **init_args)

               if 'id' not in db_creation_args:
                   db_creation_args['id'] = str(uuid.uuid4())
               db_discussion_orm = db_manager.DiscussionModel(**db_creation_args)
               session.add(db_discussion_orm)
               session.commit()
               session.expunge(db_discussion_orm)
           return cls(lollmsClient=lollms_client, db_manager=db_manager,
                      db_discussion_obj=db_discussion_orm, **init_args)
        else:
            resolved_id = kwargs.get('id') or str(uuid.uuid4())
            instance = cls(lollmsClient=lollms_client, discussion_id=resolved_id, _from_db_reload=False, **init_args)
            if instance._is_db_backed:
                with instance._session as session:
                    session.add(instance._db_discussion)
                    session.commit()
                    session.refresh(instance._db_discussion)
            return instance

    # ---------------------------------------------------------------- proxying

    def get_messages(self, branch_id=None):
        leaf_id = branch_id if branch_id is not None else self.active_branch_id
        return self.get_branch(leaf_id)

    @property
    def client(self):
        return getattr(self, 'lollmsClient', None)

    @client.setter
    def client(self, value):
        object.__setattr__(self, 'lollmsClient', value)

    def __getattr__(self, name):
        # CRITICAL GUARD: Never proxy private/internal attributes to the ORM.
        # If an underscore-prefixed attribute is missing from __dict__, raise immediately.
        # This prevents infinite recursion and confusing 'object has no attribute' errors.
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name == 'client':
            return getattr(self, 'lollmsClient', None)

        if '_db_discussion' not in self.__dict__ or self.__dict__['_db_discussion'] is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name == 'metadata':
            return getattr(self._db_discussion, 'discussion_metadata', None)
        if name == 'messages':
            return [LollmsMessage(self, msg) for msg in self._db_discussion.messages]
        return getattr(self._db_discussion, name)

    @property
    def workspace_path(self) -> Optional[str]:
        return getattr(self, '_workspace_path_val', None)

    @workspace_path.setter
    def workspace_path(self, value: Optional[Union[str, Path]]):
        if value is None:
            object.__setattr__(self, '_workspace_path_val', None)
            object.__setattr__(self, 'workspace_data_path', None)
            object.__setattr__(self, 'artefacts_metadata_path', None)
            return
        self.set_workspace_path(value)

    def set_workspace_path(self, new_workspace_path: Union[str, Path]) -> Dict[str, int]:
        """
        Manually points the discussion into a new workspace directory at runtime and refreshes the full context.

        1. Re-resolves and establishes paths (workspace_path, workspace_data_path, artefacts_metadata_path).
        2. Re-anchors ArtefactManager to the new workspace root.
        3. Clears the cached token breakdown (_token_cache) so context status reflects the new files.
        4. Performs a full bidirectional synchronization with disk as the single source of truth.
        5. Touches and commits the discussion state.

        Returns:
            Dict[str, int]: A report of the synchronization operations.
        """
        # A user-supplied workspace path is used verbatim as the workspace data
        # root. We do NOT create a nested "workspace_data" folder inside it.
        # Defensive normalization: strip a trailing "workspace_data" segment so
        # legacy callers passing "<dir>/workspace_data" still land on a clean root.
        parent_dir = Path(new_workspace_path).resolve()
        if parent_dir.name == "workspace_data" and parent_dir.parent.name:
            parent_dir = parent_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        ws_data_dir = parent_dir
        # Sibling bookkeeping dir (never inside the user's external tree), so
        # repointing a discussion at a user workspace leaves that workspace
        # pristine (no `.artefacts_metadata/` polluting their project).
        metadata_dir = parent_dir.parent / f"{parent_dir.name}_artefacts"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, '_workspace_path_val', str(parent_dir))
        object.__setattr__(self, 'workspace_data_path', str(ws_data_dir))
        object.__setattr__(self, 'artefacts_metadata_path', str(metadata_dir))

        # ── AGENTIC MODE: CUSTOM WORKSPACE ⇒ NO .versions, NO DEEP SCAN ──
        # Repointing into a user-supplied workspace switches the session to agentic
        # mode: versioning is disabled (no .versions/ snapshots, no .lam twins) and
        # the initial deep scan is skipped, keeping UI responsive on huge trees.
        object.__setattr__(self, 'disable_artefact_versioning', True)

        # Re-anchor ArtefactManager (its __init__ deep scan is gated by
        # _is_versioning_enabled(), so it short-circuits in agentic mode).
        object.__setattr__(self, 'artefacts', ArtefactManager(self))
        # The workspace root just moved — drop the cached path so subsequent
        # `_get_workspace_root()` / `_resolve_confined_path()` calls resolve
        # against the new location instead of a stale one.
        try:
            self.artefacts._invalidate_workspace_cache()
        except Exception:
            pass

        # Persist the new workspace path so agentic mode survives reload, and
        # invalidate cached context telemetry so the next prompt reflects reality.
        meta = dict(getattr(self, 'metadata', {}) or {})
        meta["workspace_path"] = str(ws_data_dir)
        if "_token_cache" in meta:
            meta["_token_cache"] = {}
        self.metadata = meta

        # In agentic mode sync_workspace_to_artefacts() is a no-op (flag-gated),
        # so this is cheap and safe to call for the report shape.
        sync_report = self.sync_workspace_to_artefacts()
        self.touch()
        if self._is_db_backed:
            self.commit()

        ASCIIColors.success(f"[LollmsDiscussion] Workspace repointed to: {parent_dir} (agentic mode: versioning off, no deep scan)")
        return sync_report

    def __setattr__(self, name, value):
        internal_attrs = [
            'lollmsClient', 'client', 'db_manager', 'autosave', 'max_context_size', 'scratchpad',
            'images', 'artefacts', '_workspace_path_val', 'workspace_data_path', 'artefacts_metadata_path',
            '_session', '_db_discussion', '_message_index', '_messages_to_delete_from_db',
            '_is_db_backed', '_system_prompt', 'disable_artefact_versioning',
        ]
        if name == 'workspace_path':
            self.set_workspace_path(value) if value else object.__setattr__(self, '_workspace_path_val', None)
        elif name in internal_attrs or '_db_discussion' not in self.__dict__ or self.__dict__['_db_discussion'] is None:
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
        proxy.created_at = datetime.now(timezone.utc).replace(tzinfo=None)
        proxy.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
        if '_db_discussion' not in self.__dict__ or self.__dict__['_db_discussion'] is None:
            return
        metadata = (getattr(self._db_discussion, 'discussion_metadata', {}) or {}).copy()
        if self.images or "discussion_images" in metadata:
            metadata["discussion_images"] = self.images

        # ── PERSIST WORKSPACE PATH ──
        # Automatically save the current workspace path into metadata so it can be
        # restored when the discussion is reloaded in a future session.
        current_ws_path = getattr(self, '_workspace_path_val', None)
        ws_changed = current_ws_path is not None and metadata.get("workspace_path") != current_ws_path
        if ws_changed:
            metadata["workspace_path"] = current_ws_path

        # Persist whenever any tracked field is present or changed. The previous
        # guard only assigned when the key already existed, silently dropping the
        # first workspace_path write on image-less discussions.
        if self.images or "discussion_images" in metadata or "workspace_path" in metadata:
            setattr(self._db_discussion, 'discussion_metadata', metadata)
        if self._is_db_backed:
            from sqlalchemy.orm.attributes import flag_modified
            try:
                flag_modified(self._db_discussion, 'discussion_metadata')
            except Exception:
                pass
        try:
            _dobj = getattr(self, "_db_discussion", None)
            if _dobj is not None and getattr(_dobj, "created_at", None) is None:
                setattr(_dobj, "created_at", datetime.now(timezone.utc).replace(tzinfo=None))
        except Exception:
            pass
        setattr(self._db_discussion, 'updated_at', datetime.now(timezone.utc).replace(tzinfo=None))
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

            # ── HEAL A PENDING-ROLLBACK SESSION ──
            # A prior flush/commit failure (e.g. "database is locked") leaves
            # the Session where every subsequent operation raises
            # PendingRollbackError until rollback() is issued. Probe with a
            # flush; if it signals that state, reset the session and bail (the
            # failed transaction's pending changes were already lost, but the
            # session must be made usable again for the next turn).
            try:
                self._session.flush()
            except Exception as flush_err:
                if isinstance(flush_err, PendingRollbackError) or "rollback" in str(flush_err).lower():
                    self._reset_session_after_failure()
                    return
                # Other flush errors (e.g. a transient lock) fall through; the
                # commit retry loop below handles them.

            if self._messages_to_delete_from_db:
                for msg_id in list(self._messages_to_delete_from_db):
                    try:
                        msg_to_del = self._session.get(self.db_manager.MessageModel, msg_id)
                        if msg_to_del:
                            self._session.delete(msg_to_del)
                    except Exception as del_err:
                        ASCIIColors.warning(f"Message deletion error deferred: {del_err}")
                self._messages_to_delete_from_db.clear()

            try:
                _dobj = getattr(self, "_db_discussion", None)
                if _dobj is not None:
                    if getattr(_dobj, "created_at", None) is None:
                        _dobj.created_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    if getattr(_dobj, "updated_at", None) is None:
                        _dobj.updated_at = _dobj.created_at
            except Exception:
                pass

            # ── COMMIT WITH SQLITE "database is locked" RETRY ──
            # WAL allows concurrent readers, but a writer competing with another
            # writer (a tool subprocess, a second Session on the same file, the
            # discussion's own auto-created DB writer, ...) raises
            # OperationalError("database is locked") once busy_timeout expires.
            # This is transient: roll back, re-bind the discussion, back off and
            # retry before giving up.
            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                try:
                    self._session.commit()
                    self._rebuild_message_index()
                    return
                except Exception as e:
                    if self._is_lock_contention(e) and attempt < max_attempts:
                        try:
                            self._session.rollback()
                        except Exception:
                            pass
                        try:
                            if self._db_discussion not in self._session:
                                self._db_discussion = self._session.merge(self._db_discussion)
                        except Exception:
                            pass
                        time.sleep(0.05 * (2 ** (attempt - 1)))
                        continue
                    # Non-retryable error, or lock retries exhausted.
                    trace_exception(e)
                    self._reset_session_after_failure()
                    return

    @staticmethod
    def _is_lock_contention(exc: Exception) -> bool:
        """True when ``exc`` is a transient SQLite lock-contention error."""
        if isinstance(exc, PendingRollbackError):
            return False
        msg = str(exc).lower()
        return "database is locked" in msg or "database table is locked" in msg

    def _reset_session_after_failure(self):
        """
        Rolls back the session and re-binds the discussion ORM object so the
        Session is usable again after a failed flush/commit.

        A failed flush leaves SQLAlchemy's Session in a state where every
        subsequent operation raises PendingRollbackError. The only escape is
        ``rollback()`` followed by re-merging any detached ORM objects, then
        rebuilding the in-memory message index from a clean transaction.
        """
        try:
            self._session.rollback()
        except Exception:
            pass
        try:
            if self._db_discussion is not None and self._db_discussion not in self._session:
                self._db_discussion = self._session.merge(self._db_discussion)
        except Exception:
            pass
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
                        'created_at': datetime.now(timezone.utc).replace(tzinfo=None), **kwargs}
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
                           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
                         "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        self.images = current
        self.touch()

    def get_discussion_images(self):
        if not self.images or len(self.images) == 0 or type(self.images) is not list:
            return []
        if isinstance(self.images[0], str):
            ASCIIColors.yellow(f"Discussion {self.id}: Upgrading legacy image format.")
            upgraded = [{"data": s, "source": "user", "active": True,
                          "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")} for s in self.images]
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

    # Lightweight text extensions whose content is read directly into the DB
    # and exposed to the LLM at FULL visibility during workspace sync.
    # Anything not in this set (PDF, DOCX, images, DBs, archives, ...) is
    # registered as an as-is metadata entry at TREE_UNLOCKABLE visibility.
    _LIGHTWEIGHT_TEXT_EXTS = {
        ".txt", ".md", ".markdown", ".log", ".py", ".js", ".ts", ".tsx", ".jsx",
        ".json", ".yaml", ".yml", ".xml", ".csv", ".tsv", ".sql", ".html", ".htm",
        ".ini", ".toml", ".cfg", ".conf", ".rst", ".tex", ".sh", ".bash", ".bat",
        ".ps1", ".c", ".cpp", ".h", ".java", ".go", ".rs", ".rb", ".php", ".swift",
        ".kt", ".lua", ".css",
    }

    def _classify_artefact_type(self, file_ext: str) -> str:
        if file_ext in (".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sql", ".cir", ".net", ".op",
                        ".c", ".cpp", ".h", ".java", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".lua",
                        ".sh", ".bash", ".bat", ".ps1"):
            return "code"
        if file_ext in (".csv", ".tsv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"):
            # Data files are ordinary FILE artefacts (no dedicated DATA type).
            return "file"
        if file_ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
            return "image"
        return "document"

    def _is_lightweight_text_file(self, f_path, file_ext: str, max_size: int = 5 * 1024 * 1024) -> bool:
        """True when a workspace file is small enough text to cache in the DB.

        Called from ``sync_workspace_to_artefacts`` as
        ``self._is_lightweight_text_file(f_path, file_ext)``. The previous
        signature declared ``(self, file_ext, max_size)`` but the caller passed
        ``(f_path, file_ext)``, so ``file_ext`` was actually a ``Path`` and the
        membership test always failed — every file fell through to the binary
        branch and no text artefact was ever registered. The old body also used
        a broken ``("a" or "b" or ...)`` expression which evaluates to the first
        truthy literal only.
        """
        _TEXT_EXTS = {
            ".text", ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt",
            ".json", ".yaml", ".yml", ".xml", ".rst", ".csv", ".log",
            ".ini", ".toml", ".cfg", ".conf", ".html", ".htm", ".css",
            ".sh", ".bash", ".sql", ".tex", ".r", ".lua", ".rb", ".go",
            ".java", ".c", ".cpp", ".h", ".hpp", ".rs", ".php", ".kt",
        }
        ext = (file_ext or "").lower()
        if ext not in _TEXT_EXTS:
            return False
        try:
            return Path(f_path).stat().st_size <= max_size
        except OSError:
            return False
    def sync_workspace_to_artefacts(
        self,
        enumerate_workspace: Optional[bool] = None,
        max_files: Optional[int] = None,
        metadata_only: Optional[bool] = None,
    ) -> Dict[str, int]:
        """
        Performs filesystem-as-source-of-truth synchronization between the physical
        workspace_data directory and the LollmsDiscussion Artefact database.

        Args:
            enumerate_workspace: ``None`` (default) resolves by workspace mode:
                agentic workspaces are heal-only (no disk walk); managed
                workspaces enumerate fully. Pass ``True`` to force enumeration
                (bounded in agentic mode) or ``False`` to force heal-only.
            max_files: Hard safety ceiling on how many files may be registered in
                a single agentic enumeration pass. ``None`` resolves to
                ``2000`` for agentic workspaces and is unlimited for the managed
                default workspace. This keeps an explicit sync on a massive user
                tree bounded; files beyond the cap remain reachable via the
                workspace tools and register lazily on write/unlock.

        Disk is the single source of truth for existence and content. Visibility
        is a security gate, NOT a sync side-effect:

        1. Purge: Deletes database records for any artefacts whose physical files
           have been deleted from disk.
        2. Ingest (text files): Reads the file content into the DB so it is
           queryable and instantly unlockable, but registers it at TREE_UNLOCKABLE
           visibility. The content is NOT injected into the LLM context until the
           LLM emits <unlock_file> or the user unlocks it. This prevents mass
           context injection when a workspace contains many text files.
        3. Ingest (binary / non-text files): Registers an as-is metadata entry at
           TREE_UNLOCKABLE visibility (no content read), so the LLM knows the file
           exists and can unlock it or use a tool to inspect it.

        Security doctrine: anything not generated by the LLM (<artifact>) or
        explicitly unlocked by the user defaults to TREE_UNLOCKABLE. Only at unlock
        time is content promoted to FULL and added to the active context.

        Returns:
            Dict[str, int]: A report containing counts of synced items:
            {"new_artefacts": int, "updated_artefacts": int, "deleted_artefacts": int}
        """
        from lollms_client.lollms_artefact import ArtefactVisibility

        agentic_mode = bool(getattr(self, "disable_artefact_versioning", False))

        # ── ENUMERATION DEFAULT RESOLUTION ────────────────────────────────────
        # ``enumerate_workspace=None`` means "auto": honour the workspace mode.
        #   • Agentic (custom user tree): default to heal-only (no disk walk).
        #     A blind ``rglob`` over a potentially enormous external tree at
        #     sync time would hang the session. Callers that genuinely need a
        #     one-shot enumeration must pass ``enumerate_workspace=True``.
        #   • Managed (library default workspace): default to full enumeration,
        #     preserving the historical filesystem-as-source-of-truth behaviour.
        # An explicit ``True``/``False`` is always respected verbatim.
        if enumerate_workspace is None:
            enumerate_workspace = not agentic_mode

        # ── ENUMERATION POLICY ────────────────────────────────────────────────
        # "Disable versioning" (agentic) must NOT mean "make the workspace
        # invisible". Two distinct behaviours are decoupled here:
        #
        #   • Versioning (`.versions/` snapshots, `.lam` twins) is OFF whenever
        #     the workspace is agentic — that gate lives in ArtefactManager and
        #     is never overridden by this method.
        #   • Enumeration (registering files as artefacts so the LLM can SEE
        #     them) is still performed on an EXPLICIT sync request. The
        #     construction-time auto-sync is guarded at its call site and never
        #     reaches here in agentic mode, so huge user trees are not scanned
        #     implicitly at session creation.
        #
        # A bounded file budget prevents an explicit sync on a pathological
        # workspace from hanging the UI.
        if not enumerate_workspace:
            return {"new_artefacts": 0, "updated_artefacts": 0, "deleted_artefacts": 0}

        # ── METADATA-ONLY ENUMERATION ──
        # Reading the FULL content of every file into the DB during enumeration
        # is the dominant I/O cost on large trees (it turns a fast `stat()` walk
        # into thousands of `read_text()` calls that can take minutes on a big
        # source project). Since enumeration registers files at TREE_UNLOCKABLE
        # (content is NOT injected into the LLM context), we do not need the
        # content up front: it is read lazily at unlock time. Default to
        # metadata-only for AGENTIC workspaces (the huge external-tree case);
        # managed default workspaces keep content caching for instant unlocks.
        if metadata_only is None:
            metadata_only = agentic_mode

        if agentic_mode:
            effective_max_files = 2000 if max_files is None else max_files
        else:
            effective_max_files = max_files  # unlimited by default in managed mode

        # Defensive reset: if a prior sync aborted mid-pass, the bulk flag could
        # still be set on the manager. Clear it before doing anything else so a
        # stale flag never suppresses a future legitimate commit.
        object.__setattr__(self.artefacts, "_bulk_sync_active", False)
        object.__setattr__(self.artefacts, "_bulk_raw_cache", None)

        ws_data_path = getattr(self, "workspace_data_path", None)
        if not ws_data_path:
            return {"new_artefacts": 0, "updated_artefacts": 0, "deleted_artefacts": 0}

        workspace_dir = Path(ws_data_path)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        report = {"new_artefacts": 0, "updated_artefacts": 0, "deleted_artefacts": 0}

        # --- 1. COLLECT DISK FILES (bounded, deterministic) ---
        # We first collect the file descriptors into a sorted list, THEN register
        # them. Registering while iterating interleaves `_save_all()` / `touch()`
        # metadata commits with directory-walk state, which can drop or duplicate
        # registrations on some filesystems. A two-phase pass is deterministic
        # and guarantees every collected file is registered.
        #
        # ── PERFORMANCE: PRUNE DURING THE WALK, NOT AFTER ──
        # `Path.rglob("*")` descends into every directory FIRST and only filters
        # afterwards. A user workspace containing a `venv/` (~50k files),
        # `node_modules/` (~200k files), or a `.git/` object store makes a naive
        # walk take many seconds to minutes and performs a `stat()` on every
        # descendant just to throw it away. We instead use `os.walk` and mutate
        # `dirs[:]` in place so heavy/vendor/VCS directories are NEVER entered:
        # the walk cost is proportional to the files we actually keep.
        disk_files = []
        # Runtime database artifacts of the discussion itself must never be
        # registered as workspace files. auto_create_db writes discussion.db
        # (plus SQLite WAL sidecars -shm/-wal) at the workspace root; treating
        # them as user artefacts pollutes the artefact list and the LLM context.
        _DB_RUNTIME_NAMES = {
            "discussion.db", "discussion.db-shm", "discussion.db-wal",
            "discussion.db-journal",
        }
        # Directories never worth walking. Names are matched case-insensitively
        # against each directory component; ANY dot-prefixed directory is also
        # pruned (`.git`, `.venv`, `.tox`, `.idea`, `.vscode`, `.cache`, user
        # dot-folders, etc.) so we never descend into hidden state trees.
        _PRUNE_DIR_NAMES = {
            "venv", "env", "site-packages", "node_modules", "__pycache__",
            "build", "dist", "target", "obj", "bin", "out",
            "bower_components", "egg-info", "dist-info",
            ".venv", ".git", ".svn", ".hg", ".tox", ".nox", ".mypy_cache",
            ".pytest_cache", ".ruff_cache", ".idea", ".vscode", ".next",
            ".nuxt", ".cache", ".lollms", ".lollms_code", ".lollms_metadata",
            "htmlcov", "artefacts_metadata", ".versions", "versions",
        }

        def _prune_dir_name(name: str) -> bool:
            low = name.lower()
            if low.startswith("."):
                return True  # every dot-folder is internal state — never walk
            if low in _PRUNE_DIR_NAMES:
                return True
            return low.endswith(".egg-info") or low.endswith(".dist-info")

        for root_str, dirnames, filenames in os.walk(workspace_dir, followlinks=False):
            # In-place prune: os.walk will not descend into these at all.
            dirnames[:] = [d for d in dirnames if not _prune_dir_name(d)]
            root_path = Path(root_str)
            for filename in filenames:
                f_path = root_path / filename
                if filename in _DB_RUNTIME_NAMES:
                    continue
                try:
                    rel_parts = f_path.relative_to(workspace_dir)
                except ValueError:
                    continue
                if _is_ignored_path(rel_parts):
                    continue
                # Capture (size, mtime) once here — a later `stat()` per file in
                # the ingest loop would double the syscall cost. `None` on error
                # so the file still registers (we just can't detect "unchanged").
                try:
                    _st = f_path.stat()
                    _size, _mtime = _st.st_size, _st.st_mtime
                except OSError:
                    _size, _mtime = None, None
                disk_files.append((f_path, _size, _mtime))

        disk_files.sort(key=lambda t: str(t[0].relative_to(workspace_dir)).replace("\\", "/"))

        # ── SCAN COMPLETENESS (drives the purge decision below) ──
        # The purge may only run against a COMPLETE on-disk view. When the
        # enumeration hit `effective_max_files`, the view is partial: purging
        # from it would wrongly delete artefacts whose files simply fell
        # beyond the cap. We record completeness now, BEFORE truncation.
        scan_complete = True
        if effective_max_files is not None and len(disk_files) > effective_max_files:
            ASCIIColors.warning(
                f"[CoreMixin] sync_workspace_to_artefacts: {len(disk_files)} files found, "
                f"registering the first {effective_max_files} (agentic enumeration cap); "
                f"remaining files stay lazily tracked via workspace tools."
            )
            scan_complete = False
            disk_files = disk_files[:effective_max_files]

        # --- 1b. DISK AS SOURCE OF TRUTH: PURGE DELETED ARTEFACTS FROM DB ---
        # Runs ONLY when the scan was COMPLETE, so a bounded agentic pass never
        # purges a live artefact that merely fell beyond the cap. In managed
        # mode the scan is always complete by default. In agentic mode this is
        # the fix for the "deleted workspace file lingers as an artefact
        # forever" leak. We pass the already-collected relative paths so the
        # purge reuses this walk instead of re-scanning the tree.
        if scan_complete:
            on_disk_rel = [
                str(p[0].relative_to(workspace_dir)).replace("\\", "/")
                for p in disk_files
            ]
            purged_titles = self.artefacts._sync_index_with_disk(
                existing_paths=on_disk_rel,
                allow_purge=True,
            )
            report["deleted_artefacts"] = len(purged_titles)

        # --- 3. WORKSPACE -> DB (Upsert Active Disk Files) ---
        # PERFORMANCE: `artefacts.get()` re-materializes the WHOLE artefact list
        # (with a per-entry migration copy) on every call. Calling it 3× per file
        # is O(N²) across a sync and dominates runtime once the registry grows.
        # Take one snapshot of the raw list up front and build an O(1) lookup
        # keyed by title / physical_path / stem; refresh it after each write so
        # later files see freshly-registered entries (e.g. two files with the
        # same basename in different subfolders).
        def _safe_version(entry) -> int:
            try:
                return int(entry.get("version", 1))
            except (TypeError, ValueError):
                return 1

        raw_arts = self.artefacts._get_all_raw()
        title_index: Dict[str, Any] = {}
        for _a in raw_arts:
            for _cand in (_a.get("title"), _a.get("physical_path")):
                if not _cand:
                    continue
                _key = str(_cand).replace("\\", "/")
                _prev = title_index.get(_key)
                if _prev is None or _safe_version(_a) >= _safe_version(_prev):
                    title_index[_key] = _a

        def _lookup_existing(rel_str: str, name: str, stem: str):
            return (
                title_index.get(rel_str)
                or title_index.get(name)
                or title_index.get(Path(rel_str).name)
                or title_index.get(stem)
            )

        # ── BULK-SYNC FAST PATH: ONE COMMIT FOR THE WHOLE PASS ──
        # Each `artefacts.add()/update()` normally commits to the DB. Across a
        # large enumeration that is one DB round-trip per file — the dominant
        # cost once the directory walk is pruned. We set a transient flag that
        # makes `_save_all()` update in-memory metadata only, then commit once
        # after the loop (plus once more from the tail `self.commit()`).
        object.__setattr__(self.artefacts, "_bulk_sync_active", True)
        for f_path, file_size, file_mtime in disk_files:
            rel_parts = f_path.relative_to(workspace_dir)

            rel_str = str(rel_parts).replace("\\", "/")
            file_ext = f_path.suffix.lower()

            existing_art = _lookup_existing(rel_str, f_path.name, f_path.stem)
            if existing_art and _is_ignored_path(existing_art.get("title", "")):
                continue

            if file_size is None:
                try:
                    file_size = f_path.stat().st_size
                except OSError:
                    continue
            atype = self._classify_artefact_type(file_ext)

            art_title = f_path.name
            # Disambiguate same-basename files in different subfolders: if the
            # bare basename is already claimed by an artefact living at a
            # DIFFERENT relative path, register this file under its full rel
            # path instead. Served from the pre-built title_index — calling
            # `self.artefacts.get()` here would re-materialize the entire list
            # for every file, i.e. O(files x artefacts).
            existing_by_name = title_index.get(f_path.name)
            if existing_by_name and existing_by_name.get("physical_path") and existing_by_name.get("physical_path") != rel_str:
                art_title = rel_str

            if self._is_lightweight_text_file(f_path, file_ext):
                # ── TEXT FILE: cache content in DB unless metadata-only ──
                # Change detection: store (size, mtime) on the record and skip
                # the file entirely when BOTH still match. This makes an
                # unchanged re-sync O(stat) instead of O(read), which is the
                # difference between a fast refresh and a minute-long reload on
                # a large source tree. In metadata-only mode we never READ the
                # file — content is loaded lazily at unlock time.
                if existing_art is not None:
                    same_size = existing_art.get("file_size") == file_size
                    same_mtime = existing_art.get("file_mtime") == file_mtime
                    if same_size and same_mtime:
                        # Nothing changed on disk — leave the record untouched.
                        continue

                if metadata_only:
                    # Register/refresh the descriptor WITHOUT reading content.
                    descriptor = (
                        f"### File: `{rel_str}`\n\n"
                        f"- **Type**: {file_ext.upper().lstrip('.') if file_ext else 'Text'} (Text)\n"
                        f"- **Size**: {file_size:,} bytes\n"
                        f"- **Location**: `./{rel_str}`\n\n"
                        f"> Content is loaded on demand. Use `<unlock_file>` to load it "
                        f"into context, or a tool (read_file, Python, grep) to inspect it.\n"
                    )
                    if existing_art is None:
                        _new = self.artefacts.add(
                            title=art_title,
                            physical_path=rel_str,
                            artefact_type=atype,
                            content=descriptor,
                            logical_content=descriptor,
                            active=False,
                            visibility=ArtefactVisibility.TREE_UNLOCKABLE,
                            file_size=file_size,
                            file_mtime=file_mtime,
                            commit_message="Synced from disk (metadata only, unlockable)"
                        )
                        report["new_artefacts"] += 1
                        for _k in (art_title, rel_str):
                            if _k:
                                title_index[str(_k).replace("\\", "/")] = _new
                    else:
                        _upd = self.artefacts.update(
                            title=existing_art.get("title", art_title),
                            physical_path=rel_str,
                            new_content=descriptor,
                            logical_content=descriptor,
                            new_type=atype,
                            bump_version=False,
                            active=(existing_art.get("visibility") == ArtefactVisibility.FULL),
                            visibility=existing_art.get("visibility", ArtefactVisibility.TREE_UNLOCKABLE),
                            file_size=file_size,
                            file_mtime=file_mtime,
                            commit_message="Refreshed from disk (metadata only)"
                        )
                        report["updated_artefacts"] += 1
                        if _upd:
                            for _k in (existing_art.get("title"), rel_str):
                                if _k:
                                    title_index[str(_k).replace("\\", "/")] = _upd
                    continue

                try:
                    disk_content = f_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                if existing_art is None:
                    _new = self.artefacts.add(
                        title=art_title,
                        physical_path=rel_str,
                        artefact_type=atype,
                        content=disk_content,
                        active=False,
                        visibility=ArtefactVisibility.TREE_UNLOCKABLE,
                        file_size=file_size,
                        file_mtime=file_mtime,
                        commit_message="Synced from disk (text content cached, unlockable)"
                    )
                    report["new_artefacts"] += 1
                    for _k in (art_title, rel_str):
                        if _k:
                            title_index[str(_k).replace("\\", "/")] = _new
                else:
                    db_content = existing_art.get("content", "")
                    if db_content != disk_content:
                        current_visibility = existing_art.get("visibility", ArtefactVisibility.TREE_UNLOCKABLE)
                        _upd = self.artefacts.update(
                            title=existing_art.get("title", art_title),
                            physical_path=rel_str,
                            new_content=disk_content,
                            new_type=atype,
                            active=(current_visibility == ArtefactVisibility.FULL),
                            visibility=current_visibility,
                            file_size=file_size,
                            file_mtime=file_mtime,
                            commit_message="Updated from disk (text content modified)"
                        )
                        report["updated_artefacts"] += 1
                        if _upd:
                            for _k in (existing_art.get("title"), rel_str):
                                if _k:
                                    title_index[str(_k).replace("\\", "/")] = _upd
            else:
                # ── BINARY / NON-TEXT FILE: as-is metadata entry, no content read ──
                # Data files (.csv/.db/.xlsx/...) are ordinary FILE artefacts; the
                # physical bytes stay on disk in their native format.
                type_desc = "As-Is / Non-Text"
                content_placeholder = (
                    f"### File: `{rel_str}`\n\n"
                    f"- **Type**: {file_ext.upper().lstrip('.') if file_ext else 'Unknown'} ({type_desc})\n"
                    f"- **Size**: {file_size:,} bytes\n"
                    f"- **Location**: `./{rel_str}`\n\n"
                    f"> This file is preserved on disk in its native format. "
                    f"Use a tool (document tools, SQL query, or Python) to inspect or "
                    f"query it, or <unlock_file> it to load a portion into context.\n"
                )
                if existing_art is not None:
                    same_size = existing_art.get("file_size") == file_size
                    same_mtime = existing_art.get("file_mtime") == file_mtime
                    if same_size and same_mtime:
                        continue

                if existing_art is None:
                    _new = self.artefacts.add(
                        title=art_title,
                        physical_path=rel_str,
                        artefact_type=atype,
                        content=content_placeholder,
                        logical_content=content_placeholder,
                        active=True,
                        visibility=ArtefactVisibility.TREE_UNLOCKABLE,
                        file_size=file_size,
                        file_mtime=file_mtime,
                        commit_message="Synced from disk (as-is entry)"
                    )
                    report["new_artefacts"] += 1
                    for _k in (art_title, rel_str):
                        if _k:
                            title_index[str(_k).replace("\\", "/")] = _new
                else:
                    _upd = self.artefacts.update(
                        title=existing_art.get("title", art_title),
                        physical_path=rel_str,
                        new_content=content_placeholder,
                        logical_content=content_placeholder,
                        new_type=atype,
                        bump_version=False,
                        active=True,
                        visibility=ArtefactVisibility.TREE_UNLOCKABLE,
                        file_size=file_size,
                        file_mtime=file_mtime,
                        commit_message="Updated from disk (as-is metadata)"
                    )
                    report["updated_artefacts"] += 1
                    if _upd:
                        for _k in (existing_art.get("title"), rel_str):
                            if _k:
                                title_index[str(_k).replace("\\", "/")] = _upd

        # Exit bulk mode and persist everything registered this pass in ONE commit.
        object.__setattr__(self.artefacts, "_bulk_sync_active", False)

        if report["deleted_artefacts"] > 0 or report["new_artefacts"] > 0 or report["updated_artefacts"] > 0:
            meta = dict(getattr(self, 'metadata', {}) or {})
            if "_token_cache" in meta:
                meta["_token_cache"] = {}
                self.metadata = meta

        self.commit()
        return report
