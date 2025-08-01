import base64
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Type, Union
from ascii_colors import trace_exception
import yaml
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, JSON,
                        LargeBinary, String, Text, create_engine)
from sqlalchemy.orm import (Session, declarative_base, declared_attr,
                            relationship, sessionmaker)
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.types import TypeDecorator
from sqlalchemy import text 
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# Type hint placeholders for classes defined externally
if False:
    from lollms_client import LollmsClient
    from lollms_personality import LollmsPersonality

from lollms_client.lollms_utilities import build_image_dicts, robust_json_parser
from ascii_colors import ASCIIColors, trace_exception
from .lollms_types import MSG_TYPE

class EncryptedString(TypeDecorator):
    """A SQLAlchemy TypeDecorator for field-level database encryption.

    This class provides transparent encryption and decryption for string-based
    database columns. It derives a stable encryption key from a user-provided
    password and a fixed salt using PBKDF2HMAC, then uses Fernet for
    symmetric encryption.

    Requires the 'cryptography' library to be installed.
    """
    impl = LargeBinary
    cache_ok = True

    def __init__(self, key: str, *args, **kwargs):
        """Initializes the encryption engine.

        Args:
            key: The secret key (password) to use for encryption.
        """
        super().__init__(*args, **kwargs)
        if not ENCRYPTION_AVAILABLE:
            raise ImportError("'cryptography' is required for DB encryption.")
        
        self.salt = b'lollms-fixed-salt-for-db-encryption'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=480000,
            backend=default_backend()
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        self.fernet = Fernet(derived_key)

    def process_bind_param(self, value: Optional[str], dialect) -> Optional[bytes]:
        """Encrypts the string value before writing it to the database.

        Args:
            value: The plaintext string to encrypt.
            dialect: The database dialect in use.

        Returns:
            The encrypted value as bytes, or None if the input was None.
        """
        if value is None:
            return None
        return self.fernet.encrypt(value.encode('utf-8'))

    def process_result_value(self, value: Optional[bytes], dialect) -> Optional[str]:
        """Decrypts the byte value from the database into a string.

        Args:
            value: The encrypted bytes from the database.
            dialect: The database dialect in use.

        Returns:
            The decrypted plaintext string, a special error message if decryption
            fails, or None if the input was None.
        """
        if value is None:
            return None
        try:
            return self.fernet.decrypt(value).decode('utf-8')
        except InvalidToken:
            return "<DECRYPTION_FAILED: Invalid Key or Corrupt Data>"


def create_dynamic_models(
    discussion_mixin: Optional[Type] = None,
    message_mixin: Optional[Type] = None,
    encryption_key: Optional[str] = None
) -> tuple[Type, Type, Type]:
    """Factory to dynamically create SQLAlchemy ORM models.

    This function builds the `Discussion` and `Message` SQLAlchemy models,
    optionally including custom mixin classes for extending functionality and
    applying encryption to text fields if a key is provided.

    Args:
        discussion_mixin: An optional class to mix into the Discussion model.
        message_mixin: An optional class to mix into the Message model.
        encryption_key: An optional key to enable database field encryption.

    Returns:
        A tuple containing the declarative Base, the created Discussion model,
        and the created Message model.
    """
    Base = declarative_base()
    EncryptedText = EncryptedString(encryption_key) if encryption_key else Text

    class DiscussionBase:
        """Abstract base for the Discussion ORM model."""
        __abstract__ = True
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        system_prompt = Column(EncryptedText, nullable=True)
        user_data_zone = Column(EncryptedText, nullable=True) # Field for persistent user-specific data
        discussion_data_zone = Column(EncryptedText, nullable=True) # Field for persistent discussion-specific data
        personality_data_zone = Column(EncryptedText, nullable=True) # Field for persistent personality-specific data
        memory = Column(EncryptedText, nullable=True) # New field for long-term memory across discussions
        
        participants = Column(JSON, nullable=True, default=dict)
        active_branch_id = Column(String, nullable=True)
        discussion_metadata = Column(JSON, nullable=True, default=dict)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

        # Fields for non-destructive context pruning
        pruning_summary = Column(EncryptedText, nullable=True)
        pruning_point_id = Column(String, nullable=True)

        @declared_attr
        def messages(cls):
            return relationship("Message", back_populates="discussion", cascade="all, delete-orphan", lazy="joined")

    class MessageBase:
        """Abstract base for the Message ORM model."""
        __abstract__ = True
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        discussion_id = Column(String, ForeignKey('discussions.id'), nullable=False, index=True)
        parent_id = Column(String, ForeignKey('messages.id'), nullable=True, index=True)
        sender = Column(String, nullable=False)
        sender_type = Column(String, nullable=False)

        raw_content = Column(EncryptedText, nullable=True)
        thoughts = Column(EncryptedText, nullable=True)
        content = Column(EncryptedText, nullable=False)
        scratchpad = Column(EncryptedText, nullable=True)

        tokens = Column(Integer, nullable=True)
        binding_name = Column(String, nullable=True)
        model_name = Column(String, nullable=True)
        generation_speed = Column(Float, nullable=True)

        message_metadata = Column(JSON, nullable=True, default=dict)
        images = Column(JSON, nullable=True, default=list)
        active_images = Column(JSON, nullable=True, default=list) # New: List of booleans for image activation state
        created_at = Column(DateTime, default=datetime.utcnow)

        @declared_attr
        def discussion(cls):
            return relationship("Discussion", back_populates="messages")

    discussion_bases = (discussion_mixin, DiscussionBase, Base) if discussion_mixin else (DiscussionBase, Base)
    DynamicDiscussion = type('Discussion', discussion_bases, {'__tablename__': 'discussions'})

    message_bases = (message_mixin, MessageBase, Base) if message_mixin else (MessageBase, Base)
    DynamicMessage = type('Message', message_bases, {'__tablename__': 'messages'})

    return Base, DynamicDiscussion, DynamicMessage


class LollmsDataManager:
    """Manages database connection, session, and table creation.

    This class serves as the central point of contact for all database
    operations, abstracting away the SQLAlchemy engine and session management.
    """

    def __init__(self, db_path: str, discussion_mixin: Optional[Type] = None, message_mixin: Optional[Type] = None, encryption_key: Optional[str] = None):
        """Initializes the data manager.

        Args:
            db_path: The connection string for the SQLAlchemy database
                     (e.g., 'sqlite:///mydatabase.db').
            discussion_mixin: Optional mixin class for the Discussion model.
            message_mixin: Optional mixin class for the Message model.
            encryption_key: Optional key to enable database encryption.
        """
        if not db_path:
            raise ValueError("Database path cannot be empty.")
        
        self.Base, self.DiscussionModel, self.MessageModel = create_dynamic_models(
            discussion_mixin, message_mixin, encryption_key
        )
        self.engine = create_engine(db_path)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_and_migrate_tables()

    def create_and_migrate_tables(self):
        """Creates all tables if they don't exist and performs simple schema migrations."""
        self.Base.metadata.create_all(bind=self.engine)
        try:
            with self.engine.connect() as connection:
                print("Checking for database schema upgrades...")
                
                # Discussions table migration
                cursor = connection.execute(text("PRAGMA table_info(discussions)"))
                columns = {row[1] for row in cursor.fetchall()}
                
                if 'pruning_summary' not in columns:
                    print("  -> Upgrading 'discussions' table: Adding 'pruning_summary' column.")
                    connection.execute(text("ALTER TABLE discussions ADD COLUMN pruning_summary TEXT"))
                
                if 'pruning_point_id' not in columns:
                    print("  -> Upgrading 'discussions' table: Adding 'pruning_point_id' column.")
                    connection.execute(text("ALTER TABLE discussions ADD COLUMN pruning_point_id VARCHAR"))

                if 'data_zone' in columns:
                    print("  -> Upgrading 'discussions' table: Removing 'data_zone' column.")
                    connection.execute(text("ALTER TABLE discussions DROP COLUMN data_zone"))

                if 'user_data_zone' not in columns:
                    print("  -> Upgrading 'discussions' table: Adding 'user_data_zone' column.")
                    connection.execute(text("ALTER TABLE discussions ADD COLUMN user_data_zone TEXT"))

                if 'discussion_data_zone' not in columns:
                    print("  -> Upgrading 'discussions' table: Adding 'discussion_data_zone' column.")
                    connection.execute(text("ALTER TABLE discussions ADD COLUMN discussion_data_zone TEXT"))

                if 'personality_data_zone' not in columns:
                    print("  -> Upgrading 'discussions' table: Adding 'personality_data_zone' column.")
                    connection.execute(text("ALTER TABLE discussions ADD COLUMN personality_data_zone TEXT"))

                if 'memory' not in columns:
                    print("  -> Upgrading 'discussions' table: Adding 'memory' column.")
                    connection.execute(text("ALTER TABLE discussions ADD COLUMN memory TEXT"))

                # Messages table migration
                cursor = connection.execute(text("PRAGMA table_info(messages)"))
                columns = {row[1] for row in cursor.fetchall()}

                if 'active_images' not in columns:
                    print("  -> Upgrading 'messages' table: Adding 'active_images' column.")
                    connection.execute(text("ALTER TABLE messages ADD COLUMN active_images TEXT"))

                print("Database schema is up to date.")
                connection.commit() 

        except Exception as e:
            print(f"\n--- DATABASE MIGRATION WARNING ---")
            print(f"An error occurred during database schema migration: {e}")
            print("The application might not function correctly if the schema is outdated.")
            print("If problems persist, consider backing up and deleting the database file.")
            print("---")

    def get_session(self) -> Session:
        """Returns a new SQLAlchemy session."""
        return self.SessionLocal()

    def list_discussions(self) -> List[Dict]:
        """Retrieves a list of all discussions from the database.

        Returns:
            A list of dictionaries, where each dictionary represents a discussion.
        """
        with self.get_session() as session:
            discussions = session.query(self.DiscussionModel).all()
            return [{c.name: getattr(disc, c.name) for c in disc.__table__.columns} for disc in discussions]

    def get_discussion(self, lollms_client: 'LollmsClient', discussion_id: str, **kwargs) -> Optional['LollmsDiscussion']:
        """Retrieves a single discussion by its ID and wraps it.

        Args:
            lollms_client: The LollmsClient instance for the discussion to use.
            discussion_id: The unique ID of the discussion to retrieve.
            **kwargs: Additional arguments to pass to the LollmsDiscussion constructor.

        Returns:
            An LollmsDiscussion instance if found, otherwise None.
        """
        with self.get_session() as session:
            try:
                db_disc = session.query(self.DiscussionModel).filter_by(id=discussion_id).one()
                session.expunge(db_disc)  # Detach from session before returning
                return LollmsDiscussion(lollmsClient=lollms_client, db_manager=self, db_discussion_obj=db_disc, **kwargs)
            except NoResultFound:
                return None

    def search_discussions(self, **criteria) -> List[Dict]:
        """Searches for discussions based on provided criteria.

        Args:
            **criteria: Keyword arguments where the key is a column name and
                        the value is the string to search for.

        Returns:
            A list of dictionaries representing the matching discussions.
        """
        with self.get_session() as session:
            query = session.query(self.DiscussionModel)
            for key, value in criteria.items():
                if hasattr(self.DiscussionModel, key):
                    query = query.filter(getattr(self.DiscussionModel, key).ilike(f"%{value}%"))
            discussions = query.all()
            return [{c.name: getattr(disc, c.name) for c in disc.__table__.columns} for disc in discussions]

    def delete_discussion(self, discussion_id: str):
        """Deletes a discussion and all its associated messages from the database.

        Args:
            discussion_id: The ID of the discussion to delete.
        """
        with self.get_session() as session:
            db_disc = session.query(self.DiscussionModel).filter_by(id=discussion_id).first()
            if db_disc:
                session.delete(db_disc)
                session.commit()


class LollmsMessage:
    """A lightweight proxy wrapper for a message ORM object.

    This class provides a more direct and convenient API for interacting with a
    message's data, proxying attribute access to the underlying database object.
    """

    def __init__(self, discussion: 'LollmsDiscussion', db_message: Any):
        """Initializes the message proxy.

        Args:
            discussion: The parent LollmsDiscussion instance.
            db_message: The underlying SQLAlchemy ORM message object or a SimpleNamespace.
        """
        object.__setattr__(self, '_discussion', discussion)
        object.__setattr__(self, '_db_message', db_message)

    def __getattr__(self, name: str) -> Any:
        """Proxies attribute getting to the underlying DB object."""
        if name == 'metadata':
            return getattr(self._db_message, 'message_metadata', None)
        return getattr(self._db_message, name)

    def __setattr__(self, name: str, value: Any):
        """Proxies attribute setting to the underlying DB object and marks discussion as dirty."""
        if name == 'metadata':
            setattr(self._db_message, 'message_metadata', value)
        else:
            setattr(self._db_message, name, value)
        self._discussion.touch()

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the message."""
        return f"<LollmsMessage id={self.id} sender='{self.sender}'>"

    def get_active_images(self) -> List[str]:
        """
        Returns a list of base64 strings for only the active images.
        This is backwards-compatible with messages created before this feature.
        """
        if not self.images:
            return []
        
        # Retrocompatibility: if active_images is not set, all images are active.
        if self.active_images is None or not isinstance(self.active_images, list):
            return self.images

        # Filter images based on the active_images flag list
        return [
            img for i, img in enumerate(self.images) 
            if i < len(self.active_images) and self.active_images[i]
        ]

    def toggle_image_activation(self, index: int, active: Optional[bool] = None):
        """
        Toggles or sets the activation status of an image at a given index.
        This change is committed to the database if the discussion is DB-backed.
        
        Args:
            index: The index of the image in the 'images' list.
            active: If provided, sets the status to this boolean. If None, toggles the current status.
        """
        if not self.images or index >= len(self.images):
            raise IndexError("Image index out of range.")

        # Initialize active_images if it's missing or mismatched
        if self.active_images is None or not isinstance(self.active_images, list) or len(self.active_images) != len(self.images):
            new_active_images = [True] * len(self.images)
        else:
            new_active_images = self.active_images.copy()

        if active is None:
            new_active_images[index] = not new_active_images[index]
        else:
            new_active_images[index] = bool(active) # Ensure it's a boolean
        
        self.active_images = new_active_images
        if self._discussion._is_db_backed:
            self._discussion.commit()
    
    def set_metadata_item(self, itemname:str, item_value, discussion):
        new_metadata = (self.metadata or {}).copy()
        new_metadata[itemname] = item_value
        self.metadata = new_metadata
        discussion.commit()

class LollmsDiscussion:
    """Represents and manages a single discussion.

    This class is the primary user-facing interface for interacting with a
    conversation. It can be database-backed or entirely in-memory. It handles
    message management, branching, context formatting, and automatic,
    non-destructive context pruning.
    """

    def __init__(self,
                 lollmsClient: 'LollmsClient',
                 db_manager: Optional[LollmsDataManager] = None,
                 discussion_id: Optional[str] = None,
                 db_discussion_obj: Optional[Any] = None,
                 autosave: bool = False,
                 max_context_size: Optional[int] = None):
        """Initializes a discussion instance.

        Args:
            lollmsClient: The LollmsClient instance used for generation and token counting.
            db_manager: An optional LollmsDataManager for database persistence.
            discussion_id: The ID of the discussion to load (if db_manager is provided).
            db_discussion_obj: A pre-loaded ORM object to wrap.
            autosave: If True, commits changes to the DB automatically after modifications.
            max_context_size: The maximum number of tokens to allow in the context
                              before triggering automatic pruning.
        """
        object.__setattr__(self, 'lollmsClient', lollmsClient)
        object.__setattr__(self, 'db_manager', db_manager)
        object.__setattr__(self, 'autosave', autosave)
        object.__setattr__(self, 'max_context_size', max_context_size)
        object.__setattr__(self, 'scratchpad', "")
        object.__setattr__(self, 'images', [])
        
        # Internal state
        object.__setattr__(self, '_session', None)
        object.__setattr__(self, '_db_discussion', None)
        object.__setattr__(self, '_message_index', None)
        object.__setattr__(self, '_messages_to_delete_from_db', set())
        object.__setattr__(self, '_is_db_backed', db_manager is not None)
        
        object.__setattr__(self, '_system_prompt', None)
        if self._is_db_backed:
            if not db_discussion_obj and not discussion_id:
                raise ValueError("Either discussion_id or db_discussion_obj must be provided for DB-backed discussions.")
            
            self._session = db_manager.get_session()
            if db_discussion_obj:
                self._db_discussion = self._session.merge(db_discussion_obj)
            else:
                try:
                    self._db_discussion = self._session.query(db_manager.DiscussionModel).filter_by(id=discussion_id).one()
                except NoResultFound:
                    self._session.close()
                    raise ValueError(f"No discussion found with ID: {discussion_id}")
        else:
            self._create_in_memory_proxy(id=discussion_id)
        
        self._rebuild_message_index()

    @classmethod
    def create_new(cls, lollms_client: 'LollmsClient', db_manager: Optional[LollmsDataManager] = None, **kwargs) -> 'LollmsDiscussion':
        """Creates a new discussion and persists it if a db_manager is provided.

        This is the recommended factory method for creating new discussions.

        Args:
            lollms_client: The LollmsClient instance to associate with the discussion.
            db_manager: An optional LollmsDataManager to make the discussion persistent.
            **kwargs: Attributes for the new discussion (e.g., id, title).

        Returns:
            A new LollmsDiscussion instance.
        """
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
            return cls(lollmsClient=lollms_client, db_manager=db_manager, db_discussion_obj=db_discussion_orm, **init_args)
        else:
            return cls(lollmsClient=lollms_client, discussion_id=kwargs.get('id'), **init_args)

    def get_messages(self, branch_id: Optional[str] = None) -> Optional[List[LollmsMessage]]:
        """
        Returns a list of messages forming a branch, from root to a specific leaf.

        - If no branch_id is provided, it returns the full message list of the
          currently active branch.
        - If a branch_id is provided, it returns the list of all messages from the
          root up to (and including) the message with that ID.

        Args:
            branch_id: The ID of the leaf message of the desired branch.
                       If None, the active branch's leaf is used.

        Returns:
            A list of LollmsMessage objects for the specified branch, ordered
            from root to leaf, or None if the branch_id does not exist.
        """
        # Determine which leaf message ID to use
        leaf_id = branch_id if branch_id is not None else self.active_branch_id

        # Return the full branch leading to that leaf
        # We assume self.get_branch() correctly handles non-existent IDs by returning None or an empty list.
        return self.get_branch(leaf_id)


    def __getattr__(self, name: str) -> Any:
        """Proxies attribute getting to the underlying discussion object."""
        if name == 'metadata':
            return getattr(self._db_discussion, 'discussion_metadata', None)
        if name == 'messages':
            return [LollmsMessage(self, msg) for msg in self._db_discussion.messages]
        return getattr(self._db_discussion, name)

    def __setattr__(self, name: str, value: Any):
        """Proxies attribute setting to the underlying discussion object."""
        internal_attrs = [
            'lollmsClient', 'db_manager', 'autosave', 'max_context_size', 'scratchpad', 'images',
            '_session', '_db_discussion', '_message_index', '_messages_to_delete_from_db', '_is_db_backed',
            '_system_prompt'
        ]
        if name in internal_attrs:
            object.__setattr__(self, name, value)
        else:
            if name == 'metadata':
                setattr(self._db_discussion, 'discussion_metadata', value)
            else:
                setattr(self._db_discussion, name, value)
            self.touch()
    
    def _create_in_memory_proxy(self, id: Optional[str] = None):
        """Creates a SimpleNamespace object to mimic a DB record for in-memory discussions."""
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
    
    def _rebuild_message_index(self):
        """Rebuilds the internal dictionary mapping message IDs to message objects."""
        if self._is_db_backed and self._session.is_active and self._db_discussion in self._session:
            self._session.refresh(self._db_discussion, ['messages'])
        self._message_index = {msg.id: msg for msg in self._db_discussion.messages}

    def touch(self):
        """Marks the discussion as updated and saves it if autosave is enabled."""
        setattr(self._db_discussion, 'updated_at', datetime.utcnow())
        if self._is_db_backed and self.autosave:
            self.commit()

    def commit(self):
        """Commits all pending changes to the database.

        This includes new/modified discussion attributes and any pending message deletions.
        """
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
            self._session.rollback()
            raise e

    def close(self):
        """Commits any final changes and closes the database session."""
        if self._session:
            self.commit()
            self._session.close()

    def add_message(self, **kwargs) -> LollmsMessage:
        """Adds a new message to the discussion.

        Args:
            **kwargs: Attributes for the new message (e.g., sender, content, parent_id).

        Returns:
            The newly created LollmsMessage instance.
        """
        msg_id = kwargs.get('id', str(uuid.uuid4()))
        parent_id = kwargs.get('parent_id', self.active_branch_id)
        
        # New: Automatically initialize active_images if images are provided
        if 'images' in kwargs and kwargs['images'] and 'active_images' not in kwargs:
            kwargs['active_images'] = [True] * len(kwargs['images'])

        message_data = {
            'id': msg_id,
            'parent_id': parent_id,
            'discussion_id': self.id,
            'created_at': datetime.utcnow(),
            **kwargs
        }
        
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
            
        self._message_index[msg_id] = new_msg_orm
        self.active_branch_id = msg_id
        self.touch()
        return LollmsMessage(self, new_msg_orm)
        
    def get_branch(self, leaf_id: Optional[str]) -> List[LollmsMessage]:
        """Traces a branch of the conversation from a leaf message back to the root.

        Args:
            leaf_id: The ID of the message at the end of the branch.

        Returns:
            A list of LollmsMessage objects, ordered from the root to the leaf.
        """
        if not leaf_id:
            return []
        
        branch_orms = []
        current_id = leaf_id
        while current_id and current_id in self._message_index:
            msg_orm = self._message_index[current_id]
            branch_orms.append(msg_orm)
            current_id = msg_orm.parent_id
            
        return [LollmsMessage(self, orm) for orm in reversed(branch_orms)]

    def get_full_data_zone(self):
        """Assembles all data zones into a single, formatted string for the prompt."""
        parts = []
        # If memory is not empty, add it to the list of zones.
        if self.memory and self.memory.strip():
            parts.append(f"-- Memory --\n{self.memory.strip()}")
        if self.user_data_zone and self.user_data_zone.strip():
            parts.append(f"-- User Data Zone --\n{self.user_data_zone.strip()}")
        if self.discussion_data_zone and self.discussion_data_zone.strip():
            parts.append(f"-- Discussion Data Zone --\n{self.discussion_data_zone.strip()}")
        if self.personality_data_zone and self.personality_data_zone.strip():
            parts.append(f"-- Personality Data Zone --\n{self.personality_data_zone.strip()}")
        
        # Join the zones with double newlines for clear separation in the prompt.
        return "\n\n".join(parts)
    
    
    def chat(
        self,
        user_message: str,
        personality: Optional['LollmsPersonality'] = None,
        branch_tip_id: Optional[str | None] = None,
        use_mcps: Union[None, bool, List[str]] = None,
        use_data_store: Union[None, Dict[str, Callable]] = None,
        add_user_message: bool = True,
        max_reasoning_steps: int = 20,
        images: Optional[List[str]] = None,
        debug: bool = False,
        **kwargs
    ) -> Dict[str, 'LollmsMessage']:
        """Main interaction method that can invoke the dynamic, multi-modal agent.

        This method orchestrates the entire response generation process. It can
        trigger a simple, direct chat with the language model, or it can invoke
        the powerful `generate_with_mcp_rag` agent.

        When an agentic turn is used, the agent's full reasoning process (the
        `final_scratchpad`), tool calls, and any retrieved RAG sources are
        automatically stored in the resulting AI message object for full persistence
        and auditability. It also handles clarification requests from the agent.

        Args:
            user_message: The new message from the user.
            personality: An optional LollmsPersonality to use for the response,
                         which can influence system prompts and other behaviors.
            use_mcps: Controls MCP tool usage for the agent. Can be None (disabled),
                      True (all tools), or a list of specific tool names.
            use_data_store: Controls RAG usage for the agent. A dictionary mapping
                            store names to their query callables.
            add_user_message: If True, a new user message is created from the prompt.
                              If False, it assumes regeneration on the current active
                              user message.
            max_reasoning_steps: The maximum number of reasoning cycles for the agent
                                 before it must provide a final answer.
            images: A list of base64-encoded images provided by the user, which will
                    be passed to the agent or a multi-modal LLM.
            debug: If True, prints full prompts and raw AI responses to the console.
            **kwargs: Additional keyword arguments passed to the underlying generation
                      methods, such as 'streaming_callback'.

        Returns:
            A dictionary with 'user_message' and 'ai_message' LollmsMessage objects,
            where the 'ai_message' will contain rich metadata if an agentic turn was used.
        """
        callback = kwargs.get("streaming_callback")
        # extract personality data
        if personality is not None:
            object.__setattr__(self, '_system_prompt', personality.system_prompt)

            # --- New Data Source Handling Logic ---
            if hasattr(personality, 'data_source') and personality.data_source is not None:
                if isinstance(personality.data_source, str):
                    # --- Static Data Source ---
                    if callback:
                        callback("Loading static personality data...", MSG_TYPE.MSG_TYPE_STEP, {"id": "static_data_loading"})
                    if personality.data_source:
                        self.personality_data_zone = personality.data_source.strip()

                elif callable(personality.data_source):
                    # --- Dynamic Data Source ---
                    qg_id = None
                    if callback:
                        qg_id = callback("Generating query for dynamic personality data...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": "dynamic_data_query_gen"})

                    context_for_query = self.export('markdown')
                    query_prompt = (
                        "You are an expert query generator. Based on the current conversation, formulate a concise and specific query to retrieve relevant information from a knowledge base. "
                        "The query will be used to fetch data that will help you answer the user's latest request.\n\n"
                        f"--- Conversation History ---\n{context_for_query}\n\n"
                        "--- Instructions ---\n"
                        "Generate a single query string."
                    )
                    
                    try:
                        query_json = self.lollmsClient.generate_structured_content(
                            prompt=query_prompt,
                            output_format={"query": "Your generated search query here."},
                            system_prompt="You are an AI assistant that generates search queries in JSON format.",
                            temperature=0.0
                        )

                        if not query_json or "query" not in query_json:
                            if callback:
                                callback("Failed to generate data query.", MSG_TYPE.MSG_TYPE_EXCEPTION, {"id": qg_id})
                        else:
                            generated_query = query_json["query"]
                            if callback:
                                callback(f"Generated query: '{generated_query}'", MSG_TYPE.MSG_TYPE_STEP_END, {"id": qg_id, "query": generated_query})
                            
                            dr_id = None
                            if callback:
                                dr_id = callback("Retrieving dynamic data from personality source...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": "dynamic_data_retrieval"})
                            
                            try:
                                retrieved_data = personality.data_source(generated_query)
                                if callback:
                                    callback(f"Retrieved data successfully.", MSG_TYPE.MSG_TYPE_STEP_END, {"id": dr_id, "data_snippet": retrieved_data[:200]})
                                
                                
                                if retrieved_data:
                                    self.personality_data_zone = retrieved_data.strip()

                            except Exception as e:
                                trace_exception(e)
                                if callback:
                                    callback(f"Error retrieving dynamic data: {e}", MSG_TYPE.MSG_TYPE_EXCEPTION, {"id": dr_id})
                    except Exception as e:
                        trace_exception(e)
                        if callback:
                            callback(f"An error occurred during query generation: {e}", MSG_TYPE.MSG_TYPE_EXCEPTION, {"id": qg_id})

        # Determine effective MCPs by combining personality defaults and turn-specific overrides
        effective_use_mcps = use_mcps
        if personality and hasattr(personality, 'active_mcps') and personality.active_mcps:
            if effective_use_mcps in [None, False]:
                effective_use_mcps = personality.active_mcps
            elif isinstance(effective_use_mcps, list):
                effective_use_mcps = list(set(effective_use_mcps + personality.active_mcps))
            
        if self.max_context_size is not None:
            self.summarize_and_prune(self.max_context_size)

        # Step 1: Add user message, now including any images.
        if add_user_message:
            user_msg = self.add_message(
                sender="user", 
                sender_type="user", 
                content=user_message,
                images=images,
                **kwargs
            )
        else: # Regeneration logic
            if self.active_branch_id not in self._message_index:
                raise ValueError("Regeneration failed: active branch tip not found or is invalid.")
            user_msg_orm = self._message_index[self.active_branch_id]
            if user_msg_orm.sender_type != 'user':
                raise ValueError(f"Regeneration failed: active branch tip is a '{user_msg_orm.sender_type}' message, not 'user'.")
            user_msg = LollmsMessage(self, user_msg_orm)
            images = user_msg.images

        is_agentic_turn = (effective_use_mcps is not None and effective_use_mcps) or (use_data_store is not None and use_data_store)
        
        start_time = datetime.now()
        
        agent_result = None
        final_scratchpad = None
        final_raw_response = ""
        final_content = ""

        if is_agentic_turn:
            prompt_for_agent = self.export("markdown", branch_tip_id if branch_tip_id else self.active_branch_id)
            if debug:
                ASCIIColors.cyan("\n" + "="*50 + "\n--- DEBUG: AGENTIC TURN TRIGGERED ---\n" + f"--- PROMPT FOR AGENT (from discussion history) ---\n{prompt_for_agent}\n" + "="*50 + "\n")

            agent_result = self.lollmsClient.generate_with_mcp_rag(
                prompt=prompt_for_agent,
                use_mcps=effective_use_mcps,
                use_data_store=use_data_store,
                max_reasoning_steps=max_reasoning_steps,
                images=images,
                system_prompt = self._system_prompt,
                debug=debug,
                **kwargs
            )
            final_content = agent_result.get("final_answer", "The agent did not produce a final answer.")
            final_scratchpad = agent_result.get("final_scratchpad", "")
            final_raw_response = json.dumps(agent_result, indent=2)
        else:
            if debug:
                prompt_for_chat = self.export("markdown", branch_tip_id if branch_tip_id else self.active_branch_id)
                ASCIIColors.cyan("\n" + "="*50 + f"\n--- DEBUG: SIMPLE CHAT PROMPT ---\n{prompt_for_chat}\n" + "="*50 + "\n")

            final_raw_response = self.lollmsClient.chat(self, images=images, **kwargs) or ""
            
            if debug:
                ASCIIColors.cyan("\n" + "="*50 + f"\n--- DEBUG: RAW SIMPLE CHAT RESPONSE ---\n{final_raw_response}\n" + "="*50 + "\n")

            if isinstance(final_raw_response, dict) and final_raw_response.get("status") == "error":
                raise Exception(final_raw_response.get("message", "Unknown error from lollmsClient.chat"))
            else:
                final_content = self.lollmsClient.remove_thinking_blocks(final_raw_response)
            final_scratchpad = None

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        token_count = self.lollmsClient.count_tokens(final_content)
        tok_per_sec = (token_count / duration) if duration > 0 else 0

        message_meta = {}
        if is_agentic_turn and isinstance(agent_result, dict):
            if "tool_calls" in agent_result: message_meta["tool_calls"] = agent_result["tool_calls"]
            if "sources" in agent_result: message_meta["sources"] = agent_result["sources"]
            if agent_result.get("clarification_required", False): message_meta["clarification_required"] = True

        ai_message_obj = self.add_message(
            sender=personality.name if personality else "assistant",
            sender_type="assistant",
            content=final_content,
            raw_content=final_raw_response,
            scratchpad=final_scratchpad,
            tokens=token_count,
            generation_speed=tok_per_sec,
            parent_id=user_msg.id,
            metadata=message_meta
        )
        
        if self._is_db_backed and self.autosave:
            self.commit()
            
        return {"user_message": user_msg, "ai_message": ai_message_obj}

    def regenerate_branch(self, branch_tip_id=None, **kwargs) -> Dict[str, 'LollmsMessage']:
        """Regenerates the last AI response in the active branch.

        It deletes the previous AI response and calls chat() again with the
        same user prompt.

        Args:
            **kwargs: Additional arguments for the chat() method.

        Returns:
            A dictionary with the user and the newly generated AI message.
        """
        if not branch_tip_id:
            branch_tip_id = self.active_branch_id
        if not self.active_branch_id or self.active_branch_id not in self._message_index:
            if len(self._message_index)>0:
                ASCIIColors.warning("No active message to regenerate from.\n")
                ASCIIColors.warning(f"Using last available message:{list(self._message_index.keys())[-1]}\n")
            else:
                branch_tip_id = list(self._message_index.keys())[-1]
                raise ValueError("No active message to regenerate from.")
        
        last_message_orm = self._message_index[self.active_branch_id]
        
        if last_message_orm.sender_type == 'assistant':
            parent_id = last_message_orm.parent_id
            if not parent_id:
                raise ValueError("Cannot regenerate from an assistant message with no parent.")
                
            last_message_id = last_message_orm.id
            self._db_discussion.messages.remove(last_message_orm)
            del self._message_index[last_message_id]
            if self._is_db_backed:
                self._messages_to_delete_from_db.add(last_message_id)
            
        return self.chat(user_message="", add_user_message=False, branch_tip_id=branch_tip_id, **kwargs)
    
    def delete_branch(self, message_id: str):
        """Deletes a message and its entire descendant branch.

        This method removes the specified message and any messages that have it
        as a parent or an ancestor. After deletion, the active branch is moved
        to the parent of the deleted message.

        This operation is only supported for database-backed discussions.

        Args:
            message_id: The ID of the message at the root of the branch to be deleted.

        Raises:
            NotImplementedError: If the discussion is not database-backed.
            ValueError: If the message ID is not found in the discussion.
        """
        if not self._is_db_backed:
            raise NotImplementedError("Branch deletion is only supported for database-backed discussions.")
        
        if message_id not in self._message_index:
            raise ValueError(f"Message with ID '{message_id}' not found in the discussion.")

        # --- 1. Identify all messages to delete ---
        # We start with the target message and find all of its descendants.
        messages_to_delete_ids = set()
        queue = [message_id] # A queue for breadth-first search of descendants

        while queue:
            current_id = queue.pop(0)
            if current_id in messages_to_delete_ids:
                continue # Already processed
            
            messages_to_delete_ids.add(current_id)

            # Find all direct children of the current message
            children = [msg.id for msg in self._db_discussion.messages if msg.parent_id == current_id]
            queue.extend(children)
        
        # --- 2. Get the parent of the starting message to reset the active branch ---
        original_message_orm = self._message_index[message_id]
        new_active_branch_id = original_message_orm.parent_id

        # --- 3. Perform the deletion ---
        # Remove from the ORM object's list
        self._db_discussion.messages = [
            msg for msg in self._db_discussion.messages if msg.id not in messages_to_delete_ids
        ]
        
        # Remove from the quick-access index
        for mid in messages_to_delete_ids:
            if mid in self._message_index:
                del self._message_index[mid]
        
        # Add to the set of messages to be deleted from the DB on next commit
        self._messages_to_delete_from_db.update(messages_to_delete_ids)

        # --- 4. Update the active branch ---
        # If we deleted the branch that was active, move to its parent.
        if self.active_branch_id in messages_to_delete_ids:
            self.active_branch_id = new_active_branch_id
        
        self.touch() # Mark discussion as updated and save if autosave is on

        print(f"Marked branch starting at {message_id} ({len(messages_to_delete_ids)} messages) for deletion.")
        
    def export(self, format_type: str, branch_tip_id: Optional[str] = None, max_allowed_tokens: Optional[int] = None) -> Union[List[Dict], str]:
        """Exports the discussion history into a specified format.

        This method can format the conversation for different backends like OpenAI,
        Ollama, or the native `lollms_text` format. It intelligently handles
        context limits and non-destructive pruning summaries.

        Args:
            format_type: The target format. Can be "lollms_text", "openai_chat",
                         "ollama_chat", or "markdown".
            branch_tip_id: The ID of the message to use as the end of the context.
                           Defaults to the active branch ID.
            max_allowed_tokens: The maximum number of tokens the final prompt can contain.
                                This is primarily used by "lollms_text".

        Returns:
            A string for "lollms_text" or a list of dictionaries for "openai_chat"
            and "ollama_chat". For "markdown", returns a Markdown-formatted string.

        Raises:
            ValueError: If an unsupported format_type is provided.
        """
        branch_tip_id = branch_tip_id or self.active_branch_id
        if not branch_tip_id and format_type in ["lollms_text", "openai_chat", "ollama_chat", "markdown"]:
            if format_type in ["lollms_text", "markdown"]:
                return ""
            else:
                return []


        branch = self.get_branch(branch_tip_id)
        
        # Combine system prompt and data zones
        system_prompt_part = (self._system_prompt or "").strip()
        data_zone_part = self.get_full_data_zone() # This now returns a clean, multi-part block or an empty string
        full_system_prompt = ""

        # Combine them intelligently
        if system_prompt_part and data_zone_part:
            full_system_prompt = f"{system_prompt_part}\n\n{data_zone_part}"
        elif system_prompt_part:
            full_system_prompt = system_prompt_part
        else:
            full_system_prompt = data_zone_part


        participants = self.participants or {}

        def get_full_content(msg: 'LollmsMessage') -> str:
            content_to_use = msg.content
            # You can expand this logic to include thoughts, scratchpads etc. based on settings
            return content_to_use.strip()

        # --- NATIVE LOLLMS_TEXT FORMAT ---
        if format_type == "lollms_text":
            final_prompt_parts = []
            message_parts = [] # Temporary list for correctly ordered messages

            current_tokens = 0
            messages_to_render = branch

            summary_text = ""
            if self.pruning_summary and self.pruning_point_id:
                pruning_index = -1
                for i, msg in enumerate(branch):
                    if msg.id == self.pruning_point_id:
                        pruning_index = i
                        break
                if pruning_index != -1:
                    messages_to_render = branch[pruning_index:]
                    summary_text = f"!@>system:\n--- Conversation Summary ---\n{self.pruning_summary.strip()}\n"

            sys_msg_text = ""
            if full_system_prompt:
                sys_msg_text = f"!@>system:\n{full_system_prompt.strip()}\n"
                sys_tokens = self.lollmsClient.count_tokens(sys_msg_text)
                if max_allowed_tokens is None or sys_tokens <= max_allowed_tokens:
                    final_prompt_parts.append(sys_msg_text)
                    current_tokens += sys_tokens

            if summary_text:
                summary_tokens = self.lollmsClient.count_tokens(summary_text)
                if max_allowed_tokens is None or current_tokens + summary_tokens <= max_allowed_tokens:
                    final_prompt_parts.append(summary_text)
                    current_tokens += summary_tokens

            for msg in reversed(messages_to_render):
                sender_str = msg.sender.replace(':', '').replace('!@>', '')
                content = get_full_content(msg)
                
                active_images = msg.get_active_images()
                if active_images:
                    content += f"\n({len(active_images)} image(s) attached)"

                msg_text = f"!@>{sender_str}:\n{content}\n"
                msg_tokens = self.lollmsClient.count_tokens(msg_text)

                if max_allowed_tokens is not None and current_tokens + msg_tokens > max_allowed_tokens:
                    break

                message_parts.insert(0, msg_text)
                current_tokens += msg_tokens

            final_prompt_parts.extend(message_parts)
            return "".join(final_prompt_parts).strip()

        # --- OPENAI & OLLAMA CHAT FORMATS ---
        messages = []
        if full_system_prompt:
            if format_type == "markdown":
                messages.append(f"system: {full_system_prompt}")
            else:
                messages.append({"role": "system", "content": full_system_prompt})

        for msg in branch:
            if msg.sender_type == 'user':
                role = participants.get(msg.sender, "user")
            else:
                role = participants.get(msg.sender, "assistant")

            content = get_full_content(msg)
            active_images_b64 = msg.get_active_images()
            images = build_image_dicts(active_images_b64)

            if format_type == "openai_chat":
                if images:
                    content_parts = [{"type": "text", "text": content}] if content else []
                    for img in images:
                        img_data = img['data']
                        url = f"data:image/jpeg;base64,{img_data}" if img['type'] == 'base64' else img_data
                        content_parts.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
                    messages.append({"role": role, "content": content_parts})
                else:
                    messages.append({"role": role, "content": content})

            elif format_type == "ollama_chat":
                message_dict = {"role": role, "content": content}
                
                base64_images = [img['data'] for img in images if img['type'] == 'base64']
                if base64_images:
                    message_dict["images"] = base64_images
                messages.append(message_dict)

            elif format_type == "markdown":
                # Create Markdown content based on the role and content
                markdown_line = f"**{role.capitalize()}**: {content}\n"
                if images:
                    for img in images:
                        img_data = img['data']
                        url = f"![Image](data:image/jpeg;base64,{img_data})" if img['type'] == 'base64' else f"![Image]({img_data})"
                        markdown_line += f"\n{url}\n"
                messages.append(markdown_line)

            else:
                raise ValueError(f"Unsupported export format_type: {format_type}")

        return "\n".join(messages) if format_type == "markdown" else messages
    

    def summarize_and_prune(self, max_tokens: int, preserve_last_n: int = 4):
        """Non-destructively prunes the discussion by summarizing older messages.

        This method does NOT delete messages. Instead, it generates a summary of
        the older parts of the conversation and bookmarks the point from which
        the full conversation should resume. The `export()` method then uses this
        information to build a context-window-friendly prompt.

        Args:
            max_tokens: The token limit that triggers the pruning process.
            preserve_last_n: The number of recent messages to keep in full detail.
        """
        branch_tip_id = self.active_branch_id
        if not branch_tip_id:
            return

        current_formatted_text = self.export("lollms_text", branch_tip_id, 999999)
        current_tokens = self.lollmsClient.count_tokens(current_formatted_text)

        if current_tokens <= max_tokens:
            return

        branch = self.get_branch(branch_tip_id)
        if len(branch) <= preserve_last_n:
            return

        messages_to_prune = branch[:-preserve_last_n]
        pruning_point_message = branch[-preserve_last_n]

        text_to_summarize = "\n\n".join([f"{m.sender}: {m.content}" for m in messages_to_prune])
        summary_prompt = f"Concisely summarize this conversation excerpt, capturing all key facts, questions, and decisions:\n---\n{text_to_summarize}\n---\nSUMMARY:"
        
        try:
            print("\n[INFO] Context window is full. Summarizing older messages...")
            summary = self.lollmsClient.generate_text(summary_prompt, n_predict=512, temperature=0.1)
        except Exception as e:
            print(f"\n[WARNING] Pruning failed, couldn't generate summary: {e}")
            return

        current_summary = self.pruning_summary or ""
        self.pruning_summary = f"{current_summary}\n\n--- Summary of earlier conversation ---\n{summary.strip()}".strip()
        self.pruning_point_id = pruning_point_message.id
        
        self.touch()
        print(f"[INFO] Discussion auto-pruned. {len(messages_to_prune)} messages summarized. History preserved.")

    def memorize(self, branch_tip_id: Optional[str] = None):
        """
        Analyzes the current discussion, extracts key information suitable for long-term
        memory, and appends it to the discussion's 'memory' field.

        This is intended to build a persistent knowledge base about user preferences,
        facts, and context that can be useful across different future discussions.

        Args:
            branch_tip_id: The ID of the message to use as the end of the context
                           for memory extraction. Defaults to the active branch.
        """
        try:
            # 1. Get the current conversation context
            discussion_context = self.export("markdown", branch_tip_id=branch_tip_id)
            if not discussion_context.strip():
                print("[INFO] Memorize: Discussion is empty, nothing to memorize.")
                return

            # 2. Formulate the prompt for the LLM
            system_prompt = (
                "You are a Memory Extractor AI. Your task is to analyze a conversation "
                "and extract only the most critical pieces of information that would be "
                "valuable for a future, unrelated conversation with the same user. "
                "Focus on: \n"
                "- Explicit user preferences, goals, or facts about themselves.\n"
                "- Key decisions or conclusions reached.\n"
                "- Important entities, projects, or topics mentioned that are likely to recur.\n"
                "Format the output as a concise list of bullet points. Be brief and factual. "
                "Do not repeat information that is already in the User Data Zone or the Memory"
                "If no new, significant long-term information is present, output the single word: 'NOTHING'."
            )
            
            prompt = (
                "Analyze the following discussion and extract key information for long-term memory:\n\n"
                f"--- Conversation ---\n{discussion_context}\n\n"
                "--- Extracted Memory Points (as a bulleted list) ---"
            )

            # 3. Call the LLM to extract information
            print("[INFO] Memorize: Extracting key information from discussion...")
            extracted_info = self.lollmsClient.generate_text(
                prompt,
                system_prompt=system_prompt,
                n_predict=512, # A reasonable length for a summary
                temperature=0.1, # Low temperature for factual extraction
                top_k=10,
            )

            # 4. Process and append the information
            if extracted_info and "NOTHING" not in extracted_info.upper():
                new_memory_entry = extracted_info.strip()
                
                # Format with a timestamp for context
                timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                formatted_entry = f"\n\n--- Memory entry from {timestamp} ---\n{new_memory_entry}"

                current_memory = self.memory or ""
                self.memory = (current_memory + formatted_entry).strip()
                self.touch() # Mark as updated and save if autosave is on
                print(f"[INFO] Memorize: New information added to long-term memory.")
            else:
                print("[INFO] Memorize: No new significant information found to add to memory.")
                
        except Exception as e:
            trace_exception(e)
            print(f"[ERROR] Memorize: Failed to extract memory. {e}")

    def count_discussion_tokens(self, format_type: str, branch_tip_id: Optional[str] = None) -> int:
        """Counts the number of tokens in the exported discussion content.

        This method exports the discussion in the specified format and then uses
        the lollmsClient's tokenizer to count the tokens in the resulting text.

        Args:
            format_type: The target format (e.g., "lollms_text", "openai_chat").
            branch_tip_id: The ID of the message to use as the end of the context.
                           Defaults to the active branch ID.

        Returns:
            The total number of tokens.
        """
        exported_content = self.export(format_type, branch_tip_id)
        
        text_to_count = ""
        if isinstance(exported_content, str):
            text_to_count = exported_content
        elif isinstance(exported_content, list):
            # Handle list of dicts (OpenAI/Ollama format)
            full_content = []
            for message in exported_content:
                content = message.get("content")
                if isinstance(content, str):
                    full_content.append(content)
                elif isinstance(content, list): # Handle OpenAI content parts
                    for part in content:
                        if part.get("type") == "text":
                            full_content.append(part.get("text", ""))
            text_to_count = "\n".join(full_content)

        return self.lollmsClient.count_tokens(text_to_count)
    def get_context_status(self, branch_tip_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns a detailed breakdown of the context size and its components.

        This provides a comprehensive snapshot of the context usage. It accurately calculates
        the token count of the combined system context (prompt, all data zones, summary)
        and the message history, reflecting how the `lollms_text` export format works.

        Args:
            branch_tip_id: The ID of the message branch to measure. Defaults to the active branch.

        Returns:
            A dictionary with a detailed breakdown:
            {
                "max_tokens": int | None,
                "current_tokens": int,
                "zones": {
                    "system_context": {
                        "content": str,
                        "tokens": int,
                        "breakdown": {
                            "system_prompt": {"content": str, "tokens": int},
                            "memory": {"content": str, "tokens": int},
                            ...
                        }
                    },
                    "message_history": {
                        "content": str,
                        "tokens": int,
                        "message_count": int
                    }
                }
            }
            Zones and breakdown components are only included if they contain content.
        """
        result = {
            "max_tokens": self.max_context_size,
            "current_tokens": 0,
            "zones": {}
        }
        tokenizer = self.lollmsClient.count_tokens

        # --- 1. Assemble and Tokenize the Entire System Context Block ---
        system_prompt_text = (self._system_prompt or "").strip()
        data_zone_text = self.get_full_data_zone()
        pruning_summary_content = (self.pruning_summary or "").strip()

        pruning_summary_block = ""
        if pruning_summary_content and self.pruning_point_id:
            pruning_summary_block = f"--- Conversation Summary ---\n{pruning_summary_content}"

        full_system_content_parts = [
            part for part in [system_prompt_text, data_zone_text, pruning_summary_block] if part
        ]
        full_system_content = "\n\n".join(full_system_content_parts).strip()

        if full_system_content:
            system_block = f"!@>system:\n{full_system_content}\n"
            system_tokens = tokenizer(system_block)
            
            breakdown = {}
            if system_prompt_text:
                breakdown["system_prompt"] = {
                    "content": system_prompt_text,
                    "tokens": tokenizer(system_prompt_text)
                }

            memory_text = (self.memory or "").strip()
            if memory_text:
                breakdown["memory"] = {
                    "content": memory_text,
                    "tokens": tokenizer(memory_text)
                }

            user_data_text = (self.user_data_zone or "").strip()
            if user_data_text:
                breakdown["user_data_zone"] = {
                    "content": user_data_text,
                    "tokens": tokenizer(user_data_text)
                }
            
            discussion_data_text = (self.discussion_data_zone or "").strip()
            if discussion_data_text:
                breakdown["discussion_data_zone"] = {
                    "content": discussion_data_text,
                    "tokens": tokenizer(discussion_data_text)
                }

            personality_data_text = (self.personality_data_zone or "").strip()
            if personality_data_text:
                breakdown["personality_data_zone"] = {
                    "content": personality_data_text,
                    "tokens": tokenizer(personality_data_text)
                }

            if pruning_summary_content:
                breakdown["pruning_summary"] = {
                    "content": pruning_summary_content,
                    "tokens": tokenizer(pruning_summary_content)
                }

            result["zones"]["system_context"] = {
                "content": full_system_content,
                "tokens": system_tokens,
                "breakdown": breakdown
            }

        # --- 2. Assemble and Tokenize the Message History Block ---
        branch_tip_id = branch_tip_id or self.active_branch_id
        messages_text = ""
        message_count = 0
        if branch_tip_id:
            branch = self.get_branch(branch_tip_id)
            messages_to_render = branch
            
            if self.pruning_summary and self.pruning_point_id:
                pruning_index = -1
                for i, msg in enumerate(branch):
                    if msg.id == self.pruning_point_id:
                        pruning_index = i
                        break
                if pruning_index != -1:
                    messages_to_render = branch[pruning_index:]

            message_parts = []
            for msg in messages_to_render:
                sender_str = msg.sender.replace(':', '').replace('!@>', '')
                content = msg.content.strip()

                active_images = msg.get_active_images()
                if active_images:
                    content += f"\n({len(active_images)} image(s) attached)"
                msg_text = f"!@>{sender_str}:\n{content}\n"
                message_parts.append(msg_text)
            
            messages_text = "".join(message_parts)
            message_count = len(messages_to_render)

        if messages_text:
            tokens = tokenizer(messages_text)
            result["zones"]["message_history"] = {
                "content": messages_text,
                "tokens": tokens,
                "message_count": message_count
            }

        # --- 3. Finalize the Total Count ---
        result["current_tokens"] = self.count_discussion_tokens("lollms_text", branch_tip_id)

        return result
    
    def switch_to_branch(self, branch_id):
        self.active_branch_id = branch_id

    def auto_title(self):
        try:
            if self.metadata is None:
                self.metadata = {}
            discussion = self.export("markdown")[0:1000]
            system_prompt="You are a title builder out of a discussion."
            prompt = f"""Build a title for the following discussion:
{discussion}
...
"""
            title_generation_schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short, catchy title for the discussion."},
                },
                "required": ["title"],
                "description": "JSON object as title of the discussion."
            }
            infos = self.lollmsClient.generate_structured_content(prompt = prompt, system_prompt=system_prompt, schema = title_generation_schema)
            discussion_title = infos["title"]
            new_metadata = (self.metadata or {}).copy()
            new_metadata['title'] = discussion_title
            
            self.metadata = new_metadata
            self.commit()
            return discussion_title
        except Exception as ex:
            trace_exception(ex)

    def set_metadata_item(self, itemname:str, item_value):
        new_metadata = (self.metadata or {}).copy()
        new_metadata[itemname] = item_value
        self.metadata = new_metadata
        self.commit()