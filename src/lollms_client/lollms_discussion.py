#lollms_client/lollms_discussion.py
#author : ParisNeo

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
from lollms_client.lollms_types import MSG_TYPE 

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

    Requires the 'cryptography' library to be installed.

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
        memory = Column(EncryptedText, nullable=True) # Field for long-term memory, now managed with structured memories
        
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

    def discussion_exists(self, discussion_id: str) -> bool:
        """Checks if a discussion with the given ID exists in the database."""
        with self.get_session() as session:
            return session.query(self.DiscussionModel).filter_by(id=discussion_id).first() is not None

    @staticmethod
    def new_message(**kwargs) -> 'SimpleNamespace':
        """A static factory method to create a new message data object.

        This is a convenience method for building message objects to be passed
        to LollmsDiscussion.from_messages. It returns a SimpleNamespace that
        mimics the structure of an ORM message object for in-memory use.

        Args:
            **kwargs: Attributes for the new message (e.g., sender, content, sender_type).

        Returns:
            A SimpleNamespace object representing the message data.
        """
        # Set default sender based on sender_type if not provided
        if 'sender' not in kwargs:
            if kwargs.get('sender_type') == 'user':
                kwargs['sender'] = 'user'
            else:
                kwargs['sender'] = 'assistant'
        
        # Ensure default sender_type if not provided
        if 'sender_type' not in kwargs:
            if kwargs.get('sender') == 'user':
                kwargs['sender_type'] = 'user'
            else:
                kwargs['sender_type'] = 'assistant'

        # Default values for a new message
        message_data = {
            'id': str(uuid.uuid4()),
            'parent_id': None,  # Will be set by from_messages
            'discussion_id': None, # Will be set by from_messages
            'created_at': datetime.utcnow(),
            'raw_content': kwargs.get('content'),
            'thoughts': None,
            'scratchpad': None,
            'tokens': None,
            'binding_name': None,
            'model_name': None,
            'generation_speed': None,
            'message_metadata': {},
            'images': [],
            'active_images': [],
        }
        
        # Override defaults with user-provided kwargs
        message_data.update(kwargs)
        
        # Handle metadata alias
        if 'metadata' in message_data:
            message_data['message_metadata'] = message_data.pop('metadata')

        return SimpleNamespace(**message_data)
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
            query = query = session.query(self.DiscussionModel)
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

    def get_all_images(self) -> List[Dict[str, Union[str, bool]]]:
        """
        Returns a list of all images associated with this message, including their activation status.
        
        Returns:
            A list of dictionaries, where each dictionary represents an image.
            Example: [{"data": "base64_string", "active": True}]
        """
        if not self.images:
            return []
        
        # Retrocompatibility: if active_images is not set or mismatched, assume all are active.
        if self.active_images is None or not isinstance(self.active_images, list) or len(self.active_images) != len(self.images):
            active_flags = [True] * len(self.images)
        else:
            active_flags = self.active_images

        return [
            {"data": img_data, "active": active_flags[i]}
            for i, img_data in enumerate(self.images)
        ]

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
        This handles groups/packs: if the image belongs to a group, enabling it
        will disable others in that group.
        
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

        # Determine desired state
        if active is None:
            target_state = not new_active_images[index]
        else:
            target_state = bool(active)

        # Check grouping logic via metadata
        # We look for 'image_groups' or 'image_generation_groups' in metadata
        metadata = self.metadata or {}
        all_groups = (metadata.get("image_generation_groups", []) + metadata.get("image_groups", []))
        
        target_group = next((g for g in all_groups if index in g.get("indices", [])), None)
        
        if target_group:
            if target_state:
                # If turning ON, disable all others in this group
                for i in target_group["indices"]:
                    if 0 <= i < len(new_active_images):
                        new_active_images[i] = (i == index)
            else:
                # If turning OFF, just deactivate it.
                # (Assuming it's allowed to have no selection in a group)
                new_active_images[index] = False
        else:
            # Simple toggle for ungrouped images
            new_active_images[index] = target_state
        
        self.active_images = new_active_images
        if self._discussion._is_db_backed:
            self._discussion.commit()

    def add_image_pack(self, images: List[str], group_type: str = "generated", active_by_default: bool = True, title: str = None) -> None:
        """
        Adds a list of images as a new pack/group.
        
        - The new images are appended to the existing images list.
        - The first image in the pack is set to Active (if active_by_default is True), others to Inactive.
        - A new group entry is added to 'image_groups' metadata.
        
        Args:
            images: List of base64 image strings.
            group_type: Type label for the group (e.g., 'generated', 'upload').
            active_by_default: If True, the first image in the pack is activated. If False, all are inactive.
            title: Optional title for the image pack (e.g., the prompt).
        """
        if not images:
            return
            
        current_images = self.images or []
        start_index = len(current_images)
        
        # Append new images
        current_images.extend(images)
        self.images = current_images
        
        # Sync active_images list
        current_active = self.active_images or [True] * start_index
        # If active_images was shorter than images for some reason, pad it
        if len(current_active) < start_index:
             current_active.extend([True] * (start_index - len(current_active)))
             
        # Determine activation states for new images
        new_active_flags = [False] * len(images)
        if active_by_default and len(images) > 0:
            new_active_flags[0] = True
            
        current_active.extend(new_active_flags)
        self.active_images = current_active
        
        # Update Metadata with Group info
        metadata = (self.metadata or {}).copy()
        groups = metadata.get("image_groups", [])
        
        new_indices = list(range(start_index, start_index + len(images)))
        
        group_entry = {
            "id": str(uuid.uuid4()),
            "type": group_type,
            "indices": new_indices,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if title:
            group_entry["title"] = title
        
        groups.append(group_entry)
        
        metadata["image_groups"] = groups
        self.metadata = metadata
        
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
            lollmsClient: The LollmsClient instance for generation and token counting.
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
        
        # Internal state
        object.__setattr__(self, '_session', None)
        object.__setattr__(self, '_db_discussion', None)
        object.__setattr__(self, '_message_index', None)
        object.__setattr__(self, '_messages_to_delete_from_db', set())
        object.__setattr__(self, '_is_db_backed', db_manager is not None)
        
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
        
        object.__setattr__(self, '_system_prompt', getattr(self._db_discussion, 'system_prompt', None))

        # --- REVISED IMAGE HANDLING ---
        # Load raw image data from metadata. This might be in the old (list of strings)
        # or new (list of dicts) format. The get_discussion_images() method will
        # handle the migration and act as the single source of truth.
        metadata = getattr(self._db_discussion, 'discussion_metadata', {}) or {}
        images_data = metadata.get("discussion_images", [])
        object.__setattr__(self, 'images', images_data)
        # The separate `active_images` list is deprecated and removed to avoid inconsistency.

        self._rebuild_message_index()
        self._validate_and_set_active_branch()
        # Trigger potential migration on load to ensure data is consistent from the start.
        self.get_discussion_images()


    @classmethod
    def from_messages(
        cls,
        messages: List[Any],
        lollms_client: 'LollmsClient',
        db_manager: Optional[LollmsDataManager] = None,
        **kwargs
    ) -> 'LollmsDiscussion':
        """Creates a new discussion instance directly from a list of message objects.

        This factory is useful for creating temporary or programmatic discussions
        without manually adding each message. Messages are chained sequentially.

        Args:
            messages: A list of message-like objects (e.g., SimpleNamespace from
                      LollmsMessage.new_message).
            lollms_client: The LollmsClient instance for the discussion.
            db_manager: An optional LollmsDataManager to make the discussion persistent.
            **kwargs: Additional arguments for the new discussion (e.g., system_prompt).

        Returns:
            A new LollmsDiscussion instance populated with the provided messages.
        """
        # Create a new, empty discussion
        discussion = cls.create_new(
            lollms_client=lollms_client,
            db_manager=db_manager,
            **kwargs
        )

        last_message_id = None
        for msg_data in messages:
            # Convert the message-like object to a dict for add_message
            if isinstance(msg_data, SimpleNamespace):
                msg_kwargs = msg_data.__dict__.copy()
            elif isinstance(msg_data, dict):
                msg_kwargs = msg_data.copy()
            else:
                raise TypeError("message objects must be of type dict or SimpleNamespace")

            # Set the parent to the previous message in the list
            msg_kwargs['parent_id'] = last_message_id
            
            # Add the message and update the last_message_id for the next iteration
            new_msg = discussion.add_message(**msg_kwargs)
            last_message_id = new_msg.id
            
        # The active_branch_id is already set to the last message by add_message,
        # so no further action is needed.
        
        return discussion

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
        """
        Proxies attribute setting to the underlying discussion object, while
        keeping internal state like _system_prompt in sync.
        """
        # A list of attributes that are internal to the LollmsDiscussion wrapper
        # and should not be proxied to the underlying data object.
        internal_attrs = [
            'lollmsClient', 'db_manager', 'autosave', 'max_context_size', 'scratchpad',
            'images',
            '_session', '_db_discussion', '_message_index', '_messages_to_delete_from_db',
            '_is_db_backed', '_system_prompt'
        ]
        
        if name in internal_attrs:
            # If it's an internal attribute, set it directly on the wrapper object.
            object.__setattr__(self, name, value)
        else:
            # If we are setting 'system_prompt', we must update BOTH the internal
            # _system_prompt variable AND the underlying data object.
            if name == 'system_prompt':
                object.__setattr__(self, '_system_prompt', value)
            
            # If the attribute is 'metadata', proxy it to the correct column name.
            if name == 'metadata':
                setattr(self._db_discussion, 'discussion_metadata', value)
            else:
                # For all other attributes, proxy them directly to the underlying object.
                setattr(self._db_discussion, name, value)
            
            # Mark the discussion as dirty to trigger a save.
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
        if self._is_db_backed and self._session and self._session.is_active and self._db_discussion in self._session:
            # Ensure discussion object's messages collection is loaded/refreshed
            self._session.refresh(self._db_discussion, ['messages'])
        self._message_index = {msg.id: msg for msg in self._db_discussion.messages}

    def _find_deepest_leaf(self, start_id: Optional[str]) -> Optional[str]:
        """
        Finds the ID of the most recent leaf message in the branch starting from start_id.
        If start_id is None or not found, it finds the most recent leaf in the entire discussion.
        A leaf message is one that has no children.
        """
        if not self._message_index:
            return None

        self._rebuild_message_index() # Ensure the internal index is up-to-date

        # Build an adjacency list (children_of_parent_id -> [child1_id, child2_id])
        children_of = {msg_id: [] for msg_id in self._message_index.keys()}
        for msg_id, msg_obj in self._message_index.items():
            if msg_obj.parent_id in children_of: # Only if parent exists in current index
                children_of[msg_obj.parent_id].append(msg_id)

        # Helper to find the most recent leaf from a list of messages
        def get_most_recent_leaf_from_list(message_list: List[Any]) -> Optional[str]:
            if not message_list:
                return None
            leaves_in_list = [msg for msg in message_list if not children_of.get(msg.id)]
            if leaves_in_list:
                return max(leaves_in_list, key=lambda msg: msg.created_at).id
            return None

        if start_id and start_id in self._message_index:
            # Perform BFS to get all descendants including the start_id itself
            queue = [self._message_index[start_id]]
            visited_ids = {start_id}
            descendants_and_self = [self._message_index[start_id]]
            
            head = 0
            while head < len(queue):
                current_msg_obj = queue[head]
                head += 1
                
                for child_id in children_of.get(current_msg_obj.id, []):
                    if child_id not in visited_ids:
                        visited_ids.add(child_id)
                        child_obj = self._message_index[child_id]
                        queue.append(child_obj)
                        descendants_and_self.append(child_obj)
            
            # Now find the most recent leaf among these descendants and the start_id itself
            result_leaf_id = get_most_recent_leaf_from_list(descendants_and_self)
            if result_leaf_id:
                return result_leaf_id
            else:
                # If no actual leaves were found within the branch rooted at start_id,
                # then start_id itself is the 'leaf' of its known subgraph IF it has no children.
                if not children_of.get(start_id):
                    return start_id
                return None # The start_id is not a leaf and has no reachable leaves.

        else: # No specific starting point, find the most recent leaf in the entire discussion
            all_messages_in_discussion = list(self._message_index.values())
            return get_most_recent_leaf_from_list(all_messages_in_discussion)


    def _validate_and_set_active_branch(self):
        """
        Ensures that self.active_branch_id points to an existing message and is a leaf message.
        If it's None, points to a non-existent message, or points to a non-leaf message,
        it attempts to set it to the ID of the most recently created leaf message in the entire discussion.
        If a valid active_branch_id exists but is not a leaf, it will try to find the deepest leaf
        from that point onwards.
        This method directly updates the underlying _db_discussion object to avoid recursion.
        """
        self._rebuild_message_index() # Ensure index is fresh

        # If the discussion is empty, silently set active_branch_id to None and exit.
        if not self._message_index:
            object.__setattr__(self._db_discussion, 'active_branch_id', None)
            return

        current_active_id = self._db_discussion.active_branch_id # Access direct attribute

        # Case 1: Active branch ID is invalid or missing
        if current_active_id is None or current_active_id not in self._message_index:
            ASCIIColors.warning(f"Active branch ID '{current_active_id}' is invalid or missing for discussion {self.id}. Attempting to select a new leaf.")
            new_active_leaf_id = self._find_deepest_leaf(None) # Find most recent leaf in entire discussion
            if new_active_leaf_id:
                object.__setattr__(self._db_discussion, 'active_branch_id', new_active_leaf_id)
                ASCIIColors.success(f"New active branch ID for discussion {self.id} set to: {new_active_leaf_id} (most recent overall leaf).")
            else:
                # This else block should theoretically not be reached if _message_index is not empty,
                # as _find_deepest_leaf(None) would find a leaf. Added for robustness.
                object.__setattr__(self, '_db_discussion.active_branch_id', None) # Use setattr for direct ORM access
                ASCIIColors.yellow(f"Could not find any leaf messages in discussion {self.id}. Active branch ID remains None.")
        
        # Case 2: Active branch ID exists, but is it a leaf?
        else:
            # Determine if current_active_id is a leaf
            children_of_current_active = []
            for msg_obj in self._message_index.values():
                if msg_obj.parent_id == current_active_id:
                    children_of_current_active.append(msg_obj.id)

            if children_of_current_active: # If it has children, it's not a leaf
                ASCIIColors.warning(f"Active branch ID '{current_active_id}' is not a leaf message. Finding deepest leaf from this point.")
                new_active_leaf_id = self._find_deepest_leaf(current_active_id)
                if new_active_leaf_id and new_active_leaf_id != current_active_id:
                    object.__setattr__(self._db_discussion, 'active_branch_id', new_active_leaf_id)
                    ASCIIColors.success(f"Active branch ID for discussion {self.id} updated to: {new_active_leaf_id} (deepest leaf descendant).")
                elif new_active_leaf_id is None: # Should not happen if current_active_id exists
                    ASCIIColors.warning(f"Could not find a deeper leaf from '{current_active_id}'. Keeping current ID.")


    def touch(self):
        """Marks the discussion as updated, persists images, and saves if autosave is on."""
        # Persist in-memory discussion images to the metadata field before saving.
        # self.images is the single source of truth and is guaranteed to be in the
        # correct dictionary format by the get_discussion_images() lazy migration.
        metadata = (getattr(self._db_discussion, 'discussion_metadata', {}) or {}).copy()
        
        if self.images or "discussion_images" in metadata:
            metadata["discussion_images"] = self.images
            setattr(self._db_discussion, 'discussion_metadata', metadata)

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
            self._rebuild_message_index() # Rebuild index after commit to reflect DB state
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
        
        kwargs.setdefault('images', [])
        kwargs.setdefault('active_images', [])

        if 'sender_type' not in kwargs:
            if kwargs.get('sender') == 'user':
                kwargs['sender_type'] = 'user'
            else:
                kwargs['sender_type'] = 'assistant'

        # --- NEW PARTICIPANT LOGIC ---
        if kwargs.get('sender_type') == 'user':
            sender_name = kwargs.get('sender')
            sender_icon = kwargs.get('sender_icon')
            if sender_name:
                if self.participants is None:
                    self.participants = {}
                # Update only if not present or icon is missing
                if sender_name not in self.participants or self.participants[sender_name].get('icon') is None:
                    self.participants[sender_name] = {"icon": sender_icon}
                    self.touch()
        # --- END NEW PARTICIPANT LOGIC ---

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
            
        self.active_branch_id = msg_id # New message is always a leaf
        self._message_index[msg_id] = new_msg_orm
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
        # Use _message_index for efficient lookup of parents
        while current_id and current_id in self._message_index:
            msg_orm = self._message_index[current_id]
            branch_orms.append(msg_orm)
            current_id = msg_orm.parent_id
            
        return [LollmsMessage(self, orm) for orm in reversed(branch_orms)]

    def get_message(self, message_id: str) -> Optional['LollmsMessage']:
        """Retrieves a single message by its ID.

        Args:
            message_id: The unique ID of the message to retrieve.

        Returns:
            An LollmsMessage instance if found, otherwise None.
        """
        db_message = self._message_index.get(message_id)
        if db_message:
            return LollmsMessage(self, db_message)
        return None
    
    def get_all_messages_flat(self) -> List[LollmsMessage]:
        """
        Retrieves all messages stored for this discussion as a flat list.
        Useful for building complex UIs or doing comprehensive data analysis.

        Returns:
            A list of LollmsMessage objects, representing all messages in the discussion.
        """
        self._rebuild_message_index() # Ensure index is fresh
        return [LollmsMessage(self, msg_obj) for msg_obj in self._message_index.values()]

    def setMemory(self, memory:str):
        """sets memory content

        Args:
            memory (str): _description_
        """
        self.memory = memory

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
        remove_thinking_blocks:bool = True,
        **kwargs
    ) -> Dict[str, Any]:
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
            remove_thinking_blocks: If True, removes any thinking blocks from the final
                                   response content, cleaning it up for user display.
            branch_tip_id: If provided, this is the ID of the message to use as the
                           starting point for the branch. If None, uses the current
                           active branch tip.
            **kwargs: Additional keyword arguments passed to the underlying generation
                      methods, such as 'streaming_callback'.

        Returns:
            A dictionary with 'user_message' and 'ai_message' LollmsMessage objects,
            where the 'ai_message' will contain rich metadata if an agentic turn was used.
        """
        callback = kwargs.get("streaming_callback")
        collected_sources = []
        

        # Step 1: Add user message, now including any images.
        if add_user_message:
            user_msg = self.add_message(
                sender=kwargs.get("user_name", "user"), 
                sender_type="user", 
                content=user_message,
                images=images,
                **kwargs
            )
        else: # Regeneration logic
            # _validate_and_set_active_branch ensures active_branch_id is valid and a leaf.
            # So, if we are regenerating, active_branch_id must be valid.
            if self.active_branch_id not in self._message_index: # Redundant check, but safe
                 raise ValueError("Regeneration failed: active branch tip not found or is invalid.")
            user_msg_orm = self._message_index[self.active_branch_id]
            if user_msg_orm.sender_type != 'user':
                raise ValueError(f"Regeneration failed: active branch tip is a '{user_msg_orm.sender_type}' message, not 'user'.")
            user_msg = LollmsMessage(self, user_msg_orm)
            # FIX: Use get_active_images() to ensure we get a list of strings, not potentially objects/dicts.
            # This prevents errors if the underlying 'images' field contains new-style structured data.
            images = user_msg.get_active_images()
                    
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
                        self.personality_data_zone = personality.data_zone.strip()

                elif callable(personality.data_source):
                    # --- Dynamic Data Source ---
                    qg_id = None
                    if callback:
                        qg_id = callback("Generating query for dynamic personality data...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": "dynamic_data_query_gen"})

                    context_for_query = self.export('markdown', suppress_system_prompt=True)
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
                                    source_item = {
                                        "title": "Personality Data Source",
                                        "content": retrieved_data,
                                        "source": personality.name if hasattr(personality, 'name') else "Personality",
                                        "query": generated_query
                                    }
                                    collected_sources.append(source_item)
                                    if callback:
                                        callback([source_item], MSG_TYPE.MSG_TYPE_SOURCES_LIST)

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

        is_agentic_turn = (effective_use_mcps is not None and effective_use_mcps) or (use_data_store is not None and use_data_store)
        
        start_time = datetime.now()
        
        agent_result = None
        final_scratchpad = None
        final_raw_response = ""
        final_content = ""

        if is_agentic_turn:
            prompt_for_agent = self.export("markdown", branch_tip_id if branch_tip_id else self.active_branch_id, suppress_system_prompt=True)
            if debug:
                ASCIIColors.cyan("\n" + "="*50 + "\n--- DEBUG: AGENTIC TURN TRIGGERED ---\n" + f"--- PROMPT FOR AGENT (from discussion history) ---\n{prompt_for_agent}\n" + "="*50 + "\n")
            
            
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
            agent_result = self.lollmsClient.generate_with_mcp_rag(
                prompt=prompt_for_agent,
                use_mcps=effective_use_mcps,
                use_data_store=use_data_store,
                max_reasoning_steps=max_reasoning_steps,
                images=images,
                system_prompt = full_system_prompt,
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

            final_raw_response = self.lollmsClient.chat(self, images=images, branch_tip_id=branch_tip_id, **kwargs) or ""
            
            if debug:
                ASCIIColors.cyan("\n" + "="*50 + f"\n--- DEBUG: RAW SIMPLE CHAT RESPONSE ---\n{final_raw_response}\n" + "="*50 + "\n")

            if isinstance(final_raw_response, dict) and final_raw_response.get("status") == "error":
                raise Exception(final_raw_response.get("message", "Unknown error from lollmsClient.chat"))
            else:
                if remove_thinking_blocks:
                    final_content = self.lollmsClient.remove_thinking_blocks(final_raw_response)
                else:
                    final_content = final_raw_response
            final_scratchpad = None

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        token_count = self.lollmsClient.count_tokens(final_content)
        tok_per_sec = (token_count / duration) if duration > 0 else 0

        message_meta = {}
        if is_agentic_turn and isinstance(agent_result, dict):
            if "tool_calls" in agent_result: message_meta["tool_calls"] = agent_result["tool_calls"]
            if "sources" in agent_result: collected_sources.extend(agent_result["sources"])
            if agent_result.get("clarification_required", False): message_meta["clarification_required"] = True

        if collected_sources:
             message_meta["sources"] = collected_sources

        ai_message_obj = self.add_message(
            sender=personality.name if personality else "assistant",
            sender_type="assistant",
            content=final_content,
            raw_content=final_raw_response,
            scratchpad=final_scratchpad,
            tokens=token_count,
            generation_speed=tok_per_sec,
            parent_id=user_msg.id,
            model_name = self.lollmsClient.llm.model_name,
            binding_name = self.lollmsClient.llm.binding_name,
            metadata=message_meta
        )
        
        if self._is_db_backed and self.autosave:
            self.commit()
            
        return {"user_message": user_msg, "ai_message": ai_message_obj, "sources": collected_sources}

    def regenerate_branch(self, branch_tip_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Regenerates the AI response for a given message or the active branch's AI response.

        Instead of deleting the old response, this method simply starts a new generation
        from the parent message, creating a new branch (sibling to the original response).

        Args:
            branch_tip_id (Optional[str]): The ID of the message to regenerate from.
                                           If None, the currently active branch tip is used.
            **kwargs: Additional arguments for the chat() method.

        Returns:
            A dictionary with the user and the newly generated AI message.
        """
        self._rebuild_message_index() # Ensure index is fresh before operations

        target_id = branch_tip_id if branch_tip_id is not None else self.active_branch_id

        if not target_id or target_id not in self._message_index:
            raise ValueError("Regeneration failed: Target message ID not found or discussion is empty.")

        target_message_orm = self._message_index[target_id]
        
        # Determine the user message ID that will be the parent for the new AI generation
        if target_message_orm.sender_type == 'assistant':
            user_parent_id = target_message_orm.parent_id
            if user_parent_id is None or user_parent_id not in self._message_index:
                raise ValueError("Regeneration failed: Assistant message has no valid user parent to regenerate from.")
            user_msg_to_regenerate_from = self._message_index[user_parent_id]
        elif target_message_orm.sender_type == 'user':
            user_msg_to_regenerate_from = target_message_orm
            user_parent_id = user_msg_to_regenerate_from.id
        else:
            raise ValueError(f"Regeneration failed: Target message '{target_id}' is of an unexpected sender type '{target_message_orm.sender_type}'.")

        # --- Phase 1: Generate new AI response ---
        # The user message for the new generation is user_msg_to_regenerate_from
        self.active_branch_id = user_msg_to_regenerate_from.id
        
        # Call chat with add_user_message=False as the user message already exists (or was just found)
        # The chat method's add_message will set the new AI message as the active_branch_id.
        return self.chat(user_message="", add_user_message=False, branch_tip_id=user_msg_to_regenerate_from.id, **kwargs)

    def delete_branch(self, message_id: str):
        """Deletes a message and its entire descendant branch.

        This method removes the specified message and any messages that have it
        as a parent or an ancestor. After deletion, the active branch is moved
        to the parent of the deleted message. If the parent doesn't exist or is also
        deleted, it finds the most recent remaining message. Crucially, it re-parented
        children of the deleted message to the parent of the deleted message.

        This operation is only supported for database-backed discussions.

        Args:
            message_id: The ID of the message at the root of the branch to be deleted.

        Raises:
            NotImplementedError: If the discussion is not database-backed.
            ValueError: If the message ID is not found in the discussion.
        """
        if not self._is_db_backed:
            raise NotImplementedError("Branch deletion is only supported for database-backed discussions.")
        
        # Ensure message index is up-to-date with current DB state
        self._rebuild_message_index() 

        if message_id not in self._message_index:
            raise ValueError(f"Message with ID '{message_id}' not found in the discussion.")

        original_message_obj = self._message_index[message_id]
        new_parent_id_for_children = original_message_obj.parent_id
        
        # Identify direct children of the message being deleted (before removal from _db_discussion.messages)
        children_of_deleted_message_ids = [
            msg_obj.id for msg_obj in self._db_discussion.messages 
            if msg_obj.parent_id == message_id and msg_obj.id != message_id
        ]

        # Identify all messages to delete (including the one specified and its descendants)
        messages_to_delete_ids = set()
        queue = [message_id]
        processed_queue_idx = 0
        while processed_queue_idx < len(queue):
            current_msg_id = queue[processed_queue_idx]
            processed_queue_idx += 1
            
            if current_msg_id in messages_to_delete_ids:
                continue
            
            messages_to_delete_ids.add(current_msg_id)

            # Find children of current_msg_id from the current _message_index
            for msg_in_index_id, msg_in_index_obj in self._message_index.items():
                if msg_in_index_obj.parent_id == current_msg_id and msg_in_index_id not in messages_to_delete_ids:
                    queue.append(msg_in_index_id)

        # Re-parent children of the deleted message to its parent BEFORE removing messages from _db_discussion.messages
        reparented_children_count = 0
        for msg_obj in self._db_discussion.messages:
            if msg_obj.id in children_of_deleted_message_ids:
                msg_obj.parent_id = new_parent_id_for_children
                reparented_children_count += 1
        
        if reparented_children_count > 0:
            ASCIIColors.info(f"Re-parented {reparented_children_count} children from deleted message '{message_id}' to '{new_parent_id_for_children}'.")

        # Update the ORM's in-memory list of messages and mark for actual DB deletion
        self._db_discussion.messages = [
            msg_obj for msg_obj in self._db_discussion.messages if msg_obj.id not in messages_to_delete_ids
        ]
        self._messages_to_delete_from_db.update(messages_to_delete_ids)

        # Update the internal message index to reflect in-memory changes immediately
        temp_message_index = {}
        for mid, mobj in self._message_index.items():
            if mid not in messages_to_delete_ids:
                temp_message_index[mid] = mobj
        object.__setattr__(self, '_message_index', temp_message_index)


        # Determine the new active_branch_id by finding the most recent leaf
        # We first try to find a leaf descendant from the `new_parent_id_for_children` path.
        # If that path does not exist or has no leaves, then we search the entire discussion for the most recent leaf.
        new_active_id = None
        if new_parent_id_for_children and new_parent_id_for_children in self._message_index:
            new_active_id = self._find_deepest_leaf(new_parent_id_for_children)
        
        if new_active_id is None: # Fallback if direct re-parenting path doesn't yield a leaf
            new_active_id = self._find_deepest_leaf(None) # Find most recent leaf in the entire remaining discussion

        self.active_branch_id = new_active_id
        
        self.touch() # Mark for update and auto-save if configured
        print(f"Branch starting at {message_id} ({len(messages_to_delete_ids)} messages) removed. New active branch: {self.active_branch_id}")
        
    def export(self, format_type: str, branch_tip_id: Optional[str] = None, max_allowed_tokens: Optional[int] = None, suppress_system_prompt=False) -> Union[List[Dict], str]:
        """Exports the discussion history into a specified format.

        This method can format the conversation for different backends like OpenAI,
        Ollama, or the native `lollms_text` format. It intelligently handles
        context limits, non-destructive pruning summaries, and discussion-level images.

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
        if not suppress_system_prompt:
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
        # Get active discussion-level images using the corrected method
        active_discussion_b64 = self.get_active_images(branch_tip_id=None) # Get all active discussion images
        
        # Handle system message, which can now contain text and/or discussion-level images
        if full_system_prompt or (active_discussion_b64 and format_type in ["openai_chat", "ollama_chat", "markdown"]):
            discussion_level_images = build_image_dicts(active_discussion_b64)

            if format_type == "openai_chat":
                content_parts = []
                if full_system_prompt:
                    content_parts.append({"type": "text", "text": full_system_prompt})
                for img in discussion_level_images:
                    img_data = img['data']
                    url = f"data:image/jpeg;base64,{img_data}" if img['type'] == 'base64' else img_data
                    content_parts.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
                if content_parts:
                    messages.append({"role": "system", "content": content_parts})

            elif format_type == "ollama_chat":
                system_message_dict = {"role": "system", "content": full_system_prompt or ""}
                base64_images = [img['data'] for img in discussion_level_images if img['type'] == 'base64']
                if base64_images:
                    system_message_dict["images"] = base64_images
                messages.append(system_message_dict)

            elif format_type == "markdown":
                system_md_parts = []
                if full_system_prompt:
                    system_md_parts.append(f"system: {full_system_prompt}")

                for img in discussion_level_images:
                    img_data = img['data']
                    url = f"![Image](data:image/jpeg;base64,{img_data})" if img['type'] == 'base64' else f"![Image]({img_data})"
                    system_md_parts.append(f"\n{url}\n")
                
                if system_md_parts:
                    messages.append("".join(system_md_parts))

            else: # Fallback for any other potential format
                if full_system_prompt:
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
        Analyzes the current discussion to create a structured memory of its essence,
        focusing on preserving detailed technical content, problems, and solutions.
        This new memory is then automatically saved and loaded into the context for immediate use.

        Args:
            branch_tip_id: The ID of the message to use as the end of the context
                        for memory extraction. Defaults to the active branch.
        """
        try:
            discussion_context = self.export("markdown", branch_tip_id=branch_tip_id)
            if not discussion_context.strip():
                print("[INFO] Memorize: Discussion is empty, nothing to memorize.")
                return

            system_prompt = (
                "You are a Technical Knowledge Extraction AI specialized in preserving detailed information. "
                "Your task is to extract and preserve the ACTUAL CONTENT and DETAILS from discussions, not just summaries.\n\n"
                
                "CRITICAL INSTRUCTIONS:\n"
                "- If equations, formulas, or code are mentioned, INCLUDE THE FULL EQUATIONS/FORMULAS/CODE in the memory\n"
                "- If technical procedures or steps are discussed, preserve the EXACT STEPS\n"
                "- If specific values, parameters, or constants are mentioned, include them\n"
                "- If problems and solutions are discussed, capture BOTH the problem statement AND the detailed solution\n"
                "- Focus on ACTIONABLE and REFERENCEABLE content that someone could use later\n"
                "- Preserve technical terminology, variable names, and specific implementation details\n"
                "- Do NOT create high-level summaries - capture the actual working content\n\n"
                
                "OUTPUT FORMAT: JSON with 'title' (descriptive but specific) and 'content' (detailed technical content)"
            )
            
            prompt = (
                "Extract the key technical content from this discussion. Focus on preserving:\n"
                "1. Complete equations, formulas, or code snippets\n"
                "2. Specific problem statements and their detailed solutions\n"
                "3. Step-by-step procedures or algorithms\n"
                "4. Important constants, values, or parameters\n"
                "5. Technical concepts with their precise definitions\n"
                "6. Any implementation details or configuration settings\n\n"
                
                "IMPORTANT: Do not summarize what was discussed - extract the actual usable content.\n"
                "If Maxwell's equations were shown, include the actual equations.\n"
                "If code was provided, include the actual code.\n"
                "If a solution method was explained, include the actual steps.\n\n"
                
                f"--- Conversation to Extract From ---\n{discussion_context}\n\n"
                
                "Extract the technical essence that would be valuable for future reference:"
            )

            print("[INFO] Memorize: Extracting detailed technical content into a new memory...")
            memory_json = self.lollmsClient.generate_structured_content(
                prompt,
                schema={
                    "title": "A descriptive title indicating the type of problem solved (e.g., 'Python Import Error Fix', 'Database Connection Issue Solution')", 
                    "content": "Structured content with PROBLEM: [detailed problem] and SOLUTION: [detailed solution] sections"
                },
                system_prompt=system_prompt,
                temperature=0.1
            )

            if memory_json and memory_json.get("title") and memory_json.get("content"):
                print(f"[INFO] Memorize: New memory created and loaded into context: '{title}'.")
                return memory_json
            else:
                print("[WARNING] Memorize: Failed to generate a valid memory from the discussion.")
                return None
        except Exception as e:
            trace_exception(e)
            print(f"[ERROR] Memorize: Failed to create memory. {e}")

        def set_memory(self, memory_text: str):
            """Sets the discussion's memory content.
            This memory is included in the system context during exports and can be
            used to provide background information or retain important details across turns.
            Args:
                memory_text: The text to set as the discussion's memory.
            """       
            self.memory = memory_text.strip()
            self.touch()

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
        It also includes the token count for any active images in the message history.

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
                        "message_count": int,
                        "breakdown": {
                            "text_tokens": int,
                            "image_tokens": int,
                            "image_details": [{"message_id": str, "index": int, "tokens": int}]
                        }
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
        tokenizer_images = self.lollmsClient.count_image_tokens

        # --- 1. Assemble and Tokenize the Entire System Context Block ---
        system_context_tokens = 0
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
            system_context_tokens = tokenizer(system_block)
            
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
                "tokens": system_context_tokens,
                "breakdown": breakdown
            }

        # --- 2. Assemble and Tokenize the Message History Block (with images) ---
        branch_tip_id = branch_tip_id or self.active_branch_id
        messages_text = ""
        message_count = 0
        history_text_tokens = 0
        total_image_tokens = 0
        image_details_list = []
        
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
                    # Count image tokens
                    for i, image_b64 in enumerate(active_images):
                        tokens = self.lollmsClient.count_image_tokens(image_b64) 
                        if tokens > 0:
                            total_image_tokens += tokens
                            image_details_list.append({"message_id": msg.id, "index": i, "tokens": tokens})

                msg_text = f"!@>{sender_str}:\n{content}\n"
                message_parts.append(msg_text)
            
            messages_text = "".join(message_parts)
            message_count = len(messages_to_render)

        if messages_text or total_image_tokens > 0:
            history_text_tokens = tokenizer(messages_text)
            result["zones"]["message_history"] = {
                "content": messages_text,
                "tokens": history_text_tokens + total_image_tokens,
                "message_count": message_count,
                "breakdown": {
                    "text_tokens": history_text_tokens,
                    "image_tokens": total_image_tokens,
                    "image_details": image_details_list
                }
            }
        
        # Calculate discussion-level image tokens separately and add them to the total.
        active_discussion_images = self.get_discussion_images()
        active_discussion_b64 = [img['data'] for img in active_discussion_images if img.get('active', True)]
        discussion_image_tokens = sum(self.lollmsClient.count_image_tokens(img) for img in active_discussion_b64)
        
        # Add a new zone for discussion images for clarity
        if discussion_image_tokens > 0:
            result["zones"]["discussion_images"] = {
                "tokens": discussion_image_tokens,
                "image_count": len(active_discussion_b64)
            }


        # --- 3. Finalize the Total Count ---
        # Sum up the tokens from all calculated zones
        total_tokens = 0
        for zone in result["zones"].values():
            total_tokens += zone.get("tokens", 0)
        
        result["current_tokens"] = total_tokens

        return result
    
    def get_all_images(self, branch_tip_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all images from all messages in the specified or active branch.

        Each image is returned as a dictionary containing its data, its active status,
        the ID of the message it belongs to, and its index within that message.

        Args:
            branch_tip_id: The ID of the leaf message of the desired branch.
                           If None, the active branch's leaf is used.

        Returns:
            A list of dictionaries, each representing an image in the discussion branch.
            Example: [{"message_id": "...", "index": 0, "data": "...", "active": True}]
        """
        all_discussion_images = []
        branch = self.get_branch(branch_tip_id or self.active_branch_id)

        if not branch:
            return []

        for message in branch:
            message_images = message.get_all_images() # This returns [{"data":..., "active":...}]
            for i, img_info in enumerate(message_images):
                all_discussion_images.append({
                    "message_id": message.id,
                    "index": i,
                    "data": img_info["data"],
                    "active": img_info["active"]
                })
        
        return all_discussion_images

    def get_active_images(self, branch_tip_id: Optional[str] = None) -> List[str]:
        """
        Retrieves all *active* images from the discussion and from all messages
        in the specified or active branch.

        This method aggregates the active images from the discussion level and
        from each message's `get_active_images` call into a single flat list.

        Args:
            branch_tip_id: The ID of the leaf message of the desired branch.
                           If None, the active branch's leaf is used.

        Returns:
            A flat list of base64-encoded strings for all active images.
        """
        # Start with active discussion-level images. get_discussion_images() ensures
        # the format is correct (list of dicts) before we filter.
        discussion_images = self.get_discussion_images()
        active_discussion_images = [
            img['data'] for img in discussion_images if img.get('active', True)
        ]
        
        branch = self.get_branch(branch_tip_id or self.active_branch_id)
        if not branch:
            return active_discussion_images

        for message in branch:
            active_discussion_images.extend(message.get_active_images())

        return active_discussion_images

    def switch_to_branch(self, branch_id: str):
        """
        Switches the active discussion branch to the specified message ID.
        It then finds the deepest leaf descendant of that message and sets it as the active branch.
        """
        if branch_id not in self._message_index:
            ASCIIColors.warning(f"Attempted to switch to non-existent branch ID: {branch_id}. No action taken.")
            return

        # Find the deepest leaf in the branch starting from the provided branch_id
        # This ensures the active_branch_id is always a leaf
        new_active_leaf_id = self._find_deepest_leaf(branch_id)
        
        if new_active_leaf_id:
            self.active_branch_id = new_active_leaf_id
            self.touch() # Mark for saving if autosave is on
            ASCIIColors.info(f"Switched active branch to leaf: {self.active_branch_id}.")
        else:
            # Fallback: If no deeper leaf is found (e.g., branch_id is already a leaf or has no valid descendants)
            # then set active_branch_id to the provided branch_id.
            if branch_id in self._message_index:
                self.active_branch_id = branch_id
                self.touch()
                ASCIIColors.warning(f"Could not find a deeper leaf from branch ID {branch_id}. Active branch set to {branch_id}.")
            else:
                self.active_branch_id = None
                self.touch()
                ASCIIColors.error(f"Failed to set active branch: provided ID {branch_id} is invalid and no leaf could be found.")


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
            infos = self.lollmsClient.generate_structured_content(prompt = prompt, system_prompt=system_prompt, schema = title_generation_schema, n_predict=512)
            if infos is None or "title" not in infos:
                raise ValueError("Title generation failed or returned invalid data.")
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

    def add_discussion_image(self, image_b64: str, source: str = "user", active: bool = True):
        """
        Adds an image at the discussion level and marks it with a source.
        
        Args:
            image_b64: A base64-encoded string of the image to add.
            source: The origin of the image ('user', 'generation', 'artefact:<name> v<version>').
            active: Whether the image should be active by default.
        """
        # Ensures self.images is in the correct format before appending.
        current_images = self.get_discussion_images()

        new_image_data = {
            "data": image_b64,
            "source": source,
            "active": active,
            "created_at": datetime.utcnow().isoformat()
        }

        current_images.append(new_image_data)
        self.images = current_images
        self.touch()

    def get_discussion_images(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all images attached to the discussion, ensuring they are
        in the new dictionary format.

        - This method performs a lazy migration: if it detects the old format
          (a list of base64 strings), it converts it to the new format (a list
          of dictionaries) and marks the discussion for saving.
        """
        if not self.images or len(self.images) == 0 or type(self.images) is not list:
            return []

        # Check if migration is needed (if the first element is a string).
        if isinstance(self.images[0], str):
            ASCIIColors.yellow(f"Discussion {self.id}: Upgrading legacy discussion image format.")
            upgraded_images = []
            for i, img_data_str in enumerate(self.images):
                upgraded_images.append({
                    "data": img_data_str,
                    "source": "user",  # Assume 'user' for old format
                    "active": True,   # Assume active for old format
                    "created_at": datetime.utcnow().isoformat()
                })
            self.images = upgraded_images
            self.touch() # Mark for saving the upgraded format.
            
        return self.images


    def toggle_discussion_image_activation(self, index: int, active: Optional[bool] = None):
        """
        Toggles or sets the activation status of a discussion-level image at a given index.
        """
        current_images = self.get_discussion_images() # Ensures format is upgraded.
        if index >= len(current_images):
            raise IndexError("Discussion image index out of range.")

        if active is None:
            # Toggle the current state, defaulting to True if key is missing.
            current_images[index]["active"] = not current_images[index].get("active", True)
        else:
            current_images[index]["active"] = bool(active)
        
        self.images = current_images
        self.touch()

    def remove_discussion_image(self, index: int, commit: bool = True):
        """
        Removes a discussion-level image at a given index.
        """
        current_images = self.get_discussion_images() # Ensures format is upgraded.
        if index >= len(current_images):
            raise IndexError("Discussion image index out of range.")
        
        del current_images[index]
        self.images = current_images
        
        self.touch()
        if commit:
            self.commit()
              
    def fix_orphan_messages(self):
        """
        Detects and re-chains orphan messages or branches in the discussion.
        
        An "orphan message" is one whose parent_id points to a message that
        does not exist, or whose lineage cannot be traced back to a root.
        """
        ASCIIColors.info(f"Checking discussion {self.id} for orphan messages...")
        
        self._rebuild_message_index()
        
        all_messages_orms = list(self._message_index.values())
        if not all_messages_orms:
            ASCIIColors.yellow("No messages in discussion. Nothing to fix.")
            return

        message_map = {msg_orm.id: msg_orm for msg_orm in all_messages_orms}
        
        root_messages = []
        children_map = {msg_id: [] for msg_id in message_map.keys()}
        
        for msg_orm in all_messages_orms:
            if msg_orm.parent_id is None:
                root_messages.append(msg_orm)
            elif msg_orm.parent_id in message_map:
                children_map[msg_orm.parent_id].append(msg_orm.id)

        root_messages.sort(key=lambda msg: msg.created_at)
        primary_root = root_messages[0] if root_messages else None

        if primary_root:
            ASCIIColors.info(f"Primary discussion root identified: {primary_root.id}")
        else:
            ASCIIColors.warning("No root message found in discussion initially.")

        reachable_messages = set()
        queue = [r.id for r in root_messages]
        reachable_messages.update(queue)

        head = 0
        while head < len(queue):
            current_msg_id = queue[head]
            head += 1
            
            for child_id in children_map.get(current_msg_id, []):
                if child_id not in reachable_messages:
                    reachable_messages.add(child_id)
                    queue.append(child_id)

        orphan_messages_ids = set(message_map.keys()) - reachable_messages
        
        if not orphan_messages_ids:
            ASCIIColors.success("No orphan messages found. Discussion chain is healthy.")
            return
            
        ASCIIColors.warning(f"Found {len(orphan_messages_ids)} orphan message(s). Attempting to fix...")
        
        orphan_branch_tops = set()
        for orphan_id in orphan_messages_ids:
            current_id = orphan_id
            while message_map[current_id].parent_id is not None and message_map[current_id].parent_id in orphan_messages_ids:
                current_id = message_map[current_id].parent_id
            orphan_branch_tops.add(current_id)

        sorted_orphan_tops_orms = sorted(
            [message_map[top_id] for top_id in orphan_branch_tops], 
            key=lambda msg: msg.created_at
        )

        reparented_count = 0
        
        if not primary_root:
            if sorted_orphan_tops_orms:
                new_primary_root_orm = sorted_orphan_tops_orms[0]
                new_primary_root_orm.parent_id = None
                ASCIIColors.success(f"Set oldest orphan '{new_primary_root_orm.id}' as new primary root.")
                primary_root = new_primary_root_orm
                reparented_count += 1
                sorted_orphan_tops_orms = sorted_orphan_tops_orms[1:]
            else:
                ASCIIColors.error("Could not create a new root. Discussion remains unrooted.")
                return

        if primary_root:
            for orphan_top_orm in sorted_orphan_tops_orms:
                if orphan_top_orm.id != primary_root.id:
                    orphan_top_orm.parent_id = primary_root.id
                    ASCIIColors.info(f"Re-parented orphan '{orphan_top_orm.id}' to primary root '{primary_root.id}'.")
                    reparented_count += 1
        
        if reparented_count > 0:
            ASCIIColors.success(f"Successfully re-parented {reparented_count} orphan(s).")
            self.touch()
            self.commit()
            self._rebuild_message_index()
            self._validate_and_set_active_branch()
        else:
            ASCIIColors.yellow("No messages were re-parented.")


    @property
    def system_prompt(self) -> str:
        """Returns the system prompt for this discussion."""
        return self._system_prompt
    
    # Artefacts management system

    def list_artefacts(self) -> List[Dict[str, Any]]:
        """
        Lists all artefacts stored in the discussion's metadata.

        - Upgrades artefacts with missing fields to the new schema on-the-fly.
        - Computes the `is_loaded` status for each artefact.
        """
        metadata = self.metadata or {}
        artefacts = metadata.get("_artefacts", [])
        now = datetime.utcnow().isoformat()

        upgraded = []
        dirty = False
        for artefact in artefacts:
            fixed = artefact.copy()
            # Schema upgrade checks
            if "title" not in fixed: fixed["title"] = "untitled"; dirty = True
            if "content" not in fixed: fixed["content"] = ""; dirty = True
            if "images" not in fixed: fixed["images"] = []; dirty = True
            if "audios" not in fixed: fixed["audios"] = []; dirty = True
            if "videos" not in fixed: fixed["videos"] = []; dirty = True
            if "zip" not in fixed: fixed["zip"] = None; dirty = True
            if "version" not in fixed: fixed["version"] = 1; dirty = True
            if "created_at" not in fixed: fixed["created_at"] = now; dirty = True
            if "updated_at" not in fixed: fixed["updated_at"] = now; dirty = True

            # Reconstruct `is_loaded` from discussion context
            section_start = f"--- Document: {fixed['title']} v{fixed['version']} ---"
            is_content_loaded = section_start in (self.discussion_data_zone or "")
            
            artefact_source_id = f"artefact:{fixed['title']} v{fixed['version']}"
            is_image_loaded = any(
                img.get("source") == artefact_source_id for img in self.get_discussion_images()
            )

            fixed["is_loaded"] = is_content_loaded or is_image_loaded
            upgraded.append(fixed)

        if dirty:
            metadata["_artefacts"] = upgraded
            self.metadata = metadata
            self.commit()

        return upgraded

    def add_artefact(self, title: str, content: str = "", images: List[str] = None, audios: List[str] = None, videos: List[str] = None, zip_content: Optional[str] = None, version: int = 1, **extra_data) -> Dict[str, Any]:
        """
        Adds or overwrites an artefact in the discussion.
        """
        new_metadata = (self.metadata or {}).copy()
        artefacts = new_metadata.get("_artefacts", [])
        
        artefacts = [a for a in artefacts if not (a.get('title') == title and a.get('version') == version)]

        new_artefact = {
            "title": title, "version": version, "content": content,
            "images": images or [], "audios": audios or [], "videos": videos or [],
            "zip": zip_content, "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(), **extra_data
        }
        artefacts.append(new_artefact)
        
        new_metadata["_artefacts"] = artefacts
        self.metadata = new_metadata
        self.commit()
        return new_artefact

    def get_artefact(self, title: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves an artefact by title. Returns the latest version if `version` is None.
        """
        artefacts = self.list_artefacts()
        candidates = [a for a in artefacts if a.get('title') == title]
        if not candidates:
            return None
            
        if version is not None:
            return next((a for a in candidates if a.get('version') == version), None)
        else:
            return max(candidates, key=lambda a: a.get('version', 0))

    def update_artefact(self, title: str, new_content: str, new_images: List[str] = None, **extra_data) -> Dict[str, Any]:
        """
        Creates a new, incremented version of an existing artefact.
        """
        latest_artefact = self.get_artefact(title)
        if latest_artefact is None:
            raise ValueError(f"Cannot update non-existent artefact '{title}'.")
            
        latest_version = latest_artefact.get("version", 0)
        
        return self.add_artefact(
            title, content=new_content, images=new_images,
            audios=latest_artefact.get("audios", []),videos=latest_artefact.get("videos", []),
            zip_content=latest_artefact.get("zip"), **extra_data
        )

    def load_artefact_into_data_zone(self, title: str, version: Optional[int] = None):
        """
        Loads an artefact's content and images into the active discussion context.
        """
        artefact = self.get_artefact(title, version)
        if not artefact:
            raise ValueError(f"Artefact '{title}' not found.")

        # Load text content
        if artefact.get('content'):
            section = (
                f"--- Document: {artefact['title']} v{artefact['version']} ---\n"
                f"{artefact['content']}\n"
                f"--- End Document: {artefact['title']} ---\n\n"
            )
            if section not in (self.discussion_data_zone or ""):
                current_zone = self.discussion_data_zone or ""
                self.discussion_data_zone = current_zone.rstrip() + "\n\n" + section

        # Load images
        artefact_source_id = f"artefact:{artefact['title']} v{artefact['version']}"
        if artefact.get('images'):
            current_images_data = [img['data'] for img in self.get_discussion_images() if img.get('source') == artefact_source_id]
            for img_b64 in artefact['images']:
                if img_b64 not in current_images_data:
                    self.add_discussion_image(img_b64, source=artefact_source_id)
        
        self.touch()
        self.commit()
        print(f"Loaded artefact '{title}' v{artefact['version']} into context.")

    def unload_artefact_from_data_zone(self, title: str, version: Optional[int] = None):
        """
        Removes an artefact's content and images from the discussion context.
        """
        artefact = self.get_artefact(title, version)
        if not artefact:
            raise ValueError(f"Artefact '{title}' not found.")

        # Unload text content
        if self.discussion_data_zone and artefact.get('content'):
            section_start = f"--- Document: {artefact['title']} v{artefact['version']} ---"
            pattern = rf"\n*\s*{re.escape(section_start)}.*?--- End Document: {re.escape(artefact['title'])} ---\s*\n*"
            self.discussion_data_zone = re.sub(pattern, "", self.discussion_data_zone, flags=re.DOTALL).strip()
        
        # Unload images
        artefact_source_id = f"artefact:{artefact['title']} v{artefact['version']}"
        all_images = self.get_discussion_images()
        
        indices_to_remove = [i for i, img in enumerate(all_images) if img.get("source") == artefact_source_id]
        
        if indices_to_remove:
            for index in sorted(indices_to_remove, reverse=True):
                self.remove_discussion_image(index, commit=False)

        self.touch()
        self.commit()
        print(f"Unloaded artefact '{title}' v{artefact['version']} from context.")


    def is_artefact_loaded(self, title: str, version: Optional[int] = None) -> bool:
        """
        Checks if any part of an artefact is currently loaded in the context.
        """
        artefact = self.get_artefact(title, version)
        if not artefact:
            return False

        section_start = f"--- Document: {artefact['title']} v{artefact['version']} ---"
        if section_start in (self.discussion_data_zone or ""):
            return True

        artefact_source_id = f"artefact:{artefact['title']} v{artefact['version']}"
        if any(img.get("source") == artefact_source_id for img in self.get_discussion_images()):
            return True

        return False
        
    def export_as_artefact(self, title: str, version: int = 1, **extra_data) -> Dict[str, Any]:
        """
        Exports the discussion_data_zone content as a new artefact.
        """
        content = (self.discussion_data_zone or "").strip()
        if not content:
            raise ValueError("Discussion data zone is empty. Nothing to export.")

        return self.add_artefact(
            title=title, content=content, version=version, **extra_data
        )

    def remove_artefact(self, title: str, version: Optional[int] = None) -> int:
        """
        Removes artefacts by title. Removes all versions if `version` is None.
        
        Returns:
            The number of artefact entries removed.
        """
        new_metadata = (self.metadata or {}).copy()
        artefacts = new_metadata.get("_artefacts", [])
        if not artefacts:
            return 0

        initial_count = len(artefacts)
        
        if version is None:
            # Remove all versions with the matching title
            kept_artefacts = [a for a in artefacts if a.get('title') != title]
        else:
            # Remove only the specific title and version
            kept_artefacts = [a for a in artefacts if not (a.get('title') == title and a.get('version') == version)]

        if len(kept_artefacts) < initial_count:
            new_metadata["_artefacts"] = kept_artefacts
            self.metadata = new_metadata
            self.commit()

        removed_count = initial_count - len(kept_artefacts)
        if removed_count > 0:
            print(f"Removed {removed_count} artefact(s) titled '{title}'.")
            
        return removed_count

    def clone_without_messages(self) -> 'LollmsDiscussion':
        """
        Creates a new discussion with the same context but no message history.
        """
        discussion_data = {
            "system_prompt": self.system_prompt,
            "user_data_zone": self.user_data_zone,
            "discussion_data_zone": self.discussion_data_zone,
            "personality_data_zone": self.personality_data_zone,
            "memory": self.memory,
            "participants": self.participants,
            "discussion_metadata": self.metadata,
            "images": [img.copy() for img in self.get_discussion_images()],
        }

        new_discussion = LollmsDiscussion.create_new(
            lollms_client=self.lollmsClient,
            db_manager=self.db_manager,
            **discussion_data
        )
        return new_discussion

    def export_to_json_str(self) -> str:
        """
        Serializes the entire discussion state to a JSON string.
        """
        export_data = {
            "id": self.id,
            "system_prompt": self.system_prompt,
            "user_data_zone": self.user_data_zone,
            "discussion_data_zone": self.discussion_data_zone,
            "personality_data_zone": self.personality_data_zone,
            "memory": self.memory,
            "participants": self.participants,
            "active_branch_id": self.active_branch_id,
            "discussion_metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "pruning_summary": self.pruning_summary,
            "pruning_point_id": self.pruning_point_id,
            "images": self.get_discussion_images(), # Ensures new format is exported
            "messages": []
        }

        for msg in self.get_all_messages_flat():
            msg_data = {
                "id": msg.id, "discussion_id": msg.discussion_id, "parent_id": msg.parent_id,
                "sender": msg.sender, "sender_type": msg.sender_type,
                "raw_content": msg.raw_content, "thoughts": msg.thoughts, "content": msg.content,
                "scratchpad": msg.scratchpad, "tokens": msg.tokens,
                "binding_name": msg.binding_name, "model_name": msg.model_name,
                "generation_speed": msg.generation_speed, "message_metadata": msg.metadata,
                "images": msg.images, "active_images": msg.active_images,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
            export_data["messages"].append(msg_data)
        
        return json.dumps(export_data, indent=2)

    @classmethod
    def import_from_json_str(
        cls,
        json_str: str,
        lollms_client: 'LollmsClient',
        db_manager: Optional[LollmsDataManager] = None
    ) -> 'LollmsDiscussion':
        """
        Creates a new LollmsDiscussion instance from a JSON string.
        """
        data = json.loads(json_str)
        
        message_data_list = data.pop("messages", [])
        
        # Clean up deprecated fields before creation if they exist in the JSON.
        data.pop("active_images", None)
        
        new_discussion = cls.create_new(
            lollms_client=lollms_client,
            db_manager=db_manager,
            **data
        )

        for msg_data in message_data_list:
            if 'created_at' in msg_data and msg_data['created_at']:
                msg_data['created_at'] = datetime.fromisoformat(msg_data['created_at'])
            
            new_discussion.add_message(**msg_data)

        new_discussion.active_branch_id = data.get('active_branch_id')
        if db_manager:
            new_discussion.commit()

        return new_discussion
