# lollms_discussion/_db.py
# Database layer: encryption helper, dynamic ORM model factory, LollmsDataManager.

import base64
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Type, List, Dict

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, JSON,
                        LargeBinary, String, Text, create_engine)
from sqlalchemy.orm import (Session, declarative_base, declared_attr,
                            relationship, sessionmaker)
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.types import TypeDecorator
from sqlalchemy import text
from types import SimpleNamespace

from ascii_colors import ASCIIColors

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Encryption helper
# ---------------------------------------------------------------------------

class EncryptedString(TypeDecorator):
    """A SQLAlchemy TypeDecorator for field-level database encryption."""
    impl = LargeBinary
    cache_ok = True

    def __init__(self, key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not ENCRYPTION_AVAILABLE:
            raise ImportError("'cryptography' is required for DB encryption.")
        self.salt = b'lollms-fixed-salt-for-db-encryption'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32, salt=self.salt,
            iterations=480000, backend=default_backend()
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        self.fernet = Fernet(derived_key)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return self.fernet.encrypt(value.encode('utf-8'))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        try:
            return self.fernet.decrypt(value).decode('utf-8')
        except InvalidToken:
            return "<DECRYPTION_FAILED: Invalid Key or Corrupt Data>"


# ---------------------------------------------------------------------------
# Dynamic model factory
# ---------------------------------------------------------------------------

def create_dynamic_models(
    discussion_mixin: Optional[Type] = None,
    message_mixin: Optional[Type] = None,
    encryption_key: Optional[str] = None
) -> tuple:
    Base = declarative_base()
    EncryptedText = EncryptedString(encryption_key) if encryption_key else Text

    class DiscussionBase:
        __abstract__ = True
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        system_prompt = Column(EncryptedText, nullable=True)
        user_data_zone = Column(EncryptedText, nullable=True)
        discussion_data_zone = Column(EncryptedText, nullable=True)
        personality_data_zone = Column(EncryptedText, nullable=True)
        memory = Column(EncryptedText, nullable=True)
        participants = Column(JSON, nullable=True, default=dict)
        active_branch_id = Column(String, nullable=True)
        discussion_metadata = Column(JSON, nullable=True, default=dict)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        pruning_summary = Column(EncryptedText, nullable=True)
        pruning_point_id = Column(String, nullable=True)

        @declared_attr
        def messages(cls):
            return relationship("Message", back_populates="discussion",
                                cascade="all, delete-orphan", lazy="joined")

    class MessageBase:
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
        active_images = Column(JSON, nullable=True, default=list)
        created_at = Column(DateTime, default=datetime.utcnow)

        @declared_attr
        def discussion(cls):
            return relationship("Discussion", back_populates="messages")

    discussion_bases = (discussion_mixin, DiscussionBase, Base) if discussion_mixin else (DiscussionBase, Base)
    DynamicDiscussion = type('Discussion', discussion_bases, {'__tablename__': 'discussions'})

    message_bases = (message_mixin, MessageBase, Base) if message_mixin else (MessageBase, Base)
    DynamicMessage = type('Message', message_bases, {'__tablename__': 'messages'})

    return Base, DynamicDiscussion, DynamicMessage


# ---------------------------------------------------------------------------
# LollmsDataManager
# ---------------------------------------------------------------------------

class LollmsDataManager:
    """Manages database connection, session, and table creation."""

    def __init__(self, db_path: str, discussion_mixin=None, message_mixin=None, encryption_key=None):
        if not db_path:
            raise ValueError("Database path cannot be empty.")
        self.Base, self.DiscussionModel, self.MessageModel = create_dynamic_models(
            discussion_mixin, message_mixin, encryption_key
        )
        self.engine = create_engine(db_path)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_and_migrate_tables()

    def discussion_exists(self, discussion_id: str) -> bool:
        with self.get_session() as session:
            return session.query(self.DiscussionModel).filter_by(id=discussion_id).first() is not None

    @staticmethod
    def new_message(**kwargs) -> 'SimpleNamespace':
        if 'sender' not in kwargs:
            kwargs['sender'] = 'user' if kwargs.get('sender_type') == 'user' else 'assistant'
        if 'sender_type' not in kwargs:
            kwargs['sender_type'] = 'user' if kwargs.get('sender') == 'user' else 'assistant'
        message_data = {
            'id': str(uuid.uuid4()), 'parent_id': None, 'discussion_id': None,
            'created_at': datetime.utcnow(), 'raw_content': kwargs.get('content'),
            'thoughts': None, 'scratchpad': None, 'tokens': None,
            'binding_name': None, 'model_name': None, 'generation_speed': None,
            'message_metadata': {}, 'images': [], 'active_images': [],
        }
        message_data.update(kwargs)
        if 'metadata' in message_data:
            message_data['message_metadata'] = message_data.pop('metadata')
        return SimpleNamespace(**message_data)

    def create_and_migrate_tables(self):
        self.Base.metadata.create_all(bind=self.engine)
        try:
            with self.engine.connect() as connection:
                cursor = connection.execute(text("PRAGMA table_info(discussions)"))
                columns = {row[1] for row in cursor.fetchall()}
                migrations = [
                    ('pruning_summary',      "ALTER TABLE discussions ADD COLUMN pruning_summary TEXT"),
                    ('pruning_point_id',     "ALTER TABLE discussions ADD COLUMN pruning_point_id VARCHAR"),
                    ('user_data_zone',       "ALTER TABLE discussions ADD COLUMN user_data_zone TEXT"),
                    ('discussion_data_zone', "ALTER TABLE discussions ADD COLUMN discussion_data_zone TEXT"),
                    ('personality_data_zone',"ALTER TABLE discussions ADD COLUMN personality_data_zone TEXT"),
                    ('memory',               "ALTER TABLE discussions ADD COLUMN memory TEXT"),
                ]
                for col, sql in migrations:
                    if col not in columns:
                        ASCIIColors.info(f"  -> Upgrading 'discussions' table: Adding '{col}' column.")
                        connection.execute(text(sql))
                if 'data_zone' in columns:
                    connection.execute(text("ALTER TABLE discussions DROP COLUMN data_zone"))

                cursor = connection.execute(text("PRAGMA table_info(messages)"))
                columns = {row[1] for row in cursor.fetchall()}
                if 'active_images' not in columns:
                    connection.execute(text("ALTER TABLE messages ADD COLUMN active_images TEXT"))
                connection.commit()
        except Exception as e:
            ASCIIColors.red(f"\n--- DATABASE MIGRATION WARNING ---\n{e}\n---")

    def get_session(self) -> Session:
        return self.SessionLocal()

    def list_discussions(self, limit=None) -> List[Dict]:
        with self.get_session() as session:
            discussions = session.query(self.DiscussionModel).all()
            return [{c.name: getattr(disc, c.name) for c in disc.__table__.columns} for disc in (discussions[:limit] if limit else discussions)]

    def get_discussion(self, lollms_client, discussion_id: str, **kwargs):
        # Imported here to avoid circular dependency at module load time.
        from lollms_client.lollms_discussion import LollmsDiscussion
        with self.get_session() as session:
            try:
                db_disc = session.query(self.DiscussionModel).filter_by(id=discussion_id).one()
                session.expunge(db_disc)
                return LollmsDiscussion(lollmsClient=lollms_client, db_manager=self,
                                        db_discussion_obj=db_disc, **kwargs)
            except NoResultFound:
                return None

    def search_discussions(self, **criteria) -> List[Dict]:
        with self.get_session() as session:
            query = session.query(self.DiscussionModel)
            for key, value in criteria.items():
                if hasattr(self.DiscussionModel, key):
                    query = query.filter(getattr(self.DiscussionModel, key).ilike(f"%{value}%"))
            discussions = query.all()
            return [{c.name: getattr(disc, c.name) for c in disc.__table__.columns} for disc in discussions]

    def delete_discussion(self, discussion_id: str):
        with self.get_session() as session:
            db_disc = session.query(self.DiscussionModel).filter_by(id=discussion_id).first()
            if db_disc:
                session.delete(db_disc)
                session.commit()
