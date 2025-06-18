import yaml
import json
import base64
import os
import uuid
import shutil
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Union, Any, Type, Callable
from pathlib import Path

from sqlalchemy import (create_engine, Column, String, Text, Integer, DateTime,
                        ForeignKey, JSON, Boolean, LargeBinary, Index)
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base
from sqlalchemy.types import TypeDecorator

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

if False: 
    from lollms_client import LollmsClient
    from lollms_client.lollms_types import MSG_TYPE

class EncryptedString(TypeDecorator):
    impl = LargeBinary
    cache_ok = True

    def __init__(self, key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not ENCRYPTION_AVAILABLE: raise ImportError("'cryptography' is required for DB encryption.")
        self.salt = b'lollms-fixed-salt-for-db-encryption'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32, salt=self.salt,
            iterations=480000, backend=default_backend()
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        self.fernet = Fernet(derived_key)

    def process_bind_param(self, value: Optional[str], dialect) -> Optional[bytes]:
        if value is None: return None
        return self.fernet.encrypt(value.encode('utf-8'))

    def process_result_value(self, value: Optional[bytes], dialect) -> Optional[str]:
        if value is None: return None
        try:
            return self.fernet.decrypt(value).decode('utf-8')
        except InvalidToken:
            return "<DECRYPTION_FAILED: Invalid Key or Corrupt Data>"

def create_dynamic_models(discussion_mixin: Optional[Type] = None, message_mixin: Optional[Type] = None, encryption_key: Optional[str] = None):
    Base = declarative_base()
    EncryptedText = EncryptedString(encryption_key) if encryption_key else Text

    class DiscussionBase(Base):
        __abstract__ = True
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        system_prompt = Column(EncryptedText, nullable=True)
        participants = Column(JSON, nullable=True, default=dict)
        active_branch_id = Column(String, nullable=True)
        discussion_metadata = Column(JSON, nullable=True, default=dict)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    class MessageBase(Base):
        __abstract__ = True
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        discussion_id = Column(String, ForeignKey('discussions.id'), nullable=False)
        parent_id = Column(String, ForeignKey('messages.id'), nullable=True)
        sender = Column(String, nullable=False)
        sender_type = Column(String, nullable=False)
        content = Column(EncryptedText, nullable=False)
        message_metadata = Column(JSON, nullable=True, default=dict)
        images = Column(JSON, nullable=True, default=list)
        created_at = Column(DateTime, default=datetime.utcnow)
    
    discussion_attrs = {'__tablename__': 'discussions'}
    if hasattr(discussion_mixin, '__table_args__'):
        discussion_attrs['__table_args__'] = discussion_mixin.__table_args__
    if discussion_mixin:
        for attr, col in discussion_mixin.__dict__.items():
            if isinstance(col, Column):
                discussion_attrs[attr] = col

    message_attrs = {'__tablename__': 'messages'}
    if hasattr(message_mixin, '__table_args__'):
        message_attrs['__table_args__'] = message_mixin.__table_args__
    if message_mixin:
        for attr, col in message_mixin.__dict__.items():
            if isinstance(col, Column):
                message_attrs[attr] = col
    
    discussion_bases = (discussion_mixin, DiscussionBase) if discussion_mixin else (DiscussionBase,)
    DynamicDiscussion = type('Discussion', discussion_bases, discussion_attrs)

    message_bases = (message_mixin, MessageBase) if message_mixin else (MessageBase,)
    DynamicMessage = type('Message', message_bases, message_attrs)
    
    DynamicDiscussion.messages = relationship(DynamicMessage, back_populates="discussion", cascade="all, delete-orphan", lazy="joined")
    DynamicMessage.discussion = relationship(DynamicDiscussion, back_populates="messages")

    return Base, DynamicDiscussion, DynamicMessage

class DatabaseManager:
    def __init__(self, db_path: str, discussion_mixin: Optional[Type] = None, message_mixin: Optional[Type] = None, encryption_key: Optional[str] = None):
        if not db_path: raise ValueError("Database path cannot be empty.")
        self.Base, self.DiscussionModel, self.MessageModel = create_dynamic_models(
            discussion_mixin, message_mixin, encryption_key
        )
        self.engine = create_engine(db_path)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()

    def create_tables(self):
        self.Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()
        
    def list_discussions(self) -> List[Dict]:
        session = self.get_session()
        discussions = session.query(self.DiscussionModel).all()
        session.close()
        discussion_list = []
        for disc in discussions:
            disc_dict = {c.name: getattr(disc, c.name) for c in disc.__table__.columns}
            discussion_list.append(disc_dict)
        return discussion_list

    def get_discussion(self, lollms_client: 'LollmsClient', discussion_id: str, **kwargs) -> Optional['LollmsDiscussion']:
        session = self.get_session()
        db_disc = session.query(self.DiscussionModel).filter_by(id=discussion_id).first()
        session.close()
        if db_disc:
            return LollmsDiscussion(lollmsClient=lollms_client, discussion_id=discussion_id, db_manager=self, **kwargs)
        return None

    def search_discussions(self, **criteria) -> List[Dict]:
        session = self.get_session()
        query = session.query(self.DiscussionModel)
        for key, value in criteria.items():
            query = query.filter(getattr(self.DiscussionModel, key).ilike(f"%{value}%"))
        discussions = query.all()
        session.close()
        discussion_list = []
        for disc in discussions:
            disc_dict = {c.name: getattr(disc, c.name) for c in disc.__table__.columns}
            discussion_list.append(disc_dict)
        return discussion_list

    def delete_discussion(self, discussion_id: str):
        session = self.get_session()
        db_disc = session.query(self.DiscussionModel).filter_by(id=discussion_id).first()
        if db_disc:
            session.delete(db_disc)
            session.commit()
        session.close()

class LollmsDiscussion:
    def __init__(self, lollmsClient: 'LollmsClient', discussion_id: Optional[str] = None, db_manager: Optional[DatabaseManager] = None, autosave: bool = False, max_context_size: Optional[int] = None):
        self.lollmsClient = lollmsClient
        self.db_manager = db_manager
        self.autosave = autosave
        self.max_context_size = max_context_size
        self._is_db_backed = db_manager is not None
        
        self.session = None
        self.db_discussion = None
        self._messages_to_delete = []

        self._reset_in_memory_state()

        if self._is_db_backed:
            if not discussion_id: raise ValueError("A discussion_id is required for database-backed discussions.")
            self.session = db_manager.get_session()
            self._load_from_db(discussion_id)
        else:
            self.id = discussion_id or str(uuid.uuid4())
            self.created_at = datetime.utcnow()
            self.updated_at = self.created_at

    def _reset_in_memory_state(self):
        self.id: str = ""
        self.system_prompt: Optional[str] = None
        self.participants: Dict[str, str] = {}
        self.active_branch_id: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.scratchpad: str = ""
        self.messages: List[Dict] = []
        self.message_index: Dict[str, Dict] = {}
        self.created_at: Optional[datetime] = None
        self.updated_at: Optional[datetime] = None

    def _load_from_db(self, discussion_id: str):
        self.db_discussion = self.session.query(self.db_manager.DiscussionModel).filter(self.db_manager.DiscussionModel.id == discussion_id).one()
        
        self.id = self.db_discussion.id
        self.system_prompt = self.db_discussion.system_prompt
        self.participants = self.db_discussion.participants or {}
        self.active_branch_id = self.db_discussion.active_branch_id
        self.metadata = self.db_discussion.discussion_metadata or {}
        
        self.messages = []
        self.message_index = {}
        for msg in self.db_discussion.messages:
            msg_dict = {c.name: getattr(msg, c.name) for c in msg.__table__.columns}
            if 'message_metadata' in msg_dict:
                msg_dict['metadata'] = msg_dict.pop('message_metadata')
            self.messages.append(msg_dict)
            self.message_index[msg.id] = msg_dict
            
    def commit(self):
        if not self._is_db_backed or not self.session: return
        
        if self.db_discussion:
            self.db_discussion.system_prompt = self.system_prompt
            self.db_discussion.participants = self.participants
            self.db_discussion.active_branch_id = self.active_branch_id
            self.db_discussion.discussion_metadata = self.metadata
            self.db_discussion.updated_at = datetime.utcnow()

        for msg_id in self._messages_to_delete:
            msg_to_del = self.session.query(self.db_manager.MessageModel).filter_by(id=msg_id).first()
            if msg_to_del: self.session.delete(msg_to_del)
        self._messages_to_delete.clear()

        for msg_data in self.messages:
            msg_id = msg_data['id']
            msg_orm = self.session.query(self.db_manager.MessageModel).filter_by(id=msg_id).first()
            
            if 'metadata' in msg_data:
                msg_data['message_metadata'] = msg_data.pop('metadata',None)

            if not msg_orm:
                msg_data_copy = msg_data.copy()
                valid_keys = {c.name for c in self.db_manager.MessageModel.__table__.columns}
                filtered_msg_data = {k: v for k, v in msg_data_copy.items() if k in valid_keys}
                msg_orm = self.db_manager.MessageModel(**filtered_msg_data)
                self.session.add(msg_orm)
            else:
                for key, value in msg_data.items():
                    if hasattr(msg_orm, key):
                        setattr(msg_orm, key, value)
        
        self.session.commit()

    def touch(self):
        self.updated_at = datetime.utcnow()
        if self._is_db_backed and self.autosave:
            self.commit()

    @classmethod
    def create_new(cls, lollms_client: 'LollmsClient', db_manager: Optional[DatabaseManager] = None, **kwargs) -> 'LollmsDiscussion':
        init_args = {
            'autosave': kwargs.pop('autosave', False),
            'max_context_size': kwargs.pop('max_context_size', None)
        }
        
        if db_manager:
            session = db_manager.get_session()
            valid_keys = db_manager.DiscussionModel.__table__.columns.keys()
            db_creation_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            db_discussion = db_manager.DiscussionModel(**db_creation_args)
            session.add(db_discussion)
            session.commit()
            return cls(lollmsClient=lollms_client, discussion_id=db_discussion.id, db_manager=db_manager, **init_args)
        else:
            discussion_id = kwargs.get('discussion_id')
            return cls(lollmsClient=lollms_client, discussion_id=discussion_id, **init_args)

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        self.touch()

    def set_participants(self, participants: Dict[str, str]):
        for name, role in participants.items():
            if role not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role '{role}' for participant '{name}'")
        self.participants = participants
        self.touch()

    def add_message(self, **kwargs) -> Dict:
        msg_id = kwargs.get('id', str(uuid.uuid4()))
        parent_id = kwargs.get('parent_id', self.active_branch_id or None)
        
        message_data = {
            'id': msg_id, 'parent_id': parent_id,
            'discussion_id': self.id, 'created_at': datetime.utcnow(),
            **kwargs
        }
        
        self.messages.append(message_data)
        self.message_index[msg_id] = message_data
        self.active_branch_id = msg_id
        self.touch()
        return message_data

    def get_branch(self, leaf_id: Optional[str]) -> List[Dict]:
        if not leaf_id: return []
        branch = []
        current_id: Optional[str] = leaf_id
        while current_id and current_id in self.message_index:
            msg = self.message_index[current_id]
            branch.append(msg)
            current_id = msg.get('parent_id')
        return list(reversed(branch))
        
    def chat(self, user_message: str, show_thoughts: bool = False, **kwargs) -> Dict:
        if self.max_context_size is not None:
            self.summarize_and_prune(self.max_context_size)
        
        if user_message:
            self.add_message(sender="user", sender_type="user", content=user_message)

        from lollms_client.lollms_types import MSG_TYPE
        
        is_streaming = "streaming_callback" in kwargs and kwargs["streaming_callback"] is not None

        if is_streaming:
            full_response_parts = []
            token_buffer = ""
            in_thought_block = False
            original_callback = kwargs.get("streaming_callback")

            def accumulating_callback(token: str, msg_type: MSG_TYPE = MSG_TYPE.MSG_TYPE_CHUNK):
                nonlocal token_buffer, in_thought_block
                continue_streaming = True
                
                if token: token_buffer += token

                while True:
                    if in_thought_block:
                        end_tag_pos = token_buffer.find("</think>")
                        if end_tag_pos != -1:
                            thought_chunk = token_buffer[:end_tag_pos]
                            if show_thoughts and original_callback and thought_chunk:
                                if not original_callback(thought_chunk, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK): continue_streaming = False
                            in_thought_block = False
                            token_buffer = token_buffer[end_tag_pos + len("</think>"):]
                        else:
                            if show_thoughts and original_callback and token_buffer:
                                if not original_callback(token_buffer, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK): continue_streaming = False
                            token_buffer = ""
                            break
                    else:
                        start_tag_pos = token_buffer.find("<think>")
                        if start_tag_pos != -1:
                            response_chunk = token_buffer[:start_tag_pos]
                            if response_chunk:
                                full_response_parts.append(response_chunk)
                                if original_callback:
                                    if not original_callback(response_chunk, MSG_TYPE.MSG_TYPE_CHUNK): continue_streaming = False
                            in_thought_block = True
                            token_buffer = token_buffer[start_tag_pos + len("<think>"):]
                        else:
                            if token_buffer:
                                full_response_parts.append(token_buffer)
                                if original_callback:
                                    if not original_callback(token_buffer, MSG_TYPE.MSG_TYPE_CHUNK): continue_streaming = False
                            token_buffer = ""
                            break
                return continue_streaming
            
            kwargs["streaming_callback"] = accumulating_callback
            kwargs["stream"] = True
            
            self.lollmsClient.chat(self, **kwargs)
            ai_response = "".join(full_response_parts)
        else:
            kwargs["stream"] = False
            raw_response = self.lollmsClient.chat(self, **kwargs)
            ai_response = self.lollmsClient.remove_thinking_blocks(raw_response) if raw_response else ""

        ai_message_obj = self.add_message(sender="assistant", sender_type="assistant", content=ai_response)

        if self._is_db_backed and not self.autosave:
            self.commit()
            
        return ai_message_obj

    def regenerate_branch(self, show_thoughts: bool = False, **kwargs) -> Dict:
        last_message = self.message_index.get(self.active_branch_id)
        if not last_message or last_message['sender_type'] != 'assistant':
            raise ValueError("Can only regenerate from an assistant's message.")
        
        parent_id = last_message['parent_id']
        self.active_branch_id = parent_id
        
        self.messages = [m for m in self.messages if m['id'] != last_message['id']]
        self._messages_to_delete.append(last_message['id'])
        self._rebuild_in_memory_indexes()

        new_ai_response_obj = self.chat("", show_thoughts, **kwargs)
        return new_ai_response_obj

    def delete_branch(self, message_id: str):
        if not self._is_db_backed:
            raise NotImplementedError("Branch deletion is only supported for database-backed discussions.")
        
        if message_id not in self.message_index:
            raise ValueError("Message not found.")

        msg_to_delete = self.session.query(self.db_manager.MessageModel).filter_by(id=message_id).first()
        if msg_to_delete:
            parent_id = msg_to_delete.parent_id
            self.session.delete(msg_to_delete)
            self.active_branch_id = parent_id
            self.commit()
            self._load_from_db(self.id)

    def switch_to_branch(self, message_id: str):
        if message_id not in self.message_index:
            raise ValueError(f"Message ID '{message_id}' not found in the current discussion.")
        self.active_branch_id = message_id
        if self._is_db_backed:
            self.db_discussion.active_branch_id = message_id
            if self.autosave: self.commit()

    def format_discussion(self, max_allowed_tokens: int, branch_tip_id: Optional[str] = None) -> str:
        return self.export("lollms_text", branch_tip_id, max_allowed_tokens)

    def _get_full_system_prompt(self) -> Optional[str]:
        full_sys_prompt_parts = []
        if self.scratchpad:
            full_sys_prompt_parts.append("--- KNOWLEDGE SCRATCHPAD ---")
            full_sys_prompt_parts.append(self.scratchpad.strip())
            full_sys_prompt_parts.append("--- END SCRATCHPAD ---")
        
        if self.system_prompt and self.system_prompt.strip():
            full_sys_prompt_parts.append(self.system_prompt.strip())
            
        return "\n\n".join(full_sys_prompt_parts) if full_sys_prompt_parts else None

    def export(self, format_type: str, branch_tip_id: Optional[str] = None, max_allowed_tokens: Optional[int] = None) -> Union[List[Dict], str]:
        if branch_tip_id is None: branch_tip_id = self.active_branch_id
        if not branch_tip_id and format_type in ["lollms_text", "openai_chat", "ollama_chat"]:
            return "" if format_type == "lollms_text" else []

        branch = self.get_branch(branch_tip_id)
        full_system_prompt = self._get_full_system_prompt()
        
        participants = self.participants or {}

        if format_type == "lollms_text":
            prompt_parts = []
            current_tokens = 0
            
            if full_system_prompt:
                sys_msg_text = f"!@>system:\n{full_system_prompt}\n"
                sys_tokens = self.lollmsClient.count_tokens(sys_msg_text)
                if max_allowed_tokens is None or sys_tokens <= max_allowed_tokens:
                    prompt_parts.append(sys_msg_text)
                    current_tokens += sys_tokens
            
            for msg in reversed(branch):
                sender_str = msg['sender'].replace(':', '').replace('!@>', '')
                content = msg['content'].strip()
                if msg.get('images'): content += f"\n({len(msg['images'])} image(s) attached)"
                msg_text = f"!@>{sender_str}:\n{content}\n"
                msg_tokens = self.lollmsClient.count_tokens(msg_text)

                if max_allowed_tokens is not None and current_tokens + msg_tokens > max_allowed_tokens: break
                prompt_parts.insert(1 if full_system_prompt else 0, msg_text)
                current_tokens += msg_tokens
            return "".join(prompt_parts).strip()

        messages = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        
        for msg in branch:
            role = participants.get(msg['sender'], "user")
            content = msg.get('content', '').strip()
            images = msg.get('images', [])

            if format_type == "openai_chat":
                if images:
                    content_parts = [{"type": "text", "text": content}] if content else []
                    for img in images:
                        image_url = img['data'] if img['type'] == 'url' else f"data:image/jpeg;base64,{img['data']}"
                        content_parts.append({"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}})
                    messages.append({"role": role, "content": content_parts})
                else:
                    messages.append({"role": role, "content": content})
            elif format_type == "ollama_chat":
                message_dict = {"role": role, "content": content}
                base64_images = [img['data'] for img in images or [] if img['type'] == 'base64']
                if base64_images:
                    message_dict["images"] = base64_images
                messages.append(message_dict)
            else:
                raise ValueError(f"Unsupported export format_type: {format_type}")
        
        return messages

    def summarize_and_prune(self, max_tokens: int, preserve_last_n: int = 4):
        branch_tip_id = self.active_branch_id
        if not branch_tip_id: return

        current_prompt_text = self.format_discussion(999999, branch_tip_id)
        current_tokens = self.lollmsClient.count_tokens(current_prompt_text)
        if current_tokens <= max_tokens: return

        branch = self.get_branch(branch_tip_id)
        if len(branch) <= preserve_last_n: return

        messages_to_prune = branch[:-preserve_last_n]
        text_to_summarize = "\n\n".join([f"{m['sender']}: {m['content']}" for m in messages_to_prune])
        
        summary_prompt = f"Concisely summarize this conversation excerpt:\n---\n{text_to_summarize}\n---\nSUMMARY:"
        try:
            summary = self.lollmsClient.generate_text(summary_prompt, n_predict=300, temperature=0.1)
        except Exception as e:
            print(f"\n[WARNING] Pruning failed, couldn't generate summary: {e}")
            return

        new_scratchpad_content = f"{self.scratchpad}\n\n--- Summary of earlier conversation ---\n{summary.strip()}"
        self.scratchpad = new_scratchpad_content.strip()

        pruned_ids = {msg['id'] for msg in messages_to_prune}
        self.messages = [m for m in self.messages if m['id'] not in pruned_ids]
        self._messages_to_delete.extend(list(pruned_ids))
        self._rebuild_in_memory_indexes()

        print(f"\n[INFO] Discussion auto-pruned. {len(messages_to_prune)} messages summarized.")

    def to_dict(self):
        messages_copy = [msg.copy() for msg in self.messages]
        for msg in messages_copy:
            if 'created_at' in msg and isinstance(msg['created_at'], datetime):
                msg['created_at'] = msg['created_at'].isoformat()
            if 'message_metadata' in msg:
                msg['metadata'] = msg.pop('message_metadata')

        return {
            "id": self.id, "system_prompt": self.system_prompt,
            "participants": self.participants, "active_branch_id": self.active_branch_id,
            "metadata": self.metadata, "scratchpad": self.scratchpad,
            "messages": messages_copy,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    def load_from_dict(self, data: Dict):
        self._reset_in_memory_state()
        self.id = data.get("id", str(uuid.uuid4()))
        self.system_prompt = data.get("system_prompt")
        self.participants = data.get("participants", {})
        self.active_branch_id = data.get("active_branch_id")
        self.metadata = data.get("metadata", {})
        self.scratchpad = data.get("scratchpad", "")
        
        loaded_messages = data.get("messages", [])
        for msg in loaded_messages:
            if 'created_at' in msg and isinstance(msg['created_at'], str):
                try:
                    msg['created_at'] = datetime.fromisoformat(msg['created_at'])
                except ValueError:
                    msg['created_at'] = datetime.utcnow()
        self.messages = loaded_messages

        self.created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.utcnow()
        self.updated_at = datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else self.created_at
        self._rebuild_in_memory_indexes()

    def _rebuild_in_memory_indexes(self):
        self.message_index = {msg['id']: msg for msg in self.messages}

    @staticmethod
    def migrate(lollms_client: 'LollmsClient', db_manager: DatabaseManager, folder_path: Union[str, Path]):
        folder = Path(folder_path)
        if not folder.is_dir():
            print(f"Error: Path '{folder}' is not a valid directory.")
            return

        print(f"\n--- Starting Migration from '{folder}' ---")
        discussion_files = list(folder.glob("*.json")) + list(folder.glob("*.yaml"))
        session = db_manager.get_session()
        for i, file_path in enumerate(discussion_files):
            print(f"Migrating file {i+1}/{len(discussion_files)}: {file_path.name} ... ", end="")
            try:
                in_memory_discussion = LollmsDiscussion.create_new(lollms_client=lollms_client)
                if file_path.suffix.lower() == ".json":
                    with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f: data = yaml.safe_load(f)
                
                in_memory_discussion.load_from_dict(data)
                discussion_id = in_memory_discussion.id

                existing = session.query(db_manager.DiscussionModel).filter_by(id=discussion_id).first()
                if existing:
                    print("SKIPPED (already exists)")
                    continue

                valid_disc_keys = {c.name for c in db_manager.DiscussionModel.__table__.columns}
                valid_msg_keys = {c.name for c in db_manager.MessageModel.__table__.columns}
                
                discussion_data = {
                    'id': in_memory_discussion.id,
                    'system_prompt': in_memory_discussion.system_prompt,
                    'participants': in_memory_discussion.participants,
                    'active_branch_id': in_memory_discussion.active_branch_id,
                    'discussion_metadata': in_memory_discussion.metadata,
                    'created_at': in_memory_discussion.created_at,
                    'updated_at': in_memory_discussion.updated_at
                }
                project_name = in_memory_discussion.metadata.get('project_name', file_path.stem)
                if 'project_name' in valid_disc_keys:
                    discussion_data['project_name'] = project_name
                
                db_discussion = db_manager.DiscussionModel(**discussion_data)
                session.add(db_discussion)
                
                for msg_data in in_memory_discussion.messages:
                    msg_data['discussion_id'] = db_discussion.id
                    if 'metadata' in msg_data:
                        msg_data['message_metadata'] = msg_data.pop('metadata')
                    filtered_msg_data = {k: v for k, v in msg_data.items() if k in valid_msg_keys}
                    msg_orm = db_manager.MessageModel(**filtered_msg_data)
                    session.add(msg_orm)
                
                print("OK")
            except Exception as e:
                print(f"FAILED. Error: {e}")
                session.rollback()
        session.commit()
        session.close()