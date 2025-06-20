import yaml
import json
import base64
import os
import uuid
import shutil
import re
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Union, Any, Type, Callable
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import (create_engine, Column, String, Text, Integer, DateTime,
                        ForeignKey, JSON, Boolean, LargeBinary, Index, Float)
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base, declared_attr
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm.exc import NoResultFound

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# Type hint placeholders for classes defined externally
if False: 
    from lollms_client import LollmsClient
    from lollms_client.lollms_types import MSG_TYPE
    from lollms_personality import LollmsPersonality

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

    def process_bind_param(self, value: Optional[str], dialect) -> Optional[bytes]:
        if value is None:
            return None
        return self.fernet.encrypt(value.encode('utf-8'))

    def process_result_value(self, value: Optional[bytes], dialect) -> Optional[str]:
        if value is None:
            return None
        try:
            return self.fernet.decrypt(value).decode('utf-8')
        except InvalidToken:
            return "<DECRYPTION_FAILED: Invalid Key or Corrupt Data>"

def create_dynamic_models(discussion_mixin: Optional[Type] = None, message_mixin: Optional[Type] = None, encryption_key: Optional[str] = None):
    """Factory to dynamically create SQLAlchemy ORM models with custom mixins."""
    Base = declarative_base()
    EncryptedText = EncryptedString(encryption_key) if encryption_key else Text

    class DiscussionBase:
        __abstract__ = True
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        system_prompt = Column(EncryptedText, nullable=True)
        participants = Column(JSON, nullable=True, default=dict)
        active_branch_id = Column(String, nullable=True)
        discussion_metadata = Column(JSON, nullable=True, default=dict)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        @declared_attr
        def messages(cls):
            return relationship("Message", back_populates="discussion", cascade="all, delete-orphan", lazy="joined")

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
    """Manages database connection, session, and table creation."""
    def __init__(self, db_path: str, discussion_mixin: Optional[Type] = None, message_mixin: Optional[Type] = None, encryption_key: Optional[str] = None):
        if not db_path:
            raise ValueError("Database path cannot be empty.")
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
        with self.get_session() as session:
            discussions = session.query(self.DiscussionModel).all()
            return [{c.name: getattr(disc, c.name) for c in disc.__table__.columns} for disc in discussions]

    def get_discussion(self, lollms_client: 'LollmsClient', discussion_id: str, **kwargs) -> Optional['LollmsDiscussion']:
        with self.get_session() as session:
            try:
                db_disc = session.query(self.DiscussionModel).filter_by(id=discussion_id).one()
                session.expunge(db_disc)
                return LollmsDiscussion(lollmsClient=lollms_client, db_manager=self, db_discussion_obj=db_disc, **kwargs)
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

class LollmsMessage:
    """A wrapper for a message ORM object, providing direct attribute access."""
    def __init__(self, discussion: 'LollmsDiscussion', db_message: Any):
        object.__setattr__(self, '_discussion', discussion)
        object.__setattr__(self, '_db_message', db_message)

    def __getattr__(self, name: str) -> Any:
        if name == 'metadata':
            return getattr(self._db_message, 'message_metadata', None)
        return getattr(self._db_message, name)

    def __setattr__(self, name: str, value: Any):
        if name == 'metadata':
            setattr(self._db_message, 'message_metadata', value)
        else:
            setattr(self._db_message, name, value)
        self._discussion.touch()

    def __repr__(self) -> str:
        return f"<LollmsMessage id={self.id} sender='{self.sender}'>"

class LollmsDiscussion:
    """Represents and manages a single discussion, acting as a high-level interface."""
    def __init__(self, lollmsClient: 'LollmsClient', db_manager: Optional[LollmsDataManager] = None, 
                 discussion_id: Optional[str] = None, db_discussion_obj: Optional[Any] = None,
                 autosave: bool = False, max_context_size: Optional[int] = None):
        
        object.__setattr__(self, 'lollmsClient', lollmsClient)
        object.__setattr__(self, 'db_manager', db_manager)
        object.__setattr__(self, 'autosave', autosave)
        object.__setattr__(self, 'max_context_size', max_context_size)
        object.__setattr__(self, 'scratchpad', "")
        object.__setattr__(self, 'show_thoughts', False)
        object.__setattr__(self, 'include_thoughts_in_context', False)
        object.__setattr__(self, 'thought_placeholder', "<thought process hidden>")
        
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
        self._rebuild_message_index()
    
    @property
    def remaining_tokens(self) -> Optional[int]:
        """Calculates the remaining tokens available in the context window."""
        binding = self.lollmsClient.binding
        if not binding or not hasattr(binding, 'ctx_size') or not binding.ctx_size:
            return None
        max_ctx = binding.ctx_size
        current_prompt = self.format_discussion(max_ctx)
        current_tokens = self.lollmsClient.count_tokens(current_prompt)
        return max_ctx - current_tokens

    @classmethod
    def create_new(cls, lollms_client: 'LollmsClient', db_manager: Optional[LollmsDataManager] = None, **kwargs) -> 'LollmsDiscussion':
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

    def __getattr__(self, name: str) -> Any:
        if name == 'metadata':
            return getattr(self._db_discussion, 'discussion_metadata', None)
        if name == 'messages':
            return [LollmsMessage(self, msg) for msg in self._db_discussion.messages]
        return getattr(self._db_discussion, name)

    def __setattr__(self, name: str, value: Any):
        internal_attrs = [
            'lollmsClient','db_manager','autosave','max_context_size','scratchpad',
            'show_thoughts', 'include_thoughts_in_context', 'thought_placeholder',
            '_session','_db_discussion','_message_index','_messages_to_delete_from_db', '_is_db_backed'
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
        proxy = SimpleNamespace()
        proxy.id, proxy.system_prompt, proxy.participants = id or str(uuid.uuid4()), None, {}
        proxy.active_branch_id, proxy.discussion_metadata = None, {}
        proxy.created_at, proxy.updated_at = datetime.utcnow(), datetime.utcnow()
        proxy.messages = []
        object.__setattr__(self, '_db_discussion', proxy)
    
    def _rebuild_message_index(self):
        if self._is_db_backed and self._session.is_active and self._db_discussion in self._session:
            self._session.refresh(self._db_discussion, ['messages'])
        self._message_index = {msg.id: msg for msg in self._db_discussion.messages}

    def touch(self):
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
            self._session.rollback()
            raise e

    def close(self):
        if self._session:
            self.commit()
            self._session.close()

    def add_message(self, **kwargs) -> LollmsMessage:
        msg_id, parent_id = kwargs.get('id', str(uuid.uuid4())), kwargs.get('parent_id', self.active_branch_id)
        message_data = {'id': msg_id, 'parent_id': parent_id, 'discussion_id': self.id, 'created_at': datetime.utcnow(), **kwargs}
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
        self._message_index[msg_id], self.active_branch_id = new_msg_orm, msg_id
        self.touch()
        return LollmsMessage(self, new_msg_orm)
        
    def get_branch(self, leaf_id: Optional[str]) -> List[LollmsMessage]:
        if not leaf_id:
            return []
        branch_orms, current_id = [], leaf_id
        while current_id and current_id in self._message_index:
            msg_orm = self._message_index[current_id]
            branch_orms.append(msg_orm)
            current_id = msg_orm.parent_id
        return [LollmsMessage(self, orm) for orm in reversed(branch_orms)]

    def chat(self, user_message: str, personality: Optional['LollmsPersonality'] = None, **kwargs) -> LollmsMessage:
        if self.max_context_size is not None:
            self.summarize_and_prune(self.max_context_size)

        if user_message:
            self.add_message(sender="user", sender_type="user", content=user_message)

        rag_context = None
        original_system_prompt = self.system_prompt
        if personality:
            self.system_prompt = personality.system_prompt
            if user_message:
                rag_context = personality.get_rag_context(user_message)
        
        if rag_context:
            self.system_prompt = f"{original_system_prompt or ''}\n\n--- Relevant Information ---\n{rag_context}\n---"

        from lollms_client.lollms_types import MSG_TYPE
        is_streaming = "streaming_callback" in kwargs and kwargs.get("streaming_callback") is not None
        
        final_raw_response = ""
        start_time = datetime.now()

        if personality and personality.script_module and hasattr(personality.script_module, 'run'):
            try:
                print(f"[{personality.name}] Running custom script...")
                final_raw_response = personality.script_module.run(self, kwargs.get("streaming_callback"))
            except Exception as e:
                print(f"[{personality.name}] Error in custom script: {e}")
                final_raw_response = f"Error executing personality script: {e}"
        else:
            raw_response_accumulator = []
            if is_streaming:
                full_response_parts, token_buffer, in_thought_block = [], "", False
                original_callback = kwargs.get("streaming_callback")
                def accumulating_callback(token: str, msg_type: MSG_TYPE = MSG_TYPE.MSG_TYPE_CHUNK):
                    nonlocal token_buffer, in_thought_block
                    raw_response_accumulator.append(token)
                    continue_streaming = True
                    if token: token_buffer += token
                    while True:
                        if in_thought_block:
                            end_tag_pos = token_buffer.find("</think>")
                            if end_tag_pos != -1:
                                thought_chunk = token_buffer[:end_tag_pos]
                                if self.show_thoughts and original_callback and thought_chunk:
                                    if not original_callback(thought_chunk, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK): continue_streaming = False
                                in_thought_block, token_buffer = False, token_buffer[end_tag_pos + len("</think>"):]
                            else:
                                if self.show_thoughts and original_callback and token_buffer:
                                    if not original_callback(token_buffer, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK): continue_streaming = False
                                token_buffer = ""; break
                        else:
                            start_tag_pos = token_buffer.find("<think>")
                            if start_tag_pos != -1:
                                response_chunk = token_buffer[:start_tag_pos]
                                if response_chunk:
                                    full_response_parts.append(response_chunk)
                                    if original_callback:
                                        if not original_callback(response_chunk, MSG_TYPE.MSG_TYPE_CHUNK): continue_streaming = False
                                in_thought_block, token_buffer = True, token_buffer[start_tag_pos + len("<think>"):]
                            else:
                                if token_buffer:
                                    full_response_parts.append(token_buffer)
                                    if original_callback:
                                        if not original_callback(token_buffer, MSG_TYPE.MSG_TYPE_CHUNK): continue_streaming = False
                                token_buffer = ""; break
                    return continue_streaming
                kwargs["streaming_callback"], kwargs["stream"] = accumulating_callback, True
                self.lollmsClient.chat(self, **kwargs)
                final_raw_response = "".join(raw_response_accumulator)
            else:
                kwargs["stream"] = False
                final_raw_response = self.lollmsClient.chat(self, **kwargs) or ""

        end_time = datetime.now()
        if rag_context:
            self.system_prompt = original_system_prompt
        
        duration = (end_time - start_time).total_seconds()
        thoughts_match = re.search(r"<think>(.*?)</think>", final_raw_response, re.DOTALL)
        thoughts_text = thoughts_match.group(1).strip() if thoughts_match else None
        final_content = self.lollmsClient.remove_thinking_blocks(final_raw_response)
        token_count = self.lollmsClient.count_tokens(final_content)
        tok_per_sec = (token_count / duration) if duration > 0 else 0
        
        ai_message_obj = self.add_message(
            sender="assistant", sender_type="assistant", content=final_content,
            raw_content=final_raw_response, thoughts=thoughts_text, tokens=token_count,
            binding_name=self.lollmsClient.binding.binding_name, model_name=self.lollmsClient.binding.model_name,
            generation_speed=tok_per_sec
        )

        if self._is_db_backed and not self.autosave:
            self.commit()
        return ai_message_obj

    def process_and_summarize(self, large_text: str, user_prompt: str, chunk_size: int = 4096, **kwargs) -> LollmsMessage:
        user_msg = self.add_message(sender="user", sender_type="user", content=user_prompt)
        chunks = [large_text[i:i + chunk_size] for i in range(0, len(large_text), chunk_size)]
        current_summary, total_chunks = "", len(chunks)
        for i, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {i+1}/{total_chunks}...")
            if i == 0:
                prompt = f"""The user wants to know: "{user_prompt}"\nHere is the first part of the document (chunk 1 of {total_chunks}). \nRead it and create a detailed summary of all information relevant to the user's prompt.\n\nDOCUMENT CHUNK:\n---\n{chunk}\n---\nSUMMARY:"""
            else:
                prompt = f"""The user wants to know: "{user_prompt}"\nYou are processing a large document sequentially. Here is the summary of the previous chunks and the content of the next chunk ({i+1} of {total_chunks}).\nUpdate your summary by integrating new relevant information from the new chunk. Do not repeat information you already have. Output ONLY the new, updated, complete summary.\n\nPREVIOUS SUMMARY:\n---\n{current_summary}\n---\n\nNEW DOCUMENT CHUNK:\n---\n{chunk}\n---\nUPDATED SUMMARY:"""
            current_summary = self.lollmsClient.generate_text(prompt, **kwargs).strip()
        final_prompt = f"""Based on the following comprehensive summary of a document, provide a final answer to the user's original prompt.\nUser's prompt: "{user_prompt}"\n\nCOMPREHENSIVE SUMMARY:\n---\n{current_summary}\n---\nFINAL ANSWER:"""
        final_answer = self.lollmsClient.generate_text(final_prompt, **kwargs).strip()
        ai_message_obj = self.add_message(
            sender="assistant", sender_type="assistant", content=final_answer,
            scratchpad=current_summary, parent_id=user_msg.id
        )
        if self._is_db_backed and not self.autosave:
            self.commit()
        return ai_message_obj

    def regenerate_branch(self, **kwargs) -> LollmsMessage:
        if not self.active_branch_id or self.active_branch_id not in self._message_index:
            raise ValueError("No active message to regenerate from.")
        last_message_orm = self._message_index[self.active_branch_id]
        if last_message_orm.sender_type != 'assistant':
            raise ValueError("Can only regenerate from an assistant's message.")
        parent_id, last_message_id = last_message_orm.parent_id, last_message_orm.id
        self._db_discussion.messages.remove(last_message_orm)
        del self._message_index[last_message_id]
        if self._is_db_backed:
            self._messages_to_delete_from_db.add(last_message_id)
        self.active_branch_id = parent_id
        self.touch()
        return self.chat("", **kwargs)

    def delete_branch(self, message_id: str):
        if not self._is_db_backed:
            raise NotImplementedError("Branch deletion is only supported for database-backed discussions.")
        if message_id not in self._message_index:
            raise ValueError("Message not found.")
        msg_to_delete = self._session.query(self.db_manager.MessageModel).filter_by(id=message_id).first()
        if msg_to_delete:
            self.active_branch_id = msg_to_delete.parent_id
            self._session.delete(msg_to_delete)
            self.commit()

    def switch_to_branch(self, message_id: str):
        if message_id not in self._message_index:
            raise ValueError(f"Message ID '{message_id}' not found in the current discussion.")
        self.active_branch_id = message_id
        self.touch()

    def format_discussion(self, max_allowed_tokens: int, branch_tip_id: Optional[str] = None) -> str:
        return self.export("lollms_text", branch_tip_id, max_allowed_tokens)

    def _get_full_system_prompt(self) -> Optional[str]:
        parts = []
        if self.scratchpad:
            parts.extend(["--- KNOWLEDGE SCRATCHPAD ---", self.scratchpad.strip(), "--- END SCRATCHPAD ---"])
        if self.system_prompt and self.system_prompt.strip():
            parts.append(self.system_prompt.strip())
        return "\n\n".join(parts) if parts else None

    def export(self, format_type: str, branch_tip_id: Optional[str] = None, max_allowed_tokens: Optional[int] = None) -> Union[List[Dict], str]:
        branch_tip_id = branch_tip_id or self.active_branch_id
        if not branch_tip_id and format_type in ["lollms_text", "openai_chat", "ollama_chat"]:
            return "" if format_type == "lollms_text" else []
        branch, full_system_prompt, participants = self.get_branch(branch_tip_id), self._get_full_system_prompt(), self.participants or {}

        def get_full_content(msg: LollmsMessage) -> str:
            content_to_use = msg.content
            if self.include_thoughts_in_context and msg.sender_type == 'assistant' and msg.raw_content:
                if self.thought_placeholder:
                    content_to_use = re.sub(r"<think>.*?</think>", f"<think>{self.thought_placeholder}</think>", msg.raw_content, flags=re.DOTALL)
                else:
                    content_to_use = msg.raw_content
            
            parts = [f"--- Internal Scratchpad ---\n{msg.scratchpad.strip()}\n---"] if msg.scratchpad and msg.scratchpad.strip() else []
            parts.append(content_to_use.strip())
            return "\n".join(parts)

        if format_type == "lollms_text":
            prompt_parts, current_tokens = [], 0
            if full_system_prompt:
                sys_msg_text = f"!@>system:\n{full_system_prompt}\n"
                sys_tokens = self.lollmsClient.count_tokens(sys_msg_text)
                if max_allowed_tokens is None or sys_tokens <= max_allowed_tokens:
                    prompt_parts.append(sys_msg_text)
                    current_tokens += sys_tokens
            for msg in reversed(branch):
                sender_str = msg.sender.replace(':', '').replace('!@>', '')
                content = get_full_content(msg)
                if msg.images:
                    content += f"\n({len(msg.images)} image(s) attached)"
                msg_text = f"!@>{sender_str}:\n{content}\n"
                msg_tokens = self.lollmsClient.count_tokens(msg_text)
                if max_allowed_tokens is not None and current_tokens + msg_tokens > max_allowed_tokens:
                    break
                prompt_parts.insert(1 if full_system_prompt else 0, msg_text)
                current_tokens += msg_tokens
            return "".join(prompt_parts).strip()
        
        messages = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        for msg in branch:
            role, content, images = participants.get(msg.sender, "user"), get_full_content(msg), msg.images or []
            if format_type == "openai_chat":
                if images:
                    content_parts = [{"type": "text", "text": content}] if content else []
                    for img in images:
                        content_parts.append({"type": "image_url", "image_url": {"url": img['data'] if img['type'] == 'url' else f"data:image/jpeg;base64,{img['data']}", "detail": "auto"}})
                    messages.append({"role": role, "content": content_parts})
                else:
                    messages.append({"role": role, "content": content})
            elif format_type == "ollama_chat":
                message_dict = {"role": role, "content": content}
                base64_images = [img['data'] for img in images if img['type'] == 'base64']
                if base64_images:
                    message_dict["images"] = base64_images
                messages.append(message_dict)
            else:
                raise ValueError(f"Unsupported export format_type: {format_type}")
        return messages

    def summarize_and_prune(self, max_tokens: int, preserve_last_n: int = 4):
        branch_tip_id = self.active_branch_id
        if not branch_tip_id:
            return
        current_tokens = self.lollmsClient.count_tokens(self.format_discussion(999999, branch_tip_id))
        if current_tokens <= max_tokens:
            return
        branch = self.get_branch(branch_tip_id)
        if len(branch) <= preserve_last_n:
            return
        messages_to_prune = branch[:-preserve_last_n]
        text_to_summarize = "\n\n".join([f"{m.sender}: {m.content}" for m in messages_to_prune])
        summary_prompt = f"Concisely summarize this conversation excerpt:\n---\n{text_to_summarize}\n---\nSUMMARY:"
        try:
            summary = self.lollmsClient.generate_text(summary_prompt, n_predict=300, temperature=0.1)
        except Exception as e:
            print(f"\n[WARNING] Pruning failed, couldn't generate summary: {e}")
            return
        self.scratchpad = f"{self.scratchpad}\n\n--- Summary of earlier conversation ---\n{summary.strip()}".strip()
        pruned_ids = {msg.id for msg in messages_to_prune}
        if self._is_db_backed:
            self._messages_to_delete_from_db.update(pruned_ids)
            self._db_discussion.messages = [m for m in self._db_discussion.messages if m.id not in pruned_ids]
        else:
            self._db_discussion.messages = [m for m in self._db_discussion.messages if m.id not in pruned_ids]
        self._rebuild_message_index()
        self.touch()
        print(f"\n[INFO] Discussion auto-pruned. {len(messages_to_prune)} messages summarized.")

    def to_dict(self):
        return {
            "id": self.id, "system_prompt": self.system_prompt, "participants": self.participants,
            "active_branch_id": self.active_branch_id, "metadata": self.metadata, "scratchpad": self.scratchpad,
            "messages": [{ 'id': m.id, 'parent_id': m.parent_id, 'discussion_id': m.discussion_id, 'sender': m.sender,
                           'sender_type': m.sender_type, 'content': m.content, 'scratchpad': m.scratchpad, 'images': m.images,
                           'created_at': m.created_at.isoformat(), 'metadata': m.metadata } for m in self.messages],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    def load_from_dict(self, data: Dict):
        self._create_in_memory_proxy(id=data.get("id"))
        self.system_prompt, self.participants = data.get("system_prompt"), data.get("participants", {})
        self.active_branch_id, self.metadata = data.get("active_branch_id"), data.get("metadata", {})
        self.scratchpad = data.get("scratchpad", "")
        for msg_data in data.get("messages", []):
            if 'created_at' in msg_data and isinstance(msg_data['created_at'], str):
                try:
                    msg_data['created_at'] = datetime.fromisoformat(msg_data['created_at'])
                except ValueError:
                    msg_data['created_at'] = datetime.utcnow()
            self.add_message(**msg_data)
        self.created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.utcnow()
        self.updated_at = datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else self.created_at

    @staticmethod
    def migrate(lollms_client: 'LollmsClient', db_manager: LollmsDataManager, folder_path: Union[str, Path]):
        folder = Path(folder_path)
        if not folder.is_dir():
            print(f"Error: Path '{folder}' is not a valid directory.")
            return
        print(f"\n--- Starting Migration from '{folder}' ---")
        files = list(folder.glob("*.json")) + list(folder.glob("*.yaml"))
        with db_manager.get_session() as session:
            valid_disc_keys = {c.name for c in db_manager.DiscussionModel.__table__.columns}
            valid_msg_keys = {c.name for c in db_manager.MessageModel.__table__.columns}
            for i, file_path in enumerate(files):
                print(f"Migrating file {i+1}/{len(files)}: {file_path.name} ... ", end="")
                try:
                    data = yaml.safe_load(file_path.read_text(encoding='utf-8'))
                    discussion_id = data.get("id", str(uuid.uuid4()))
                    if session.query(db_manager.DiscussionModel).filter_by(id=discussion_id).first():
                        print("SKIPPED (already exists)")
                        continue
                    discussion_data = data.copy()
                    if 'metadata' in discussion_data:
                        discussion_data['discussion_metadata'] = discussion_data.pop('metadata')
                    for key in ['created_at', 'updated_at']:
                        if key in discussion_data and isinstance(discussion_data[key], str):
                            discussion_data[key] = datetime.fromisoformat(discussion_data[key])
                    db_discussion = db_manager.DiscussionModel(**{k: v for k, v in discussion_data.items() if k in valid_disc_keys})
                    session.add(db_discussion)
                    for msg_data in data.get("messages", []):
                        msg_data['discussion_id'] = db_discussion.id
                        if 'metadata' in msg_data:
                            msg_data['message_metadata'] = msg_data.pop('metadata')
                        if 'created_at' in msg_data and isinstance(msg_data['created_at'], str):
                            msg_data['created_at'] = datetime.fromisoformat(msg_data['created_at'])
                        msg_orm = db_manager.MessageModel(**{k: v for k, v in msg_data.items() if k in valid_msg_keys})
                        session.add(msg_orm)
                    session.flush()
                    print("OK")
                except Exception as e:
                    print(f"FAILED. Error: {e}")
                    session.rollback()
                    continue
            session.commit()
        print("--- Migration Finished ---")