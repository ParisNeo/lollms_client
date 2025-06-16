# lollms_discussion.py

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
import uuid
from collections import defaultdict

# It's good practice to forward-declare the type for the client to avoid circular imports.
if False:
    from lollms.client import LollmsClient


@dataclass
class LollmsMessage:
    """
    Represents a single message in a LollmsDiscussion, including its content,
    sender, and relationship within the discussion tree.
    """
    sender: str
    sender_type: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    metadata: str = "{}"
    images: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the message object to a dictionary."""
        return {
            'sender': self.sender,
            'sender_type': self.sender_type,
            'content': self.content,
            'id': self.id,
            'parent_id': self.parent_id,
            'metadata': self.metadata,
            'images': self.images
        }


class LollmsDiscussion:
    """
    Manages a branching conversation tree, including system prompts, participants,
    an internal knowledge scratchpad, and context pruning capabilities.
    """

    def __init__(self, lollmsClient: 'LollmsClient'):
        """
        Initializes a new LollmsDiscussion instance.

        Args:
            lollmsClient: An instance of LollmsClient, required for tokenization.
        """
        self.lollmsClient = lollmsClient
        self.version: int = 3  # Current version of the format with scratchpad support
        self._reset_state()

    def _reset_state(self):
        """Helper to reset all discussion attributes to their defaults."""
        self.messages: List[LollmsMessage] = []
        self.active_branch_id: Optional[str] = None
        self.message_index: Dict[str, LollmsMessage] = {}
        self.children_index: Dict[Optional[str], List[str]] = defaultdict(list)
        self.participants: Dict[str, str] = {}
        self.system_prompt: Optional[str] = None
        self.scratchpad: Optional[str] = None

    # --- Scratchpad Management Methods ---
    def set_scratchpad(self, content: str):
        """Sets or replaces the entire content of the internal scratchpad."""
        self.scratchpad = content

    def update_scratchpad(self, new_content: str, append: bool = True):
        """
        Updates the scratchpad. By default, it appends with a newline separator.

        Args:
            new_content: The new text to add to the scratchpad.
            append: If True, appends to existing content. If False, replaces it.
        """
        if append and self.scratchpad:
            self.scratchpad += f"\n{new_content}"
        else:
            self.scratchpad = new_content

    def get_scratchpad(self) -> Optional[str]:
        """Returns the current content of the scratchpad."""
        return self.scratchpad

    def clear_scratchpad(self):
        """Clears the scratchpad content."""
        self.scratchpad = None

    # --- Configuration Methods ---
    def set_system_prompt(self, prompt: str):
        """Sets the main system prompt for the discussion."""
        self.system_prompt = prompt

    def set_participants(self, participants: Dict[str, str]):
        """
        Defines the participants and their roles ('user' or 'assistant').

        Args:
            participants: A dictionary mapping sender names to roles.
        """
        for name, role in participants.items():
            if role not in ["user", "assistant"]:
                raise ValueError(f"Invalid role '{role}' for participant '{name}'")
        self.participants = participants

    # --- Core Message Tree Methods ---
    def add_message(
        self,
        sender: str,
        sender_type: str,
        content: str,
        metadata: Optional[Dict] = None,
        parent_id: Optional[str] = None,
        images: Optional[List[Dict[str, str]]] = None,
        override_id: Optional[str] = None
    ) -> str:
        """
        Adds a new message to the discussion tree.
        """
        if parent_id is None:
            parent_id = self.active_branch_id
        if parent_id is None:
            parent_id = "main_root"

        message = LollmsMessage(
            sender=sender, sender_type=sender_type, content=content,
            parent_id=parent_id, metadata=str(metadata or {}), images=images or []
        )
        if override_id:
            message.id = override_id

        self.messages.append(message)
        self.message_index[message.id] = message
        self.children_index[parent_id].append(message.id)
        self.active_branch_id = message.id
        return message.id

    def get_branch(self, leaf_id: str) -> List[LollmsMessage]:
        """Gets the full branch of messages from the root to the specified leaf."""
        branch = []
        current_id: Optional[str] = leaf_id
        while current_id and current_id in self.message_index:
            msg = self.message_index[current_id]
            branch.append(msg)
            current_id = msg.parent_id
        return list(reversed(branch))

    def set_active_branch(self, message_id: str):
        """Sets the active message, effectively switching to a different branch."""
        if message_id not in self.message_index:
            raise ValueError(f"Message ID {message_id} not found in discussion.")
        self.active_branch_id = message_id

    # --- Persistence ---
    def save_to_disk(self, file_path: str):
        """Saves the entire discussion state to a YAML file."""
        data = {
            'version': self.version, 'active_branch_id': self.active_branch_id,
            'system_prompt': self.system_prompt, 'participants': self.participants,
            'scratchpad': self.scratchpad, 'messages': [m.to_dict() for m in self.messages]
        }
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, allow_unicode=True, sort_keys=False)

    def load_from_disk(self, file_path: str):
        """Loads a discussion state from a YAML file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        self._reset_state()
        version = data.get("version", 1)
        if version > self.version:
            raise ValueError(f"File version {version} is newer than supported version {self.version}.")

        self.active_branch_id = data.get('active_branch_id')
        self.system_prompt = data.get('system_prompt', None)
        self.participants = data.get('participants', {})
        self.scratchpad = data.get('scratchpad', None)

        for msg_data in data.get('messages', []):
            msg = LollmsMessage(
                sender=msg_data['sender'], sender_type=msg_data.get('sender_type', 'user'),
                content=msg_data['content'], parent_id=msg_data.get('parent_id'),
                id=msg_data.get('id', str(uuid.uuid4())), metadata=msg_data.get('metadata', '{}'),
                images=msg_data.get('images', [])
            )
            self.messages.append(msg)
            self.message_index[msg.id] = msg
            self.children_index[msg.parent_id].append(msg.id)

    # --- Context Management and Formatting ---
    def _get_full_system_prompt(self) -> Optional[str]:
        """Combines the scratchpad and system prompt into a single string for the LLM."""
        full_sys_prompt_parts = []
        if self.scratchpad and self.scratchpad.strip():
            full_sys_prompt_parts.append("--- KNOWLEDGE SCRATCHPAD ---")
            full_sys_prompt_parts.append(self.scratchpad.strip())
            full_sys_prompt_parts.append("--- END SCRATCHPAD ---")
        
        if self.system_prompt and self.system_prompt.strip():
            full_sys_prompt_parts.append(self.system_prompt.strip())
        return "\n\n".join(full_sys_prompt_parts) if full_sys_prompt_parts else None

    def summarize_and_prune(self, max_tokens: int, preserve_last_n: int = 4, branch_tip_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Checks context size and, if exceeded, summarizes the oldest messages
        into the scratchpad and prunes them to free up token space.
        """
        if branch_tip_id is None: branch_tip_id = self.active_branch_id
        if not branch_tip_id: return {"pruned": False, "reason": "No active branch."}

        full_prompt_text = self.export("lollms_text", branch_tip_id)
        current_tokens = len(self.lollmsClient.binding.tokenize(full_prompt_text))
        if current_tokens <= max_tokens: return {"pruned": False, "reason": "Token count within limit."}

        branch = self.get_branch(branch_tip_id)
        if len(branch) <= preserve_last_n: return {"pruned": False, "reason": "Not enough messages to prune."}

        messages_to_prune = branch[:-preserve_last_n]
        messages_to_keep = branch[-preserve_last_n:]
        text_to_summarize = "\n\n".join([f"{self.participants.get(m.sender, 'user').capitalize()}: {m.content}" for m in messages_to_prune])
        
        summary_prompt = (
            "You are a summarization expert. Read the following conversation excerpt and create a "
            "concise, factual summary of all key information, decisions, and outcomes. This summary "
            "will be placed in a knowledge scratchpad for future reference. Omit conversational filler.\n\n"
            f"CONVERSATION EXCERPT:\n---\n{text_to_summarize}\n---\n\nCONCISE SUMMARY:"
        )
        try:
            summary = self.lollmsClient.generate_text(summary_prompt, max_new_tokens=300, temperature=0.1)
        except Exception as e:
            return {"pruned": False, "reason": f"Failed to generate summary: {e}"}

        summary_block = f"--- Summary of earlier conversation (pruned on {uuid.uuid4().hex[:8]}) ---\n{summary.strip()}"
        self.update_scratchpad(summary_block, append=True)

        ids_to_prune = {msg.id for msg in messages_to_prune}
        new_root_of_branch = messages_to_keep[0]
        original_parent_id = messages_to_prune[0].parent_id

        self.message_index[new_root_of_branch.id].parent_id = original_parent_id
        if original_parent_id in self.children_index:
            self.children_index[original_parent_id] = [mid for mid in self.children_index[original_parent_id] if mid != messages_to_prune[0].id]
            self.children_index[original_parent_id].append(new_root_of_branch.id)

        for msg_id in ids_to_prune:
            self.message_index.pop(msg_id, None)
            self.children_index.pop(msg_id, None)
        self.messages = [m for m in self.messages if m.id not in ids_to_prune]

        new_prompt_text = self.export("lollms_text", branch_tip_id)
        new_tokens = len(self.lollmsClient.binding.tokenize(new_prompt_text))
        return {"pruned": True, "tokens_saved": current_tokens - new_tokens, "summary_added": True}

    def format_discussion(self, max_allowed_tokens: int, splitter_text: str = "!@>", branch_tip_id: Optional[str] = None) -> str:
        """
        Formats the discussion into a single string for instruct models,
        truncating from the start to respect the token limit.

        Args:
            max_allowed_tokens: The maximum token limit for the final prompt.
            splitter_text: The separator token to use (e.g., '!@>').
            branch_tip_id: The ID of the branch to format. Defaults to active.

        Returns:
            A single, truncated prompt string.
        """
        if branch_tip_id is None:
            branch_tip_id = self.active_branch_id
        
        branch_msgs = self.get_branch(branch_tip_id) if branch_tip_id else []
        full_system_prompt = self._get_full_system_prompt()
        
        prompt_parts = []
        current_tokens = 0
        
        # Start with the system prompt if defined
        if full_system_prompt:
            sys_msg_text = f"{splitter_text}system:\n{full_system_prompt}\n"
            sys_tokens = len(self.lollmsClient.binding.tokenize(sys_msg_text))
            if sys_tokens <= max_allowed_tokens:
                prompt_parts.append(sys_msg_text)
                current_tokens += sys_tokens
        
        # Iterate from newest to oldest to fill the remaining context
        for msg in reversed(branch_msgs):
            sender_str = msg.sender.replace(':', '').replace(splitter_text, '')
            content = msg.content.strip()
            if msg.images:
                content += f"\n({len(msg.images)} image(s) attached)"

            msg_text = f"{splitter_text}{sender_str}:\n{content}\n"
            msg_tokens = len(self.lollmsClient.binding.tokenize(msg_text))

            if current_tokens + msg_tokens > max_allowed_tokens:
                break # Stop if adding the next message exceeds the limit

            prompt_parts.insert(1 if full_system_prompt else 0, msg_text) # Prepend after system prompt
            current_tokens += msg_tokens
            
        return "".join(prompt_parts).strip()


    def export(self, format_type: str, branch_tip_id: Optional[str] = None) -> Union[List[Dict], str]:
        """
        Exports the full, untruncated discussion history in a specific format.
        """
        if branch_tip_id is None: branch_tip_id = self.active_branch_id
        if branch_tip_id is None and not self._get_full_system_prompt(): return "" if format_type in ["lollms_text", "openai_completion"] else []

        branch = self.get_branch(branch_tip_id) if branch_tip_id else []
        full_system_prompt = self._get_full_system_prompt()

        if format_type == "openai_chat":
            messages = []
            if full_system_prompt: messages.append({"role": "system", "content": full_system_prompt})
            def openai_image_block(image: Dict[str, str]) -> Dict:
                image_url = image['data'] if image['type'] == 'url' else f"data:image/jpeg;base64,{image['data']}"
                return {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}
            for msg in branch:
                role = self.participants.get(msg.sender, "user")
                if msg.images:
                    content_parts = [{"type": "text", "text": msg.content.strip()}] if msg.content.strip() else []
                    content_parts.extend(openai_image_block(img) for img in msg.images)
                    messages.append({"role": role, "content": content_parts})
                else: messages.append({"role": role, "content": msg.content.strip()})
            return messages

        elif format_type == "ollama_chat":
            messages = []
            if full_system_prompt: messages.append({"role": "system", "content": full_system_prompt})
            for msg in branch:
                role = self.participants.get(msg.sender, "user")
                message_dict = {"role": role, "content": msg.content.strip()}
                ollama_images = [img['data'] for img in msg.images if img['type'] == 'base64']
                if ollama_images: message_dict["images"] = ollama_images
                messages.append(message_dict)
            return messages

        elif format_type == "lollms_text":
            full_prompt_parts = []
            if full_system_prompt: full_prompt_parts.append(f"!@>system:\n{full_system_prompt}")
            for msg in branch:
                sender_str = msg.sender.replace(':', '').replace('!@>', '')
                content = msg.content.strip()
                if msg.images: content += f"\n({len(msg.images)} image(s) attached)"
                full_prompt_parts.append(f"!@>{sender_str}:\n{content}")
            return "\n".join(full_prompt_parts)

        elif format_type == "openai_completion":
            full_prompt_parts = []
            if full_system_prompt: full_prompt_parts.append(f"System:\n{full_system_prompt}")
            for msg in branch:
                role_label = self.participants.get(msg.sender, "user").capitalize()
                content = msg.content.strip()
                if msg.images: content += f"\n({len(msg.images)} image(s) attached)"
                full_prompt_parts.append(f"{role_label}:\n{content}")
            return "\n\n".join(full_prompt_parts)

        else: raise ValueError(f"Unsupported export format_type: {format_type}")


if __name__ == "__main__":
    class MockBinding:
        def tokenize(self, text: str) -> List[int]: return text.split()
    class MockLollmsClient:
        def __init__(self): self.binding = MockBinding()
        def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str: return "This is a generated summary."

    print("--- Initializing Mock Client and Discussion ---")
    mock_client = MockLollmsClient()
    discussion = LollmsDiscussion(mock_client)
    discussion.set_participants({"User": "user", "Project Lead": "assistant"})
    discussion.set_system_prompt("This is a formal discussion about Project Phoenix.")
    discussion.set_scratchpad("Initial State: Project Phoenix is in the planning phase.")

    print("\n--- Creating a long discussion history ---")
    parent_id = None
    long_text = "extra text to increase token count"
    for i in range(10):
        user_msg = f"Message #{i*2+1}: Update on task {i+1}? {long_text}"
        user_id = discussion.add_message("User", "user", user_msg, parent_id=parent_id)
        assistant_msg = f"Message #{i*2+2}: Task {i+1} status is blocked. {long_text}"
        assistant_id = discussion.add_message("Project Lead", "assistant", assistant_msg, parent_id=user_id)
        parent_id = assistant_id
    
    initial_tokens = len(mock_client.binding.tokenize(discussion.export("lollms_text")))
    print(f"Initial message count: {len(discussion.messages)}, Initial tokens: {initial_tokens}")

    print("\n--- Testing Pruning ---")
    prune_result = discussion.summarize_and_prune(max_tokens=200, preserve_last_n=4)
    if prune_result.get("pruned"):
        print("✅ Pruning was successful!")
        assert "Summary" in discussion.get_scratchpad()
    else: print(f"❌ Pruning failed: {prune_result.get('reason')}")

    print("\n--- Testing format_discussion (Instruct Model Format) ---")
    truncated_prompt = discussion.format_discussion(max_allowed_tokens=80)
    truncated_tokens = len(mock_client.binding.tokenize(truncated_prompt))
    print(f"Truncated prompt tokens: {truncated_tokens}")
    print("Truncated Prompt:\n" + "="*20 + f"\n{truncated_prompt}\n" + "="*20)

    # Verification
    assert truncated_tokens <= 80
    # Check that it contains the newest message that fits
    assert "Message #19" in truncated_prompt or "Message #20" in truncated_prompt 
    print("✅ format_discussion correctly truncated the prompt.")