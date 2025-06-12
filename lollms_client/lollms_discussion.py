import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import uuid
import os
from collections import defaultdict

# LollmsMessage Class with parent_id support
@dataclass
class LollmsMessage:
    sender: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    metadata: str = "{}"
    images: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self):
        return {
            'sender': self.sender,
            'content': self.content,
            'id': self.id,
            'parent_id': self.parent_id,
            'metadata': self.metadata,
            'images': self.images
        }
 


# Enhanced LollmsDiscussion Class with branching support
class LollmsDiscussion:
    def __init__(self, lollmsClient: 'LollmsClient'):
        self.messages: List[LollmsMessage] = []
        self.lollmsClient = lollmsClient
        self.active_branch_id: Optional[str] = None
        self.message_index: Dict[str, LollmsMessage] = {}
        self.children_index: Dict[Optional[str], List[str]] = defaultdict(list)
        self.version: int = 2  # Current version of the format
        self.participants: Dict[str, str] = {}  # name -> type ("user" or "assistant")
        self.system_prompt: Optional[str] = None

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def set_participants(self, participants: Dict[str, str]):
        for name, role in participants.items():
            if role not in ["user", "assistant"]:
                raise ValueError(f"Invalid role '{role}' for participant '{name}'")
        self.participants = participants

    def add_message(
        self,
        sender: str,
        content: str,
        metadata: Dict = {},
        parent_id: Optional[str] = None,
        images: Optional[List[Dict[str, str]]] = None,
        override_id: Optional[str] = None
    ) -> str:
        if parent_id is None:
            parent_id = self.active_branch_id
        if parent_id is None:
            parent_id = "main"

        message = LollmsMessage(
            sender=sender,
            content=content,
            parent_id=parent_id,
            metadata=str(metadata),
            images=images or []
        )
        if override_id:
            message.id = override_id

        self.messages.append(message)
        self.message_index[message.id] = message
        self.children_index[parent_id].append(message.id)

        self.active_branch_id = message.id
        return message.id


    def get_branch(self, leaf_id: str) -> List[LollmsMessage]:
        """Get full branch from root to specified leaf"""
        branch = []
        current_id = leaf_id
        
        while current_id in self.message_index:
            msg = self.message_index[current_id]
            branch.append(msg)
            current_id = msg.parent_id
        
        # Return from root to leaf
        return list(reversed(branch))

    def set_active_branch(self, message_id: str):
        if message_id not in self.message_index:
            raise ValueError(f"Message ID {message_id} not found")
        self.active_branch_id = message_id

    def remove_message(self, message_id: str):
        if message_id not in self.message_index:
            return

        msg = self.message_index[message_id]
        parent_id = msg.parent_id
        
        # Reassign children to parent
        for child_id in self.children_index[message_id]:
            child = self.message_index[child_id]
            child.parent_id = parent_id
            self.children_index[parent_id].append(child_id)
        
        # Clean up indexes
        del self.message_index[message_id]
        del self.children_index[message_id]
        
        # Remove from parent's children list
        if parent_id in self.children_index and message_id in self.children_index[parent_id]:
            self.children_index[parent_id].remove(message_id)
        
        # Remove from main messages list
        self.messages = [m for m in self.messages if m.id != message_id]
        
        # Update active branch if needed
        if self.active_branch_id == message_id:
            self.active_branch_id = parent_id

    def save_to_disk(self, file_path: str):
        data = {
            'version': self.version,
            'active_branch_id': self.active_branch_id,
            'system_prompt': self.system_prompt,
            'participants': self.participants,
            'messages': [m.to_dict() for m in self.messages]
        }
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, allow_unicode=True)


    def load_from_disk(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # Reset
        self.messages = []
        self.message_index = {}
        self.children_index = defaultdict(list)

        if isinstance(data, list):
            # Legacy v1 format
            prev_id = None
            for msg_data in data:
                msg = LollmsMessage(
                    sender=msg_data['sender'],
                    content=msg_data['content'],
                    parent_id=prev_id,
                    id=msg_data.get('id', str(uuid.uuid4())),
                    metadata=msg_data.get('metadata', '{}')
                )
                self.messages.append(msg)
                self.message_index[msg.id] = msg
                self.children_index[prev_id].append(msg.id)
                prev_id = msg.id
            self.active_branch_id = prev_id if self.messages else None
            self.system_prompt = None
            self.participants = {}
            self.save_to_disk(file_path)  # Upgrade
            return

        # v2 format
        version = data.get("version", 1)
        if version != self.version:
            raise ValueError(f"Unsupported version: {version}")

        self.active_branch_id = data.get('active_branch_id')
        self.system_prompt = data.get('system_prompt', None)
        self.participants = data.get('participants', {})

        for msg_data in data.get('messages', []):
            # FIXED: Added `images=msg_data.get('images', [])` to correctly load images from the file.
            msg = LollmsMessage(
                sender=msg_data['sender'],
                content=msg_data['content'],
                parent_id=msg_data.get('parent_id'),
                id=msg_data.get('id'),
                metadata=msg_data.get('metadata', '{}'),
                images=msg_data.get('images', []) 
            )
            self.messages.append(msg)
            self.message_index[msg.id] = msg
            self.children_index[msg.parent_id].append(msg.id)


    def format_discussion(self, max_allowed_tokens: int, splitter_text: str = "!@>", branch_tip_id: Optional[str] = None) -> str:
        if branch_tip_id is None:
            branch_tip_id = self.active_branch_id

        branch_msgs = self.get_branch(branch_tip_id) if branch_tip_id else []
        formatted_text = ""
        current_tokens = 0

        # Start with system prompt if defined
        if self.system_prompt:
            sys_msg = f"!>system:\n{self.system_prompt.strip()}\n"
            sys_tokens = len(self.lollmsClient.tokenize(sys_msg))
            if max_allowed_tokens and current_tokens + sys_tokens <= max_allowed_tokens:
                formatted_text += sys_msg
                current_tokens += sys_tokens

        for msg in reversed(branch_msgs):
            content = msg.content.strip()
            # FIXED: Add a placeholder for images to represent them in text-only formats.
            if msg.images:
                content += f"\n({len(msg.images)} image(s) attached)"

            msg_text = f"{splitter_text}{msg.sender.replace(':', '').replace('!@>', '')}:\n{content}\n"
            msg_tokens = len(self.lollmsClient.tokenize(msg_text))
            if current_tokens + msg_tokens > max_allowed_tokens:
                break
            formatted_text = msg_text + formatted_text
            current_tokens += msg_tokens

        return formatted_text.strip()

    # gradio helpers -------------------------
    def get_branch_as_chatbot_history(self, branch_tip_id: Optional[str] = None) -> List[List[str]]:
        """
        Converts a discussion branch into Gradio's chatbot list format.
        [[user_msg, ai_reply], [user_msg, ai_reply], ...]
        """
        if branch_tip_id is None:
            branch_tip_id = self.active_branch_id
        if not branch_tip_id:
            return []
        
        branch = self.get_branch(branch_tip_id)
        history = []
        for msg in branch:
            # Determine the role from participants, default to 'user'
            role = self.participants.get(msg.sender, "user")

            if role == "user":
                history.append([msg.content, None])
            else:  # assistant
                # If the last user message has no reply yet, append to it
                if history and history[-1][1] is None:
                    history[-1][1] = msg.content
                else:  # Standalone assistant message (e.g., the first message)
                    history.append([None, msg.content])
        return history

    def render_discussion_tree(self, active_branch_highlight: bool = True) -> str:
        """
        Renders the entire discussion tree as formatted Markdown for display.
        """
        if not self.messages:
            return "No messages yet."

        tree_markdown = "### Discussion Tree\n\n"
        tree_markdown += "Click a message in the dropdown to switch branches.\n\n"
        
        # Find root nodes (messages with no parent)
        root_ids = [msg.id for msg in self.messages if msg.parent_id is None]
        
        # Recursive function to render a node and its children
        def _render_node(node_id: str, depth: int) -> str:
            node = self.message_index.get(node_id)
            if not node:
                return ""

            indent = "    " * depth
            # Highlight the active message
            is_active = ""
            if active_branch_highlight and node.id == self.active_branch_id:
                is_active = "  <span class='activ'>[ACTIVE]</span>"
            
            # Format the message line
            prefix = f"{indent}- **{node.sender}**: "
            content_preview = node.content.replace('\n', ' ').strip()[:80]
            line = f"{prefix} _{content_preview}..._{is_active}\n"
            
            # Recursively render children
            children_ids = self.children_index.get(node.id, [])
            for child_id in children_ids:
                line += _render_node(child_id, depth + 1)
            
            return line

        for root_id in root_ids:
            tree_markdown += _render_node(root_id, 0)
            
        return tree_markdown

    def get_message_choices(self) -> List[tuple]:
        """
        Creates a list of (label, id) tuples for a Gradio Dropdown component.
        """
        choices = [(f"{msg.sender}: {msg.content[:40]}... (ID: ...{msg.id[-4:]})", msg.id) for msg in self.messages]
        # Sort by message creation order (assuming self.messages is ordered)
        return choices
    
    
    def export(self, format_type: str, branch_tip_id: Optional[str] = None) -> Union[List[Dict], str]:
        """
        Exports the discussion history in a specific format suitable for different model APIs.

        Args:
            format_type (str): The target format. Supported values are:
                - "openai_chat": For OpenAI, llama.cpp, and other compatible chat APIs.
                - "ollama_chat": For Ollama's chat API.
                - "lollms_text": For the native lollms-webui text/image endpoints.
                - "openai_completion": For legacy text completion APIs.
            branch_tip_id (Optional[str]): The ID of the message to use as the
                tip of the conversation branch. Defaults to the active branch.

        Returns:
            Union[List[Dict], str]: The formatted conversation history, either as a
            list of dictionaries (for chat formats) or a single string.
        """
        if branch_tip_id is None:
            branch_tip_id = self.active_branch_id
        
        # Handle case of an empty or uninitialized discussion
        if branch_tip_id is None:
            return "" if format_type in ["lollms_text", "openai_completion"] else []

        branch = self.get_branch(branch_tip_id)

        # --------------------- OpenAI Chat Format ---------------------
        # Used by: OpenAI API, llama.cpp server, and many other compatible services.
        # Structure: List of dictionaries with 'role' and 'content'.
        # Images are handled via multi-part 'content'.
        # --------------------------------------------------------------
        if format_type == "openai_chat":
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt.strip()})

            def openai_image_block(image: Dict[str, str]) -> Dict:
                """Creates a dict for an image URL, either from a URL or base64 data."""
                image_url = image['data'] if image['type'] == 'url' else f"data:image/jpeg;base64,{image['data']}"
                return {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}

            for msg in branch:
                role = self.participants.get(msg.sender, "user")
                if msg.images:
                    content_parts = []
                    if msg.content.strip(): # Add text part only if content exists
                        content_parts.append({"type": "text", "text": msg.content.strip()})
                    content_parts.extend(openai_image_block(img) for img in msg.images)
                    messages.append({"role": role, "content": content_parts})
                else:
                    messages.append({"role": role, "content": msg.content.strip()})
            return messages

        # --------------------- Ollama Chat Format ---------------------
        # Used by: Ollama's '/api/chat' endpoint.
        # Structure: List of dictionaries with 'role', 'content', and an optional 'images' key.
        # Images must be a list of base64-encoded strings. URLs are ignored.
        # --------------------------------------------------------------
        elif format_type == "ollama_chat":
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt.strip()})
            
            for msg in branch:
                role = self.participants.get(msg.sender, "user")
                message_dict = {"role": role, "content": msg.content.strip()}
                
                # Filter for and add base64 images, as required by Ollama
                ollama_images = [img['data'] for img in msg.images if img['type'] == 'base64']
                if ollama_images:
                    message_dict["images"] = ollama_images
                
                messages.append(message_dict)
            return messages

        # --------------------- LoLLMs Native Text Format ---------------------
        # Used by: lollms-webui's '/lollms_generate' and '/lollms_generate_with_images' endpoints.
        # Structure: A single string with messages separated by special tokens like '!@>user:'.
        # Images are not part of the string but are sent separately by the binding.
        # --------------------------------------------------------------------
        elif format_type == "lollms_text":
            full_prompt_parts = []
            if self.system_prompt:
                full_prompt_parts.append(f"!@>system:\n{self.system_prompt.strip()}")
            
            for msg in branch:
                sender_str = msg.sender.replace(':', '').replace('!@>', '')
                content = msg.content.strip()
                # Images are handled separately by the binding, but a placeholder can be useful for context
                if msg.images:
                    content += f"\n({len(msg.images)} image(s) attached)"
                full_prompt_parts.append(f"!@>{sender_str}:\n{content}")
            
            return "\n".join(full_prompt_parts)

        # ------------------ Legacy OpenAI Completion Format ------------------
        # Used by: Older text-completion models.
        # Structure: A single string with human-readable roles (e.g., "User:", "Assistant:").
        # Images are represented by a text placeholder.
        # ----------------------------------------------------------------------
        elif format_type == "openai_completion":
            full_prompt_parts = []
            if self.system_prompt:
                full_prompt_parts.append(f"System:\n{self.system_prompt.strip()}")
            
            for msg in branch:
                role_label = self.participants.get(msg.sender, "user").capitalize()
                content = msg.content.strip()
                if msg.images:
                    content += f"\n({len(msg.images)} image(s) attached)"
                full_prompt_parts.append(f"{role_label}:\n{content}")
            
            return "\n\n".join(full_prompt_parts)

        else:
            raise ValueError(f"Unsupported export format_type: {format_type}")
# Example usage
if __name__ == "__main__":
    import base64

    # 🔧 Mock client for token counting
    from lollms_client import LollmsClient
    client = LollmsClient(binding_name="ollama",model_name="mistral:latest")
    discussion = LollmsDiscussion(client)

    # 👥 Set participants
    discussion.set_participants({
        "Alice": "user",
        "Bob": "assistant"
    })
    
    # 📝 Set a system prompt
    discussion.set_system_prompt("You are a helpful and friendly assistant.")

    # 📩 Add root message
    msg1 = discussion.add_message('Alice', 'Hello!')

    # 📩 Add reply
    msg2 = discussion.add_message('Bob', 'Hi there!')

    # 🌿 Branch from msg1 with an image
    msg3 = discussion.add_message(
        'Alice',
        'Here is an image of my dog.',
        parent_id=msg1,
        images=[{"type": "url", "data": "https://example.com/alices_dog.jpg"}]
    )

    # 🖼️ FIXED: Add another message with images using the 'images' parameter directly.
    sample_base64 = base64.b64encode(b'This is a test image of a cat').decode('utf-8')
    msg4 = discussion.add_message(
        'Bob',
        "Nice! Here's my cat.",
        parent_id=msg3,
        images=[
            {"type": "url", "data": "https://example.com/bobs_cat.jpg"},
            {"type": "base64", "data": sample_base64}
        ]
    )

    # 🌿 Switch to the new branch
    discussion.set_active_branch(msg4)

    # 📁 Save and load discussion
    discussion.save_to_disk("test_discussion.yaml")

    print("\n💾 Discussion saved to test_discussion.yaml")
    
    new_discussion = LollmsDiscussion(client)
    new_discussion.load_from_disk("test_discussion.yaml")
    # Participants must be set again as they are part of the runtime configuration
    # but the loader now correctly loads them from the file.
    print("📂 Discussion loaded from test_discussion.yaml")


    # 🧾 Format the discussion
    formatted = new_discussion.format_discussion(1000)
    print("\n📜 Formatted discussion (text-only with placeholders):\n", formatted)

    # 🔁 Export to OpenAI Chat format
    openai_chat = new_discussion.export("openai_chat")
    print("\n📦 OpenAI Chat format:\n", yaml.dump(openai_chat, allow_unicode=True, sort_keys=False))

    # 🔁 Export to OpenAI Completion format
    openai_completion = new_discussion.export("openai_completion")
    print("\n📜 OpenAI Completion format:\n", openai_completion)

    # 🔁 Export to Ollama Chat format
    ollama_export = new_discussion.export("ollama_chat")
    print("\n🤖 Ollama Chat format:\n", yaml.dump(ollama_export, allow_unicode=True, sort_keys=False))

    # Test that images were loaded correctly
    final_message = new_discussion.message_index[new_discussion.active_branch_id]
    assert len(final_message.images) == 2
    assert final_message.images[1]['type'] == 'base64'
    print("\n✅ Verification successful: Images were loaded correctly from the file.")
