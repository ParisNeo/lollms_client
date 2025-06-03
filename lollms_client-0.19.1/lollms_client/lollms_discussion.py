import yaml
from lollms_client.lollms_core import LollmsClient
from dataclasses import dataclass, field
from typing import List
import uuid
import os

# LollmsMessage Class
@dataclass
class LollmsMessage:
    sender: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self):
        return {'sender': self.sender, 'content': self.content, 'id': self.id}

# LollmsDiscussion Class
class LollmsDiscussion:
    def __init__(self, lollmsClient:LollmsClient):
        self.messages:List[LollmsMessage] = []
        self.lollmsClient = lollmsClient

    def add_message(self, sender, content):
        message = LollmsMessage(sender, content)
        self.messages.append(message)

    def save_to_disk(self, file_path):
        with open(file_path, 'w') as file:
            yaml_data = [message.to_dict() for message in self.messages]
            yaml.dump(yaml_data, file)


    def format_discussion(self, max_allowed_tokens, splitter_text="!@>"):
        formatted_text = ""
        for message in reversed(self.messages):  # Start from the newest message
            formatted_message = f"{splitter_text}{message.sender.replace(':','').replace('!@>','')}:\n{message.content}\n"
            tokenized_message = self.lollmsClient.tokenize(formatted_message)
            if len(tokenized_message) + len(self.lollmsClient.tokenize(formatted_text)) <= max_allowed_tokens:
                formatted_text = formatted_message + formatted_text
            else:
                break  # Stop if adding the next message would exceed the limit
        return formatted_text

if __name__=="__main__":
    # Usage
    discussion = LollmsDiscussion()
    discussion.add_message(sender='Alice', content='Hi there, welcome to Lollms!')
    discussion.add_message(sender='Bob', content='See ya, thanks for using Lollms!')
    discussion.save_to_disk('lollms_discussion.yaml')

    # Dependency Installation
    # Ensure to install the PyYAML library using pip:
    # pip install PyYAML
