---
name: Lollms Text Generation
description: Guides the model on performing basic text generation, counting tokens, detokenizing lists, and generating structurally validated JSON outputs with LollmsClient.
author: ParisNeo
version: 1.0.0
category: lollms_client/base
created: 2026-05-24
---

# Lollms Text Generation

This skill teaches how to interact with the core LLM capabilities of the `LollmsClient` class. It covers basic generation, token management, and structured schema generation.

## 1. Initializing LollmsClient
The client acts as the central interface for all interactions. It dynamically handles the loading of different local and remote binding backends.

```python
from lollms_client import LollmsClient

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={
        "model_name": "gemma4:e2b",
        "host_address": "http://localhost:11434"
    },
    user_name="ParisNeo",
    ai_name="Lollms"
)
```

## 2. Basic Text Generation
Perform simple completions using raw strings.

```python
# Generate text directly
response = client.generate_text(
    prompt="Why is Rust chosen for system-level programming?",
    system_prompt="You are a senior software architect. Answer concisely.",
    temperature=0.7,
    n_predict=512
)
print(response)
```

## 3. Token Management
Since different models have varying token behaviors, always use the client's tokenization APIs to safely calculate context footprints.

```python
# Tokenize a string into token IDs
tokens = client.tokenize("systems programming")

# Detokenize token IDs back to a string
decoded_string = client.detokenize(tokens)

# Directly count tokens of a text string
token_count = client.count_tokens("systems programming")
print(f"Token count: {token_count}")
```

## 4. Generating Structured JSON Content
Generating unstructured text that needs to be parsed can be fragile. Use the `generate_structured_content` or `generate_structured_content_pydantic` methods to enforce schema validation at generation time.

```python
# Generate structured outputs matching a Pydantic schema
from pydantic import BaseModel, Field

class APIEndpoint(BaseModel):
    route: str = Field(description="The URL route path, e.g. /api/users")
    method: str = Field(description="HTTP Method, e.g. GET, POST")
    description: str = Field(description="A short explanation of what the route does")

api_schema = client.generate_structured_content(
    prompt="Design a simple user registration endpoint",
    schema=APIEndpoint,
    temperature=0.1
)
print(api_schema)
# Output matches: {"route": "/api/register", "method": "POST", "description": "Registers a new user..."}
```