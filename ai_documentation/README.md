# User Guide for `lollms_client` library

This guide provides a comprehensive overview of the `lollms_client` module, its classes, methods, and usage examples. The `LollmsClient` class is designed to interact with various AI models and APIs for text generation, including support for image-based prompts and multiple generation formats.

---

## Table of Contents

1. **Introduction**
2. **Installation Requirements**
3. **Class Overview**
   - `LollmsClient`
   - `ELF_GENERATION_FORMAT`
   - `ELF_COMPLETION_FORMAT`
4. **Key Methods**
   - Initialization
   - Tokenization and Detokenization
   - Text Generation
   - Image-Based Generation
   - Model and Personality Management
   - Code Generation
5. **Examples**
   - Basic Text Generation
   - Image-Based Prompt Generation
   - Code Generation
6. **Error Handling**
7. **Advanced Features**
8. **Conclusion**

---

## 1. Introduction

The `lollms_client.py` module provides a Python client for interacting with AI models hosted on a server. It supports multiple generation formats, including LOLLMS, OpenAI, Ollama, LiteLLM, Transformers, and VLLM. The client can generate text, handle image-based prompts, and extract code blocks from generated responses.

---

## 2. Installation Requirements

Before using the `LollmsClient`, ensure the following dependencies are installed:

- `requests`
- `ascii_colors`
- `tiktoken`
- `pipmaster`
- `torch` (if using Transformers)
- `transformers` (if using Transformers)

To install the required packages, run:

```bash
pip install requests ascii_colors tiktoken pipmaster torch torchvision torchaudio transformers
```

---

## 3. Class Overview

### `LollmsClient`

The main class for interacting with AI models. It provides methods for text generation, image-based generation, and code extraction.

#### Parameters:
- `host_address` (str): The server address hosting the AI model (default: `http://localhost:9600`).
- `model_name` (str): The name of the model to use (optional if you are using lollms as backend).
- `ctx_size` (int): Context size for the model (default: 32000).
- `n_predict` (int): Number of tokens to predict (default: 4096).
- `temperature` (float): Sampling temperature (default: 0.1).
- `top_k` (int): Top-k sampling parameter (default: 50).
- `top_p` (float): Top-p (nucleus) sampling parameter (default: 0.95).
- `repeat_penalty` (float): Penalty for repeating tokens (default: 0.8).
- `n_threads` (int): Number of threads to use (default: 8).
- `default_generation_mode` (ELF_GENERATION_FORMAT): Default generation format (default: `LOLLMS`).

---

### `ELF_GENERATION_FORMAT`

An enumeration of supported generation formats:
- `LOLLMS`
- `OPENAI`
- `OLLAMA`
- `LITELLM`
- `TRANSFORMERS`
- `VLLM`

---

### `ELF_COMPLETION_FORMAT`

An enumeration of completion formats:
- `Instruct`
- `Chat`

---

## 4. Key Methods
### Initialization

```python
from lollms_client import LollmsClient

client = LollmsClient(
    host_address="http://localhost:9600", #optional
    model_name="gpt-3.5-turbo", # optional
    n_predict=512, # optional (default 4096)
    temperature=0.7, # optional (default 0.1)
    default_generation_mode=ELF_GENERATION_FORMAT.OPENAI # optional (default ELF_GENERATION_FORMAT.LOLLMS)
)
```

---

### Tokenization and Detokenization

#### `tokenize(prompt: str) -> list`
Tokenizes a given prompt into tokens.

```python
tokens = client.tokenize("Hello, world!")
print(tokens)
```

#### `detokenize(tokens_list: list) -> str`
Detokenizes a list of tokens into a string.

```python
text = client.detokenize(tokens)
print(text)
```

---

### Text Generation

#### `generate(prompt: str, ...) -> str`
Generates text based on a given prompt.

```python
response = client.generate("What is the capital of France?")
print(response)
```

#### `generate_with_images(prompt: str, images: list, ...) -> str`
Generates text based on a prompt and a list of images.

```python
response = client.generate_with_images("Describe this image:", ["image1.jpg", "image2.jpg"])
print(response)
```

---

### Code Generation

#### `generate_code(prompt: str, ...) -> str`
Generates a single code block based on a prompt.

```python
code = client.generate_code("Write a Python function to calculate the factorial of a number.")
print(code)
```

#### `generate_codes(prompt: str, ...) -> list`
Generates multiple code blocks based on a prompt.

```python
codes = client.generate_codes("Write Python and JavaScript functions to calculate the factorial of a number.")
for code in codes:
    print(code)
```

#### `extract_code_blocks(text: str) -> list`
Extracts code blocks from a given text.

```python
code_blocks = client.extract_code_blocks(response)
print(code_blocks)
```

---

### Model and Personality Management

#### `listMountedPersonalities() -> list`
Lists all mounted personalities on the server.

```python
personalities = client.listMountedPersonalities()
print(personalities)
```

#### `list_models() -> list`
Lists all available models on the server.

```python
models = client.list_models()
print(models)
```


---

## 5. Examples

### Example 1: Basic Text Generation

```python
response = client.generate("Tell me a joke.")
print(response)
```

### Example 2: Image-Based Prompt Generation

```python
response = client.generate_with_images("Describe this image:", ["image1.jpg"])
print(response)
```

### Example 3: Code Generation

```python
code = client.generate_code("Write a Python function to reverse a string.")
print(code)
```

---

## 6. Error Handling

The client returns error messages in the form of dictionaries with `status` and `error` keys. Example:

```python
response = client.generate("Invalid prompt")
if isinstance(response, dict) and not response["status"]:
    print("Error:", response["error"])
```

---

## 7. Advanced Features

### Streaming Responses

The `generate` and `generate_with_images` methods support streaming responses using a callback function.

```python
def streaming_callback(chunk, msg_type):
    print(chunk, end="")
    return True

response = client.generate("Stream this response.", streaming_callback=streaming_callback)
```

---

## 8. Conclusion

The `LollmsClient` provides a versatile interface for interacting with AI models, supporting text and image-based prompts, multiple generation formats, and advanced features like streaming and code extraction. Use this guide to integrate the client into your projects effectively.
