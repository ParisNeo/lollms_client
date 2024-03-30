# LoLLMs Client Library

LoLLMs Client is a Python library for interacting with the LoLLMs generate endpoint. This library simplifies the process of sending POST requests to the endpoint and handling responses.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [API Reference](#api-reference)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

To install the LoLLMs Client library, use pip:

```bash
pip install lollms_client
```

## Usage

Here's an example of how to use the LoLLMs Client library:

```python
from lollms_client import send_post_request

response = send_post_request(host_address="http://localhost:9600", prompt="Your prompt here")
print(response)
```

## API Reference

### send_post_request

Sends a POST request to the specified LoLLMs generate endpoint.

```python
send_post_request(
    host_address: str,
    prompt: str,
    model_name: Optional[str] = None,
    personality: int = -1,
    n_predict: int = 1024,
    stream: bool = False,
    temperature: float = 0.1,
    top_k: int = 50,
    top_p: float = 0.95,
    repeat_penalty: float = 0.8,
    repeat_last_n: int = 40,
    seed: Optional[int] = None,
    n_threads: int = 8
)
```

#### Parameters
- `host_address` (str): The host address of the LoLLMs generate endpoint (e.g., 'http://localhost:9600').
- `prompt` (str): The prompt to be sent to the LoLLMs generate endpoint.
- `model_name` (Optional[str]): The name of the model to be used (default: None).
- `personality` (int): The personality to be used (default: -1).
- `n_predict` (int): The number of tokens to predict (default: 1024).
- `stream` (bool): Whether to stream the response (default: False).
- `temperature` (float): The temperature for sampling (default: 0.1).
- `top_k` (int): The number of top choices for sampling (default: 50).
- `top_p` (float): The cumulative probability for top-p sampling (default: 0.95).
- `repeat_penalty` (float): The penalty for repeating previous tokens (default: 0.8).
- `repeat_last_n` (int): The number of previous tokens to consider for repeat penalty (default: 40).
- `seed` (Optional[int]): The seed for random number generation (default: None).
- `n_threads` (int): The number of threads to use for token prediction (default: 8).

#### Returns
- If the request is successful, the function returns the response text.
- If the request fails, the function returns a dictionary with the status set to False and the error message in the response: `{"status": False, "error": str(ex)}`.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please open a pull request with your proposed changes.

## License

LoLLMs Client is released under the [Apache 2.0 License](LICENSE).