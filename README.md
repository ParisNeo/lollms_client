# lollms_client

[![Python Version](https://img.shields.io/pypi/pyversions/lollms-client)](https://pypi.org/project/lollms-client/) [![PyPI Downloads](https://img.shields.io/pypi/dw/lollms-client)](https://pypi.org/project/lollms-client/) [![Apache License](https://img.shields.io/apachie/2.0)](https://www.apache.org/licenses/LICENSE-2.0)

Welcome to the lollms_client repository! This library is built by [ParisNeo](https://github.com/ParisNeo) and provides a convenient way to interact with the lollms (Lord Of Large Language Models) API. It is available on [PyPI](https://pypi.org/project/lollms-client/) and distributed under the Apache 2.0 License.

## Installation

To install the library from PyPI using `pip`, run:

```
pip install lollms-client
```

## Usage

To use the lollms_client, first import the necessary classes:

```python
from lollms_client import LollmsClient

# Initialize the LollmsClient instance
lc = LollmsClient("http://localhost:9600")
```

### Text Generation

Use `generate_text()` for generating text from the lollms API.

```python
response = lc.generate_text(prompt="Once upon a time", stream=False, temperature=0.5)
print(response)
```

### Completion

Use `generate_completion()` for getting a completion of the prompt from the lollms API.

```python
response = lc.generate_completion(prompt="What is the capital of France", stream=False, temperature=0.5)
print(response)
```

### List Mounted Personalities

List mounted personalities of the lollms API with the `listMountedPersonalities()` method.

```python
response = lc.listMountedPersonalities()
print(response)
```

### List Models

List available models of the lollms API with the `listModels()` method.

```python
response = lc.listModels()
print(response)
```

## Complete Example

```python
from lollms_client import LollmsClient

# Initialize the LollmsClient instance
lc = LollmsClient("http://localhost:9600")

# Generate Text
response = lc.generate_text(prompt="Once upon a time", stream=False, temperature=0.5)
print(response)

# Generate Completion
response = lc.generate_completion(prompt="What is the capital of France", stream=False, temperature=0.5)
print(response)

# List Mounted Personalities
response = lc.listMountedPersonalities()
print(response)

# List Models
response = lc.listModels()
print(response)
```

Feel free to contribute to the project by submitting issues or pull requests. Follow [ParisNeo](https://github.com/ParisNeo) on [GitHub](https://github.com/ParisNeo), [Twitter](https://twitter.com/ParisNeo_AI), [Discord](https://discord.gg/BDxacQmv), [Sub-Reddit](r/lollms), and [Instagram](https://www.instagram.com/spacenerduino/) for updates and news.

Happy coding!