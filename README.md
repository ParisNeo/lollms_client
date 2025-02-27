# lollms_client

[![Python Version](https://img.shields.io/pypi/pyversions/lollms-client)](https://pypi.org/project/lollms-client/) [![PyPI Downloads](https://img.shields.io/pypi/dw/lollms-client)](https://pypi.org/project/lollms-client/) [![Apache License](https://img.shields.io/apachie/2.0)](https://www.apache.org/licenses/LICENSE-2.0)

Welcome to the lollms_client repository! This library is built by [ParisNeo](https://github.com/ParisNeo) and provides a convenient way to interact with the lollms (Lord Of Large Language Models) API. It is available on [PyPI](https://pypi.org/project/lollms-client/) and distributed under the Apache 2.0 License.

## Installation

To install the library from PyPI using `pip`, run:

```
pip install lollms-client
``` 

## Getting Started

The LollmsClient class is the gateway to interacting with the lollms API. Here's how you can instantiate it in various ways to suit your needs:

```python
from lollms_client import LollmsClient

# Default instantiation using the local lollms service - hosted at http://localhost:9600
lc = LollmsClient()

# Specify a custom host and port
lc = LollmsClient(host_adress="http://some.server:9600")

# Use a specific model with a local or remote ollama server
from lollms_client import ELF_GENERATION_FORMAT
lc = LollmsClient(model_name="phi4:latest", default_generation_mode = ELF_GENERATION_FORMAT.OLLAMA)

# Use a specific model with a local or remote OpenAI server (you can either set your key as an environment variable or pass it here)
lc = LollmsClient(model_name="gpt-3.5-turbo-0125", default_generation_mode = ELF_GENERATION_FORMAT.OPENAI)

# Use a specific model with an Ollama binding on the server, with a context size of 32800
lc = LollmsClient(
    host_adress="http://some.other.server:11434",
    model_name="phi4:latest",
    ctx_size=32800,
    default_generation_mode=ELF_GENERATION_FORMAT.OLLAMA
)
```

### Text Generation

Use `generate()` for generating text from the lollms API.

```python
response = lc.generate(prompt="Once upon a time", stream=False, temperature=0.5)
print(response)
```

### Code Generation

The `generate_code()` function allows you to generate code snippets based on your input. Here's how you can use it:

```python
# A generic case to generate a snippet in python
response = lc.generate_code(prompt="Create a function to add all numbers of a list", language='python')
print(response)

# generate_code can also be used to generate responses ready to be parsed - with json or yaml for instance
response = lc.generate_code(prompt="Mr Alex Brown presents himself to the pharmacist. He is 20 years old and seeks an appointment for the 12th of October. Fill out his application.", language='json', template="""
{
 "name":"the first name of the person"
 "family_name":"the family name of the person"
 "age":"the age of the person"
 "appointment_date":"the date of the appointment"
 "reason":"the reason for the appointment. if not specified fill out with 'N/A'"
}
""")
data = json.loads(response)
print(data['name'], data['family_name'], "- Reason:", data['reason'])
```


### List Mounted Personalities (only on lollms)

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
import json
from datetime import datetime

# Assuming LollmsClient is already imported and instantiated as lc
lc = LollmsClient()

# Generate code using the LollmsClient
response = lc.generate_code(
    prompt="Mr Alex Brown presents himself to the pharmacist. He is 20 years old and seeks an appointment for the 12th of October. Fill out his application.",
    language='json',
    template="""
    {
        "name": "the first name of the person",
        "family_name": "the family name of the person",
        "age": "the age of the person",
        "appointment_date": "the date of the appointment in the format DD/MM/YYYY",
        "reason": "the reason for the appointment. if not specified fill out with 'N/A'"
    }
    """
)

# Parse the JSON response
data = json.loads(response)

# Function to validate the data
def validate_data(data):
    try:
        # Validate age
        if not (0 < int(data['age']) < 120):
            raise ValueError("Invalid age provided.")
        
        # Validate appointment date
        appointment_date = datetime.strptime(data['appointment_date'], '%d/%m/%Y')
        if appointment_date < datetime.now():
            raise ValueError("Appointment date cannot be in the past.")
        
        # Validate name fields
        if not data['name'] or not data['family_name']:
            raise ValueError("Name fields cannot be empty.")
        
        return True
    except Exception as e:
        print(f"Validation Error: {e}")
        return False

# Function to simulate a response to the user
def simulate_response(data):
    if validate_data(data):
        print(f"Appointment confirmed for {data['name']} {data['family_name']}.")
        print(f"Date: {data['appointment_date']}")
        print(f"Reason: {data['reason']}")
    else:
        print("Failed to confirm appointment due to invalid data.")

# Execute the simulation
simulate_response(data)
```

Feel free to contribute to the project by submitting issues or pull requests. Follow [ParisNeo](https://github.com/ParisNeo) on [GitHub](https://github.com/ParisNeo), [Twitter](https://twitter.com/ParisNeo_AI), [Discord](https://discord.gg/BDxacQmv), [Sub-Reddit](r/lollms), and [Instagram](https://www.instagram.com/spacenerduino/) for updates and news.

Happy coding!
