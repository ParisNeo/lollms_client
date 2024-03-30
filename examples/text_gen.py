# Import the library
from lollms_client import generate_text

# Define the host address and prompt
host_address = "http://localhost:9600"
prompt = "Your prompt here"

# Send a POST request
response = generate_text(host_address, prompt)

# Print the response
print(response)