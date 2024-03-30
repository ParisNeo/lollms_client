# Import the library
from lollms_client import send_post_request

# Define the host address and prompt
host_address = "http://localhost:9600"
prompt = "Your prompt here"

# Send a POST request
response = send_post_request(host_address, prompt)

# Print the response
print(response)