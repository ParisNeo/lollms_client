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