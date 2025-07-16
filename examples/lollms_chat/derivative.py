from lollms_client import LollmsClient
import json

# Make sure you use your key
lc = LollmsClient(
    "openai",
    "http://localhost:9642/v1/",
    service_key="lollms_y-uyV-p2_AQGo5Ut6uHDmfIoRk6rKfmf0Rz6xQx-Zkl8cNyVUSFM"  # make sure you generate your own key
)

# if you want to see what binding/model does the server support, use this:
models = lc.listModels()
print(f"Found models:\n{models}")

lc.set_model_name("ollama/gemma3:27b")  # Or your preferred binding/model

function = input("Enter the function (e.g., x^2 + 2*x): ")
parameter = input("Enter the parameter to differentiate with respect to (e.g., x): ")

# Construct a detailed prompt
system_prompt = (
    "You are a symbolic differentiation engine. You receive a mathematical function "
    "and a parameter as input, and you return the derivative of the function with respect to that parameter. "
    "The function can include variables, numbers, and common mathematical operations. "
    "Return the derivative as a string. If the function or parameter is invalid, return 'Error'."
)

template = '"{derivative}": the derivative of the function with respect to the parameter'

# Include the function and parameter in the user prompt. This is important!
user_prompt = f"Find the derivative of the function '{function}' with respect to '{parameter}'."

# Generate the code
generation_output = lc.generate_code(
    user_prompt,
    system_prompt=system_prompt,
    template=template
)

try:
    # Attempt to parse the JSON response
    generation_output = json.loads(generation_output)
    derivative = generation_output["derivative"]
    print(f"Derivative: {derivative}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON response: {generation_output}")
except KeyError:
    print(f"Error: 'derivative' key not found in JSON response: {generation_output}")
