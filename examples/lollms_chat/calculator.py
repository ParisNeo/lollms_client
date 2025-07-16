  from lollms_client import LollmsClient
import json
import math  # Import the math module for calculations

# Make sure you use your key
lc = LollmsClient(
    "openai",
    "http://localhost:9642/v1/",
    service_key="lollms_y-uyV-p2_AQGo5Ut6uHDmfIoRk6rKfmf0Rz6xQx-Zkl8cNyVUSFM"# make sure you generate your own key
)

# if you want to see what binding/model does the server support, use this:
models = lc.listModels()
print(f"Found models:\n{models}")

lc.set_model_name("ollama/gemma3:27b")  # Or your preferred binding/model

expression = input("Give an expression to evaluate: ")

# Construct a detailed prompt
system_prompt = (
    "You are a highly accurate calculator.  You receive a mathematical expression "
    "as input and return the result as a JSON object.  "
    "The expression can include numbers, basic arithmetic operators (+, -, *, /), "
    "parentheses, and common mathematical functions like sin, cos, tan, pi, sqrt, and log.  "
    "Always evaluate the expression and return the final numeric result.  If the expression is invalid, return 'Error'."
)

template = '{"result": the numeric result of the evaluated expression}'

# Include the expression in the user prompt.  This is important!
user_prompt = f"Evaluate the following expression: {expression}"

# Generate the code
generation_output = lc.generate_code(
    user_prompt,
    system_prompt=system_prompt,
    template=template
)

try:
    # Attempt to parse the JSON response
    generation_output = json.loads(generation_output)
    result = generation_output["result"]

    # Attempt to convert the result to a float
    try:
        result = float(result)
        print(f"Result: {result}")
    except ValueError:
        print(f"Result: {result} (Could not convert to a number)") #Handles cases where the LLM returns non-numeric output
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON response: {generation_output}")
except KeyError:
    print(f"Error: 'result' key not found in JSON response: {generation_output}")
