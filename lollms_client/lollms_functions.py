from functools import partial
from lollms_client.lollms_tasks import TasksLibrary
from typing import Dict, Any, List

class FunctionCalling_Library:
    def __init__(self, tasks_library:TasksLibrary):
        self.tl = tasks_library
        self.function_definitions = []

    def register_function(self, function_name, function_callable, function_description, function_parameters):
        self.function_definitions.append({
            "function_name": function_name,
            "function": function_callable,
            "function_description": function_description,
            "function_parameters": function_parameters
        })

    def unregister_function(self, function_name):
        self.function_definitions = [func for func in self.function_definitions if func["function_name"] != function_name]



    def execute_function_calls(self, function_calls: List[Dict[str, Any]]) -> List[Any]:
        """
        Executes the function calls with the parameters extracted from the generated text,
        using the original functions list to find the right function to execute.

        Args:
            function_calls (List[Dict[str, Any]]): A list of dictionaries representing the function calls.
            function_definitions (List[Dict[str, Any]]): The original list of functions with their descriptions and callable objects.

        Returns:
            List[Any]: A list of results from executing the function calls.
        """
        results = []
        # Convert function_definitions to a dict for easier lookup
        functions_dict = {func['function_name']: func['function'] for func in self.function_definitions}

        for call in function_calls:
            function_name = call.get("function_name")
            parameters = call.get("function_parameters", [])
            function = functions_dict.get(function_name)

            if function:
                try:
                    # Assuming parameters is a dictionary that maps directly to the function's arguments.
                    if type(parameters)==list:
                        result = function(*parameters)
                    elif type(parameters)==dict:
                        result = function(**parameters)
                    results.append(result)
                except TypeError as e:
                    # Handle cases where the function call fails due to incorrect parameters, etc.
                    results.append(f"Error calling {function_name}: {e}")
            else:
                results.append(f"Function {function_name} not found.")

        return results
    
    def generate_with_functions(self, prompt, stream=False, temperature=0.5, streaming_callback=None):
        # Assuming generate_with_function_calls is a method from TasksLibrary
        ai_response, function_calls = self.tl.generate_with_function_calls(prompt, self.function_definitions, callback=streaming_callback)
        return ai_response, function_calls

    def generate_with_functions_and_images(self, prompt, images=[], stream=False, temperature=0.5, streaming_callback=None):
        # Assuming generate_with_function_calls_and_images is a method from TasksLibrary
        if len(images) > 0:
            ai_response, function_calls = self.tl.generate_with_function_calls_and_images(prompt, images, self.function_definitions, callback=streaming_callback)
        else:
            ai_response, function_calls = self.tl.generate_with_function_calls(prompt, self.function_definitions)

        return ai_response, function_calls