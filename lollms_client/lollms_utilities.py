import urllib
import numpy
from pathlib import Path
import pipmaster as pm
from PIL import Image
import io
import base64
import re
import numpy as np

import json
from ascii_colors import ASCIIColors, trace_exception

def dict_to_markdown(d, indent=0):
    """
    Formats a dictionary (with potential nested lists and dicts) as a markdown list.

    Args:
        d (dict): The dictionary to format.
        indent (int): Current indentation level (used recursively).

    Returns:
        str: The formatted markdown string.
    """
    lines = []
    indent_str = ' ' * (indent * 2)
    
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionary
            lines.append(f"{indent_str}- {key}:")
            lines.append(dict_to_markdown(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{indent_str}- {key}:")
            for item in value:
                if isinstance(item, dict):
                    # Render nested dicts in the list
                    lines.append(dict_to_markdown(item, indent + 1))
                else:
                    # Render strings or other simple items in the list
                    lines.append(f"{' ' * (indent + 1) * 2}- {item}")
        else:
            # Simple key-value pair
            lines.append(f"{indent_str}- {key}: {value}")
    
    return "\n".join(lines)

def is_base64(s):
    """Check if the string is a valid base64 encoded string."""
    try:
        # Try to decode and then encode back to check for validity
        import base64
        base64.b64decode(s)
        return True
    except Exception as e:
        return False

def build_image_dicts(images):
    """
    Convert a list of image strings (base64 or URLs) into a list of dictionaries with type and data.

    Args:
        images (list): List of image strings (either base64-encoded or URLs).

    Returns:
        list: List of dictionaries in the format {'type': 'base64'/'url', 'data': <image string>}.
    """
    result = []

    for img in images:
        if isinstance(img, str):
            if is_base64(img):
                result.append({'type': 'base64', 'data': img})
            else:
                # Assuming it's a URL if not base64
                result.append({'type': 'url', 'data': img})
        else:
            result.append(img)

    return result

def robust_json_parser(json_string: str) -> dict:
    """
    Parses a possibly malformed JSON string using a series of corrective strategies.

    Args:
        json_string: A string expected to represent a JSON object or array.

    Returns:
        A dictionary parsed from the JSON string.

    Raises:
        ValueError: If parsing fails after all correction attempts.
    """


    # STEP 0: Attempt to parse directly
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        trace_exception(e)
        pass
    
    # STEP 1: Remove code block wrappers if present (e.g., ```json ... ```)
    json_string = re.sub(r"^```(?:json)?\s*|\s*```$", '', json_string.strip())

    # STEP 2: Attempt to parse directly
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass

    # STEP 2: Extract likely JSON substring
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', json_string)
    cleaned_string = json_match.group(0) if json_match else json_string

    try:
        # STEP 3a: Normalize Python/JS booleans/nulls
        cleaned_string = re.sub(r'\bTrue\b', 'true', cleaned_string)
        cleaned_string = re.sub(r'\bFalse\b', 'false', cleaned_string)
        cleaned_string = re.sub(r'\bNone\b', 'null', cleaned_string)

        # STEP 3b: Remove comments (single-line and block)
        cleaned_string = re.sub(r'//.*', '', cleaned_string)
        cleaned_string = re.sub(r'/\*[\s\S]*?\*/', '', cleaned_string)

        # STEP 3c: Remove bad escape sequences like \_ or \*
        cleaned_string = re.sub(r'\\([_`*#\-])', r'\1', cleaned_string)

        # STEP 3d: Remove trailing commas
        cleaned_string = re.sub(r',\s*(\}|\])', r'\1', cleaned_string)

        # STEP 3e: Escape unescaped newlines inside string literals
        def escape_newlines_in_strings(text: str) -> str:
            in_string = False
            result = []
            i = 0
            while i < len(text):
                c = text[i]
                if c == '"' and (i == 0 or text[i - 1] != '\\'):
                    in_string = not in_string
                if in_string and c == '\n':
                    result.append('\\n')
                else:
                    result.append(c)
                i += 1
            return ''.join(result)

        cleaned_string = escape_newlines_in_strings(cleaned_string)

        # STEP 3f: Escape unescaped inner double quotes inside strings
        def escape_unescaped_inner_quotes(text: str) -> str:
            def fix(match):
                s = match.group(0)
                inner = s[1:-1]
                # Escape double quotes that aren't already escaped
                inner_fixed = re.sub(r'(?<!\\)"', r'\\"', inner)
                return f'"{inner_fixed}"'
            return re.sub(r'"(?:[^"\\]|\\.)*"', fix, text)

        cleaned_string = escape_unescaped_inner_quotes(cleaned_string)

        # STEP 3g: Convert single-quoted strings to double quotes (arrays or object keys)
        cleaned_string = re.sub(
            r"(?<=[:\[,])\s*'([^']*?)'\s*(?=[,\}\]])", 
            lambda m: '"' + m.group(1).replace('"', '\\"') + '"', 
            cleaned_string
        )
        cleaned_string = re.sub(
            r"(?<=\{)\s*'([^']*?)'\s*:", 
            lambda m: '"' + m.group(1).replace('"', '\\"') + '":', 
            cleaned_string
        )

        # STEP 3h: Remove non-breaking spaces and control characters
        cleaned_string = re.sub(r'[\x00-\x1F\x7F\u00A0]', '', cleaned_string)

        # STEP 3i: Fix smart quotes
        cleaned_string = cleaned_string.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        # STEP 3j: Remove line breaks between JSON tokens that don't belong
        cleaned_string = re.sub(r'"\s*\n\s*"', '"\\n"', cleaned_string)

        # Final parse
        return json.loads(cleaned_string)

    except json.JSONDecodeError as e:
        print("\n--- JSONDecodeError ---")
        print(e)
        print("\n--- Original String ---")
        print(json_string)
        print("\n--- Final Cleaned String Attempted ---")
        print(cleaned_string)
        raise ValueError(f"Failed to parse JSON. Final error: {e}") from e


class PromptReshaper:
    def __init__(self, template:str):
        self.template = template
    def replace(self, placeholders:dict)->str:
        template = self.template
        # Calculate the number of tokens for each placeholder
        for placeholder, text in placeholders.items():
            template = template.replace(placeholder, text)
        return template
    def build(self, placeholders:dict, tokenize, detokenize, max_nb_tokens:int, place_holders_to_sacrifice:list=[])->str:
        # Tokenize the template without placeholders
        template_text = self.template
        template_tokens = tokenize(template_text)
        
        # Calculate the number of tokens in the template without placeholders
        template_tokens_count = len(template_tokens)
        
        # Calculate the number of tokens for each placeholder
        placeholder_tokens_count = {}
        all_count = template_tokens_count
        for placeholder, text in placeholders.items():
            text_tokens = tokenize(text)
            placeholder_tokens_count[placeholder] = len(text_tokens)
            all_count += placeholder_tokens_count[placeholder]

        def fill_template(template, data):
            for key, value in data.items():
                placeholder = "{{" + key + "}}"
                n_text_tokens = len(tokenize(template))
                if key in place_holders_to_sacrifice:
                    n_remaining = max_nb_tokens - n_text_tokens
                    t_value = tokenize(value)
                    n_value = len(t_value)
                    if n_value<n_remaining:
                        template = template.replace(placeholder, value)
                    else:
                        value = detokenize(t_value[-n_remaining:])
                        template = template.replace(placeholder, value)
                        
                else:
                    template = template.replace(placeholder, value)
            return template
        
        return fill_template(self.template, placeholders)


# Function to encode the image
def encode_image(image_path, max_image_width=-1):
    image = Image.open(image_path)
    width, height = image.size

    # Check and convert image format if needed
    if image.format not in ['PNG', 'JPEG', 'GIF', 'WEBP']:
        image = image.convert('JPEG')


    if max_image_width != -1 and width > max_image_width:
        ratio = max_image_width / width
        new_width = max_image_width
        new_height = int(height * ratio)
        f = image.format
        image = image.resize((new_width, new_height))
        image.format = f


    # Save the image to a BytesIO object
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=image.format)
    byte_arr = byte_arr.getvalue()

    return base64.b64encode(byte_arr).decode('utf-8')

def discussion_path_to_url(file_path:str|Path)->str:
    """
    This function takes a file path as an argument and converts it into a URL format. It first removes the initial part of the file path until the "outputs" string is reached, then replaces backslashes with forward slashes and quotes each segment with urllib.parse.quote() before joining them with forward slashes to form the final URL.

    :param file_path: str, the file path in the format of a Windows system
    :return: str, the converted URL format of the given file path
    """
    file_path = str(file_path)
    url = "/"+file_path[file_path.index("discussion_databases"):].replace("\\","/").replace("discussion_databases","discussions")
    return "/".join([urllib.parse.quote(p, safe="") for p in url.split("/")])

def personality_path_to_url(file_path:str|Path)->str:
    """
    This function takes a file path as an argument and converts it into a URL format. It first removes the initial part of the file path until the "outputs" string is reached, then replaces backslashes with forward slashes and quotes each segment with urllib.parse.quote() before joining them with forward slashes to form the final URL.

    :param file_path: str, the file path in the format of a Windows system
    :return: str, the converted URL format of the given file path
    """
    file_path = str(file_path)
    url = "/"+file_path[file_path.index("personalities_zoo"):].replace("\\","/").replace("personalities_zoo","personalities")
    return "/".join([urllib.parse.quote(p, safe="") for p in url.split("/")])



def remove_text_from_string(string: str, text_to_find:str):
    """
    Removes everything from the first occurrence of the specified text in the string (case-insensitive).

    Parameters:
    string (str): The original string.
    text_to_find (str): The text to find in the string.

    Returns:
    str: The updated string.
    """
    index = string.lower().find(text_to_find.lower())

    if index != -1:
        string = string[:index]

    return string



def process_ai_output(output, images, output_folder):
    if not pm.is_installed("opencv-python"):
        pm.install("opencv-python")
    import cv2
    images = [cv2.imread(str(img)) for img in images]
    # Find all bounding box entries in the output
    bounding_boxes = re.findall(r'boundingbox\((\d+), ([^,]+), ([^,]+), ([^,]+), ([^,]+), ([^,]+)\)', output)

    # Group bounding boxes by image index
    image_boxes = {}
    for box in bounding_boxes:
        image_index = int(box[0])
        if image_index not in image_boxes:
            image_boxes[image_index] = []
        image_boxes[image_index].append(box[1:])

    # Process each image and its bounding boxes
    for image_index, boxes in image_boxes.items():
        # Get the corresponding image
        image = images[image_index]

        # Draw bounding boxes on the image
        for box in boxes:
            label, left, top, width, height = box
            left, top, width, height = float(left), float(top), float(width), float(height)
            x, y, w, h = int(left * image.shape[1]), int(top * image.shape[0]), int(width * image.shape[1]), int(height * image.shape[0])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the modified image
        random_stuff = np.random
        output_path = Path(output_folder)/f"image_{image_index}_{random_stuff}.jpg"
        cv2.imwrite(str(output_path), image)

    # Remove bounding box text from the output
    output = re.sub(r'boundingbox\([^)]+\)', '', output)

    # Append img tags for the generated images
    for image_index in image_boxes.keys():
        url = discussion_path_to_url(Path(output_folder)/f"image_{image_index}.jpg")
        output += f'\n<img src="{url}">'

    return output

