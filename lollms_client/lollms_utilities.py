import urllib
import numpy
from pathlib import Path
import pipmaster as pm
from PIL import Image
import io
import base64
import re
import numpy as np
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

