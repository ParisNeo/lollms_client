## DOCS FOR: `lollms_client/lollms_utilities.py`

**Purpose:**
This module provides a collection of utility functions and classes used across the `lollms_client` library, primarily for tasks like prompt reshaping, image encoding, path conversions, and text processing related to AI outputs.

---
### `PromptReshaper`

*   **Purpose**: A class designed to help format and potentially truncate prompts based on a template and token limits.
*   **Key Attributes**:
    *   `template` (str): A string template containing placeholders (e.g., `{{placeholder_name}}`).
*   **Methods**:
    *   **`__init__(template: str)`**:
        *   **Purpose**: Initializes the `PromptReshaper`.
        *   **Parameters**: `template` (str): The prompt template string.
    *   **`replace(placeholders: dict) -> str`**:
        *   **Purpose**: Performs a simple replacement of placeholders in the template with their corresponding text values. No token counting or truncation is done.
        *   **Parameters**: `placeholders` (dict): A dictionary where keys are placeholder names (without `{{}}`) and values are the text to substitute.
        *   **Returns**: The template string with placeholders replaced.
    *   **`build(placeholders: dict, tokenize: Callable, detokenize: Callable, max_nb_tokens: int, place_holders_to_sacrifice: list = []) -> str`**:
        *   **Purpose**: Fills the template with provided placeholder texts while attempting to adhere to a maximum token limit (`max_nb_tokens`). It prioritizes fitting all content, but if truncation is necessary, it can be guided by `place_holders_to_sacrifice` (sacrificing from the end of those placeholder's content first).
        *   **Parameters**:
            *   `placeholders` (dict): Placeholder names and their full text values.
            *   `tokenize` (Callable): A function that takes a string and returns a list of tokens.
            *   `detokenize` (Callable): A function that takes a list of tokens and returns a string.
            *   `max_nb_tokens` (int): The maximum allowed number of tokens for the final formatted prompt.
            *   `place_holders_to_sacrifice` (list, optional): A list of placeholder names that can be truncated if the total token count exceeds `max_nb_tokens`. Truncation happens from the end of the content of these placeholders.
        *   **Returns**: The formatted prompt string, potentially with some placeholder content truncated to fit token limits.

---
### Utility Functions

*   **`encode_image(image_path: Union[str, Path], max_image_width: int = -1) -> str`**:
    *   **Purpose**: Encodes an image file into a base64 string. Optionally resizes the image if its width exceeds `max_image_width`. Converts images to JPEG if they are not already PNG, JPEG, GIF, or WEBP to ensure broader compatibility for web/API transmission.
    *   **Parameters**:
        *   `image_path` (Union[str, Path]): The path to the image file.
        *   `max_image_width` (int, optional): If greater than -1, the image will be resized to this width, maintaining aspect ratio, if its original width is larger.
    *   **Returns**: A base64 encoded string representation of the image.
    *   **Dependencies**: `Pillow (PIL)`, `io`, `base64`.

*   **`discussion_path_to_url(file_path: Union[str, Path]) -> str`**:
    *   **Purpose**: Converts a local file path (typically within a LoLLMs discussion database structure) into a URL-friendly format suitable for web UIs. It targets paths under "discussion_databases" and transforms them to be accessible under a "/discussions/" route, with URL quoting.
    *   **Parameters**: `file_path` (Union[str, Path]): The local file path.
    *   **Returns**: A URL-friendly string.
    *   **Dependencies**: `urllib.parse`.

*   **`personality_path_to_url(file_path: Union[str, Path]) -> str`**:
    *   **Purpose**: Similar to `discussion_path_to_url`, but targets paths under "personalities_zoo" and transforms them for a "/personalities/" route.
    *   **Parameters**: `file_path` (Union[str, Path]): The local file path.
    *   **Returns**: A URL-friendly string.
    *   **Dependencies**: `urllib.parse`.

*   **`remove_text_from_string(string: str, text_to_find: str) -> str`**:
    *   **Purpose**: Removes the first occurrence of `text_to_find` (case-insensitive) and all subsequent text from the `string`.
    *   **Parameters**:
        *   `string` (str): The original string.
        *   `text_to_find` (str): The substring to find.
    *   **Returns**: The modified string, truncated from the first occurrence of `text_to_find`.

*   **`process_ai_output(output: str, images: List[Union[str, Path]], output_folder: Union[str, Path]) -> str`**:
    *   **Purpose**: Processes AI-generated text output that may contain special "boundingbox" directives. It draws these bounding boxes on the corresponding input images, saves the modified images to `output_folder`, and then removes the "boundingbox" directives from the text, appending `<img>` tags for the newly saved images.
    *   **Format of "boundingbox" directive**: `boundingbox(image_index, label, left, top, width, height)` where coordinates are normalized (0.0 to 1.0).
    *   **Parameters**:
        *   `output` (str): The AI-generated text potentially containing bounding box directives.
        *   `images` (List[Union[str, Path]]): A list of paths to the original input images referenced by `image_index` in the directives.
        *   `output_folder` (Union[str, Path]): The directory where modified images (with bounding boxes drawn) will be saved.
    *   **Returns**: The modified output string with "boundingbox" directives removed and `<img>` tags (using `discussion_path_to_url` for `src`) appended.
    *   **Dependencies**: `opencv-python` (cv2), `numpy`, `re`.

*   **`chunk_text(text: str, tokenizer: Callable, detokenizer: Callable, chunk_size: int, overlap: int, use_separators: bool = True) -> List[str]`**:
    *   **Purpose**: Splits a long text into smaller chunks based on token count, with a specified overlap between chunks. Optionally tries to split at natural text separators (paragraphs, sentences) for more coherent chunks.
    *   **Parameters**:
        *   `text` (str): The input text to be chunked.
        *   `tokenizer` (Callable): A function that converts text to a list of tokens.
        *   `detokenizer` (Callable): A function that converts a list of tokens back to text.
        *   `chunk_size` (int): The desired number of tokens in each chunk.
        *   `overlap` (int): The number of tokens to overlap between consecutive chunks.
        *   `use_separators` (bool, optional): If `True` (default), the function attempts to make cuts at paragraph or sentence boundaries near the `chunk_size` limit. If `False`, it cuts strictly by token count.
    *   **Returns**: A list of text strings, each representing a chunk.
    *   **Dependencies**: `re` (if `use_separators` is True).

**Usage Example (PromptReshaper):**
```python
from lollms_client.lollms_utilities import PromptReshaper

# Dummy tokenizer/detokenizer for example
def my_tokenize(text): return text.split()
def my_detokenize(tokens): return " ".join(tokens)

template = "Context: {{context_data}}\n\nQuestion: {{user_question}}\n\nAnswer:"
reshaper = PromptReshaper(template)

placeholders = {
    "context_data": "This is a very long piece of context information that might need to be truncated based on token limits. It talks about many things.",
    "user_question": "What is the main point of the context?"
}

# Simple replacement
filled_prompt_simple = reshaper.replace(placeholders)
# print(f"Simple Filled Prompt:\n{filled_prompt_simple}")

# Build with token limit (e.g., 20 tokens)
# This will likely truncate 'context_data' if 'user_question' is not in place_holders_to_sacrifice
filled_prompt_limited = reshaper.build(
    placeholders,
    my_tokenize,
    my_detokenize,
    max_nb_tokens=20,
    place_holders_to_sacrifice=["context_data"] # Allow 'context_data' to be shortened
)
print(f"\nLimited Token Prompt (approx 20 tokens):\n{filled_prompt_limited}")
# Expected: Context might be shortened, question should remain intact.
```

**Dependencies:**
*   `pipmaster` (used by `process_ai_output` to ensure `opencv-python` if not present).
*   `Pillow (PIL)`
*   `numpy`
*   `opencv-python` (optional, installed on demand by `process_ai_output`)
*   `re`, `io`, `base64`, `urllib.parse` (standard libraries).
