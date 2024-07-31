# Documentation Introduction

# Documentation Introduction

Welcome to the documentation for `lollms_tasks.py`, a key component of the `lollms_client` package. This module provides a comprehensive set of tools for managing and executing various tasks related to language model interactions. It is designed to facilitate the generation of text, image processing, and data extraction, among other functionalities.

The `TasksLibrary` class serves as the primary interface for users, offering a range of methods to generate prompts, process text, and handle callbacks. With support for advanced features such as multi-choice questions, text summarization, and function call execution, this module is equipped to handle complex tasks efficiently.

Before diving into the specifics, please ensure that you have the necessary dependencies installed, including `typing`, `json`, and various components from the `lollms_client` package. This documentation will guide you through the available classes, methods, and their respective functionalities, enabling you to leverage the full potential of the `lollms_tasks.py` module in your projects.

# Information for lollms_tasks.py

File: `lollms_client\lollms_tasks.py`

## Dependencies

- typing
- lollms_client.lollms_utilities
- functools
- sys
- ascii_colors
- json
- safe_store.document_decomposer
- lollms_client.lollms_core
- datetime
- lollms_client.lollms_types

## Classes

### TasksLibrary

```python
class TasksLibrary:
    def __init__(self, lollms: LollmsClient, callback: Callable[([str, MSG_TYPE, dict, list], bool)] = None) -> None
    def print_prompt(self, title, prompt) -> Any
    def setCallback(self, callback: Callable[([str, MSG_TYPE, dict, list], bool)]) -> Any
    def process(self, text: str, message_type: MSG_TYPE, callback = None, show_progress = False) -> Any
    def generate(self, prompt, max_size, temperature = None, top_k = None, top_p = None, repeat_penalty = None, repeat_last_n = None, callback = None, debug = False, show_progress = False, stream = False) -> Any
    def fast_gen(self, prompt: str, max_generation_size: int = None, placeholders: dict = {}, sacrifice: list = ['previous_discussion'], debug: bool = False, callback = None, show_progress = False, temperature = None, top_k = None, top_p = None, repeat_penalty = None, repeat_last_n = None) -> str
    def generate_with_images(self, prompt, images, max_size, temperature = None, top_k = None, top_p = None, repeat_penalty = None, repeat_last_n = None, callback = None, debug = False, show_progress = False, stream = False) -> Any
    def fast_gen_with_images(self, prompt: str, images: list, max_generation_size: int = None, placeholders: dict = {}, sacrifice: list = ['previous_discussion'], debug: bool = False, callback = None, show_progress = False) -> str
    def step_start(self, step_text, callback: Callable[([str, MSG_TYPE, dict, list], bool)] = None) -> Any
    def step_end(self, step_text, status = True, callback: Callable[([str, int, dict, list], bool)] = None) -> Any
    def step(self, step_text, callback: Callable[([str, MSG_TYPE, dict, list], bool)] = None) -> Any
    def sink(self, s = None, i = None, d = None) -> Any
    def build_prompt(self, prompt_parts: List[str], sacrifice_id: int = -1, context_size: int = None, minimum_spare_context_size: int = None) -> Any
    def translate_text_chunk(self, text_chunk, output_language: str = 'french', host_address: str = None, model_name: str = None, temperature = 0.1, max_generation_size = 3000) -> Any
    def extract_code_blocks(self, text: str) -> List[dict]
    def yes_no(self, question: str, context: str = '', max_answer_length: int = 50, conditionning = '') -> bool
    def multichoice_question(self, question: str, possible_answers: list, context: str = '', max_answer_length: int = 50, conditionning = '') -> int
    def summerize_text(self, text, summary_instruction = 'summerize', doc_name = 'chunk', answer_start = '', max_generation_size = 3000, max_summary_size = 512, callback = None, chunk_summary_post_processing = None, summary_mode = SUMMARY_MODE.SUMMARY_MODE_SEQUENCIAL) -> Any
    def smart_data_extraction(self, text, data_extraction_instruction = 'summerize', final_task_instruction = 'reformulate with better wording', doc_name = 'chunk', answer_start = '', max_generation_size = 3000, max_summary_size = 512, callback = None, chunk_summary_post_processing = None, summary_mode = SUMMARY_MODE.SUMMARY_MODE_SEQUENCIAL) -> Any
    def summerize_chunks(self, chunks, summary_instruction = 'summerize', doc_name = 'chunk', answer_start = '', max_generation_size = 3000, callback = None, chunk_summary_post_processing = None, summary_mode = SUMMARY_MODE.SUMMARY_MODE_SEQUENCIAL) -> Any
    def _upgrade_prompt_with_function_info(self, prompt: str, functions: List[Dict[(str, Any)]]) -> str
    def extract_function_calls_as_json(self, text: str) -> List[Dict[(str, Any)]]
    def execute_function_calls(self, function_calls: List[Dict[(str, Any)]], function_definitions: List[Dict[(str, Any)]]) -> List[Any]
    def generate_with_function_calls(self, prompt: str, functions: List[Dict[(str, Any)]], max_answer_length: Optional[int] = None, callback: Callable[([str, MSG_TYPE], bool)] = None) -> List[Dict[(str, Any)]]
    def generate_with_function_calls_and_images(self, prompt: str, images: list, functions: List[Dict[(str, Any)]], max_answer_length: Optional[int] = None, callback: Callable[([str, MSG_TYPE], bool)] = None) -> List[Dict[(str, Any)]]
```

