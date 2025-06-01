# Class Information for lollms_tasks.py

File: `lollms_client\lollms_tasks.py`

## Classes

### TasksLibrary

```python
class TasksLibrary:
    def __init__(self, lollms: LollmsClient) -> None
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
    def translate_text_chunk(self, text_chunk, output_language: str = 'french', host_address: str = None, model_name: str = None, temperature = 0.7, max_generation_size = 3000) -> Any
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

