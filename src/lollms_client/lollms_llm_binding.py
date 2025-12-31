# lollms_binding.py
from abc import abstractmethod
import importlib
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict, Any
from ascii_colors import trace_exception, ASCIIColors
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_utilities import ImageTokenizer, robust_json_parser
from lollms_client.lollms_base_binding import LollmsBaseBinding
import re
import yaml
import json

def load_known_contexts():
    """
    Loads the known_contexts data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the known_contexts data, or None if an error occurs.
    """
    try:
        file_path = Path(__file__).parent / "assets" / "models_ctx_sizes.json"
        with open(file_path, "r") as f:
            known_contexts = json.load(f)
        return known_contexts
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

class LollmsLLMBinding(LollmsBaseBinding):
    """Abstract base class for all LOLLMS LLM bindings"""
    
    def __init__(self, 
                 binding_name: Optional[str] ="unknown",
                 **kwargs
        ):
        """
        Initialize the LollmsLLMBinding base class.

        Args:
            binding_name (Optional[str]): The name of the bindingto be used
        """
        super().__init__(binding_name=binding_name, **kwargs)
        self.model_name = None #Must be set by the instance
        self.default_ctx_size = kwargs.get("ctx_size") 
        self.default_n_predict = kwargs.get("n_predict")
        self.default_stream = kwargs.get("stream")
        self.default_temperature = kwargs.get("temperature")
        self.default_top_k = kwargs.get("top_k")
        self.default_top_p = kwargs.get("top_p")
        self.default_repeat_penalty = kwargs.get("repeat_penalty")
        self.default_repeat_last_n = kwargs.get("repeat_last_n")
        self.default_seed = kwargs.get("seed")
        self.default_n_threads = kwargs.get("n_threads")
        self.default_streaming_callback = kwargs.get("streaming_callback")

        # Prompt Formatting defaults
        self.user_name = kwargs.get("user_name", "user")
        self.ai_name = kwargs.get("ai_name", "assistant")
        self.start_header_id_template = "!@>"
        self.end_header_id_template = ": "
        self.system_message_template = "system"
    
    @property
    def system_full_header(self) -> str:
        return f"{self.start_header_id_template}{self.system_message_template}{self.end_header_id_template}"

    @property
    def user_full_header(self) -> str:
        return f"{self.start_header_id_template}{self.user_name}{self.end_header_id_template}"

    @property
    def ai_full_header(self) -> str:
        return f"{self.start_header_id_template}{self.ai_name}{self.end_header_id_template}"

    @abstractmethod
    def generate_text(self,
                    prompt: str,
                    images: Optional[List[str]] = None,
                    system_prompt: str = "",
                    n_predict: Optional[int] = None,
                    stream: Optional[bool] = None,
                    temperature: Optional[float] = None,
                    top_k: Optional[int] = None,
                    top_p: Optional[float] = None,
                    repeat_penalty: Optional[float] = None,
                    repeat_last_n: Optional[int] = None,
                    seed: Optional[int] = None,
                    n_threads: Optional[int] = None,
                    ctx_size: int | None = None,
                    streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                    split:Optional[bool]=False, # put to true if the prompt is a discussion
                    user_keyword:Optional[str]="!@>user:",
                    ai_keyword:Optional[str]="!@>assistant:",
                    think: Optional[bool] = False,
                    reasoning_effort: Optional[bool] = "low", # low, medium, high
                    reasoning_summary: Optional[bool] = "auto", # auto
                    **kwargs
                    ) -> Union[str, dict]:
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.
        """
        pass

    def generate_from_messages(self,
                    messages: List[Dict],
                    n_predict: Optional[int] = None,
                    stream: Optional[bool] = None,
                    temperature: Optional[float] = None,
                    top_k: Optional[int] = None,
                    top_p: Optional[float] = None,
                    repeat_penalty: Optional[float] = None,
                    repeat_last_n: Optional[int] = None,
                    seed: Optional[int] = None,
                    n_threads: Optional[int] = None,
                    ctx_size: int | None = None,
                    streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                    think: Optional[bool] = False,
                    reasoning_effort: Optional[bool] = "low", # low, medium, high
                    reasoning_summary: Optional[bool] = "auto", # auto
                    **kwargs
                    ) -> Union[str, dict]:
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.
        """
        ASCIIColors.red("This binding does not support generate_from_messages")

    @abstractmethod
    def _chat(self,
            discussion: LollmsDiscussion,
            branch_tip_id: Optional[str] = None,
            n_predict: Optional[int] = None,
            stream: Optional[bool] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repeat_penalty: Optional[float] = None,
            repeat_last_n: Optional[int] = None,
            seed: Optional[int] = None,
            n_threads: Optional[int] = None,
            ctx_size: Optional[int] = None,
            streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
            think: Optional[bool] = False,
            reasoning_effort: Optional[bool] = "low", # low, medium, high
            reasoning_summary: Optional[bool] = "auto", # auto
            **kwargs
            ) -> Union[str, dict]:
        """
        A method to conduct a chat session with the model using a LollmsDiscussion object.
        """
        pass

    def chat(
        self,
        discussion: LollmsDiscussion,
        branch_tip_id: Optional[str] = None,
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        seed: Optional[int] = None,
        n_threads: Optional[int] = None,
        ctx_size: Optional[int] = None,
        streaming_callback: Optional[Callable[[str, MSG_TYPE, Dict], bool]] = None,
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",  # low, medium, high
        reasoning_summary: Optional[str] = "auto",  # auto
        **kwargs
    ) -> Union[str, dict]:
        """
        High-level method to perform a chat generation using a LollmsDiscussion object.
        
        This is the recommended method for conversational interactions. It uses the 
        discussion object to correctly format the context for the model, including 
        system prompts, roles, and multi-modal content.
        
        Args:
            discussion: The discussion object to use for context.
            branch_tip_id: The ID of the message to use as the end of the conversation 
                branch. If None, the active branch is used.
            n_predict: Maximum number of tokens to generate. Uses instance default if None.
            stream: Whether to stream the output. Uses instance default if None.
            temperature: Sampling temperature. Uses instance default if None.
            top_k: Top-k sampling parameter. Uses instance default if None.
            top_p: Top-p sampling parameter. Uses instance default if None.
            repeat_penalty: Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n: Number of previous tokens to consider for repeat penalty. 
                Uses instance default if None.
            seed: Random seed for generation. Uses instance default if None.
            n_threads: Number of threads to use. Uses instance default if None.
            ctx_size: Context size override for this generation.
            streaming_callback: Callback for streaming output.
            think: Enable thinking mode.
            reasoning_effort: Level of reasoning effort (low, medium, high).
            reasoning_summary: Reasoning summary mode.
            
        Returns:
            Generated text or an error dictionary if failed.
        """
        # Build the kwargs dictionary with resolved parameters
        chat_kwargs = {
            'discussion': discussion,
            'branch_tip_id': branch_tip_id,
            'n_predict': n_predict if n_predict is not None else self.default_n_predict,
            'stream': stream if stream is not None else (True if streaming_callback is not None else self.default_stream),
            'temperature': temperature if temperature is not None else self.default_temperature,
            'top_k': top_k if top_k is not None else self.default_top_k,
            'top_p': top_p if top_p is not None else self.default_top_p,
            'repeat_penalty': repeat_penalty if repeat_penalty is not None else self.default_repeat_penalty,
            'repeat_last_n': repeat_last_n if repeat_last_n is not None else self.default_repeat_last_n,
            'seed': seed if seed is not None else self.default_seed,
            'n_threads': n_threads if n_threads is not None else self.default_n_threads,
            'ctx_size': ctx_size if ctx_size is not None else self.default_ctx_size,
            'streaming_callback': streaming_callback if streaming_callback is not None else self.default_streaming_callback,
            'think': think,
            'reasoning_effort': reasoning_effort,
            'reasoning_summary': reasoning_summary,
        }
        
        # Call the internal chat method with all resolved parameters
        return self._chat(**chat_kwargs)

    def get_ctx_size(self, model_name: Optional[str|None] = None) -> Optional[int]:
        """
        Retrieves context size for a model from a hardcoded list.
        """
        if model_name is None:
            model_name = self.model_name

        known_contexts = load_known_contexts()

        normalized_model_name = model_name.lower().strip()
        sorted_base_models = sorted(known_contexts.keys(), key=len, reverse=True)

        for base_name in sorted_base_models:
            if base_name in normalized_model_name:
                context_size = known_contexts[base_name]
                ASCIIColors.warning(
                    f"Using hardcoded context size for model '{model_name}' "
                    f"based on base name '{base_name}': {context_size}"
                )
                return context_size

        ASCIIColors.warning(f"Context size not found for model '{model_name}' in the hardcoded list.")
        return self.default_ctx_size


    @abstractmethod
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of tokens.
        """
        pass
    
    @abstractmethod
    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text.
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text.
        """        
        pass

    def count_image_tokens(self, image: str) -> int:
        """
        Estimate the number of tokens for an image using ImageTokenizer based on self.model_name.
        """
        try:
            return ImageTokenizer(self.model_name).count_image_tokens(image)
        except Exception as e:
            ASCIIColors.warning(f"Could not estimate image tokens: {e}")
            return -1
    @abstractmethod
    def embed(self, text: str, **kwargs) -> list:
        """
        Get embeddings for the input text using Ollama API
        """
        pass    
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Return information about the current model.
        """
        pass

    def get_zoo(self) -> List[Dict[str, Any]]:
        """
        Returns a list of models available for download.
        each entry is a dict with:
        name, description, size, type, link
        """
        return []

    def download_from_zoo(self, index: int, progress_callback: Callable[[dict], None] = None) -> dict:
        """
        Downloads a model from the zoo using its index.
        """
        return {"status": False, "message": "Not implemented"}

    @abstractmethod
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model.
        """
        pass


    def split_discussion(self, lollms_prompt_string: str, system_keyword="!@>system:", user_keyword="!@>user:", ai_keyword="!@>assistant:") -> list:
        """
        Splits a LoLLMs prompt into a list of OpenAI-style messages.
        """
        pattern = r"(?={}|{}|{})".format(
            re.escape(system_keyword),
            re.escape(user_keyword),
            re.escape(ai_keyword)
        )
        parts = re.split(pattern, lollms_prompt_string)
        messages = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part.startswith(system_keyword):
                role = "system"
                content = part[len(system_keyword):].strip()
            elif part.startswith(user_keyword):
                role = "user"
                content = part[len(user_keyword):].strip()
            elif part.startswith(ai_keyword):
                role = "assistant"
                content = part[len(ai_keyword):].strip()
            else:
                if not messages:
                    role = "system"
                    content = part
                else:
                    continue

            messages.append({"role": role, "content": content})
            if messages[-1]["content"]=="":
                del messages[-1]
        return messages
    
    def ps(self):
        """
        List models (simulating a process status command).
        Since Lollms/OpenAI API doesn't have a specific 'ps' endpoint for running models with memory stats,
        we list available models and populate structure with available info, leaving hardware stats empty.
        """
        # Since there is no dedicated ps endpoint to see *running* models in the standard OpenAI API,
        # we list available models and try to map relevant info.
        models = self.list_models()
        standardized_models = []
        for m in models:
            standardized_models.append({
                "model_name": m.get("model_name"),
                "size": None,
                "vram_size": None,
                "gpu_usage_percent": None,
                "cpu_usage_percent": None,
                "expires_at": None,
                "parameters_size": None,
                "quantization_level": None,
                "parent_model": None,
                "context_size": m.get("context_length"),
                "owned_by": m.get("owned_by"),
                "created": m.get("created")
            })
        return standardized_models
        
    def get_context_size(self) -> Optional[int]:
        """
        Returns the default context size for the binding.
        """
        return self.default_ctx_size

    # --- High Level Operations ---

    def extract_thinking_blocks(self, text: str) -> List[str]:
        """
        Extracts content between <thinking> or <think> tags from a given text.
        """
        pattern = r'<(thinking|think)>(.*?)</\1>'
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        thinking_blocks = [match.group(2).strip() for match in matches]
        return thinking_blocks

    def remove_thinking_blocks(self, text: str) -> str:
        """
        Removes thinking blocks (either <thinking> or <think>) from text including the tags.
        """
        pattern = r'<(thinking|think)>.*?</\1>\s*'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
        return cleaned_text

    def extract_code_blocks(self, text: str, format: str = "markdown") -> List[dict]:
        """
        Extracts code blocks from text in Markdown or HTML format.
        """
        code_blocks = []
        remaining = text
        first_index = 0
        indices = []

        if format.lower() == "markdown":
            while remaining:
                try:
                    index = remaining.index("```")
                    indices.append(index + first_index)
                    remaining = remaining[index + 3:]
                    first_index += index + 3
                except ValueError:
                    if len(indices) % 2 == 1:
                        indices.append(first_index + len(remaining))
                    break

        elif format.lower() == "html":
            cursor = 0
            while cursor < len(text):
                try:
                    start_index = text.index("<code", cursor)
                    try:
                        end_of_opening = text.index(">", start_index)
                    except ValueError:
                        break

                    indices.append(start_index)
                    opening_tag_end = end_of_opening + 1
                    cursor = opening_tag_end

                    nest_level = 0
                    temp_cursor = cursor
                    found_closing = False
                    while temp_cursor < len(text):
                        if text[temp_cursor:].startswith("<code"):
                            nest_level += 1
                            try:
                                temp_cursor = text.index(">", temp_cursor) + 1
                            except ValueError:
                                break 
                        elif text[temp_cursor:].startswith("</code>"):
                            if nest_level == 0:
                                indices.append(temp_cursor)
                                cursor = temp_cursor + len("</code>")
                                found_closing = True
                                break
                            nest_level -= 1
                            temp_cursor += len("</code>")
                        else:
                            temp_cursor += 1

                    if not found_closing:
                        indices.append(len(text))
                        break

                except ValueError:
                    break
        else:
            raise ValueError("Format must be 'markdown' or 'html'")

        for i in range(0, len(indices), 2):
            block_infos = {
                'index': i // 2,
                'file_name': "",
                'content': "",
                'type': 'language-specific',
                'is_complete': False
            }

            start_pos = indices[i]
            search_area_start = max(0, start_pos - 200)
            preceding_text_segment = text[search_area_start:start_pos]
            lines = preceding_text_segment.strip().splitlines()
            if lines:
                last_line = lines[-1].strip()
                if last_line.startswith("<file_name>") and last_line.endswith("</file_name>"):
                    block_infos['file_name'] = last_line[len("<file_name>"):-len("</file_name>")].strip()
                elif last_line.lower().startswith("file:") or last_line.lower().startswith("filename:"):
                    block_infos['file_name'] = last_line.split(":", 1)[1].strip()

            if format.lower() == "markdown":
                content_start = start_pos + 3
                if i + 1 < len(indices):
                    end_pos = indices[i + 1]
                    content_raw = text[content_start:end_pos]
                    block_infos['is_complete'] = True
                else:
                    content_raw = text[content_start:]
                    block_infos['is_complete'] = False

                first_line_end = content_raw.find('\n')
                if first_line_end != -1:
                    first_line = content_raw[:first_line_end].strip()
                    if first_line and not first_line.isspace() and ' ' not in first_line:
                        block_infos['type'] = first_line
                        content = content_raw[first_line_end + 1:].strip()
                    else:
                        content = content_raw.strip()
                else:
                    content = content_raw.strip()
                    if content and not content.isspace() and ' ' not in content and len(content)<20:
                         block_infos['type'] = content
                         content = ""

            elif format.lower() == "html":
                try:
                    opening_tag_end = text.index(">", start_pos) + 1
                except ValueError:
                    continue

                opening_tag = text[start_pos:opening_tag_end]

                if i + 1 < len(indices):
                    end_pos = indices[i + 1]
                    content = text[opening_tag_end:end_pos].strip()
                    block_infos['is_complete'] = True
                else:
                    content = text[opening_tag_end:].strip()
                    block_infos['is_complete'] = False

                match = re.search(r'class\s*=\s*["\']([^"\']*)["\']', opening_tag)
                if match:
                    classes = match.group(1).split()
                    for cls in classes:
                        if cls.startswith("language-"):
                            block_infos['type'] = cls[len("language-"):]
                            break

            block_infos['content'] = content
            if block_infos['content'] or block_infos['is_complete']:
                code_blocks.append(block_infos)

        return code_blocks

    def generate_codes(
                        self,
                        prompt,
                        images=[],
                        template=None,
                        language="json",
                        code_tag_format="markdown",
                        n_predict = None,
                        temperature = None,
                        top_k = None,
                        top_p=None,
                        repeat_penalty=None,
                        repeat_last_n=None,
                        callback=None,
                        think: Optional[bool] = False,
                        reasoning_effort: Optional[bool] = "low",
                        reasoning_summary: Optional[bool] = "auto",
                        debug=False,
                        **kwargs
                        ):
        """Generates multiple code blocks based on a prompt."""
        response_full = ""
        system_prompt = f"""Act as a code generation assistant that generates code from user prompt."""

        if template:
            system_prompt += "Here is a template of the answer:\n"
            if code_tag_format=="markdown":
                system_prompt += f"""You must answer with the code placed inside the markdown code tag like this:
```{language}
{template}
```
{"Make sure you fill all fields and to use the exact same keys as the template." if language in ["json","yaml","xml"] else ""}
The code tag is mandatory.
Don't forget encapsulate the code inside a markdown code tag. This is mandatory.
"""
            elif code_tag_format=="html":
                system_prompt +=f"""You must answer with the code placed inside the html code tag like this:
<code language="{language}">
{template}
</code>
{"Make sure you fill all fields and to use the exact same keys as the template." if language in ["json","yaml","xml"] else ""}
The code tag is mandatory.
Don't forget encapsulate the code inside a html code tag. This is mandatory.
"""
        system_prompt += f"""Do not split the code in multiple tags."""

        response = self.generate_text(
            prompt,
            images=images,
            system_prompt=system_prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            streaming_callback=callback,
            think=think,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary
            )

        if isinstance(response, dict) and not response.get("status", True):
             ASCIIColors.error(f"Code generation failed: {response.get('error')}")
             return []

        response_full += response
        codes = self.extract_code_blocks(response, format=code_tag_format)
        return codes

    def generate_code(
                        self,
                        prompt:str,
                        images=[],
                        system_prompt:str|None=None,
                        template:str|None=None,
                        language="json",
                        code_tag_format="markdown",
                        n_predict:int|None = None,
                        temperature:float|None = None,
                        top_k:int|None= None,
                        top_p:float|None=None,
                        repeat_penalty:float|None=None,
                        repeat_last_n:int|None=None,
                        callback=None,
                        debug:bool=False,
                        override_all_prompts:bool=False,
                        **kwargs ):
        """Generates a single code block based on a prompt."""
        if override_all_prompts:
            final_system_prompt = system_prompt if system_prompt else ""
            final_prompt = prompt
        else:
            if not system_prompt:
                system_prompt = f"""Act as a code generation assistant that generates code from user prompt."""

            if template and template !="{}":
                if language in ["json","yaml","xml"]:
                    system_prompt += f"\nMake sure the generated context follows the following schema:\n```{language}\n{template}\n```\n"
                else:
                    system_prompt += f"\nHere is a template of the answer:\n```{language}\n{template}\n```\n"
                
                if code_tag_format=="markdown":
                    system_prompt += f"""You must answer with the code placed inside the markdown code tag:
```{language}
```
"""
            elif code_tag_format=="html":
                system_prompt +=f"""You must answer with the code placed inside the html code tag:
<code language="{language}">
</code>
"""
            system_prompt += f"""You must return a single code tag.
Do not split the code in multiple tags.
{self.ai_full_header}"""
            
            final_system_prompt = system_prompt
            final_prompt = prompt

        response = self.generate_text(
            final_prompt,
            images=images,
            system_prompt=final_system_prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            streaming_callback=callback
            )

        if isinstance(response, dict) and not response.get("status", True):
            ASCIIColors.error(f"Code generation failed: {response.get('error')}")
            return None

        codes = self.extract_code_blocks(response, format=code_tag_format)
        code_content = None

        if codes:
            last_code = codes[-1]
            code_content = last_code["content"]

            if not override_all_prompts:
                max_retries = 3
                retries = 0
                while not last_code["is_complete"] and retries < max_retries:
                    retries += 1
                    ASCIIColors.info(f"Code block seems incomplete. Attempting continuation ({retries}/{max_retries})...")
                    continuation_prompt = f"{prompt}\n\nAssistant:\n{code_content}\n\n{self.user_full_header}The previous code block was incomplete. Continue the code exactly from where it left off. Do not repeat the previous part. Only provide the continuation inside a single {code_tag_format} code tag.\n{self.ai_full_header}"

                    continuation_response = self.generate_text(
                        continuation_prompt,
                        images=images,
                        n_predict=n_predict,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        repeat_last_n=repeat_last_n,
                        streaming_callback=callback
                    )

                    if isinstance(continuation_response, dict) and not continuation_response.get("status", True):
                        break

                    continuation_codes = self.extract_code_blocks(continuation_response, format=code_tag_format)

                    if continuation_codes:
                        new_code_part = continuation_codes[0]["content"]
                        code_content += "\n" + new_code_part
                        last_code["is_complete"] = continuation_codes[0]["is_complete"]
                        if last_code["is_complete"]:
                            break
                    else:
                        break

        return code_content

    def update_code(
        self,
        original_code: str,
        modification_prompt: str,
        language: str = "python",
        images=[],
        system_prompt: str | None = None,
        patch_format: str = "unified",
        n_predict: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        repeat_last_n: int | None = None,
        callback=None,
        debug: bool = False,
        max_retries: int = 3
    ):
        """Updates existing code based on a modification prompt by generating and applying patches."""
        if not original_code or not original_code.strip():
            return {
                "success": False,
                "updated_code": None,
                "patch": None,
                "error": "Original code is empty"
            }
        
        if patch_format == "simple":
            patch_system_prompt = f"""You are a code modification assistant.
You will receive {language} code and a modification request.
Generate a patch using this EXACT format:

PATCH_START
REPLACE_LINE: <line_number>
OLD: <exact_old_line>
NEW: <new_line>
REPLACE_LINE: <another_line_number>
OLD: <exact_old_line>
NEW: <new_line>
PATCH_END

For adding lines:
ADD_AFTER: <line_number>
NEW: <line_to_add>

For removing lines:
REMOVE_LINE: <line_number>

Rules:
- Line numbers start at 1
- Match OLD lines EXACTLY including whitespace
- Only include lines that need changes
- Keep changes minimal and focused"""

        else:
            patch_system_prompt = f"""You are a code modification assistant.
You will receive {language} code and a modification request.
Generate a unified diff patch showing the changes.

Format your response as:
```diff
@@ -start_line,count +start_line,count @@
 context_line (unchanged)
-removed_line
+added_line
 context_line (unchanged)
```

Rules:
- Use standard unified diff format
- Include 1-2 lines of context around changes
- Be precise with line numbers
- Keep changes minimal"""

        if system_prompt:
            patch_system_prompt = system_prompt + "\n\n" + patch_system_prompt
        
        numbered_code = "\n".join(
            f"{i+1:4d}: {line}" 
            for i, line in enumerate(original_code.split("\n"))
        )
        
        patch_prompt = f"""Original {language} code (with line numbers for reference):
```{language}
{numbered_code}
```

Modification request: {modification_prompt}

Generate a patch to apply these changes. Follow the format specified in your instructions exactly."""

        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                response = self.generate_text(
                    patch_prompt,
                    images=images,
                    system_prompt=patch_system_prompt,
                    n_predict=n_predict or 2000,
                    temperature=temperature or 0.3,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    repeat_last_n=repeat_last_n,
                    streaming_callback=callback
                )
                
                if isinstance(response, dict) and not response.get("status", True):
                    raise Exception(f"Patch generation failed: {response.get('error')}")
                
                if patch_format == "simple":
                    updated_code, patch_text = self._apply_simple_patch(original_code, response, debug)
                else:
                    updated_code, patch_text = self._apply_unified_patch(original_code, response, debug)
                
                if updated_code:
                    return {
                        "success": True,
                        "updated_code": updated_code,
                        "patch": patch_text,
                        "error": None
                    }
                else:
                    raise Exception("Failed to apply patch - no valid changes found")
                    
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                if retry_count < max_retries and patch_format == "unified":
                    patch_format = "simple"
        
        return {
            "success": False,
            "updated_code": None,
            "patch": None,
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}"
        }

    def _apply_simple_patch(self, original_code: str, patch_response: str, debug: bool = False):
        try:
            lines = original_code.split("\n")
            patch_lines = []
            
            if "PATCH_START" in patch_response and "PATCH_END" in patch_response:
                start_idx = patch_response.index("PATCH_START")
                end_idx = patch_response.index("PATCH_END")
                patch_content = patch_response[start_idx + len("PATCH_START"):end_idx].strip()
            else:
                patch_content = patch_response
            
            modifications = []
            
            for line in patch_content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                patch_lines.append(line)
                
                if line.startswith("REPLACE_LINE:"):
                    try:
                        line_num = int(line.split(":")[1].strip()) - 1
                        idx = patch_content.index(line)
                        remaining = patch_content[idx:].split("\n")
                        old_line = None
                        new_line = None
                        for next_line in remaining[1:]:
                            if next_line.strip().startswith("OLD:"):
                                old_line = next_line[next_line.index("OLD:") + 4:].strip()
                            elif next_line.strip().startswith("NEW:"):
                                new_line = next_line[next_line.index("NEW:") + 4:].strip()
                                break
                        if old_line is not None and new_line is not None:
                            modifications.append(("replace", line_num, old_line, new_line))
                    except: pass
                elif line.startswith("ADD_AFTER:"):
                    try:
                        line_num = int(line.split(":")[1].strip()) - 1
                        idx = patch_content.index(line)
                        remaining = patch_content[idx:].split("\n")
                        for next_line in remaining[1:]:
                            if next_line.strip().startswith("NEW:"):
                                new_line = next_line[next_line.index("NEW:") + 4:].strip()
                                modifications.append(("add_after", line_num, None, new_line))
                                break
                    except: pass
                elif line.startswith("REMOVE_LINE:"):
                    try:
                        line_num = int(line.split(":")[1].strip()) - 1
                        modifications.append(("remove", line_num, None, None))
                    except: pass
            
            if not modifications:
                return None, None
            
            modifications.sort(key=lambda x: x[1], reverse=True)
            
            for mod_type, line_num, old_line, new_line in modifications:
                if line_num < 0 or line_num >= len(lines):
                    continue
                if mod_type == "replace":
                    if old_line and lines[line_num].strip() == old_line.strip():
                        indent = len(lines[line_num]) - len(lines[line_num].lstrip())
                        lines[line_num] = " " * indent + new_line.lstrip()
                elif mod_type == "add_after":
                    indent = len(lines[line_num]) - len(lines[line_num].lstrip())
                    lines.insert(line_num + 1, " " * indent + new_line.lstrip())
                elif mod_type == "remove":
                    del lines[line_num]
            
            updated_code = "\n".join(lines)
            patch_text = "\n".join(patch_lines)
            return updated_code, patch_text
        except:
            return None, None

    def _apply_unified_patch(self, original_code: str, patch_response: str, debug: bool = False):
        try:
            lines = original_code.split("\n")
            diff_pattern = r'```diff\n(.*?)\n```'
            diff_match = re.search(diff_pattern, patch_response, re.DOTALL)
            patch_text = diff_match.group(1) if diff_match else (patch_response if "@@" in patch_response else None)
            
            if not patch_text: return None, None
            
            hunk_pattern = r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@'
            hunks = re.finditer(hunk_pattern, patch_text)
            changes = []
            
            for hunk in hunks:
                old_start = int(hunk.group(1)) - 1
                hunk_start = hunk.end()
                next_hunk = re.search(hunk_pattern, patch_text[hunk_start:])
                hunk_end = hunk_start + next_hunk.start() if next_hunk else len(patch_text)
                hunk_lines = patch_text[hunk_start:hunk_end].strip().split("\n")
                
                for line in hunk_lines:
                    if not line: continue
                    if line.startswith("-"): changes.append(("remove", old_start, line[1:].strip()))
                    elif line.startswith("+"): changes.append(("add", old_start, line[1:].strip()))
            
            changes.sort(key=lambda x: x[1], reverse=True)
            
            for change_type, line_num, content in changes:
                if line_num < 0: continue
                if change_type == "remove":
                    if line_num < len(lines) and lines[line_num].strip() == content: del lines[line_num]
                elif change_type == "add":
                    indent = 0
                    if line_num > 0 and line_num-1 < len(lines):
                        indent = len(lines[line_num - 1]) - len(lines[line_num - 1].lstrip())
                    lines.insert(line_num, " " * indent + content)
            
            updated_code = "\n".join(lines)
            return updated_code, patch_text
        except: return None, None

    def generate_structured_content(self, prompt, images=None, schema=None, system_prompt=None, max_retries=1, use_override=False, **kwargs):
        """Enhanced structured content generation with optional prompt override."""
        images = [] if images is None else images
        schema = {} if schema is None else schema
        
        try:
            from jsonschema import validate
            has_validator = True
        except ImportError:
            has_validator = False

        if isinstance(schema, dict):
            schema_obj = schema
        elif isinstance(schema, str):
            try:
                schema_obj = json.loads(schema)
            except:
                raise ValueError("The provided schema string is not valid JSON")
        else:
            raise TypeError("schema must be a dict or a JSON string.")

        if "type" not in schema_obj and "properties" not in schema_obj and all(isinstance(v, dict) for v in schema_obj.values()):
            schema_obj = {"type": "object", "properties": schema_obj, "required": list(schema_obj.keys())}

        def _instance_skeleton(s):
            if not isinstance(s, dict): return {}
            if "const" in s: return s["const"]
            if "default" in s: return s["default"]
            t = s.get("type")
            if t == "string": return ""
            if t == "integer": return 0
            if t == "number": return 0.0
            if t == "boolean": return False
            if t == "array": return []
            if t == "object":
                out = {}
                for k, v in s.get("properties", {}).items(): out[k] = _instance_skeleton(v)
                return out
            return {}

        schema_str = json.dumps(schema_obj, indent=2, ensure_ascii=False)
        example_str = json.dumps(_instance_skeleton(schema_obj), indent=2, ensure_ascii=False)

        if use_override:
            final_system_prompt = system_prompt or ""
            final_prompt = prompt
            override_prompts = True
        else:
            base_system = (
                "Your objective is to generate a JSON object that satisfies the user's request and conforms to the provided schema.\n"
                f"Schema (reference only):\n```json\n{schema_str}\n```\n\n"
                f"Correct example:\n```json\n{example_str}\n```"
            )
            final_system_prompt = f"{system_prompt}\n\n{base_system}" if system_prompt else base_system
            final_prompt = prompt
            override_prompts = False

        for attempt in range(max_retries + 1):
            json_string = self.generate_code(
                prompt=final_prompt, images=images, system_prompt=final_system_prompt,
                template=example_str if not use_override else None,
                language="json", code_tag_format="markdown", override_all_prompts=override_prompts, **kwargs
            )
            
            if not json_string: continue
            
            try:
                parsed_json = robust_json_parser(json_string)
                if parsed_json is None: continue
                if has_validator:
                    try:
                        validate(instance=parsed_json, schema=schema_obj)
                        return parsed_json
                    except:
                        if attempt < max_retries: continue
                        return parsed_json
                return parsed_json
            except: break
        return None

    def generate_structured_content_pydantic(self, prompt, pydantic_model, images=None, system_prompt=None, max_retries=1, use_override=False, **kwargs):
        try:
            from pydantic import BaseModel, ValidationError
        except ImportError:
            ASCIIColors.error("Pydantic is required for this method.")
            return None
        
        if isinstance(pydantic_model, type) and issubclass(pydantic_model, BaseModel):
            model_class = pydantic_model
        elif isinstance(pydantic_model, BaseModel):
            model_class = type(pydantic_model)
        else:
            raise TypeError("pydantic_model must be a Pydantic BaseModel")
        
        try:
            schema = model_class.model_json_schema()
            schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
        except: return None
        
        if use_override:
            final_system_prompt = system_prompt or ""
            override_prompts = True
        else:
            base_system = f"Generate JSON strictly conforming to this Pydantic schema:\n```json\n{schema_str}\n```"
            final_system_prompt = f"{system_prompt}\n\n{base_system}" if system_prompt else base_system
            override_prompts = False
        
        for attempt in range(max_retries + 1):
            json_string = self.generate_code(
                prompt=prompt, images=images or [], system_prompt=final_system_prompt,
                language="json", code_tag_format="markdown", override_all_prompts=True, **kwargs
            )
            if not json_string: continue
            
            try:
                parsed_json = robust_json_parser(json_string)
                if parsed_json is None: continue
                try:
                    return model_class.model_validate(parsed_json)
                except ValidationError:
                    if attempt < max_retries: continue
                    return parsed_json
            except: break
        return None

    def yes_no(self, question: str, context: str = "", max_answer_length: int = None, conditionning: str = "", return_explanation: bool = False, callback = None) -> bool | dict:
        prompt = f"{self.system_full_header}{conditionning}\n{self.user_full_header}Based on the context, answer the question with only 'true' or 'false' and provide a brief explanation.\nContext:\n{context}\nQuestion: {question}\n{self.ai_full_header}"
        template = """{"answer": true, "explanation": "explanation"}"""
        
        json_str = self.generate_code(prompt=prompt, language="json", template=template, n_predict=max_answer_length, callback=callback)
        if not json_str: return {"answer": False, "explanation": "Failed"} if return_explanation else False
        
        try:
            data = robust_json_parser(json_str)
            ans = data.get("answer")
            if isinstance(ans, str): ans = ans.lower() == 'true'
            if return_explanation: return {"answer": ans, "explanation": data.get("explanation", "")}
            return ans
        except: return {"answer": False, "explanation": "Error"} if return_explanation else False

    def multichoice_question(self, question: str, possible_answers: list, context: str = "", max_answer_length: int = None, conditionning: str = "", return_explanation: bool = False, callback = None) -> int | dict:
        choices = "\n".join([f"{i}. {ans}" for i, ans in enumerate(possible_answers)])
        prompt = f"{self.system_full_header}{conditionning}\n{self.user_full_header}Answer the multiple-choice question. Return JSON with index.\nContext:\n{context}\nQuestion:\n{question}\nAnswers:\n{choices}\n{self.ai_full_header}"
        template = """{"index": 0, "explanation": "reason"}"""
        
        json_str = self.generate_code(prompt=prompt, template=template, language="json", n_predict=max_answer_length, callback=callback)
        if not json_str: return {"index": -1} if return_explanation else -1
        
        try:
            data = robust_json_parser(json_str)
            idx = data.get("index")
            if return_explanation: return {"index": idx, "explanation": data.get("explanation", "")}
            return idx
        except: return {"index": -1} if return_explanation else -1

    def multichoice_ranking(self, question: str, possible_answers: list, context: str = "", max_answer_length: int = None, conditionning: str = "", return_explanation: bool = False, callback = None) -> dict:
        choices = "\n".join([f"{i}. {ans}" for i, ans in enumerate(possible_answers)])
        prompt = f"{self.system_full_header}{conditionning}\n{self.user_full_header}Rank answers from best to worst. Return JSON list of indices.\nContext:\n{context}\nQuestion:\n{question}\nAnswers:\n{choices}\n{self.ai_full_header}"
        template = """{"ranking": [0, 1], "explanations": []}"""
        
        json_str = self.generate_code(prompt=prompt, template=template, language="json", n_predict=max_answer_length, callback=callback)
        if not json_str: return {"ranking": []}
        
        try:
            data = robust_json_parser(json_str)
            return {"ranking": data.get("ranking", []), "explanations": data.get("explanations", []) if return_explanation else []}
        except: return {"ranking": []}

class LollmsLLMBindingManager:
    """Manages binding discovery and instantiation"""

    def __init__(self, llm_bindings_dir: Union[str, Path] = Path(__file__).parent.parent / "llm_bindings"):
        self.llm_bindings_dir = Path(llm_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        """Dynamically load a specific binding implementation from the llm bindings directory."""
        binding_dir = self.llm_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.llm_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName)
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load binding {binding_name}: {str(e)}")

    def create_binding(self, 
                      binding_name: str,
                      **kwargs) -> Optional[LollmsLLMBinding]:
        """
        Create an instance of a specific binding.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)
        
        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            return binding_class(**kwargs)
        return None
    @staticmethod
    def _get_fallback_description(binding_name: str) -> Dict:
        """
        Generates a default description dictionary for a binding without a description.yaml file.
        """
        return {
            "binding_name": binding_name,
            "title": binding_name.replace("_", " ").title(),
            "author": "Unknown",
            "creation_date": "N/A",
            "last_update_date": "N/A",
            "description": f"A binding for {binding_name}. No description.yaml file was found, so common parameters are shown as a fallback.",
            "input_parameters": [
                {
                    "name": "model_name",
                    "type": "str",
                    "description": "The model name, ID, or filename to be used.",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "host_address",
                    "type": "str",
                    "description": "The host address of the service (for API-based bindings).",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "models_path",
                    "type": "str",
                    "description": "The path to the models directory (for local bindings).",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "service_key",
                    "type": "str",
                    "description": "The API key or service key for authentication (if applicable).",
                    "mandatory": False,
                    "default": ""
                }
            ]
        }

    @staticmethod
    def get_bindings_list(llm_bindings_dir: Union[str, Path]) -> List[Dict]:
        """
        Lists all available LLM bindings by scanning a directory.
        """
        bindings_dir = Path(llm_bindings_dir)
        if not bindings_dir.is_dir():
            return []

        bindings_list = []
        for binding_folder in bindings_dir.iterdir():
            if binding_folder.is_dir() and (binding_folder / "__init__.py").exists():
                binding_name = binding_folder.name
                description_file = binding_folder / "description.yaml"
                
                binding_info = {}
                if description_file.exists():
                    try:
                        with open(description_file, 'r', encoding='utf-8') as f:
                            binding_info = yaml.safe_load(f)
                        binding_info['binding_name'] = binding_name
                    except Exception as e:
                        print(f"Error loading description.yaml for {binding_name}: {e}")
                        binding_info = LollmsLLMBindingManager._get_fallback_description(binding_name)
                else:
                    binding_info = LollmsLLMBindingManager._get_fallback_description(binding_name)
                
                bindings_list.append(binding_info)

        return sorted(bindings_list, key=lambda b: b.get('title', b['binding_name']))
    
    def get_available_bindings(self) -> List[Dict]:
        """
        Retrieves a list of all available LLM bindings with their full descriptions.
        """
        return LollmsLLMBindingManager.get_bindings_list(self.llm_bindings_dir)

def get_available_bindings(llm_bindings_dir: Union[str, Path] = None) -> List[Dict]:
    """
    Lists all available LLM bindings with their detailed descriptions.
    """
    if llm_bindings_dir is None:
        llm_bindings_dir = Path(__file__).parent / "llm_bindings"
    return LollmsLLMBindingManager.get_bindings_list(llm_bindings_dir)

def list_binding_models(llm_binding_name: str, llm_binding_config: Optional[Dict[str, any]]|None = None, llm_bindings_dir: str|Path = Path(__file__).parent / "llm_bindings") -> List[Dict]:
    """
    Lists all available models for a specific binding.
    """
    binding = LollmsLLMBindingManager(llm_bindings_dir).create_binding(
        binding_name=llm_binding_name,
        **{
            k: v
            for k, v in (llm_binding_config or {}).items()
            if k != "binding_name"
        }
    )

    return binding.list_models() if binding else []