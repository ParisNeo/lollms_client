import requests
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager
import json
from enum import Enum
import base64
import requests
import pipmaster as pm
from typing import List, Optional, Callable, Union, Dict
import numpy as np
import pipmaster as pm
from pathlib import Path
import os

        
class LollmsClient():
    """Core class for interacting with LOLLMS bindings"""
    def __init__(self, 
                 binding_name: str = "lollms",
                 host_address: Optional[str] = None,
                 model_name: str = "",
                 service_key: Optional[str] = None,
                 verify_ssl_certificate: bool = True,
                 personality: Optional[int] = None,
                 llm_bindings_dir: Path = Path(__file__).parent / "llm_bindings",
                 binding_config: Optional[Dict[str, any]] = None,
                 ctx_size: Optional[int] = 8192,
                 n_predict: Optional[int] = 4096,
                 stream: bool = False,
                 temperature: float = 0.1,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 repeat_penalty: float = 0.8,
                 repeat_last_n: int = 40,
                 seed: Optional[int] = None,
                 n_threads: int = 8,
                 streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                 user_name ="user",
                 ai_name = "assistant"):
        """
        Initialize the LollmsCore with a binding and generation parameters.

        Args:
            binding_name (str): Name of the binding to use (e.g., "lollms", "ollama").
            host_address (Optional[str]): Host address for the service. Overrides binding default if provided.
            model_name (str): Name of the model to use. Defaults to empty string.
            service_key (Optional[str]): Authentication key for the service.
            verify_ssl_certificate (bool): Whether to verify SSL certificates. Defaults to True.
            personality (Optional[int]): Personality ID (used only by LOLLMS binding).
            llm_bindings_dir (Path): Directory containing binding implementations. 
                Defaults to the "bindings" subdirectory relative to this file's location.
            binding_config (Optional[Dict[str, any]]): Additional configuration for the binding.
            n_predict (Optional[int]): Maximum number of tokens to generate. Default for generate_text.
            stream (bool): Whether to stream the output. Defaults to False for generate_text.
            temperature (float): Sampling temperature. Defaults to 0.1 for generate_text.
            top_k (int): Top-k sampling parameter. Defaults to 50 for generate_text.
            top_p (float): Top-p sampling parameter. Defaults to 0.95 for generate_text.
            repeat_penalty (float): Penalty for repeated tokens. Defaults to 0.8 for generate_text.
            repeat_last_n (int): Number of previous tokens to consider for repeat penalty. Defaults to 40.
            seed (Optional[int]): Random seed for generation. Default for generate_text.
            n_threads (int): Number of threads to use. Defaults to 8 for generate_text.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.
                Default for generate_text. Takes a string chunk and an MSG_TYPE enum value.

        Raises:
            ValueError: If the specified binding cannot be created.
        """
        self.binding_manager = LollmsLLMBindingManager(llm_bindings_dir)
        self.binding_config = binding_config or {}
        
        # Store generation parameters as instance variables
        self.default_ctx_size = ctx_size
        self.default_n_predict = n_predict
        self.default_stream = stream
        self.default_temperature = temperature
        self.default_top_k = top_k
        self.default_top_p = top_p
        self.default_repeat_penalty = repeat_penalty
        self.default_repeat_last_n = repeat_last_n
        self.default_seed = seed
        self.default_n_threads = n_threads
        self.default_streaming_callback = streaming_callback
        
        # Create the binding instance
        self.binding = self.binding_manager.create_binding(
            binding_name=binding_name,
            host_address=host_address,
            model_name=model_name,
            service_key=service_key,
            verify_ssl_certificate=verify_ssl_certificate,
            personality=personality
        )
        
        if self.binding is None:
            raise ValueError(f"Failed to create binding: {binding_name}. Available bindings: {self.binding_manager.get_available_bindings()}")
        
        # Apply additional configuration if provided
        if binding_config:
            for key, value in binding_config.items():
                setattr(self.binding, key, value)
        self.user_name = user_name
        self.ai_name = ai_name
        self.service_key = service_key

        self.verify_ssl_certificate = verify_ssl_certificate
        self.start_header_id_template ="!@>"
        self.end_header_id_template =": "
        self.system_message_template ="system"
        self.separator_template ="!@>"
        self.start_user_header_id_template ="!@>"
        self.end_user_header_id_template =": "
        self.end_user_message_id_template =""
        self.start_ai_header_id_template ="!@>"
        self.end_ai_header_id_template =": "
        self.end_ai_message_id_template =""


    @property
    def system_full_header(self) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_header_id_template}{self.system_message_template}{self.end_header_id_template}"
    
    def system_custom_header(self, ai_name) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_header_id_template}{ai_name}{self.end_header_id_template}"
    
    @property
    def user_full_header(self) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_user_header_id_template}{self.user_name}{self.end_user_header_id_template}"
    
    def user_custom_header(self, user_name="user") -> str:
        """Get the start_header_id_template."""
        return f"{self.start_user_header_id_template}{user_name}{self.end_user_header_id_template}"
    
    @property
    def ai_full_header(self) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_ai_header_id_template}{self.ai_name}{self.end_ai_header_id_template}"

    def ai_custom_header(self, ai_name) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_ai_header_id_template}{ai_name}{self.end_ai_header_id_template}"

    def sink(self, s=None,i=None,d=None):
        pass
    def tokenize(self, text: str) -> list:
        """
        Tokenize text using the active binding.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of tokens.
        """
        return self.binding.tokenize(text)
    
    def detokenize(self, tokens: list) -> str:
        """
        Detokenize tokens using the active binding.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            str: Detokenized text.
        """
        return self.binding.detokenize(tokens)
    
    def get_model_details(self) -> dict:
        """
        Get model information from the active binding.

        Returns:
            dict: Model information dictionary.
        """
        return self.binding.get_model_info()
    
    def switch_model(self, model_name: str) -> bool:
        """
        Load a new model in the active binding.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        return self.binding.load_model(model_name)
    
    def get_available_bindings(self) -> List[str]:
        """
        Get list of available bindings.

        Returns:
            List[str]: List of binding names that can be used.
        """
        return self.binding_manager.get_available_bindings()
    
    def generate_text(self, 
                     prompt: str,
                     images: Optional[List[str]] = None,
                     n_predict: Optional[int] = None,
                     stream: Optional[bool] = None,
                     temperature: Optional[float] = None,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None,
                     repeat_penalty: Optional[float] = None,
                     repeat_last_n: Optional[int] = None,
                     seed: Optional[int] = None,
                     n_threads: Optional[int] = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None) -> str:
        """
        Generate text using the active binding, using instance defaults if parameters are not provided.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths for multimodal generation.
            n_predict (Optional[int]): Maximum number of tokens to generate. Uses instance default if None.
            stream (Optional[bool]): Whether to stream the output. Uses instance default if None.
            temperature (Optional[float]): Sampling temperature. Uses instance default if None.
            top_k (Optional[int]): Top-k sampling parameter. Uses instance default if None.
            top_p (Optional[float]): Top-p sampling parameter. Uses instance default if None.
            repeat_penalty (Optional[float]): Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty. Uses instance default if None.
            seed (Optional[int]): Random seed for generation. Uses instance default if None.
            n_threads (Optional[int]): Number of threads to use. Uses instance default if None.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.
                Uses instance default if None.
                - First parameter (str): The chunk of text received from the stream.
                - Second parameter (MSG_TYPE): The message type enum (e.g., MSG_TYPE.MSG_TYPE_CHUNK).

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
        """
        return self.binding.generate_text(
            prompt=prompt,
            images=images,
            n_predict=n_predict if n_predict is not None else self.default_n_predict,
            stream=stream if stream is not None else self.default_stream,
            temperature=temperature if temperature is not None else self.default_temperature,
            top_k=top_k if top_k is not None else self.default_top_k,
            top_p=top_p if top_p is not None else self.default_top_p,
            repeat_penalty=repeat_penalty if repeat_penalty is not None else self.default_repeat_penalty,
            repeat_last_n=repeat_last_n if repeat_last_n is not None else self.default_repeat_last_n,
            seed=seed if seed is not None else self.default_seed,
            n_threads=n_threads if n_threads is not None else self.default_n_threads,
            streaming_callback=streaming_callback if streaming_callback is not None else self.default_streaming_callback
        )

    
    def embed(self, text):
        self.binding.embed(text)


    def listModels(self):
        self.binding.listModels()



    def generate_codes(
                        self, 
                        prompt, 
                        images=[], 
                        template=None,
                        language="json",
                        code_tag_format="markdown", # or "html"
                        max_size = None, 
                        temperature = None, 
                        top_k = None, 
                        top_p=None, 
                        repeat_penalty=None, 
                        repeat_last_n=None, 
                        callback=None, 
                        debug=False 
                        ):
        response_full = ""
        full_prompt = f"""{self.system_full_header}Act as a code generation assistant that generates code from user prompt.    
{self.user_full_header} 
{prompt}
"""
        if template:
            full_prompt += "Here is a template of the answer:\n"
            if code_tag_format=="markdown":
                full_prompt += f"""You must answer with the code placed inside the markdown code tag like this:
```{language}
{template}
```
{"Make sure you fill all fields and to use the exact same keys as the template." if language in ["json","yaml","xml"] else ""}
The code tag is mandatory.
Don't forget encapsulate the code inside a markdown code tag. This is mandatory.
"""
            elif code_tag_format=="html":
                full_prompt +=f"""You must answer with the code placed inside the html code tag like this:
<code language="{language}">
{template}
</code>
{"Make sure you fill all fields and to use the exact same keys as the template." if language in ["json","yaml","xml"] else ""}
The code tag is mandatory.
Don't forget encapsulate the code inside a html code tag. This is mandatory.
"""
        full_prompt += f"""Do not split the code in multiple tags.
{self.ai_full_header}"""

        if len(self.image_files)>0:
            response = self.generate_text_with_images(full_prompt, self.image_files, max_size, temperature, top_k, top_p, repeat_penalty, repeat_last_n, callback, debug=debug)
        elif  len(images)>0:
            response = self.generate_text_with_images(full_prompt, images, max_size, temperature, top_k, top_p, repeat_penalty, repeat_last_n, callback, debug=debug)
        else:
            response = self.generate_text(full_prompt, max_size, temperature, top_k, top_p, repeat_penalty, repeat_last_n, callback, debug=debug)
        response_full += response
        codes = self.extract_code_blocks(response)
        return codes
    
    def generate_code(
                        self, 
                        prompt, 
                        images=[],
                        template=None,
                        language="json",
                        code_tag_format="markdown", # or "html"                         
                        max_size = None,  
                        temperature = None,
                        top_k = None,
                        top_p=None,
                        repeat_penalty=None,
                        repeat_last_n=None,
                        callback=None,
                        debug=False ):
        
        full_prompt = f"""{self.system_full_header}Act as a code generation assistant that generates code from user prompt.    
{self.user_full_header} 
{prompt}
"""
        if template:
            full_prompt += "Here is a template of the answer:\n"
            if code_tag_format=="markdown":
                full_prompt += f"""You must answer with the code placed inside the markdown code tag like this:
```{language}
{template}
```
{"Make sure you fill all fields and to use the exact same keys as the template." if language in ["json","yaml","xml"] else ""}
The code tag is mandatory.
Don't forget encapsulate the code inside a markdown code tag. This is mandatory.
"""
            elif code_tag_format=="html":
                full_prompt +=f"""You must answer with the code placed inside the html code tag like this:
<code language="{language}">
{template}
</code>
{"Make sure you fill all fields and to use the exact same keys as the template." if language in ["json","yaml","xml"] else ""}
The code tag is mandatory.
Don't forget encapsulate the code inside a html code tag. This is mandatory.
"""
        full_prompt += f"""You must return a single code tag.
Do not split the code in multiple tags.
{self.ai_full_header}"""
        response = self.generate_text(full_prompt, images, max_size, temperature, top_k, top_p, repeat_penalty, repeat_last_n, streaming_callback=callback)
        codes = self.extract_code_blocks(response)
        if len(codes)>0:
            if not codes[-1]["is_complete"]:
                code = "\n".join(codes[-1]["content"].split("\n")[:-1])
                while not codes[-1]["is_complete"]:
                    response = self.generate_text(prompt+code+self.user_full_header+"continue the code. Start from last line and continue the code. Put the code inside a markdown code tag."+self.separator_template+self.ai_full_header, max_size, temperature, top_k, top_p, repeat_penalty, repeat_last_n, streaming_callback=callback)
                    codes = self.extract_code_blocks(response)
                    if len(codes)==0:
                        break
                    else:
                        if not codes[-1]["is_complete"]:
                            code +="\n"+ "\n".join(codes[-1]["content"].split("\n")[:-1])
                        else:
                            code +="\n"+ "\n".join(codes[-1]["content"].split("\n"))
            else:
                code = codes[-1]["content"]

            return code
        else:
            return None

    def extract_code_blocks(self, text: str) -> List[dict]:
        """
        This function extracts code blocks from a given text.

        Parameters:
        text (str): The text from which to extract code blocks. Code blocks are identified by triple backticks (```).

        Returns:
        List[dict]: A list of dictionaries where each dictionary represents a code block and contains the following keys:
            - 'index' (int): The index of the code block in the text.
            - 'file_name' (str): The name of the file extracted from the preceding line, if available.
            - 'content' (str): The content of the code block.
            - 'type' (str): The type of the code block. If the code block starts with a language specifier (like 'python' or 'java'), this field will contain that specifier. Otherwise, it will be set to 'language-specific'.
            - 'is_complete' (bool): True if the block has a closing tag, False otherwise.

        Note:
        The function assumes that the number of triple backticks in the text is even.
        If the number of triple backticks is odd, it will consider the rest of the text as the last code block.
        """        
        remaining = text
        bloc_index = 0
        first_index = 0
        indices = []
        while len(remaining) > 0:
            try:
                index = remaining.index("```")
                indices.append(index + first_index)
                remaining = remaining[index + 3:]
                first_index += index + 3
                bloc_index += 1
            except Exception as ex:
                if bloc_index % 2 == 1:
                    index = len(remaining)
                    indices.append(index)
                remaining = ""

        code_blocks = []
        is_start = True
        for index, code_delimiter_position in enumerate(indices):
            block_infos = {
                'index': index,
                'file_name': "",
                'section': "",
                'content': "",
                'type': "",
                'is_complete': False
            }
            if is_start:
                # Check the preceding line for file name
                preceding_text = text[:code_delimiter_position].strip().splitlines()
                if preceding_text:
                    last_line = preceding_text[-1].strip()
                    if last_line.startswith("<file_name>") and last_line.endswith("</file_name>"):
                        file_name = last_line[len("<file_name>"):-len("</file_name>")].strip()
                        block_infos['file_name'] = file_name
                    elif last_line.startswith("## filename:"):
                        file_name = last_line[len("## filename:"):].strip()
                        block_infos['file_name'] = file_name
                    if last_line.startswith("<section>") and last_line.endswith("</section>"):
                        section = last_line[len("<section>"):-len("</section>")].strip()
                        block_infos['section'] = section

                sub_text = text[code_delimiter_position + 3:]
                if len(sub_text) > 0:
                    try:
                        find_space = sub_text.index(" ")
                    except:
                        find_space = int(1e10)
                    try:
                        find_return = sub_text.index("\n")
                    except:
                        find_return = int(1e10)
                    next_index = min(find_return, find_space)
                    if '{' in sub_text[:next_index]:
                        next_index = 0
                    start_pos = next_index
                    if code_delimiter_position + 3 < len(text) and text[code_delimiter_position + 3] in ["\n", " ", "\t"]:
                        block_infos["type"] = 'language-specific'
                    else:
                        block_infos["type"] = sub_text[:next_index]

                    if index + 1 < len(indices):
                        next_pos = indices[index + 1] - code_delimiter_position
                        if next_pos - 3 < len(sub_text) and sub_text[next_pos - 3] == "`":
                            block_infos["content"] = sub_text[start_pos:next_pos - 3].strip()
                            block_infos["is_complete"] = True
                        else:
                            block_infos["content"] = sub_text[start_pos:next_pos].strip()
                            block_infos["is_complete"] = False
                    else:
                        block_infos["content"] = sub_text[start_pos:].strip()
                        block_infos["is_complete"] = False
                    code_blocks.append(block_infos)
                is_start = False
            else:
                is_start = True
                continue

        return code_blocks

    def extract_thinking_blocks(self, text: str) -> List[str]:
        """
        Extracts content between <thinking> or <think> tags from a given text.
        
        Parameters:
        text (str): The text containing thinking blocks
        
        Returns:
        List[str]: List of extracted thinking contents
        """
        import re
        
        # Pattern to match both <thinking> and <think> blocks with matching tags
        pattern = r'<(thinking|think)>(.*?)</\1>'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        # Extract content from the second group (index 2) and clean
        thinking_blocks = [match.group(2).strip() for match in matches]
        
        return thinking_blocks

    def remove_thinking_blocks(self, text: str) -> str:
        """
        Removes thinking blocks (either <thinking> or <think>) from text including the tags.
        
        Parameters:
        text (str): The text containing thinking blocks
        
        Returns:
        str: Text with thinking blocks removed
        """
        import re
        
        # Pattern to remove both <thinking> and <think> blocks with matching tags
        pattern = r'<(thinking|think)>.*?</\1>'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Remove extra whitespace and normalize newlines
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text.strip())
        
        return cleaned_text

    def yes_no(
            self,
            question: str,
            context: str = "",
            max_answer_length: int = None,
            conditionning: str = "",
            return_explanation: bool = False,
            callback = None
        ) -> bool | dict:
        """
        Answers a yes/no question.

        Args:
            question (str): The yes/no question to answer.
            context (str, optional): Additional context to provide for the question.
            max_answer_length (int, optional): Maximum string length allowed for the response. Defaults to None.
            conditionning (str, optional): An optional system message to put at the beginning of the prompt.
            return_explanation (bool, optional): If True, returns a dictionary with the answer and explanation. Defaults to False.

        Returns:
            bool or dict: 
                - If return_explanation is False, returns a boolean (True for 'yes', False for 'no').
                - If return_explanation is True, returns a dictionary with the answer and explanation.
        """
        if not callback:
            callback=self.sink

        prompt = f"{conditionning}\nQuestion: {question}\nContext: {context}\n"
        
        template = """
        {
            "answer": true | false,
            "explanation": "Optional explanation if return_explanation is True"
        }
        """
        
        response = self.generate_text_code(
            prompt=prompt,
            template=template,
            language="json",
            code_tag_format="markdown",
            max_size=max_answer_length,
            callback=callback
        )
        
        try:
            parsed_response = json.loads(response)
            answer = parsed_response.get("answer", False)
            explanation = parsed_response.get("explanation", "")
            
            if return_explanation:
                return {"answer": answer, "explanation": explanation}
            else:
                return answer
        except json.JSONDecodeError:
            return False

    def multichoice_question(
            self, 
            question: str, 
            possible_answers: list, 
            context: str = "", 
            max_answer_length: int = None, 
            conditionning: str = "", 
            return_explanation: bool = False,
            callback = None
        ) -> dict:
        """
        Interprets a multi-choice question from a user's response. This function expects only one choice as true. 
        All other choices are considered false. If none are correct, returns -1.

        Args:
            question (str): The multi-choice question posed by the user.
            possible_answers (List[Any]): A list containing all valid options for the chosen value.
            context (str, optional): Additional context to provide for the question.
            max_answer_length (int, optional): Maximum string length allowed while interpreting the user's responses. Defaults to None.
            conditionning (str, optional): An optional system message to put at the beginning of the prompt.
            return_explanation (bool, optional): If True, returns a dictionary with the choice and explanation. Defaults to False.

        Returns:
            dict: 
                - If return_explanation is False, returns a JSON object with only the selected choice index.
                - If return_explanation is True, returns a JSON object with the selected choice index and an explanation.
                - Returns {"index": -1} if no match is found among the possible answers.
        """
        if not callback:
            callback=self.sink
        
        prompt = f"""
        {conditionning}\n
        QUESTION:\n{question}\n
        POSSIBLE ANSWERS:\n"""
        for i, answer in enumerate(possible_answers):
            prompt += f"{i}. {answer}\n"
        
        if context:
            prompt += f"\nADDITIONAL CONTEXT:\n{context}\n"
        
        prompt += "\nRespond with a JSON object containing:\n"
        if return_explanation:
            prompt += "{\"index\": (the selected answer index), \"explanation\": (reasoning for selection)}"
        else:
            prompt += "{\"index\": (the selected answer index)}"
        
        response = self.generate_text_code(prompt, language="json", max_size=max_answer_length, 
            accept_all_if_no_code_tags_is_present=True, return_full_generated_code=False, callback=callback)
        
        try:
            result = json.loads(response)
            if return_explanation:
                if "index" in result and isinstance(result["index"], int):
                    return result["index"], result["index"]
            else:
                if "index" in result and isinstance(result["index"], int):
                    return result["index"]
        except json.JSONDecodeError:
            if return_explanation:
                return -1, "failed to decide"
            else:
                return -1
            
    def multichoice_ranking(
            self, 
            question: str, 
            possible_answers: list, 
            context: str = "", 
            max_answer_length: int = 512, 
            conditionning: str = "", 
            return_explanation: bool = False,
            callback = None
        ) -> dict:
        """
        Ranks answers for a question from best to worst. Returns a JSON object containing the ranked order.

        Args:
            question (str): The question for which the answers are being ranked.
            possible_answers (List[Any]): A list of possible answers to rank.
            context (str, optional): Additional context to provide for the question.
            max_answer_length (int, optional): Maximum string length allowed for the response. Defaults to 50.
            conditionning (str, optional): An optional system message to put at the beginning of the prompt.
            return_explanation (bool, optional): If True, returns a dictionary with the ranked order and explanations. Defaults to False.

        Returns:
            dict: 
                - If return_explanation is False, returns a JSON object with only the ranked order.
                - If return_explanation is True, returns a JSON object with the ranked order and explanations.
        """
        if not callback:
            callback=self.sink
        
        prompt = f"""
        {conditionning}\n
        QUESTION:\n{question}\n
        POSSIBLE ANSWERS:\n"""
        for i, answer in enumerate(possible_answers):
            prompt += f"{i}. {answer}\n"
        
        if context:
            prompt += f"\nADDITIONAL CONTEXT:\n{context}\n"
        
        prompt += "\nRespond with a JSON object containing:\n"
        if return_explanation:
            prompt += "{\"ranking\": (list of indices ordered from best to worst), \"explanations\": (list of reasons for each ranking)}"
        else:
            prompt += "{\"ranking\": (list of indices ordered from best to worst)}"
        
        response = self.generate_text_code(prompt, language="json", return_full_generated_code=False, callback=callback)
        
        try:
            result = json.loads(response)
            if "ranking" in result and isinstance(result["ranking"], list):
                return result
        except json.JSONDecodeError:
            return {"ranking": []}
        
        
    def sequential_summarize(
                                self, 
                                text:str,
                                chunk_processing_prompt:str="Extract relevant information from the current text chunk and update the memory if needed.",
                                chunk_processing_output_format="markdown",
                                final_memory_processing_prompt="Create final summary using this memory.",
                                final_output_format="markdown",
                                ctx_size:int=None,
                                chunk_size:int=None,
                                bootstrap_chunk_size:int=None,
                                bootstrap_steps:int=None,
                                callback = None,
                                debug:bool= False):
        """
            This function processes a given text in chunks and generates a summary for each chunk.
            It then combines the summaries to create a final summary.

            Parameters:
            text (str): The input text to be summarized.
            chunk_processing_prompt (str, optional): The prompt used for processing each chunk. Defaults to "".
            chunk_processing_output_format (str, optional): The format of the output for each chunk. Defaults to "markdown".
            final_memory_processing_prompt (str, optional): The prompt used for processing the final memory. Defaults to "Create final summary using this memory.".
            final_output_format (str, optional): The format of the final output. Defaults to "markdown".
            ctx_size (int, optional): The size of the context. Defaults to None.
            chunk_size (int, optional): The size of each chunk. Defaults to None.
            callback (callable, optional): A function to be called after processing each chunk. Defaults to None.
            debug (bool, optional): A flag to enable debug mode. Defaults to False.

            Returns:
            The final summary in the specified format.
        """
        if ctx_size is None:
            ctx_size = self.ctx_size
        
        if chunk_size is None:
            chunk_size = ctx_size//4
        
        # Tokenize entire text
        all_tokens = self.tokenize(text)
        total_tokens = len(all_tokens)
        
        # Initialize memory and chunk index
        memory = ""
        start_token_idx = 0
        
        # Create static prompt template
        static_prompt_template = f"""{self.system_full_header}
You are a structured sequential text summary assistant that processes documents chunk by chunk, updating a memory of previously generated information at each step.

Your goal is to extract and combine relevant information from each text chunk with the existing memory, ensuring no key details are omitted or invented.

If requested, infer metadata like titles or authors from the content.

{self.user_full_header}
Update the memory by merging previous information with new details from this text chunk.
Only add information explicitly present in the chunk. Retain all relevant prior memory unless clarified or updated by the current chunk.

----
# Text chunk:
# Chunk number: {{chunk_id}}
----
```markdown
{{chunk}}
```

{{custom_prompt}}

Before updating, verify each requested detail:
1. Does the chunk explicitly mention the information?
2. Should prior memory be retained, updated, or clarified?

Include only confirmed details in the output.
Rewrite the full memory including the updates and keeping relevant data.
Do not discuss the information inside thememory, just put the relevant information without comments.

----
# Current document analysis memory:
----
```{chunk_processing_output_format}
{{memory}}
```
{self.ai_full_header}
""" 
        # Calculate static prompt tokens (with empty memory and chunk)
        chunk_id=0
        example_prompt = static_prompt_template.format(custom_prompt=chunk_processing_prompt if chunk_processing_prompt else '', memory="", chunk="", chunk_id=chunk_id)
        static_tokens = len(self.tokenize(example_prompt))
        
        # Process text in chunks
        while start_token_idx < total_tokens:
            # Calculate available tokens for chunk
            current_memory_tokens = len(self.tokenize(memory))
            available_tokens = ctx_size - static_tokens - current_memory_tokens
            
            if available_tokens <= 0:
                raise ValueError("Memory too large - consider reducing chunk size or increasing context window")
            
            # Get chunk tokens
            if bootstrap_chunk_size is not None and chunk_id < bootstrap_steps:
                end_token_idx = min(start_token_idx + bootstrap_chunk_size, total_tokens)
            else:                
                end_token_idx = min(start_token_idx + chunk_size, total_tokens)
            chunk_tokens = all_tokens[start_token_idx:end_token_idx]
            chunk = self.detokenize(chunk_tokens)
            chunk_id +=1
            
            # Generate memory update
            prompt = static_prompt_template.format(custom_prompt=chunk_processing_prompt if chunk_processing_prompt else '', memory=memory, chunk=chunk, chunk_id=chunk_id)
            if debug:
                ASCIIColors.yellow(f" ----- {chunk_id-1} ------")
                ASCIIColors.red(prompt)
            
            memory = self.generate_text(prompt, n_predict=ctx_size//4, streaming_callback=callback).strip()
            code = self.extract_code_blocks(memory)
            if code:
                memory=code[0]["content"]
                
            if debug:
                ASCIIColors.yellow(f" ----- OUT ------")
                ASCIIColors.yellow(memory)
                ASCIIColors.yellow(" ----- ------")
            # Move to next chunk
            start_token_idx = end_token_idx
        
        # Prepare final summary prompt
        final_prompt_template = f"""!@>system:
You are a memory summarizer assistant that helps users format their memory information into coherant text in a specific style or format.
{final_memory_processing_prompt}.
!@>user:
Here is my document analysis memory:
```{chunk_processing_output_format}
{memory}
```
The output must be put inside a {final_output_format} markdown tag.
The updated memory must be put in a {chunk_processing_output_format} markdown tag.
!@>assistant:
"""
        # Truncate memory if needed for final prompt
        example_final_prompt = final_prompt_template
        final_static_tokens = len(self.tokenize(example_final_prompt))
        available_final_tokens = ctx_size - final_static_tokens
        
        memory_tokens = self.tokenize(memory)
        if len(memory_tokens) > available_final_tokens:
            memory = self.detokenize(memory_tokens[:available_final_tokens])
        
        # Generate final summary
        final_prompt = final_prompt_template
        memory = self.generate_text(final_prompt, streaming_callback=callback)
        code = self.extract_code_blocks(memory)
        if code:
            memory=code[0]["content"]
        return memory

    def deep_analyze(
            self,
            query: str,
            text: str = None,
            files: list = None,
            search_prompt: str = "Extract information related to the query from the current text chunk and update the memory with new findings.",
            aggregation_prompt: str = None,
            output_format: str = "markdown",
            ctx_size: int = None,
            chunk_size: int = None,
            bootstrap_chunk_size: int = None,
            bootstrap_steps: int = None,
            callback=None,
            debug: bool = False
        ):
            """
            Searches for specific information related to a query in a long text or a list of files.
            Processes the input in chunks, updates a memory with relevant findings, and optionally aggregates them.

            Parameters:
            - query (str): The query to search for.
            - text (str, optional): The input text to search in. Defaults to None.
            - files (list, optional): List of file paths to search in. Defaults to None.
            - search_prompt (str, optional): Prompt for processing each chunk. Defaults to a standard extraction prompt.
            - aggregation_prompt (str, optional): Prompt for aggregating findings. Defaults to None.
            - output_format (str, optional): Output format. Defaults to "markdown".
            - ctx_size (int, optional): Context size for the model. Defaults to None (uses self.ctx_size).
            - chunk_size (int, optional): Size of each chunk. Defaults to None (ctx_size // 4). Smaller chunk sizes yield better results but is slower.
            - bootstrap_chunk_size (int, optional): Size for initial chunks. Defaults to None.
            - bootstrap_steps (int, optional): Number of initial chunks using bootstrap size. Defaults to None.
            - callback (callable, optional): Function called after each chunk. Defaults to None.
            - debug (bool, optional): Enable debug output. Defaults to False.

            Returns:
            - str: The search findings or aggregated output in the specified format.
            """
            # Set defaults
            if ctx_size is None:
                ctx_size = self.ctx_size
            if chunk_size is None:
                chunk_size = ctx_size // 4

            # Prepare input
            if files:
                all_texts = [(file, open(file, 'r', encoding='utf-8').read()) for file in files]
            elif text:
                all_texts = [("input_text", text)]
            else:
                raise ValueError("Either text or files must be provided.")

            # Initialize memory and chunk counter
            memory = ""
            chunk_id = 0

            # Define search prompt template using f-string and the provided search_prompt
            search_prompt_template = f"""{self.system_full_header}
You are a search assistant that processes documents chunk by chunk to find information related to a query, updating a memory of findings at each step.

Your goal is to extract and combine relevant information from each text chunk with the existing memory, ensuring no key details are omitted or invented.


----
# Chunk number: {{chunk_id}}
# Text chunk:
```markdown
{{chunk}}
```

Current findings memory:
```markdown
{{memory}}
```
{self.user_full_header}
Query: '{query}'
Task: {search_prompt}

Update the memory by adding new relevant information from this chunk. Retain all prior findings unless contradicted or updated. Only include explicitly relevant details.
Make sure to extrafct only information relevant to be able to answer the query of the user or at least gives important contextual information that can be completed to answer the user query.
{self.ai_full_header}
"""

            # Calculate static prompt tokens
            example_prompt = search_prompt_template.replace("{{chunk_id}}", "0")\
                                                .replace("{{memory}}", "")\
                                                .replace("{{chunk}}", "")
            static_tokens = len(self.tokenize(example_prompt))

            # Process each text (file or input)
            for file_name, file_text in all_texts:
                file_tokens = self.tokenize(file_text)
                start_token_idx = 0

                while start_token_idx < len(file_tokens):
                    # Calculate available tokens
                    current_memory_tokens = len(self.tokenize(memory))
                    available_tokens = ctx_size - static_tokens - current_memory_tokens
                    if available_tokens <= 0:
                        raise ValueError("Memory too large - consider reducing chunk size or increasing context window")

                    # Adjust chunk size
                    actual_chunk_size = (
                        min(bootstrap_chunk_size, available_tokens)
                        if bootstrap_chunk_size is not None and bootstrap_steps is not None and chunk_id < bootstrap_steps
                        else min(chunk_size, available_tokens)
                    )
                    
                    end_token_idx = min(start_token_idx + actual_chunk_size, len(file_tokens))
                    chunk_tokens = file_tokens[start_token_idx:end_token_idx]
                    chunk = self.detokenize(chunk_tokens)

                    # Generate updated memory
                    prompt = search_prompt_template.replace("{chunk_id}", str(chunk_id))\
                                                .replace("{memory}", memory)\
                                                .replace("{chunk}", chunk)
                    if debug:
                        print(f"----- Chunk {chunk_id} from {file_name} ------")
                        print(prompt)

                    output = self.generate_text(prompt, n_predict=ctx_size // 4, streaming_callback=callback).strip()
                    code = self.extract_code_blocks(output)
                    memory = code[0]["content"] if code else output

                    if debug:
                        print("----- Updated Memory ------")
                        print(memory)
                        print("---------------------------")

                    start_token_idx = end_token_idx
                    chunk_id += 1

            # Aggregate findings if requested
            if aggregation_prompt:
                final_prompt = f"""{self.system_full_header}
You are a search results aggregator.

{self.user_full_header}
{aggregation_prompt}

Collected findings:
```markdown
{memory}
```

Provide the final output in {output_format} format.
{self.ai_full_header}
"""
                final_output = self.generate_text(final_prompt, streaming_callback=callback)
                code = self.extract_code_blocks(final_output)
                return code[0]["content"] if code else final_output
            return memory
def error(self, content, duration:int=4, client_id=None, verbose:bool=True):
    ASCIIColors.error(content)



if __name__=="__main__":
    lc = LollmsClient("ollama", model_name="mistral-nemo:latest")
    #lc = LollmsClient("http://localhost:11434", model_name="mistral-nemo:latest", default_generation_mode=ELF_GENERATION_FORMAT.OLLAMA)
    #lc = LollmsClient(model_name="gpt-3.5-turbo-0125", default_generation_mode=ELF_GENERATION_FORMAT.OPENAI)
    print(lc.listModels())
    code = lc.generate_code("Build a simple json that containes name and age. put the output inside a json markdown tag")
    print(code)

    code ="""<thinking>
Hello world thinking!
How you doing?
    
</thinking>
This is no thinking

<think>
Hello world think!
How you doing?
    
</think>

"""
    print(lc.extract_thinking_blocks(code))
    print(lc.remove_thinking_blocks(code))
