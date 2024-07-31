from lollms_client.lollms_core import LollmsClient
from lollms_client.lollms_types import  SUMMARY_MODE, MSG_TYPE
from lollms_client.lollms_utilities import remove_text_from_string, PromptReshaper, process_ai_output
from typing import List, Callable, Dict, Any, Optional
from ascii_colors import ASCIIColors
from functools import partial
import json
import sys
from datetime import datetime
from lollmsvectordb.text_chunker import TextChunker

class TasksLibrary:
    def __init__(self, lollms:LollmsClient, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None) -> None:
        self.lollms = lollms
        self.callback = callback

    def print_prompt(self, title, prompt):
        ASCIIColors.red("*-*-*-*-*-*-*-* ", end="")
        ASCIIColors.red(title, end="")
        ASCIIColors.red(" *-*-*-*-*-*-*-*")
        ASCIIColors.yellow(prompt)
        ASCIIColors.red(" *-*-*-*-*-*-*-*")

    def setCallback(self, callback: Callable[[str, MSG_TYPE, dict, list], bool]):
        self.callback = callback

    def process(self, text:str, message_type:MSG_TYPE, callback=None, show_progress=False):
        if callback is None:
            callback = self.callback
        if text is None:
            return True
        if message_type==MSG_TYPE.MSG_TYPE_CHUNK:
            bot_says = self.bot_says + text
        elif  message_type==MSG_TYPE.MSG_TYPE_FULL:
            bot_says = text

        if show_progress:
            if self.nb_received_tokens==0:
                self.start_time = datetime.now()
            dt =(datetime.now() - self.start_time).seconds
            if dt==0:
                dt=1
            spd = self.nb_received_tokens/dt
            ASCIIColors.green(f"Received {self.nb_received_tokens} tokens (speed: {spd:.2f}t/s)              ",end="\r",flush=True)
            sys.stdout = sys.__stdout__
            sys.stdout.flush()
            self.nb_received_tokens+=1


        antiprompt = self.detect_antiprompt(bot_says)
        if antiprompt:
            self.bot_says = remove_text_from_string(bot_says,antiprompt)
            ASCIIColors.warning(f"\n{antiprompt} detected. Stopping generation")
            return False
        else:
            if callback:
                callback(text,message_type)
            self.bot_says = bot_says
            return True
    def generate(self, prompt, max_size, temperature = None, top_k = None, top_p=None, repeat_penalty=None, repeat_last_n=None, callback=None, debug=False, show_progress=False, stream= False ):
        ASCIIColors.info("Text generation started: Warming up")
        self.nb_received_tokens = 0
        self.bot_says = ""
        if debug:
            self.print_prompt("gen",prompt)

        bot_says = self.lollms.generate(
                                prompt,
                                max_size,
                                stream=stream,
                                streaming_callback=partial(self.process, callback=callback, show_progress=show_progress),
                                temperature= temperature if temperature is not None else self.lollms.temperature,
                                top_k= top_k if top_k is not None else self.lollms.top_k ,
                                top_p= top_p if top_p is not None else self.lollms.top_p ,
                                repeat_penalty= repeat_penalty if repeat_penalty is not None else self.lollms.repeat_penalty,
                                repeat_last_n= repeat_last_n if repeat_last_n is not None else self.lollms.repeat_last_n,
                                ).strip()
        return self.bot_says if stream else bot_says


    def fast_gen(
                    self, 
                    prompt: str, 
                    max_generation_size: int=None, 
                    placeholders: dict = {}, 
                    sacrifice: list = ["previous_discussion"], 
                    debug: bool  = False, 
                    callback=None, 
                    show_progress=False, 
                    temperature = None, 
                    top_k = None, 
                    top_p=None, 
                    repeat_penalty=None, 
                    repeat_last_n=None
                ) -> str:
        """
        Fast way to generate code

        This method takes in a prompt, maximum generation size, optional placeholders, sacrifice list, and debug flag.
        It reshapes the context before performing text generation by adjusting and cropping the number of tokens.

        Parameters:
        - prompt (str): The input prompt for text generation.
        - max_generation_size (int): The maximum number of tokens to generate.
        - placeholders (dict, optional): A dictionary of placeholders to be replaced in the prompt. Defaults to an empty dictionary.
        - sacrifice (list, optional): A list of placeholders to sacrifice if the window is bigger than the context size minus the number of tokens to generate. Defaults to ["previous_discussion"].
        - debug (bool, optional): Flag to enable/disable debug mode. Defaults to False.

        Returns:
        - str: The generated text after removing special tokens ("<s>" and "</s>") and stripping any leading/trailing whitespace.
        """
        if max_generation_size is None:
            prompt_size = self.lollms.tokenize(prompt)
            max_generation_size = self.lollms.ctx_size - len(prompt_size)

        pr = PromptReshaper(prompt)
        prompt = pr.build(placeholders,
                        self.lollms.tokenize,
                        self.lollms.detokenize,
                        self.lollms.ctx_size - max_generation_size,
                        sacrifice
                        )
        ntk = len(self.lollms.tokenize(prompt))
        max_generation_size = min(self.lollms.ctx_size - ntk, max_generation_size)
        # TODO : add show progress

        gen = self.generate(prompt, max_generation_size, temperature = temperature, top_k = top_k, top_p=top_p, repeat_penalty=repeat_penalty, repeat_last_n=repeat_last_n, callback=callback, show_progress=show_progress).strip().replace("</s>", "").replace("<s>", "")
        if debug:
            self.print_prompt("prompt", prompt+gen)

        return gen

    def generate_with_images(self, prompt, images, max_size, temperature = None, top_k = None, top_p=None, repeat_penalty=None, repeat_last_n=None, callback=None, debug=False, show_progress=False, stream=False ):
        ASCIIColors.info("Text generation started: Warming up")
        self.nb_received_tokens = 0
        self.bot_says = ""
        if debug:
            self.print_prompt("gen",prompt)

        bot_says = self.lollms.generate_with_images(
                                prompt,
                                images,
                                max_size,
                                stream=stream,
                                streaming_callback= partial(self.process, callback=callback, show_progress=show_progress),
                                temperature=self.lollms.temperature if temperature is None else temperature,
                                top_k=self.lollms.top_k if top_k is None else top_k,
                                top_p=self.lollms.top_p if top_p is None else top_p,
                                repeat_penalty=self.lollms.repeat_penalty if repeat_penalty is None else repeat_penalty,
                                repeat_last_n = self.lollms.repeat_last_n if repeat_last_n is None else repeat_last_n
                                ).strip()
        return self.bot_says if stream else bot_says

    
    def fast_gen_with_images(self, prompt: str, images:list, max_generation_size: int=None, placeholders: dict = {}, sacrifice: list = ["previous_discussion"], debug: bool  = False, callback=None, show_progress=False) -> str:
        """
        Fast way to generate text from text and images

        This method takes in a prompt, maximum generation size, optional placeholders, sacrifice list, and debug flag.
        It reshapes the context before performing text generation by adjusting and cropping the number of tokens.

        Parameters:
        - prompt (str): The input prompt for text generation.
        - max_generation_size (int): The maximum number of tokens to generate.
        - placeholders (dict, optional): A dictionary of placeholders to be replaced in the prompt. Defaults to an empty dictionary.
        - sacrifice (list, optional): A list of placeholders to sacrifice if the window is bigger than the context size minus the number of tokens to generate. Defaults to ["previous_discussion"].
        - debug (bool, optional): Flag to enable/disable debug mode. Defaults to False.

        Returns:
        - str: The generated text after removing special tokens ("<s>" and "</s>") and stripping any leading/trailing whitespace.
        """
        prompt = "\n".join([
            "!@>system: I am an AI assistant that can converse and analyze images. When asked to locate something in an image you send, I will reply with:",
            "boundingbox(image_index, label, left, top, width, height)",
            "Where:",
            "image_index: 0-based index of the image",
            "label: brief description of what is located",
            "left, top: x,y coordinates of top-left box corner (0-1 scale)",
            "width, height: box dimensions as fraction of image size",
            "Coordinates have origin (0,0) at top-left, (1,1) at bottom-right.",
            "For other queries, I will respond conversationally to the best of my abilities.",
            prompt
        ])

        if max_generation_size is None:
            prompt_size = self.lollms.tokenize(prompt)
            max_generation_size = self.lollms.ctx_size - len(prompt_size)

        pr = PromptReshaper(prompt)
        prompt = pr.build(placeholders,
                        self.lollms.tokenize,
                        self.lollms.detokenize,
                        self.lollms.ctx_size - max_generation_size,
                        sacrifice
                        )
        ntk = len(self.lollms.tokenize(prompt))
        max_generation_size = min(self.lollms.ctx_size - ntk, max_generation_size)
        # TODO : add show progress

        gen = self.generate_with_images(prompt, images, max_generation_size, callback=callback, show_progress=show_progress).strip().replace("</s>", "").replace("<s>", "")
        try:
            gen = process_ai_output(gen, images, "/discussions/")
        except Exception as ex:
            pass
        if debug:
            self.print_prompt("prompt", prompt+gen)

        return gen    



    def step_start(self, step_text, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This triggers a step start

        Args:
            step_text (str): The step text
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the step start to. Defaults to None.
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(step_text, MSG_TYPE.MSG_TYPE_STEP_START)

    def step_end(self, step_text, status=True, callback: Callable[[str, int, dict, list], bool]=None):
        """This triggers a step end

        Args:
            step_text (str): The step text
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the step end to. Defaults to None.
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(step_text, MSG_TYPE.MSG_TYPE_STEP_END, {'status':status})

    def step(self, step_text, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This triggers a step information

        Args:
            step_text (str): The step text
            callback (callable, optional): A callable with this signature (str, MSG_TYPE, dict, list) to send the step to. Defaults to None.
            The callback has these fields:
            - chunk
            - Message Type : the type of message
            - Parameters (optional) : a dictionary of parameters
            - Metadata (optional) : a list of metadata
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(step_text, MSG_TYPE.MSG_TYPE_STEP)


    def sink(self, s=None,i=None,d=None):
        pass

    def build_prompt(self, prompt_parts:List[str], sacrifice_id:int=-1, context_size:int=None, minimum_spare_context_size:int=None):
        """
        Builds the prompt for code generation.

        Args:
            prompt_parts (List[str]): A list of strings representing the parts of the prompt.
            sacrifice_id (int, optional): The ID of the part to sacrifice.
            context_size (int, optional): The size of the context.
            minimum_spare_context_size (int, optional): The minimum spare context size.

        Returns:
            str: The built prompt.
        """
        if context_size is None:
            context_size = self.lollms.ctx_size
        if minimum_spare_context_size is None:
            minimum_spare_context_size = self.lollms.min_n_predict

        if sacrifice_id == -1 or len(prompt_parts[sacrifice_id])<50:
            return "\n".join([s for s in prompt_parts if s!=""])
        else:
            part_tokens=[]
            nb_tokens=0
            for i,part in enumerate(prompt_parts):
                tk = self.lollms.tokenize(part)
                part_tokens.append(tk)
                if i != sacrifice_id:
                    nb_tokens += len(tk)
            if len(part_tokens[sacrifice_id])>0:
                sacrifice_tk = part_tokens[sacrifice_id]
                sacrifice_tk= sacrifice_tk[-(context_size-nb_tokens-minimum_spare_context_size):]
                sacrifice_text = self.lollms.detokenize(sacrifice_tk)
            else:
                sacrifice_text = ""
            prompt_parts[sacrifice_id] = sacrifice_text
            return "\n".join([s for s in prompt_parts if s!=""])

    def translate_text_chunk(self, text_chunk, output_language:str="french", host_address:str=None, model_name: str = None, temperature=0.1, max_generation_size=3000):
        """
        This function translates a given text chunk into a specified language.

        Parameters:
        text_chunk (str): The text to be translated.
        output_language (str): The language into which the text should be translated. Defaults to 'french'.
        host_address (str): The address of the host where the translation model is located. Defaults to None.
        model_name (str): The name of the translation model to be used. Defaults to None.
        temperature (float): The temperature value for the translation model. This value affects the randomness of the translation. Defaults to 0.1.
        max_generation_size (int): The maximum length of the translated text. Defaults to 3000.

        Returns:
        str: The translated text.
        """        
        translated = self.lollms.generate_text(
                                "\n".join([
                                    f"!@>system:",
                                    f"Translate the following text to {output_language}.",
                                    "Be faithful to the original text and do not add or remove any information.",
                                    "Respond only with the translated text.",
                                    "Do not add comments or explanations.",
                                    f"!@>text to translate:",
                                    f"{text_chunk}",
                                    f"!@>translation:",
                                    ]),
                                    host_address,
                                    model_name,
                                    personality = -1,
                                    n_predict=max_generation_size,
                                    stream=False,
                                    temperature=temperature
                                    )
        return translated

    def extract_code_blocks(self, text: str) -> List[dict]:
        """
        This function extracts code blocks from a given text.

        Parameters:
        text (str): The text from which to extract code blocks. Code blocks are identified by triple backticks (```).

        Returns:
        List[dict]: A list of dictionaries where each dictionary represents a code block and contains the following keys:
            - 'index' (int): The index of the code block in the text.
            - 'file_name' (str): An empty string. This field is not used in the current implementation.
            - 'content' (str): The content of the code block.
            - 'type' (str): The type of the code block. If the code block starts with a language specifier (like 'python' or 'java'), this field will contain that specifier. Otherwise, it will be set to 'language-specific'.

        Note:
        The function assumes that the number of triple backticks in the text is even.
        If the number of triple backticks is odd, it will consider the rest of the text as the last code block.
        """        
        remaining = text
        bloc_index = 0
        first_index=0
        indices = []
        while len(remaining)>0:
            try:
                index = remaining.index("```")
                indices.append(index+first_index)
                remaining = remaining[index+3:]
                first_index += index+3
                bloc_index +=1
            except Exception as ex:
                if bloc_index%2==1:
                    index=len(remaining)
                    indices.append(index)
                remaining = ""

        code_blocks = []
        is_start = True
        for index, code_delimiter_position in enumerate(indices):
            block_infos = {
                'index':index,
                'file_name': "",
                'content': "",
                'type':""
            }
            if is_start:

                sub_text = text[code_delimiter_position+3:]
                if len(sub_text)>0:
                    try:
                        find_space = sub_text.index(" ")
                    except:
                        find_space = int(1e10)
                    try:
                        find_return = sub_text.index("\n")
                    except:
                        find_return = int(1e10)
                    next_index = min(find_return, find_space)
                    start_pos = next_index
                    if code_delimiter_position+3<len(text) and text[code_delimiter_position+3] in ["\n"," ","\t"] :
                        # No
                        block_infos["type"]='language-specific'
                    else:
                        block_infos["type"]=sub_text[:next_index]

                    next_pos = indices[index+1]-code_delimiter_position
                    if sub_text[next_pos-3]=="`":
                        block_infos["content"]=sub_text[start_pos:next_pos-3].strip()
                    else:
                        block_infos["content"]=sub_text[start_pos:next_pos].strip()
                    code_blocks.append(block_infos)
                is_start = False
            else:
                is_start = True
                continue

        return code_blocks

    def yes_no(self, question: str, context:str="", max_answer_length: int = 50, conditionning="") -> bool:
        """
        Analyzes the user prompt and answers whether it is asking to generate an image.

        Args:
            question (str): The user's message.
            max_answer_length (int, optional): The maximum length of the generated answer. Defaults to 50.
            conditionning: An optional system message to put at the beginning of the prompt
        Returns:
            bool: True if the user prompt is asking to generate an image, False otherwise.
        """
        return self.multichoice_question(question, ["no","yes"], context, max_answer_length, conditionning=conditionning)>0

    def multichoice_question(self, question: str, possible_answers:list, context:str = "", max_answer_length: int = 50, conditionning="") -> int:
        """
        Interprets a multi-choice question from a users response. This function expects only one choice as true. All other choices are considered false. If none are correct, returns -1.

        Args:
            question (str): The multi-choice question posed by the user.
            possible_ansers (List[Any]): A list containing all valid options for the chosen value. For each item in the list, either 'True', 'False', None or another callable should be passed which will serve as the truth test function when checking against the actual user input.
            max_answer_length (int, optional): Maximum string length allowed while interpreting the users' responses. Defaults to 50.
            conditionning: An optional system message to put at the beginning of the prompt

        Returns:
            int: Index of the selected option within the possible_ansers list. Or -1 if there was not match found among any of them.
        """
        choices = "\n".join([f"{i}. {possible_answer}" for i, possible_answer in enumerate(possible_answers)])
        elements = [conditionning] if conditionning!="" else []
        elements += [
                "!@>system:",
                "Answer this multi choices question.",
        ]
        if context!="":
            elements+=[
                       "!@>Context:",
                        f"{context}",
                    ]
        elements +=[
                "Answer with an id from the possible answers.",
                "Do not answer with an id outside this possible answers.",
                "Do not explain your reasons or add comments.",
                "the output should be an integer."
        ]
        elements += [
                f"!@>question: {question}",
                "!@>possible answers:",
                f"{choices}",
        ]
        elements += ["!@>answer:"]
        prompt = self.build_prompt(elements)

        gen = self.lollms.generate(prompt, max_answer_length, temperature=0.1, top_k=50, top_p=0.9, repeat_penalty=1.0, repeat_last_n=50, streaming_callback=self.sink).strip().replace("</s>","").replace("<s>","")
        if len(gen)>0:
            selection = gen.strip().split()[0].replace(",","").replace(".","")
            self.print_prompt("Multi choice selection",prompt+gen)
            try:
                return int(selection)
            except:
                ASCIIColors.cyan("Model failed to answer the question")
                return -1
        else:
            return -1


    def summerize_text(
                        self,
                        text,
                        summary_instruction="summerize",
                        doc_name="chunk",
                        answer_start="",
                        max_generation_size=3000,
                        max_summary_size=512,
                        callback=None,
                        chunk_summary_post_processing=None,
                        summary_mode=SUMMARY_MODE.SUMMARY_MODE_SEQUENCIAL
                    ):
        depth=0
        tk = self.lollms.tokenize(text)
        prev_len = len(tk)
        document_chunks=None
        while len(tk)>max_summary_size and (document_chunks is None or len(document_chunks)>1):
            self.step_start(f"Comprerssing {doc_name}... [depth {depth+1}]")
            chunk_size = int(self.lollms.ctx_size*0.6)
            document_chunks = TextChunker.chunk_text(text, self.lollms, chunk_size, 0, True)
            text = self.summerize_chunks(
                                            document_chunks,
                                            summary_instruction, 
                                            doc_name, 
                                            answer_start, 
                                            max_generation_size, 
                                            callback, 
                                            chunk_summary_post_processing=chunk_summary_post_processing,
                                            summary_mode=summary_mode)
            tk = self.lollms.tokenize(text)
            tk = self.lollms.tokenize(text)
            dtk_ln=prev_len-len(tk)
            prev_len = len(tk)
            self.step(f"Current text size : {prev_len}, max summary size : {max_summary_size}")
            self.step_end(f"Comprerssing {doc_name}... [depth {depth+1}]")
            depth += 1
            if dtk_ln<=10: # it is not sumlmarizing
                break
        return text

    def smart_data_extraction(
                                self,
                                text,
                                data_extraction_instruction="summerize",
                                final_task_instruction="reformulate with better wording",
                                doc_name="chunk",
                                answer_start="",
                                max_generation_size=3000,
                                max_summary_size=512,
                                callback=None,
                                chunk_summary_post_processing=None,
                                summary_mode=SUMMARY_MODE.SUMMARY_MODE_SEQUENCIAL
                            ):
        depth=0
        tk = self.lollms.tokenize(text)
        prev_len = len(tk)
        while len(tk)>max_summary_size:
            self.step_start(f"Comprerssing... [depth {depth+1}]")
            chunk_size = int(self.lollms.ctx_size*0.6)
            document_chunks = TextChunker.chunk_text(text, self.lollms, chunk_size, 0, True)
            text = self.summerize_chunks(
                                            document_chunks, 
                                            data_extraction_instruction, 
                                            doc_name, 
                                            answer_start, 
                                            max_generation_size, 
                                            callback, 
                                            chunk_summary_post_processing=chunk_summary_post_processing, 
                                            summary_mode=summary_mode
                                        )
            tk = self.lollms.tokenize(text)
            dtk_ln=prev_len-len(tk)
            prev_len = len(tk)
            self.step(f"Current text size : {prev_len}, max summary size : {max_summary_size}")
            self.step_end(f"Comprerssing... [depth {depth+1}]")
            depth += 1
            if dtk_ln<=10: # it is not sumlmarizing
                break
        self.step_start(f"Rewriting ...")
        text = self.summerize_chunks(
                                        [text],
                                        final_task_instruction, 
                                        doc_name, answer_start, 
                                        max_generation_size, 
                                        callback, 
                                        chunk_summary_post_processing=chunk_summary_post_processing
                                    )
        self.step_end(f"Rewriting ...")

        return text

    def summerize_chunks(
                            self,
                            chunks,
                            summary_instruction="summerize",
                            doc_name="chunk",
                            answer_start="",
                            max_generation_size=3000,
                            callback=None,
                            chunk_summary_post_processing=None,
                            summary_mode=SUMMARY_MODE.SUMMARY_MODE_SEQUENCIAL
                        ):
        if summary_mode==SUMMARY_MODE.SUMMARY_MODE_SEQUENCIAL:
            summary = ""
            for i, chunk in enumerate(chunks):
                self.step_start(f" Summary of {doc_name} - Processing chunk : {i+1}/{len(chunks)}")
                summary = f"{answer_start}"+ self.fast_gen(
                            "\n".join([
                                f"!@>Document_chunk: {doc_name}:",
                                f"{summary}",
                                f"{chunk}",
                                f"!@>instruction: {summary_instruction}",
                                f"Answer directly with the summary with no extra comments.",
                                f"!@>summary:",
                                f"{answer_start}"
                                ]),
                                max_generation_size=max_generation_size,
                                callback=callback)
                if chunk_summary_post_processing:
                    summary = chunk_summary_post_processing(summary)
                self.step_end(f" Summary of {doc_name} - Processing chunk : {i+1}/{len(chunks)}")
            return summary
        else:
            summeries = []
            for i, chunk in enumerate(chunks):
                self.step_start(f" Summary of {doc_name} - Processing chunk : {i+1}/{len(chunks)}")
                summary = f"{answer_start}"+ self.fast_gen(
                            "\n".join([
                                f"!@>Document_chunk: {doc_name}:",
                                f"{chunk}",
                                f"!@>instruction: {summary_instruction}",
                                f"Answer directly with the summary with no extra comments.",
                                f"!@>summary:",
                                f"{answer_start}"
                                ]),
                                max_generation_size=max_generation_size,
                                callback=callback)
                if chunk_summary_post_processing:
                    summary = chunk_summary_post_processing(summary)
                summeries.append(summary)
                self.step_end(f" Summary of {doc_name} - Processing chunk : {i+1}/{len(chunks)}")
            return "\n".join(summeries)

    #======================= Function calls
    def _upgrade_prompt_with_function_info(self, prompt: str, functions: List[Dict[str, Any]]) -> str:
        """
        Upgrades the prompt with information about function calls.

        Args:
            prompt (str): The original prompt.
            functions (List[Dict[str, Any]]): A list of dictionaries describing functions that can be called.

        Returns:
            str: The upgraded prompt that includes information about the function calls.
        """
        function_descriptions = ["!@>information: If you need to call a function to fulfull the user request, use a function markdown tag with the function call as the following json format:",
                                 "```function",
                                 "{",
                                 '"function_name":the name of the function to be called,',
                                 '"function_parameters": a list of  parameter values',
                                 "}",
                                 "```",
                                 "You can call multiple functions in one generation.",
                                 "Each function call needs to be in a separate function markdown tag.",
                                 "Do not add status of the execution as it will be added automatically by the system.",
                                 "If you want to get the output of the function before answering the user, then use the keyword @<NEXT>@ at the end of your message.",
                                 "!@>List of possible functions to be called:\n"]
        for function in functions:
            description = f"{function['function_name']}: {function['function_description']}\nparameters:{function['function_parameters']}"
            function_descriptions.append(description)

        # Combine the function descriptions with the original prompt.
        function_info = ' '.join(function_descriptions)
        upgraded_prompt = f"{function_info}\n{prompt}"

        return upgraded_prompt
    def extract_function_calls_as_json(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts function calls formatted as JSON inside markdown code blocks.

        Args:
            text (str): The generated text containing JSON markdown entries for function calls.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the function calls.
        """
        # Extract markdown code blocks that contain JSON.
        code_blocks = self.extract_code_blocks(text)

        # Filter out and parse JSON entries.
        function_calls = []
        for block in code_blocks:
            if block["type"]=="function":
                content = block.get("content", "")
                try:
                    # Attempt to parse the JSON content of the code block.
                    function_call = json.loads(content)
                    if type(function_call)==dict:
                        function_calls.append(function_call)
                    elif type(function_call)==list:
                        function_calls+=function_call
                except json.JSONDecodeError:
                    # If the content is not valid JSON, skip it.
                    continue

        return function_calls    
    def execute_function_calls(self, function_calls: List[Dict[str, Any]], function_definitions: List[Dict[str, Any]]) -> List[Any]:
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
        functions_dict = {func['function_name']: func['function'] for func in function_definitions}

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
    def generate_with_function_calls(self, prompt: str, functions: List[Dict[str, Any]], max_answer_length: Optional[int] = None, callback: Callable[[str,MSG_TYPE],bool]=None) -> List[Dict[str, Any]]:
        """
        Performs text generation with function calls.

        Args:
            prompt (str): The full prompt (including conditioning, user discussion, extra data, and the user prompt).
            functions (List[Dict[str, Any]]): A list of dictionaries describing functions that can be called.
            max_answer_length (int, optional): Maximum string length allowed for the generated text.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with the function names and parameters to execute.
        """
        # Upgrade the prompt with information about the function calls.
        upgraded_prompt = self._upgrade_prompt_with_function_info(prompt, functions)

        # Generate the initial text based on the upgraded prompt.
        generated_text = self.fast_gen(upgraded_prompt, max_answer_length, callback=callback)

        # Extract the function calls from the generated text.
        function_calls = self.extract_function_calls_as_json(generated_text)

        return generated_text, function_calls

    def generate_with_function_calls_and_images(self, prompt: str, images:list, functions: List[Dict[str, Any]], max_answer_length: Optional[int] = None, callback: Callable[[str,MSG_TYPE],bool]=None) -> List[Dict[str, Any]]:
        """
        Performs text generation with function calls.

        Args:
            prompt (str): The full prompt (including conditioning, user discussion, extra data, and the user prompt).
            functions (List[Dict[str, Any]]): A list of dictionaries describing functions that can be called.
            max_answer_length (int, optional): Maximum string length allowed for the generated text.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with the function names and parameters to execute.
        """
        # Upgrade the prompt with information about the function calls.
        upgraded_prompt = self._upgrade_prompt_with_function_info(prompt, functions)

        # Generate the initial text based on the upgraded prompt.
        generated_text = self.fast_gen_with_images(upgraded_prompt, images, max_answer_length, callback=callback)

        # Extract the function calls from the generated text.
        function_calls = self.extract_function_calls_as_json(generated_text)

        return generated_text, function_calls
