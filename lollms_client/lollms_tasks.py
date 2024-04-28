from lollms_client.lollms_core import LollmsClient
from typing import List
from ascii_colors import ASCIIColors

class TasksLibrary:
    def __init__(self, lollms:LollmsClient) -> None:
        self.lollms = lollms

    def print_prompt(self, title, prompt):
        ASCIIColors.red("*-*-*-*-*-*-*-* ", end="")
        ASCIIColors.red(title, end="")
        ASCIIColors.red(" *-*-*-*-*-*-*-*")
        ASCIIColors.yellow(prompt)
        ASCIIColors.red(" *-*-*-*-*-*-*-*")

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