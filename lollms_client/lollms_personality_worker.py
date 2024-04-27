from typing import List, Optional, Callable, Dict, Any
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_config import TypedConfig
from ascii_colors import trace_exception, ASCIIColors
import yaml
import json

class StateMachine:
    def __init__(self, states_dict):
        """
        states structure is the following
        [
            {
                "name": the state name,
                "commands": [ # list of commands
                    "command": function
                ],
                "default": default function
            }
        ]
        """
        self.states_dict = states_dict
        self.current_state_id = 0
        self.callback = None

    def goto_state(self, state):
        """
        Transition to the state with the given name or index.

        Args:
            state (str or int): The name or index of the state to transition to.

        Raises:
            ValueError: If no state is found with the given name or index.
        """
        if isinstance(state, str):
            for i, state_dict in enumerate(self.states_dict):
                if state_dict["name"] == state:
                    self.current_state_id = i
                    return
        elif isinstance(state, int):
            if 0 <= state < len(self.states_dict):
                self.current_state_id = state
                return
        raise ValueError(f"No state found with name or index: {state}")



    def process_state(self, command, full_context, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None, client=None):
        """
        Process the given command based on the current state.

        Args:
            command: The command to process.

        Raises:
            ValueError: If the current state doesn't have the command and no default function is defined.
        """
        if callback:
            self.callback=callback

        current_state = self.states_dict[self.current_state_id]
        commands = current_state["commands"]
        command = command.strip()

        for cmd, func in commands.items():
            if cmd == command[0:len(cmd)]:
                try:
                    func(command, full_context,client)
                except:# retrocompatibility
                    func(command, full_context)
                return

        default_func = current_state.get("default")
        if default_func is not None:
            default_func(command, full_context)
        else:
            raise ValueError(f"Command '{command}' not found in current state and no default function defined.")


class LollmsPersonalityWorker(StateMachine):
    """
    Template class for implementing personality processor classes in the APScript framework.

    This class provides a basic structure and placeholder methods for processing model inputs and outputs.
    Personality-specific processor classes should inherit from this class and override the necessary methods.
    """
    def __init__(
                    self,
                    personality         :LollmsPersonality,
                    personality_config  :TypedConfig,
                    states_dict         :dict   = {},
                    callback            = None
                ) -> None:
        super().__init__(states_dict)
        self.notify                             = personality.notify

        self.personality                        = personality
        self.personality_config                 = personality_config
        self.configuration_file_path            = self.personality.personality_config_dir/f"config.yaml"

        self.personality_config.config.file_path    = self.configuration_file_path

        self.callback = callback

        # Installation
        if (not self.configuration_file_path.exists()):
            self.install()
            self.personality_config.config.save_config()
        else:
            self.load_personality_config()

    def sink(self, s=None,i=None,d=None):
        pass

    def settings_updated(self):
        """
        To be implemented by the processor when the settings have changed
        """
        pass

    def mounted(self):
        """
        triggered when mounted
        """
        pass

    def selected(self):
        """
        triggered when mounted
        """
        pass

    def execute_command(self, command: str, parameters:list=[], client=None):
        """
        Recovers user commands and executes them. Each personality can define a set of commands that they can receive and execute
        Args:
            command: The command name
            parameters: A list of the command parameters

        """
        try:
            self.process_state(command, "", self.callback, client)
        except Exception as ex:
            trace_exception(ex)
            self.warning(f"Couldn't execute command {command}")

    async def handle_request(self, request) -> Dict[str, Any]:
        """
        Handle client requests.

        Args:
            data (dict): A dictionary containing the request data.

        Returns:
            dict: A dictionary containing the response, including at least a "status" key.

        This method should be implemented by a class that inherits from this one.

        Example usage:
        ```
        handler = YourHandlerClass()
        request_data = {"command": "some_command", "parameters": {...}}
        response = await handler.handle_request(request_data)
        ```
        """
        return {"status":True}


    def load_personality_config(self):
        """
        Load the content of local_config.yaml file.

        The function reads the content of the local_config.yaml file and returns it as a Python dictionary.

        Args:
            None

        Returns:
            dict: A dictionary containing the loaded data from the local_config.yaml file.
        """
        try:
            self.personality_config.config.load_config()
        except:
            self.personality_config.config.save_config()
        self.personality_config.sync()

    def install(self):
        """
        Installation procedure (to be implemented)
        """
        ASCIIColors.blue("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        ASCIIColors.red(f"Installing {self.personality.personality_folder_name}")
        ASCIIColors.blue("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")


    def uninstall(self):
        """
        Installation procedure (to be implemented)
        """
        ASCIIColors.blue("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        ASCIIColors.red(f"Uninstalling {self.personality.personality_folder_name}")
        ASCIIColors.blue("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")


    def add_file(self, path, client, callback=None, process=True):
        self.personality.add_file(path, client=client,callback=callback, process=process)
        if callback is not None:
            callback("File added successfully",MSG_TYPE.MSG_TYPE_INFO)
        return True

    def remove_file(self, path):
        if path in self.personality.text_files:
            self.personality.text_files.remove(path)
        elif path in self.personality.image_files:
            self.personality.image_files.remove(path)


    def load_config_file(self, path, default_config=None):
        """
        Load the content of local_config.yaml file.

        The function reads the content of the local_config.yaml file and returns it as a Python dictionary.
        If a default_config is provided, it fills any missing entries in the loaded dictionary.
        If at least one field from default configuration was not present in the loaded configuration, the updated
        configuration is saved.

        Args:
            path (str): The path to the local_config.yaml file.
            default_config (dict, optional): A dictionary with default values to fill missing entries.

        Returns:
            dict: A dictionary containing the loaded data from the local_config.yaml file, with missing entries filled
            by default_config if provided.
        """
        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        if default_config:
            updated = False
            for key, value in default_config.items():
                if key not in data:
                    data[key] = value
                    updated = True

            if updated:
                self.save_config_file(path, data)

        return data

    def save_config_file(self, path, data):
        """
        Save the configuration data to a local_config.yaml file.

        Args:
            path (str): The path to save the local_config.yaml file.
            data (dict): The configuration data to be saved.

        Returns:
            None
        """
        with open(path, 'w') as file:
            yaml.dump(data, file)

    def generate_with_images(self, prompt, images, max_size, temperature = None, top_k = None, top_p=None, repeat_penalty=None, repeat_last_n=None, callback=None, debug=False ):
        return self.personality.generate_with_images(prompt, images, max_size, temperature, top_k, top_p, repeat_penalty, repeat_last_n, callback, debug=debug)

    def generate(self, prompt, max_size, temperature = None, top_k = None, top_p=None, repeat_penalty=None, repeat_last_n=None, callback=None, debug=False ):
        return self.personality.generate(prompt, max_size, temperature, top_k, top_p, repeat_penalty, repeat_last_n, callback, debug=debug)

    from lollms.client_session import Client
    def run_workflow(self, prompt:str, previous_discussion_text:str="", callback: Callable[[str, MSG_TYPE, dict, list], bool]=None, context_details:dict=None, client:Client=None):
        """
        This function generates code based on the given parameters.

        Args:
            full_prompt (str): The full prompt for code generation.
            prompt (str): The prompt for code generation.
            context_details (dict): A dictionary containing the following context details for code generation:
                - conditionning (str): The conditioning information.
                - documentation (str): The documentation information.
                - knowledge (str): The knowledge information.
                - user_description (str): The user description information.
                - discussion_messages (str): The discussion messages information.
                - positive_boost (str): The positive boost information.
                - negative_boost (str): The negative boost information.
                - force_language (str): The force language information.
                - fun_mode (str): The fun mode conditionning text
                - ai_prefix (str): The AI prefix information.
            n_predict (int): The number of predictions to generate.
            client_id: The client ID for code generation.
            callback (function, optional): The callback function for code generation.

        Returns:
            None
        """

        return None


    # ================================================= Advanced methods ===========================================
    def compile_latex(self, file_path, pdf_latex_path=None):
        try:
            # Determine the pdflatex command based on the provided or default path
            if pdf_latex_path:
                pdflatex_command = pdf_latex_path
            else:
                pdflatex_command = self.personality.config.pdf_latex_path if self.personality.config.pdf_latex_path is not None else 'pdflatex'

            # Set the execution path to the folder containing the tmp_file
            execution_path = file_path.parent
            # Run the pdflatex command with the file path
            result = subprocess.run([pdflatex_command, "-interaction=nonstopmode", file_path], check=True, capture_output=True, text=True, cwd=execution_path)
            # Check the return code of the pdflatex command
            if result.returncode != 0:
                error_message = result.stderr.strip()
                return {"status":False,"error":error_message}

            # If the compilation is successful, you will get a PDF file
            pdf_file = file_path.with_suffix('.pdf')
            print(f"PDF file generated: {pdf_file}")
            return {"status":True,"file_path":pdf_file}

        except subprocess.CalledProcessError as e:
            print(f"Error occurred while compiling LaTeX: {e}")
            return {"status":False,"error":e}

    def find_numeric_value(self, text):
        pattern = r'\d+[.,]?\d*'
        match = re.search(pattern, text)
        if match:
            return float(match.group().replace(',', '.'))
        else:
            return None
    def remove_backticks(self, text):
        if text.startswith("```"):
            split_text = text.split("\n")
            text = "\n".join(split_text[1:])
        if text.endswith("```"):
            text= text[:-3]
        return text

    def search_duckduckgo(self, query: str, max_results: int = 10, instant_answers: bool = True, regular_search_queries: bool = True, get_webpage_content: bool = False) -> List[Dict[str, Union[str, None]]]:
        """
        Perform a search using the DuckDuckGo search engine and return the results as a list of dictionaries.

        Args:
            query (str): The search query to use in the search. This argument is required.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            instant_answers (bool, optional): Whether to include instant answers in the search results. Defaults to True.
            regular_search_queries (bool, optional): Whether to include regular search queries in the search results. Defaults to True.
            get_webpage_content (bool, optional): Whether to retrieve and include the website content for each result. Defaults to False.

        Returns:
            list[dict]: A list of dictionaries containing the search results. Each dictionary will contain 'title', 'body', and 'href' keys.

        Raises:
            ValueError: If neither instant_answers nor regular_search_queries is set to True.
        """
        if not PackageManager.check_package_installed("duckduckgo_search"):
            PackageManager.install_package("duckduckgo_search")
        from duckduckgo_search import DDGS
        if not (instant_answers or regular_search_queries):
            raise ValueError("One of ('instant_answers', 'regular_search_queries') must be True")

        query = query.strip("\"'")

        with DDGS() as ddgs:
            if instant_answers:
                answer_list = list(ddgs.answers(query))
                if answer_list:
                    answer_dict = answer_list[0]
                    answer_dict["title"] = query
                    answer_dict["body"] = next((item['Text'] for item in answer_dict['AbstractText']), None)
                    answer_dict["href"] = answer_dict.get('FirstURL', '')
            else:
                answer_list = []

            if regular_search_queries:
                results = ddgs.text(query, safe=False, result_type='link')
                for result in results[:max_results]:
                    title = result['Text'] or query
                    body = None
                    href = result['FirstURL'] or ''
                    answer_dict = {'title': title, 'body': body, 'href': href}
                    answer_list.append(answer_dict)

            if get_webpage_content:
                for i, result in enumerate(answer_list):
                    try:
                        response = requests.get(result['href'])
                        if response.status_code == 200:
                            content = response.text
                            answer_list[i]['body'] = content
                    except Exception as e:
                        print(f"Error retrieving webpage content for {result['href']}: {str(e)}")

            return answer_list


    def translate(self, text_chunk, output_language="french", max_generation_size=3000):
        translated = self.fast_gen(
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
                                    max_generation_size=max_generation_size, callback=self.sink)
        return translated

    def summerize_text(
                        self,
                        text,
                        summary_instruction="summerize",
                        doc_name="chunk",
                        answer_start="",
                        max_generation_size=3000,
                        max_summary_size=512,
                        callback=None,
                        chunk_summary_post_processing=None
                    ):
        depth=0
        tk = self.personality.model.tokenize(text)
        prev_len = len(tk)
        while len(tk)>max_summary_size:
            self.step_start(f"Comprerssing {doc_name}... [depth {depth+1}]")
            chunk_size = int(self.personality.config.ctx_size*0.6)
            document_chunks = DocumentDecomposer.decompose_document(text, chunk_size, 0, self.personality.model.tokenize, self.personality.model.detokenize, True)
            text = self.summerize_chunks(document_chunks,summary_instruction, doc_name, answer_start, max_generation_size, callback, chunk_summary_post_processing=chunk_summary_post_processing)
            tk = self.personality.model.tokenize(text)
            tk = self.personality.model.tokenize(text)
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
                                chunk_summary_post_processing=None
                            ):
        depth=0
        tk = self.personality.model.tokenize(text)
        prev_len = len(tk)
        while len(tk)>max_summary_size:
            self.step_start(f"Comprerssing... [depth {depth+1}]")
            chunk_size = int(self.personality.config.ctx_size*0.6)
            document_chunks = DocumentDecomposer.decompose_document(text, chunk_size, 0, self.personality.model.tokenize, self.personality.model.detokenize, True)
            text = self.summerize_chunks(document_chunks, data_extraction_instruction, doc_name, answer_start, max_generation_size, callback, chunk_summary_post_processing=chunk_summary_post_processing)
            tk = self.personality.model.tokenize(text)
            dtk_ln=prev_len-len(tk)
            prev_len = len(tk)
            self.step(f"Current text size : {prev_len}, max summary size : {max_summary_size}")
            self.step_end(f"Comprerssing... [depth {depth+1}]")
            depth += 1
            if dtk_ln<=10: # it is not sumlmarizing
                break
        self.step_start(f"Rewriting ...")
        text = self.summerize_chunks([text],
        final_task_instruction, doc_name, answer_start, max_generation_size, callback, chunk_summary_post_processing=chunk_summary_post_processing)
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
                            chunk_summary_post_processing=None
                        ):
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

    def sequencial_chunks_summary(
                            self,
                            chunks,
                            summary_instruction="summerize",
                            doc_name="chunk",
                            answer_start="",
                            max_generation_size=3000,
                            callback=None,
                            chunk_summary_post_processing=None
                        ):
        summeries = []
        for i, chunk in enumerate(chunks):
            if i<len(chunks)-1:
                chunk1 = chunks[i+1]
            else:
                chunk1=""
            if i>0:
                chunk=summary
            self.step_start(f" Summary of {doc_name} - Processing chunk : {i+1}/{len(chunks)}")
            summary = f"{answer_start}"+ self.fast_gen(
                        "\n".join([
                            f"!@>Document_chunk: {doc_name}:",
                            f"Block1:",
                            f"{chunk}",
                            f"Block2:",
                            f"{chunk1}",
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
            context_size = self.personality.config.ctx_size
        if minimum_spare_context_size is None:
            minimum_spare_context_size = self.personality.config.min_n_predict

        if sacrifice_id == -1 or len(prompt_parts[sacrifice_id])<50:
            return "\n".join([s for s in prompt_parts if s!=""])
        else:
            part_tokens=[]
            nb_tokens=0
            for i,part in enumerate(prompt_parts):
                tk = self.personality.model.tokenize(part)
                part_tokens.append(tk)
                if i != sacrifice_id:
                    nb_tokens += len(tk)
            if len(part_tokens[sacrifice_id])>0:
                sacrifice_tk = part_tokens[sacrifice_id]
                sacrifice_tk= sacrifice_tk[-(context_size-nb_tokens-minimum_spare_context_size):]
                sacrifice_text = self.personality.model.detokenize(sacrifice_tk)
            else:
                sacrifice_text = ""
            prompt_parts[sacrifice_id] = sacrifice_text
            return "\n".join([s for s in prompt_parts if s!=""])
    # ================================================= Sending commands to ui ===========================================
    def add_collapsible_entry(self, title, content, subtitle=""):
        return "\n".join(
        [
        f'<details class="flex w-full rounded-xl border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900 mb-3.5 max-w-full svelte-1escu1z" open="">',
        f'    <summary class="grid w-full select-none grid-cols-[40px,1fr] items-center gap-2.5 p-2 svelte-1escu1z">',
        f'        <dl class="leading-4">',
        f'          <dd class="text-sm"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-arrow-right">',
        f'          <line x1="5" y1="12" x2="19" y2="12"></line>',
        f'          <polyline points="12 5 19 12 12 19"></polyline>',
        f'          </svg>',
        f'          </dd>',
        f'        </dl>',
        f'        <dl class="leading-4">',
        f'        <dd class="text-sm"><h3>{title}</h3></dd>',
        f'        <dt class="flex items-center gap-1 truncate whitespace-nowrap text-[.82rem] text-gray-400">{subtitle}</dt>',
        f'        </dl>',
        f'    </summary>',
        f' <div class="content px-5 pb-5 pt-4">',
        content,
        f' </div>',
        f' </details>\n'
        ])

    def internet_search_with_vectorization(self, query, quick_search:bool=False ):
        """
        Do internet search and return the result
        """
        return self.personality.internet_search_with_vectorization(query, quick_search=quick_search)


    def vectorize_and_query(self, text, query, max_chunk_size=512, overlap_size=20, internet_vectorization_nb_chunks=3):
        vectorizer = TextVectorizer(VectorizationMethod.TFIDF_VECTORIZER, model = self.personality.model)
        decomposer = DocumentDecomposer()
        chunks = decomposer.decompose_document(text, max_chunk_size, overlap_size,self.personality.model.tokenize,self.personality.model.detokenize)
        for i, chunk in enumerate(chunks):
            vectorizer.add_document(f"chunk_{i}", self.personality.model.detokenize(chunk))
        vectorizer.index()
        docs, sorted_similarities, document_ids = vectorizer.recover_text(query, internet_vectorization_nb_chunks)
        return docs, sorted_similarities


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

    def exception(self, ex, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This sends exception to the client

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
            callback(str(ex), MSG_TYPE.MSG_TYPE_EXCEPTION)

    def warning(self, warning:str, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This sends exception to the client

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
            callback(warning, MSG_TYPE.MSG_TYPE_EXCEPTION)

    def info(self, info:str, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This sends exception to the client

        Args:
            inf (str): The information to be sent
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
            callback(info, MSG_TYPE.MSG_TYPE_INFO)

    def json(self, title:str, json_infos:dict, callback: Callable[[str, int, dict, list], bool]=None, indent=4):
        """This sends json data to front end

        Args:
            step_text (dict): The step text
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
            callback("", MSG_TYPE.MSG_TYPE_JSON_INFOS, metadata = [{"title":title, "content":json.dumps(json_infos, indent=indent)}])

    def ui(self, html_ui:str, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This sends ui elements to front end

        Args:
            step_text (dict): The step text
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
            callback(html_ui, MSG_TYPE.MSG_TYPE_UI)

    def code(self, code:str, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This sends code to front end

        Args:
            step_text (dict): The step text
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
            callback(code, MSG_TYPE.MSG_TYPE_CODE)

    def chunk(self, full_text:str, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This sends full text to front end

        Args:
            step_text (dict): The step text
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the text to. Defaults to None.
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(full_text, MSG_TYPE.MSG_TYPE_CHUNK)


    def full(self, full_text:str, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None, msg_type:MSG_TYPE = MSG_TYPE.MSG_TYPE_FULL):
        """This sends full text to front end

        Args:
            step_text (dict): The step text
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the text to. Defaults to None.
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(full_text, msg_type)

    def full_invisible_to_ai(self, full_text:str, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This sends full text to front end (INVISIBLE to AI)

        Args:
            step_text (dict): The step text
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the text to. Defaults to None.
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(full_text, MSG_TYPE.MSG_TYPE_FULL_INVISIBLE_TO_AI)

    def full_invisible_to_user(self, full_text:str, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This sends full text to front end (INVISIBLE to user)

        Args:
            step_text (dict): The step text
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the text to. Defaults to None.
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(full_text, MSG_TYPE.MSG_TYPE_FULL_INVISIBLE_TO_USER)




    def execute_python(self, code, code_folder=None, code_file_name=None):
        if code_folder is not None:
            code_folder = Path(code_folder)

        """Executes Python code and returns the output as JSON."""
        # Create a temporary file.
        root_folder = code_folder if code_folder is not None else self.personality.personality_output_folder
        root_folder.mkdir(parents=True,exist_ok=True)
        tmp_file = root_folder/(code_file_name if code_file_name is not None else f"ai_code.py")
        with open(tmp_file,"w") as f:
            f.write(code)

        # Execute the Python code in a temporary file.
        process = subprocess.Popen(
            ["python", str(tmp_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=root_folder
        )

        # Get the output and error from the process.
        output, error = process.communicate()

        # Check if the process was successful.
        if process.returncode != 0:
            # The child process threw an exception.
            error_message = f"Error executing Python code: {error.decode('utf8')}"
            return error_message

        # The child process was successful.
        return output.decode("utf8")

    def build_python_code(self, prompt, max_title_length=4096):
        if not PackageManager.check_package_installed("autopep8"):
            PackageManager.install_package("autopep8")
        import autopep8
        global_prompt = "\n".join([
            f"{prompt}",
            "!@>Extra conditions:",
            "- The code must be complete, not just snippets, and should be put inside a single python markdown code.",
            "-Preceive each python codeblock with a line using this syntax:",
            "$$file_name|the file path relative to the root folder of the project$$",
            "```python",
            "# Placeholder. Here you need to put the code for the file",
            "```",
            "!@>Code Builder:"
        ])
        code = self.fast_gen(global_prompt, max_title_length)
        code_blocks = self.extract_code_blocks(code)
        try:
            back_quote_index = code.index("```")  # Remove trailing backticks
            if back_quote_index>=0:
                # Removing any extra text
                code = code[:back_quote_index]
        except:
            pass
        formatted_code = autopep8.fix_code(code)  # Fix indentation errors
        return formatted_code


    def make_title(self, prompt, max_title_length: int = 50):
        """
        Generates a title for a given prompt.

        Args:
            prompt (str): The prompt for which a title needs to be generated.
            max_title_length (int, optional): The maximum length of the generated title. Defaults to 50.

        Returns:
            str: The generated title.
        """
        global_prompt = f"!@>instructions: Based on the provided prompt, suggest a concise and relevant title that captures the main topic or theme of the conversation. Only return the suggested title, without any additional text or explanation.\n!@>prompt: {prompt}\n!@>title:"
        title = self.fast_gen(global_prompt,max_title_length)
        return title


    def plan_with_images(self, request: str, images:list, actions_list:list=[LoLLMsAction], context:str = "", max_answer_length: int = 512) -> List[LoLLMsAction]:
        """
        creates a plan out of a request and a context

        Args:
            request (str): The request posed by the user.
            max_answer_length (int, optional): Maximum string length allowed while interpreting the users' responses. Defaults to 50.

        Returns:
            int: Index of the selected option within the possible_ansers list. Or -1 if there was not match found among any of them.
        """
        template = """!@>instruction:
Act as plan builder, a tool capable of making plans to perform the user requested operation.
"""
        if len(actions_list)>0:
            template +="""The plan builder is an AI that responds in json format. It should plan a succession of actions in order to reach the objective.
!@>list of action types information:
[
{{actions_list}}
]
The AI should respond in this format using data from actions_list:
{
    "actions": [
    {
        "name": name of the action 1,
        "parameters":[
            parameter name: parameter value
        ]
    },
    {
        "name": name of the action 2,
        "parameters":[
            parameter name: parameter value
        ]
    }
    ...
    ]
}
"""
        if context!="":
            template += """!@>Context:
{{context}}Ok
"""
        template +="""!@>request: {{request}}
"""
        template +="""!@>plan: To acheive the requested objective, this is the list of actions to follow, formatted as requested in json format:\n```json\n"""
        pr  = PromptReshaper(template)
        prompt = pr.build({
                "context":context,
                "request":request,
                "actions_list":",\n".join([f"{action}" for action in actions_list])
                },
                self.personality.model.tokenize,
                self.personality.model.detokenize,
                self.personality.model.config.ctx_size,
                ["previous_discussion"]
                )
        gen = self.generate_with_images(prompt, images, max_answer_length).strip().replace("</s>","").replace("<s>","")
        gen = self.remove_backticks(gen)
        self.print_prompt("full",prompt+gen)
        gen = fix_json(gen)
        return generate_actions(actions_list, gen)

    def plan(self, request: str, actions_list:list=[LoLLMsAction], context:str = "", max_answer_length: int = 512) -> List[LoLLMsAction]:
        """
        creates a plan out of a request and a context

        Args:
            request (str): The request posed by the user.
            max_answer_length (int, optional): Maximum string length allowed while interpreting the users' responses. Defaults to 50.

        Returns:
            int: Index of the selected option within the possible_ansers list. Or -1 if there was not match found among any of them.
        """
        template = """!@>instruction:
Act as plan builder, a tool capable of making plans to perform the user requested operation.
"""
        if len(actions_list)>0:
            template +="""The plan builder is an AI that responds in json format. It should plan a succession of actions in order to reach the objective.
!@>list of action types information:
[
{{actions_list}}
]
The AI should respond in this format using data from actions_list:
{
    "actions": [
    {
        "name": name of the action 1,
        "parameters":[
            parameter name: parameter value
        ]
    },
    {
        "name": name of the action 2,
        "parameters":[
            parameter name: parameter value
        ]
    }
    ...
    ]
}
"""
        if context!="":
            template += """!@>Context:
{{context}}Ok
"""
        template +="""!@>request: {{request}}
"""
        template +="""!@>plan: To acheive the requested objective, this is the list of actions to follow, formatted as requested in json format:\n```json\n"""
        pr  = PromptReshaper(template)
        prompt = pr.build({
                "context":context,
                "request":request,
                "actions_list":",\n".join([f"{action}" for action in actions_list])
                },
                self.personality.model.tokenize,
                self.personality.model.detokenize,
                self.personality.model.config.ctx_size,
                ["previous_discussion"]
                )
        gen = self.generate(prompt, max_answer_length).strip().replace("</s>","").replace("<s>","")
        gen = self.remove_backticks(gen).strip()
        if gen[-1]!="}":
            gen+="}"
        self.print_prompt("full",prompt+gen)
        gen = fix_json(gen)
        return generate_actions(actions_list, gen)


    def parse_directory_structure(self, structure):
        paths = []
        lines = structure.strip().split('\n')
        stack = []

        for line in lines:
            line = line.rstrip()
            level = (len(line) - len(line.lstrip())) // 4

            if '/' in line or line.endswith(':'):
                directory = line.strip(' ├─└│').rstrip(':').rstrip('/')

                while stack and level < stack[-1][0]:
                    stack.pop()

                stack.append((level, directory))
                path = '/'.join([dir for _, dir in stack]) + '/'
                paths.append(path)
            else:
                file = line.strip(' ├─└│')
                if stack:
                    path = '/'.join([dir for _, dir in stack]) + '/' + file
                    paths.append(path)

        return paths

    def extract_code_blocks(self, text: str) -> List[dict]:
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



    def build_and_execute_python_code(self,context, instructions, execution_function_signature, extra_imports=""):
        code = "```python\n"+self.fast_gen(
            self.build_prompt([
            "!@>context!:",
            context,
            f"!@>system:",
            f"{instructions}",
            f"Here is the signature of the function:\n{execution_function_signature}",
            "Don't call the function, just write it",
            "Do not provide usage example.",
            "The code must me without comments",
            f"!@>coder: Sure, in the following code, I import the necessary libraries, then define the function as you asked.",
            "The function is ready to be used in your code and performs the task as you asked:",
            "```python\n"
            ],2), callback=self.sink)
        code = code.replace("```python\n```python\n", "```python\n").replace("```\n```","```")
        code=self.extract_code_blocks(code)

        if len(code)>0:
            # Perform the search query
            code = code[0]["content"]
            code = "\n".join([
                        extra_imports,
                        code
                    ])
            ASCIIColors.magenta(code)
            module_name = 'custom_module'
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            exec(code, module.__dict__)
            return module, code


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

        gen = self.generate(prompt, max_answer_length, temperature=0.1, top_k=50, top_p=0.9, repeat_penalty=1.0, repeat_last_n=50, callback=self.sink).strip().replace("</s>","").replace("<s>","")
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

    def multichoice_ranking(self, question: str, possible_answers:list, context:str = "", max_answer_length: int = 50, conditionning="") -> int:
        """
        Ranks answers for a question from best to worst. returns a list of integers

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
                "!@>instructions:",
                "Answer this multi choices question.",
                "Answer with an id from the possible answers.",
                "Do not answer with an id outside this possible answers.",
                f"!@>question: {question}",
                "!@>possible answers:",
                f"{choices}",
        ]
        if context!="":
            elements+=[
                       "!@>Context:",
                        f"{context}",
                    ]

        elements += ["!@>answer:"]
        prompt = self.build_prompt(elements)

        gen = self.generate(prompt, max_answer_length, temperature=0.1, top_k=50, top_p=0.9, repeat_penalty=1.0, repeat_last_n=50).strip().replace("</s>","").replace("<s>","")
        self.print_prompt("Multi choice ranking",prompt+gen)
        if gen.index("]")>=0:
            try:
                ranks = eval(gen.split("]")[0]+"]")
                return ranks
            except:
                ASCIIColors.red("Model failed to rank inputs")
                return None
        else:
            ASCIIColors.red("Model failed to rank inputs")
            return None



    def build_html5_integration(self, html, ifram_name="unnamed"):
        """
        This function creates an HTML5 iframe with the given HTML content and iframe name.

        Args:
        html (str): The HTML content to be displayed in the iframe.
        ifram_name (str, optional): The name of the iframe. Defaults to "unnamed".

        Returns:
        str: The HTML string for the iframe.
        """
        return "\n".join(
            '<div style="width: 80%; margin: 0 auto;">',
            f'<iframe id="{ifram_name}" srcdoc="',
            html,
            '" style="width: 100%; height: 600px; border: none;"></iframe>',
            '</div>'
        )



    def info(self, info_text:str, callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This sends info text to front end

        Args:
            step_text (dict): The step text
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the info to. Defaults to None.
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(info_text, MSG_TYPE.MSG_TYPE_FULL)

    def step_progress(self, step_text:str, progress:float, callback: Callable[[str, MSG_TYPE, dict, list, LollmsPersonality], bool]=None):
        """This sends step rogress to front end

        Args:
            step_text (dict): The step progress in %
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the progress to. Defaults to None.
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(step_text, MSG_TYPE.MSG_TYPE_STEP_PROGRESS, {'progress':progress})

    def new_message(self, message_text:str, message_type:MSG_TYPE= MSG_TYPE.MSG_TYPE_FULL, metadata=[], callback: Callable[[str, int, dict, list, LollmsPersonality], bool]=None):
        """This sends step rogress to front end

        Args:
            step_text (dict): The step progress in %
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the progress to. Defaults to None.
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(message_text, MSG_TYPE.MSG_TYPE_NEW_MESSAGE, parameters={'type':message_type.value,'metadata':metadata},personality = self.personality)

    def finished_message(self, message_text:str="", callback: Callable[[str, MSG_TYPE, dict, list], bool]=None):
        """This sends step rogress to front end

        Args:
            step_text (dict): The step progress in %
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the progress to. Defaults to None.
        """
        if not callback and self.callback:
            callback = self.callback

        if callback:
            callback(message_text, MSG_TYPE.MSG_TYPE_FINISHED_MESSAGE)

    def print_prompt(self, title, prompt):
        ASCIIColors.red("*-*-*-*-*-*-*-* ", end="")
        ASCIIColors.red(title, end="")
        ASCIIColors.red(" *-*-*-*-*-*-*-*")
        ASCIIColors.yellow(prompt)
        ASCIIColors.red(" *-*-*-*-*-*-*-*")


    def fast_gen_with_images(self, prompt: str, images:list, max_generation_size: int= None, placeholders: dict = {}, sacrifice: list = ["previous_discussion"], debug: bool = False, callback=None, show_progress=False) -> str:
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
        return self.personality.fast_gen_with_images(prompt=prompt, images=images, max_generation_size=max_generation_size,placeholders=placeholders, sacrifice=sacrifice, debug=debug, callback=callback, show_progress=show_progress)

    def fast_gen(self, prompt: str, max_generation_size: int= None, placeholders: dict = {}, sacrifice: list = ["previous_discussion"], debug: bool = False, callback=None, show_progress=False) -> str:
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
        return self.personality.fast_gen(prompt=prompt,max_generation_size=max_generation_size,placeholders=placeholders, sacrifice=sacrifice, debug=debug, callback=callback, show_progress=show_progress)


    #Helper method to convert outputs path to url
    def path2url(file):
        file = str(file).replace("\\","/")
        pth = file.split('/')
        idx = pth.index("outputs")
        pth = "/".join(pth[idx:])
        file_path = f"![](/{pth})\n"
        return file_path

    def build_a_document_block(self, title="Title", link="", content="content"):
        if link!="":
            return f'''
<div style="width: 100%; border: 1px solid #ccc; border-radius: 5px; padding: 20px; font-family: Arial, sans-serif; margin-bottom: 20px; box-sizing: border-box;">
    <h3 style="margin-top: 0;">
        <a href="{link}" target="_blank" style="text-decoration: none; color: #333;">{title}</a>
    </h3>
    <pre style="white-space: pre-wrap;color: #666;">{content}</pre>
</div>
'''
        else:
            return f'''
<div style="width: 100%; border: 1px solid #ccc; border-radius: 5px; padding: 20px; font-family: Arial, sans-serif; margin-bottom: 20px; box-sizing: border-box;">
    <h3 style="margin-top: 0;">
        <p style="text-decoration: none; color: #333;">{title}</p>
    </h3>
    <pre style="white-space: pre-wrap;color: #666;">{content}</pre>
</div>
'''

    def build_a_folder_link(self, folder_path, link_text="Open Folder"):
        folder_path = str(folder_path).replace('\\','/')
        return '''
<a href="#" onclick="path=\''''+f'{folder_path}'+'''\';
fetch('/open_folder', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ path: path })
    })
    .then(response => response.json())
    .then(data => {
    if (data.status) {
        console.log('Folder opened successfully');
    } else {
        console.error('Error opening folder:', data.error);
    }
    })
    .catch(error => {
    console.error('Error:', error);
    });
">'''+f'''{link_text}</a>'''
    def build_a_file_link(self, file_path, link_text="Open Folder"):
        file_path = str(file_path).replace('\\','/')
        return '''
<a href="#" onclick="path=\''''+f'{file_path}'+'''\';
fetch('/open_file', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ path: path })
    })
    .then(response => response.json())
    .then(data => {
    if (data.status) {
        console.log('Folder opened successfully');
    } else {
        console.error('Error opening folder:', data.error);
    }
    })
    .catch(error => {
    console.error('Error:', error);
    });
">'''+f'''{link_text}</a>'''
# ===========================================================
    def compress_js(self, code):
        return compress_js(code)
    def compress_python(self, code):
        return compress_python(code)
    def compress_html(self, code):
        return compress_html(code)

# ===========================================================
    def select_model(self, binding_name, model_name):
        self.personality.app.select_model(binding_name, model_name)

class AIPersonalityInstaller:
    def __init__(self, personality:LollmsPersonality) -> None:
        self.personality = personality
