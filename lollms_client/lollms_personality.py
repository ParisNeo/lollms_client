from pathlib import Path
from lollms_client import LollmsClient
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_utilities import PromptReshaper
import pkg_resources
from typing import List, Optional, Union, Callable

from lollmsvectordb.vector_database import VectorDatabase
from lollmsvectordb.text_document_loader import TextDocumentsLoader
from lollmsvectordb.text_chunker import TextChunker

from PIL import Image
import importlib
import yaml
from ascii_colors import ASCIIColors


class LollmsPersonality:

    # Extra
    def __init__(
            self,
            lollmsClient:LollmsClient,  # Assuming the type for lollmsClient is defined elsewhere
            personality_work_dir: Union[str, Path],
            personality_config_dir: Union[str, Path],
            callback:  Callable[[str, MSG_TYPE, dict, list], bool],
            personality_package_path: Optional[Union[str, Path]] = None,
            author: Optional[str] = None,
            name: Optional[str] = None,
            user_name: Optional[str] = None,
            category: Optional[str] = None,
            category_desc: Optional[str] = None,
            language: Optional[str] = None,
            supported_languages: Optional[List[str]] = None,
            personality_description: Optional[str] = None,
            personality_conditioning: Optional[str] = None,
            welcome_message: Optional[str] = None,
            include_welcome_message_in_discussion: Optional[bool] = None,
            user_message_prefix: Optional[str] = None,
            ai_message_prefix: Optional[str] = None,
            anti_prompts: Optional[List[str]] = None,
            model_temperature: Optional[float] = 0.1,
            model_n_predicts: Optional[int] = 2048,
            model_top_k: Optional[int] = 50,
            model_top_p: Optional[float] = 0.95,
            model_repeat_penalty: Optional[float] = 1.3,
            model_repeat_last_n: Optional[int] = 40,
            logo: Optional[Image.Image] = None,
        ):
        """
        Initialize an LollmsPersonality instance.

        Parameters:
        personality_package_path (str or Path): The path to the folder containing the personality package.
        Other optional parameters are attributes of the personality that can be provided upon initialization.

        Raises:
        ValueError: If the provided path is not a folder or does not contain a config.yaml file.
        """
        self.bot_says = ""

        self.lollmsClient = lollmsClient
        self.personality_work_dir = Path(personality_work_dir)
        self.personality_config_dir =Path(personality_config_dir)

        self.callback = callback

        self.text_files = []
        self.image_files = []
        self.audio_files = []
        self.images_descriptions = []
        self.vectorizer = None

        # Whisper to transcribe audio
        self.whisper = None

        # First setup a default personality
        # Version
        self._version = pkg_resources.get_distribution('lollms_client').version

        # General information
        self.author: str = author if author is not None else "ParisNeo"
        self.name: str = name if name is not None else "lollms"
        self.user_name: str = user_name if user_name is not None else "user"
        self.category: str = category if category is not None else "General"
        self.category_desc: str = category_desc if category_desc is not None else "General purpose AI"
        self.language: str = language if language is not None else "english"
        self.supported_languages: List[str] = supported_languages if supported_languages is not None else []

        # Conditioning
        self.personality_description: str = personality_description if personality_description is not None else "This personality is a helpful and Kind AI ready to help you solve your problems"
        self.personality_conditioning: str = personality_conditioning if personality_conditioning is not None else "\n".join([
            "!@>system:",
            "lollms (Lord of LLMs) is a smart and helpful Assistant built by the computer geek ParisNeo.",
            "It is compatible with many bindings to LLM models such as llama, gpt4all, gptj, autogptq etc.",
            "It can discuss with humans and assist them on many subjects.",
            "It runs locally on your machine. No need to connect to the internet.",
            "It answers the questions with precise details",
            "Its performance depends on the underlying model size and training.",
            "Try to answer with as much details as you can",
            "Date: {{date}}",
        ])
        self.welcome_message: str =  welcome_message if welcome_message is not None else "Welcome! I am lollms (Lord of LLMs) A free and open assistant built by ParisNeo. What can I do for you today?"
        self.include_welcome_message_in_discussion: bool = include_welcome_message_in_discussion if include_welcome_message_in_discussion is not None else True
        self.user_message_prefix: str = user_message_prefix if user_message_prefix is not None else "!@>human: "
        self.link_text: str = "\n"
        self.ai_message_prefix: str = ai_message_prefix if ai_message_prefix is not None else "!@>lollms:"
        self.anti_prompts:list = anti_prompts if anti_prompts is not None else ["!@>"]

        # Extra
        self.dependencies: List[str] = []

        # Disclaimer
        self.disclaimer: str = ""
        self.help: str = ""
        self.commands: list = []

        # Default model parameters
        self.model_temperature: float = model_temperature # higher: more creative, lower more deterministic
        self.model_n_predicts: int = model_n_predicts # higher: generates many words, lower generates
        self.model_top_k: int = model_top_k
        self.model_top_p: float = model_top_p
        self.model_repeat_penalty: float = model_repeat_penalty
        self.model_repeat_last_n: int = model_repeat_last_n

        self.logo: Optional[Image.Image] = logo
        self.processor = None
        self.data = None

        if personality_package_path is None:
            self.config = {}
            self.assets_list = []
            self.personality_package_path = None

            self.conditionning_len = len(self.lollmsClient.tokenize(self.personality_conditioning+"\n"))
            self.welcome_len = len(self.lollmsClient.tokenize(self.ai_message_prefix+self.welcome_message+"\n"))
            self.preambule_len=self.conditionning_len
            if self.include_welcome_message_in_discussion:
                self.preambule = self.personality_conditioning+"\n"+self.ai_message_prefix+self.welcome_message+"\n"
                self.preambule_len+=self.welcome_len

            return
        else:
            self.personality_package_path = Path(personality_package_path)
            # Validate that the path exists
            if not self.personality_package_path.exists():
                raise ValueError(f"Could not find the personality package:{self.personality_package_path}")

            # Validate that the path format is OK with at least a config.yaml file present in the folder
            if not self.personality_package_path.is_dir():
                raise ValueError(f"Personality package path is not a folder:{self.personality_package_path}")

            self.personality_folder_name = self.personality_package_path.stem

            # Open and store the personality
            self.load_personality()
            self.personality_work_dir.mkdir(parents=True, exist_ok=True)
            self.personality_config_dir.mkdir(parents=True, exist_ok=True)

            self.conditionning_len = len(self.lollmsClient.tokenize(self.personality_conditioning+"\n"))
            self.welcome_len = len(self.lollmsClient.tokenize(self.ai_message_prefix+self.welcome_message+"\n"))
            self.preambule_len=self.conditionning_len
            if self.include_welcome_message_in_discussion:
                self.preambule = self.personality_conditioning+"\n"+self.ai_message_prefix+self.welcome_message+"\n"
                self.preambule_len+=self.welcome_len



    def load_personality(self, package_path=None):
        """
        Load personality parameters from a YAML configuration file.

        Args:
            package_path (str or Path): The path to the package directory.

        Raises:
            ValueError: If the configuration file does not exist.
        """
        if package_path is None:
            package_path = self.personality_package_path
        else:
            package_path = Path(package_path)

        # Verify that there is at least a configuration file
        config_file = package_path / "config.yaml"
        if not config_file.exists():
            raise ValueError(f"The provided folder {package_path} does not exist.")

        with open(config_file, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)

        secret_file = package_path / "secret.yaml"
        if secret_file.exists():
            with open(secret_file, "r", encoding='utf-8') as f:
                self._secret_cfg = yaml.safe_load(f)
        else:
            self._secret_cfg = None

        languages = package_path / "languages"

        if languages.exists():
            self._supported_languages = []
            for language in [l for l in languages.iterdir()]:
                self._supported_languages.append(language.stem)

            if self._selected_language is not None and self._selected_language in self._supported_languages:
                config_file = languages / (self._selected_language+".yaml")
                with open(config_file, "r", encoding='utf-8') as f:
                    config = yaml.safe_load(f)



        # Load parameters from the configuration file
        self._version = config.get("version", self._version)
        self._author = config.get("author", self._author)
        self._name = config.get("name", self._name)
        self._user_name = config.get("user_name", self._user_name)
        self._category_desc = config.get("category", self._category)
        self._language = config.get("language", self._language)

        self._ignore_discussion_documents_rag = config.get("ignore_discussion_documents_rag", self._ignore_discussion_documents_rag)


        self._personality_description = config.get("personality_description", self._personality_description)
        self._personality_conditioning = config.get("personality_conditioning", self._personality_conditioning)
        self._prompts_list = config.get("prompts_list", self._prompts_list)
        self._welcome_message = config.get("welcome_message", self._welcome_message)
        self._include_welcome_message_in_discussion = config.get("include_welcome_message_in_discussion", self._include_welcome_message_in_discussion)

        self._user_message_prefix = config.get("user_message_prefix", self._user_message_prefix)
        self._link_text = config.get("link_text", self._link_text)
        self._ai_message_prefix = config.get("ai_message_prefix", self._ai_message_prefix)
        self._dependencies = config.get("dependencies", self._dependencies)
        self._disclaimer = config.get("disclaimer", self._disclaimer)
        self._help = config.get("help", self._help)
        self._commands = config.get("commands", self._commands)
        self._model_temperature = config.get("model_temperature", self._model_temperature)
        self._model_top_k = config.get("model_top_k", self._model_top_k)
        self._model_top_p = config.get("model_top_p", self._model_top_p)
        self._model_repeat_penalty = config.get("model_repeat_penalty", self._model_repeat_penalty)
        self._model_repeat_last_n = config.get("model_repeat_last_n", self._model_repeat_last_n)

        # Script parameters (for example keys to connect to search engine or any other usage)
        self._processor_cfg = config.get("processor_cfg", self._processor_cfg)


        #set package path
        self.personality_package_path = package_path

        # Check for a logo file
        self.logo_path = self.personality_package_path / "assets" / "logo.png"
        if self.logo_path.is_file():
            self._logo = Image.open(self.logo_path)

        # Get the assets folder path
        self.assets_path = self.personality_package_path / "assets"
        # Get the scripts folder path
        self.scripts_path = self.personality_package_path / "scripts"
        # Get the languages folder path
        self.languages_path = self.personality_package_path / "languages"
        # Get the data folder path
        self.data_path = self.personality_package_path / "data"
        # Get the data folder path
        self.audio_path = self.personality_package_path / "audio"
        # Get the data folder path
        self.welcome_audio_path = self.personality_package_path / "welcome_audio"


        # If not exist recreate
        self.assets_path.mkdir(parents=True, exist_ok=True)

        # If not exist recreate
        self.scripts_path.mkdir(parents=True, exist_ok=True)

        # If not exist recreate
        self.audio_path.mkdir(parents=True, exist_ok=True)

        # samples
        self.audio_samples = [f for f in self.audio_path.iterdir()]

        # Verify if the persona has a data folder
        if self.data_path.exists():
            self.database_path = self.data_path / "db.sqlite"
            from lollmsvectordb.lollms_tokenizers.tiktoken_tokenizer import TikTokenTokenizer

            if self.config.rag_vectorizer == "semantic":
                from lollmsvectordb.lollms_vectorizers.semantic_vectorizer import SemanticVectorizer
                v = SemanticVectorizer()
            elif self.config.rag_vectorizer == "tfidf":
                from lollmsvectordb.lollms_vectorizers.tfidf_vectorizer import TFIDFVectorizer
                v = TFIDFVectorizer()
            elif self.config.rag_vectorizer == "openai":
                from lollmsvectordb.lollms_vectorizers.openai_vectorizer import OpenAIVectorizer
                v = OpenAIVectorizer(api_key=self.config.rag_vectorizer_openai_key)

            self.persona_data_vectorizer = VectorDatabase(self.database_path, v, TikTokenTokenizer(), self.config.rag_chunk_size, self.config.rag_overlap)

            files = [f for f in self.data_path.iterdir() if f.suffix.lower() in ['.asm', '.bat', '.c', '.cpp', '.cs', '.csproj', '.css',
                '.csv', '.docx', '.h', '.hh', '.hpp', '.html', '.inc', '.ini', '.java', '.js', '.json', '.log',
                '.lua', '.map', '.md', '.pas', '.pdf', '.php', '.pptx', '.ps1', '.py', '.rb', '.rtf', '.s', '.se', '.sh', '.sln',
                '.snippet', '.snippets', '.sql', '.sym', '.ts', '.txt', '.xlsx', '.xml', '.yaml', '.yml', '.msg'] ]
            dl = TextDocumentsLoader()

            for f in files:
                text = dl.read_file(f)
                self.persona_data_vectorizer.add_document(f.name, text, f)
                # data_vectorization_chunk_size: 512 # chunk size
                # data_vectorization_overlap_size: 128 # overlap between chunks size
                # data_vectorization_nb_chunks: 2 # number of chunks to use
            self.persona_data_vectorizer.build_index()

        else:
            self.persona_data_vectorizer = None
            self._data = None

        self.personality_output_folder = self.lollms_paths.personal_outputs_path/self.name
        self.personality_output_folder.mkdir(parents=True, exist_ok=True)


        if self.run_scripts:
            # Search for any processor code
            processor_file_name = "processor.py"
            self.processor_script_path = self.scripts_path / processor_file_name
            if self.processor_script_path.exists():
                module_name = processor_file_name[:-3]  # Remove the ".py" extension
                module_spec = importlib.util.spec_from_file_location(module_name, str(self.processor_script_path))
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                if hasattr(module, "Processor"):
                    self._processor = module.Processor(self, callback=self.callback)
                else:
                    self._processor = None
            else:
                self._processor = None
        # Get a list of all files in the assets folder
        contents = [str(file) for file in self.assets_path.iterdir() if file.is_file()]

        self._assets_list = contents
        return config

    
    def notify(self, notification):
        print(notification)        

    def generate(self, discussion:LollmsDiscussion, prompt:str, n_predict=1024, stream = False, callback=None):
        if callback is None:
            callback=self.callback
        if self.processor:
            self.processor.run_workflow(prompt, discussion.format_discussion(self.lollmsClient.ctx_size-n_predict), callback)
        else:
            discussion.add_message(self.user_message_prefix, prompt)
            discussion.add_message(self.ai_message_prefix, "")
            full_discussion = self.preambule + discussion.format_discussion(self.lollmsClient.ctx_size-n_predict-self.preambule_len)
            discussion.messages[-1].content = self.lollmsClient.generate(full_discussion, n_predict, stream=stream, streaming_callback=callback)

    def fast_gen(self, prompt: str, max_generation_size: int=None, placeholders: dict = {}, sacrifice: list = ["previous_discussion"], debug: bool  = False, callback=None, show_progress=False) -> str:
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
        if debug == False:
            debug = self.config.debug

        if max_generation_size is None:
            prompt_size = self.model.tokenize(prompt)
            max_generation_size = self.model.config.ctx_size - len(prompt_size)

        pr = PromptReshaper(prompt)
        prompt = pr.build(placeholders,
                        self.lollmsClient.tokenize,
                        self.lollmsClient.detokenize,
                        self.lollmsClient.ctx_size - max_generation_size,
                        sacrifice
                        )
        ntk = len(self.lollmsClient.tokenize(prompt))
        max_generation_size = min(self.lollmsClient.ctx_size - ntk, max_generation_size)
        # TODO : add show progress

        gen = self.lollmsClient.generate(prompt, max_generation_size, callback=callback).strip().replace("</s>", "").replace("<s>", "")
        if debug:
            self.print_prompt("prompt", prompt+gen)

        return gen

    def print_prompt(self, title, prompt):
        ASCIIColors.red("*-*-*-*-*-*-*-* ", end="")
        ASCIIColors.red(title, end="")
        ASCIIColors.red(" *-*-*-*-*-*-*-*")
        ASCIIColors.yellow(prompt)
        ASCIIColors.red(" *-*-*-*-*-*-*-*")