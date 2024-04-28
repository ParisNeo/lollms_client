import requests
from ascii_colors import ASCIIColors
from lollms_client.lollms_types import MSG_TYPE
import json
from enum import Enum
import tiktoken

class ELF_GENERATION_FORMAT(Enum):
    LOLLMS = 0
    OPENAI = 1
    OLLAMA = 2
    LITELLM = 2

class ELF_COMPLETION_FORMAT(Enum):
    Instruct = 0
    Chat = 1

class LollmsClient():
    def __init__(
                    self, 
                    host_address=None,
                    model_name=None,
                    ctx_size=4096,
                    personality=-1, 
                    n_predict=1024,
                    min_n_predict=512, 
                    temperature=0.1, 
                    top_k=50, 
                    top_p=0.95, 
                    repeat_penalty=0.8, 
                    repeat_last_n=40, 
                    seed=None, 
                    n_threads=8, 
                    service_key:str="",
                    tokenizer=None,
                    default_generation_mode=ELF_GENERATION_FORMAT.LOLLMS
                ) -> None:
        import tiktoken

        self.host_address=host_address
        self.model_name = model_name
        self.ctx_size = ctx_size
        self.n_predict = n_predict
        self.min_n_predict = min_n_predict
        self.personality = personality
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.seed = seed
        self.n_threads = n_threads
        self.service_key = service_key
        self.default_generation_mode = default_generation_mode
        self.tokenizer = tiktoken.model.encoding_for_model("gpt-3.5-turbo-1106") if tokenizer is None else tokenizer


    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        tokens_list = self.tokenizer.encode(prompt)

        return tokens_list

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        text = self.tokenizer.decode(tokens_list)

        return text

    def generate(self, prompt, n_predict=None, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8, service_key:str="", streaming_callback=None):
        if self.default_generation_mode == ELF_GENERATION_FORMAT.LOLLMS:
            return self.lollms_generate(prompt, self.host_address, self.model_name, -1, n_predict, stream, temperature, top_k, top_p, repeat_penalty, repeat_last_n, seed, n_threads, service_key, streaming_callback)
        elif self.default_generation_mode == ELF_GENERATION_FORMAT.OPENAI:
            return self.openai_generate(prompt, self.host_address, self.model_name, -1, n_predict, stream, temperature, top_k, top_p, repeat_penalty, repeat_last_n, seed, n_threads, ELF_COMPLETION_FORMAT.Instruct, service_key, streaming_callback)
        elif self.default_generation_mode == ELF_GENERATION_FORMAT.OLLAMA:
            return self.ollama_generate(prompt, self.host_address, self.model_name, -1, n_predict, stream, temperature, top_k, top_p, repeat_penalty, repeat_last_n, seed, n_threads, ELF_COMPLETION_FORMAT.Instruct, service_key, streaming_callback)
        elif self.default_generation_mode == ELF_GENERATION_FORMAT.LITELLM:
            return self.litellm_generate(prompt, self.host_address, self.model_name, -1, n_predict, stream, temperature, top_k, top_p, repeat_penalty, repeat_last_n, seed, n_threads, ELF_COMPLETION_FORMAT.Instruct, service_key, streaming_callback)


    def generate_text(self, prompt, host_address=None, model_name=None, personality=None, n_predict=None, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8, service_key:str="", streaming_callback=None):
        self.lollms_generate(prompt, host_address, model_name, personality, n_predict, stream, temperature, top_k, top_p, repeat_penalty, repeat_last_n, seed, n_threads, service_key, streaming_callback)

    def lollms_generate(self, prompt, host_address=None, model_name=None, personality=None, n_predict=None, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8, service_key:str="", streaming_callback=None):
        # Set default values to instance variables if optional arguments are None
        host_address = host_address if host_address else self.host_address
        model_name = model_name if model_name else self.model_name
        n_predict = n_predict if n_predict else self.n_predict
        personality = personality if personality is not None else self.personality
        # Set temperature, top_k, top_p, repeat_penalty, repeat_last_n, seed, n_threads to the instance variables if they are not provided or None
        temperature = temperature if temperature is not None else self.temperature
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p
        repeat_penalty = repeat_penalty if repeat_penalty is not None else self.repeat_penalty
        repeat_last_n = repeat_last_n if repeat_last_n is not None else self.repeat_last_n
        seed = seed or self.seed  # Use the instance seed if not provided
        n_threads = n_threads if n_threads else self.n_threads


        url = f"{host_address}/lollms_generate"
        if service_key!="":
            headers = {
                'Content-Type': 'application/json;',
                'Authorization': f'Bearer {service_key}',
            }
        else:
            headers = {
                'Content-Type': 'application/json',
            }
        data = {
            "prompt": prompt,
            "model_name": self.model_name,
            "personality": self.personality,
            "n_predict": n_predict,
            "stream": stream,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "seed": seed,
            "n_threads": n_threads
        }

        response = requests.post(url, json=data, headers=headers, stream=stream)
        if not stream:
            if response.status_code == 200:
                try:
                    text = response.text.strip().replace('\"','"')
                    if text[0]=='"':
                        text = text[1:]
                    if text[-1]=='"':
                        text = text[:-1]
                    return text
                except Exception as ex:
                    return {"status": False, "error": str(ex)}
            else:
                return {"status": False, "error": response.text}
        else:
            text = ""
            if response.status_code==200:
                try:
                    for line in response.iter_lines():
                        chunk = line.decode("utf-8")
                        text += chunk
                        if streaming_callback:
                            streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK)
                    if text[0]=='"':
                        text = text[1:]
                    if text[-1]=='"':
                        text = text[:-1]
                    return text
                except Exception as ex:
                    return {"status": False, "error": str(ex)}
            else:
                return {"status": False, "error": response.text}

    def openai_generate(self, prompt, host_address=None, model_name=None, personality=None, n_predict=None, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8, completion_format:ELF_COMPLETION_FORMAT=ELF_COMPLETION_FORMAT.Instruct, service_key:str="", streaming_callback=None):
        # Set default values to instance variables if optional arguments are None
        host_address = host_address if host_address else self.host_address
        model_name = model_name if model_name else self.model_name
        n_predict = n_predict if n_predict else self.n_predict
        personality = personality if personality is not None else self.personality
        # Set temperature, top_k, top_p, repeat_penalty, repeat_last_n, seed, n_threads to the instance variables if they are not provided or None
        temperature = temperature if temperature is not None else self.temperature
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p
        repeat_penalty = repeat_penalty if repeat_penalty is not None else self.repeat_penalty
        repeat_last_n = repeat_last_n if repeat_last_n is not None else self.repeat_last_n
        seed = seed or self.seed  # Use the instance seed if not provided
        n_threads = n_threads if n_threads else self.n_threads

    
        if service_key!="":
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {service_key}',
            }
        else:
            headers = {
                'Content-Type': 'application/json',
            }

        if completion_format==ELF_COMPLETION_FORMAT.Instruct:
            data = {
                'model':model_name,#self.config.model_name,
                'prompt': prompt,
                "stream":True,
                "temperature": float(temperature),
                "max_tokens": n_predict
            }
            completion_format_path = "/v1/completions"
        elif completion_format==ELF_COMPLETION_FORMAT.Chat:
            data = {
                'model':model_name,
                'messages': [{
                    'role': "user",
                    'content': prompt
                }],
                "stream":True,
                "temperature": float(temperature),
                "max_tokens": n_predict
            }
            completion_format_path = "/v1/chat/completions"


        if host_address.endswith("/"):
            host_address = host_address[:-1]
        url = f'{host_address}{completion_format_path}'


        response = requests.post(url, json=data, headers=headers)

        if response.status_code==400:
            try:
                content = response.content.decode("utf8")
                content = json.loads(content)
                self.error(content["error"]["message"])
                return
            except:
                content = response.content.decode("utf8")
                content = json.loads(content)
                self.error(content["message"])
                return
        elif response.status_code==404:
            ASCIIColors.error(response.content.decode("utf-8", errors='ignore'))
        text = ""
        for line in response.iter_lines():
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                try:
                    json_data = json.loads(decoded[5:].strip())
                    if completion_format==ELF_COMPLETION_FORMAT.Chat:
                        try:
                            chunk = json_data["choices"][0]["delta"]["content"]
                        except:
                            chunk = ""
                    else:
                        chunk = json_data["choices"][0]["text"]
                    ## Process the JSON data here
                    text +=chunk
                    if streaming_callback:
                        if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
                except:
                    break
            else:
                if decoded.startswith("{"):
                    for line_ in response.iter_lines():
                        decoded += line_.decode("utf-8")
                    try:
                        json_data = json.loads(decoded)
                        if json_data["object"]=="error":
                            self.error(json_data["message"])
                            break
                    except:
                        self.error("Couldn't generate text, verify your key or model name")
                else:
                    text +=decoded
                    if streaming_callback:
                        if not streaming_callback(decoded, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
            return text

    def ollama_generate(self, prompt, host_address=None, model_name=None, personality=None, n_predict=None, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8, completion_format:ELF_COMPLETION_FORMAT=ELF_COMPLETION_FORMAT.Instruct, service_key:str="", streaming_callback=None):
        # Set default values to instance variables if optional arguments are None
        host_address = host_address if host_address else self.host_address
        model_name = model_name if model_name else self.model_name
        n_predict = n_predict if n_predict else self.n_predict
        personality = personality if personality is not None else self.personality
        # Set temperature, top_k, top_p, repeat_penalty, repeat_last_n, seed, n_threads to the instance variables if they are not provided or None
        temperature = temperature if temperature is not None else self.temperature
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p
        repeat_penalty = repeat_penalty if repeat_penalty is not None else self.repeat_penalty
        repeat_last_n = repeat_last_n if repeat_last_n is not None else self.repeat_last_n
        seed = seed or self.seed  # Use the instance seed if not provided
        n_threads = n_threads if n_threads else self.n_threads

        if service_key!="":
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {service_key}',
            }
        else:
            headers = {
                'Content-Type': 'application/json',
            }

        data = {
            'model':model_name,
            'prompt': prompt,
            "stream":True,
            "temperature": float(temperature),
            "max_tokens": n_predict
        }
        completion_format_path = "/api/generate"
        if host_address.endswith("/"):
            host_address = host_address[:-1]
        url = f'{host_address}{completion_format_path}'

        response = requests.post(url, json=data, headers=headers)

        if response.status_code==404:
            ASCIIColors.error(response.content.decode("utf-8", errors='ignore'))
        text = ""
        for line in response.iter_lines():
            decoded = line.decode("utf-8")
            json_data = json.loads(decoded)
            chunk = json_data["response"]
            ## Process the JSON data here
            text +=chunk
            if streaming_callback:
                if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                    break            
            return text

    def litellm_generate(self, prompt, host_address=None, model_name=None, personality=None, n_predict=None, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8, completion_format:ELF_COMPLETION_FORMAT=ELF_COMPLETION_FORMAT.Instruct, service_key:str="", streaming_callback=None):
        # Set default values to instance variables if optional arguments are None
        host_address = host_address if host_address else self.host_address
        model_name = model_name if model_name else self.model_name
        n_predict = n_predict if n_predict else self.n_predict
        personality = personality if personality is not None else self.personality
        # Set temperature, top_k, top_p, repeat_penalty, repeat_last_n, seed, n_threads to the instance variables if they are not provided or None
        temperature = temperature if temperature is not None else self.temperature
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p
        repeat_penalty = repeat_penalty if repeat_penalty is not None else self.repeat_penalty
        repeat_last_n = repeat_last_n if repeat_last_n is not None else self.repeat_last_n
        seed = seed or self.seed  # Use the instance seed if not provided
        n_threads = n_threads if n_threads else self.n_threads

        if service_key!="":
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {service_key}',
            }
        else:
            headers = {
                'Content-Type': 'application/json',
            }

        data = {
            'model':model_name,
            'prompt': prompt,
            "stream":True,
            "temperature": float(temperature),
            "max_tokens": n_predict
        }
        completion_format_path = "/api/generate"
        if host_address.endswith("/"):
            host_address = host_address[:-1]
        url = f'{host_address}{completion_format_path}'

        response = requests.post(url, json=data, headers=headers)

        if response.status_code==404:
            ASCIIColors.error(response.content.decode("utf-8", errors='ignore'))
        text = ""
        for line in response.iter_lines():
            decoded = line.decode("utf-8")
            if decoded.startswith("{"):
                json_data = json.loads(decoded)
                if "error" in json_data:
                    self.error(json_data["error"]["message"])
                    break
            else:
                text +=decoded
                if streaming_callback:
                    if not streaming_callback(decoded, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
       
            return text

####

    def listMountedPersonalities(self, host_address:str=None):
        host_address = host_address if host_address else self.host_address
        url = f"{host_address}/list_mounted_personalities"

        response = requests.get(url)

        if response.status_code == 200:
            try:
                text = json.loads(response.content.decode("utf-8"))
                return text
            except Exception as ex:
                return {"status": False, "error": str(ex)}
        else:
            return {"status": False, "error": response.text}

    def listModels(self, host_address:str=None):
        host_address = host_address if host_address else self.host_address
        url = f"{host_address}/list_models"

        response = requests.get(url)

        if response.status_code == 200:
            try:
                text = json.loads(response.content.decode("utf-8"))
                return text
            except Exception as ex:
                return {"status": False, "error": str(ex)}
        else:
            return {"status": False, "error": response.text}

if __name__=="__main__":
    lc = LollmsClient("http://localhost:9600")
    print(lc.listMountedPersonalities())
    print(lc.listModels())