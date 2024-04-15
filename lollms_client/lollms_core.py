import requests
from ascii_colors import ASCIIColors
from lollms_client.lollms_types import MSG_TYPE
import json

elf_completion_formats={
    "openai instruct":"/v1/completions",
    "openai chat":"/v1/chat/completions",
    "vllm instruct":"/v1/completions",
    "vllm chat":"/v1/chat/completions",
    "ollama chat":"/api/generate",
    "litellm chat":"/chat/completions",
    "lollms":"/generate"
}

class LollmsClient():
    def __init__(self, host_address=None, model_name=None, personality=-1, n_predict=1024, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8, service_key:str="") -> None:
        self.host_address=host_address
        self.model_name = model_name
        self.n_predict = n_predict
        self.personality = personality
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.seed = seed
        self.n_threads = n_threads

        self.service_key = service_key


    def generate_text(self, prompt, host_address=None, model_name=None, personality=None, n_predict=None, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8, service_key:str="", streaming_callback=None):
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
                'Content-Type': 'application/json; charset=utf-8',
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


    def generate_completion(self, prompt, host_address=None, model_name=None, personality=None, n_predict=None, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8, completion_format="vllm instruct", service_key:str="", streaming_callback=None):
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

        if completion_format=="openai instruct":
            data = {
                'model':model_name,#self.config.model_name,
                'prompt': prompt,
                "stream":True,
                "temperature": float(temperature),
                "max_tokens": n_predict
            }
        elif completion_format=="openai chat":
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
        elif completion_format=="vllm instruct":
            data = {
                'model':model_name,
                'prompt': prompt,
                "stream":True,
                "temperature": float(temperature),
                "max_tokens": n_predict
            }
        elif completion_format=="vllm chat":
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
        elif completion_format=="ollama chat":
            data = {
                'model':model_name,
                'prompt': prompt,
                "stream":True,
                "temperature": float(temperature),
                "max_tokens": n_predict
            }
        elif completion_format=="litellm chat":
            data = {
                'model':model_name,
                'prompt': prompt,
                "stream":True,
                "temperature": float(temperature),
                "max_tokens": n_predict
            }

        if host_address.endswith("/"):
            host_address = host_address[:-1]
        url = f'{host_address}{elf_completion_formats[completion_format]}'


        response = requests.post(url, json=data, headers=headers)

        if response.status_code==400:
            if "openai" in completion_format:
                content = response.content.decode("utf8")
                content = json.loads(content)
                self.error(content["error"]["message"])
                return
            elif "vllm" in completion_format:
                content = response.content.decode("utf8")
                content = json.loads(content)
                self.error(content["message"])
                return
        elif response.status_code==404:
            ASCIIColors.error(response.content.decode("utf-8", errors='ignore'))
        text = ""
        for line in response.iter_lines():
            if completion_format=="litellm chat":
                text +=chunk
                if streaming_callback:
                    if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                        break            
            else:
                decoded = line.decode("utf-8")
                if completion_format=="ollama chat":
                    json_data = json.loads(decoded)
                    chunk = json_data["response"]
                    ## Process the JSON data here
                    text +=chunk
                    if streaming_callback:
                        if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                            break            
                else:
                    if decoded.startswith("data: "):
                        try:
                            json_data = json.loads(decoded[5:].strip())
                            if "chat" in completion_format:
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
    print(lc.listMountedPersonalities())