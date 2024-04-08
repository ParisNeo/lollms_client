import requests
import json



class LollmsClient():
    def __init__(self, host_address=None, model_name=None, personality=-1, n_predict=1024, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8) -> None:
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


    def generate_text(self, prompt, host_address=None, model_name=None, personality=None, n_predict=None, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8):
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

        data = {
            "prompt": prompt,
            "model_name": self.model_name,
            "personality": self.personality,
            "n_predict": self.n_predict,
            "stream": stream,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "seed": seed,
            "n_threads": n_threads
        }

        response = requests.post(url, json=data)

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

    def generate_completion(self, prompt, host_address=None, model_name=None, personality=None, n_predict=None, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8):
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

        data = {
            "prompt": prompt,
            "model_name": model_name,
            "n_predict": n_predict,
            "stream": stream,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "seed": seed,
            "n_threads": n_threads
        }

        response = requests.post(url, json=data)

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