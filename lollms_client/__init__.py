import requests
import json
def generate_text(host_address, prompt, model_name=None, personality=-1, n_predict=1024, stream=False, temperature=0.1, top_k=50, top_p=0.95, repeat_penalty=0.8, repeat_last_n=40, seed=None, n_threads=8):
    url = f"{host_address}/lollms_generate"

    data = {
        "prompt": prompt,
        "model_name": model_name,
        "personality": personality,
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


def listMountedPersonalities(host_address):
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

if __name__=="__main__":
    print(listMountedPersonalities("http://localhost:9600"))