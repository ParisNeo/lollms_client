import requests

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
        return response.text
    else:
        return {"status": False, "error": response.text}
