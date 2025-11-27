@router.get("/list_models")
def list_models_endpoint():
    models = []

    # 1) Local models - ensure dict format
    local_files = list_local_models_endpoint()
    for model_name in local_files:
        models.append({
            "model_name": model_name,
            "display_name": model_name,
            "description": "(Local) Folder model" if not model_name.endswith(".safetensors") else "(Local) Local safetensors file",
            "owned_by": "local_user"
        })

    # 2) HF Public models - already dicts from HF_PUBLIC_MODELS
    for category, hf_models in HF_PUBLIC_MODELS.items():
        for model_info in hf_models:
            models.append({
                "model_name": model_info["model_name"],
                "display_name": model_info["display_name"],
                "description": f"({category}) {model_info['desc']}",
                "owned_by": "huggingface"
            })

    # 3) Gated models - same
    if state.config.get("hf_token"):
        for category, gated_models in HF_GATED_MODELS.items():
            for model_info in gated_models:
                models.append({
                    "model_name": model_info["model_name"],
                    "display_name": model_info["display_name"],
                    "description": f"({category}) {model_info['desc']}",
                    "owned_by": "huggingface"
                })

    # 4) Civitai models - ensure dict format
    for key, info in CIVITAI_MODELS.items():
        models.append({
            "model_name": key,
            "display_name": info["display_name"],
            "description": f"(Civitai) {info['description']}",
            "owned_by": info["owned_by"]
        })

    return models  # Plain list of dicts - JSON serializable
