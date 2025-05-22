def get_model(model_type: str, model_name: str, provider: str, bucket_name: str = None, model_path: str = None):
    """Factory method to load models either from HuggingFace or Google Cloud."""
    print("get_model() reached ✅")
    loader = None

    # Check the provider
    if provider == "huggingface":
       pass
    elif provider == "google_cloud":
        pass
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Load model based on type (embedding or llm)
    # if model_type == "embedding":
    #     return loader.load_embedding_model(model_name)
    # elif model_type == "llm":
    #     return loader.load_llm_model(model_name)
    # else:
    #     raise ValueError(f"Unsupported model type: {model_type}")
