from .huggingface_model import HuggingFaceModelLoader
from .openai_model import OpenAIModelLoader

def get_model(model_type: str, model_name: str, provider: str):
    print("get_model() reached ✅")
    loader = None

    if provider == "huggingface":
        loader = HuggingFaceModelLoader()
    elif provider == "openai":
        loader = OpenAIModelLoader()
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if model_type == "embedding":
        return loader.load_embedding_model(model_name)
    elif model_type == "llm":
        return loader.load_llm_model(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
