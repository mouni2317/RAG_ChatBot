from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def get_model(model_type: str, model_name: str):
    """
    Dynamically load models based on type and name.
    
    model_type: "embedding" | "llm"
    model_name: huggingface model name
    """
    if model_type == "embedding":
        return HuggingFaceEmbeddings(model_name=model_name)
    
    if model_type == "llm":
        hf_pipeline = pipeline("text-generation", model=model_name)
        return HuggingFacePipeline(pipeline=hf_pipeline)

    raise ValueError(f"Unsupported model type: {model_type}")
