from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

class HuggingFaceModelLoader:
    def load_embedding_model(self, model_name: str):
        """Load embedding model from Hugging Face"""
        return HuggingFaceEmbeddings(model_name=model_name)

    def load_llm_model(self, model_name: str):
        """Load LLM model from Hugging Face"""
        hf_pipeline = pipeline("text-generation", model=model_name)
        return hf_pipeline
