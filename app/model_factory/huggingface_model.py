from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from .base import ModelLoader

class HuggingFaceModelLoader(ModelLoader):
    print("huggingface_model.py loaded ✅")
    def load_embedding_model(self, model_name: str):
        return HuggingFaceEmbeddings(model_name=model_name)

    # def load_llm_model(self, model_name: str):
    #     hf_pipeline = pipeline("text-generation", model=model_name)
    #     return HuggingFacePipeline(pipeline=hf_pipeline)
