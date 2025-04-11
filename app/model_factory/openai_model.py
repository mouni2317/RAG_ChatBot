from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from .base import ModelLoader

class OpenAIModelLoader(ModelLoader):
    def load_embedding_model(self, model_name: str):
        return OpenAIEmbeddings(model=model_name)

    def load_llm_model(self, model_name: str):
        return ChatOpenAI(model_name=model_name)
