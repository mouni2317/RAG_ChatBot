class ModelLoader:
    def load_embedding_model(self, model_name: str):
        raise NotImplementedError

    def load_llm_model(self, model_name: str):
        raise NotImplementedError
