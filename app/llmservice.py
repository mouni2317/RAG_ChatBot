from app.model_factory.factory import get_model
from app.app_config import CONFIG
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

class LLMService:
    def __init__(self, provider="huggingface", model_name=None):
        self.model_name = model_name or "tiiuae/falcon-7b-instruct"
        self.provider = provider
        self.llm = self._load_model()
        self.query_engine = self._load_query_engine()

    def _load_model(self):
        print(f"Loading LLM from {self.provider} ✅")
        return get_model(model_type="llm", model_name=self.model_name, provider=self.provider)

    def _load_query_engine(self):
        print("Loading FAISS index and embedding model for retrieval ✅")
        embedding_model = HuggingFaceEmbeddings(model_name=CONFIG.EMBEDDING_MODEL_NAME)
        index = FAISS.load_local(CONFIG.FAISS_INDEX_PATH, embedding_model)
        retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)

    def generate_response(self, prompt: str) -> str:
        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(prompt)
        elif hasattr(self.llm, "__call__"):
            return self.llm(prompt)
        else:
            raise ValueError("Loaded LLM does not support invoking.")

    def rag_query(self, question: str) -> str:
        print(f"Performing RAG query for: {question}")
        return self.query_engine.run(question)
