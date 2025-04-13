from app.embeddings import load_faiss_index
from app.model_factory.factory import get_model
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS

class LLMService:
    def __init__(self, provider="huggingface", model_name=None):
        # You can change the default model here if needed
        self.model_name = model_name or "tiiuae/falcon-7b-instruct"
        self.provider = provider
        self.llm = self._load_model()
        self.vector_db = self._load_vector_db()

    def _load_model(self):
        print(f"Loading LLM from {self.provider} ✅")
        return get_model(model_type="llm", model_name=self.model_name, provider=self.provider)

    def _load_vector_db(self):
        """Load FAISS vector database."""
        print("Loading FAISS index... ✅")
        return load_faiss_index()

    def generate_response(self, query: str) -> str:
        """Generate response using the LLM with retrieved context from FAISS."""
        if not self.vector_db:
            raise ValueError("Vector database (FAISS) is not loaded.")

        # Perform a similarity search on the FAISS index
        print(f"Querying FAISS with: {query} ✅")
        docs = self.vector_db.similarity_search(query, k=5)  # Adjust `k` as needed for more docs

        # Combine the retrieved documents and the query into a prompt for the LLM
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # Generate response using the LLM
        response = self.llm(prompt)
        return response
