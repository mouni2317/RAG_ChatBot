from app.model_factory.factory import get_model
from app.DBServices.chroma_writer import ChromaWriter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import requests
from fastapi import HTTPException
from app.app_config import CONFIG

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {CONFIG.HUGGING_FACE_API_KEY}"}

class LLMService:
    def __init__(self, provider="huggingface", model_name=None):
        # You can change the default model here if needed
        self.model_name = None
        self.provider = None
        self.llm = None
        self.vector_db = self._load_vector_db()
        self.headers = {"Authorization": f"Bearer {CONFIG.HUGGING_FACE_API_KEY}"}

    def _load_model(self):
        print(f"Loading LLM from {self.provider} ✅")
        return get_model(model_type="llm", model_name=self.model_name, provider=self.provider)

    def _load_vector_db(self):
        """Load Chroma vector database."""
        print("Loading Chroma DB... ✅")
        # Initialize ChromaWriter with appropriate parameters
        return ChromaWriter(persist_directory="./chroma_data", embedding_function=get_model("embedding", "sentence-transformers/all-MiniLM-L6-v2", "huggingface"))

    def generate_response(self, query: str) -> str:
        """Generate response using the LLM with retrieved context from Chroma DB."""
        if not self.vector_db:
            raise ValueError("Vector database (Chroma) is not loaded.")

        # Perform a similarity search on the Chroma DB
        print(f"Querying Chroma DB with: {query} ✅")
        docs = self.vector_db.get_embeddings(query, k=5)  # Adjust `k` as needed for more docs

        # Combine the retrieved documents and the query into a prompt for the LLM
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # Generate response using the LLM
        #response = self.llm(prompt)
        response = self.generate_response_remote(prompt);
        return response

    def generate_response_remote(self, query: str) -> str:
        """Generate response using the Hugging Face API."""
        API_URL = "https://api-inference.huggingface.co/models/gpt2"
        headers = {"Authorization": f"Bearer {CONFIG.HUGGING_FACE_API_KEY}"}

        payload = {
            "inputs": query,
            "parameters": {"max_length": 50}
        }

        response = requests.post(API_URL, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Error in Hugging Face API call")
        