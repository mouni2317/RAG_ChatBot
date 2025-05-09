from app.model_factory.factory import get_model
from app.DBServices.chroma_writer import ChromaWriter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import requests
from fastapi import HTTPException
from app.app_config import CONFIG

API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
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
        docs = self.vector_db.get_embeddings(query, k=3)  # Retrieve top 3 documents

        # Combine the retrieved documents and the query into a prompt for the LLM
        context = "\n".join([doc.page_content[:500] for doc in docs])  # Limit context size
        prompt = (
            "You are an expert assistant. ONLY use the context below to answer the question. "
            "If the answer is not in the context, respond with \"I don't know.\"\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        print(f"Prompt for LLM: {prompt}")

        # Generate response using the Hugging Face API
        response = self.generate_response_remote(prompt)
        # Sample structure:
# response = [{"generated_text": "Line 1\nLine 2"}, {"generated_text": "Line 3"}]

        response_text=response

        if not response_text:
            raise ValueError("Empty or invalid respons or invalide from the model")

        return response_text

    def generate_response_remote(self, query: str) -> str:
        """Generate response using the Hugging Face API."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": 512,
            "model": "accounts/fireworks/models/deepseek-v3-0324"
        }

        response = requests.post(API_URL, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Error in Hugging Face API call")
