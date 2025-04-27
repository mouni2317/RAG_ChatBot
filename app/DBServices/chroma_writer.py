from langchain.embeddings.base import Embeddings
from typing import Optional
from app.DBServices.client_provider import ChromaClientLC

class ChromaWriter:
    def __init__(self, persist_directory: str, embedding_function: Embeddings, collection_name: str = "default"):
        self.client = ChromaClientLC(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name
        )

    def write(self, texts: list[str], metadatas: Optional[list[dict]] = None):
        """Add documents to the Chroma database."""
        self.client.add_documents(texts, metadatas)
        self.client.persist()

    def get_embeddings(self, query: str, k: int = 5):
        """Retrieve embeddings similar to the query."""
        return self.client.similarity_search(query, k=k)
    
    