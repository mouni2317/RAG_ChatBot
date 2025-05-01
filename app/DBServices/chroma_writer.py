from langchain.embeddings.base import Embeddings
from typing import Optional
from app.DBServices.client_provider import ChromaClientLC
from langchain.vectorstores import Chroma

class ChromaWriter:
    def __init__(self, persist_directory: str, embedding_function: Embeddings, collection_name: str = "default"):
        self.client = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name
        )

    def write(self, texts: list[str], ids: list[str], metadatas: Optional[list[dict]] = None):
        """Add documents to the Chroma database."""
        print(f"MetaDatas: {metadatas}")
        print(f"IDs: {ids}")
        self.client.add_texts(texts=texts, ids=ids, metadatas=metadatas)
        self.client.persist()

    def get_embeddings(self, query: str, k: int = 5):
        """Retrieve embeddings similar to the query."""
        return self.client.similarity_search(query, k=k)
    
    def clear(self):
        """Delete all documents from the Chroma vector store."""
        self.client.delete(ids = ['null'])  # Deletes all entries
        self.client.persist()
        print("🧹 Chroma DB cleared successfully.")
        