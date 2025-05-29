from langchain.vectorstores import Chroma as LangChainChroma
from langchain.embeddings.base import Embeddings
from typing import Optional

class ChromaClientLC:
    def __init__(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        collection_name: str = "default",
    ):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name

        self.vectorstore = LangChainChroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )

    def add_documents(self, texts: list[str], metadatas: Optional[list[dict]] = None):
        """Ingest plain texts (and optional metadata) into Chroma DB."""
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 3):
        return self.vectorstore.similarity_search(query, k=k)

    def persist(self):
        self.vectorstore.persist()
