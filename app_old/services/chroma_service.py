import os
from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

# Path to local sentence transformer model
MODEL_DIR = "models/all-MiniLM-L6-v2"

# Initialize embedding model globally (or consider passing it in a real app)
# Make sure the model is downloaded and available at MODEL_DIR
try:
    embedding_model = SentenceTransformerEmbeddings(model_name=MODEL_DIR)
except Exception as e:
    print(f"Warning: Could not load embedding model from {MODEL_DIR}. Please ensure the model is downloaded. Error: {e}")
    # Fallback or raise error if model loading is critical
    embedding_model = None # Or a dummy embedding function

class ChromaService:
    def __init__(self, persist_directory: str = "./chroma_data", collection_name: str = "my_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initializes or loads the ChromaDB vector store."""
        print(f"Initializing ChromaDB with persist directory: {self.persist_directory} and collection: {self.collection_name}")
        # Ensure embedding_model is loaded before initializing Chroma
        if embedding_model is None:
             raise ValueError("Embedding model is not loaded. Cannot initialize ChromaDB.")
             
        return Chroma(persist_directory=self.persist_directory, embedding_function=embedding_model, collection_name=self.collection_name)

    def add_documents(self, documents: List[Document]):
        """Adds a list of LangChain Document objects to ChromaDB."""
        if not documents:
            print("No documents to add.")
            return
        try:
            ids = [str(uuid.uuid4()) for _ in documents] # Generate UUIDs if documents don't have them
            # Or use existing IDs if available in metadata/document object structure not covered by LangChain Document default
            # For simplicity, using UUIDs here. Adjust if your Document objects have intrinsic IDs.
            
            # LangChain's add_documents method automatically handles splitting and embedding if not already done
            # However, for clarity in the graph, we'll handle splitting and embedding in separate nodes.
            # If adding pre-split and pre-embedded documents, use add_embeddings or add_texts with metadatas and ids

            # If you want to add pre-split texts with metadata and potentially pre-generated IDs:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            # If your Document objects had IDs, you'd extract them here:
            # ids = [doc.metadata.get('id') or str(uuid.uuid4()) for doc in documents]
            # For now, let's stick to add_documents and assume it handles embedding with the provided embedding_function
            
            # Using add_documents which expects a list of LangChain Document objects
            # This method will use the initialized embedding_function to create embeddings
            self.vector_store.add_documents(documents)
            print(f"Successfully added {len(documents)} documents to ChromaDB collection '{self.collection_name}'.")

        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            raise # Re-raise the exception after printing

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieves relevant documents from ChromaDB based on a query."""
        if embedding_model is None:
             print("Warning: Embedding model not loaded. Cannot perform similarity search.")
             return []
        try:
            # Assuming similarity_search is available and works with the initialized vector_store
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error retrieving documents from ChromaDB: {e}")
            return []

    # Add other useful methods like delete_collection, get_collection, etc. as needed

import uuid # Import uuid for generating IDs if needed 