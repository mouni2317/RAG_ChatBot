import os
from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
import uuid

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

        # Use the specified collection name
        return Chroma(persist_directory=self.persist_directory, embedding_function=embedding_model, collection_name=self.collection_name)

    def get_embeddings(self, query_text: str, k: int = 5) -> List[Document]:
        """Retrieve similar documents/embeddings from ChromaDB based on a query."""
        if embedding_model is None:
             print("Warning: Embedding model not loaded. Cannot perform similarity search.")
             return []
        try:
            return self.vector_store.similarity_search(query_text, k=k)
        except Exception as e:
            print(f"Error retrieving documents from ChromaDB: {e}")
            # Depending on desired behavior, you might re-raise or return empty
            return [] # Returning empty list on error for this service method

    def add_documents(self, documents: List[Document]):
        """Adds a list of LangChain Document objects to ChromaDB."""
        if not documents:
            print("No documents to add.")
            return
        try:
            # Using add_documents which expects a list of LangChain Document objects
            # This method will use the initialized embedding_function to create embeddings
            self.vector_store.add_documents(documents)
            print(f"Successfully added {len(documents)} documents to ChromaDB collection '{self.collection_name}'.")

        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            raise # Re-raise the exception after printing

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = [], ids: List[str] = []):
        """Adds a list of texts with optional metadata and ids to ChromaDB."""
        if not texts:
            print("No texts to add.")
            return
        try:
            # add_texts is useful when you already have text chunks and metadata
            # It can generate IDs if not provided
            self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids if ids else None)
            print(f"Successfully added {len(texts)} texts to ChromaDB collection '{self.collection_name}'.")
        except Exception as e:
            print(f"Error adding texts to ChromaDB: {e}")
            raise # Re-raise the exception after printing

    def delete_collection(self):
        """Deletes the current collection."""
        try:
            # ChromaDB's client should have a delete_collection method
            # This assumes vector_store has an underlying client with this capability or a direct method
            # Depending on your ChromaDB setup (in-memory, persistent, client-server), the exact method might vary.
            # LangChain's Chroma wrapper doesn't expose delete_collection directly on the vector store object in all versions.
            # You might need to access the underlying client or recreate the Chroma object pointing to a new collection.
            # For a simple persistent directory, deleting the directory is an option, but not recommended while app is running.
            # A safer way in a service might be to use the underlying client:
            
            # Example (might need adjustment based on actual Chroma client): 
            # self.vector_store._client.delete_collection(name=self.collection_name)
            
            # As a workaround or simpler approach for file-based Chroma: Re-initializing might clear it or use a new collection name
            # For now, let's add a placeholder and a print statement as direct deletion via vector_store object is not standard.
            print(f"Attempting to clear or delete data from ChromaDB collection: {self.collection_name}")
            # This method is often not directly available or reliable across all Chroma versions via the VectorStore base class.
            # A common pattern is to re-initialize or manage collections via the client directly.
            # Placeholder for future implementation based on specific Chroma client usage if needed.
            print("Note: Direct deletion of collection via LangChain VectorStore object may not be supported or reliable.")
            # If you need to clear the collection, consider restarting the app with a new persist_directory or manually clearing the directory.
            # Or use the underlying chromadb client methods directly if accessed.
            pass # Placeholder

        except Exception as e:
            print(f"Error deleting ChromaDB collection: {e}")
            raise # Re-raise the exception

    def get_collection(self):
         """Gets the current collection object."""
         # This might not be needed often, but useful if direct Chroma client operations are necessary
         return self.vector_store._collection # Accessing internal attribute, may break in future versions

import uuid # Import uuid for generating IDs if needed 