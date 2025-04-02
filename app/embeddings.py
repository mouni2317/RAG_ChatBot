from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import config

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_faiss_index(documents):
    """Convert documents to vector embeddings and store in FAISS."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.create_documents(documents)

    # Store embeddings in FAISS
    vector_db = FAISS.from_documents(split_docs, embedding_model)
    vector_db.save_local(config.FAISS_INDEX_PATH)
    return vector_db

def load_faiss_index():
    """Load FAISS index from disk."""
    return FAISS.load_local(config.FAISS_INDEX_PATH, embedding_model) if os.path.exists(config.FAISS_INDEX_PATH) else None
