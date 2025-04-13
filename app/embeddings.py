from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import config
from .model_factory.factory import get_model
import json
from app.app_config import CONFIG 

embedding_model = get_model("embedding", "sentence-transformers/all-MiniLM-L6-v2", "huggingface")

#to load google cloud embedding model
# embedding_model = get_model(
#     "embedding", 
#     "your-model-name", 
#     "google_cloud", 
#     bucket_name="your-bucket-name", 
#     model_path="path/to/your/model"
# )

def create_faiss_index(documents):
    """Convert documents to vector embeddings and store in FAISS."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.create_documents(documents)

     # Get raw embeddings
    texts = [doc.page_content for doc in split_docs]
    embeddings = embedding_model.embed_documents(texts)

    # Save embeddings to file for inspection
    with open("embeddings_output.json", "w") as f:
        json.dump(
            [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)],
            f,
            indent=2
        )

    # Ensure index directory exists
    dir_path = os.path.dirname(CONFIG.FAISS_INDEX_PATH)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Store embeddings in FAISS
    vector_db = FAISS.from_documents(split_docs, embedding_model)
    vector_db.save_local(CONFIG.FAISS_INDEX_PATH)

    # return vector_db

def load_faiss_index():
    """Load FAISS index from disk."""
    if os.path.exists(CONFIG.FAISS_INDEX_PATH):
        return FAISS.load_local(CONFIG.FAISS_INDEX_PATH, embedding_model)
    return None