from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import config
from .model_factory.factory import get_model
import json

embedding_model = get_model("embedding", "sentence-transformers/all-MiniLM-L6-v2", "huggingface")


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

    # Store embeddings in FAISS
    # vector_db = FAISS.from_documents(split_docs, embedding_model)
    # vector_db.save_local(config.FAISS_INDEX_PATH)
    # return vector_db

def load_faiss_index():
    """Load FAISS index from disk."""
    return FAISS.load_local(config.FAISS_INDEX_PATH, embedding_model) if os.path.exists(config.FAISS_INDEX_PATH) else None
