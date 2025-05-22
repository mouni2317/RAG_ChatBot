
from sentence_transformers import SentenceTransformer
from .model_factory.factory import get_model
from app.app_config import CONFIG 

# embedding_model = get_model("embedding", "sentence-transformers/all-MiniLM-L6-v2", "huggingface")
LOCAL_MODEL_PATH = "models/all-MiniLM-L6-v2"  # Update this path as needed
embedding_model = SentenceTransformer(LOCAL_MODEL_PATH)

#to load google cloud embedding model
# embedding_model = get_model(
#     "embedding", 
#     "your-model-name", 
#     "google_cloud", 
#     bucket_name="your-bucket-name", 
#     model_path="path/to/your/model"
# )

# def create_faiss_index(documents):
#     """Convert documents to vector embeddings and store in FAISS."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) #Refactor to custom chunk_szie and chunk_overlap
#     split_docs = text_splitter.create_documents(documents)

#      # Get raw embeddings
#     texts = [doc.page_content for doc in split_docs]
#     embeddings = embedding_model.embed_documents(texts)

#     # Save embeddings to file for inspection
#     with open("embeddings_output.json", "w") as f:
#         json.dump(
#             [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)],
#             f,
#             indent=2
#         )

#     # Ensure index directory exists
#     dir_path = os.path.dirname(CONFIG.FAISS_INDEX_PATH)
#     if dir_path:
#         os.makedirs(dir_path, exist_ok=True)

#     # DB Write Service ot handle Local wrties or remote writes
#     # Store embeddings in FAISS
#     vector_db = FAISS.from_documents(split_docs, embedding_model)
#     vector_db.save_local(CONFIG.FAISS_INDEX_PATH)

#     # return vector_db -> Croma Neo4j FAISS

# def load_faiss_index():
#     """Load FAISS index from disk."""
#     #Config from cloud or local
#     if os.path.exists(CONFIG.FAISS_INDEX_PATH):
#         return FAISS.load_local(CONFIG.FAISS_INDEX_PATH, embedding_model)
#     return None