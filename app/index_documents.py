from embeddings import create_faiss_index

# Sample documents (load from files or databases)
docs = ["RAG is a powerful AI technique.", "LLMs work well with vector search.", "FAISS enables fast similarity search."]

# Create FAISS index
create_faiss_index(docs)

print("FAISS index updated!")
