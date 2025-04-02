from embeddings import load_faiss_index, create_faiss_index

class FAISSDatabase:
    def __init__(self):
        self.vector_db = load_faiss_index()
        if not self.vector_db:
            print("No FAISS index found. Please index documents first.")

    def search(self, query, k=2):
        """Retrieve top-k similar documents."""
        return self.vector_db.similarity_search(query, k=k) if self.vector_db else []

    def update_index(self, documents):
        """Recreate FAISS index from new documents."""
        self.vector_db = create_faiss_index(documents)
