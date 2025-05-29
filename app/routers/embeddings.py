from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from app.models import EmbeddingQuery, DocumentData
from app.services.chroma_service import ChromaService

# Initialize ChromaService (consider using dependency injection in a real app)
# Assuming a default collection name or configure as needed
chroma_service = ChromaService()

router = APIRouter()

@router.post("/get-embeddings/")
async def get_embeddings(query: EmbeddingQuery):
    """Retrieve similar documents/embeddings from ChromaDB based on a query."""
    try:
        # Use the ChromaService to get embeddings
        results = chroma_service.get_embeddings(query.query_text, k=query.k)

        return {
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in results
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/write-embeddings/")
async def write_embeddings(data: DocumentData):
    """Write documents/embeddings to ChromaDB."""
    try:
        # Use the ChromaService to write documents
        # ChromaService.add_texts is suitable for adding pre-chunked texts with metadata
        chroma_service.add_texts(texts=data.texts, metadatas=data.metadatas, ids=data.ids)

        return {"message": f"Successfully added {len(data.texts)} documents to ChromaDB"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 