from fastapi import APIRouter, HTTPException
from typing import Optional, List
import os

from app.models import QuestionRequest
from app.services.chroma_service import ChromaService
from langchain.schema import Document # Import Document for type hinting

# Initialize ChromaService (consider using dependency injection in a real app)
# Assuming the ChromaService is initialized with the correct persist_directory
chroma_service = ChromaService()

router = APIRouter()

@router.post("/ask-question/")
async def ask_question_endpoint(request: QuestionRequest):
    """
    Accepts a user question, performs similarity search using ChromaDB,
    writes the retrieved context and sources to 'data.txt', and returns a preview.
    """
    try:
        # Use the ChromaService to perform similarity search
        results: List[Document] = chroma_service.get_relevant_documents(request.query, k=request.k)

        if not results:
            return {"message": "No relevant documents found.", "answer": "No relevant documents found.", "preview": ""}

        # Combine retrieved document content
        combined_text = "\n\n---\n\n".join([doc.page_content for doc in results])

        # Extract source information from metadata
        sources = []
        for doc in results:
            source_info = []
            if doc.metadata.get('source_file'):
                source_info.append(f"File: {doc.metadata['source_file']}")
            if doc.metadata.get('title'):
                source_info.append(f"Title: {doc.metadata['title']}")
            if doc.metadata.get('url'):
                 source_info.append(f"URL: {doc.metadata['url']}")
            # Add other relevant metadata fields as needed

            if source_info:
                sources.append("; ".join(source_info))

        source_info_string = "\n".join(sources) if sources else "No source information available."

        # Format the full document content
        full_document = f"Query: {request.query}\n\nRetrieved Context:\n{combined_text}\n\nSources:\n{source_info_string}"

        # Define the output file path (relative to the current working directory)
        output_file_path = "./data.txt"

        # Save to the specified file
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(full_document)

        # Return a response with confirmation and preview
        return {
            "message": "Question processed and result written to data.txt.",
            "file_path": output_file_path,
            "preview": full_document[:500] + ("..." if len(full_document) > 500 else ""), # Optional: show first 500 chars
            "answer": "Context retrieved and saved. An LLM would typically generate an answer based on this context." # Indicate that this step is context retrieval, not LLM answer
        }

    except Exception as e:
        print(f"Error processing ask-question request: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}") 