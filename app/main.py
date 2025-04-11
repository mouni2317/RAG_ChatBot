from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.document_processor import DocumentProcessor
from app.embeddings import create_faiss_index

app = FastAPI(title="RAG-based Chatbot", version="1.0")

# @app.post("/chat", response_model=QueryResponse)
# async def chat_rag(request: QueryRequest):
#     """Handle user queries using RAG."""
#     try:
#         response = generate_rag_response(request.question)
#         return QueryResponse(response=response)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
class DocumentRequest(BaseModel):
    path_or_url: str


@app.post("/upload")
async def upload_document(doc: DocumentRequest):
    """API Endpoint to handle file upload (TXT, JSON, CSV)."""
    processor = DocumentProcessor(doc.path_or_url)
    result = await processor.process()
    return {"message": "Document processed and stored", **result}

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/info")
async def get_info():
    """Basic server info endpoint."""
    return {"app_name": "RAG-based Chatbot", "version": "1.0"}

@app.post("/embedding")
async def create_embedding(file: UploadFile = File(...)):
    """Create embeddings from uploaded document."""
    try:
        # Read the file content
        contents = await file.read()
        text = contents.decode("utf-8")  # Assuming it's text-based (TXT/CSV/JSON with text)

        # Create FAISS index from text
        # index = create_faiss_index([text])  # Wrap in list to match expected input
        create_faiss_index([text])
        return {"message": f"Embeddings created for {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))