from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.document_processor import DocumentProcessor
from app.DBServices.db_write_service import DBWriteService  # Import DBWriteService from the appropriate module
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.WebCrawler import WebCrawlerManager, ConfluenceStrategy
from app.embeddings import create_faiss_index
from app.llmservice import LLMService 
from app.model_factory.factory import get_model

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

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str

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
async def create_embedding():
    """Create embeddings from uploaded document."""
    try:
        # Read the file content
        # contents = await file.read()
        # text = contents.decode("utf-8")  # Assuming it's text-based (TXT/CSV/JSON with text)
        # HTML Files to embedding / Read from DB layer
        # Create FAISS index from text
        # index = create_faiss_index([text])  # Wrap in list to match expected input
        db_service = DBWriteService(db_type="chroma")
        #Finalise the event Schema
        db_service.process_event({"title": "Web crawling initiated", "content": "Crawling in progress..."})
        # Query

        return {"message": f"Embeddings created for {results}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/webcrawl")
async def webcrawl():
    """Web crawling endpoint."""
    # Implement web crawling logic here
    # For now, just return a placeholder response
    # write to DB
    base_url = 'https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/overview'
    strategy = ConfluenceStrategy()
    manager = WebCrawlerManager(base_url, strategy, max_workers=10)
    manager.crawl()
    # write to DB
    #Refactor to handle by workers for event processing
    db_service = DBWriteService(db_type="mongo")
    db_service.process_event({"title": "Web crawling initiated", "content": "Crawling in progress..."})
    return {"message": "Web crawling initiated"}

@app.post("/chat", response_model=QueryResponse)
async def chat_rag(request: QueryRequest):
    """Handle user queries using RAG."""
    try:
        # Initialize the LLM service (you can customize the model name/provider here)
        llm_service = LLMService()  # Optionally, pass model name/provider for more flexibility
        response = llm_service.generate_response(request.question)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transfer-to-chroma")
async def transfer_to_chroma():
    """Fetch data from MongoDB and write to Chroma DB."""
    try:
        # Initialize MongoDB service
        mongo_service = DBWriteService(db_type="mongo")
        # Fetch data from MongoDB
        # Assuming a method 'fetch_all' exists in MongoWriter to retrieve all documents
        documents = mongo_service.db_writer.fetch_all() # Donot fetch unchanged events that is already vectorised 

        #Define the Schmea fo Chroma
        # Transform data for Chroma
        texts = [doc.get('jwt', '') for doc in documents]  # Assuming 'content' is the field to be indexed
        metadatas = [{'user_id': doc.get('user_id', '')} for doc in documents]  # Optional metadata

        # Initialize Chroma DB service
        chroma_service = DBWriteService(db_type="chroma")
        # Write to Chroma DB
        chroma_service.process_event({'texts': texts, 'metadatas': metadatas})
        chroma_service

        return {"message": "Data transferred to Chroma DB successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/get-embeddings")
async def get_embeddings(query: str):
    """Retrieve embeddings similar to the query."""
    try:
        # Initialize Chroma DB service
        chroma_service = DBWriteService(db_type="chroma")
        # Get embeddings
        results = chroma_service.db_writer.get_embeddings(query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))