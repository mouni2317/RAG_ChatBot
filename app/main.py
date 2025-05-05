import random
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.document_processor import DocumentProcessor
from app.DBServices.db_write_service import DBWriteService  # Import DBWriteService from the appropriate module
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.WebCrawler import WebCrawlerManager, ConfluenceStrategy, AllLinksStrategy
from app.embeddings import create_faiss_index
from app.llmservice import LLMService 
from app.model_factory.factory import get_model
from langchain_community.vectorstores.utils import filter_complex_metadata
import uuid

app = FastAPI(title="RAG-based Chatbot", version="1.0")
app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "app", "static")), name="static")

RESTRICTED_KEYWORDS = ['abc', 'abc company']

class DocumentRequest(BaseModel):
    path_or_url: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str

class AskRequest(BaseModel):
    prompt: str

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

def is_prompt_restricted(prompt: str, banned_words: List[str]) -> bool:
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in banned_words)

llm_service = LLMService()

@app.post("/upload")
async def upload_document(doc: DocumentRequest):
    """API Endpoint to handle file upload (TXT, JSON, CSV)."""
    processor = DocumentProcessor(doc.path_or_url)
    result = await processor.process()
    return {"message": "Document processed and stored", **result}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file_path = os.path.join(os.getcwd(), "app", "static", "index.html")
    with open(index_file_path, "r") as f:
        return f.read()

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
    print(f"""Web crawling endpoint.""")
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

@app.get("/webcrawl-all-links")
async def webcrawl_all_links():
    """Web crawling endpoint using AllLinksStrategy."""
    try:
        # Define the base URL for crawling
        base_url = 'https://www.investopedia.com/articles/optioninvestor/02/091802.aspm' 

        # Use AllLinksStrategy for parsing links
        strategy = AllLinksStrategy()
        manager = WebCrawlerManager(base_url, strategy, max_workers=10)

        # Start crawling and collect scraped data
        scraped_data = manager.crawl()  # Assume this returns a list of scraped objects

        # Initialize DBWriteService
        db_service = DBWriteService(db_type="mongo")  # Change to "chroma" if using ChromaWriter

        # Process each scraped object
        for data in scraped_data:
            event = {
                "texts": [data.get("content", "")],  # Replace "content" with the actual field containing text
                "ids": [data.get("id", str(random.randint(1000, 9999)))],  # Generate a random ID if not present
                "metadatas": [{"url": data.get("url", "")}]  # Add metadata like URL
            }
            db_service.process_event(event)

        return {"message": "Web crawling using AllLinksStrategy completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(query: AskRequest):
    prompt = query.prompt

    # Check if the prompt contains restricted keywords
    if is_prompt_restricted(prompt, RESTRICTED_KEYWORDS):
        return {"response": "Sorry, this topic is restricted."}

    try:
        # Try RAG first
        rag_response = llm_service.generate_response(prompt)
        if rag_response and "not found in the provided data" not in rag_response.lower():
            return {"response": rag_response}
    except Exception as e:
        print("RAG failed:", e)

    # Fallback to Web Search if RAG fails
    web_response = llm_service.fetch_web_answer(prompt)
    if web_response:
        return {"response": f"This information is retrieved from the web: {web_response}"}

    # Fallback to LLM if web search also fails
    try:
        fallback = llm_service.generate_response_remote(prompt)
        return {"response": fallback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM fallback failed: {e}")

# === CHAT Endpoint ===
@app.post("/chat")
async def chat(chat_data: ChatRequest):
    if not chat_data.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_message = chat_data.messages[-1]
    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    # Check if the prompt contains restricted keywords
    if is_prompt_restricted(last_message.content, RESTRICTED_KEYWORDS):
        return {"response": "Sorry, this topic is restricted."}

    try:
        response = llm_service._call_hf_chat([
            {"role": msg.role, "content": msg.content} for msg in chat_data.messages
        ])
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat API failed: {e}")

@app.post("/transfer-to-chroma")
async def transfer_to_chroma():
    """Fetch data from MongoDB and write to Chroma DB."""
    try:
        # Initialize MongoDB service
        mongo_service = DBWriteService(db_type="mongo")
        # Fetch data from MongoDB
        documents = mongo_service.db_writer.fetch_all()  # Do not fetch unchanged events that are already vectorized

        # Define the Schema for Chroma
        chroma_service = DBWriteService(db_type="chroma")
        chroma_service.db_writer.clear()

        for doc in documents:
            texts = []
            metadatas = []
            ids = []

            # Use the content field directly for Investopedia and Confluence
            content_texts = [doc.get('content', '')]

            # Log to verify content is being used
            print(f"Size of Extracted Content Texts: {len(content_texts[0]) if content_texts[0] else 0} characters")

            # Check if content is not empty or just whitespace
            if content_texts and content_texts[0].strip():
                # Join all text segments into a single string (optional: use separator like \n)
                combined_text = '\n'.join(content_texts)
                texts.append(combined_text)

                # Add metadata like page_id or others as needed
                page_id = str(doc.get('page_id', ''))
                if not page_id:
                    page_id = str(uuid.uuid4())
                metadata = {
                    'page_id': page_id,
                    'title': doc.get('title', ''),
                    'author': doc.get('author', 'Unknown'),
                    'published_date': doc.get('published_date', 'Unknown'),
                    'url': doc.get('url', ''),
                    #'child_page_ids': json.dumps(doc.get('child_page_ids', []))
                }
                metadatas.append(metadata)
                

                ids = [page_id]  # Use page_id as the unique ID for Chroma

                print(f"Metadatas: {metadatas}")
                print(f"IDs: {ids}")
            else:
                # If content is empty, log the skipping and continue
                print(f"Skipping document with Page ID: {doc.get('page_id', '')} due to empty content.")
                continue  # Skip to the next document

            # Ensure texts are non-empty before writing to Chroma
            if texts:
                # Write to Chroma DB
                chroma_service.process_event({'texts': texts, 'ids': ids, 'metadatas': metadatas})
                print(f"Successfully written to Chroma for Page ID: {doc.get('page_id', '')}")
            else:
                print(f"Skipping document with Page ID: {doc.get('page_id', '')} due to empty texts.")

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
    
@app.delete("/clean-chroma")
async def clean_chroma():
    """Delete all data from Chroma DB."""
    try:
        # Initialize Chroma DB service
        chroma_service = DBWriteService(db_type="chroma")
        
        # Clear Chroma DB
        chroma_service.db_writer.clear()  #not so useful rn, need to change
        
        return {"message": "Chroma DB cleaned successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Folder path
FOLDER_PATH = 'C:\\Users\\mouni\\rag\\confluence_data'

@app.post("/insert_html_json/")
async def insert_html_json_files():
    
    db_write_service = DBWriteService(db_type="mongo")  

    inserted_files = []
    skipped_files = []

    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith('html.json'):
            file_path = os.path.join(FOLDER_PATH, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if data:
                        db_write_service.process_event(data)
                        inserted_files.append(filename)
                    else:
                        skipped_files.append(filename)
            except Exception as e:
                skipped_files.append(filename)
                print(f"⚠️ Error processing {filename}: {e}")

    return {
        "status": "completed",
        "inserted_files": inserted_files,
        "skipped_files": skipped_files
    }

@app.post("/insertData/")
async def insertData():
    db_write_service = DBWriteService(db_type="mongo")  

    inserted_files = []
    skipped_files = []

    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith('html.json'):  # Assuming all json files are relevant
            file_path = os.path.join(FOLDER_PATH, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if data:
                        # Determine if the file is Confluence or Investopedia based on its structure
                        if 'data' in data:  # Confluence format
                            content_texts = []
                            # Process heading and link types
                            for entry in data.get('data', []):
                                if entry.get('type') in ['heading', 'link'] and 'text' in entry:
                                    content_texts.append(entry['text'])
                            
                            # Check for confluence_tables and process them if present
                            if 'confluence_tables' in data:
                                for table in data['confluence_tables']:
                                    # If there are actual table data, we process it
                                    if table and isinstance(table, dict):
                                        # Convert table into a readable string format (could be JSON or table-like string)
                                        table_content = str(table)  
                                        content_texts.append(table_content)
                            
                            # Combine all content into a single string (optional separator)
                            combined_content = '\n'.join(content_texts)
                            
                            # Prepare the data in the same format for MongoDB

                            confluence_data = {
                                
                                "title": data.get("title", ""),
                                "author": data.get("author", "Unknown"),
                                "published_date": data.get("published_date", "Unknown"),
                                "content": combined_content,
                                "url": data.get("url", ""),
                                "page_id": data.get("page_id", ""),
                                "child_page_ids": data.get("child_page_ids", [])
                            }
                            db_write_service.process_event(confluence_data)

                        elif 'content' in data:  # Investopedia format
                            # Directly map the Investopedia data to MongoDB schema
                            investopedia_data = {
                                "title": data.get("title", ""),
                                "author": data.get("author", "Unknown"),
                                "published_date": data.get("published_date", "Unknown"),
                                "content": data.get("content", ""),
                                "url": data.get("url", "")
                            }
                            db_write_service.process_event(investopedia_data)
                        else:
                            skipped_files.append(filename)

                        inserted_files.append(filename)
                    else:
                        skipped_files.append(filename)
            except Exception as e:
                skipped_files.append(filename)
                print(f"⚠️ Error processing {filename}: {e}")

    return {
        "inserted_files": inserted_files,
        "skipped_files": skipped_files
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("app/static/index.html") as f:
        return HTMLResponse(content=f.read())