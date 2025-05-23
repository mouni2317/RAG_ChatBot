from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
import uuid

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

# Path to local sentence transformer model
MODEL_DIR = "models/all-MiniLM-L6-v2"

# Initialize ChromaDB vector store and embedding model
persist_directory = "./chroma_data"
embedding_model = SentenceTransformerEmbeddings(model_name=MODEL_DIR)

# Initialize ChromaDB client and collection
# Note: Ensure the ChromaDB server is running or use an in-memory instance
# For simplicity, this example uses a file-based persistent directory.
# In a real application, you might connect to a separate Chroma server.

vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

app = FastAPI(title="Embedding Service", version="1.0")


class EmbeddingQuery(BaseModel):
    query_text: str
    k: int = 5


class DocumentData(BaseModel):
    texts: List[str]
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []


# New model for requesting to insert Investopedia articles
class InvestopediaInsertRequest(BaseModel):
    file_path: str


@app.post("/get-embeddings/")
async def get_embeddings(query: EmbeddingQuery):
    """Retrieve similar documents/embeddings from ChromaDB based on a query."""
    try:
        # Perform similarity search
        results = vector_store.similarity_search(query.query_text, k=query.k)

        return {
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in results
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/write-embeddings/")
async def write_embeddings(data: DocumentData):
    """Write documents/embeddings to ChromaDB."""
    try:
        docs = []
        for i, text in enumerate(data.texts):
            metadata = data.metadatas[i] if i < len(data.metadatas) else {}
            doc_id = data.ids[i] if i < len(data.ids) else None
            docs.append(Document(page_content=text, metadata=metadata, id=doc_id))

        # Add the documents to the vector store
        vector_store.add_documents(docs)

        return {"message": f"Successfully added {len(docs)} documents to ChromaDB"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New endpoint to handle large JSON file with Investopedia articles
@app.post("/insert-investopedia-articles/")
async def insert_investopedia_articles(request: InvestopediaInsertRequest):
    """Read a large JSON file with a list of articles and add them to ChromaDB."""
    file_path = request.file_path
    inserted_count = 0
    skipped_count = 0

    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        if not isinstance(articles, list):
            raise HTTPException(status_code=400, detail="JSON file should contain a list of articles")

        texts_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        for article in articles:
            try:
                # Assuming the structure of each article matches the sample provided (Investopedia)
                content = article.get("content", "").strip()
                if not content:
                    skipped_count += 1
                    print(f"⚠️ Skipping article: Empty content.")
                    continue

                metadata = {
                    "title": article.get("title", ""),
                    "author": article.get("author", "Unknown"),
                    "published_date": article.get("published_date", "Unknown"),
                    "url": article.get("url", ""),
                    # Add source file information
                    "source_file": os.path.basename(file_path)
                }
                # Generate a simple ID if not available (optional, Chroma can do this)
                # Using a hash of content + url could be a more robust approach if unique IDs are needed
                article_id = metadata.get('url', None) or str(uuid.uuid4()) # Use URL as ID if present, otherwise generate UUID
                ids_to_add.append(article_id)
                texts_to_add.append(content)
                metadatas_to_add.append(metadata)
                inserted_count += 1

            except Exception as e:
                skipped_count += 1
                print(f"⚠️ Error processing article: {e}")
                # Optionally log the article data that caused the error

        # Add documents to Chroma in batches (Chroma handles this internally with add_texts)
        if texts_to_add:
            vector_store.add_texts(texts=texts_to_add, metadatas=metadatas_to_add, ids=ids_to_add)

        return {
            "message": "Finished processing Investopedia articles file.",
            "inserted_count": inserted_count,
            "skipped_count": skipped_count
        }

    except FileNotFoundError:
         raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/insert_html_json/")
async def insert_html_json_files():

    inserted_files = []
    skipped_files = []
    FOLDER_PATH = '/Users/shreyaspandey/Library/Mobile Documents/com~apple~CloudDocs/RAG_ChatBot/confluence_data'
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith('html.json'):
            file_path = os.path.join(FOLDER_PATH, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if data:
                        # Adapt parsing logic from the previous /insertData/ endpoint
                        content_texts = []
                        metadata = {}
                        doc_id = None # Initialize doc_id

                        if 'data' in data:  # Confluence format
                            for entry in data.get('data', []):
                                if entry.get('type') in ['heading', 'link'] and 'text' in entry:
                                    content_texts.append(entry['text'])
                            if 'confluence_tables' in data:
                                for table in data['confluence_tables']:
                                    if table and isinstance(table, dict):
                                        table_content = str(table)
                                        content_texts.append(table_content)

                            combined_content = '\n'.join(content_texts)
                            content_to_add = [combined_content] # Ensure texts is a list

                            # Prepare metadata for Chroma
                            metadata = {
                                "title": data.get("title", ""),
                                "author": data.get("author", "Unknown"),
                                "published_date": data.get("published_date", "Unknown"),
                                "url": data.get("url", ""),
                                "file_name": filename # Add filename as metadata
                            }
                            # Use page_id as doc_id if available
                            doc_id = data.get("page_id", None)

                        elif 'content' in data:  # Investopedia format or similar structure
                            content_to_add = [data.get("content", "")] # Ensure texts is a list

                            # Prepare metadata for Chroma
                            metadata = {
                                "title": data.get("title", ""),
                                "author": data.get("author", "Unknown"),
                                "published_date": data.get("published_date", "Unknown"),
                                "url": data.get("url", ""),
                                "file_name": filename # Add filename as metadata
                            }
                            # No specific ID from this format, Chroma will generate one if doc_id is None
                            doc_id = None
                        else:
                            # If format is not recognized, skip the file
                            skipped_files.append(filename)
                            print(f"⚠️ Skipping {filename}: Unrecognized data format")
                            continue # Skip to the next file

                        # Write to Chroma vector store
                        if content_to_add and content_to_add[0].strip(): # Only add if content is not empty
                             # Ensure metadatas is a list of dicts matching texts length
                            metadatas_list = [metadata] if metadata else [{}]
                            # Use add_texts which handles adding documents with generated IDs if none are provided
                            vector_store.add_texts(texts=content_to_add, metadatas=metadatas_list, ids=[doc_id] if doc_id else None)
                            inserted_files.append(filename)
                        else:
                            skipped_files.append(filename)
                            print(f"⚠️ Skipping {filename}: Empty content after parsing")

                    else:
                        skipped_files.append(filename)
                        print(f"⚠️ Skipping {filename}: Empty JSON data")

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
    # This endpoint is now redundant as insert_html_json_files handles writing to Chroma
    return {"message": "This endpoint is deprecated. Use /insert_html_json/ instead."}

@app.get("/")
async def read_root():
    return {"message": "Embedding service is running"}
