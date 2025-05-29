from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import uuid

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from app_old.routers import ingestion_graph_router

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

app = FastAPI(title="Embedding and Ingestion Service", version="1.0")

# Include the routers
app.include_router(ingestion_graph_router.router, prefix="")

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


@app.get("/get-embeddings/")
async def get_embeddings(query: EmbeddingQuery):
    """Retrieve similar documents/embeddings from ChromaDB based on a query."""
    try:
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

@app.post("/insert-investopedia-articles/")
async def insert_investopedia_articles():
    """Reads a set of JSON files, each containing a single article, and adds them to ChromaDB."""
    folder_path = 'C:\\Users\\mouni\\rag\\investopedia'  # Set this to the folder where your .json files are located
    inserted_count = 0
    skipped_count = 0

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")

    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    if not files:
        raise HTTPException(status_code=400, detail="No JSON files found in the specified folder.")

    texts_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    # Process each file
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                article = json.load(f)  # Each file contains a single article

            if not isinstance(article, dict):
                skipped_count += 1
                print(f"⚠️ Skipping {file_name}: Not a valid article.")
                continue

            content = article.get("content", "").strip()
            if not content:
                skipped_count += 1
                print(f"⚠️ Skipping {file_name}: Empty content.")
                continue

            metadata = {
                "title": article.get("title", ""),
                "author": article.get("author", "Unknown"),
                "published_date": article.get("published_date", "Unknown"),
                "url": article.get("url", ""),
                "source_file": file_name  # Add the source file name for reference
            }

            article_id = metadata.get('url', None) or str(uuid.uuid4())  # Use URL or generate UUID
            ids_to_add.append(article_id)
            texts_to_add.append(content)
            metadatas_to_add.append(metadata)
            inserted_count += 1

        except Exception as e:
            skipped_count += 1
            print(f"⚠️ Error processing {file_name}: {e}")

    # Add documents to Chroma
    if texts_to_add:
        vector_store.add_texts(texts=texts_to_add, metadatas=metadatas_to_add, ids=ids_to_add)

    return {
        "message": f"Finished processing {len(files)} files.",
        "inserted_count": inserted_count,
        "skipped_count": skipped_count
    }

@app.post("/insert_html_json/")
async def insert_html_json_files():
    inserted_files = []
    skipped_files = []

    FOLDER_PATH = 'C:\\Users\\mouni\\rag\\confluence_data'

    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith('html.json'):
            file_path = os.path.join(FOLDER_PATH, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if data:
                        content_texts = []
                        metadata = {}
                        doc_id = None

                        if 'data' in data:  # Confluence format
                            for entry in data.get('data', []):
                                if entry.get('type') in ['heading', 'link'] and 'text' in entry:
                                    content_texts.append(entry['text'])

                            if 'confluence_tables' in data:
                                for table in data['confluence_tables']:
                                    if table and isinstance(table, dict):
                                        content_texts.append(str(table))

                            combined_content = '\n'.join(content_texts)
                            content_to_add = [combined_content]

                            # Use page_id if url is missing
                            url_value = data.get("url") or data.get("page_id", "")

                            metadata = {
                                "title": data.get("title", ""),
                                "author": data.get("author", "Unknown"),
                                "published_date": data.get("published_date", "Unknown"),
                                "url": url_value,
                                "source_file": filename  # updated field name
                            }

                            doc_id = data.get("page_id", None)

                        elif 'content' in data:  # Investopedia or similar
                            content_to_add = [data.get("content", "")]
                            url_value = data.get("url") or data.get("page_id", "")

                            metadata = {
                                "title": data.get("title", ""),
                                "author": data.get("author", "Unknown"),
                                "published_date": data.get("published_date", "Unknown"),
                                "url": url_value,
                                "source_file": filename
                            }

                            doc_id = None

                        else:
                            skipped_files.append(filename)
                            print(f"⚠️ Skipping {filename}: Unrecognized data format")
                            continue

                        if content_to_add and content_to_add[0].strip():
                            metadatas_list = [metadata] if metadata else [{}]
                            vector_store.add_texts(
                                texts=content_to_add,
                                metadatas=metadatas_list,
                                ids=[doc_id] if doc_id else None
                            )
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
    return {"message": "Embedding and Ingestion service is running"}

class QuestionRequest(BaseModel):
    query: str
    k: Optional[int] = 5  # Top-k documents to retrieve

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    """
    Accepts a user question, performs similarity search, writes result to 'data.txt', and returns preview.
    """
    try:
        results = vector_store.similarity_search(request.query, k=request.k)

        if not results:
            return {"message": "No relevant documents found."}

        combined_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        sources = [f"Source: {doc.metadata.get('source_file', '')}, Title: {doc.metadata.get('title', '')}" for doc in results]
        source_info = "\n".join(sources)

        full_document = f"Query: {request.query}\n\n{combined_text}\n\nSources:\n{source_info}"

        # Save to fixed file name
        file_path = "./data.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_document)

        return {
            "message": "data.txt written successfully.",
            "file_path": file_path,
            "preview": full_document[:500] + "..."  # Optional: show first 500 chars
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

