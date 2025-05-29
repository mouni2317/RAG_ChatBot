from fastapi import APIRouter, HTTPException
import os
import json
import uuid
from typing import List, Dict, Any

from app.models import InvestopediaInsertRequest
from app.services.chroma_service import ChromaService

# Initialize ChromaService (consider using dependency injection in a real app)
# Assuming a default collection name or configure as needed
# Using a specific collection name for ingestion for clarity
chroma_service = ChromaService(collection_name="ingested_data")

router = APIRouter()

@router.post("/insert-investopedia-articles/")
async def insert_investopedia_articles():
    """Reads a set of JSON files, each containing a single article, and adds them to ChromaDB."""
    # NOTE: This endpoint assumes a fixed folder path. Consider making this path configurable or accepting a file list.
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
        chroma_service.add_texts(texts=texts_to_add, metadatas=metadatas_to_add, ids=ids_to_add)

    return {
        "message": f"Finished processing {len(files)} files.",
        "inserted_count": inserted_count,
        "skipped_count": skipped_count
    }

@router.post("/insert_html_json/")
async def insert_html_json_files():
    """Processes HTML JSON files from a predefined folder and adds content to ChromaDB."""
    # NOTE: This endpoint assumes a fixed folder path. Consider making this path configurable or accepting a file list.
    inserted_files = []
    skipped_files = []

    FOLDER_PATH = 'C:\\Users\\mouni\\rag\\confluence_data' # Make sure this path is correct

    if not os.path.exists(FOLDER_PATH):
        return {"status": "error", "message": f"Folder not found: {FOLDER_PATH}"}

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
                            chroma_service.add_texts(
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