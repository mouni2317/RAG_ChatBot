import os
import json
import uuid
from typing import TypedDict, List, Dict, Any, Union, Literal, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain.document_loaders import TextLoader # Using TextLoader as a simple example
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import the ChromaService
from app.services.chroma_service import ChromaService

# Define the graph state
class AgentState(TypedDict):
    """Represents the state of our agent graph."""
    # Input structure: either file_path or a trigger for a predefined ingestion method
    input_type: Literal['file', 'investopedia_folder', 'confluence_folder']
    file_path: Optional[str] # Used when input_type is 'file'
    folder_path: Optional[str] # Used when input_type is 'investopedia_folder' or 'confluence_folder'
    # You could add more fields for other ingestion types or parameters
    
    # State for processing
    documents: List[Document]
    processed_count: int
    skipped_count: int
    status: str
    message: str

# Initialize ChromaService (using a specific collection for this agent)
chroma_service = ChromaService(collection_name="agent_ingested_data")

# --- Nodes for the Graph ---

def route_input(state: AgentState) -> Literal['process_file', 'process_investopedia', 'process_confluence']:
    """Routes the input based on its type."""
    print(f"Routing input type: {state['input_type']}")
    if state['input_type'] == 'file':
        return 'process_file'
    elif state['input_type'] == 'investopedia_folder':
        return 'process_investopedia'
    elif state['input_type'] == 'confluence_folder':
        return 'process_confluence'
    else:
        # Handle unknown types or raise an error
        print(f"Error: Unknown input type {state['input_type']}")
        # Depending on desired behavior, you might raise an exception or route to an error handler node
        raise ValueError(f"Unknown input type: {state['input_type']}")

def process_file(state: AgentState) -> AgentState:
    """Loads, splits, and prepares a single file for ChromaDB."""
    file_path = state['file_path']
    print(f"Processing single file: {file_path}")
    if not file_path or not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return {**state, 'documents': [], 'status': 'failed', 'message': f'File not found: {file_path}'}

    try:
        # Load document (using TextLoader, adapt as needed)
        loader = TextLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s).")

        # Split document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)
        print(f"Split into {len(split_documents)} chunks.")

        return {**state, 'documents': split_documents, 'status': 'processed', 'processed_count': len(split_documents), 'skipped_count': 0, 'message': f'Processed and split {file_path}'}

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {**state, 'documents': [], 'status': 'failed', 'message': f'Error processing file {file_path}: {e}'}

def process_investopedia(state: AgentState) -> AgentState:
    """Processes JSON files from a predefined Investopedia-like folder structure."""
    # This logic is adapted from the old /insert-investopedia-articles/ endpoint
    folder_path = state['folder_path'] # Use folder_path from state
    if not folder_path or not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return {**state, 'documents': [], 'status': 'failed', 'message': f'Folder not found: {folder_path}'}
        
    print(f"Processing Investopedia folder: {folder_path}")

    inserted_count = 0
    skipped_count = 0
    all_documents = []

    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    if not files:
        print("No JSON files found in the specified folder.")
        return {**state, 'documents': [], 'status': 'completed', 'message': 'No JSON files found in the specified folder.', 'processed_count': 0, 'skipped_count': 0}

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                article = json.load(f)

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
                "source_file": file_name
            }

            # Create a LangChain Document object for each article
            doc = Document(page_content=content, metadata=metadata)
            all_documents.append(doc)
            inserted_count += 1

        except Exception as e:
            skipped_count += 1
            print(f"⚠️ Error processing {file_name}: {e}")

    print(f"Finished processing folder. Inserted: {inserted_count}, Skipped: {skipped_count}")
    # Pass the list of Document objects to the next node
    return {**state, 'documents': all_documents, 'processed_count': inserted_count, 'skipped_count': skipped_count, 'status': 'processed_folder', 'message': f'Processed {inserted_count} articles from {folder_path}'}


def process_confluence(state: AgentState) -> AgentState:
    """Processes HTML JSON files from a predefined Confluence-like folder structure."""
    # This logic is adapted from the old /insert_html_json/ endpoint
    folder_path = state['folder_path'] # Use folder_path from state
    if not folder_path or not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return {**state, 'documents': [], 'status': 'failed', 'message': f'Folder not found: {folder_path}'}

    print(f"Processing Confluence folder: {folder_path}")

    inserted_files = []
    skipped_files = []
    all_documents = []

    files = [f for f in os.listdir(folder_path) if f.endswith('html.json')]
    if not files:
        print("No HTML JSON files found in the specified folder.")
        return {**state, 'documents': [], 'status': 'completed', 'message': 'No HTML JSON files found in the specified folder.', 'processed_count': 0, 'skipped_count': 0}

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if not data:
                skipped_files.append(filename)
                print(f"⚠️ Skipping {filename}: Empty JSON data")
                continue

            content_texts = []
            metadata = {}

            if 'data' in data:  # Confluence format
                for entry in data.get('data', []):
                    if entry.get('type') in ['heading', 'link'] and 'text' in entry:
                        content_texts.append(entry['text'])

                if 'confluence_tables' in data:
                    for table in data['confluence_tables']:
                        if table and isinstance(table, dict):
                            content_texts.append(str(table))

                combined_content = '\n'.join(content_texts)

                url_value = data.get("url") or data.get("page_id", "")

                metadata = {
                    "title": data.get("title", ""),
                    "author": data.get("author", "Unknown"),
                    "published_date": data.get("published_date", "Unknown"),
                    "url": url_value,
                    "source_file": filename
                }

            elif 'content' in data:  # Investopedia or similar within a single file
                combined_content = data.get("content", "")
                url_value = data.get("url") or data.get("page_id", "")

                metadata = {
                    "title": data.get("title", ""),
                    "author": data.get("author", "Unknown"),
                    "published_date": data.get("published_date", "Unknown"),
                    "url": url_value,
                    "source_file": filename
                }
            else:
                skipped_files.append(filename)
                print(f"⚠️ Skipping {filename}: Unrecognized data format")
                continue

            if combined_content.strip():
                 # Create a LangChain Document object
                doc = Document(page_content=combined_content, metadata=metadata)
                all_documents.append(doc)
                inserted_files.append(filename)
            else:
                skipped_files.append(filename)
                print(f"⚠️ Skipping {filename}: Empty content after parsing")

        except Exception as e:
            skipped_files.append(filename)
            print(f"⚠️ Error processing {filename}: {e}")

    print(f"Finished processing folder. Inserted files: {len(inserted_files)}, Skipped files: {len(skipped_files)}")
     # Pass the list of Document objects to the next node
    return {**state, 'documents': all_documents, 'processed_count': len(inserted_files), 'skipped_count': len(skipped_files), 'status': 'processed_folder', 'message': f'Processed {len(inserted_files)} files from {folder_path}'}

def add_documents_to_chroma(state: AgentState) -> AgentState:
    """Adds the list of Document objects in the state to ChromaDB."""
    documents = state['documents']
    if not documents:
        print("No documents to add to ChromaDB.")
        return {**state, 'status': 'completed', 'message': state.get('message','No documents processed or found.')}

    print(f"Adding {len(documents)} documents to ChromaDB...")
    try:
        # Use the add_documents method of ChromaService
        chroma_service.add_documents(documents)
        print(f"Successfully added {len(documents)} documents to ChromaDB.")
        return {**state, 'status': 'completed', 'message': f'Successfully added {len(documents)} documents to ChromaDB.'}
    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")
        return {**state, 'status': 'failed', 'message': f'Error adding documents to ChromaDB: {e}'}

# --- Build the Graph ---

workflow = StateGraph(AgentState)

# Add the routing node
workflow.add_node("route_input", route_input)

# Add processing nodes for different input types
workflow.add_node("process_file", process_file)
workflow.add_node("process_investopedia", process_investopedia)
workflow.add_node("process_confluence", process_confluence)

# Add the node to add documents to Chroma (common endpoint for processing nodes)
workflow.add_node("add_to_chroma", add_documents_to_chroma)

# Set the entry point
workflow.set_entry_point("route_input")

# Add conditional edges from the router
workflow.add_conditional_edges(
    "route_input",
    route_input, # The function to determine the next node
    {
        "process_file": "process_file",
        "process_investopedia": "process_investopedia",
        "process_confluence": "process_confluence",
    },
)

# Add edges from processing nodes to the add_to_chroma node
workflow.add_edge('process_file', 'add_to_chroma')
workflow.add_edge('process_investopedia', 'add_to_chroma')
workflow.add_edge('process_confluence', 'add_to_chroma')

# Add the edge from add_to_chroma to the end
workflow.add_edge('add_to_chroma', END)

# Compile the graph
ingestion_agent_graph = workflow.compile()

# Example usage (can be called from a FastAPI endpoint or script)
# if __name__ == "__main__":
#     # Example 1: Process a single text file
#     # file_to_ingest = "data.txt" # Make sure data.txt exists in the root
#     # initial_state: AgentState = {'input_type': 'file', 'file_path': file_to_ingest, 'documents': [], 'processed_count': 0, 'skipped_count': 0, 'status': 'initial', 'message': ''}
#     # print(f"\n--- Invoking graph for single file: {file_to_ingest} ---")
#     # result = ingestion_agent_graph.invoke(initial_state)
#     # print("Graph execution finished.")
#     # print(f"Final state: {result}")

#     # Example 2: Process Investopedia folder (replace with your path)
#     # investopedia_folder_path = '/Users/shreyaspandey/Library/Mobile Documents/com~apple~CloudDocs/RAG_ChatBot/investopedia_data' # Adjust path
#     # initial_state_investopedia: AgentState = {'input_type': 'investopedia_folder', 'folder_path': investopedia_folder_path, 'documents': [], 'processed_count': 0, 'skipped_count': 0, 'status': 'initial', 'message': ''}
#     # print(f"\n--- Invoking graph for Investopedia folder: {investopedia_folder_path} ---")
#     # result_investopedia = ingestion_agent_graph.invoke(initial_state_investopedia)
#     # print("Graph execution finished.")
#     # print(f"Final state: {result_investopedia}")

#     # Example 3: Process Confluence folder (replace with your path)
#     # confluence_folder_path = '/Users/shreyaspandey/Library/Mobile Documents/com~apple~CloudDocs/RAG_ChatBot/confluence_data' # Adjust path
#     # initial_state_confluence: AgentState = {'input_type': 'confluence_folder', 'folder_path': confluence_folder_path, 'documents': [], 'processed_count': 0, 'skipped_count': 0, 'status': 'initial', 'message': ''}
#     # print(f"\n--- Invoking graph for Confluence folder: {confluence_folder_path} ---")
#     # result_confluence = ingestion_agent_graph.invoke(initial_state_confluence)
#     # print("Graph execution finished.")
#     # print(f"Final state: {result_confluence}") 