import os
from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain.document_loaders import TextLoader # Using TextLoader as a simple example
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import the ChromaService
from app.services.chroma_service import ChromaService

# Define the graph state
class GraphState(TypedDict):
    """Represents the state of our graph."""
    file_path: str
    documents: List[Document]

# Initialize ChromaService
chroma_service = ChromaService(collection_name="ingested_documents")

# Define the nodes
def load_document(state: GraphState) -> GraphState:
    """Loads a document from the given file path."""
    file_path = state['file_path']
    print(f"Loading document from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return {**state, 'documents': []} # Return empty documents list

    try:
        # Use TextLoader for simplicity. Adapt for other file types if needed.
        loader = TextLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s).")
        return {**state, 'documents': documents}
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        return {**state, 'documents': []}

def split_document(state: GraphState) -> GraphState:
    """Splits the loaded document into chunks."""
    documents = state['documents']
    if not documents:
        print("No documents to split.")
        return state

    print(f"Splitting {len(documents)} document(s) into chunks...")
    # Configure your text splitter appropriately
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)
    print(f"Split into {len(split_documents)} chunks.")
    return {**state, 'documents': split_documents}

def add_to_chroma(state: GraphState) -> GraphState:
    """Adds the document chunks to ChromaDB."""
    documents = state['documents']
    if not documents:
        print("No documents to add to Chroma.")
        return state

    print(f"Adding {len(documents)} documents to ChromaDB...")
    try:
        chroma_service.add_documents(documents)
        print("Successfully added documents to ChromaDB.")
    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")

    return state

# Build the graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("load", load_document)
workflow.add_node("split", split_document)
workflow.add_node("add_to_chroma", add_to_chroma)

# Set entry point
workflow.set_entry_point("load")

# Add edges
workflow.add_edge('load', 'split')
workflow.add_edge('split', 'add_to_chroma')

# Set the end point
workflow.add_edge('add_to_chroma', END)

# Compile the graph
ingestion_graph = workflow.compile()

# Example usage (you can call this from another script or endpoint)
# if __name__ == "__main__":
#     # Replace with the actual path to your file
#     # For example, using data.txt from the project layout
#     file_to_ingest = "data.txt"
#     initial_state = {'file_path': file_to_ingest, 'documents': []}
#     result = ingestion_graph.invoke(initial_state)
#     print("\nGraph execution finished.")
#     # The result state will contain the final state after execution
#     # print(f"Final state: {result}") # Uncomment to see the final state 