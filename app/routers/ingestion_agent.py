from fastapi import APIRouter, HTTPException
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel

# Import the agent graph and its state
from app.ingestion_agent_graph import ingestion_agent_graph, AgentState

router = APIRouter()

# Define the request model for the ingestion agent endpoint
class IngestionRequest(BaseModel):
    input_type: Literal['file', 'investopedia_folder', 'confluence_folder']
    file_path: Optional[str] = None # Path for 'file' type
    folder_path: Optional[str] = None # Path for folder types
    # Add validation logic here if needed, e.g., ensure file_path is provided for 'file' input_type


@router.post("/ingest-with-agent/")
async def ingest_with_agent(request: IngestionRequest):
    """Triggers the LangChain ingestion agent graph."""
    print(f"Received ingestion request: {request.input_type}")

    # Prepare the initial state for the agent graph from the request data
    initial_state: AgentState = {
        'input_type': request.input_type,
        'file_path': request.file_path,
        'folder_path': request.folder_path,
        'documents': [], # Initialize documents list
        'processed_count': 0,
        'skipped_count': 0,
        'status': 'initial',
        'message': 'Ingestion started.'
    }

    try:
        # Invoke the ingestion agent graph
        # Consider running this in a background task for long-running processes
        print("Invoking ingestion agent graph...")
        final_state = ingestion_agent_graph.invoke(initial_state)
        print("Ingestion agent graph execution finished.")

        return {"message": "Ingestion process completed.", "final_state": final_state}

    except Exception as e:
        print(f"Error during ingestion agent graph execution: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during ingestion: {str(e)}") 