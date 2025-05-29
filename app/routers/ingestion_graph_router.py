from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
# Reuse the existing model
from app.ingestion_graph import ingestion_graph, GraphState # Import the graph and state

router = APIRouter()
class InvestopediaInsertRequest(BaseModel):
    file_path: str

@router.post("/ingest-file-graph/")
async def ingest_file_via_graph(request: InvestopediaInsertRequest):
    """Triggers the LangChain ingestion graph for a given file path."""
    file_path = request.file_path
    print(f"Received request to ingest file via graph: {file_path}")

    # Define the initial state for the graph
    initial_state: GraphState = {
        'file_path': file_path,
        'documents': [] # documents will be populated by the graph nodes
    }

    try:
        # Invoke the ingestion graph
        # Note: For long-running processes, consider running this in a background task
        # or using a job queue (like Celery) to avoid blocking the FastAPI event loop.
        # For simplicity in this example, we are running it directly.
        print("Invoking ingestion graph...")
        result = ingestion_graph.invoke(initial_state)
        print("Ingestion graph execution finished.")

        # You can inspect the final state if needed
        # print(f"Final graph state: {result}")

        return {"message": f"Ingestion graph triggered for file: {file_path}", "status": "completed"}

    except Exception as e:
        print(f"Error during graph execution: {e}")
        raise HTTPException(status_code=500, detail=f"Error during graph execution: {e}") 