from fastapi import FastAPI
from app.routers import embeddings, ingestion, qa, ingestion_agent

# Initialize FastAPI application
app = FastAPI(title="RAG-based Chatbot Service", version="1.0")

# Include the routers
app.include_router(embeddings.router, prefix="")
app.include_router(ingestion.router, prefix="")
app.include_router(qa.router, prefix="")
app.include_router(ingestion_agent.router, prefix="")

@app.get("/")
async def read_root():
    return {"message": "RAG-based Chatbot Service is running"}

