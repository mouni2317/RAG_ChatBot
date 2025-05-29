from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class EmbeddingQuery(BaseModel):
    query_text: str
    k: int = 5


class DocumentData(BaseModel):
    texts: List[str]
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []


class InvestopediaInsertRequest(BaseModel):
    file_path: str


class QuestionRequest(BaseModel):
    query: str
    k: Optional[int] = 5  # Top-k documents to retrieve 