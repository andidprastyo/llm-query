from pydantic import BaseModel
from typing import Optional, List, Dict

class Query(BaseModel):
    text: str
    max_results: Optional[int] = 3

class Response(BaseModel):
    query: str
    response: str
    processing_time: float
    metadata: Dict = {}  # Add metadata field
    error: Optional[str] = None
    status: str = "success"