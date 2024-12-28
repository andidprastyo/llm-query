from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_URL: str = "http://ollama:11434/api/generate"
    MODEL_NAME: str = "llama3.2:1b"
    EMBEDDINGS_PATH: str = "/app/data/embeddings.pkl"
    DATA_PATH: str = "/app/data/test_data.jsonl"
    
settings = Settings()

# fastapi/app/models.py
from pydantic import BaseModel
from typing import Optional, List

class Query(BaseModel):
    text: str
    max_results: Optional[int] = 3

class Response(BaseModel):
    query: str
    response: str
    processing_time: float