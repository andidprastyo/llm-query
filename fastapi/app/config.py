from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_URL: str = "http://ollama:11434/api/generate"
    MODEL_NAME: str = "llama3.2:1b"
    EMBEDDINGS_PATH: str = str(Path(__file__).parent / "data" / "embeddings.pkl")
    DATA_PATH: str = str(Path(__file__).parent / "data" / "test_data.jsonl")
    
settings = Settings()
