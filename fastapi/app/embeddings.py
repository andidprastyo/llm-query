import pickle
import json
import numpy as np
from typing import List, Dict, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import asyncio
from pathlib import Path
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self):
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = None
        self.documents = []
        self.is_initialized = False
        self.lock = asyncio.Lock()
        
    async def preprocess_record(self, record: Dict) -> str:
        """Convert record to searchable text"""
        formatted = []
        for k, v in record.items():
            if v:
                if isinstance(v, (int, float)):
                    formatted.append(f"{k}: {v}")
                elif isinstance(v, bool):
                    formatted.append(f"{k}: {'yes' if v else 'no'}")
                elif isinstance(v, str):
                    v = v.strip().replace('\n', ' ')
                    if v:
                        formatted.append(f"{k}: {v}")
        return " | ".join(formatted)

    async def init_embeddings(self):
        """Initialize embeddings asynchronously"""
        async with self.lock:
            if self.is_initialized:
                return

            logger.info("Starting embeddings initialization...")
            
            try:
                # Load data
                with open(settings.DATA_PATH, 'r') as f:
                    raw_data = [json.loads(line) for line in f if line.strip()]
                
                # Process documents asynchronously
                tasks = []
                for record in raw_data:
                    tasks.append(self.preprocess_record(record))
                document_contents = await asyncio.gather(*tasks)
                
                self.documents = [{"content": content, "metadata": record} 
                                for content, record in zip(document_contents, raw_data)]
                
                # Create embeddings
                embeddings = self.encoder.encode([doc["content"] for doc in self.documents])
                
                # Initialize FAISS index
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(embeddings.astype('float32'))
                
                # Save to pickle
                embeddings_dir = Path(settings.EMBEDDINGS_PATH).parent
                embeddings_dir.mkdir(parents=True, exist_ok=True)
                
                with open(settings.EMBEDDINGS_PATH, 'wb') as f:
                    pickle.dump({
                        'index': faiss.serialize_index(self.index),
                        'documents': self.documents
                    }, f)
                
                self.is_initialized = True
                logger.info("Embeddings initialization completed")
                
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {e}")
                raise

    async def get_index(self) -> Tuple[faiss.Index, List[Dict]]:
        """Get initialized index and documents"""
        if not self.is_initialized:
            try:
                # Try to load existing embeddings
                if Path(settings.EMBEDDINGS_PATH).exists():
                    with open(settings.EMBEDDINGS_PATH, 'rb') as f:
                        data = pickle.load(f)
                        self.index = faiss.deserialize_index(data['index'])
                        self.documents = data['documents']
                        self.is_initialized = True
                else:
                    # Generate new embeddings
                    await self.init_embeddings()
            except Exception as e:
                logger.error(f"Error in get_index: {e}")
                await self.init_embeddings()
        
        return self.index, self.documents

    async def get_initialization_status(self) -> Dict:
        """Get current initialization status"""
        return {
            "initialized": self.is_initialized,
            "document_count": len(self.documents) if self.documents else 0,
            "embeddings_file_exists": Path(settings.EMBEDDINGS_PATH).exists()
        }

# Create a global instance
embedding_manager = EmbeddingManager()