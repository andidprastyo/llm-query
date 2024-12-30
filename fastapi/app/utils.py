import aiohttp
from typing import Dict
import logging
from app.config import settings

logger = logging.getLogger(__name__)

async def query_llm(prompt: str) -> Dict:
    """Query Ollama API with better error handling"""
    payload = {
        "model": settings.MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "top_p": 0.95
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.OLLAMA_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "response": data['response']
                    }
                else:
                    error_msg = await response.text()
                    return {
                        "status": "error",
                        "response": f"LLM API error: {error_msg}",
                        "error_code": response.status
                    }
    except Exception as e:
        logger.error(f"LLM query failed: {str(e)}")
        return {
            "status": "error",
            "response": f"Failed to connect to LLM service: {str(e)}",
            "error": str(e)
        }