from fastapi import FastAPI, HTTPException
from app.models import Query, Response
from app.embeddings import embedding_manager
from app.utils import query_llm
import time
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Query API")

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime object"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except:
        return None

def extract_date_records(records: List[Dict], year: Optional[int] = None, 
                        month: Optional[int] = None, 
                        end_date: bool = False) -> List[Dict]:
    """
    Extract records matching specific date criteria
    
    Args:
        records: List of document records
        year: Specific year to match
        month: Specific month to match
        end_date: If True, match end_kontrak, else match start_kontrak
    """
    matching_records = []
    
    for record in records:
        metadata = record['metadata']
        date_field = 'end_kontrak' if end_date else 'start_kontrak'
        date_str = metadata.get(date_field)
        
        if not date_str:
            continue
            
        date_obj = parse_date(date_str)
        if not date_obj:
            continue
            
        # Match criteria
        if year and month:
            if date_obj.year == year and date_obj.month == month:
                matching_records.append(record)
        elif year:
            if date_obj.year == year:
                matching_records.append(record)
        elif month:
            if date_obj.month == month:
                matching_records.append(record)
                
    return matching_records

async def retrieve_documents(query: str, index, documents: List[Dict], k: int) -> List[Dict]:
    """Retrieve relevant documents with date matching"""
    try:
        query_lower = query.lower()
        
        # Extract year if mentioned
        year = None
        for word in query_lower.split():
            if word.isdigit() and len(word) == 4:  # Year format YYYY
                year = int(word)
                break
                
        # Handle date-specific queries
        if year:
            initial_docs = documents  # Use all documents for date filtering
            if 'end' in query_lower or 'ending' in query_lower:
                return extract_date_records(initial_docs, year=year, end_date=True)
            elif 'start' in query_lower or 'starting' in query_lower:
                return extract_date_records(initial_docs, year=year, end_date=False)
            else:
                # If year mentioned but no start/end specified, check both
                start_matches = extract_date_records(initial_docs, year=year, end_date=False)
                end_matches = extract_date_records(initial_docs, year=year, end_date=True)
                return list(set(start_matches + end_matches))
        
        # Default embedding-based retrieval for non-date queries
        query_embedding = embedding_manager.encoder.encode([query]).astype('float32')
        D, I = index.search(query_embedding, k)
        return [documents[idx] for idx in I[0]]
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(embedding_manager.init_embeddings())

@app.get("/status")
async def get_status():
    return await embedding_manager.get_initialization_status()

@app.post("/generate-embeddings")
async def generate_embeddings():
    try:
        await embedding_manager.init_embeddings()
        return {"status": "success", "message": "Embeddings generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=Response)
async def process_query(query: Query):
    start_time = time.time()
    
    try:
        # Get index and documents
        index, documents = await embedding_manager.get_index()
        
        if not index or not documents:
            return Response(
                query=query.text,
                response="Embeddings not ready. Please try again in a moment.",
                processing_time=time.time() - start_time,
                status="error",
                error="Embeddings not initialized",
                metadata={"embeddings_status": "initializing"}
            )

        # Retrieve relevant documents
        relevant_docs = await retrieve_documents(
            query.text, 
            index, 
            documents, 
            query.max_results
        )
        
        if not relevant_docs:
            return Response(
                query=query.text,
                response="No relevant documents found for your query.",
                processing_time=time.time() - start_time,
                status="success",
                metadata={"docs_found": 0}
            )

        # Prepare context
        context_parts = []
        for doc in relevant_docs:
            metadata = doc['metadata']
            context_part = f"""CONTRACT DETAILS:
ID: {metadata.get('id', 'N/A')}
Service Name: {metadata.get('nama', 'N/A')}
Customer: {metadata.get('customer', 'N/A')}
Contract Start: {metadata.get('start_kontrak', 'N/A')}
Contract End: {metadata.get('end_kontrak', 'N/A')}
Service ID (SID): {metadata.get('sid', 'N/A')}
TSAT Service ID: {metadata.get('sid_tsat', 'N/A')}
Work Order: {metadata.get('no_wo', 'N/A')}

SERVICE INFO:
Type: {metadata.get('layanan', 'N/A')}
Segment: {metadata.get('segmen', 'N/A')}
Location: {metadata.get('provinsi', 'N/A')} - {metadata.get('kabupaten', 'N/A')}"""
            context_parts.append(context_part)
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""Based on these service and contract records:

{context}

Question: {query.text}

IMPORTANT INSTRUCTIONS:
1. List ONLY the records shown above
2. Keep all dates in YYYY-MM-DD format exactly as shown
3. Do not modify or calculate any dates
4. Format the response as bullet points with:
   - Contract ID
   - Service Name
   - Exact dates as shown
   - Customer name
5. If no dates are shown, say "date not available"

Answer:"""
        
        # Query LLM
        llm_result = await query_llm(prompt)
        
        return Response(
            query=query.text,
            response=llm_result["response"],
            processing_time=time.time() - start_time,
            status=llm_result.get("status", "success"),
            error=llm_result.get("error"),
            metadata={
                "docs_found": len(relevant_docs),
                "llm_status": llm_result.get("status"),
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return Response(
            query=query.text,
            response="An error occurred while processing your query.",
            processing_time=time.time() - start_time,
            status="error",
            error=str(e),
            metadata={
                "error_type": type(e).__name__,
                "error_details": str(e)
            }
        )