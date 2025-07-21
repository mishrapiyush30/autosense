from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import os
import psycopg
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AutoSense Agentic RAG API",
    description="AI diagnostic platform for connected cars",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for services
model: Optional[SentenceTransformer] = None
qdrant: Optional[QdrantClient] = None
DB_URL: Optional[str] = None
COLLECTION_NAME = "autosense"


# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    vin: Optional[str] = Field(None, max_length=17, description="Vehicle identification number")
    filter_type: Optional[str] = Field(None, description="Filter by type: 'dtc' or 'recall'")


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total_found: int


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]


# Dependency functions
def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the embedding model."""
    global model
    if model is None:
        try:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Initialized embedding model")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise HTTPException(status_code=500, detail="Embedding model not available")
    return model


def get_qdrant_client() -> QdrantClient:
    """Get or initialize the Qdrant client."""
    global qdrant
    if qdrant is None:
        try:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant = QdrantClient(qdrant_url)
            # Test connection
            qdrant.get_collections()
            logger.info(f"Connected to Qdrant at {qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise HTTPException(status_code=500, detail="Vector database not available")
    return qdrant


def get_db_url() -> str:
    """Get the database URL."""
    global DB_URL
    if DB_URL is None:
        DB_URL = os.environ.get("DATABASE_URL")
        if not DB_URL:
            raise HTTPException(status_code=500, detail="Database URL not configured")
    return DB_URL


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services = {}
    
    # Check embedding model
    try:
        get_embedding_model()
        services["embedding_model"] = "healthy"
    except Exception as e:
        services["embedding_model"] = f"unhealthy: {str(e)}"
    
    # Check Qdrant
    try:
        get_qdrant_client()
        services["qdrant"] = "healthy"
    except Exception as e:
        services["qdrant"] = f"unhealthy: {str(e)}"
    
    # Check database
    try:
        db_url = get_db_url()
        with psycopg.connect(db_url) as conn:
            conn.execute("SELECT 1")
        services["database"] = "healthy"
    except Exception as e:
        services["database"] = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if all("healthy" in status for status in services.values()) else "degraded"
    
    return HealthResponse(status=overall_status, services=services)


# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    embedding_model: SentenceTransformer = Depends(get_embedding_model),
    qdrant_client: QdrantClient = Depends(get_qdrant_client)
):
    """Search for relevant automotive diagnostic information."""
    try:
        # Create embedding for the query
        query_embedding = embedding_model.encode(request.query, normalize_embeddings=True)
        
        # Build filter if needed
        query_filter = None
        if request.vin or request.filter_type:
            must_conditions = []
            
            if request.vin:
                must_conditions.append(
                    models.FieldCondition(key="vin", match=models.MatchValue(value=request.vin))
                )
            
            if request.filter_type:
                must_conditions.append(
                    models.FieldCondition(key="type", match=models.MatchValue(value=request.filter_type))
                )
            
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=request.k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        results = []
        for result in search_results:
            result_dict = {
                "score": float(result.score),
                **result.payload
            }
            results.append(result_dict)
        
        return SearchResponse(
            results=results,
            query=request.query,
            total_found=len(results)
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Get DTC information endpoint
@app.get("/dtc/{code}")
async def get_dtc_info(code: str, db_url: str = Depends(get_db_url)):
    """Get detailed information about a specific DTC code."""
    try:
        with psycopg.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT code, category, description FROM dtc WHERE code = %s",
                (code.upper(),)
            )
            result = cur.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail=f"DTC code {code} not found")
            
            return {
                "code": result[0],
                "category": result[1],
                "description": result[2]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching DTC {code}: {e}")
        raise HTTPException(status_code=500, detail="Database error")


# Get recalls for a VIN endpoint
@app.get("/recalls/{vin}")
async def get_recalls_for_vin(vin: str, db_url: str = Depends(get_db_url)):
    """Get all recalls for a specific VIN."""
    try:
        with psycopg.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT nhtsa_id, date, summary 
                   FROM recall 
                   WHERE vin = %s 
                   ORDER BY date DESC""",
                (vin.upper(),)
            )
            results = cur.fetchall()
            
            recalls = []
            for row in results:
                recalls.append({
                    "nhtsa_id": row[0],
                    "date": str(row[1]) if row[1] else None,
                    "summary": row[2]
                })
            
            return {"vin": vin, "recalls": recalls, "count": len(recalls)}
    except Exception as e:
        logger.error(f"Error fetching recalls for VIN {vin}: {e}")
        raise HTTPException(status_code=500, detail="Database error")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AutoSense Agentic RAG API",
        "version": "0.1.0",
        "description": "AI diagnostic platform for connected cars",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "dtc_info": "/dtc/{code}",
            "recalls": "/recalls/{vin}"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 