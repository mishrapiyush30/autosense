from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import sqlite3
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AutoSense Agentic RAG API (Local)",
    description="AI diagnostic platform for connected cars - Local Version",
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

# Global variables
model: Optional[SentenceTransformer] = None
bm25_index: Optional[BM25Okapi] = None
documents: List[Dict[str, Any]] = []
DB_PATH = "autosense.db"
COLLECTION_NAME = "autosense_local"

# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    vin: Optional[str] = Field(None, max_length=17, description="Vehicle identification number")
    use_hybrid: bool = Field(default=True, description="Use hybrid search (vector + BM25)")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total_found: int
    search_type: str

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]

def init_database():
    """Initialize SQLite database with sample data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dtc (
            code TEXT PRIMARY KEY,
            category TEXT,
            description TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recall (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nhtsa_id INTEGER,
            vin TEXT,
            date TEXT,
            summary TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle (
            id INTEGER PRIMARY KEY,
            vin TEXT UNIQUE,
            make TEXT,
            model TEXT,
            year INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_reading (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            ts TIMESTAMP,
            sensor TEXT,
            value REAL,
            FOREIGN KEY (vehicle_id) REFERENCES vehicle (id)
        )
    ''')
    
    # Insert sample DTC data
    sample_dtcs = [
        ("P0420", "Engine", "Catalyst System Efficiency Below Threshold (Bank 1)"),
        ("P0300", "Engine", "Random/Multiple Cylinder Misfire Detected"),
        ("P0171", "Engine", "System Too Lean (Bank 1)"),
        ("P0174", "Engine", "System Too Lean (Bank 2)"),
        ("P0128", "Engine", "Coolant Thermostat Temperature Below Regulating Temperature"),
        ("P0442", "Evaporative Emission Control", "Evaporative Emission Control System Leak Detected (Small Leak)"),
        ("P0455", "Evaporative Emission Control", "Evaporative Emission Control System Leak Detected (Gross Leak)"),
        ("P0506", "Engine", "Idle Control System RPM Lower Than Expected"),
        ("P0507", "Engine", "Idle Control System RPM Higher Than Expected"),
        ("P0700", "Transmission", "Transmission Control System Malfunction"),
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO dtc (code, category, description) VALUES (?, ?, ?)",
        sample_dtcs
    )
    
    # Insert sample recall data
    sample_recalls = [
        (12345, "2HGFC2F59JH000001", "2024-01-15", "Safety recall for airbag deployment issue"),
        (12346, "2HGFC2F59JH000002", "2024-02-20", "Recall for brake system software update"),
        (12347, "2HGFC2F59JH000003", "2024-03-10", "Fuel system component replacement required"),
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO recall (nhtsa_id, vin, date, summary) VALUES (?, ?, ?, ?)",
        sample_recalls
    )
    
    conn.commit()
    conn.close()
    logger.info("Database initialized with sample data")

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

def build_bm25_index():
    """Build BM25 index from database documents."""
    global bm25_index, documents
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all documents
    documents = []
    
    # DTC codes
    cursor.execute("SELECT code, category, description FROM dtc")
    for code, category, desc in cursor.fetchall():
        text = f"DTC {code} ({category}): {desc}"
        documents.append({
            "text": text,
            "type": "dtc",
            "code": code,
            "category": category,
            "description": desc
        })
    
    # Recalls
    cursor.execute("SELECT nhtsa_id, vin, date, summary FROM recall")
    for rid, vin, date, summ in cursor.fetchall():
        text = f"Recall {rid} ({date}): {summ}"
        documents.append({
            "text": text,
            "type": "recall",
            "rid": str(rid),
            "vin": vin,
            "date": str(date) if date else None,
            "summary": summ
        })
    
    conn.close()
    
    if documents:
        # Tokenize documents for BM25
        tokenized_docs = [doc["text"].lower().split() for doc in documents]
        bm25_index = BM25Okapi(tokenized_docs)
        logger.info(f"Built BM25 index with {len(documents)} documents")
    else:
        logger.warning("No documents found for BM25 index")


def hybrid_search(query: str, k: int = 5, vin: Optional[str] = None) -> List[Dict[str, Any]]:
    """Perform hybrid search combining vector and BM25 scores."""
    global model, bm25_index, documents
    
    if not documents or not bm25_index:
        return search_local(query, k, vin)
    
    # Vector search
    vector_results = search_local(query, k * 2, vin)  # Get more results for reranking
    
    # BM25 search
    query_tokens = query.lower().split()
    bm25_scores = bm25_index.get_scores(query_tokens)
    
    # Combine scores
    combined_results = []
    for i, doc in enumerate(documents):
        if vin and doc.get("vin") and doc["vin"] != vin:
            continue  # Filter by VIN if provided
        
        # Find corresponding vector result
        vector_score = 0.0
        for vec_result in vector_results:
            if (doc.get("code") == vec_result.get("code") or 
                doc.get("rid") == vec_result.get("rid")):
                vector_score = vec_result.get("score", 0.0)
                break
        
        # Normalize BM25 score (0-1 range)
        bm25_score = min(bm25_scores[i] / 10.0, 1.0) if bm25_scores[i] > 0 else 0.0
        
        # Combine scores (weighted average)
        combined_score = 0.7 * vector_score + 0.3 * bm25_score
        
        combined_results.append({
            **doc,
            "score": combined_score,
            "vector_score": vector_score,
            "bm25_score": bm25_score
        })
    
    # Sort by combined score and return top k
    combined_results.sort(key=lambda x: x["score"], reverse=True)
    return combined_results[:k]


def search_local(query: str, k: int = 5, vin: Optional[str] = None) -> List[Dict[str, Any]]:
    """Simple local search implementation."""
    embedding_model = get_embedding_model()
    
    # Get all DTC codes
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT code, category, description FROM dtc")
    dtc_results = cursor.fetchall()
    
    cursor.execute("SELECT nhtsa_id, vin, date, summary FROM recall")
    recall_results = cursor.fetchall()
    
    conn.close()
    
    # Create searchable documents
    documents = []
    
    # Add DTC documents
    for code, category, description in dtc_results:
        text = f"DTC {code} ({category}): {description}"
        documents.append({
            "text": text,
            "type": "dtc",
            "code": code,
            "category": category,
            "description": description,
            "score": 0.0
        })
    
    # Add recall documents
    for nhtsa_id, recall_vin, date, summary in recall_results:
        text = f"Recall {nhtsa_id} ({date}): {summary}"
        documents.append({
            "text": text,
            "type": "recall",
            "rid": nhtsa_id,
            "vin": recall_vin,
            "date": date,
            "summary": summary,
            "score": 0.0
        })
    
    # Simple keyword-based scoring
    query_lower = query.lower()
    for doc in documents:
        score = 0.0
        text_lower = doc["text"].lower()
        
        # Exact matches get high scores
        if query_lower in text_lower:
            score += 0.8
        
        # Word overlap
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        overlap = len(query_words.intersection(text_words))
        score += overlap * 0.1
        
        # VIN filtering
        if vin and doc.get("vin") and vin.upper() in doc["vin"].upper():
            score += 0.5
        
        doc["score"] = min(score, 1.0)
    
    # Sort by score and return top k
    documents.sort(key=lambda x: x["score"], reverse=True)
    return documents[:k]

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
    
    # Check database
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        services["database"] = "healthy"
    except Exception as e:
        services["database"] = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if all("healthy" in status for status in services.values()) else "degraded"
    
    return HealthResponse(status=overall_status, services=services)

@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    embedding_model: SentenceTransformer = Depends(get_embedding_model)
):
    """Search for relevant automotive diagnostic information."""
    try:
        results = hybrid_search(request.query, request.k, request.vin)
        
        return SearchResponse(
            results=results,
            query=request.query,
            total_found=len(results),
            search_type="hybrid" if request.use_hybrid else "vector"
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/dtc/{code}")
async def get_dtc_info(code: str):
    """Get detailed information about a DTC code."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT code, category, description FROM dtc WHERE code = ?",
            (code.upper(),)
        )
        result = cursor.fetchone()
        conn.close()
        
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

@app.get("/recalls/{vin}")
async def get_recalls_for_vin(vin: str):
    """Get all recalls for a specific VIN."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            """SELECT nhtsa_id, date, summary 
               FROM recall 
               WHERE vin = ? 
               ORDER BY date DESC""",
            (vin.upper(),)
        )
        results = cursor.fetchall()
        conn.close()
        
        recalls = []
        for row in results:
            recalls.append({
                "nhtsa_id": row[0],
                "date": row[1],
                "summary": row[2]
            })
        
        return {"vin": vin, "recalls": recalls, "count": len(recalls)}
    except Exception as e:
        logger.error(f"Error fetching recalls for VIN {vin}: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/sensors/{vin}")
async def get_sensor_data(vin: str, sensor: Optional[str] = None, limit: int = 100):
    """Get sensor data for a specific vehicle."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get vehicle ID
        cursor.execute("SELECT id FROM vehicle WHERE vin = ?", (vin,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Vehicle with VIN {vin} not found")
        
        vehicle_id = result[0]
        
        # Build query
        query = """
        SELECT sr.ts, sr.sensor, sr.value, v.vin, v.make, v.model, v.year
        FROM sensor_reading sr
        JOIN vehicle v ON sr.vehicle_id = v.id
        WHERE sr.vehicle_id = ?
        """
        params = [vehicle_id]
        
        if sensor:
            query += " AND sr.sensor = ?"
            params.append(sensor)
        
        query += " ORDER BY sr.ts DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        conn.close()
        
        return {
            "vin": vin,
            "sensor_data": [
                {
                    "timestamp": row[0],
                    "sensor": row[1],
                    "value": row[2],
                    "vehicle_info": {
                        "vin": row[3],
                        "make": row[4],
                        "model": row[5],
                        "year": row[6]
                    }
                }
                for row in results
            ],
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sensors/{vin}/analytics")
async def get_sensor_analytics(vin: str, sensor: Optional[str] = None):
    """Get sensor analytics for a specific vehicle."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get vehicle ID
        cursor.execute("SELECT id FROM vehicle WHERE vin = ?", (vin,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Vehicle with VIN {vin} not found")
        
        vehicle_id = result[0]
        
        # Build analytics query
        query = """
        SELECT 
            sr.sensor,
            AVG(sr.value) as avg_value,
            MIN(sr.value) as min_value,
            MAX(sr.value) as max_value,
            COUNT(*) as reading_count,
            MAX(sr.ts) as latest_reading
        FROM sensor_reading sr
        WHERE sr.vehicle_id = ?
        """
        params = [vehicle_id]
        
        if sensor:
            query += " AND sr.sensor = ?"
            params.append(sensor)
        
        query += " GROUP BY sr.sensor"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        conn.close()
        
        return {
            "vin": vin,
            "analytics": [
                {
                    "sensor": row[0],
                    "average": round(row[1], 2),
                    "minimum": row[2],
                    "maximum": row[3],
                    "reading_count": row[4],
                    "latest_reading": row[5]
                }
                for row in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting sensor analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sensors/{vin}/anomalies")
async def get_sensor_anomalies(vin: str):
    """Get sensor anomalies for a specific vehicle."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get vehicle ID
        cursor.execute("SELECT id FROM vehicle WHERE vin = ?", (vin,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Vehicle with VIN {vin} not found")
        
        vehicle_id = result[0]
        
        # Define thresholds
        thresholds = {
            'engine_temp': {'min': 160, 'max': 230, 'unit': '°F'},
            'rpm': {'min': 500, 'max': 3500, 'unit': 'RPM'},
            'fuel_level': {'min': 5, 'max': 100, 'unit': '%'},
            'speed': {'min': 0, 'max': 120, 'unit': 'mph'},
            'oil_pressure': {'min': 15, 'max': 70, 'unit': 'psi'},
            'battery_voltage': {'min': 11.5, 'max': 14.5, 'unit': 'V'}
        }
        
        anomalies = []
        
        for sensor, threshold in thresholds.items():
            query = """
            SELECT sr.ts, sr.value
            FROM sensor_reading sr
            WHERE sr.vehicle_id = ? AND sr.sensor = ? 
            AND (sr.value < ? OR sr.value > ?)
            ORDER BY sr.ts DESC
            """
            params = [vehicle_id, sensor, threshold['min'], threshold['max']]
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            for row in results:
                anomalies.append({
                    'timestamp': row[0],
                    'sensor': sensor,
                    'value': row[1],
                    'threshold_min': threshold['min'],
                    'threshold_max': threshold['max'],
                    'unit': threshold['unit'],
                    'severity': 'high' if abs(row[1] - (threshold['min'] + threshold['max']) / 2) > (threshold['max'] - threshold['min']) / 2 else 'medium'
                })
        
        conn.close()
        
        return {
            "vin": vin,
            "anomalies": anomalies,
            "count": len(anomalies)
        }
        
    except Exception as e:
        logger.error(f"Error getting sensor anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AutoSense Agentic RAG API (Local)",
        "version": "0.1.0",
        "description": "AI diagnostic platform for connected cars - Local Version",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "dtc_info": "/dtc/{code}",
            "recalls": "/recalls/{vin}"
        }
    }

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_database()
    build_bm25_index() # Build index on startup

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 