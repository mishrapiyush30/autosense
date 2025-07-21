import os
import psycopg
import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import Iterator, Tuple, Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB = os.environ["DATABASE_URL"]
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "autosense"


def get_qdrant_client() -> QdrantClient:
    """Initialize and return Qdrant client."""
    try:
        client = QdrantClient(QDRANT_URL)
        # Test connection
        client.get_collections()
        logger.info(f"Connected to Qdrant at {QDRANT_URL}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise


def get_embedding_model() -> SentenceTransformer:
    """Initialize and return the embedding model."""
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def iter_text() -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Iterate through all text data to be embedded."""
    with psycopg.connect(DB) as conn, conn.cursor() as cur:
        # Get DTC codes
        cur.execute("SELECT code, category, description FROM dtc")
        for code, category, desc in cur:
            text = f"DTC {code} ({category}): {desc}"
            yield text, {"type": "dtc", "code": code, "category": category}
        
        # Get recalls
        cur.execute("SELECT nhtsa_id, vin, date, summary FROM recall")
        for rid, vin, date, summ in cur:
            text = f"Recall {rid} ({date}): {summ}"
            yield text, {"type": "recall", "rid": int(rid), "vin": vin, "date": str(date) if date else None}


def build_index() -> None:
    """Build the vector index in Qdrant."""
    client = get_qdrant_client()
    model = get_embedding_model()
    
    # Get all text and metadata
    texts_and_metas = list(iter_text())
    if not texts_and_metas:
        logger.warning("No data found to index")
        return
    
    logger.info(f"Found {len(texts_and_metas)} documents to index")
    
    # Create embeddings
    vecs, metas, ids = [], [], []
    for i, (text, meta) in enumerate(tqdm.tqdm(texts_and_metas, desc="Creating embeddings")):
        try:
            vec = model.encode(text, normalize_embeddings=True)
            vecs.append(vec)
            metas.append(meta)
            ids.append(i)
        except Exception as e:
            logger.error(f"Error encoding text {i}: {e}")
            continue
    
    if not vecs:
        logger.error("No embeddings created")
        return
    
    # Create or recreate collection
    vector_size = len(vecs[0])
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "size": vector_size,
                "distance": "Cosine"
            }
        )
        logger.info(f"Created collection '{COLLECTION_NAME}' with vector size {vector_size}")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return
    
    # Upload vectors
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": id_val,
                    "vector": vec.tolist(),
                    "payload": meta
                }
                for id_val, vec, meta in zip(ids, vecs, metas)
            ]
        )
        logger.info(f"Successfully indexed {len(ids)} documents")
    except Exception as e:
        logger.error(f"Failed to upload vectors: {e}")


def get_collection_info() -> Dict[str, Any]:
    """Get information about the indexed collection."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "name": info.name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        return {}


if __name__ == "__main__":
    logger.info("Starting vector index build...")
    build_index()
    
    # Print collection info
    info = get_collection_info()
    if info:
        logger.info(f"Collection info: {info}") 