from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from typing import Any

# Initialize FastAPI
app = FastAPI(title="Embedding Service with Qwen3 + Qdrant")

# Load environment variables from .env (if present)
load_dotenv()

# Configuration via environment with sensible defaults
MODEL_NAME = os.getenv("MODEL_NAME", "ibm-granite/granite-embedding-278m-multilingual")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "")

# Load embedding model
model = SentenceTransformer(MODEL_NAME, use_auth_token=HUGGINGFACE_HUB_TOKEN if HUGGINGFACE_HUB_TOKEN else None)

# Connect to Qdrant
qdrant = QdrantClient(url=QDRANT_URL)

# Helper to ensure a collection exists (creates if missing).
def ensure_collection(name: str):
    try:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )
    except Exception:
        # If collection already exists or any other non-fatal error, ignore.
        pass

# Note: Default collection is no longer created on startup.
# A collection name must be provided in API requests.

# --------- Request Schemas ---------
class InsertRequest(BaseModel):
    id: int
    text: str
    metadata: Optional[dict] = None
    collection: Optional[str] = None

class BulkInsertRequest(BaseModel):
    items: List[InsertRequest]
    collection: str

class LoadUsersRequest(BaseModel):
    source_file: Optional[str] = None  # Optional: if not using file upload

class MetadataFilter(BaseModel):
    field: str
    value: Optional[Any] = None   # Any JSON value
    gte: Optional[float] = None
    lte: Optional[float] = None

class CollectionSearch(BaseModel):
    name: str
    filters: Optional[List[MetadataFilter]] = None

class SearchRequest(BaseModel):
    query: Optional[str] = None   # optional: can do pure filter search
    top_k: int = 5
    score_threshold: Optional[float] = None
    collections: List[CollectionSearch]

# --------- Helper Functions ---------
def embed_text(text: str):
    # SentenceTransformer handles tokenization and embedding generation
    embeddings = model.encode(text)
    return embeddings

# --------- API Routes ---------
@app.post("/insert")
def insert_doc(req: InsertRequest):
    collection = req.collection
    ensure_collection(collection)
    vector = embed_text(req.text)
    qdrant.upsert(
        collection_name=collection,
        points=[
            PointStruct(id=req.id, vector=vector, payload=req.metadata or {"text": req.text})
        ]
    )
    return {"status": "inserted", "id": req.id, "collection": collection}

@app.post("/bulk_insert")
def bulk_insert(req: BulkInsertRequest):
    collection = req.collection
    ensure_collection(collection)
    points = []
    for item in req.items:
        # Per-item collection can override top-level collection if provided
        item_collection = item.collection
        if item_collection != collection:
            # Ensure the per-item collection exists if different
            ensure_collection(item_collection)
        vector = embed_text(item.text)
        points.append(PointStruct(id=item.id, vector=vector, payload=item.metadata or {"text": item.text}))
    qdrant.upsert(collection_name=collection, points=points)
    return {"status": "bulk_inserted", "count": len(points), "collection": collection}

@app.post("/search")
def search(req: SearchRequest):
    results_by_collection: dict[str, List[ScoredPoint]] = {}

    query_vec = embed_text(req.query) if req.query else None

    for col in req.collections:
        qdrant_filter = None
        if col.filters:
            conditions = []
            for f in col.filters:
                if f.value is not None:
                    conditions.append(FieldCondition(
                        key=f.field,
                        match=MatchValue(value=f.value)
                    ))
                if f.gte is not None or f.lte is not None:
                    conditions.append(FieldCondition(
                        key=f.field,
                        range=Range(gte=f.gte, lte=f.lte)
                    ))
            qdrant_filter = Filter(must=conditions)

        try:
            results = qdrant.search(
                collection_name=col.name,
                query_vector=query_vec,
                limit=req.top_k,
                score_threshold=req.score_threshold,
                query_filter=qdrant_filter
            )
            results_by_collection[col.name] = results
        except Exception:
            results_by_collection[col.name] = []

    return {"results": results_by_collection}
