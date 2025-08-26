from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import ScoredPoint
from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv

# Initialize FastAPI
app = FastAPI(title="Embedding Service with Qwen3 + Qdrant")

# Load environment variables from .env (if present)
load_dotenv()

# Configuration via environment with sensible defaults
MODEL_NAME = os.getenv("MODEL_NAME", "ibm-granite/granite-embedding-278m-multilingual")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Connect to Qdrant
qdrant = QdrantClient(url=QDRANT_URL)

# Helper to ensure a collection exists (creates if missing).
def ensure_collection(name: str):
    try:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=model.config.hidden_size, distance=Distance.COSINE),
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

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: Optional[float] = None
    collections: List[str]

# --------- Helper Functions ---------
def embed_text(text: str):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings[0].cpu().numpy()

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
    
    query_vec = embed_text(req.query)

    for collection_name in req.collections:
        try:
            # Do not automatically create collection for search - return empty if missing
            results = qdrant.search(
                collection_name=collection_name,
                query_vector=query_vec,
                limit=req.top_k, # Fetch top_k from each collection
                score_threshold=req.score_threshold
            )
            results_by_collection[collection_name] = results
        except Exception:
            # If an error occurs (e.g., collection not found), assign an empty list
            results_by_collection[collection_name] = []

    return {"results": results_by_collection}
