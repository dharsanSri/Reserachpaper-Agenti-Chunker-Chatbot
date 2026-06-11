import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """
    Manages vector storage inside a Qdrant cluster.
    """
    
    def __init__(self):
        self.client = None

    def connect(self) -> None:
        """
        Connects to the Qdrant cluster using credentials from .env.
        Supports both QDRANT_URL/QDRANT_ENDPOINT and QDRANT_API_KEY/QDRANT_API.
        """
        if self.client is None:
            url = os.getenv("QDRANT_URL") or os.getenv("QDRANT_ENDPOINT")
            api_key = os.getenv("QDRANT_API_KEY") or os.getenv("QDRANT_API")
            
            if not url:
                raise ValueError("QDRANT_URL or QDRANT_ENDPOINT is not configured in environment variables.")
                
            try:
                from qdrant_client import QdrantClient
                logger.info(f"Connecting to Qdrant cluster at {url}...")
                self.client = QdrantClient(url=url, api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant cluster: {e}")
                raise e

    def collection_exists(self, collection_name: str) -> bool:
        """
        Checks whether the specified collection exists in the Qdrant cluster.
        """
        self.connect()
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.warning(f"Error checking collection existence for '{collection_name}': {e}")
            return False

    def create_collection(self, collection_name: str, vector_size: int, distance: str = "COSINE") -> None:
        """
        Creates a new collection in Qdrant if it does not already exist.
        """
        self.connect()
        if self.collection_exists(collection_name):
            print("Collection already exists.")
            return

        from qdrant_client.http import models
        
        # Translate distance metric
        dist_enum = models.Distance.COSINE
        if distance.upper() == "EUCLIDEAN":
            dist_enum = models.Distance.EUCLID
        elif distance.upper() == "DOT":
            dist_enum = models.Distance.DOT

        logger.info(f"Creating collection '{collection_name}' (dim: {vector_size}, distance: {distance})...")
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=dist_enum
                )
            )
            logger.info(f"Collection '{collection_name}' successfully created.")
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise e

    def upsert_chunks(
        self, 
        collection_name: str, 
        chunks: List[Dict[str, Any]], 
        embedded_chunks: List[Dict[str, Any]], 
        source_pdf: str, 
        batch_size: int = 100
    ) -> None:
        """
        Batch uploads chunk vectors and payloads to Qdrant.
        """
        self.connect()
        
        # Map embeddings by chunk_id
        emb_map = {emb["chunk_id"]: emb["embedding"] for emb in embedded_chunks}
        
        from qdrant_client.http import models
        
        points = []
        for idx, chunk in enumerate(chunks, 1):
            chunk_id = chunk["chunk_id"]
            embedding = emb_map.get(chunk_id)
            if not embedding:
                logger.warning(f"No embedding found for chunk ID '{chunk_id}'. Skipping.")
                continue
                
            # Build payload metadata
            payload = {
                "chunk_id": chunk_id,
                "section_name": chunk["section_name"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "strategy": chunk["strategy"],
                "char_count": chunk["char_count"],
                "source_pdf": source_pdf,
                "collection_name": collection_name,
                "text": chunk["text"]
            }
            
            # Determine integer point ID based on sequential chunk id suffix
            try:
                point_id = int(chunk_id.split("_")[-1])
            except (ValueError, IndexError):
                point_id = idx
                
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
            
        # Batch upsert points
        logger.info(f"Uploading {len(points)} points to collection '{collection_name}' in batches of {batch_size}...")
        try:
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
            logger.info(f"Finished uploading all points to collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to upsert points into collection '{collection_name}': {e}")
            raise e

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Retrieves statistics for the specified collection.
        """
        self.connect()
        try:
            info = self.client.get_collection(collection_name=collection_name)
            
            total_vectors = getattr(info, "points_count", 0) or getattr(info, "vectors_count", 0)
            vector_size = 384
            distance_metric = "COSINE"
            
            config = getattr(info, "config", None)
            if config:
                params = getattr(config, "params", None)
                if params:
                    vectors = getattr(params, "vectors", None)
                    if vectors:
                        if hasattr(vectors, "size"):
                            vector_size = vectors.size
                            distance_metric = getattr(vectors, "distance", "COSINE")
                        elif isinstance(vectors, dict):
                            first_val = next(iter(vectors.values()), None)
                            if first_val and hasattr(first_val, "size"):
                                vector_size = first_val.size
                                distance_metric = getattr(first_val, "distance", "COSINE")

            # Extract clean string for distance
            if hasattr(distance_metric, "name"):
                distance_metric = distance_metric.name
            elif hasattr(distance_metric, "value"):
                distance_metric = distance_metric.value

            return {
                "total_vectors": total_vectors,
                "vector_size": vector_size,
                "distance_metric": str(distance_metric).upper(),
                "status": getattr(info, "status", "UNKNOWN")
            }
        except Exception as e:
            logger.error(f"Failed to retrieve stats for collection '{collection_name}': {e}")
            raise e

    def search(self, collection_name: str, query_vector: List[float], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Searches the specified Qdrant collection with a query vector.
        """
        self.connect()
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit
            )
            ret = []
            for r in results.points:
                payload = r.payload or {}
                chunk_id = payload.get("chunk_id", f"chunk_{r.id:04d}")
                ret.append({
                    "chunk_id": chunk_id,
                    "score": r.score,
                    "text": payload.get("text", ""),
                    "section_name": payload.get("section_name", ""),
                    "payload": payload
                })
            return ret
        except Exception as e:
            logger.error(f"Failed to search collection '{collection_name}': {e}")
            raise e
