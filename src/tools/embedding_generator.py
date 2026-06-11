import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates vector embeddings for validated text chunks using local Sentence Transformers.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def load_model(self) -> None:
        """
        Loads the SentenceTransformer model if it has not been loaded yet.
        """
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model '{self.model_name}': {e}")
                raise e

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates a normalized vector embedding for a single text chunk.
        
        Args:
            text: The text content of the chunk.
            
        Returns:
            A list of floats representing the embedding vector.
        """
        self.load_model()
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates normalized vector embeddings in bulk for a list of chunks.
        
        Args:
            chunks: A list of validated chunk dictionaries.
            
        Returns:
            A list of dictionaries containing chunk ID, section name, embedding vector, and dimensions.
        """
        if not chunks:
            return []
            
        self.load_model()
        texts = [chunk["text"] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks in bulk...")
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        results = []
        for idx, chunk in enumerate(chunks):
            emb_vector = embeddings[idx].tolist() if hasattr(embeddings[idx], "tolist") else list(embeddings[idx])
            results.append({
                "chunk_id": chunk["chunk_id"],
                "section_name": chunk["section_name"],
                "embedding": emb_vector,
                "embedding_dimension": len(emb_vector)
            })
            
        logger.info(f"Successfully generated embeddings for {len(chunks)} chunks.")
        return results
