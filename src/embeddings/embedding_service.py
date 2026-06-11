import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    A service class to generate chunk embeddings programmatically using SentenceTransformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model '{model_name}'...")
            self.model = SentenceTransformer(model_name)
            logger.info("SentenceTransformer model successfully loaded.")
        except Exception as e:
            logger.exception(f"Failed to initialize SentenceTransformer model '{model_name}': {e}")
            raise e

    def generate_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Generates embeddings for a list of chunks in batches.
        
        Args:
            chunks: A list of generated chunk dictionaries (must contain 'chunk_id' and 'content').
            batch_size: The batch size for generation.
            
        Returns:
            A list of embedding records containing chunk_id and the embedding float list.
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation.")
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks (batch size: {batch_size})...")
        contents = [c["content"] for c in chunks]
        
        try:
            # Batch encoding via sentence-transformers
            embeddings = self.model.encode(
                contents,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            records = []
            for chunk, emb in zip(chunks, embeddings):
                records.append({
                    "chunk_id": chunk["chunk_id"],
                    "embedding": emb.tolist()
                })
            
            logger.info("Embedding generation completed successfully.")
            return records
        except Exception as e:
            logger.exception(f"Error occurred during embedding generation: {e}")
            raise e

    def embed_and_save(self, chunks_file_path: str, output_file_path: str) -> None:
        """
        Loads repaired chunks, generates embeddings, and saves them to outputs/embeddings.json.
        """
        chunks_path = Path(chunks_file_path)
        output_path = Path(output_file_path)

        if not chunks_path.exists():
            raise FileNotFoundError(f"Missing repaired chunks file at {chunks_path}")

        logger.info(f"Loading repaired chunks from {chunks_path.absolute()}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data.get("chunks", [])
        
        # Generate embeddings
        records = self.generate_embeddings(chunks)

        # Save to output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
            
        logger.info(f"Saved {len(records)} embedding records to {output_path.absolute()}")
