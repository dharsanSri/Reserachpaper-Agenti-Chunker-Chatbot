import re
import math
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ChunkGenerator:
    @staticmethod
    def generate_chunks(document_text: str, section_metadata: Any, chunk_plan: Any) -> List[Dict[str, Any]]:
        """
        Extracts section text using boundaries and splits it programmatically.
        Supports both Pydantic objects and dictionary inputs.
        
        Args:
            document_text: The full text of the PDF.
            section_metadata: Metadata for the section containing boundaries.
            chunk_plan: Plan parameters containing strategy, target size, and overlap size.
            
        Returns:
            A list of generated chunk dictionaries matching the schema.
        """
        # Resolve section metadata values (support dict and object access)
        if isinstance(section_metadata, dict):
            sec_name = section_metadata.get("section_name") or section_metadata.get("title")
            start_pos = section_metadata.get("start_position") or section_metadata.get("start_char", 0)
            end_pos = section_metadata.get("end_position") or section_metadata.get("end_char", len(document_text))
            char_count = section_metadata.get("char_count")
        else:
            sec_name = getattr(section_metadata, "section_name", None) or getattr(section_metadata, "title", None)
            start_pos = getattr(section_metadata, "start_position", 0) or getattr(section_metadata, "start_char", 0)
            end_pos = getattr(section_metadata, "end_position", len(document_text)) or getattr(section_metadata, "end_char", len(document_text))
            char_count = getattr(section_metadata, "char_count", None)

        # Resolve chunk plan parameters (support dict and object access)
        if isinstance(chunk_plan, dict):
            target_size = chunk_plan.get("target_chunk_size", 1000)
            overlap_size = chunk_plan.get("overlap_size", 100)
            strategy = chunk_plan.get("strategy", "semantic")
        else:
            target_size = getattr(chunk_plan, "target_chunk_size", 1000)
            overlap_size = getattr(chunk_plan, "overlap_size", 100)
            strategy = getattr(chunk_plan, "strategy", "semantic")

        # Clamp target_size to prevent oversized chunks violating hard limits
        target_size = min(max(target_size, 300), 1200)

        # Step 1: Extract section text
        section_text = document_text[start_pos:end_pos]
        L = len(section_text)
        
        if L == 0:
            return []

        # Handle sections smaller than 1500 chars (single chunk)
        if L <= 1500:
            raw_chunks = [{"start": 0, "end": L, "text": section_text}]
        else:
            raw_chunks = []
            start = 0
            
            while start < L:
                # Adjust final chunk backwards if too small
                if L - start < 300:
                    if raw_chunks:
                        last_chunk = raw_chunks[-1]
                        if (L - last_chunk["start"]) <= 1500:
                            last_chunk["end"] = L
                            last_chunk["text"] = section_text[last_chunk["start"]:L]
                            break
                    start = max(0, L - target_size)
                    end = L
                    raw_chunks.append({"start": start, "end": end, "text": section_text[start:end]})
                    break
                    
                ideal_end = start + target_size
                if ideal_end >= L:
                    end = L
                    raw_chunks.append({"start": start, "end": end, "text": section_text[start:end]})
                    break
                    
                split_pos = -1
                
                # Apply strategies
                if strategy.lower() in ["paragraph", "hybrid"]:
                    # Look for double newline or single newline
                    search_start = max(start + 300, ideal_end - 200)
                    search_end = min(L, ideal_end + 200)
                    window = section_text[search_start:search_end]
                    
                    idx = window.rfind("\n\n")
                    if idx != -1:
                        split_pos = search_start + idx + 2
                    else:
                        idx = window.rfind("\n")
                        if idx != -1:
                            split_pos = search_start + idx + 1
                else: # semantic
                    # Look for sentence boundary
                    search_start = max(start + 300, ideal_end - 250)
                    search_end = min(L, ideal_end + 250)
                    window = section_text[search_start:search_end]
                    
                    matches = list(re.finditer(r'(?<=[.?!])\s+', window))
                    if matches:
                        split_pos = search_start + matches[-1].end()
                    else:
                        idx = window.rfind("\n")
                        if idx != -1:
                            split_pos = search_start + idx + 1
                            
                # Fallback if no clean split point or split violates size constraints
                if split_pos == -1 or (split_pos - start) > 1500 or (split_pos - start) < 300:
                    end = ideal_end
                else:
                    end = split_pos
                    
                # Enforce hard limits
                if end - start > 1500:
                    end = start + 1200
                if end - start < 300:
                    end = start + 300
                    
                raw_chunks.append({"start": start, "end": end, "text": section_text[start:end]})
                
                # Advance start by end - overlap
                next_start = end - overlap_size
                if next_start <= start:
                    next_start = start + 300  # Enforce positive movement
                start = next_start

        # Step 3: Format chunk objects matching schema
        total_chunks = len(raw_chunks)
        formatted_chunks = []
        
        for idx, rc in enumerate(raw_chunks, 1):
            chunk_global_start = start_pos + rc["start"]
            chunk_global_end = start_pos + rc["end"]
            
            # Format clean chunk id
            chunk_id = f"chunk_{idx:04d}"
            
            formatted_chunks.append({
                "chunk_id": chunk_id,
                "section_name": sec_name,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "start_char": chunk_global_start,
                "end_char": chunk_global_end,
                "char_count": rc["end"] - rc["start"],
                "text": rc["text"],
                "strategy": strategy.lower()
            })
            
        return formatted_chunks
