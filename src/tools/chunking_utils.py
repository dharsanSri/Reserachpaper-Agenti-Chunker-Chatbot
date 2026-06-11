import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Matches subheadings like "A. Modalities", "1. Convolution", "1) Max Pooling", "I. Related"
SUBHEADING_PATTERN = re.compile(
    r'^\s*(?:[A-Z]\.\s+|[0-9]+\.\s+|[0-9]+\)\s+|[a-z]\)\s+|[IVXLCDM]+\.\s+)[A-Za-z]'
)

_embedding_model = None

def get_embedding_model():
    """Lazy loads the SentenceTransformer model and caches it."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2' for local semantic chunking...")
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            raise e
    return _embedding_model

def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences, avoiding false splits on common abbreviations."""
    sentence_end = re.compile(
        r'(?<!\be\.g)(?<!\bi\.e)(?<!\bFig)(?<!\bal)(?<!\bvol)(?<!\bpp)'
        r'(?<!\bDr)(?<!\bvs)(?<!\bNo)(?<!\bSt)(?<!\bJan)(?<!\bFeb)'
        r'(?<!\bMar)(?<!\bApr)(?<!\bJun)(?<!\bJul)(?<!\bAug)(?<!\bSep)'
        r'(?<!\bOct)(?<!\bNov)(?<!\bDec)(?<=[.!?])\s+'
    )
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]

def partition_text_into_logical_units(text: str) -> List[str]:
    """
    Partitions raw section content into an ordered list of logical text units (sentences, tables, lists).
    Ensures that tables, bullet lists, and figure/table descriptions are kept whole.
    """
    lines = text.split("\n")
    logical_units = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 1. Skip empty lines
        if not stripped:
            i += 1
            continue
            
        # 2. Check for Table block (lines containing pipe symbols)
        if "|" in line:
            table_lines = []
            while i < len(lines) and ("|" in lines[i] or not lines[i].strip()):
                if lines[i].strip():
                    table_lines.append(lines[i])
                i += 1
            if table_lines:
                logical_units.append("\n".join(table_lines))
            continue
            
        # 3. Check for Bullet or Numbered List block
        list_marker = re.match(r'^\s*(?:[\*\-\+]\s|\d+\.\s)', line)
        if list_marker:
            list_lines = []
            while i < len(lines):
                curr_line = lines[i]
                curr_stripped = curr_line.strip()
                if curr_stripped:
                    is_curr_list = re.match(r'^\s*(?:[\*\-\+]\s|\d+\.\s)', curr_line)
                    if not is_curr_list and curr_line.startswith(curr_stripped):
                        break
                    list_lines.append(curr_line)
                else:
                    if i + 1 < len(lines) and (lines[i+1].startswith(" ") or lines[i+1].startswith("\t")):
                        list_lines.append("")
                    else:
                        break
                i += 1
            if list_lines:
                while list_lines and not list_lines[-1].strip():
                    list_lines.pop()
                logical_units.append("\n".join(list_lines))
            continue
            
        # 4. Check for Figure/Table captions
        caption_marker = re.match(r'^\s*(?:Figure|Fig\.|Table|TABLE)\s+\d+', line, re.IGNORECASE)
        if caption_marker:
            caption_lines = []
            while i < len(lines) and lines[i].strip():
                caption_lines.append(lines[i])
                i += 1
            if caption_lines:
                logical_units.append("\n".join(caption_lines))
            continue
            
        # 5. Standard Text Paragraph
        paragraph_lines = []
        while i < len(lines):
            curr_line = lines[i]
            curr_stripped = curr_line.strip()
            if not curr_stripped:
                break
            if "|" in curr_line or re.match(r'^\s*(?:[\*\-\+]\s|\d+\.\s)', curr_line) or re.match(r'^\s*(?:Figure|Fig\.|Table|TABLE)\s+\d+', curr_line, re.IGNORECASE):
                break
            paragraph_lines.append(curr_line)
            i += 1
            
        if paragraph_lines:
            paragraph_text = " ".join([l.strip() for l in paragraph_lines])
            sentences = split_into_sentences(paragraph_text)
            logical_units.extend(sentences)
            
    return logical_units

def partition_into_paragraphs(text: str) -> List[str]:
    """Partitions raw section content into paragraphs, list blocks, tables, or figure descriptions."""
    lines = text.split("\n")
    paragraph_blocks = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 1. Skip empty lines
        if not stripped:
            i += 1
            continue
            
        # 2. Check for Table block
        if "|" in line:
            table_lines = []
            while i < len(lines) and ("|" in lines[i] or not lines[i].strip()):
                if lines[i].strip():
                    table_lines.append(lines[i])
                i += 1
            if table_lines:
                paragraph_blocks.append("\n".join(table_lines))
            continue
            
        # 3. Check for Bullet or Numbered List block
        list_marker = re.match(r'^\s*(?:[\*\-\+]\s|\d+\.\s)', line)
        if list_marker:
            list_lines = []
            while i < len(lines):
                curr_line = lines[i]
                curr_stripped = curr_line.strip()
                if curr_stripped:
                    is_curr_list = re.match(r'^\s*(?:[\*\-\+]\s|\d+\.\s)', curr_line)
                    if not is_curr_list and curr_line.startswith(curr_stripped):
                        break
                    list_lines.append(curr_line)
                else:
                    if i + 1 < len(lines) and (lines[i+1].startswith(" ") or lines[i+1].startswith("\t")):
                        list_lines.append("")
                    else:
                        break
                i += 1
            if list_lines:
                while list_lines and not list_lines[-1].strip():
                    list_lines.pop()
                paragraph_blocks.append("\n".join(list_lines))
            continue
            
        # 4. Check for Figure/Table captions
        caption_marker = re.match(r'^\s*(?:Figure|Fig\.|Table|TABLE)\s+\d+', line, re.IGNORECASE)
        if caption_marker:
            caption_lines = []
            while i < len(lines) and lines[i].strip():
                caption_lines.append(lines[i])
                i += 1
            if caption_lines:
                paragraph_blocks.append("\n".join(caption_lines))
            continue
            
        # 5. Standard Paragraph
        paragraph_lines = []
        while i < len(lines):
            curr_line = lines[i]
            curr_stripped = curr_line.strip()
            if not curr_stripped:
                break
            if "|" in curr_line or re.match(r'^\s*(?:[\*\-\+]\s|\d+\.\s)', curr_line) or re.match(r'^\s*(?:Figure|Fig\.|Table|TABLE)\s+\d+', curr_line, re.IGNORECASE):
                break
            paragraph_lines.append(curr_line)
            i += 1
            
        if paragraph_lines:
            paragraph_blocks.append(" ".join([l.strip() for l in paragraph_lines]))
            
    return paragraph_blocks

def create_chunks_with_overlap(logical_units: List[str], target_size: int, overlap_size: int) -> List[str]:
    """Groups logical units into coherent text chunks respecting target size and overlap boundaries."""
    chunks = []
    current_units = []
    current_len = 0
    
    i = 0
    while i < len(logical_units):
        unit = logical_units[i]
        unit_len = len(unit)
        
        # If adding this unit exceeds target size, finalize the current chunk
        if current_len + unit_len > target_size and current_units:
            chunk_content = "\n".join(current_units)
            chunks.append(chunk_content)
            
            # Carry over overlapping units
            overlap_units = []
            overlap_len = 0
            for u in reversed(current_units):
                if overlap_len + len(u) <= overlap_size:
                    overlap_units.insert(0, u)
                    overlap_len += len(u) + 1  # +1 for newline separator
                else:
                    break
            
            # Ensure at least one unit is carried over if overlap is specified
            if not overlap_units and current_units and overlap_size > 0:
                overlap_units = [current_units[-1]]
                overlap_len = len(current_units[-1])
                
            current_units = overlap_units
            current_len = overlap_len
            
        current_units.append(unit)
        current_len += unit_len + (1 if current_len > 0 else 0)
        i += 1
        
    if current_units:
        chunks.append("\n".join(current_units))
        
    return chunks

def chunk_by_paragraph(text: str, target_size: int, overlap_size: int) -> List[str]:
    """Splits text by paragraph boundaries, keeping paragraph structures intact."""
    logger.info(f"Chunking paragraph-based text (length: {len(text)}, target: {target_size}, overlap: {overlap_size})")
    paragraphs = partition_into_paragraphs(text)
    return create_chunks_with_overlap(paragraphs, target_size, overlap_size)

def chunk_by_heading(text: str, target_size: int, overlap_size: int) -> List[str]:
    """Splits text by subheading. If a subsection exceeds target_size, falls back to paragraph splitting."""
    logger.info(f"Chunking heading-based text (length: {len(text)}, target: {target_size}, overlap: {overlap_size})")
    lines = text.split("\n")
    
    heading_blocks = []
    current_block = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        is_subheading = SUBHEADING_PATTERN.match(line) and len(stripped) < 80
        if is_subheading and current_block:
            heading_blocks.append("\n".join(current_block))
            current_block = []
            
        current_block.append(line)
        
    if current_block:
        heading_blocks.append("\n".join(current_block))
        
    final_chunks = []
    for block in heading_blocks:
        if len(block) <= target_size:
            final_chunks.append(block)
        else:
            sub_chunks = chunk_by_paragraph(block, target_size, overlap_size)
            final_chunks.extend(sub_chunks)
            
    return final_chunks

def chunk_hybrid(text: str, target_size: int, overlap_size: int) -> List[str]:
    """Splits text using subheading, falling back to paragraph, then to logical sentence structures."""
    logger.info(f"Chunking hybrid text (length: {len(text)}, target: {target_size}, overlap: {overlap_size})")
    
    # Split by subheadings
    lines = text.split("\n")
    heading_blocks = []
    current_block = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        is_subheading = SUBHEADING_PATTERN.match(line) and len(stripped) < 80
        if is_subheading and current_block:
            heading_blocks.append("\n".join(current_block))
            current_block = []
            
        current_block.append(line)
        
    if current_block:
        heading_blocks.append("\n".join(current_block))
        
    final_chunks = []
    for block in heading_blocks:
        if len(block) <= target_size:
            final_chunks.append(block)
        else:
            paragraphs = partition_into_paragraphs(block)
            units = []
            for para in paragraphs:
                if len(para) <= target_size:
                    units.append(para)
                else:
                    sub_units = partition_text_into_logical_units(para)
                    units.extend(sub_units)
            
            sub_chunks = create_chunks_with_overlap(units, target_size, overlap_size)
            final_chunks.extend(sub_chunks)
            
    return final_chunks

def chunk_semantically(text: str, target_size: int, overlap_size: int) -> List[str]:
    """
    Splits text semantically using SentenceTransformer embeddings.
    Groups sentences into segments by detecting significant changes in cosine similarity.
    """
    logger.info(f"Performing local semantic chunking (length: {len(text)}, target: {target_size}, overlap: {overlap_size})")
    logical_units = partition_text_into_logical_units(text)
    if not logical_units or len(logical_units) <= 2:
        return create_chunks_with_overlap(logical_units, target_size, overlap_size)
        
    try:
        model = get_embedding_model()
        embeddings = model.encode(logical_units, convert_to_tensor=True)
        
        from sentence_transformers.util import cosine_similarity
        import numpy as np
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i+1]).item()
            similarities.append(sim)
            
        if similarities:
            # Splits occur where similarity falls below the 20th percentile (topic shifts)
            threshold = min(0.65, np.percentile(similarities, 20))
            split_indices = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
        else:
            split_indices = []
            
        semantic_segments = []
        current_segment = []
        for idx, unit in enumerate(logical_units):
            if idx in split_indices and current_segment:
                semantic_segments.append("\n".join(current_segment))
                current_segment = []
            current_segment.append(unit)
            
        if current_segment:
            semantic_segments.append("\n".join(current_segment))
            
        final_chunks = []
        for segment in semantic_segments:
            if len(segment) <= target_size:
                final_chunks.append(segment)
            else:
                sub_chunks = create_chunks_with_overlap(
                    partition_text_into_logical_units(segment),
                    target_size,
                    overlap_size
                )
                final_chunks.extend(sub_chunks)
        return final_chunks
    except Exception as e:
        logger.warning(f"Local semantic chunking failed: {e}. Falling back to paragraph heuristics.")
        return chunk_by_paragraph(text, target_size, overlap_size)

def generate_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
    section: str = "Unknown",
    start_idx: int = 1
) -> List[dict]:
    """
    Deterministic chunk generator.
    Splits text into chunks preserving paragraph and sentence boundaries, avoiding sentence fragmentation, and maintaining overlap.
    
    Returns a list of dicts:
    [
      {
        "chunk_id": "chunk_0001",
        "content": "...",
        "char_count": 985,
        "section": "Introduction"
      }
    ]
    """
    # We use chunk_hybrid because it splits by heading, falling back to paragraph, then logical sentence structures.
    # This prevents sentence fragmentation and preserves paragraph boundaries.
    chunks_content = chunk_hybrid(text, target_size=chunk_size, overlap_size=overlap)
    
    results = []
    for idx, content in enumerate(chunks_content, start_idx):
        results.append({
            "chunk_id": f"chunk_{idx:04d}",
            "content": content,
            "char_count": len(content),
            "section": section
        })
    return results

