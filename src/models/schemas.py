from pydantic import BaseModel, Field
from typing import List

class DocumentSection(BaseModel):
    """Pydantic model representing a section boundary and its preview."""
    section_name: str = Field(..., description="The name of the detected section.")
    start_position: int = Field(0, description="The character start index of the section in the full text.")
    end_position: int = Field(0, description="The character end index of the section in the full text.")
    char_count: int = Field(..., description="The character count of the section.")
    content_preview: str = Field("", description="A short preview or summary of the content in this section.")

class DocumentAnalysis(BaseModel):
    """Pydantic model representing high-level analysis and section hierarchy."""
    document_type: str = Field(..., description="The type of document, e.g., Research Paper.")
    sections: List[DocumentSection] = Field(..., description="The detected section hierarchy.")
    section_ordering_valid: bool = Field(True, description="Whether the section ordering is standard and logical.")
    missing_sections: List[str] = Field(default_factory=list, description="Any missing standard sections.")
    observations: str = Field("", description="High-level observations about the document structure.")

class ChunkPlan(BaseModel):
    """Pydantic model representing the planned chunking details for a section."""
    section_name: str = Field(..., description="The name of the section.")
    planned_chunks: int = Field(..., description="The number of chunks planned for this section.")
    target_chunk_size: int = Field(..., description="The target character size for each chunk (500-1200).")
    overlap_size: int = Field(..., description="The character overlap size between chunks (50-150).")
    strategy: str = Field(..., description="The chunking strategy to use (semantic, paragraph, heading, hybrid).")
    reasoning: str = Field(..., description="The reasoning behind this strategy choice.")

class ChunkPlanningOutput(BaseModel):
    """Pydantic model representing the overall chunk plan for the document."""
    plans: List[ChunkPlan] = Field(..., description="The planned chunking strategies for each section.")

class GeneratedChunk(BaseModel):
    """Pydantic model representing a generated chunk of text."""
    chunk_id: str = Field(..., description="The unique sequential identifier of the chunk (e.g., chunk_0001).")
    section_name: str = Field(..., description="The name of the section this chunk belongs to.")
    chunk_index: int = Field(..., description="The 1-based index of this chunk within the section.")
    total_chunks: int = Field(..., description="The total number of chunks generated for this section.")
    strategy: str = Field(..., description="The chunking strategy used to generate this chunk.")
    char_count: int = Field(..., description="The character length of the chunk content.")
    token_estimate: int = Field(..., description="The estimated token count of the chunk (1 token ≈ 4 characters).")
    content: str = Field(..., description="The actual text content of the chunk.")

class GeneratedChunkCollection(BaseModel):
    """Pydantic model representing a collection of generated chunks."""
    chunks: List[GeneratedChunk] = Field(..., description="A list of generated chunks.")

class ChunkQuality(BaseModel):
    """Pydantic model representing quality metrics for a single chunk."""
    chunk_id: str = Field(..., description="The unique identifier of the chunk.")
    semantic_coherence: float = Field(..., description="Score between 0.0 and 1.0 indicating if the chunk discusses a coherent topic.")
    context_preservation: float = Field(..., description="Score between 0.0 and 1.0 indicating if context from neighboring chunks is preserved.")
    boundary_quality: float = Field(..., description="Score between 0.0 and 1.0 indicating if chunk starts/ends naturally.")
    size_quality: float = Field(..., description="Score between 0.0 and 1.0 indicating closeness to optimal size range.")
    retrieval_readiness: float = Field(..., description="Score between 0.0 and 1.0 indicating standalone retrieval utility.")
    overall_score: float = Field(..., description="The weighted average quality score.")
    is_valid: bool = Field(..., description="Whether the chunk satisfies the minimum validation quality threshold.")
    validation_reason: str = Field(..., description="The explanation for the quality score and validity result.")

class ChunkQualityReport(BaseModel):
    """Pydantic model representing a report of chunk quality scores."""
    evaluations: List[ChunkQuality] = Field(..., description="The list of chunk quality metrics.")

class RepairResult(BaseModel):
    """Pydantic model representing the result of a single chunk repair operation."""
    original_chunk_id: str = Field(..., description="The identifier of the chunk prior to repair.")
    repaired_chunk_ids: List[str] = Field(..., description="The identifiers of the chunks produced by the repair.")
    action_taken: str = Field(..., description="The type of repair action executed.")
    success: bool = Field(..., description="Whether the repair was successful.")
    initial_score: float = Field(..., description="The chunk overall score before repair.")
    repaired_score: float = Field(..., description="The chunk overall score after repair.")

class RepairReport(BaseModel):
    """Pydantic model representing a summary of the self-healing loop execution."""
    repaired_chunks: int = Field(..., description="Total number of chunks successfully repaired.")
    merged_chunks: int = Field(..., description="Number of merge operations performed.")
    split_chunks: int = Field(..., description="Number of split operations performed.")
    optimized_chunks: int = Field(..., description="Number of overlap/boundary adjustments made.")
    results: List[RepairResult] = Field(..., description="The detailed results of each repair attempt.")

class EmbeddingRecord(BaseModel):
    """Pydantic model representing a chunk embedding record."""
    chunk_id: str = Field(..., description="The identifier of the chunk.")
    embedding: List[float] = Field(..., description="The generated embedding vector.")

class EmbeddingCollection(BaseModel):
    """Pydantic model representing a list of embedding records."""
    records: List[EmbeddingRecord] = Field(..., description="List of chunk embedding records.")

class QueryExpansionOutput(BaseModel):
    """Pydantic model representing the expanded queries."""
    expanded_queries: List[str] = Field(..., description="List of exactly 5 semantically related search queries.")

class ChunkSelectionOutput(BaseModel):
    """Pydantic model representing the selected chunk IDs."""
    selected_chunk_ids: List[str] = Field(..., description="List of the 5 most relevant chunk IDs.")
