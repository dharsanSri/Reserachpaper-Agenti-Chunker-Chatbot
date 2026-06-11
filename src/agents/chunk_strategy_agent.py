from crewai import Agent
from config.llm_factory import get_llm

def get_chunk_strategy_agent() -> Agent:
    """
    Creates and returns the Chunk Strategy Agent.
    This agent determines the optimal chunking parameters (count, size, overlap, and strategy)
    per section based on length and document analysis.
    """
    return Agent(
        role="Chunk Planning Specialist",
        goal="Determine optimal chunk counts, target sizes, overlap sizes, and strategies for each document section.",
        backstory=(
            "You are a Search Relevance Architect. You specialize in designing retrieval-optimized "
            "chunking strategies for RAG indexing. You know how to balance document length, topic density, "
            "and model limits using strategies like heading, semantic, paragraph, and hybrid."
        ),
        verbose=True,
        llm=get_llm()
    )
