from crewai import Agent
from config.llm_factory import get_llm

def get_query_expansion_agent() -> Agent:
    """
    Creates and returns the Query Expansion Agent.
    This agent expands user queries into exactly 5 semantically related search queries.
    """
    return Agent(
        role="Research Query Expansion Specialist",
        goal="Understand the user's intent and generate exactly 5 semantically related search queries to improve retrieval coverage.",
        backstory=(
            "You are a search relevance specialist. You understand user search intent, query phrasings, "
            "broad/narrow concepts, and keyword variations. You generate high-quality academic and research-oriented "
            "search queries to maximize RAG document retrieval."
        ),
        verbose=True,
        llm=get_llm()
    )
