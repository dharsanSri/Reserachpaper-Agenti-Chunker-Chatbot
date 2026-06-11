from crewai import Task
from models.schemas import QueryExpansionOutput

def create_query_expansion_task(query: str, agent) -> Task:
    """
    Creates and returns the Query Expansion Task.
    """
    description = (
        "You are tasked with expanding a user search query into exactly 5 semantically related search queries.\n"
        "Here is the user query:\n"
        f"'{query}'\n\n"
        "Instructions:\n"
        "1. Understand the user's intent from the query.\n"
        "2. Generate alternative phrasings.\n"
        "3. Generate broader concepts related to the query.\n"
        "4. Generate narrower concepts related to the query.\n"
        "5. Generate keyword-focused variants.\n"
        "6. Generate academic/research-oriented variants.\n"
        "7. Ensure the output is a strict JSON object with a list of exactly 5 expanded queries."
    )
    
    return Task(
        description=description,
        expected_output="A strict JSON object containing the list of expanded queries.",
        agent=agent,
        output_pydantic=QueryExpansionOutput,
        cache=False
    )
