import json
import logging
from typing import Any
from crewai import Task
from crewai.tools import tool
from agents.chunk_strategy_agent import get_chunk_strategy_agent
from models.schemas import ChunkPlanningOutput

logger = logging.getLogger(__name__)

def create_strategy_task(sections_metadata_json: str, agent: Any = None) -> Task:
    """
    Creates the Chunk Strategy task.
    This task uses the document sections metadata passed in parameter to determine chunking parameters,
    and programmatically sanitizes the results.
    """
    assigned_agent = agent if agent is not None else get_chunk_strategy_agent()

    @tool("Sanitize Chunk Plans")
    def sanitize_chunk_plans(plans_json: str) -> str:
        """
        Validates and programmatically corrects the planned chunk parameters based on section size boundaries.
        Args:
            plans_json: A JSON string containing the decided plans list.
        Returns:
            A JSON string containing the sanitized plans list.
        """
        try:
            plans = json.loads(plans_json)
            with open("outputs/document_analysis.json", "r") as f:
                doc_analysis = json.load(f)
            sections = doc_analysis.get("sections", [])
            sec_sizes = {s["section_name"]: s["char_count"] for s in sections}
            
            for plan in plans:
                sec_name = plan["section_name"]
                size = sec_sizes.get(sec_name, 5000)
                
                # Rule 1: Under 1500 chars -> Exactly 1 chunk, 0 overlap
                if size < 1500:
                    plan["planned_chunks"] = 1
                    plan["overlap_size"] = 0
                    plan["target_chunk_size"] = size
                    plan["reasoning"] = f"Sanitized: Section size ({size}) < 1500. Under rule, enforced 1 chunk, 0 overlap."
                else:
                    # Rule 2: Bound target_chunk_size and overlap_size
                    plan["target_chunk_size"] = min(max(plan.get("target_chunk_size", 800), 500), 1200)
                    plan["overlap_size"] = min(max(plan.get("overlap_size", 100), 50), 150)
                    
                    import math
                    computed_chunks = math.ceil(size / plan["target_chunk_size"])
                    
                    if size <= 10000:
                        plan["planned_chunks"] = min(max(computed_chunks, 2), 4)
                    else:
                        plan["planned_chunks"] = min(max(computed_chunks, 4), 10)
            return json.dumps(plans)
        except Exception as e:
            logger.exception(f"Error sanitizing plans: {e}")
            return plans_json

    if assigned_agent.tools is None:
        assigned_agent.tools = []
    # Only assign Sanitize Chunk Plans tool
    assigned_agent.tools = [sanitize_chunk_plans]

    description = (
        "You are tasked with planning the optimal chunking parameters for each section of the document.\n"
        "Here is the list of document sections along with their character counts:\n"
        f"{sections_metadata_json}\n\n"
        "Instructions:\n"
        "1. Read the list of sections and character counts from the provided metadata.\n"
        "2. For each section, select the best strategy (semantic, paragraph, heading, hybrid) and plan parameters.\n"
        "3. Formulate your decided plans, then pass them to the `Sanitize Chunk Plans` tool to enforce sizing rules.\n"
        "4. Output the sanitized results matching the ChunkPlanningOutput schema."
    )

    return Task(
        description=description,
        expected_output="A structured JSON representation containing a list of chunk plans for each section.",
        agent=assigned_agent,
        output_pydantic=ChunkPlanningOutput,
        tools=[sanitize_chunk_plans],
        cache=False
    )
