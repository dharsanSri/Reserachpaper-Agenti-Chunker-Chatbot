import logging
from typing import Dict, Any
import json

from crewai import Crew
from agents.chunk_strategy_agent import get_chunk_strategy_agent
from tasks.strategy_task import create_strategy_task
from models.schemas import ChunkPlanningOutput

logger = logging.getLogger(__name__)

def run_crew(sections_metadata_json: str) -> Dict[str, Any]:
    """
    Runs ONLY the Chunk Strategy Agent and the Strategy Task.
    Does NOT write any output files or use caching.
    """
    logger.info("[CHUNK STRATEGY AGENT] Planning chunk sizes.")

    # 1. Initialize only the Chunk Strategy Agent
    strategy_agent = get_chunk_strategy_agent()

    # 2. Create only the Chunk Strategy Task
    strategy_task = create_strategy_task(sections_metadata_json, agent=strategy_agent)
    # Ensure strategy_task.output_file is None to prevent file writing
    strategy_task.output_file = None

    # 3. Assemble and kickoff the single-task crew
    crew = Crew(
        agents=[strategy_agent],
        tasks=[strategy_task],
        verbose=True
    )
    
    # Run the single task
    results = crew.kickoff()
    logger.info("Chunk Strategy Agent Crew execution completed.")

    # 4. Extract the Pydantic results
    chunk_planning_output = None
    if hasattr(results, "pydantic") and results.pydantic:
        chunk_planning_output = results.pydantic
    else:
        try:
            raw_text = results.raw
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                if lines[0].startswith("```"):
                    lines.pop(0)
                if lines and lines[-1].startswith("```"):
                    lines.pop()
                raw_text = "\n".join(lines).strip()
            data = json.loads(raw_text)
            chunk_planning_output = ChunkPlanningOutput.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse raw output as Pydantic: {e}")
            raise e

    return {
        "chunk_planning_output": chunk_planning_output
    }
