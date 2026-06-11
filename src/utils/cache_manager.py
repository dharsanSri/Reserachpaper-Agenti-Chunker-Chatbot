import os
import re
import json
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache/chunk_plans")

def _get_cache_path(pdf_name: str) -> Path:
    """
    Sanitizes the PDF filename to generate a unique cache filename and returns the Path.
    """
    base = pdf_name
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    base = base.lower()
    base = re.sub(r'[\s\-]+', '_', base)
    base = re.sub(r'[^a-z0-9_]', '', base)
    return CACHE_DIR / f"{base}.json"

def get_chunk_plan_cache(pdf_name: str) -> Dict[str, Any] | None:
    """
    Checks if a cached chunk plan exists and returns it as a dict, or None.
    """
    cache_file = _get_cache_path(pdf_name)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cached chunk plan from {cache_file}: {e}")
            return None
    return None

def save_chunk_plan_cache(pdf_name: str, chunk_planning_output: Any) -> None:
    """
    Saves the chunk planning output to the cache directory.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _get_cache_path(pdf_name)
    
    # Serialize chunk_planning_output (support both Pydantic models and dict inputs)
    plans_list = []
    
    if hasattr(chunk_planning_output, "plans"):
        plans = chunk_planning_output.plans
    elif isinstance(chunk_planning_output, dict):
        plans = chunk_planning_output.get("plans", [])
    else:
        plans = []
        
    for plan in plans:
        if isinstance(plan, dict):
            plans_list.append(plan)
        else:
            plans_list.append({
                "section_name": getattr(plan, "section_name", ""),
                "planned_chunks": getattr(plan, "planned_chunks", 0),
                "target_chunk_size": getattr(plan, "target_chunk_size", 0),
                "overlap_size": getattr(plan, "overlap_size", 0),
                "strategy": getattr(plan, "strategy", ""),
                "reasoning": getattr(plan, "reasoning", "")
            })
            
    cache_data = {
        "pdf_name": pdf_name,
        "generated_at": datetime.datetime.now().isoformat(),
        "plans": plans_list
    }
    
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Chunk plan saved to cache at {cache_file}")
    except Exception as e:
        logger.error(f"Failed to save chunk plan to cache: {e}")

def validate_chunk_plan_cache(cached_data: Dict[str, Any], current_sections: List[Dict[str, Any]]) -> bool:
    """
    Validates the cached chunk plan data against current detected sections.
    """
    if not isinstance(cached_data, dict):
        return False
        
    plans = cached_data.get("plans")
    if not isinstance(plans, list) or len(plans) == 0:
        return False
        
    # Extract section names from cache
    cached_section_names = []
    for plan in plans:
        if isinstance(plan, dict) and "section_name" in plan:
            cached_section_names.append(plan["section_name"])
            
    # Extract section names from current detection
    current_section_names = []
    for sec in current_sections:
        if isinstance(sec, dict) and "section_name" in sec:
            current_section_names.append(sec["section_name"])
        elif hasattr(sec, "section_name"):
            current_section_names.append(getattr(sec, "section_name"))
            
    # Check that cached section names match current structure detection in both order and text
    return cached_section_names == current_section_names
