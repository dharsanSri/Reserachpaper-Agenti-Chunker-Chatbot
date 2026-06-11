import os
import sys
import json
import re
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure stdout/stderr encoding to utf-8 on Windows to prevent charmap encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Configure structured logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("research_paper_chunker")

# Add the src directory to sys.path to ensure imports resolve correctly
src_dir = Path(__file__).resolve().parent
if src_dir not in sys.path:
    sys.path.insert(0, str(src_dir))

from tools.pdf_parser import PDFParser, PDFParserError
from tools.structure_detector import StructureDetector
from crews.chunking_crew import run_crew
from models.exceptions import (
    PDFExtractionError,
    LLMInitializationError,
    CrewExecutionError
)

def main() -> None:
    """
    Main orchestrator implementing the refactored pipeline:
    1. Parse PDF text and page count.
    2. Detect sections programmatically in Python using StructureDetector (no truncation).
    3. Print structure output and calculate coverage percentage.
    4. Save outputs (raw_text.txt, document_analysis.json) to disk.
    5. Pass section metadata to the Chunk Strategy Agent (Groq) to plan chunk sizes.
    6. Print chunk planning output response.
    """
    # Load environment variables
    load_dotenv()

    import argparse
    parser = argparse.ArgumentParser(description="Research Paper Chunker / Query Expansion tool")
    parser.add_argument("--query", type=str, help="User query for expansion")
    args, unknown = parser.parse_known_args()

    if args.query:
        logger.info(f"Running Query Expansion for user query: '{args.query}'")
        from agents.query_expansion_agent import get_query_expansion_agent
        from tasks.query_expansion_task import create_query_expansion_task
        from crewai import Crew
        
        agent = get_query_expansion_agent()
        task = create_query_expansion_task(args.query, agent)
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False
        )
        
        logger.info("Executing Query Expansion Crew...")
        try:
            results = crew.kickoff()
        except Exception as e:
            logger.error(f"Query expansion Crew execution failed: {e}")
            raise CrewExecutionError(f"Error occurred during CrewAI execution: {e}") from e

        # Extract output data
        output_data = results.pydantic
        if not output_data:
            try:
                raw_json = json.loads(results.raw)
                from models.schemas import QueryExpansionOutput
                output_data = QueryExpansionOutput(expanded_queries=raw_json.get("expanded_queries", []))
            except Exception:
                raise ValueError("Query expansion returned invalid or empty output.")

        # Save to cache
        cache_path = Path("cache/query_expansion.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(output_data.model_dump(), f, indent=2)

        # Print expected terminal output format
        print("\n# QUERY EXPANSION OUTPUT\n")
        print(f"Original Query: {args.query}")
        print("Expanded Queries:\n")
        for idx, eq in enumerate(output_data.expanded_queries, 1):
            print(f"{idx}. {eq}")
        print()

        # 10. Qdrant Retrieval and Reciprocal Rank Fusion (RRF)
        # Determine collection name from resources directory PDF
        resources_dir = Path(__file__).resolve().parent.parent / "resources"
        pdf_files = sorted(list(resources_dir.glob("*.pdf")))
        if not pdf_files:
            raise PDFExtractionError("No PDF files found in resources directory for Qdrant search.")
        pdf_path = pdf_files[0]
        pdf_filename = pdf_path.name
        
        # Sanitization
        collection_name = pdf_filename
        if collection_name.lower().endswith(".pdf"):
            collection_name = collection_name[:-4]
        collection_name = collection_name.lower()
        collection_name = re.sub(r'[\s\-]+', '_', collection_name)
        collection_name = re.sub(r'[^a-z0-9_]', '', collection_name)
        
        logger.info(f"Retrieving chunks from collection: {collection_name}")
        
        # Setup embedding generator & vector store
        from tools.embedding_generator import EmbeddingGenerator
        from tools.vector_store import QdrantVectorStore
        
        embedder = EmbeddingGenerator()
        store = QdrantVectorStore()
        store.connect()
        
        # Verify collection exists
        if not store.collection_exists(collection_name):
            raise ValueError(f"Qdrant collection '{collection_name}' does not exist. Please run chunking/indexing first.")
            
        # Collect all queries to search (original + expanded)
        all_queries = [args.query] + output_data.expanded_queries
        
        logger.info(f"Searching Qdrant for {len(all_queries)} queries...")
        
        search_results_by_query = []
        for q in all_queries:
            q_vector = embedder.generate_embedding(q)
            results = store.search(collection_name, q_vector, limit=20)
            search_results_by_query.append(results)
            
        # Apply Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        chunk_map = {}
        k = 60
        
        for results in search_results_by_query:
            for rank, item in enumerate(results, 1):
                chunk_id = item["chunk_id"]
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    chunk_map[chunk_id] = item
                rrf_scores[chunk_id] += 1.0 / (k + rank)
                
        # Sort and take top 20
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:20]
        top_20_chunks = [chunk_map[cid] for cid in sorted_chunk_ids]
        
        logger.info(f"RRF complete. Retained {len(top_20_chunks)} unique chunks.")
        
        # Format Top-20 retrieved chunks as string context
        retrieved_chunks_str = ""
        for idx, chunk in enumerate(top_20_chunks, 1):
            retrieved_chunks_str += (
                f"--- CHUNK {idx} ---\n"
                f"Chunk ID: {chunk['chunk_id']}\n"
                f"Section: {chunk['section_name']}\n"
                f"Score: {chunk['score']}\n"
                f"Content:\n{chunk['text']}\n"
                f"--- END CHUNK {idx} ---\n\n"
            )
            
        # 11. Chunk Selection and Answer Generation without an Agent
        # Take the top 5 chunks directly from RRF (top_20_chunks is already sorted by score)
        top_5_chunks = top_20_chunks[:5]
        selected_chunk_ids = [c["chunk_id"] for c in top_5_chunks]
        
        # Save results to cache/selected_chunks.json
        selection_output = {"selected_chunk_ids": selected_chunk_ids}
        selection_cache_path = Path("cache/selected_chunks.json")
        selection_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(selection_cache_path, "w", encoding="utf-8") as f:
            json.dump(selection_output, f, indent=2)
            
        # Print Chunk Selection Output
        print("\n# CHUNK SELECTION OUTPUT\n")
        print("Query:")
        print(args.query + "\n")
        print("Selected Chunks:")
        for scid in selected_chunk_ids:
            print(scid)
        print()
        
        # Answer Generation
        logger.info("Generating answer directly using Central LLM factory...")
        from config.llm_factory import get_llm
        llm = get_llm()
        
        prompt = (
            "You are a helpful research assistant. Answer the user query using ONLY the provided context chunks.\n"
            "If the context does not contain enough information to answer, state that clearly.\n\n"
            f"Query: {args.query}\n\n"
            "Context Chunks:\n"
        )
        for chunk in top_5_chunks:
            prompt += f"Chunk ID: {chunk['chunk_id']}\nContent:\n{chunk['text']}\n\n"
            
        messages = [{"role": "user", "content": prompt}]
        try:
            answer = llm.call(messages=messages)
        except Exception as e:
            logger.error(f"Direct LLM answer generation failed: {e}")
            answer = f"Error generating answer: {e}"
            
        # Print Answer Generation Output
        print("\n# ANSWER GENERATION OUTPUT\n")
        print("Query:")
        print(args.query + "\n")
        print("Answer:")
        print(answer.strip() + "\n")
        print("Context Used:")
        for scid in selected_chunk_ids:
            print(f"- {scid}")
        print()
        return

    logger.info("Initializing Research Paper Chunker (Programmatic Structure + Strategy Agent Mode)...")

    # PDF Discovery
    resources_dir = Path(__file__).resolve().parent.parent / "resources"
    if not resources_dir.is_dir():
        logger.error(f"Resources directory does not exist at: {resources_dir.absolute()}")
        raise PDFExtractionError(f"Resources folder '{resources_dir.absolute()}' is missing.")

    pdf_files = sorted(list(resources_dir.glob("*.pdf")))
    if not pdf_files:
        logger.error("No PDF files found in resources directory.")
        raise PDFExtractionError("No PDF files found in resources directory.")
    
    pdf_path = pdf_files[0]
    logger.info(f"PDF Discovery: Selected PDF file '{pdf_path.name}'.")

    # Extract text & metadata
    logger.info(f"Reading and parsing PDF file: {pdf_path.name}")
    try:
        page_count = PDFParser.get_page_count(str(pdf_path))
        extracted_text = PDFParser.extract_text(str(pdf_path))
    except Exception as e:
        logger.error(f"PDF text/metadata extraction failed: {e}")
        raise PDFExtractionError(f"Failed to read or parse the PDF document: {e}") from e

    char_count = len(extracted_text)
    logger.info(f"Successfully extracted document text. Pages: {page_count}, Characters: {char_count}")

    # 1. Programmatic Structure Detection
    structure = StructureDetector.detect_structure(extracted_text)
    sections = structure["sections"]
    
    logger.info(f"[STRUCTURE DETECTOR] Detected {len(sections)} sections.")

    # Compute Coverage
    total_covered_chars = sum(sec["char_count"] for sec in sections)
    coverage_pct = (total_covered_chars / char_count) * 100 if char_count > 0 else 0.0

    # 2. Print STRUCTURE DETECTION OUTPUT
    print("\n" + "=" * 50)
    print("STRUCTURE DETECTION OUTPUT")
    print("=" * 26 + "\n")
    print(f"Total Sections: {len(sections)}\n")
    for idx, sec in enumerate(sections, 1):
        print(f"Section {idx}: {sec['section_name']}")
    print(f"\nCoverage %: {coverage_pct:.2f}%")
    print("=" * 50 + "\n")

    # 3. Save Output Files to disk (needed by strategy task's programmatic tool)
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Write raw_text.txt
    with open(outputs_dir / "raw_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    # Write document_analysis.json (fields matched to schemas.py output formatting)
    formatted_analysis = {
        "document_type": structure["document_type"],
        "sections": [
            {
                "section_name": s["section_name"],
                "start_position": s["start_position"],
                "end_position": s["end_position"],
                "char_count": s["char_count"],
                "content_preview": s["content_preview"]
            }
            for s in sections
        ]
    }
    with open(outputs_dir / "document_analysis.json", "w", encoding="utf-8") as f:
        json.dump(formatted_analysis, f, indent=2)

    # 4. Format metadata for Strategy Agent (Only section name and char_count)
    strategy_input_metadata = [
        {
            "section_name": s["section_name"],
            "char_count": s["char_count"]
        }
        for s in sections
    ]
    strategy_input_json = json.dumps(strategy_input_metadata, indent=2)

    # 5. Check and load Chunk Strategy cache (or run Groq Chunk Strategy Agent Crew)
    from utils.cache_manager import (
        get_chunk_plan_cache,
        save_chunk_plan_cache,
        validate_chunk_plan_cache
    )
    
    pdf_filename = pdf_path.name
    cached_data = get_chunk_plan_cache(pdf_filename)
    
    use_cache = False
    if cached_data:
        if validate_chunk_plan_cache(cached_data, sections):
            use_cache = True
            
    if use_cache:
        # Load from cache
        from models.schemas import ChunkPlanningOutput, ChunkPlan
        plans_list = []
        for p in cached_data["plans"]:
            plans_list.append(ChunkPlan(
                section_name=p["section_name"],
                planned_chunks=p["planned_chunks"],
                target_chunk_size=p["target_chunk_size"],
                overlap_size=p["overlap_size"],
                strategy=p["strategy"],
                reasoning=p["reasoning"]
            ))
        chunk_planning_output = ChunkPlanningOutput(plans=plans_list)
        
        # Get cache path for display
        base = pdf_filename
        if base.lower().endswith(".pdf"):
            base = base[:-4]
        base = base.lower()
        base = re.sub(r'[\s\-]+', '_', base)
        base = re.sub(r'[^a-z0-9_]', '', base)
        cache_path_str = f"cache/chunk_plans/{base}.json"
        
        # Case 1 output
        print("\n" + "=" * 50)
        print("CHUNK PLAN CACHE HIT")
        print("=" * 50)
        print("PDF:")
        print(pdf_filename + "\n")
        print("Cache File:")
        print(cache_path_str + "\n")
        print("Status:")
        print("Loaded Successfully\n")
        print("Groq Calls Saved:")
        print("1")
        print("=" * 50 + "\n")
    else:
        # Case 2 output
        print("\n" + "=" * 50)
        print("CHUNK PLAN CACHE MISS")
        print("=" * 50)
        print("No existing chunk plan found.\n")
        print("Running Chunk Strategy Agent...\n")
        
        logger.info("[CHUNK STRATEGY AGENT] Planning chunk sizes.")
        try:
            results = run_crew(strategy_input_json)
        except Exception as e:
            logger.error(f"Strategy planning Crew execution failed: {e}")
            raise CrewExecutionError(f"Error occurred during CrewAI execution: {e}") from e

        chunk_planning_output = results.get("chunk_planning_output")
        
        if chunk_planning_output:
            save_chunk_plan_cache(pdf_filename, chunk_planning_output)
            print("Saving plan to cache...\n")
            print("Status:")
            print("Success")
        else:
            print("Status:")
            print("Failed (No output)")
        print("=" * 50 + "\n")

    # 7. Programmatic Chunk Generation
    from tools.chunk_generator import ChunkGenerator
    
    plans_by_section = {}
    if chunk_planning_output and chunk_planning_output.plans:
        plans_by_section = {p.section_name: p for p in chunk_planning_output.plans}
        
    print("\n" + "=" * 50)
    print("CHUNK GENERATION REPORT")
    print("=" * 50 + "\n")
    
    all_chunks = []
    
    for idx, sec in enumerate(sections, 1):
        sec_name = sec["section_name"]
        plan = plans_by_section.get(sec_name)
        if not plan:
            # Safe programmatic fallback if Strategy Agent missed a section
            plan = {
                "target_chunk_size": 1000,
                "overlap_size": 100 if (sec["char_count"] > 1000) else 0,
                "strategy": "paragraph" if "REFERENCES" in sec_name.upper() else ("hybrid" if sec["char_count"] > 5000 else "semantic")
            }
            
        # Generate chunks for this section
        chunks = ChunkGenerator.generate_chunks(extracted_text, sec, plan)
        all_chunks.extend(chunks)
        
        # Compute metrics
        num_chunks = len(chunks)
        if num_chunks > 0:
            sizes = [c["char_count"] for c in chunks]
            avg_size = sum(sizes) / num_chunks
            min_size = min(sizes)
            max_size = max(sizes)
            
            # Count unique character coverage
            unique_indices = set()
            for chunk in chunks:
                for i in range(chunk["start_char"], chunk["end_char"]):
                    unique_indices.add(i)
            sec_len = sec["end_position"] - sec["start_position"]
            coverage_pct = (len(unique_indices) / sec_len) * 100 if sec_len > 0 else 0.0
        else:
            avg_size = 0.0
            min_size = 0
            max_size = 0
            coverage_pct = 0.0
            
        print(f"Section: {sec_name}")
        print(f"Total Chunks: {num_chunks}")
        print(f"Average Chunk Size: {avg_size:.2f}")
        print(f"Min Chunk Size: {min_size}")
        print(f"Max Chunk Size: {max_size}")
        print(f"Coverage %: {coverage_pct:.2f}%")
        print("-" * 48)
        
        # Print every chunk
        for chunk in chunks:
            print(f"Chunk {chunk['chunk_index']}")
            print(f"Size: {chunk['char_count']}")
            # Clean preview without newlines
            clean_preview = chunk["text"][:150].strip().replace("\n", " ")
            if len(chunk["text"]) > 150:
                clean_preview += "..."
            print(f"Preview: {clean_preview}\n")
            
        print("=" * 50 + "\n")

    # Rename chunk_ids to be globally sequential
    for idx, chunk in enumerate(all_chunks, 1):
        chunk["chunk_id"] = f"chunk_{idx:04d}"

    # 8. Programmatic Chunk Validation
    logger.info("Running programmatic chunk validation...")
    from tools.chunk_validator import ChunkValidator
    
    validation_results, summary = ChunkValidator.validate_chunks(all_chunks, sections)
    
    if summary["failed"] > 0:
        logger.error(f"Validation failed: {summary['failed']} chunks failed verification rules.")
        raise ValueError(f"Chunk validation failed with {summary['failed']} failed chunks. Stopping pipeline.")
        
    print("\n" + "=" * 50)
    print("VALIDATION PASSED")
    print("=" * 50 + "\n")

    # 9. Embedding Generation Pipeline
    logger.info("Running programmatic embedding generation pipeline...")
    from tools.embedding_generator import EmbeddingGenerator
    
    embedder = EmbeddingGenerator()
    embedded_chunks = embedder.generate_embeddings(all_chunks)
    
    print("\n" + "=" * 50)
    print("EMBEDDING GENERATION REPORT")
    print("=" * 50)
    print(f"Total Embedded Chunks: {len(embedded_chunks)}")
    if embedded_chunks:
        print(f"Model:                 {embedder.model_name}")
        print(f"Embedding Dimension:   {embedded_chunks[0]['embedding_dimension']}")
    print("=" * 50 + "\n")

    # 10. Qdrant Storage Layer
    import time
    
    # Generate sanitized collection name from PDF filename
    pdf_filename = pdf_path.name
    # 1. Remove ".pdf"
    collection_name = pdf_filename
    if collection_name.lower().endswith(".pdf"):
        collection_name = collection_name[:-4]
    # 2. Convert to lowercase
    collection_name = collection_name.lower()
    # 3. Replace spaces with "_"
    # 4. Replace hyphens with "_"
    collection_name = re.sub(r'[\s\-]+', '_', collection_name)
    # 5. Remove invalid characters (keep only alphanumeric and underscores)
    collection_name = re.sub(r'[^a-z0-9_]', '', collection_name)
    
    logger.info(f"Generated collection name: {collection_name}")
    
    from tools.vector_store import QdrantVectorStore
    
    store = QdrantVectorStore()
    store.connect()
    
    # Check if exists, print, then create
    if store.collection_exists(collection_name):
        print("Collection already exists.")
    else:
        # Determine vector size from the first embedding
        vector_size = embedded_chunks[0]["embedding_dimension"] if embedded_chunks else 384
        store.create_collection(collection_name, vector_size, distance="COSINE")
        
    logger.info("Uploading chunks and embeddings to Qdrant...")
    start_time = time.time()
    store.upsert_chunks(
        collection_name=collection_name,
        chunks=all_chunks,
        embedded_chunks=embedded_chunks,
        source_pdf=pdf_filename,
        batch_size=100
    )
    upload_time = time.time() - start_time
    
    # Retrieve stats
    stats = store.get_collection_stats(collection_name)
    uploaded_vectors = stats["total_vectors"]
    
    # Print Storage Report
    print("\n" + "=" * 50)
    print("QDRANT STORAGE REPORT")
    print("=" * 50)
    print("Cluster:")
    print("research_chunker\n")
    print("Collection:")
    print(collection_name + "\n")
    print(f"Vector Dimension:\n{stats['vector_size']}\n")
    print(f"Vectors Uploaded:\n{len(embedded_chunks)}\n")
    print(f"Upload Time:\n{upload_time:.2f} sec\n")
    print("Status:")
    is_success = uploaded_vectors == len(embedded_chunks)
    print("SUCCESS" if is_success else "FAIL")
    print("=" * 50 + "\n")
    
    # Print Collection Summary
    print("Collection Name:")
    print(collection_name)
    print("Vector Count:")
    print(uploaded_vectors)
    print("Embedding Dimension:")
    print(stats["vector_size"])
    print("Distance Metric:")
    print(stats["distance_metric"])
    print()
    
    # Validate Success Criteria
    if not is_success:
        logger.error(f"Storage validation failed: Qdrant count ({uploaded_vectors}) does not match generated count ({len(embedded_chunks)}).")
        raise ValueError(f"Qdrant storage verification failed: uploaded_vectors ({uploaded_vectors}) != generated_embeddings ({len(embedded_chunks)}). Stopping pipeline.")

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logger.critical(f"Pipeline crashed: {ex}", exc_info=True)
        sys.exit(1)
