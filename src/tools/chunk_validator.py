import re
import hashlib
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ChunkValidator:
    """
    Validates generated chunks using deterministic Python rules.
    No LLMs, embeddings, or vector databases are used.
    """

    @staticmethod
    def validate_chunks(
        chunks: List[Dict[str, Any]], 
        sections_metadata: List[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validates all chunks against the 8 validation rules.
        
        Args:
            chunks: A list of generated chunk dictionaries.
            sections_metadata: Optional list of section metadata dictionaries.
            
        Returns:
            A tuple containing:
            1. A list of ValidationResult dictionaries.
            2. A Summary dictionary.
        """
        logger.info(f"Starting programmatic validation for {len(chunks)} chunks...")

        # Resolve sections metadata
        if sections_metadata is None:
            try:
                analysis_path = Path("outputs/document_analysis.json")
                if analysis_path.exists():
                    with open(analysis_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        sections_metadata = data.get("sections", [])
            except Exception as e:
                logger.warning(f"Could not load sections_metadata from outputs/document_analysis.json: {e}")

        # Fallback if still not available
        if not sections_metadata:
            sections_metadata = []
            sec_to_chunks = {}
            for c in chunks:
                sec_name = c["section_name"]
                if sec_name not in sec_to_chunks:
                    sec_to_chunks[sec_name] = []
                sec_to_chunks[sec_name].append(c)
            for sec_name, s_chunks in sec_to_chunks.items():
                min_start = min(c["start_char"] for c in s_chunks)
                max_end = max(c["end_char"] for c in s_chunks)
                sections_metadata.append({
                    "section_name": sec_name,
                    "start_position": min_start,
                    "end_position": max_end,
                    "char_count": max_end - min_start
                })

        # Calculate coverages per section first (Rule 4)
        section_coverages = {}
        for sec in sections_metadata:
            sec_name = sec["section_name"]
            sec_start = sec.get("start_position", 0)
            sec_end = sec.get("end_position", 0)
            sec_len = sec.get("char_count") or (sec_end - sec_start)
            
            sec_chunks = [c for c in chunks if c["section_name"] == sec_name]
            if not sec_chunks:
                section_coverages[sec_name] = 0.0 if sec_len > 0 else 1.0
                continue
                
            # Merge ranges
            intervals = []
            for c in sec_chunks:
                c_start = max(c["start_char"], sec_start)
                c_end = min(c["end_char"], sec_end)
                if c_start < c_end:
                    intervals.append((c_start, c_end))
            
            intervals.sort()
            merged = []
            for start, end in intervals:
                if not merged or merged[-1][1] < start:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            
            covered_chars = sum(end - start for start, end in merged)
            coverage = (covered_chars / sec_len) if sec_len > 0 else 1.0
            section_coverages[sec_name] = coverage

        # Duplicate checking (Rule 2)
        text_hashes = {}
        for c in chunks:
            text_hash = hashlib.md5(c["text"].strip().encode("utf-8")).hexdigest()
            if text_hash not in text_hashes:
                text_hashes[text_hash] = []
            text_hashes[text_hash].append(c["chunk_id"])

        # Validate each chunk
        results = []
        passed_count = 0
        warning_count = 0
        failed_count = 0

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            sec_name = chunk["section_name"]
            text = chunk["text"]
            char_count = chunk["char_count"]
            start = chunk["start_char"]
            end = chunk["end_char"]
            
            issues = []
            status = "PASS"
            is_fail = False
            is_warning = False

            # RULE 1: No empty chunks
            if len(text.strip()) == 0:
                issues.append("Rule 1 Fail: Chunk text is empty.")
                is_fail = True

            # RULE 2: No duplicate chunks
            text_hash = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
            duplicate_ids = text_hashes[text_hash]
            if len(duplicate_ids) > 1:
                other_ids = [tid for tid in duplicate_ids if tid != chunk_id]
                issues.append(f"Rule 2 Fail: Duplicate chunk detected (shares identical content with: {', '.join(other_ids)}).")
                is_fail = True

            # RULE 3: Chunk size limits
            if char_count > 1500:
                issues.append(f"Rule 3 Fail: Chunk size ({char_count}) exceeds maximum limit of 1500 characters.")
                is_fail = True
            elif char_count < 300:
                # Exception: Entire section is smaller than 300 characters
                sec_metadata = next((s for s in sections_metadata if s["section_name"] == sec_name), None)
                sec_len = 0
                if sec_metadata:
                    sec_len = sec_metadata.get("char_count") or (sec_metadata.get("end_position", 0) - sec_metadata.get("start_position", 0))
                if sec_len >= 300:
                    issues.append(f"Rule 3 Warning: Chunk size ({char_count}) is below 300 characters (Section length: {sec_len}).")
                    is_warning = True

            # RULE 4: Coverage validation
            coverage = section_coverages.get(sec_name, 0.0)
            if coverage < 0.95:
                issues.append(f"Rule 4 Fail: Section '{sec_name}' coverage is {coverage * 100:.2f}%, which is below the 95% threshold.")
                is_fail = True

            # RULE 5: Overlap validation
            if chunk["chunk_index"] > 1:
                # Find previous chunk in the same section
                sec_chunks_sorted = sorted([c for c in chunks if c["section_name"] == sec_name], key=lambda x: x["chunk_index"])
                prev_chunk = next((c for c in sec_chunks_sorted if c["chunk_index"] == chunk["chunk_index"] - 1), None)
                if prev_chunk:
                    overlap = prev_chunk["end_char"] - start
                    if not (90 <= overlap <= 110):
                        issues.append(f"Rule 5 Fail: Overlap with chunk {prev_chunk['chunk_id']} is {overlap} characters, which is outside the expected 100 ± 10 range.")
                        is_fail = True

            # RULE 6: Boundary validation
            if start >= end:
                issues.append(f"Rule 6 Fail: start_char ({start}) is not less than end_char ({end}).")
                is_fail = True
            
            sec_metadata = next((s for s in sections_metadata if s["section_name"] == sec_name), None)
            if sec_metadata:
                sec_start = sec_metadata.get("start_position", 0)
                sec_end = sec_metadata.get("end_position", 0)
                if end > sec_end:
                    issues.append(f"Rule 6 Fail: end_char ({end}) exceeds section_end ({sec_end}).")
                    is_fail = True
                if start < sec_start:
                    issues.append(f"Rule 6 Fail: start_char ({start}) is less than section_start ({sec_start}).")
                    is_fail = True

            # RULE 7: Text quality checks
            quality_warnings = ChunkValidator._check_text_quality(text, sec_name)
            for qw in quality_warnings:
                issues.append(f"Rule 7 Warning: {qw}")
                is_warning = True

            # RULE 8: Section consistency
            # Check range crosses boundaries
            if sec_metadata:
                sec_start = sec_metadata.get("start_position", 0)
                sec_end = sec_metadata.get("end_position", 0)
                if start < sec_start or end > sec_end:
                    issues.append(f"Rule 8 Fail: Chunk boundaries [{start}, {end}] cross outside section '{sec_name}' [{sec_start}, {sec_end}].")
                    is_fail = True

            # Check if other section headings appear in chunk text
            for other_sec in sections_metadata:
                other_name = other_sec["section_name"]
                if other_name == sec_name:
                    continue
                # If section name is present on a line by itself in the text
                lines = [l.strip() for l in text.split("\n")]
                if any(line == other_name for line in lines):
                    issues.append(f"Rule 8 Fail: Chunk contains heading line of other section '{other_name}'.")
                    is_fail = True

            # Set status
            if is_fail:
                status = "FAIL"
                failed_count += 1
            elif is_warning:
                status = "WARNING"
                warning_count += 1
            else:
                passed_count += 1

            results.append({
                "chunk_id": chunk_id,
                "section_name": sec_name,
                "status": status,
                "issues": issues
            })

        # Calculate overall document coverage percentage
        total_section_len = sum((s.get("char_count") or (s.get("end_position", 0) - s.get("start_position", 0))) for s in sections_metadata)
        
        # Calculate overall covered chars across all sections
        total_covered_chars = 0
        for sec in sections_metadata:
            sec_name = sec["section_name"]
            sec_start = sec.get("start_position", 0)
            sec_end = sec.get("end_position", 0)
            sec_len = sec.get("char_count") or (sec_end - sec_start)
            coverage = section_coverages.get(sec_name, 0.0)
            total_covered_chars += int(coverage * sec_len)

        overall_coverage = (total_covered_chars / total_section_len) * 100 if total_section_len > 0 else 100.0

        summary = {
            "total_chunks": len(chunks),
            "passed": passed_count,
            "warnings": warning_count,
            "failed": failed_count,
            "coverage_percentage": overall_coverage,
            "section_coverages": section_coverages
        }

        # Print report
        ChunkValidator._print_report(results, summary)

        return results, summary

    @staticmethod
    def _check_text_quality(text: str, section_name: str) -> List[str]:
        warnings = []
        stripped = text.strip()
        if not stripped:
            return []

        # 1. Contains only citations (e.g. [12], [13], [14])
        citations_only = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', stripped)
        citations_only = re.sub(r'[\s,\[\]\-]+', '', citations_only)
        if len(citations_only) == 0:
            warnings.append("Chunk contains only citations.")
            return warnings

        # 2. Contains mostly figure/table labels or page numbers
        noise_cleaned = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', stripped)
        noise_cleaned = re.sub(r'\b(?:FIGURE|Figure|Fig\.|TABLE|Table)\s+[A-Z0-9IVX]+\b', '', noise_cleaned, flags=re.IGNORECASE)
        noise_cleaned = re.sub(r'\d+', '', noise_cleaned)
        noise_cleaned = re.sub(r'[^\w]', '', noise_cleaned)

        if len(noise_cleaned) < 15 and len(stripped) > 0:
            if re.search(r'\b(?:FIGURE|Figure|Fig\.|TABLE|Table)\s+[A-Z0-9IVX]+\b', stripped, flags=re.IGNORECASE):
                warnings.append("Chunk contains mostly figure/table labels or page numbers.")
            elif re.search(r'\d+', stripped):
                warnings.append("Chunk contains mostly page numbers or digits.")

        # 3. Contains mostly references (when outside REFERENCES section)
        if "REFERENCES" not in section_name.upper() and "BIBLIOGRAPHY" not in section_name.upper():
            lines = [line.strip() for line in stripped.split("\n") if line.strip()]
            if lines:
                ref_lines = sum(1 for line in lines if re.match(r'^\[\d+\]', line))
                if ref_lines / len(lines) > 0.5:
                    warnings.append("Chunk contains mostly reference entries outside of the REFERENCES section.")

        return warnings

    @staticmethod
    def _print_report(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        print("\n" + "=" * 50)
        print("CHUNK VALIDATION REPORT")
        print("=" * 50)
        print(f"Total Chunks: {summary['total_chunks']}")
        print(f"Passed:       {summary['passed']}")
        print(f"Warnings:     {summary['warnings']}")
        print(f"Failed:       {summary['failed']}")
        print("=" * 50)

        # FAILED CHUNKS
        failed_results = [r for r in results if r["status"] == "FAIL"]
        print("\nFAILED CHUNKS")
        print("-" * 50)
        if failed_results:
            for r in failed_results:
                print(f"Chunk ID: {r['chunk_id']}")
                print(f"Section:  {r['section_name']}")
                print("Issues:")
                for issue in r["issues"]:
                    print(f"  - {issue}")
                print("-" * 50)
        else:
            print("None")
            print("-" * 50)

        # WARNING CHUNKS
        warning_results = [r for r in results if r["status"] == "WARNING"]
        print("\nWARNING CHUNKS")
        print("-" * 50)
        if warning_results:
            for r in warning_results:
                print(f"Chunk ID: {r['chunk_id']}")
                print(f"Section:  {r['section_name']}")
                print("Issues:")
                for issue in r["issues"]:
                    print(f"  - {issue}")
                print("-" * 50)
        else:
            print("None")
            print("-" * 50)

        # Coverage Report
        print("\nCoverage Report")
        print("-" * 50)
        for sec_name, cov in summary["section_coverages"].items():
            print(f"Section:  {sec_name}")
            print(f"Coverage: {cov * 100:.2f}%")
            print("-" * 50)
        
        print(f"Overall Coverage: {summary['coverage_percentage']:.2f}%")
        print("=" * 50 + "\n")
