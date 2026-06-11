import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class StructureDetector:
    @staticmethod
    def detect_sections(text: str) -> List[Dict[str, Any]]:
        """
        Receives the full extracted PDF text.
        Detects section headings using regex, including Roman numeral, alphabetical, and unnumbered headings.
        Returns a list of dicts with keys: title, start_char, end_char, char_count, content.
        """
        logger.info("Initializing Python programmatic structure detection...")
        
        # Compile patterns matching the requested formats
        patterns = [
            # Roman numeral headings, e.g. I. INTRODUCTION, II. MACHINE LEARNING ARCHITECTURES
            re.compile(r'^[ \t]*[IVXLCDM]+\.[ \t]+([a-zA-Z \t,–\-\(\)]{3,})[ \t]*$', re.MULTILINE | re.IGNORECASE),
            # Alphabetical headings, e.g. A. TYPES OF MEDICAL IMAGING
            re.compile(r'^[ \t]*[A-Z]\.[ \t]+([a-zA-Z \t,–\-\(\)]{3,})[ \t]*$', re.MULTILINE | re.IGNORECASE),
            # Standard unnumbered keywords on a line by themselves
            re.compile(r'^[ \t]*(ABSTRACT|REFERENCES|ACKNOWLEDGMENT|INTRODUCTION|CONCLUSION)[ \t]*$', re.MULTILINE | re.IGNORECASE)
        ]
        
        # Special pattern for inline ABSTRACT at the beginning of a line/paragraph
        abstract_inline = re.compile(r'^[ \t]*(ABSTRACT)\b', re.MULTILINE | re.IGNORECASE)
        
        matches = []
        
        # 1. Search with standard patterns
        for pattern in patterns:
            for m in pattern.finditer(text):
                start = m.start()
                title = m.group(0).strip()
                # Avoid duplicate matches on the same start position
                if not any(x["start_char"] == start for x in matches):
                    matches.append({
                        "title": title,
                        "start_char": start
                    })
                    
        # 2. Search for inline abstract
        for m in abstract_inline.finditer(text):
            start = m.start()
            if not any(x["start_char"] == start for x in matches):
                matches.append({
                    "title": "ABSTRACT",
                    "start_char": start
                })
                
        # Sort matches in order of their occurrence in the text
        matches.sort(key=lambda x: x["start_char"])
        
        # Filter out any headings detected after the REFERENCES section starts
        ref_start = None
        for m in matches:
            if m["title"].upper() == "REFERENCES":
                ref_start = m["start_char"]
                break
                
        if ref_start is not None:
            matches = [m for m in matches if m["start_char"] <= ref_start]
            
        if not matches:
            logger.warning("No headings detected in the document text.")
            return []
            
        # Calculate boundaries and slice content
        sections = []
        for i in range(len(matches)):
            curr = matches[i]
            start = curr["start_char"]
            
            if i + 1 < len(matches):
                end = matches[i+1]["start_char"]
            else:
                end = len(text)
                
            content = text[start:end]
            sections.append({
                "title": curr["title"],
                "start_char": start,
                "end_char": end,
                "char_count": len(content),
                "content": content
            })
            
        logger.info(f"Programmatic structure detection successfully identified {len(sections)} sections.")
        return sections

    @staticmethod
    def detect_structure(text: str) -> Dict[str, Any]:
        """
        Detects headings and boundaries and returns a structured dictionary
        representing the document sections.
        """
        raw_sections = StructureDetector.detect_sections(text)
        
        sections_output = []
        for sec in raw_sections:
            # Generate clean content preview (no newlines, truncated to 200 characters)
            preview = sec["content"][:200].strip().replace("\n", " ")
            if len(sec["content"]) > 200:
                preview += "..."
                
            sections_output.append({
                "section_name": sec["title"],
                "start_position": sec["start_char"],
                "end_position": sec["end_char"],
                "char_count": sec["char_count"],
                "content_preview": preview
            })
            
        return {
            "document_type": "Research Paper",
            "sections": sections_output
        }
