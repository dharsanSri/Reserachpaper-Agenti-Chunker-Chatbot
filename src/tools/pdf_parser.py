import logging
from pathlib import Path
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

from models.exceptions import PDFExtractionError

class PDFParserError(PDFExtractionError):
    """Custom exception class for PDF parsing errors, inheriting from PDFExtractionError."""
    pass

class PDFParser:
    """A parser utility for extracting text and metadata from PDF files using PyMuPDF."""

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """
        Extracts all textual content from the specified PDF file.

        Args:
            pdf_path: The file path to the PDF document as a string.

        Returns:
            The concatenated text extracted from all pages of the PDF.

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
            PDFParserError: If the PDF file is corrupted or cannot be parsed.
        """
        path = Path(pdf_path)
        if not path.is_file():
            logger.error(f"File not found: {path}")
            raise PDFParserError(f"The PDF file at {path.absolute()} does not exist.")

        try:
            doc = fitz.open(path)
        except Exception as e:
            logger.error(f"Failed to open PDF document {path.name}: {e}")
            raise PDFParserError(f"Failed to open or parse corrupted PDF file: {e}") from e

        try:
            text_blocks = []
            for page in doc:
                text_blocks.append(page.get_text())
            
            full_text = "\n".join(text_blocks)
            logger.info(f"Successfully extracted {len(full_text)} characters from {path.name}")
            return full_text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF document {path.name}: {e}")
            raise PDFParserError(f"Error occurred during text extraction: {e}") from e
        finally:
            doc.close()

    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """
        Retrieves the page count of the specified PDF file.

        Args:
            pdf_path: The file path to the PDF document as a string.

        Returns:
            The number of pages in the PDF document.

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
            PDFParserError: If the PDF file is corrupted or cannot be parsed.
        """
        path = Path(pdf_path)
        if not path.is_file():
            logger.error(f"File not found: {path}")
            raise PDFParserError(f"The PDF file at {path.absolute()} does not exist.")

        try:
            doc = fitz.open(path)
            page_count = len(doc)
            doc.close()
            logger.info(f"PDF {path.name} has {page_count} pages")
            return page_count
        except Exception as e:
            logger.error(f"Failed to read page count from PDF document {path.name}: {e}")
            raise PDFParserError(f"Failed to read page count of PDF file: {e}") from e
