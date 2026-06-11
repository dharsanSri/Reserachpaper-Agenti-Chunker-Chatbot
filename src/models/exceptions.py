class PDFExtractionError(Exception):
    """Raised when PDF file reading, text extraction, or page count retrieval fails."""
    pass

class LLMInitializationError(Exception):
    """Raised when LLM configuration, environment variable validation, or model instantiation fails."""
    pass

class CrewExecutionError(Exception):
    """Raised when CrewAI agents or task executions fail."""
    pass

class JSONSerializationError(Exception):
    """Raised when JSON serialization or Pydantic output validation fails."""
    pass
