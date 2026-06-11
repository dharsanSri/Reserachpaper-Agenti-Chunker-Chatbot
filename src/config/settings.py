import os

# Configuration Settings for Research Paper Chunker

# Truncation limits: None means no truncation (process full document).
# Can be configured via environment variable MAX_ANALYZE_CHARS.
MAX_ANALYZE_CHARS_RAW = os.environ.get("MAX_ANALYZE_CHARS")
MAX_ANALYZE_CHARS = None

if MAX_ANALYZE_CHARS_RAW is not None:
    try:
        MAX_ANALYZE_CHARS = int(MAX_ANALYZE_CHARS_RAW)
    except ValueError:
        MAX_ANALYZE_CHARS = None

# Validation sample rate (percentage of chunks validated by Quality Validation Agent/LLM)
try:
    VALIDATION_SAMPLE_RATE = float(os.environ.get("VALIDATION_SAMPLE_RATE", 0.1))
except ValueError:
    VALIDATION_SAMPLE_RATE = 0.1

