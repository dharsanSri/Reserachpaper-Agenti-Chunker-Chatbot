import os
import logging
import re
import collections
import time
from dotenv import load_dotenv
from crewai import LLM
from models.exceptions import LLMInitializationError

# Monkey-patch CrewAI's cache breakpoint logic to prevent failures on Groq
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except Exception:
    pass

# Ensure LiteLLM drops any unsupported parameters
try:
    import litellm
    litellm.drop_params = True
except Exception:
    pass

logger = logging.getLogger(__name__)

# Token Tracking and TPM Estimation
_token_history = collections.deque()

def track_tpm(tokens: int) -> int:
    """Tracks token usage in a rolling 60-second window to estimate TPM."""
    now = time.time()
    _token_history.append((now, tokens))
    while _token_history and _token_history[0][0] < now - 60:
        _token_history.popleft()
    return sum(t[1] for t in _token_history)

try:
    def litellm_success_callback(kwargs, response_obj, start_time, end_time):
        try:
            model = kwargs.get("model", "unknown")
            usage = getattr(response_obj, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)
                rolling_tpm = track_tpm(total_tokens)
                logger.info(
                    f"[LiteLLM Usage Logging] Model: {model} | "
                    f"Input Tokens: {prompt_tokens} | "
                    f"Output Tokens: {completion_tokens} | "
                    f"Total Tokens: {total_tokens} | "
                    f"Est. 60s TPM: {rolling_tpm}/12000"
                )
        except Exception as e:
            logger.debug(f"Error in success callback: {e}")

    litellm.success_callback = [litellm_success_callback]
except Exception as e:
    logger.debug(f"Failed to register LiteLLM success callback: {e}")

# Globally throttle and retry LLM calls to respect Groq rate limits with backoff
try:
    original_llm_call = LLM.call
    
    def patched_llm_call(self, *args, **kwargs):
        max_attempts = 6
        attempt = 0
        backoff = 3
        
        while attempt < max_attempts:
            attempt += 1
            try:
                # Minimum delay between sequential requests to prevent immediate rate limit spikes
                time.sleep(1)
                return original_llm_call(self, *args, **kwargs)
            except Exception as e:
                err_str = str(e)
                # Detect rate limit errors
                is_rate_limit = any(x in err_str.lower() for x in ["rate_limit", "429", "rate limit"]) or "RateLimitError" in type(e).__name__
                
                if not is_rate_limit:
                    raise e
                    
                if attempt >= max_attempts:
                    logger.error(f"[Groq Throttler] Max retry attempts ({max_attempts}) exceeded for rate limit error: {err_str}")
                    raise e
                
                # Check for suggested retry cooldown from Groq message
                wait_time = backoff
                match = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                if match:
                    wait_time = float(match.group(1)) + 0.5  # Add a small buffer
                    logger.warning(
                        f"[Groq Rate Limit] Hit rate limit (TPM/RPM exceeded). "
                        f"Parsed suggested wait: {wait_time:.3f}s. "
                        f"Attempt {attempt}/{max_attempts}."
                    )
                else:
                    logger.warning(
                        f"[Groq Rate Limit] Hit rate limit. "
                        f"No suggested wait parsed. Using backoff: {wait_time}s. "
                        f"Attempt {attempt}/{max_attempts}."
                    )
                
                time.sleep(wait_time)
                backoff *= 2  # Exponential backoff
                
    LLM.call = patched_llm_call
except Exception as e:
    logger.error(f"Failed to apply LLM call rate limit patch: {e}")

# Cache for singleton LLM instance
_llm_instance = None

def get_llm() -> LLM:
    """
    Centralized factory that loads environment variables, validates the LLM configuration,
    and returns a shared, thread-safe instance of the CrewAI LLM configured for Groq.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
        
    # Load environment variables
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY")
    model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Startup validation
    if not api_key:
        logger.error("GROQ_API_KEY environment variable is missing.")
        raise LLMInitializationError("GROQ_API_KEY environment variable is missing.")

    print("\n" + "=" * 50)
    print("STARTUP CONFIGURATION")
    print("=" * 50)
    print("Provider: Groq")
    print(f"Model:    {model_name}")
    print("=" * 50 + "\n")

    try:
        crewai_model_string = f"groq/{model_name}"
        logger.info(f"Initializing centralized CrewAI LLM with model string: '{crewai_model_string}'")

        _llm_instance = LLM(
            model=crewai_model_string,
            api_key=api_key,
            temperature=0.1,
            max_tokens=2048
        )
        logger.info("Centralized LLM instance successfully constructed.")
        return _llm_instance
    except Exception as e:
        logger.exception(f"Exception raised during LLM instantiation: {e}")
        raise LLMInitializationError(f"Failed to initialize CrewAI LLM: {e}") from e
