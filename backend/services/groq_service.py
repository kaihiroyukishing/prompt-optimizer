"""
Groq Optimization Service for Prompt Optimizer Backend

This module handles prompt optimization using the Groq API.
It integrates with the context-aware optimization flow by:
1. Building system prompts with context from similar successful optimizations
2. Including user preferences and feedback patterns
3. Calling Groq API to generate optimized prompts
4. Cleaning and parsing responses
5. Returning optimized prompts with metadata (tokens used, timing)
"""

import logging
import time
from typing import Any, Dict, Optional

from groq import Groq

from backend.app.core.config import settings
from backend.models.prompt import Session as SessionModel

logger = logging.getLogger(__name__)

# Module-level Groq client (initialized lazily)
_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    """
    Get or initialize Groq client.
    
    Validates API key before initializing client.
    Client is cached as module-level variable after first initialization.
    
    Returns:
        Initialized Groq client instance
        
    Raises:
        ValueError: If API key is missing or invalid
    """
    global _groq_client
    
    if _groq_client is not None:
        return _groq_client
    
    api_key = settings.GROQ_API_KEY
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "Groq API key is not configured. "
            "Please set GROQ_API_KEY in your .env file."
        )
    
    try:
        _groq_client = Groq(api_key=api_key)
        logger.info("Groq client initialized successfully")
        return _groq_client
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise ValueError(f"Failed to initialize Groq client: {e}") from e


def _build_system_prompt(context: str, session: SessionModel) -> str:
    """
    Build system prompt with context and user preferences.
    
    Constructs a comprehensive system prompt that includes:
    1. Base role definition (prompt optimization expert)
    2. Context from similar successful optimizations (if available)
    3. User preferences and feedback patterns (if available)
    4. Instructions for handling missing information
    
    Args:
        context: Context string from similar successful optimizations (can be empty)
        session: Session object containing user preferences and feedback patterns
        
    Returns:
        Complete system prompt string with all sections properly formatted
    """
    sections = []
    
    # Base prompt
    sections.append("You are a prompt optimization expert.")
    
    # Context section (if available)
    if context and context.strip():
        sections.append("")
        sections.append("CONTEXT FROM SIMILAR SUCCESSFUL OPTIMIZATIONS:")
        sections.append(context.strip())
    
    # User preferences section (if available)
    if session:
        preference_lines = []
        
        # Extract preferred style
        preferred_style = session.preferred_style
        if preferred_style:
            preference_lines.append(f"- Style: {preferred_style}")
        
        # Extract feedback patterns
        feedback_patterns = session.get_feedback_patterns()
        if feedback_patterns:
            # Format content preferences
            content_prefs = feedback_patterns.get("content", {})
            if content_prefs:
                structure_pref = content_prefs.get("structure_preference")
                length_pref = content_prefs.get("length_preference")
                
                if structure_pref and structure_pref != "any":
                    preference_lines.append(f"- Structure preference: {structure_pref}")
                
                if length_pref and length_pref != "any":
                    preference_lines.append(f"- Length preference: {length_pref}")
            
            # Format success patterns
            success_patterns = feedback_patterns.get("success", {})
            if success_patterns:
                best_intents = success_patterns.get("best_intent_types", [])
                if best_intents:
                    intents_str = ", ".join(best_intents[:3])  # Top 3
                    preference_lines.append(f"- Works well with: {intents_str}")
            
            # Format usage patterns
            usage_patterns = feedback_patterns.get("usage", {})
            if usage_patterns:
                modification_rate = usage_patterns.get("modification_rate")
                if modification_rate is not None:
                    if modification_rate > 0.5:
                        preference_lines.append("- User frequently modifies optimized prompts")
                    elif modification_rate < 0.4:
                        preference_lines.append("- User typically uses optimized prompts as-is")
        
        # Add preferences section if we have any preferences
        if preference_lines:
            sections.append("")
            sections.append("USER PREFERENCES:")
            sections.extend(preference_lines)
    
    # Handling instructions
    sections.append("")
    sections.append("HANDLING MISSING INFORMATION:")
    sections.append("- Make reasonable inferences to fill in missing details when possible")
    sections.append("- Use your general knowledge to add specificity (e.g., 'Python code' if context suggests programming)")
    sections.append("- Only use placeholders like [specific task] when information truly cannot be inferred")
    sections.append("- Aim to create complete, actionable prompts rather than leaving gaps")
    
    # Output format instructions
    sections.append("")
    sections.append("OUTPUT FORMAT:")
    sections.append("- Return ONLY the optimized prompt text")
    sections.append("- Do NOT include prefixes like \"Optimized:\", \"Here's the optimized prompt:\", etc.")
    sections.append("- Do NOT include explanations, notes, or meta-commentary")
    sections.append("- Do NOT include phrases like \"Remember:\", \"Note:\", etc.")
    sections.append("- Return the optimized prompt text directly, nothing else")
    
    # Combine all sections with double newlines for readability
    return "\n".join(sections)


def _build_user_message(original_prompt: str) -> str:
    """
    Build user message with the original prompt to optimize.
    
    Formats the original prompt into a user message that instructs
    the LLM to optimize it. Handles edge cases like empty prompts
    and very long prompts.
    
    Args:
        original_prompt: The original prompt text to optimize
        
    Returns:
        Formatted user message string
        
    Raises:
        ValueError: If prompt is empty or None
    """
    # Validate prompt is not empty
    if not original_prompt or not original_prompt.strip():
        raise ValueError("Original prompt cannot be empty")
    
    # Strip whitespace
    prompt = original_prompt.strip()
    
    # Check for very long prompts (Groq has token limits, but we'll let it handle truncation)
    # We could truncate here, but it's better to let Groq handle it with max_tokens
    # Just log a warning if it's extremely long
    if len(prompt) > 5000:  # Rough estimate: ~5000 chars = ~1250 tokens
        logger.warning(
            f"Original prompt is very long ({len(prompt)} chars). "
            "Groq may truncate the response."
        )
    
    # Format user message
    # The prompt is already a string, so we don't need to escape it
    # Groq API will handle the JSON encoding
    user_message = f'Now optimize: "{prompt}"'
    
    return user_message


def _call_groq_api(system_prompt: str, user_message: str) -> Dict[str, Any]:
    """
    Call Groq API to optimize a prompt with retry logic.
    
    Makes the actual API call to Groq with the system prompt and user message.
    Includes retry logic with exponential backoff for rate limits and server errors.
    Measures the time taken and extracts response content and metadata.
    
    Args:
        system_prompt: The system prompt with context and preferences
        user_message: The user message with the original prompt to optimize
        
    Returns:
        Dict containing:
            - optimized_prompt: The raw response content from Groq (str)
            - tokens_used: Number of tokens used (int, optional)
            - optimization_time_ms: Time taken in milliseconds (int)
            
    Raises:
        RuntimeError: If API call fails after all retries (rate limit, network error, etc.)
    """
    client = _get_groq_client()
    
    # Build messages list
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Groq API parameters
    model = "llama-3.3-70b-versatile"  # Updated from llama-3.1-70b-versatile (decommissioned)
    temperature = 0.3
    max_tokens = 800
    
    # Retry configuration
    max_retries = 3
    base_delay = 1  # seconds
    
    # Measure total API call time
    total_start_time = time.time()
    
    last_exception = None
    
    for attempt in range(max_retries):
        # Measure individual attempt time
        attempt_start_time = time.time()
        
        try:
            # Call Groq API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Calculate time taken
            attempt_end_time = time.time()
            optimization_time_ms = int((attempt_end_time - total_start_time) * 1000)
            
            # Extract response content
            if not response.choices or len(response.choices) == 0:
                raise RuntimeError("Groq API returned empty response")
            
            optimized_prompt = response.choices[0].message.content
            
            if not optimized_prompt:
                raise RuntimeError("Groq API returned empty content")
            
            # Extract token usage metadata
            tokens_used = None
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
            
            logger.info(
                f"Groq API call successful (attempt {attempt + 1}): "
                f"{len(optimized_prompt)} chars, {tokens_used} tokens, {optimization_time_ms}ms"
            )
            
            return {
                "optimized_prompt": optimized_prompt,
                "tokens_used": tokens_used,
                "optimization_time_ms": optimization_time_ms
            }
            
        except Exception as e:
            # Calculate attempt time
            attempt_end_time = time.time()
            attempt_time_ms = int((attempt_end_time - attempt_start_time) * 1000)
            
            last_exception = e
            error_message = str(e)
            error_type = type(e).__name__
            
            # Try to catch specific Groq exception types (if they exist)
            # Groq library may have specific exception types similar to OpenAI
            try:
                from groq import RateLimitError, APIError, APIConnectionError, APITimeoutError
                
                if isinstance(e, RateLimitError):
                    is_rate_limit = True
                    is_retryable = True
                elif isinstance(e, APIConnectionError):
                    is_rate_limit = False
                    is_retryable = True  # Network errors are retryable
                elif isinstance(e, APITimeoutError):
                    is_rate_limit = False
                    is_retryable = True  # Timeouts are retryable
                elif isinstance(e, APIError):
                    # Check status code for retryable errors
                    status_code = getattr(e, 'status_code', None)
                    is_rate_limit = (status_code == 429)
                    is_retryable = (status_code in [429, 500, 503])  # Rate limit, server errors
                else:
                    is_rate_limit = False
                    is_retryable = False
                    
            except ImportError:
                # Groq library doesn't have specific exception types, use string matching
                is_rate_limit = ("rate limit" in error_message.lower() or "429" in error_message)
                is_retryable = (
                    is_rate_limit or
                    "503" in error_message or
                    "500" in error_message or
                    "timeout" in error_message.lower() or
                    "connection" in error_message.lower()
                )
            
            # Log error with context
            logger.warning(
                f"Groq API call failed (attempt {attempt + 1}/{max_retries}) "
                f"after {attempt_time_ms}ms: {error_type} - {error_message}"
            )
            
            # Check if this is the last attempt
            if attempt == max_retries - 1:
                # Final attempt failed, raise error
                total_time_ms = int((attempt_end_time - total_start_time) * 1000)
                
                # Handle specific error types with appropriate messages
                if is_rate_limit:
                    logger.error(
                        f"Groq API rate limit exceeded after {max_retries} attempts "
                        f"and {total_time_ms}ms total"
                    )
                    raise RuntimeError(
                        "Groq API rate limit exceeded. Please try again later."
                    ) from e
                
                # Check for authentication errors (not retryable)
                if "401" in error_message or "unauthorized" in error_message.lower():
                    logger.error(
                        f"Groq API authentication failed after {max_retries} attempts "
                        f"and {total_time_ms}ms total"
                    )
                    raise RuntimeError(
                        "Groq API authentication failed. Please check your API key."
                    ) from e
                
                # Check for invalid request (not retryable)
                if "400" in error_message or "bad request" in error_message.lower():
                    logger.error(
                        f"Groq API invalid request after {max_retries} attempts "
                        f"and {total_time_ms}ms total: {error_message}"
                    )
                    raise RuntimeError(
                        f"Groq API invalid request: {error_message}"
                    ) from e
                
                # Generic error handling
                logger.error(
                    f"Groq API call failed after {max_retries} attempts "
                    f"and {total_time_ms}ms total: {error_type} - {error_message}"
                )
                raise RuntimeError(
                    f"Groq API call failed after {max_retries} attempts: {error_message}"
                ) from e
            
            # If error is retryable, wait before retrying
            if is_retryable:
                # Exponential backoff: delay = base_delay * (2 ^ attempt)
                delay = base_delay * (2 ** attempt)
                logger.info(
                    f"Retrying Groq API call after {delay}s (attempt {attempt + 2}/{max_retries})"
                )
                time.sleep(delay)
            else:
                # Non-retryable error, raise immediately
                total_time_ms = int((attempt_end_time - total_start_time) * 1000)
                logger.error(
                    f"Groq API call failed with non-retryable error after {total_time_ms}ms: "
                    f"{error_type} - {error_message}"
                )
                raise RuntimeError(
                    f"Groq API call failed: {error_message}"
                ) from e
    
    # Should never reach here, but just in case
    if last_exception:
        raise RuntimeError(
            f"Groq API call failed after {max_retries} attempts"
        ) from last_exception
    else:
        raise RuntimeError("Groq API call failed for unknown reason")


def _clean_optimized_prompt(raw_response: str) -> str:
    """
    Clean and parse the Groq response to extract just the optimized prompt.
    
    Removes meta-commentary, prefixes, and explanations that the LLM might add.
    Handles edge cases like empty responses or responses that answer the prompt
    instead of optimizing it.
    
    Args:
        raw_response: The raw response content from Groq API
        
    Returns:
        Cleaned optimized prompt string
        
    Raises:
        ValueError: If response is empty or invalid after cleaning
    """
    if not raw_response:
        raise ValueError("Groq response is empty")
    
    # Strip leading/trailing whitespace
    cleaned = raw_response.strip()
    
    # Split into lines for processing
    lines = cleaned.split('\n')
    
    # Filter out lines that are meta-commentary or explanations
    # Common patterns that indicate meta-text:
    meta_prefixes = [
        'User prompt:',
        'Optimized:',
        'Optimized prompt:',
        'Here\'s the optimized',
        'Here is the optimized',
        'The optimized',
        'Remember:',
        'Note:',
        'Keep in mind:',
        'Here',
        'I can help',
        'I\'ll help',
        'I will help',
        'Let me',
        'Sure,',
        'Of course,',
    ]
    
    cleaned_lines = []
    for line in lines:
        trimmed = line.strip()
        # Skip empty lines
        if not trimmed:
            continue
        
        # Check if line starts with any meta prefix (case-insensitive)
        matched_prefix = None
        for prefix in meta_prefixes:
            if trimmed.lower().startswith(prefix.lower()):
                matched_prefix = prefix
                break
        
        # If line starts with a meta prefix, check if it has content after the prefix
        if matched_prefix:
            # Remove the prefix and check remaining content length
            # This handles cases like "Optimized: Please help me write code"
            # vs "Remember: Use best practices." (short meta-commentary)
            prefix_len = len(matched_prefix)
            content_after_prefix = trimmed[prefix_len:].lstrip(':').strip()
            
            # If there's substantial content after the prefix (likely the actual prompt),
            # keep the line and let prefix removal handle it later
            # If it's short (likely just meta-commentary), filter it out
            if len(content_after_prefix) > 30:  # Substantial content = keep it
                cleaned_lines.append(line)
                continue
            else:
                # Short content after prefix = likely just meta-commentary, filter it out
                continue
        
        # Check if line looks like it's answering instead of optimizing
        # (e.g., "I can help you with..." instead of optimized prompt)
        if any(phrase in trimmed.lower() for phrase in ['i can help', 'i\'ll help', 'i will help']):
            continue
        
        cleaned_lines.append(line)
    
    # Join lines back together
    cleaned = '\n'.join(cleaned_lines).strip()
    
    # Remove common prefixes that might be on the first line
    # (in case they weren't on their own line)
    for prefix in meta_prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            # Remove the prefix and any following colon/space
            cleaned = cleaned[len(prefix):].lstrip(':').strip()
            break
    
    # Validate cleaned response
    if not cleaned or len(cleaned) < 10:
        raise ValueError(
            "Groq response appears to be empty or too short after cleaning. "
            "The model may have answered the prompt instead of optimizing it."
        )
    
    # Check if response seems to be answering instead of optimizing
    # (contains phrases that suggest it's trying to help/answer)
    answer_indicators = [
        'i can help',
        'i\'ll help',
        'i will help',
        'let me help',
        'sure, i can',
        'of course, i can',
    ]
    
    cleaned_lower = cleaned.lower()
    if any(indicator in cleaned_lower for indicator in answer_indicators):
        logger.warning(
            "Groq response appears to be answering the prompt instead of optimizing it. "
            "Using response as-is, but it may need further processing."
        )
        # We'll still return it, but log a warning
    
    return cleaned


def optimize_with_groq(
    original_prompt: str, context: str, session: SessionModel
) -> Dict[str, Any]:
    """
    Optimize a prompt using Groq API with context and user preferences.

    This function orchestrates the complete Groq optimization flow:
    1. Validates inputs
    2. Builds system prompt with context and user preferences
    3. Formats user message with original prompt
    4. Calls Groq API with appropriate parameters
    5. Cleans and parses the response
    6. Returns optimized prompt with metadata

    Args:
        original_prompt: The original prompt text to optimize (required)
        context: Context string from similar successful optimizations (can be empty)
        session: Session object containing user preferences and feedback patterns

    Returns:
        Dict containing:
            - optimized_prompt: The optimized prompt text (str, required)
            - tokens_used: Number of tokens used in the API call (int, optional)
            - optimization_time_ms: Time taken for optimization in milliseconds (int, required)

    Raises:
        ValueError: If inputs are invalid (empty prompt, None session)
        RuntimeError: If Groq API call fails (rate limit, network error, etc.)
    """
    # Input validation
    # Validate original_prompt
    if original_prompt is None:
        logger.error("optimize_with_groq called with None original_prompt")
        raise ValueError("original_prompt cannot be None")
    
    if not isinstance(original_prompt, str):
        logger.error(f"optimize_with_groq called with invalid original_prompt type: {type(original_prompt)}")
        raise ValueError(f"original_prompt must be a string, got {type(original_prompt).__name__}")
    
    if not original_prompt.strip():
        logger.error("optimize_with_groq called with empty original_prompt")
        raise ValueError("original_prompt cannot be empty or whitespace only")
    
    # Check for very long prompts (max 10000 characters)
    if len(original_prompt) > 10000:
        logger.error(f"optimize_with_groq called with prompt too long: {len(original_prompt)} chars")
        raise ValueError(
            f"original_prompt is too long ({len(original_prompt)} chars). "
            "Maximum length is 10000 characters."
        )
    
    # Validate context (can be None or empty, but if provided must be string)
    if context is not None and not isinstance(context, str):
        logger.error(f"optimize_with_groq called with invalid context type: {type(context)}")
        raise ValueError(f"context must be a string or None, got {type(context).__name__}")
    
    # Normalize context: convert None to empty string for easier handling
    if context is None:
        context = ""
    
    # Validate session
    if session is None:
        logger.error("optimize_with_groq called with None session")
        raise ValueError("session cannot be None")
    
    if not isinstance(session, SessionModel):
        logger.error(f"optimize_with_groq called with invalid session type: {type(session)}")
        raise ValueError(
            f"session must be a SessionModel instance, got {type(session).__name__}"
        )
    
    logger.debug(
        f"Input validation passed: prompt_length={len(original_prompt)}, "
        f"context_length={len(context) if context else 0}, session_id={session.id}"
    )
    
    # Start optimization flow
    logger.info(
        f"Starting Groq optimization for session {session.id}, "
        f"prompt length: {len(original_prompt)} chars"
    )
    
    try:
        # Step 1: Build system prompt with context and user preferences
        logger.debug("Building system prompt with context and preferences")
        system_prompt = _build_system_prompt(context, session)
        logger.debug(f"System prompt built: {len(system_prompt)} chars")
        
        # Step 2: Build user message with original prompt
        logger.debug("Building user message")
        user_message = _build_user_message(original_prompt)
        logger.debug(f"User message built: {len(user_message)} chars")
        
        # Step 3: Call Groq API
        logger.info("Calling Groq API for optimization")
        api_result = _call_groq_api(system_prompt, user_message)
        
        raw_optimized_prompt = api_result["optimized_prompt"]
        tokens_used = api_result.get("tokens_used")
        optimization_time_ms = api_result["optimization_time_ms"]
        
        logger.debug(
            f"Groq API returned: {len(raw_optimized_prompt)} chars, "
            f"{tokens_used} tokens, {optimization_time_ms}ms"
        )
        
        # Step 4: Clean and parse the response
        logger.debug("Cleaning optimized prompt response")
        cleaned_prompt = _clean_optimized_prompt(raw_optimized_prompt)
        logger.debug(f"Cleaned prompt: {len(cleaned_prompt)} chars (was {len(raw_optimized_prompt)} chars)")
        
        # Step 5: Build and return result
        result = {
            "optimized_prompt": cleaned_prompt,
            "tokens_used": tokens_used,
            "optimization_time_ms": optimization_time_ms
        }
        
        logger.info(
            f"Groq optimization completed successfully: "
            f"{len(cleaned_prompt)} chars, {tokens_used} tokens, {optimization_time_ms}ms"
        )
        
        return result
        
    except ValueError as e:
        # Input validation or cleaning errors
        logger.error(f"Groq optimization failed with ValueError: {e}")
        raise
    except RuntimeError as e:
        # API call errors
        logger.error(f"Groq optimization failed with RuntimeError: {e}")
        raise
    except Exception as e:
        # Unexpected errors
        logger.error(
            f"Groq optimization failed with unexpected error: {type(e).__name__} - {e}",
            exc_info=True
        )
        raise RuntimeError(
            f"Unexpected error during Groq optimization: {e}"
        ) from e

