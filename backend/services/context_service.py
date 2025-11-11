"""
Context Builder Service for Prompt Optimizer Backend

This module provides functions for building context strings from similar prompts and user
preferences. These context strings are used to enhance Groq's prompt optimization by
providing examples of successful optimizations and user-specific preferences.

Key Functions:
- build_optimization_context(): Build context string from similar prompts and session preferences
"""

import logging
from typing import List

from sqlalchemy.orm import Session

from backend.models.prompt import Prompt, Session as SessionModel
from backend.services.cache_service import get_context_cache, set_context_cache

logger = logging.getLogger(__name__)


def _format_similar_prompts(prompts: List[Prompt], max_prompts: int = 5) -> str:
    """
    Format similar prompts into a readable context string.

    Formats a list of similar prompts into a structured text format that can be
    included in the Groq system prompt. Shows examples of successful optimizations
    to guide the optimization process.

    Args:
        prompts: List of Prompt objects to format
        max_prompts: Maximum number of prompts to include (default: 5)

    Returns:
        Formatted string with header and prompt examples
        Returns empty string if prompts list is empty
    """
    # Handle empty list
    if not prompts:
        return ""

    # Limit to top N prompts (most similar)
    prompts_to_format = prompts[:max_prompts]

    # Build formatted string
    lines = ["SIMILAR PROMPTS THAT WORKED WELL:"]

    for prompt in prompts_to_format:
        # Extract fields with None handling
        original = prompt.original_prompt or "[No original prompt]"
        optimized = prompt.optimized_prompt or "[No optimized prompt]"
        quality_score = prompt.chatgpt_quality_score

        # Format quality score
        if quality_score is not None:
            quality_str = f"{quality_score:.2f}"
        else:
            quality_str = "N/A"

        # Format each prompt entry
        # Format: "- Original: '...' → Optimized: '...' (Quality Score: 0.85)"
        prompt_entry = (
            f"- Original: '{original}'\n"
            f"  Optimized: '{optimized}'\n"
            f"  Quality Score: {quality_str}"
        )

        lines.append(prompt_entry)

    # Join with double newline for readability
    return "\n\n".join(lines)


def _format_user_preferences(session: SessionModel) -> str:
    """
    Format user preferences from session into a readable context string.

    Extracts and formats user preferences and feedback patterns from the session
    to provide personalized context for prompt optimization.

    Args:
        session: Session object containing user preferences and patterns

    Returns:
        Formatted string with header and user preferences
        Returns empty string if session is None or has no preferences
    """
    # Handle None session
    if not session:
        return ""

    lines = []
    has_preferences = False

    # Extract preferred style
    preferred_style = session.preferred_style
    if preferred_style:
        lines.append(f"Preferred style: {preferred_style}")
        has_preferences = True

    # Extract feedback patterns
    feedback_patterns = session.get_feedback_patterns()
    if feedback_patterns:
        # Format content preferences
        content_prefs = feedback_patterns.get("content", {})
        if content_prefs:
            structure_pref = content_prefs.get("structure_preference")
            length_pref = content_prefs.get("length_preference")
            
            if structure_pref and structure_pref != "any":
                lines.append(f"Structure preference: {structure_pref}")
                has_preferences = True
            
            if length_pref and length_pref != "any":
                lines.append(f"Length preference: {length_pref}")
                has_preferences = True

        # Format success patterns (what works well)
        success_patterns = feedback_patterns.get("success", {})
        if success_patterns:
            avg_quality = success_patterns.get("average_quality_score")
            if avg_quality and avg_quality > 0:
                lines.append(f"Average quality score: {avg_quality:.2f}")
                has_preferences = True
            
            best_intents = success_patterns.get("best_intent_types", [])
            if best_intents:
                intents_str = ", ".join(best_intents[:3])  # Top 3
                lines.append(f"Works well with intent types: {intents_str}")
                has_preferences = True

        # Format usage patterns
        usage_patterns = feedback_patterns.get("usage", {})
        if usage_patterns:
            modification_rate = usage_patterns.get("modification_rate")
            if modification_rate is not None:
                if modification_rate > 0.5:
                    lines.append("User frequently modifies optimized prompts")
                elif modification_rate < 0.4:
                    lines.append("User typically uses optimized prompts as-is")
                has_preferences = True

    # Return empty string if no preferences found
    if not has_preferences:
        return ""

    # Add header and join lines
    formatted_lines = ["USER PREFERENCES:"] + lines
    return "\n".join(formatted_lines)


def _combine_context_parts(similar_prompts_str: str, preferences_str: str) -> str:
    """
    Combine formatted similar prompts and user preferences into final context string.

    Merges the two formatted context parts (similar prompts and user preferences)
    into a single, well-formatted context string ready for inclusion in the Groq
    system prompt.

    Args:
        similar_prompts_str: Formatted string from _format_similar_prompts()
        preferences_str: Formatted string from _format_user_preferences()

    Returns:
        Combined context string with proper formatting and separators
        Returns empty string if both inputs are empty
    """
    # Handle both empty
    if not similar_prompts_str and not preferences_str:
        return ""

    # Build parts list (only include non-empty strings)
    parts = []
    if similar_prompts_str:
        parts.append(similar_prompts_str)
    if preferences_str:
        parts.append(preferences_str)

    # Combine with double newline separator for clear section separation
    # This ensures readability when inserted into the Groq system prompt
    combined = "\n\n".join(parts)

    return combined


def build_optimization_context(
    similar_prompts: List[Prompt],
    session: SessionModel,
    embedding: List[float],
    db: Session,
) -> str:
    """
    Build context string for Groq from similar prompts and user preferences.

    This function creates a formatted context string that includes examples of successful
    prompt optimizations and user preferences. The context is cached to improve performance.

    Args:
        similar_prompts: List of similar Prompt objects (from similarity search)
        session: Session object containing user preferences and patterns
        embedding: Query embedding vector (used for cache key generation)
        db: Database session for cache operations

    Returns:
        Formatted context string ready to be inserted into Groq system prompt
        Format includes similar prompts examples and user preferences

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If context building fails
    """
    # Input validation
    if embedding is None or not embedding:
        raise ValueError("embedding cannot be None or empty")
    if db is None:
        raise ValueError("db session cannot be None")

    # Extract preferred_style for cache key generation
    preferred_style = session.preferred_style if session else None

    # Check cache first
    cached_context = get_context_cache(embedding, preferred_style, db)
    if cached_context is not None:
        logger.info(f"Context cache hit for embedding hash (style: {preferred_style or 'any'})")
        return cached_context

    logger.debug(f"Context cache miss for embedding hash (style: {preferred_style or 'any'})")

    # Build context from similar prompts and user preferences
    try:
        # Format similar prompts
        similar_prompts_str = _format_similar_prompts(similar_prompts)

        # Format user preferences
        preferences_str = _format_user_preferences(session) if session else ""

        # Combine into final context string
        context = _combine_context_parts(similar_prompts_str, preferences_str)

        # Store in cache
        try:
            set_context_cache(embedding, preferred_style, context, db)
            logger.debug(f"Context cached successfully (style: {preferred_style or 'any'})")
        except Exception as cache_error:
            # Log cache error but don't fail the function
            logger.warning(f"Failed to cache context: {cache_error}")

        return context

    except Exception as e:
        logger.error(f"Error building optimization context: {e}")
        raise RuntimeError(f"Failed to build optimization context: {e}") from e

