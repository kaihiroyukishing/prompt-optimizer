"""
Validation utilities for Prompt Optimizer Backend

This module provides validation functions for ChatGPT output quality,
prompt data integrity, and session data consistency.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def validate_chatgpt_output(output: Optional[str]) -> Dict:
    """
    Validate ChatGPT output quality and structure.

    Args:
        output: ChatGPT output text

    Returns:
        dict with validation results:
        {
            "valid": bool,
            "issues": list of str,
            "quality_score": float (0-1),
            "warnings": list of str
        }
    """
    result = {
        "valid": True,
        "issues": [],
        "quality_score": 0.0,
        "warnings": [],
    }

    if not output:
        result["valid"] = False
        result["issues"].append("ChatGPT output is empty")
        return result

    if isinstance(output, str):
        output_length = len(output.strip())
    else:
        result["valid"] = False
        result["issues"].append("ChatGPT output is not a string")
        return result

    # Check for minimum length
    if output_length < 10:
        result["valid"] = False
        result["issues"].append("ChatGPT output is too short (< 10 characters)")

    # Check for maximum reasonable length (prevent garbage data)
    if output_length > 100000:  # 100k characters
        result["warnings"].append("ChatGPT output is extremely long (> 100k chars)")

    # Check for malformed content
    if output.strip() == "":
        result["valid"] = False
        result["issues"].append("ChatGPT output is only whitespace")

    # Check for common issues
    if output.count("\n") > output_length / 2:
        result["warnings"].append(
            "Output has excessive line breaks (possible formatting issue)"
        )

    # Calculate basic quality score
    quality_score = 1.0
    if result["issues"]:
        quality_score -= len(result["issues"]) * 0.3
    if result["warnings"]:
        quality_score -= len(result["warnings"]) * 0.1
    quality_score = max(quality_score, 0.0)

    result["quality_score"] = round(quality_score, 2)
    return result


def validate_chatgpt_quality_score(score: Optional[float]) -> bool:
    """
    Validate quality score is in valid range.

    Args:
        score: Quality score (expected 0-1 or 1-5)

    Returns:
        bool: True if valid, False otherwise
    """
    if score is None:
        return True  # None is acceptable (nullable field)

    # Accept either 0-1 scale or 1-5 scale
    if 0.0 <= score <= 1.0:
        return True
    if 1.0 <= score <= 5.0:
        return True

    return False


def validate_prompt_data(prompt) -> Dict:
    """
    Validate all prompt data integrity.

    Args:
        prompt: Prompt model instance

    Returns:
        dict with validation results:
        {
            "valid": bool,
            "issues": list of str,
            "warnings": list of str
        }
    """
    result = {"valid": True, "issues": [], "warnings": []}

    if not prompt:
        result["valid"] = False
        result["issues"].append("Prompt object is None")
        return result

    # Validate required fields
    if not prompt.original_prompt:
        result["valid"] = False
        result["issues"].append("original_prompt is required but missing")

    if not prompt.session_id:
        result["valid"] = False
        result["issues"].append("session_id is required but missing")

    # Validate optional but related fields
    if prompt.optimized_prompt:
        if (
            len(prompt.optimized_prompt.strip())
            < len(prompt.original_prompt.strip()) / 2
        ):
            result["warnings"].append("Optimized prompt is much shorter than original")

    # Validate quality scores
    if prompt.chatgpt_quality_score is not None:
        if not validate_chatgpt_quality_score(prompt.chatgpt_quality_score):
            result["issues"].append(
                f"chatgpt_quality_score out of valid range: {prompt.chatgpt_quality_score}"
            )

    # Validate ChatGPT output if present
    if prompt.chatgpt_output:
        output_validation = validate_chatgpt_output(prompt.chatgpt_output)
        if not output_validation["valid"]:
            result["issues"].extend(output_validation["issues"])
            result["valid"] = False
        result["warnings"].extend(output_validation["warnings"])

    # Validate context_prompts is valid JSON if present
    if prompt.context_prompts:
        try:
            import json

            json.loads(prompt.context_prompts)
        except (json.JSONDecodeError, TypeError):
            result["issues"].append("context_prompts is not valid JSON")
            result["valid"] = False

    return result


def validate_session_data(session, prompts: Optional[List] = None) -> Dict:
    """
    Validate session data consistency.

    Args:
        session: Session model instance
        prompts: Optional list of Prompt instances for this session (for cross-validation)

    Returns:
        dict with validation results:
        {
            "valid": bool,
            "issues": list of str,
            "warnings": list of str
        }
    """
    result = {"valid": True, "issues": [], "warnings": []}

    if not session:
        result["valid"] = False
        result["issues"].append("Session object is None")
        return result

    # Validate required fields
    if not session.id:
        result["valid"] = False
        result["issues"].append("Session id is required but missing")

    # Validate is_active is 0 or 1
    if session.is_active not in [0, 1]:
        result["warnings"].append(
            f"is_active has unexpected value: {session.is_active} (expected 0 or 1)"
        )

    # Cross-validate analytics with actual prompts if provided
    if prompts:
        actual_prompt_count = len(prompts)
        if session.total_prompts != actual_prompt_count:
            result["warnings"].append(
                f"total_prompts mismatch: session says {session.total_prompts}, "
                f"but found {actual_prompt_count} prompts"
            )

        # Validate ratings match (using ChatGPT quality scores as proxy)
        if session.average_optimization_rating is not None:
            quality_prompts = [
                p for p in prompts if p.chatgpt_quality_score is not None
            ]
            if quality_prompts:
                actual_avg = sum(
                    p.chatgpt_quality_score for p in quality_prompts
                ) / len(quality_prompts)
                if abs(session.average_optimization_rating - actual_avg) > 0.1:
                    result["warnings"].append(
                        f"average_optimization_rating mismatch: session says "
                        f"{session.average_optimization_rating}, calculated {actual_avg}"
                    )

    # Validate preferred_style enum values
    if session.preferred_style:
        valid_styles = ["concise", "detailed", "technical", "moderate"]
        if session.preferred_style not in valid_styles:
            result["warnings"].append(
                f"preferred_style has unexpected value: {session.preferred_style} "
                f"(expected: {valid_styles})"
            )

    # Validate common_feedback_patterns is valid JSON if present
    if session.common_feedback_patterns:
        try:
            import json

            json.loads(session.common_feedback_patterns)
        except (json.JSONDecodeError, TypeError):
            result["issues"].append("common_feedback_patterns is not valid JSON")
            result["valid"] = False

    return result


def validate_session_id(session_id: Optional[str]) -> bool:
    """
    Validate session ID format.

    Args:
        session_id: Session identifier string

    Returns:
        bool: True if valid, False otherwise
    """
    if not session_id:
        return False

    if not isinstance(session_id, str):
        return False

    # Basic length check
    if len(session_id) < 1 or len(session_id) > 255:
        return False

    return True


def validate_prompt_text(
    prompt_text: Optional[str], min_length: int = 1, max_length: int = 10000
) -> Dict:
    """
    Validate prompt text content.

    Args:
        prompt_text: Prompt text to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        dict with validation results:
        {
            "valid": bool,
            "issues": list of str
        }
    """
    result = {"valid": True, "issues": []}

    if not prompt_text:
        result["valid"] = False
        result["issues"].append("Prompt text is empty")
        return result

    if not isinstance(prompt_text, str):
        result["valid"] = False
        result["issues"].append("Prompt text is not a string")
        return result

    text_length = len(prompt_text.strip())

    if text_length < min_length:
        result["valid"] = False
        result["issues"].append(f"Prompt text is too short (< {min_length} characters)")

    if text_length > max_length:
        result["valid"] = False
        result["issues"].append(f"Prompt text is too long (> {max_length} characters)")

    if prompt_text.strip() == "":
        result["valid"] = False
        result["issues"].append("Prompt text is only whitespace")

    return result
