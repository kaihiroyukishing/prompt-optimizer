"""
Database utility functions for Prompt Optimizer Backend

This module provides helper functions for ChatGPT data analysis,
intent alignment, session pattern analysis, and optimization effectiveness.
"""

import json
import re
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()


def detect_structure_preference(original_prompt: str) -> Dict:
    """
    Detect user's structure preference from original prompt.

    Args:
        original_prompt: Original user prompt text

    Returns:
        dict with structure preference:
        {
            "preference": str ("minimal", "simple", "structured", "any"),
            "confidence": float
        }
    """
    if not original_prompt:
        return {
            "preference": "any",
            "confidence": 0.0,
        }

    prompt_lower = original_prompt.lower()

    # Detect explicit structure requests
    minimal_keywords = [
        "one word",
        "one sentence",
        "brief",
        "quick",
        "simple answer",
        "just tell me",
        "direct answer",
    ]
    simple_keywords = ["list", "bullet", "steps", "simple", "basic", "outline"]
    structured_keywords = [
        "detailed",
        "comprehensive",
        "organized",
        "structured",
        "section",
        "guide",
        "tutorial",
        "explain with",
        "break down",
    ]

    has_minimal = any(kw in prompt_lower for kw in minimal_keywords)
    has_simple = any(kw in prompt_lower for kw in simple_keywords)
    has_structured = any(kw in prompt_lower for kw in structured_keywords)

    # Determine preference
    if has_minimal and not (has_simple or has_structured):
        preference = "minimal"
        confidence = 0.9
    elif has_simple and not has_structured:
        preference = "simple"
        confidence = 0.8
    elif has_structured and not has_minimal:
        preference = "structured"
        confidence = 0.9
    else:
        # No explicit preference - infer from length preference
        length_pref = detect_length_preference(original_prompt)
        if length_pref["preference"] == "short":
            preference = "minimal"
            confidence = 0.6
        elif length_pref["preference"] == "long":
            preference = "structured"
            confidence = 0.7
        else:
            preference = "any"
            confidence = 0.3

    return {
        "preference": preference,
        "confidence": confidence,
    }


def calculate_default_structure_score(
    structure_elements: Dict, element_count: int
) -> float:
    """
    Calculate default structure score when no user preference is detected.

    Args:
        structure_elements: Dict with boolean flags for each element type
        element_count: Number of structural elements present

    Returns:
        structure_score: float (0-1)
    """
    structure_score = 0.0

    # Default scoring: more structure = better (general purpose)
    if structure_elements["paragraphs"]:
        structure_score += 0.3
    if structure_elements["headers"]:
        structure_score += 0.3
    if structure_elements["lists"]:
        structure_score += 0.2
    if structure_elements["formatting"]:
        structure_score += 0.2

    return min(structure_score, 1.0)


def detect_length_preference(original_prompt: str) -> Dict:
    """
    Detect user's length preference from original prompt.

    Args:
        original_prompt: Original user prompt text

    Returns:
        dict with length preference:
        {
            "preference": str ("short", "medium", "long", "any"),
            "min_expected": int,
            "max_expected": int,
            "confidence": float
        }
    """
    if not original_prompt:
        return {
            "preference": "any",
            "min_expected": 50,
            "max_expected": 2000,
            "confidence": 0.0,
        }

    prompt_lower = original_prompt.lower()

    # Detect explicit length requests
    short_keywords = [
        "short",
        "brief",
        "concise",
        "one word",
        "one sentence",
        "quick",
        "summarize",
    ]
    long_keywords = [
        "detailed",
        "comprehensive",
        "thorough",
        "explain in detail",
        "full explanation",
        "elaborate",
        "extensive",
        "in depth",
        "complete",
    ]
    medium_keywords = ["explain", "describe", "tell me about", "what is"]

    has_short = any(kw in prompt_lower for kw in short_keywords)
    has_long = any(kw in prompt_lower for kw in long_keywords)
    has_medium = any(kw in prompt_lower for kw in medium_keywords)

    # Determine preference
    if has_short and not has_long:
        preference = "short"
        min_expected, max_expected = 10, 200
        confidence = 0.9
    elif has_long and not has_short:
        preference = "long"
        min_expected, max_expected = 500, 5000
        confidence = 0.9
    elif has_medium:
        preference = "medium"
        min_expected, max_expected = 200, 1500
        confidence = 0.7
    else:
        # No explicit preference - infer from complexity
        word_count = len(original_prompt.split())
        if word_count <= 5:  # Very simple question
            preference = "short"
            min_expected, max_expected = 20, 300
            confidence = 0.6
        elif word_count <= 15:  # Simple question
            preference = "medium"
            min_expected, max_expected = 100, 800
            confidence = 0.5
        else:  # Complex question
            preference = "any"
            min_expected, max_expected = 200, 2000
            confidence = 0.3

    return {
        "preference": preference,
        "min_expected": min_expected,
        "max_expected": max_expected,
        "confidence": confidence,
    }


def analyze_chatgpt_quality(prompt, session=None) -> Dict:
    """
    Analyze ChatGPT output quality metrics relative to user intent.

    Args:
        prompt: Prompt model instance with chatgpt_output and original_prompt
        session: Optional Session instance for user preferences

    Returns:
        dict with quality metrics:
        {
            "quality_score": float (0-1),
            "response_length": int,
            "has_code": bool,
            "token_efficiency": float,
            "structure_score": float,
            "length_fit_score": float  # How well length matches user intent
        }
    """
    if not prompt or not prompt.chatgpt_output:
        return {
            "quality_score": 0.0,
            "response_length": 0,
            "has_code": False,
            "token_efficiency": 0.0,
            "structure_score": 0.0,
            "length_fit_score": 0.0,
        }

    output = prompt.chatgpt_output
    response_length = len(output)

    # Get original prompt to understand user intent
    original_prompt = (
        prompt.original_prompt if hasattr(prompt, "original_prompt") else None
    )

    # Check if response contains code
    code_indicators = [
        r"```[\w]*\n",  # Code blocks
        r"def\s+\w+\s*\(",  # Python functions
        r"function\s+\w+\s*\(",  # JavaScript functions
        r"class\s+\w+",  # Classes
        r"import\s+\w+",  # Imports
    ]
    has_code = any(
        re.search(pattern, output, re.IGNORECASE) for pattern in code_indicators
    )

    # Detect user's structure preference from original prompt
    structure_pref = (
        detect_structure_preference(original_prompt) if original_prompt else None
    )

    # Consider session preferences if available
    if session and hasattr(session, "preferred_style"):
        if session.preferred_style == "concise":
            # User prefers minimal structure
            if structure_pref and structure_pref["preference"] == "any":
                structure_pref["preference"] = "minimal"
                structure_pref["confidence"] = max(structure_pref["confidence"], 0.5)

    # Calculate structure score based on what user expects
    structure_elements = {
        "paragraphs": bool("\n\n" in output),
        "headers": bool(re.search(r"#{1,6}\s+\w+", output)),
        "lists": bool(re.search(r"^[\*\-\+]\s+", output, re.MULTILINE)),
        "formatting": bool("```" in output or "`" in output),
    }

    # Count structural elements
    element_count = sum(structure_elements.values())

    if structure_pref and structure_pref["confidence"] > 0.3:
        # User has detectable structure preference
        expected_level = structure_pref["preference"]

        if expected_level == "minimal":
            # User wants simple/minimal structure
            # Too much structure is bad, some is okay
            if element_count == 0:
                structure_score = 0.8  # No structure is fine for minimal
            elif element_count == 1:
                structure_score = 1.0  # One element is perfect
            elif element_count == 2:
                structure_score = 0.6  # Two is acceptable but getting too structured
            else:
                structure_score = 0.3  # Too much structure for minimal request

        elif expected_level == "simple":
            # User wants simple structure (lists, maybe paragraphs)
            if element_count == 0:
                structure_score = 0.4  # Some structure expected
            elif element_count <= 2 and (
                structure_elements["lists"] or structure_elements["paragraphs"]
            ):
                structure_score = 1.0  # Lists/paragraphs are perfect
            elif element_count <= 2:
                structure_score = 0.7  # Some structure but not ideal type
            elif element_count == 3:
                structure_score = 0.5  # Getting too complex
            else:
                structure_score = 0.3  # Way too complex

        elif expected_level == "structured":
            # User wants organized structure (headers, lists, paragraphs)
            if element_count == 0:
                structure_score = 0.2  # No structure is bad
            elif element_count == 1:
                structure_score = 0.5  # Needs more structure
            elif element_count == 2:
                structure_score = 0.8  # Good structure
            elif element_count >= 3:
                structure_score = 1.0  # Well-structured is perfect
            else:
                structure_score = 0.3
        else:
            # "any" or unknown - use balanced scoring
            structure_score = calculate_default_structure_score(
                structure_elements, element_count
            )
    else:
        # No clear preference - use balanced default scoring
        structure_score = calculate_default_structure_score(
            structure_elements, element_count
        )

    structure_score = min(structure_score, 1.0)

    # Token efficiency (characters per token estimate, higher is better)
    # Rough estimate: 1 token ≈ 4 characters
    token_count_estimate = response_length / 4
    token_efficiency = (
        response_length / token_count_estimate if token_count_estimate > 0 else 0.0
    )

    # Detect user's length preference from original prompt
    length_pref = detect_length_preference(original_prompt) if original_prompt else None

    # Consider session preferences if available
    if session and hasattr(session, "preferred_style"):
        if session.preferred_style == "concise":
            # User generally prefers short responses
            if length_pref and length_pref["preference"] == "any":
                length_pref["preference"] = "short"
                length_pref["min_expected"] = min(length_pref["min_expected"], 50)
                length_pref["max_expected"] = min(length_pref["max_expected"], 500)
                length_pref["confidence"] = max(length_pref["confidence"], 0.5)
        elif session.preferred_style == "detailed":
            # User generally prefers detailed responses
            if length_pref and length_pref["preference"] == "any":
                length_pref["preference"] = "long"
                length_pref["min_expected"] = max(length_pref["min_expected"], 500)
                length_pref["max_expected"] = max(length_pref["max_expected"], 3000)
                length_pref["confidence"] = max(length_pref["confidence"], 0.5)

    # Calculate length quality score relative to user's expected range
    if length_pref and length_pref["confidence"] > 0.3:
        # User has a detectable preference - score relative to their expectation
        min_expected = length_pref["min_expected"]
        max_expected = length_pref["max_expected"]
        mid_point = (min_expected + max_expected) / 2

        if response_length < min_expected:
            # Too short relative to expectation
            if response_length < min_expected * 0.5:
                length_quality = 0.2  # Way too short
            else:
                # Slightly short - linear ramp
                length_quality = (
                    0.2 + ((response_length / min_expected) - 0.5) * 0.6
                )  # 0.2 to 0.8
        elif response_length <= max_expected:
            # Within expected range - optimal
            if min_expected == max_expected:  # Exact length requested
                if abs(response_length - min_expected) / min_expected < 0.2:
                    length_quality = 1.0  # Perfect match
                else:
                    length_quality = 0.7  # Close but not exact
            else:
                # Range - score based on distance from midpoint (bell curve)
                distance_from_mid = abs(response_length - mid_point)
                max_distance = (max_expected - min_expected) / 2
                if max_distance > 0:
                    normalized_distance = min(distance_from_mid / max_distance, 1.0)
                    length_quality = (
                        1.0 - (normalized_distance**2) * 0.3
                    )  # 1.0 to 0.7
                else:
                    length_quality = 1.0
        else:
            # Longer than expected - penalize based on how much longer
            excess_ratio = (response_length - max_expected) / max_expected
            if excess_ratio < 0.5:  # Slightly longer
                length_quality = 0.8 - excess_ratio * 0.3  # 0.8 to 0.65
            elif excess_ratio < 1.0:  # Moderately longer
                length_quality = 0.65 - (excess_ratio - 0.5) * 0.4  # 0.65 to 0.45
            else:  # Way too long
                length_quality = max(
                    0.45 - (excess_ratio - 1.0) * 0.2, 0.3
                )  # Cap at 0.3

        # Store length fit score for analysis
        length_fit_score = length_quality
    else:
        # No clear preference - use default optimal range (general purpose)
        if response_length < 50:
            length_quality = response_length / 50 * 0.3
        elif response_length < 100:
            length_quality = 0.3 + ((response_length - 50) / 50) * 0.4
        elif response_length <= 2000:
            normalized_length = (response_length - 100) / 1900
            length_quality = 0.7 + (normalized_length**0.7) * 0.3
        elif response_length <= 5000:
            length_quality = 1.0 - ((response_length - 2000) / 3000) * 0.2
        else:
            length_quality = max(0.8 - ((response_length - 5000) / 10000) * 0.3, 0.5)

        length_fit_score = 0.5  # Neutral since we don't know user preference

    # Overall quality score (weighted) - General purpose, not code-specific
    quality_score = (
        (length_quality * 0.4)  # Length factor (40%) - now uses improved calculation
        + (structure_score * 0.5)  # Structure factor (50%)
        + (min(token_efficiency / 10, 1.0) * 0.1)  # Efficiency factor (10%)
    )
    # Note: Code presence is tracked as metadata (has_code), but doesn't affect quality score
    # A well-structured explanation is just as valuable as code
    quality_score = min(quality_score, 1.0)

    return {
        "quality_score": round(quality_score, 3),
        "response_length": response_length,
        "has_code": has_code,
        "token_efficiency": round(token_efficiency, 2),
        "structure_score": round(structure_score, 3),
        "length_fit_score": round(
            length_fit_score, 3
        ),  # How well length matches user intent
    }


def extract_chatgpt_insights(prompt) -> Dict:
    """
    Extract insights from ChatGPT interaction.

    Args:
        prompt: Prompt model instance

    Returns:
        dict with insights:
        {
            "response_type": str ("code", "explanation", "both"),
            "complexity": str ("simple", "moderate", "complex"),
            "keywords": list,
            "suggested_topics": list
        }
    """
    if not prompt or not prompt.chatgpt_output:
        return {
            "response_type": "unknown",
            "complexity": "unknown",
            "keywords": [],
            "suggested_topics": [],
        }

    output = prompt.chatgpt_output.lower()

    # Determine response type
    has_code = bool(re.search(r"```|def\s+|function\s+|class\s+", output))
    has_explanation = len(output.split("\n\n")) > 2 or len(output) > 200

    if has_code and has_explanation:
        response_type = "both"
    elif has_code:
        response_type = "code"
    else:
        response_type = "explanation"

    # Determine complexity based on response structure and length (general purpose)
    # Use word count and structural elements instead of code-specific metrics
    word_count = len(output.split())
    paragraph_count = len(output.split("\n\n"))
    has_structure = bool(
        re.search(r"#{1,6}\s+|^\d+\.\s+|^[\*\-\+]\s+", output, re.MULTILINE)
    )

    if word_count > 500 or paragraph_count > 5 or (has_structure and word_count > 200):
        complexity = "complex"
    elif word_count > 100 or paragraph_count > 2:
        complexity = "moderate"
    else:
        complexity = "simple"

    # Extract keywords (general purpose - detect common topics across domains)
    keywords = []
    # General keywords that appear in various contexts (not code-specific)
    common_keywords = [
        "example",
        "important",
        "note",
        "consider",
        "recommend",
        "suggest",
        "key",
        "factor",
        "benefit",
        "advantage",
        "disadvantage",
        "approach",
        "method",
        "process",
        "step",
        "guide",
        "explanation",
        "summary",
        "overview",
        "detail",
        "specific",
        "general",
        "context",
        "application",
    ]
    for keyword in common_keywords:
        if keyword in output.lower():
            keywords.append(keyword)

    return {
        "response_type": response_type,
        "complexity": complexity,
        "keywords": keywords[:10],  # Limit to top 10
        "suggested_topics": keywords[:5],
    }


def calculate_optimization_improvement(
    original_prompt: str, optimized_prompt: Optional[str], chatgpt_output: Optional[str]
) -> float:
    """
    Calculate how much the optimization improved the ChatGPT response.

    Args:
        original_prompt: Original user prompt
        optimized_prompt: Optimized version
        chatgpt_output: ChatGPT's response

    Returns:
        Improvement score (0-1), where 1.0 = maximum improvement
    """
    if not optimized_prompt or not chatgpt_output:
        return 0.0

    # Simple heuristic: compare prompt clarity and ChatGPT response quality
    # In a real implementation, this would use more sophisticated analysis

    original_length = len(original_prompt)
    optimized_length = len(optimized_prompt)

    # Length improvement (more detail often = better)
    length_improvement = (
        min((optimized_length - original_length) / original_length, 1.0)
        if original_length > 0
        else 0.0
    )

    # Structure improvement (check if optimized prompt has better structure)
    original_structure = bool(re.search(r"[.!?]", original_prompt))
    optimized_structure = bool(re.search(r"[.!?]", optimized_prompt))
    structure_improvement = (
        0.5 if optimized_structure and not original_structure else 0.0
    )

    # Response quality (if ChatGPT output is comprehensive, optimization worked)
    response_quality = min(
        len(chatgpt_output) / 1000, 1.0
    )  # Normalize to 1.0 at 1000 chars

    # Weighted average
    improvement = (
        (length_improvement * 0.3)
        + (structure_improvement * 0.2)
        + (response_quality * 0.5)
    )

    return min(improvement, 1.0)


def extract_user_intent(prompt: str) -> Dict:
    """
    Extract user intent from original prompt.

    Args:
        prompt: Original user prompt text

    Returns:
        dict with intent information:
        {
            "intent_type": str ("question", "command", "request"),
            "keywords": list,
            "complexity": str ("simple", "moderate", "complex"),
            "action_verbs": list
        }
    """
    if not prompt:
        return {
            "intent_type": "unknown",
            "keywords": [],
            "complexity": "unknown",
            "action_verbs": [],
        }

    prompt_lower = prompt.lower()

    # Determine intent type
    question_words = ["how", "what", "why", "when", "where", "which", "who", "?"]
    command_words = ["write", "create", "make", "build", "generate", "implement"]

    if any(word in prompt_lower for word in question_words):
        intent_type = "question"
    elif any(word in prompt_lower for word in command_words):
        intent_type = "command"
    else:
        intent_type = "request"

    # Extract action verbs
    action_verbs = [
        "write",
        "create",
        "make",
        "build",
        "generate",
        "implement",
        "optimize",
        "fix",
        "explain",
        "show",
    ]
    found_verbs = [verb for verb in action_verbs if verb in prompt_lower]

    # Determine complexity
    word_count = len(prompt.split())
    if word_count < 10:
        complexity = "simple"
    elif word_count < 30:
        complexity = "moderate"
    else:
        complexity = "complex"

    # Extract keywords (general purpose - detect various domains)
    keywords = []
    # Keywords across different domains (business, creative, technical, academic, etc.)
    domain_keywords = [
        # Action/process words
        "create",
        "analyze",
        "explain",
        "compare",
        "evaluate",
        "design",
        "develop",
        # Content types
        "document",
        "report",
        "summary",
        "list",
        "plan",
        "strategy",
        "proposal",
        # General concepts
        "example",
        "case",
        "scenario",
        "situation",
        "context",
        "approach",
        "method",
    ]
    keywords = [kw for kw in domain_keywords if kw in prompt_lower]

    return {
        "intent_type": intent_type,
        "keywords": keywords,
        "complexity": complexity,
        "action_verbs": found_verbs,
    }


def calculate_intent_alignment(
    chatgpt_output: Optional[str], user_intent: Dict
) -> float:
    """
    Calculate how well ChatGPT output matches user intent.

    Args:
        chatgpt_output: ChatGPT's response
        user_intent: Intent dict from extract_user_intent()

    Returns:
        Alignment score (0-1), where 1.0 = perfect alignment
    """
    if not chatgpt_output or not user_intent:
        return 0.0

    output_lower = chatgpt_output.lower()
    alignment_score = 0.0

    # Check if output contains user's keywords
    intent_keywords = user_intent.get("keywords", [])
    if intent_keywords:
        keyword_matches = sum(1 for kw in intent_keywords if kw in output_lower)
        alignment_score += (keyword_matches / len(intent_keywords)) * 0.4

    # Check if output contains action verbs (indicates response to command)
    action_verbs = user_intent.get("action_verbs", [])
    if action_verbs:
        verb_matches = sum(1 for verb in action_verbs if verb in output_lower)
        alignment_score += (verb_matches / max(len(action_verbs), 1)) * 0.3

    # Check intent type match (general purpose - not code-specific)
    intent_type = user_intent.get("intent_type", "")
    if intent_type == "question":
        # Questions should provide answers - check for response indicators
        answer_indicators = [
            "because",
            "since",
            "due to",
            "in order to",
            "here's",
            "the answer",
        ]
        if (
            any(indicator in output_lower for indicator in answer_indicators)
            or len(output_lower) > 100
        ):
            alignment_score += 0.2
    elif intent_type == "command":
        # Commands should produce results - check for actionable content
        # Could be code, instructions, structured output, etc.
        result_indicators = [
            "```",
            "step",
            "first",
            "second",
            "then",
            "finally",
            "1.",
            "2.",
        ]
        if (
            any(indicator in output_lower for indicator in result_indicators)
            or len(output_lower) > 150
        ):
            alignment_score += 0.2
    elif intent_type == "request":
        alignment_score += 0.1  # Requests are general, give partial credit

    # Length appropriateness
    complexity = user_intent.get("complexity", "simple")
    output_length = len(chatgpt_output)
    if complexity == "simple" and 50 < output_length < 500:
        alignment_score += 0.1
    elif complexity == "moderate" and 200 < output_length < 1500:
        alignment_score += 0.1
    elif complexity == "complex" and output_length > 500:
        alignment_score += 0.1

    return min(alignment_score, 1.0)


def analyze_session_patterns(session, prompts: List) -> Dict:
    """
    Analyze patterns across all prompts in a session.

    Args:
        session: Session model instance
        prompts: List of Prompt instances for this session

    Returns:
        dict with session patterns:
        {
            "preferred_style": str,
            "common_feedback_patterns": dict,
            "average_prompt_length": float,
            "most_common_intent": str,
            "success_rate": float
        }
    """
    if not prompts:
        return {
            "preferred_style": "unknown",
            "common_feedback_patterns": {},
            "average_prompt_length": 0.0,
            "most_common_intent": "unknown",
            "success_rate": 0.0,
        }

    # Calculate average prompt length
    prompt_lengths = [
        len(p.original_prompt) if p.original_prompt else 0 for p in prompts
    ]
    average_prompt_length = (
        sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0
    )

    # Determine preferred style based on prompt lengths and optimization patterns
    if average_prompt_length < 50:
        preferred_style = "concise"
    elif average_prompt_length > 200:
        preferred_style = "detailed"
    else:
        preferred_style = "moderate"

    # Extract common feedback patterns
    feedback_patterns = {}
    high_rated_prompts = [
        p
        for p in prompts
        if p.user_rating_optimization and p.user_rating_optimization >= 4
    ]
    if high_rated_prompts:
        feedback_patterns["high_rated_count"] = len(high_rated_prompts)
        feedback_patterns["high_rated_ratio"] = len(high_rated_prompts) / len(prompts)

    # Analyze intents
    intents = [
        extract_user_intent(p.original_prompt) for p in prompts if p.original_prompt
    ]
    intent_types = [intent.get("intent_type", "unknown") for intent in intents]
    most_common_intent = (
        max(set(intent_types), key=intent_types.count) if intent_types else "unknown"
    )

    # Calculate success rate (prompts with high ChatGPT quality scores)
    successful_prompts = [
        p for p in prompts if p.chatgpt_quality_score and p.chatgpt_quality_score >= 0.7
    ]
    success_rate = len(successful_prompts) / len(prompts) if prompts else 0.0

    return {
        "preferred_style": preferred_style,
        "common_feedback_patterns": feedback_patterns,
        "average_prompt_length": round(average_prompt_length, 2),
        "most_common_intent": most_common_intent,
        "success_rate": round(success_rate, 3),
    }


def extract_success_patterns(prompts: List) -> Dict:
    """
    Extract patterns from high-rated prompts.

    Args:
        prompts: List of Prompt instances

    Returns:
        dict with success patterns:
        {
            "common_phrases": list,
            "average_length": float,
            "optimization_techniques": list,
            "effective_keywords": list
        }
    """
    if not prompts:
        return {
            "common_phrases": [],
            "average_length": 0.0,
            "optimization_techniques": [],
            "effective_keywords": [],
        }

    # Filter high-quality prompts
    high_quality = [
        p
        for p in prompts
        if (p.chatgpt_quality_score and p.chatgpt_quality_score >= 0.7)
        or (p.user_rating_optimization and p.user_rating_optimization >= 4)
    ]

    if not high_quality:
        return {
            "common_phrases": [],
            "average_length": 0.0,
            "optimization_techniques": [],
            "effective_keywords": [],
        }

    # Calculate average length of successful prompts
    lengths = [
        len(p.optimized_prompt) if p.optimized_prompt else 0 for p in high_quality
    ]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0

    # Extract common keywords from successful optimizations (general purpose)
    all_text = " ".join(
        [p.optimized_prompt for p in high_quality if p.optimized_prompt]
    ).lower()
    # General keywords that appear in successful prompts across domains
    keywords = [
        "example",
        "explanation",
        "detail",
        "specific",
        "clear",
        "comprehensive",
        "step",
        "process",
        "method",
        "approach",
        "consider",
        "important",
        "note",
    ]
    effective_keywords = [kw for kw in keywords if kw in all_text]

    return {
        "common_phrases": [],  # Would need NLP for this
        "average_length": round(average_length, 2),
        "optimization_techniques": [],  # Would need to analyze patterns
        "effective_keywords": effective_keywords[:10],
    }
