"""
Database utility functions for Prompt Optimizer Backend

This module provides helper functions for ChatGPT data analysis,
intent alignment, session pattern analysis, and optimization effectiveness.
"""

import json
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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
    minimal_keywords = ["one word", "one sentence", "brief", "quick", "simple answer", 
                        "just tell me", "direct answer"]
    simple_keywords = ["list", "bullet", "steps", "simple", "basic", "outline"]
    structured_keywords = ["detailed", "comprehensive", "organized", "structured", 
                          "section", "guide", "tutorial", "explain with", "break down"]
    
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


def calculate_default_structure_score(structure_elements: Dict, element_count: int) -> float:
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
    short_keywords = ["short", "brief", "concise", "one word", "one sentence", "quick", "summarize"]
    long_keywords = ["detailed", "comprehensive", "thorough", "explain in detail", "full explanation", 
                     "elaborate", "extensive", "in depth", "complete"]
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
    original_prompt = prompt.original_prompt if hasattr(prompt, 'original_prompt') else None

    # Check if response contains code
    code_indicators = [
        r"```[\w]*\n",  # Code blocks
        r"def\s+\w+\s*\(",  # Python functions
        r"function\s+\w+\s*\(",  # JavaScript functions
        r"class\s+\w+",  # Classes
        r"import\s+\w+",  # Imports
    ]
    has_code = any(re.search(pattern, output, re.IGNORECASE) for pattern in code_indicators)

    # Detect user's structure preference from original prompt
    structure_pref = detect_structure_preference(original_prompt) if original_prompt else None
    
    # Consider session preferences if available
    if session and hasattr(session, 'preferred_style') and session.preferred_style:
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
            elif element_count <= 2 and (structure_elements["lists"] or structure_elements["paragraphs"]):
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
            structure_score = calculate_default_structure_score(structure_elements, element_count)
    else:
        structure_score = calculate_default_structure_score(structure_elements, element_count)
    
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
    if session and hasattr(session, 'preferred_style') and session.preferred_style:
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
                length_quality = 0.2 + ((response_length / min_expected) - 0.5) * 0.6  # 0.2 to 0.8
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
                    length_quality = 1.0 - (normalized_distance ** 2) * 0.3  # 1.0 to 0.7
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
                length_quality = max(0.45 - (excess_ratio - 1.0) * 0.2, 0.3)  # Cap at 0.3
        
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
            length_quality = 0.7 + (normalized_length ** 0.7) * 0.3
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
        "length_fit_score": round(length_fit_score, 3),  # How well length matches user intent
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
    has_structure = bool(re.search(r"#{1,6}\s+|^\d+\.\s+|^[\*\-\+]\s+", output, re.MULTILINE))

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
        "example", "important", "note", "consider", "recommend", "suggest",
        "key", "factor", "benefit", "advantage", "disadvantage", "approach",
        "method", "process", "step", "guide", "explanation", "summary",
        "overview", "detail", "specific", "general", "context", "application"
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
    length_improvement = min((optimized_length - original_length) / original_length, 1.0) if original_length > 0 else 0.0

    # Structure improvement (check if optimized prompt has better structure)
    original_structure = bool(re.search(r"[.!?]", original_prompt))
    optimized_structure = bool(re.search(r"[.!?]", optimized_prompt))
    structure_improvement = 0.5 if optimized_structure and not original_structure else 0.0

    # Response quality (if ChatGPT output is comprehensive, optimization worked)
    response_quality = min(len(chatgpt_output) / 1000, 1.0)  # Normalize to 1.0 at 1000 chars

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
    action_verbs = ["write", "create", "make", "build", "generate", "implement", "optimize", "fix", "explain", "show"]
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
        "create", "analyze", "explain", "compare", "evaluate", "design", "develop",
        # Content types
        "document", "report", "summary", "list", "plan", "strategy", "proposal",
        # General concepts
        "example", "case", "scenario", "situation", "context", "approach", "method"
    ]
    keywords = [kw for kw in domain_keywords if kw in prompt_lower]

    return {
        "intent_type": intent_type,
        "keywords": keywords,
        "complexity": complexity,
        "action_verbs": found_verbs,
    }


def calculate_intent_alignment(chatgpt_output: Optional[str], user_intent: Dict) -> float:
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
        answer_indicators = ["because", "since", "due to", "in order to", "here's", "the answer"]
        if any(indicator in output_lower for indicator in answer_indicators) or len(output_lower) > 100:
            alignment_score += 0.2
    elif intent_type == "command":
        # Commands should produce results - check for actionable content
        # Could be code, instructions, structured output, etc.
        result_indicators = ["```", "step", "first", "second", "then", "finally", "1.", "2."]
        if any(indicator in output_lower for indicator in result_indicators) or len(output_lower) > 150:
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
            "common_feedback_patterns": {
                "usage": {
                    "used_count": 0,
                    "used_ratio": 0.0,
                    "modification_rate": 0.0,
                    "modification_types": {
                        "added_details": 0,
                        "removed_parts": 0,
                        "simplified": 0,
                        "unchanged": 0
                    }
                },
                "success": {
                    "high_quality_count": 0,
                    "high_quality_ratio": 0.0,
                    "average_quality_score": 0.0,
                    "best_intent_types": [],
                    "best_complexity": [],
                    "intent_success_rates": {},
                    "complexity_success_rates": {}
                },
                "optimization": {
                    "average_improvement": 0.0,
                    "effective_methods": []
                },
                "content": {
                    "structure_preference": "any",
                    "length_preference": "any"
                }
            },
            "average_prompt_length": 0.0,
            "most_common_intent": "unknown",
            "success_rate": 0.0,
        }
    prompt_lengths = [len(p.original_prompt) if p.original_prompt else 0 for p in prompts]
    average_prompt_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0
    
    if average_prompt_length < 50:
        preferred_style = "concise"
    elif average_prompt_length > 200:
        preferred_style = "detailed"
    else:
        preferred_style = "moderate"

    feedback_patterns = {}
    
    used_prompts = [
        p for p in prompts if p.final_prompt_used is not None
    ]
    
    if used_prompts:
        feedback_patterns["usage"] = {
            "used_count": len(used_prompts),
            "used_ratio": round(len(used_prompts) / len(prompts) if prompts else 0, 3),
        }
        
        # Calculate modification patterns
        modified_prompts = [p for p in used_prompts if p.was_modified()]
        modification_rate = len(modified_prompts) / len(used_prompts) if used_prompts else 0.0
        feedback_patterns["usage"]["modification_rate"] = round(modification_rate, 3)
        
        # Analyze modification types
        modification_types = {
            "added_details": 0,
            "removed_parts": 0,
            "simplified": 0,
            "unchanged": 0
        }
        
        for prompt in used_prompts:
            if not prompt.was_modified():
                modification_types["unchanged"] += 1
            else:
                optimized_len = len(prompt.optimized_prompt) if prompt.optimized_prompt else 0
                final_len = len(prompt.final_prompt_used) if prompt.final_prompt_used else 0
                
                if optimized_len == 0:
                    modification_types["unchanged"] += 1
                elif final_len > optimized_len * 1.2:
                    modification_types["added_details"] += 1
                elif final_len < optimized_len * 0.8:
                    modification_types["removed_parts"] += 1
                else:
                    modification_types["simplified"] += 1
        
        feedback_patterns["usage"]["modification_types"] = modification_types
    else:
        feedback_patterns["usage"] = {
            "used_count": 0,
            "used_ratio": 0.0,
            "modification_rate": 0.0,
            "modification_types": {
                "added_details": 0,
                "removed_parts": 0,
                "simplified": 0,
                "unchanged": 0
            }
        }
    
    quality_threshold = 0.7
    min_sample_size = 3
    
    high_quality_prompts = [
        p for p in prompts if p.chatgpt_quality_score and p.chatgpt_quality_score >= quality_threshold
    ]
    
    quality_scores = [
        p.chatgpt_quality_score for p in prompts if p.chatgpt_quality_score is not None
    ]
    
    feedback_patterns["success"] = {
        "high_quality_count": len(high_quality_prompts),
        "high_quality_ratio": round(len(high_quality_prompts) / len(prompts) if prompts else 0.0, 3),
        "average_quality_score": round(sum(quality_scores) / len(quality_scores) if quality_scores else 0.0, 3),
        "best_intent_types": [],
        "best_complexity": [],
        "intent_success_rates": {},
        "complexity_success_rates": {}
    }
    
    intents = [extract_user_intent(p.original_prompt) for p in prompts if p.original_prompt]
    intent_performance = {}
    
    for i, prompt in enumerate(prompts):
        if prompt.chatgpt_quality_score is None:
            continue
        intent_type = intents[i].get("intent_type", "unknown") if i < len(intents) else "unknown"
        
        if intent_type not in intent_performance:
            intent_performance[intent_type] = {
                "scores": [],
                "high_quality_count": 0,
                "total_count": 0
            }
        
        intent_performance[intent_type]["scores"].append(prompt.chatgpt_quality_score)
        intent_performance[intent_type]["total_count"] += 1
        if prompt.chatgpt_quality_score >= quality_threshold:
            intent_performance[intent_type]["high_quality_count"] += 1
    
    intent_stats = []
    for intent_type, data in intent_performance.items():
        if data["total_count"] >= min_sample_size:
            avg_score = sum(data["scores"]) / len(data["scores"])
            success_rate = data["high_quality_count"] / data["total_count"]
            intent_stats.append({
                "intent_type": intent_type,
                "average_score": avg_score,
                "success_rate": success_rate,
                "count": data["total_count"]
            })
            feedback_patterns["success"]["intent_success_rates"][intent_type] = round(success_rate, 3)
    

    intent_stats.sort(key=lambda x: x["average_score"], reverse=True)
    feedback_patterns["success"]["best_intent_types"] = [
        stat["intent_type"] for stat in intent_stats[:2]
    ]
    

    complexity_performance = {}
    
    for i, prompt in enumerate(prompts):
        if prompt.chatgpt_quality_score is None:
            continue
        complexity = intents[i].get("complexity", "unknown") if i < len(intents) else "unknown"
        
        if complexity not in complexity_performance:
            complexity_performance[complexity] = {
                "scores": [],
                "high_quality_count": 0,
                "total_count": 0
            }
        
        complexity_performance[complexity]["scores"].append(prompt.chatgpt_quality_score)
        complexity_performance[complexity]["total_count"] += 1
        if prompt.chatgpt_quality_score >= quality_threshold:
            complexity_performance[complexity]["high_quality_count"] += 1
    

    complexity_stats = []
    for complexity, data in complexity_performance.items():
        if data["total_count"] >= min_sample_size:
            avg_score = sum(data["scores"]) / len(data["scores"])
            success_rate = data["high_quality_count"] / data["total_count"]
            complexity_stats.append({
                "complexity": complexity,
                "average_score": avg_score,
                "success_rate": success_rate,
                "count": data["total_count"]
            })
            feedback_patterns["success"]["complexity_success_rates"][complexity] = round(success_rate, 3)
    

    complexity_stats.sort(key=lambda x: x["average_score"], reverse=True)
    feedback_patterns["success"]["best_complexity"] = [
        stat["complexity"] for stat in complexity_stats[:2]
    ]

    # Phase 3: Optimization Effectiveness
    min_sample_size = 3
    
    effectiveness_scores = [
        p.optimization_effectiveness for p in prompts 
        if p.optimization_effectiveness is not None
    ]
    
    feedback_patterns["optimization"] = {
        "average_improvement": round(
            sum(effectiveness_scores) / len(effectiveness_scores) 
            if effectiveness_scores else 0.0, 
            3
        ),
        "effective_methods": []
    }
    
    # Analyze optimization methods performance
    method_performance = {}
    
    for prompt in prompts:
        if prompt.optimization_effectiveness is None or not prompt.optimization_method:
            continue
        
        method = prompt.optimization_method
        
        if method not in method_performance:
            method_performance[method] = {
                "scores": [],
                "count": 0
            }
        
        method_performance[method]["scores"].append(prompt.optimization_effectiveness)
        method_performance[method]["count"] += 1
    
    # Calculate averages and find top methods
    method_stats = []
    for method, data in method_performance.items():
        if data["count"] >= min_sample_size:
            avg_effectiveness = sum(data["scores"]) / len(data["scores"])
            method_stats.append({
                "method": method,
                "average_effectiveness": avg_effectiveness,
                "count": data["count"]
            })
    
    # Get top 2 methods by average effectiveness
    method_stats.sort(key=lambda x: x["average_effectiveness"], reverse=True)
    feedback_patterns["optimization"]["effective_methods"] = [
        stat["method"] for stat in method_stats[:2]
    ]

    # Phase 4: Content Patterns
    structure_preferences = []
    length_preferences = []
    
    for prompt in prompts:
        if prompt.original_prompt:
            structure_pref = detect_structure_preference(prompt.original_prompt)
            length_pref = detect_length_preference(prompt.original_prompt)
            
            if structure_pref["preference"] != "any":
                structure_preferences.append(structure_pref["preference"])
            if length_pref["preference"] != "any":
                length_preferences.append(length_pref["preference"])
    
    feedback_patterns["content"] = {
        "structure_preference": max(set(structure_preferences), key=structure_preferences.count) if structure_preferences else "any",
        "length_preference": max(set(length_preferences), key=length_preferences.count) if length_preferences else "any"
    }

    intent_types = [intent.get("intent_type", "unknown") for intent in intents]
    most_common_intent = max(set(intent_types), key=intent_types.count) if intent_types else "unknown"
    
    success_rate = feedback_patterns["success"]["high_quality_ratio"]

    return {
        "preferred_style": preferred_style,
        "common_feedback_patterns": feedback_patterns,
        "average_prompt_length": round(average_prompt_length, 2),
        "most_common_intent": most_common_intent,
        "success_rate": round(success_rate, 3),
    }

