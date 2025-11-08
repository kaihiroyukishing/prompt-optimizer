"""
Embedding Service for Prompt Optimizer Backend

This module provides functions for generating embeddings from prompts using OpenAI API.
Embeddings are vector representations of text that capture semantic meaning, enabling
similarity search and context-aware prompt optimization.

Key Functions:
- generate_embedding(): Generate embedding vector from prompt text (with caching)
- parse_embedding_from_db(): Parse embedding JSON string from database to Python list
"""

import json
import logging
from typing import List, Optional

from openai import OpenAI
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.services.cache_service import get_embedding_cache, set_embedding_cache

logger = logging.getLogger(__name__)

# Module-level OpenAI client (initialized lazily)
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    """
    Get or initialize OpenAI client.
    
    Validates API key and model configuration before initializing client.
    Client is cached as module-level variable after first initialization.
    
    Returns:
        Initialized OpenAI client instance
        
    Raises:
        ValueError: If API key is missing or invalid
    """
    global _openai_client
    
    if _openai_client is not None:
        return _openai_client
    
    api_key = settings.OPENAI_API_KEY
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "OpenAI API key is not configured. "
            "Please set OPENAI_API_KEY in your .env file."
        )
    
    embedding_model = settings.EMBEDDING_MODEL
    if not embedding_model:
        embedding_model = "text-embedding-3-small"
        logger.warning("EMBEDDING_MODEL not set, using default: text-embedding-3-small")
    
    try:
        _openai_client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized with model: {embedding_model}")
        return _openai_client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise ValueError(f"Failed to initialize OpenAI client: {e}") from e


def _call_openai_api(prompt: str) -> List[float]:
    """
    Call OpenAI API to generate embedding vector for a prompt.
    
    This is a private helper function that performs the actual API call.
    It does not handle caching - that's done by the public generate_embedding() function.
    
    Args:
        prompt: The prompt text to generate an embedding for
        
    Returns:
        List of floats representing the embedding vector (1536 dimensions)
        
    Raises:
        ValueError: If prompt is empty or response is invalid
        RuntimeError: If OpenAI API call fails (rate limits, network errors, etc.)
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    client = _get_openai_client()
    model = settings.EMBEDDING_MODEL or "text-embedding-3-small"
    expected_dimension = settings.EMBEDDING_DIMENSION or 1536
    
    try:
        logger.debug(f"Calling OpenAI API for embedding with model: {model}")
        response = client.embeddings.create(
            model=model,
            input=prompt
        )
        
        if not response or not response.data or len(response.data) == 0:
            raise RuntimeError("OpenAI API returned empty response")
        
        embedding = response.data[0].embedding
        
        if not isinstance(embedding, list):
            raise RuntimeError(f"Expected list from OpenAI API, got {type(embedding)}")
        
        if len(embedding) != expected_dimension:
            logger.warning(
                f"Embedding dimension mismatch: expected {expected_dimension}, "
                f"got {len(embedding)}"
            )
        
        if not all(isinstance(x, (int, float)) for x in embedding):
            raise RuntimeError("Embedding contains non-numeric values")
        
        embedding_floats = [float(x) for x in embedding]
        logger.debug(f"Successfully generated embedding of dimension {len(embedding_floats)}")
        return embedding_floats
        
    except ValueError as e:
        raise
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            raise RuntimeError(
                "OpenAI API rate limit exceeded. Please try again later."
            ) from e
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            raise RuntimeError(
                "Network error connecting to OpenAI API. Please check your connection."
            ) from e
        elif "authentication" in error_msg.lower() or "invalid" in error_msg.lower():
            raise RuntimeError(
                f"OpenAI API authentication error: {error_msg}"
            ) from e
        else:
            raise RuntimeError(
                f"OpenAI API call failed: {error_msg}"
            ) from e


def generate_embedding(prompt: str, db: Session) -> List[float]:
    """
    Generate embedding vector from prompt text using OpenAI API.
    
    This function checks the cache first, and only calls the OpenAI API if no cached
    embedding is found. The result is stored in cache for future use.
    
    Args:
        prompt: The prompt text to generate an embedding for
        db: Database session for cache operations
        
    Returns:
        List of floats representing the embedding vector (1536 dimensions)
        
    Raises:
        ValueError: If prompt is None, empty, or API key is missing
        RuntimeError: If OpenAI API call fails
        
    Example:
        >>> embedding = generate_embedding("help me write code", db)
        >>> len(embedding)
        1536
    """
    if prompt is None:
        raise ValueError("Prompt cannot be None")
    
    if not isinstance(prompt, str):
        raise ValueError(f"Prompt must be a string, got {type(prompt)}")
    
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    if db is None:
        raise ValueError("Database session cannot be None")
    
    logger.info(f"Generating embedding for prompt (length: {len(prompt)})")
    
    cached_embedding = None
    try:
        cached_embedding = get_embedding_cache(prompt, db)
        if cached_embedding:
            logger.info("Cache hit for embedding - returning cached result")
            return cached_embedding
    except Exception as e:
        logger.warning(f"Cache read error (continuing without cache): {e}")
    
    logger.info("Cache miss for embedding - calling OpenAI API")
    try:
        embedding = _call_openai_api(prompt)
        logger.info(f"Successfully generated embedding (dimension: {len(embedding)})")
    except ValueError as e:
        logger.error(f"Validation error in embedding generation: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"OpenAI API error in embedding generation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in embedding generation: {e}")
        raise RuntimeError(f"Failed to generate embedding: {e}") from e
    
    try:
        set_embedding_cache(prompt, embedding, db)
        logger.debug("Successfully stored embedding in cache")
    except Exception as e:
        logger.warning(f"Cache write error (embedding generated but not cached): {e}")
    
    return embedding


def parse_embedding_from_db(embedding_json: str) -> List[float]:
    """
    Parse embedding JSON string from database to Python list of floats.
    
    This function is used when reading embeddings stored in the Prompt.embedding
    column, which is stored as a JSON string.
    
    Args:
        embedding_json: JSON string representation of embedding vector
        
    Returns:
        List of floats representing the embedding vector
        
    Raises:
        ValueError: If input is None, JSON is malformed, or contains invalid data
        
    Example:
        >>> json_str = '[0.123, -0.456, 0.789]'
        >>> embedding = parse_embedding_from_db(json_str)
        >>> len(embedding)
        3
    """
    if embedding_json is None:
        raise ValueError("embedding_json cannot be None")
    
    if not isinstance(embedding_json, str):
        raise ValueError(f"Expected string, got {type(embedding_json)}")
    
    if not embedding_json.strip():
        raise ValueError("embedding_json cannot be empty")
    
    try:
        parsed_data = json.loads(embedding_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in embedding_json: {e}") from e
    
    if not isinstance(parsed_data, list):
        raise ValueError(
            f"Expected list from JSON, got {type(parsed_data)}"
        )
    
    if len(parsed_data) == 0:
        logger.warning("Parsed embedding is empty list")
        return []
    
    if not all(isinstance(x, (int, float)) for x in parsed_data):
        raise ValueError(
            "Embedding contains non-numeric values. "
            "All elements must be numbers."
        )
    
    embedding_floats = [float(x) for x in parsed_data]
    logger.debug(f"Successfully parsed embedding of dimension {len(embedding_floats)}")
    return embedding_floats

