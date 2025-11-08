"""
Caching for:
- Similarity search results
- Context strings for Groq optimization
- Embedding vectors
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.models.prompt import CacheEntry

logger = logging.getLogger(__name__)


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt text for consistent cache keys."""
    return prompt.lower().strip().replace("  ", " ")


def generate_embedding_hash(embedding: List[float]) -> str:
    """Generate hash from embedding vector (using first 10 dimensions for consistency)."""
    if not embedding or len(embedding) < 10:
        return ""
    first_10 = tuple(embedding[:10])
    hash_obj = hashlib.md5(str(first_10).encode())
    return hash_obj.hexdigest()


def generate_similarity_cache_key(embedding: List[float]) -> str:
    """Generate cache key for similarity search results."""
    embedding_hash = generate_embedding_hash(embedding)
    return f"similarity:{embedding_hash}"


def generate_context_cache_key(embedding: List[float], preferred_style: Optional[str] = None) -> str:
    """Generate cache key for context strings."""
    embedding_hash = generate_embedding_hash(embedding)
    style = preferred_style or "any"
    return f"context:{embedding_hash}:{style}"


def generate_embedding_cache_key(prompt: str) -> str:
    """Generate cache key for embedding vectors."""
    normalized = normalize_prompt(prompt)
    hash_obj = hashlib.md5(normalized.encode())
    return f"embedding:{hash_obj.hexdigest()}"


def verify_cache_key_match(cache_entry: CacheEntry, input_data: any, cache_type: str) -> bool:
    """Verify cached data matches input (collision detection)."""
    if not cache_entry or cache_entry.cache_type != cache_type:
        return False
    
    try:
        cached_value = json.loads(cache_entry.cache_value)
        
        if cache_type == "similarity":
            return isinstance(cached_value, dict) and "prompt_ids" in cached_value
        
        if cache_type == "context":
            return isinstance(cached_value, str) or (isinstance(cached_value, dict) and "context_string" in cached_value)
        
        if cache_type == "embedding":
            return isinstance(cached_value, list) and len(cached_value) > 0
        
        return False
    except (json.JSONDecodeError, TypeError):
        return False


def get_similarity_cache(embedding: List[float], db: Session) -> Optional[Dict]:
    """Get cached similarity search results."""
    if not settings.CACHE_ENABLED:
        return None
    
    try:
        cache_key = generate_similarity_cache_key(embedding)
        cache_entry = db.query(CacheEntry).filter(
            CacheEntry.cache_key == cache_key,
            CacheEntry.cache_type == "similarity"
        ).first()
        
        if not cache_entry:
            logger.debug(f"Cache miss for similarity: {cache_key}")
            return None
        
        if cache_entry.is_expired():
            logger.debug(f"Cache expired for similarity: {cache_key}")
            return None
        
        if not verify_cache_key_match(cache_entry, embedding, "similarity"):
            logger.warning(f"Cache key collision detected for similarity: {cache_key}")
            return None
        
        cache_entry.increment_hit_count()
        db.commit()
        
        cached_data = json.loads(cache_entry.cache_value)
        logger.debug(f"Cache hit for similarity: {cache_key}")
        return cached_data
    
    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted cache value for similarity: {cache_key}, error: {e}")
        return None
    except SQLAlchemyError as e:
        logger.warning(f"Database error retrieving similarity cache: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error retrieving similarity cache: {e}")
        return None


def get_context_cache(embedding: List[float], preferred_style: Optional[str], db: Session) -> Optional[str]:
    """Get cached context string."""
    if not settings.CACHE_ENABLED:
        return None
    
    try:
        cache_key = generate_context_cache_key(embedding, preferred_style)
        cache_entry = db.query(CacheEntry).filter(
            CacheEntry.cache_key == cache_key,
            CacheEntry.cache_type == "context"
        ).first()
        
        if not cache_entry:
            logger.debug(f"Cache miss for context: {cache_key}")
            return None
        
        if cache_entry.is_expired():
            logger.debug(f"Cache expired for context: {cache_key}")
            return None
        
        if not verify_cache_key_match(cache_entry, embedding, "context"):
            logger.warning(f"Cache key collision detected for context: {cache_key}")
            return None
        
        cache_entry.increment_hit_count()
        db.commit()
        
        cached_data = json.loads(cache_entry.cache_value)
        context_string = cached_data if isinstance(cached_data, str) else cached_data.get("context_string", "")
        logger.debug(f"Cache hit for context: {cache_key}")
        return context_string
    
    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted cache value for context: {cache_key}, error: {e}")
        return None
    except SQLAlchemyError as e:
        logger.warning(f"Database error retrieving context cache: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error retrieving context cache: {e}")
        return None


def get_embedding_cache(prompt: str, db: Session) -> Optional[List[float]]:
    """Get cached embedding vector."""
    if not settings.CACHE_ENABLED:
        return None
    
    try:
        cache_key = generate_embedding_cache_key(prompt)
        cache_entry = db.query(CacheEntry).filter(
            CacheEntry.cache_key == cache_key,
            CacheEntry.cache_type == "embedding"
        ).first()
        
        if not cache_entry:
            logger.debug(f"Cache miss for embedding: {cache_key}")
            return None
        
        if cache_entry.is_expired():
            logger.debug(f"Cache expired for embedding: {cache_key}")
            return None
        
        if not verify_cache_key_match(cache_entry, prompt, "embedding"):
            logger.warning(f"Cache key collision detected for embedding: {cache_key}")
            return None
        
        cache_entry.increment_hit_count()
        db.commit()
        
        cached_data = json.loads(cache_entry.cache_value)
        logger.debug(f"Cache hit for embedding: {cache_key}")
        return cached_data if isinstance(cached_data, list) else cached_data.get("embedding", [])
    
    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted cache value for embedding: {cache_key}, error: {e}")
        return None
    except SQLAlchemyError as e:
        logger.warning(f"Database error retrieving embedding cache: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error retrieving embedding cache: {e}")
        return None


def set_similarity_cache(embedding: List[float], results: Dict, db: Session) -> None:
    """Store similarity search results in cache."""
    if not settings.CACHE_ENABLED:
        return
    
    if not embedding or not results:
        logger.debug("Skipping cache write: invalid input data")
        return
    
    try:
        cache_key = generate_similarity_cache_key(embedding)
        if not cache_key:
            logger.warning("Failed to generate cache key for similarity")
            return
        
        expires_at = datetime.utcnow() + timedelta(seconds=settings.CACHE_TTL_SECONDS)
        
        cache_entry = db.query(CacheEntry).filter(
            CacheEntry.cache_key == cache_key,
            CacheEntry.cache_type == "similarity"
        ).first()
        
        cache_value = json.dumps(results)
        
        if cache_entry:
            cache_entry.cache_value = cache_value
            cache_entry.expires_at = expires_at
            cache_entry.last_accessed = datetime.utcnow()
        else:
            cache_entry = CacheEntry(
                cache_key=cache_key,
                cache_value=cache_value,
                cache_type="similarity",
                expires_at=expires_at
            )
            db.add(cache_entry)
        
        db.commit()
        logger.debug(f"Cache write for similarity: {cache_key}")
    
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid data for similarity cache: {e}")
        db.rollback()
    except SQLAlchemyError as e:
        logger.warning(f"Database error storing similarity cache: {e}")
        db.rollback()
    except Exception as e:
        logger.warning(f"Unexpected error storing similarity cache: {e}")
        db.rollback()


def set_context_cache(embedding: List[float], preferred_style: Optional[str], context: str, db: Session, quality_score: Optional[float] = None) -> None:
    """Store context string in cache."""
    if not settings.CACHE_ENABLED:
        return
    
    if not embedding or not context:
        logger.debug("Skipping cache write: invalid input data")
        return
    
    try:
        cache_key = generate_context_cache_key(embedding, preferred_style)
        if not cache_key:
            logger.warning("Failed to generate cache key for context")
            return
        
        expires_at = datetime.utcnow() + timedelta(seconds=settings.CACHE_TTL_SECONDS)
        
        cache_entry = db.query(CacheEntry).filter(
            CacheEntry.cache_key == cache_key,
            CacheEntry.cache_type == "context"
        ).first()
        
        cache_value = json.dumps({"context_string": context})
        
        if cache_entry:
            cache_entry.cache_value = cache_value
            cache_entry.expires_at = expires_at
            cache_entry.last_accessed = datetime.utcnow()
            if quality_score is not None:
                cache_entry.chatgpt_output_quality = quality_score
        else:
            cache_entry = CacheEntry(
                cache_key=cache_key,
                cache_value=cache_value,
                cache_type="context",
                expires_at=expires_at,
                chatgpt_output_quality=quality_score
            )
            db.add(cache_entry)
        
        db.commit()
        logger.debug(f"Cache write for context: {cache_key}")
    
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid data for context cache: {e}")
        db.rollback()
    except SQLAlchemyError as e:
        logger.warning(f"Database error storing context cache: {e}")
        db.rollback()
    except Exception as e:
        logger.warning(f"Unexpected error storing context cache: {e}")
        db.rollback()


def set_embedding_cache(prompt: str, embedding: List[float], db: Session) -> None:
    """Store embedding vector in cache."""
    if not settings.CACHE_ENABLED:
        return
    
    if not prompt or not embedding:
        logger.debug("Skipping cache write: invalid input data")
        return
    
    try:
        cache_key = generate_embedding_cache_key(prompt)
        if not cache_key:
            logger.warning("Failed to generate cache key for embedding")
            return
        
        expires_at = datetime.utcnow() + timedelta(seconds=settings.CACHE_TTL_SECONDS)
        
        cache_entry = db.query(CacheEntry).filter(
            CacheEntry.cache_key == cache_key,
            CacheEntry.cache_type == "embedding"
        ).first()
        
        cache_value = json.dumps(embedding)
        
        if cache_entry:
            cache_entry.cache_value = cache_value
            cache_entry.expires_at = expires_at
            cache_entry.last_accessed = datetime.utcnow()
        else:
            cache_entry = CacheEntry(
                cache_key=cache_key,
                cache_value=cache_value,
                cache_type="embedding",
                expires_at=expires_at
            )
            db.add(cache_entry)
        
        db.commit()
        logger.debug(f"Cache write for embedding: {cache_key}")
    
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid data for embedding cache: {e}")
        db.rollback()
    except SQLAlchemyError as e:
        logger.warning(f"Database error storing embedding cache: {e}")
        db.rollback()
    except Exception as e:
        logger.warning(f"Unexpected error storing embedding cache: {e}")
        db.rollback()

