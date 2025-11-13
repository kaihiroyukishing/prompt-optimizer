"""
Similarity Search Service for Prompt Optimizer Backend

This module provides functions for finding similar prompts using FAISS vector similarity search.
FAISS (Facebook AI Similarity Search) enables fast and efficient similarity search over
large collections of embedding vectors, making it ideal for finding semantically similar prompts.

Key Functions:
- build_faiss_index(): Build FAISS index from all prompts in database
- get_or_build_index(): Get cached index or build new one from database
- update_index_with_new_prompt(): Add new prompt embedding to existing index
- find_similar_prompts(): Find similar prompts using FAISS search with quality filtering
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.models.prompt import Prompt
from backend.services.cache_service import (
    get_similarity_cache,
    set_similarity_cache,
)
from backend.services.embedding_service import parse_embedding_from_db

logger = logging.getLogger(__name__)

# Try to import C++ normalization module, fallback to None if not available
try:
    import normalization as cpp_norm
    CPP_NORMALIZATION_AVAILABLE = True
except ImportError:
    cpp_norm = None
    CPP_NORMALIZATION_AVAILABLE = False
    logger.info("C++ normalization module not available, using NumPy fallback")

# Module-level cache for FAISS index and mapping
_faiss_index: Optional[faiss.Index] = None
_index_mapping: Optional[Dict[int, int]] = None


def _create_empty_index() -> faiss.Index:
    """
    Create an empty FAISS index for cosine similarity search.

    Uses IndexFlatIP (Inner Product) which, when combined with L2-normalized vectors,
    computes cosine similarity. This is the most efficient exact search method for
    cosine similarity in FAISS.

    Returns:
        Empty FAISS index configured for the embedding dimension

    Raises:
        ValueError: If embedding dimension is invalid (not positive integer)
        RuntimeError: If FAISS index creation fails
    """
    dimension = settings.EMBEDDING_DIMENSION or 1536

    # Validate dimension
    if not isinstance(dimension, int):
        raise ValueError(
            f"EMBEDDING_DIMENSION must be an integer, got {type(dimension)}"
        )

    if dimension <= 0:
        raise ValueError(
            f"EMBEDDING_DIMENSION must be positive, got {dimension}"
        )

    try:
        index = faiss.IndexFlatIP(dimension)
        logger.debug(f"Created empty FAISS index with dimension {dimension}")
        return index
    except Exception as e:
        logger.error(f"Failed to create FAISS index: {e}")
        raise RuntimeError(f"Failed to create FAISS index: {e}") from e


def _normalize_vector(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding vector(s) using L2 normalization.

    This function normalizes vectors to unit length, which is required for
    cosine similarity computation using Inner Product in FAISS. When vectors
    are L2-normalized, Inner Product equals cosine similarity.

    Supports both single vectors (1D array) and batches (2D array).
    Uses C++ implementation if available, falls back to NumPy otherwise.

    Args:
        embedding: Numpy array of shape [dimension] or [n_vectors, dimension]
                   Must be float32 or float64

    Returns:
        Normalized numpy array with same shape as input
        Zero vectors remain zero (no division by zero error)

    Raises:
        ValueError: If input is not a numpy array or has invalid shape
    """
    if not isinstance(embedding, np.ndarray):
        raise ValueError(
            f"embedding must be a numpy array, got {type(embedding)}"
        )

    if embedding.ndim == 0:
        raise ValueError("embedding must be at least 1-dimensional")

    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)

    if CPP_NORMALIZATION_AVAILABLE and cpp_norm is not None:
        try:
            if embedding.ndim == 1:
                vec_list = embedding.tolist()
                normalized_list = cpp_norm.normalize_vector(vec_list)
                return np.array(normalized_list, dtype=np.float32)
            else:
                vecs_list = [vec.tolist() for vec in embedding]
                normalized_list = cpp_norm.normalize_vectors_batch(vecs_list)
                normalized_array = np.array(normalized_list, dtype=np.float32)
                return normalized_array
        except Exception as e:
            logger.warning(f"C++ normalization failed, using NumPy fallback: {e}")

    norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    normalized = embedding / norm
    return normalized.astype(np.float32)


def _search_index(
    query_embedding: np.ndarray, index: faiss.Index, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for similar vectors.

    This function performs a similarity search in the FAISS index and returns
    the distances and indices of the k most similar vectors.

    Args:
        query_embedding: Query embedding vector as numpy array
                        Shape: [1536] or [1, 1536]
        index: FAISS index to search
        k: Number of similar vectors to return

    Returns:
        Tuple of (distances, indices)
        - distances: Numpy array of similarity scores (higher is more similar for IP)
        - indices: Numpy array of FAISS index positions

    Raises:
        ValueError: If query_embedding has invalid shape or k is invalid
        RuntimeError: If search fails
    """
    # Validate input
    if not isinstance(query_embedding, np.ndarray):
        raise ValueError(
            f"query_embedding must be a numpy array, got {type(query_embedding)}"
        )

    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    elif query_embedding.ndim == 2:
        if query_embedding.shape[0] != 1:
            raise ValueError(
                f"query_embedding must have shape [1, dimension] or [dimension], "
                f"got {query_embedding.shape}"
            )
    else:
        raise ValueError(
            f"query_embedding must be 1D or 2D array, got {query_embedding.ndim}D"
        )

    expected_dim = settings.EMBEDDING_DIMENSION or 1536
    if query_embedding.shape[1] != expected_dim:
        raise ValueError(
            f"query_embedding dimension {query_embedding.shape[1]} "
            f"does not match index dimension {expected_dim}"
        )

    normalized_query = _normalize_vector(query_embedding)
    normalized_query = normalized_query.astype(np.float32)

    try:
        distances, indices = index.search(normalized_query, k)

        if distances.shape != (1, k):
            raise RuntimeError(
                f"Unexpected distances shape: {distances.shape}, expected (1, {k})"
            )

        if indices.shape != (1, k):
            raise RuntimeError(
                f"Unexpected indices shape: {indices.shape}, expected (1, {k})"
            )

        return (distances[0], indices[0])

    except Exception as e:
        logger.error(f"FAISS search failed: {e}")
        raise RuntimeError(f"FAISS search failed: {e}") from e


def _map_indices_to_prompts(
    indices: np.ndarray, mapping: Dict[int, int], db: Session
) -> List[Prompt]:
    """
    Map FAISS index positions to Prompt database objects.

    Converts FAISS search result indices (positions in the index) to database
    prompt IDs using the mapping dictionary, then queries the database for
    the corresponding Prompt objects. Preserves the order from FAISS results
    (most similar first).

    Args:
        indices: Numpy array of FAISS index positions from search
        mapping: Dictionary mapping FAISS index position to prompt_id
                 Format: {faiss_index_position: prompt_id}
        db: Database session

    Returns:
        List of Prompt objects in the same order as indices (most similar first)
        Only includes prompts that exist in the database

    Raises:
        RuntimeError: If database query fails
    """
    try:
        prompt_ids = []
        missing_indices = []

        for idx in indices:
            if idx == -1:
                continue

            if idx not in mapping:
                missing_indices.append(idx)
                logger.warning(
                    f"FAISS index position {idx} not found in mapping. Skipping."
                )
                continue

            prompt_id = mapping[idx]
            prompt_ids.append(prompt_id)

        if not prompt_ids:
            logger.warning("No valid prompt IDs found from FAISS indices")
            return []

        prompts = db.query(Prompt).filter(Prompt.id.in_(prompt_ids)).all()
        prompt_dict = {prompt.id: prompt for prompt in prompts}

        result_prompts = []
        for prompt_id in prompt_ids:
            if prompt_id in prompt_dict:
                result_prompts.append(prompt_dict[prompt_id])
            else:
                logger.warning(
                    f"Prompt ID {prompt_id} from mapping not found in database. Skipping."
                )

        if len(result_prompts) != len(prompt_ids):
            logger.warning(
                f"Only {len(result_prompts)} of {len(prompt_ids)} prompts found in database"
            )

        logger.debug(
            f"Mapped {len(indices)} FAISS indices to {len(result_prompts)} Prompt objects"
        )

        return result_prompts

    except Exception as e:
        logger.error(f"Failed to map indices to prompts: {e}")
        raise RuntimeError(f"Failed to map indices to prompts: {e}") from e


def _filter_by_quality(
    prompts: List[Prompt], min_quality: float = 0.7
) -> List[Prompt]:
    """
    Filter prompts by quality score.

    Only includes prompts with chatgpt_quality_score >= min_quality.
    Prompts with None quality scores are excluded.

    Args:
        prompts: List of Prompt objects to filter
        min_quality: Minimum quality score threshold (default: 0.7)

    Returns:
        Filtered list of Prompt objects in the same order as input
        Only includes prompts with quality_score >= min_quality
    """
    if not prompts:
        return []

    filtered_prompts = []
    for prompt in prompts:
        if prompt.chatgpt_quality_score is None:
            continue

        if prompt.chatgpt_quality_score >= min_quality:
            filtered_prompts.append(prompt)

    num_filtered = len(prompts) - len(filtered_prompts)
    if num_filtered > 0:
        logger.debug(
            f"Filtered out {num_filtered} of {len(prompts)} prompts "
            f"(quality < {min_quality})"
        )

    return filtered_prompts


def _extract_embeddings_from_db(
    db: Session,
) -> Tuple[List[int], np.ndarray]:
    """
    Extract embeddings from database and convert to numpy array format.

    Queries all prompts with embeddings from the database, parses the JSON strings,
    and converts them to a numpy array suitable for FAISS indexing.

    Args:
        db: Database session

    Returns:
        Tuple of (prompt_ids, embeddings_array)
        - prompt_ids: List of prompt IDs in the same order as embeddings
        - embeddings_array: Numpy array of shape [n_prompts, 1536] with dtype float32

    Raises:
        ValueError: If embeddings have inconsistent dimensions
        RuntimeError: If database query fails or parsing fails
    """
    try:
        prompts = (
            db.query(Prompt)
            .filter(Prompt.embedding.isnot(None))
            .all()
        )

        if not prompts:
            logger.info("No prompts with embeddings found in database")
            return ([], np.array([], dtype=np.float32).reshape(0, settings.EMBEDDING_DIMENSION or 1536))

        prompt_ids = []
        embeddings_list = []

        for prompt in prompts:
            try:
                embedding = parse_embedding_from_db(prompt.embedding)

                expected_dim = settings.EMBEDDING_DIMENSION or 1536
                if len(embedding) != expected_dim:
                    logger.warning(
                        f"Prompt {prompt.id} has embedding dimension {len(embedding)}, "
                        f"expected {expected_dim}. Skipping."
                    )
                    continue

                prompt_ids.append(prompt.id)
                embeddings_list.append(embedding)

            except Exception as e:
                logger.warning(
                    f"Failed to parse embedding for prompt {prompt.id}: {e}. Skipping."
                )
                continue

        if not embeddings_list:
            logger.warning("No valid embeddings found after parsing")
            return ([], np.array([], dtype=np.float32).reshape(0, settings.EMBEDDING_DIMENSION or 1536))

        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        expected_dim = settings.EMBEDDING_DIMENSION or 1536
        if embeddings_array.shape != (len(embeddings_list), expected_dim):
            raise ValueError(
                f"Unexpected embedding array shape: {embeddings_array.shape}, "
                f"expected ({len(embeddings_list)}, {expected_dim})"
            )

        logger.info(
            f"Extracted {len(prompt_ids)} embeddings from database "
            f"(shape: {embeddings_array.shape})"
        )

        return (prompt_ids, embeddings_array)

    except Exception as e:
        logger.error(f"Failed to extract embeddings from database: {e}")
        raise RuntimeError(f"Failed to extract embeddings from database: {e}") from e


def build_faiss_index(db: Session) -> Tuple[faiss.Index, Dict[int, int]]:
    """
    Build FAISS index from all prompts with embeddings in the database.

    This function extracts all embeddings from the database, normalizes them,
    and builds a FAISS index for fast similarity search. It also creates a
    mapping from FAISS index positions to database prompt IDs.

    Args:
        db: Database session

    Returns:
        Tuple of (FAISS index, mapping dictionary)
        - Index: FAISS index containing all normalized embeddings
        - Mapping: Dictionary mapping FAISS index position to prompt_id
                  Format: {faiss_index_position: prompt_id}

    Raises:
        RuntimeError: If index building fails
    """
    try:
        prompt_ids, embeddings_array = _extract_embeddings_from_db(db)

        if len(prompt_ids) == 0:
            logger.info("No embeddings found, returning empty index")
            empty_index = _create_empty_index()
            return (empty_index, {})

        normalized_embeddings = _normalize_vector(embeddings_array)
        index = _create_empty_index()
        index.add(normalized_embeddings)

        mapping: Dict[int, int] = {}
        for index_position, prompt_id in enumerate(prompt_ids):
            mapping[index_position] = prompt_id

        if index.ntotal != len(prompt_ids):
            raise RuntimeError(
                f"Index size mismatch: index has {index.ntotal} vectors, "
                f"but expected {len(prompt_ids)}"
            )

        logger.info(
            f"Built FAISS index with {index.ntotal} vectors "
            f"(dimension: {index.d})"
        )

        return (index, mapping)

    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        raise RuntimeError(f"Failed to build FAISS index: {e}") from e


def get_or_build_index(db: Session) -> Tuple[faiss.Index, Dict[int, int]]:
    """
    Get cached FAISS index or build new one from database.

    This function implements module-level caching to avoid rebuilding the index
    on every call. The index is rebuilt on each server startup (Option 1 approach).

    Args:
        db: Database session

    Returns:
        Tuple of (FAISS index, mapping dictionary)
        - Index: FAISS index containing all normalized embeddings
        - Mapping: Dictionary mapping FAISS index position to prompt_id
    """
    global _faiss_index, _index_mapping

    if _faiss_index is not None and _index_mapping is not None:
        logger.debug("FAISS index cache hit - returning cached index")
        return (_faiss_index, _index_mapping)

    logger.info("FAISS index cache miss - building index from database")
    _faiss_index, _index_mapping = build_faiss_index(db)

    logger.info(
        f"FAISS index built and cached (vectors: {_faiss_index.ntotal})"
    )

    return (_faiss_index, _index_mapping)


def update_index_with_new_prompt(
    embedding: List[float],
    prompt_id: int,
    index: faiss.Index,
    mapping: Dict[int, int],
) -> None:
    """
    Add a new prompt embedding to an existing FAISS index.

    This function enables incremental index updates without rebuilding the entire index.
    The mapping dictionary is updated to include the new prompt.

    Args:
        embedding: Embedding vector (List of floats, 1536 dimensions)
        prompt_id: Database ID of the prompt
        index: Existing FAISS index to update
        mapping: Dictionary mapping FAISS index position to prompt_id (updated in-place)

    Returns:
        None (updates index and mapping in-place)

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If index update fails
    """
    if not isinstance(embedding, list):
        raise ValueError(f"embedding must be a list, got {type(embedding)}")

    if not embedding:
        raise ValueError("embedding cannot be empty")

    if not isinstance(prompt_id, int):
        raise ValueError(f"prompt_id must be an integer, got {type(prompt_id)}")

    if index is None:
        raise ValueError("index cannot be None")

    if mapping is None:
        raise ValueError("mapping cannot be None")

    expected_dim = settings.EMBEDDING_DIMENSION or 1536
    if len(embedding) != expected_dim:
        raise ValueError(
            f"embedding dimension {len(embedding)} does not match "
            f"expected dimension {expected_dim}"
        )

    try:
        embedding_array = np.array(embedding, dtype=np.float32)
        embedding_array = embedding_array.reshape(1, -1)
        normalized_embedding = _normalize_vector(embedding_array)
        normalized_embedding = normalized_embedding.astype(np.float32)
        index.add(normalized_embedding)

        new_index_position = index.ntotal - 1
        mapping[new_index_position] = prompt_id

        logger.info(
            f"Added prompt {prompt_id} to FAISS index at position {new_index_position} "
            f"(total vectors: {index.ntotal})"
        )

    except ValueError as e:
        logger.error(f"Validation error in index update: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to update FAISS index with new prompt: {e}")
        raise RuntimeError(f"Failed to update FAISS index: {e}") from e


def find_similar_prompts(
    embedding: List[float], session_id: str, limit: int, db: Session
) -> List[Prompt]:
    """
    Find similar prompts using FAISS vector similarity search.

    This function searches for prompts with similar embeddings, filters by quality score,
    and returns the top results. Results are cached to improve performance.

    Args:
        embedding: Query embedding vector (List of floats, 1536 dimensions)
        session_id: Session identifier (currently unused, reserved for future filtering)
        limit: Maximum number of similar prompts to return
        db: Database session

    Returns:
        List of Prompt objects sorted by similarity (most similar first)
        Only includes prompts with chatgpt_quality_score >= 0.7

    Raises:
        ValueError: If embedding is invalid or limit is invalid
        RuntimeError: If database query fails
    """
    # Validate inputs
    if embedding is None:
        raise ValueError("embedding cannot be None")
    
    if not isinstance(embedding, list):
        raise ValueError(f"embedding must be a list, got {type(embedding)}")
    
    if not embedding:
        raise ValueError("embedding cannot be empty")
    
    if limit is None:
        raise ValueError("limit cannot be None")
    
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {limit}")
    
    if db is None:
        raise ValueError("db (database session) cannot be None")

    cached_result = None
    try:
        cached_result = get_similarity_cache(embedding, db)
    except Exception as e:
        logger.warning(f"Cache read error (continuing without cache): {e}")

    if cached_result is not None:
        try:
            if isinstance(cached_result, dict) and "prompt_ids" in cached_result:
                prompt_ids = cached_result["prompt_ids"]
                
                prompts = db.query(Prompt).filter(Prompt.id.in_(prompt_ids)).all()
                prompt_dict = {prompt.id: prompt for prompt in prompts}
                
                result_prompts = []
                for prompt_id in prompt_ids:
                    if prompt_id in prompt_dict:
                        result_prompts.append(prompt_dict[prompt_id])
                
                filtered_prompts = _filter_by_quality(result_prompts, min_quality=0.7)
                result_prompts = filtered_prompts[:limit]
                
                logger.info(
                    f"Cache hit: returning {len(result_prompts)} similar prompts"
                )
                
                return result_prompts
            else:
                logger.warning("Cached result has invalid format, proceeding with FAISS search")
        except Exception as e:
            logger.warning(f"Error parsing cached result (continuing with FAISS search): {e}")

    logger.debug("Cache miss: performing FAISS similarity search")
    
    try:
        logger.debug("Getting or building FAISS index")
        index, mapping = get_or_build_index(db)
        
        if index.ntotal == 0:
            logger.info("FAISS index is empty, returning empty list")
            return []
        
        logger.debug(f"FAISS index has {index.ntotal} vectors, searching for similar prompts")
        
        query_embedding = np.array(embedding, dtype=np.float32)
        search_k = min(limit * 2, index.ntotal)
        logger.debug(f"Searching index for top {search_k} similar vectors (will filter to {limit})")
        
        distances, indices = _search_index(query_embedding, index, search_k)
        logger.debug(f"FAISS search returned {len([i for i in indices if i != -1])} results")
        
        logger.debug("Mapping FAISS indices to Prompt objects")
        prompts = _map_indices_to_prompts(indices, mapping, db)
        logger.debug(f"Mapped to {len(prompts)} Prompt objects")
        
        logger.debug("Filtering prompts by quality score (min: 0.7)")
        filtered_prompts = _filter_by_quality(prompts, min_quality=0.7)
        logger.debug(f"After quality filtering: {len(filtered_prompts)} prompts remain")
        
        result_prompts = filtered_prompts[:limit]
        logger.debug(f"Taking top {limit} results: {len(result_prompts)} prompts")
        
        prompt_id_to_distance = {}
        for i, idx in enumerate(indices):
            if idx != -1 and idx in mapping:
                prompt_id = mapping[idx]
                prompt_id_to_distance[prompt_id] = float(distances[i])
        
        cache_data = {
            "prompt_ids": [prompt.id for prompt in result_prompts],
            "similarity_scores": [
                prompt_id_to_distance.get(prompt.id, 0.0) 
                for prompt in result_prompts
            ]
        }
        
        try:
            set_similarity_cache(embedding, cache_data, db)
            logger.debug("Stored similarity search results in cache")
        except Exception as e:
            logger.warning(f"Cache write error (results generated but not cached): {e}")
        
        logger.info(
            f"Found {len(result_prompts)} similar prompts using FAISS search "
            f"(searched {search_k}, filtered to {len(filtered_prompts)}, returned {len(result_prompts)})"
        )
        
        return result_prompts
        
    except ValueError as e:
        logger.error(f"Input validation error in find_similar_prompts: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error in find_similar_prompts: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in find_similar_prompts: {e}")
        raise RuntimeError(f"Failed to find similar prompts: {e}") from e

