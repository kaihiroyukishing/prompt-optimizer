"""
Optimization API routes

This module handles all optimization-related API endpoints.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.config import settings
from backend.models.prompt import Prompt, Session as SessionModel
from backend.services import context_service
from backend.services import embedding_service
from backend.services import groq_service
from backend.services import similarity_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum prompt length (characters)
MAX_PROMPT_LENGTH = 10000
# Session ID constraints
MIN_SESSION_ID_LENGTH = 1
MAX_SESSION_ID_LENGTH = 255


def _validate_optimize_inputs(prompt: str, session_id: str) -> None:
    """
    Validate inputs for the optimize endpoint.

    Args:
        prompt: The prompt text to validate
        session_id: The session identifier to validate

    Raises:
        HTTPException 400: If validation fails with appropriate error message
    """
    # Validate prompt
    if prompt is None:
        raise HTTPException(
            status_code=400,
            detail="prompt cannot be None"
        )

    if not isinstance(prompt, str):
        raise HTTPException(
            status_code=400,
            detail=f"prompt must be a string, got {type(prompt).__name__}"
        )

    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="prompt cannot be empty or whitespace only"
        )

    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"prompt is too long (max {MAX_PROMPT_LENGTH} characters, got {len(prompt)})"
        )

    # Validate session_id
    if session_id is None:
        raise HTTPException(
            status_code=400,
            detail="session_id cannot be None"
        )

    if not isinstance(session_id, str):
        raise HTTPException(
            status_code=400,
            detail=f"session_id must be a string, got {type(session_id).__name__}"
        )

    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id cannot be empty or whitespace only"
        )

    if len(session_id) < MIN_SESSION_ID_LENGTH or len(session_id) > MAX_SESSION_ID_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"session_id length must be between {MIN_SESSION_ID_LENGTH} and {MAX_SESSION_ID_LENGTH} characters, got {len(session_id)}"
        )

    # Validate session_id format (alphanumeric, hyphens, underscores allowed)
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        raise HTTPException(
            status_code=400,
            detail="session_id contains invalid characters (only alphanumeric, hyphens, and underscores allowed)"
        )


def _get_or_create_session(session_id: str, db: Session) -> SessionModel:
    """
    Get existing session or create a new one.

    Args:
        session_id: The session identifier
        db: Database session

    Returns:
        SessionModel: The existing or newly created session

    Raises:
        HTTPException 500: If database operation fails
    """
    try:
        # Query for existing session
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()

        if session:
            logger.debug(f"Found existing session: {session_id}")
            return session

        # Create new session
        logger.info(f"Creating new session: {session_id}")
        session = SessionModel(id=session_id)
        db.add(session)
        db.commit()
        db.refresh(session)  # Refresh to get any default values from database

        logger.info(f"Successfully created session: {session_id}")
        return session

    except Exception as e:
        # Rollback on any error
        db.rollback()
        logger.error(f"Failed to get or create session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get or create session: {str(e)}"
        ) from e


def _create_prompt_record(
    session_id: str,
    original_prompt: str,
    optimized_prompt: str,
    embedding: List[float],
    prompt_ids: List[int],
    optimization_time_ms: int,
    tokens_used: Optional[int],
    db: Session
) -> Prompt:
    """
    Create and save a Prompt record to the database.

    Args:
        session_id: Session identifier
        original_prompt: Original prompt text
        optimized_prompt: Optimized prompt from Groq
        embedding: Embedding vector (will be converted to JSON string)
        prompt_ids: List of similar prompt IDs (will be converted to JSON string)
        optimization_time_ms: Optimization time in milliseconds
        tokens_used: Number of tokens used
        db: Database session

    Returns:
        Prompt: The created Prompt object with id populated

    Raises:
        HTTPException 500: If database operation fails
    """
    try:
        # Convert embedding to JSON string
        embedding_json = json.dumps(embedding) if embedding else None
        
        # Convert prompt_ids to JSON string
        context_prompts_json = json.dumps(prompt_ids) if prompt_ids else None
        
        # Create new Prompt object
        prompt = Prompt(
            session_id=session_id,
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            embedding=embedding_json,
            context_prompts=context_prompts_json,
            optimization_method="groq",
            optimization_time_ms=optimization_time_ms,
            tokens_used=tokens_used
        )
        
        # Add to database
        db.add(prompt)
        db.commit()
        db.refresh(prompt)  # Refresh to get the id and any default values
        
        logger.info(f"Successfully created prompt record (id: {prompt.id}, session: {session_id})")
        return prompt
        
    except Exception as e:
        # Rollback on any error
        db.rollback()
        logger.error(f"Failed to create prompt record (session: {session_id}): {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save prompt to database: {str(e)}"
        ) from e


def _format_optimize_response(prompt: Prompt, similar_prompts: List[Prompt]) -> Dict:
    """
    Format the optimization response for the frontend.

    Args:
        prompt: The Prompt object containing optimization results
        similar_prompts: List of similar Prompt objects used for context

    Returns:
        Dict containing formatted response with all required fields
    """
    # Calculate similar prompts count
    similar_prompts_count = len(similar_prompts) if similar_prompts else 0
    
    # Determine if context was used
    context_used = similar_prompts_count > 0
    
    # Build response dictionary with None handling
    response = {
        "original_prompt": prompt.original_prompt if prompt.original_prompt else "",
        "optimized_prompt": prompt.optimized_prompt if prompt.optimized_prompt else "",
        "session_id": prompt.session_id if prompt.session_id else "",
        "similar_prompts_count": similar_prompts_count,
        "context_used": context_used,
        "optimization_method": prompt.optimization_method if prompt.optimization_method else "groq",
        "prompt_id": prompt.id if prompt.id else None
    }
    
    return response


@router.post("/optimize")
async def optimize_prompt(prompt: str, session_id: str, db: Session = Depends(get_db)):
    """
    Optimize a prompt using context-aware AI.

    This endpoint orchestrates the complete optimization flow:
    1. Validates input parameters
    2. Gets or creates a session
    3. Generates embedding for the prompt
    4. Finds similar prompts using FAISS similarity search
    5. Builds context from similar prompts and user preferences
    6. Optimizes the prompt using Groq API with context
    7. Saves the result to the database
    8. Updates the FAISS index with the new prompt
    9. Returns the optimized prompt with metadata

    Args:
        prompt: The original prompt text to optimize (required, max 10000 characters)
        session_id: Unique session identifier for tracking user preferences (required)
        db: Database session (injected by FastAPI)

    Returns:
        Dict containing:
            - original_prompt (str): The original prompt text
            - optimized_prompt (str): The optimized prompt from Groq
            - session_id (str): The session identifier
            - similar_prompts_count (int): Number of similar prompts used for context
            - context_used (bool): Whether context was used in optimization
            - optimization_method (str): Method used (e.g., "groq")
            - prompt_id (int): Database ID of the saved prompt

    Raises:
        HTTPException 400: Invalid input parameters
        HTTPException 500: Server errors (database, configuration)
        HTTPException 503: Service unavailable (OpenAI, Groq API failures)
        HTTPException 429: Rate limit exceeded (Groq API)
    """
    # Track start time for logging
    start_time = time.time()
    
    logger.info(
        f"Starting optimization flow (session: {session_id}, prompt_length: {len(prompt) if prompt else 0})"
    )
    
    try:
        # Step 1: Validate inputs
        logger.debug("Step 1: Validating inputs")
        _validate_optimize_inputs(prompt, session_id)

        # Step 2: Get or create session
        logger.debug("Step 2: Getting or creating session")
        session = _get_or_create_session(session_id, db)

        # Step 3: Generate embedding
        logger.debug("Step 3: Generating embedding")
        try:
            logger.info(f"Generating embedding for prompt (session: {session_id})")
            embedding = embedding_service.generate_embedding(prompt, db)
            logger.info(f"Successfully generated embedding (dimension: {len(embedding)})")
        except ValueError as e:
            # Missing API key or validation error
            error_msg = str(e)
            logger.error(f"Embedding generation validation error: {error_msg}")
            if "API key" in error_msg or "not configured" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="OpenAI API key is not configured. Please set OPENAI_API_KEY in your environment."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid request for embedding generation: {error_msg}"
                )
        except RuntimeError as e:
            # OpenAI API failure (rate limits, network errors, etc.)
            error_msg = str(e)
            logger.error(f"OpenAI API error during embedding generation: {error_msg}")
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI API service unavailable: {error_msg}"
            )
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error during embedding generation: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding: {str(e)}"
            )

        # Step 4: Find similar prompts
        logger.debug("Step 4: Finding similar prompts")
        limit = settings.MAX_SIMILAR_PROMPTS or 5
        similar_prompts = []
        prompt_ids = []
        
        try:
            logger.info(f"Searching for similar prompts (limit: {limit}, session: {session_id})")
            similar_prompts = similarity_service.find_similar_prompts(
                embedding, session_id, limit, db
            )
            logger.info(f"Found {len(similar_prompts)} similar prompts")
            
            # Extract prompt IDs for database storage
            prompt_ids = [p.id for p in similar_prompts if p.id is not None]
            logger.debug(f"Extracted {len(prompt_ids)} prompt IDs for context_prompts")
            
        except ValueError as e:
            # Invalid input (embedding, limit, etc.)
            error_msg = str(e)
            logger.error(f"Similarity search validation error: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid request for similarity search: {error_msg}"
            )
        except RuntimeError as e:
            # Database errors or critical FAISS errors
            error_msg = str(e)
            logger.error(f"Database or FAISS error during similarity search: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error during similarity search: {error_msg}"
            )
        except Exception as e:
            # FAISS errors (empty index, etc.) - graceful degradation
            error_msg = str(e)
            logger.warning(f"FAISS error during similarity search (continuing with empty list): {error_msg}")
            # Continue with empty list - optimization can proceed without similar prompts
            similar_prompts = []
            prompt_ids = []

        # Step 5: Build context
        logger.debug("Step 5: Building optimization context")
        context = ""
        try:
            logger.info(f"Building optimization context (similar_prompts: {len(similar_prompts)}, session: {session_id})")
            context = context_service.build_optimization_context(
                similar_prompts, session, embedding, db
            )
            logger.info(f"Successfully built context (length: {len(context)} characters)")
        except ValueError as e:
            # Invalid inputs (embedding, db, etc.)
            error_msg = str(e)
            logger.error(f"Context building validation error: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid request for context building: {error_msg}"
            )
        except RuntimeError as e:
            # Context building failure (formatting errors, etc.)
            error_msg = str(e)
            logger.error(f"Context building error: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build optimization context: {error_msg}"
            )
        except Exception as e:
            # Cache errors or other unexpected errors - log warning but continue
            error_msg = str(e)
            logger.warning(f"Context building encountered error (continuing with empty context): {error_msg}")
            # Continue with empty context - Groq can optimize without context
            context = ""

        # Step 6: Optimize with Groq
        logger.debug("Step 6: Optimizing with Groq")
        optimized_prompt = ""
        tokens_used = None
        optimization_time_ms = 0
        
        try:
            logger.info(f"Optimizing prompt with Groq (session: {session_id}, context_length: {len(context)})")
            groq_result = groq_service.optimize_with_groq(prompt, context, session)
            
            optimized_prompt = groq_result["optimized_prompt"]
            tokens_used = groq_result.get("tokens_used")
            optimization_time_ms = groq_result["optimization_time_ms"]
            
            logger.info(
                f"Groq optimization completed successfully: "
                f"prompt_length={len(optimized_prompt)}, tokens={tokens_used}, time={optimization_time_ms}ms"
            )
            
        except ValueError as e:
            # Missing API key or validation error
            error_msg = str(e)
            logger.error(f"Groq optimization validation error: {error_msg}")
            if "API key" in error_msg or "not configured" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="Groq API key is not configured. Please set GROQ_API_KEY in your environment."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid request for Groq optimization: {error_msg}"
                )
        except RuntimeError as e:
            # Groq API failure (rate limits, network errors, etc.)
            error_msg = str(e)
            logger.error(f"Groq API error during optimization: {error_msg}")
            
            # Check if it's a rate limit error
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                raise HTTPException(
                    status_code=429,
                    detail="Groq API rate limit exceeded. Please try again later."
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Groq API service unavailable: {error_msg}"
                )
        except Exception as e:
            # Unexpected error
            error_msg = str(e)
            logger.error(f"Unexpected error during Groq optimization: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to optimize prompt with Groq: {error_msg}"
            )

        # Step 7: Save to database
        logger.debug("Step 7: Saving prompt to database")
        try:
            logger.info(f"Saving prompt to database (session: {session_id})")
            prompt_record = _create_prompt_record(
                session_id=session_id,
                original_prompt=prompt,
                optimized_prompt=optimized_prompt,
                embedding=embedding,
                prompt_ids=prompt_ids,
                optimization_time_ms=optimization_time_ms,
                tokens_used=tokens_used,
                db=db
            )
            logger.info(f"Prompt saved successfully (prompt_id: {prompt_record.id})")
        except HTTPException:
            # Re-raise HTTPException (already formatted)
            raise
        except Exception as e:
            # Unexpected database error
            logger.error(f"Unexpected error saving prompt to database: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save prompt to database: {str(e)}"
            )

        # Step 8: Update FAISS index
        logger.debug("Step 8: Updating FAISS index")
        try:
            logger.info(f"Updating FAISS index with new prompt (prompt_id: {prompt_record.id})")
            index, mapping = similarity_service.get_or_build_index(db)
            similarity_service.update_index_with_new_prompt(
                embedding, prompt_record.id, index, mapping
            )
            logger.info(f"FAISS index updated successfully (prompt_id: {prompt_record.id})")
        except Exception as e:
            # Non-blocking: if index update fails, index will be rebuilt on next search
            error_msg = str(e)
            logger.warning(
                f"FAISS index update failed (non-critical, will rebuild on next search): {error_msg}"
            )
            # Continue - this is non-blocking, system still works

        # Step 9: Format and return response
        logger.debug("Step 9: Formatting response")
        logger.info(f"Formatting response (prompt_id: {prompt_record.id}, similar_prompts: {len(similar_prompts)})")
        response = _format_optimize_response(prompt_record, similar_prompts)
        
        total_time = int((time.time() - start_time) * 1000)
        logger.info(
            f"✅ Optimization completed successfully (prompt_id: {prompt_record.id}, "
            f"total_time: {total_time}ms, session: {session_id}, "
            f"context_used: {response.get('context_used', False)}, "
            f"similar_prompts: {response.get('similar_prompts_count', 0)})"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTPException (already properly formatted)
        raise
    except ValueError as e:
        # Validation errors - convert to HTTPException 400
        error_msg = str(e)
        logger.error(
            f"Validation error in optimization flow (session: {session_id}, prompt_length: {len(prompt) if prompt else 0}): {error_msg}"
        )
        db.rollback()  # Ensure rollback on validation errors
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {error_msg}"
        )
    except RuntimeError as e:
        # Runtime errors (API failures, etc.) - convert to HTTPException 500 or 503
        error_msg = str(e)
        logger.error(
            f"Runtime error in optimization flow (session: {session_id}, prompt_length: {len(prompt) if prompt else 0}): {error_msg}"
        )
        db.rollback()  # Ensure rollback on runtime errors
        
        # Check if it's an API error (OpenAI or Groq)
        if "API" in error_msg or "OpenAI" in error_msg or "Groq" in error_msg:
            raise HTTPException(
                status_code=503,
                detail=f"External API service unavailable: {error_msg}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {error_msg}"
            )
    except Exception as e:
        # Unexpected errors - convert to HTTPException 500
        error_msg = str(e)
        error_type = type(e).__name__
        logger.error(
            f"Unexpected error in optimization flow (type: {error_type}, session: {session_id}, "
            f"prompt_length: {len(prompt) if prompt else 0}): {error_msg}",
            exc_info=True
        )
        db.rollback()  # Ensure rollback on unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {error_msg}"
        )
    finally:
        # Ensure database rollback on any unhandled exception
        # (Note: Most errors are already handled above with explicit rollbacks,
        # but this is a safety net for any edge cases)
        try:
            # Check if there's an active transaction that needs rollback
            # Individual steps handle their own rollbacks, but this ensures cleanup
            if db.is_active:
                # Only rollback if session is still active and has uncommitted changes
                # (Committed transactions don't need rollback)
                pass  # Individual steps handle their own rollbacks
        except Exception as rollback_error:
            # Log but don't raise - we're in cleanup
            logger.warning(f"Error during database cleanup: {rollback_error}")


@router.get("/prompts/{session_id}")
async def get_session_prompts(session_id: str, db: Session = Depends(get_db)):
    """
    Get all prompts for a specific session.

    Args:
        session_id: Session identifier
        db: Database session

    Returns:
        List of prompts for the session
    """
    # TODO: Implement database query
    return {"session_id": session_id, "prompts": []}
