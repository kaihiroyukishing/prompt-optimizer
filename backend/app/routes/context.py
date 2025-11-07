from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.models.prompt import Prompt, Session as SessionModel
from backend.utils.db_helpers import analyze_chatgpt_quality

router = APIRouter()


class ChatGPTResponseRequest(BaseModel):
    prompt_id: str
    conversation_id: str
    chatgpt_output: str
    chatgpt_response_time: int
    message_id: Optional[str] = None
    session_id: str
    original_prompt: Optional[str] = None
    optimized_prompt: Optional[str] = None


@router.post("/save-context")
async def save_context(
    prompt: str,
    optimized_prompt: str,
    session_id: str,
    embedding: Optional[List[float]] = None,
    db: Session = Depends(get_db),
):
    """
    Save prompt context for future optimization.

    Args:
        prompt: Original prompt
        optimized_prompt: Optimized version
        session_id: Session identifier
        embedding: Vector embedding of the prompt
        db: Database session

    Returns:
        Confirmation of saved context
    """
    # TODO: Implement context saving logic
    return {
        "message": "Context saved successfully",
        "session_id": session_id,
        "prompt_id": "placeholder_id",
    }


@router.get("/context/{session_id}")
async def get_context(session_id: str, limit: int = 10, db: Session = Depends(get_db)):
    """
    Get context for a session.

    Args:
        session_id: Session identifier
        limit: Maximum number of contexts to return
        db: Database session

    Returns:
        List of context entries
    """
    # TODO: Implement context retrieval logic
    return {"session_id": session_id, "contexts": [], "total_count": 0}


@router.delete("/context/{session_id}")
async def clear_context(session_id: str, db: Session = Depends(get_db)):
    """
    Clear all context for a session.

    Args:
        session_id: Session identifier
        db: Database session

    Returns:
        Confirmation of cleared context
    """
    # TODO: Implement context clearing logic
    return {
        "message": f"Context cleared for session {session_id}",
        "session_id": session_id,
    }


@router.post("/save-chatgpt-response")
async def save_chatgpt_response(
    request: ChatGPTResponseRequest,
    db: Session = Depends(get_db),
):
    session_obj = db.query(SessionModel).filter(SessionModel.id == request.session_id).first()
    if not session_obj:
        session_obj = SessionModel(id=request.session_id)
        db.add(session_obj)
        db.flush()
    
    prompt = Prompt(
        session_id=request.session_id,
        original_prompt=request.original_prompt or "",
        optimized_prompt=request.optimized_prompt,
        chatgpt_output=request.chatgpt_output,
        chatgpt_response_time=request.chatgpt_response_time,
    )
    db.add(prompt)
    db.flush()
    
    quality_analysis = analyze_chatgpt_quality(prompt)
    prompt.chatgpt_quality_score = quality_analysis.get("quality_score", 0.0)
    prompt.optimization_effectiveness = prompt.calculate_effectiveness_score()
    
    session_obj.update_analytics(prompt)
    session_obj.total_chatgpt_interactions += 1
    
    db.commit()
    
    return {
        "success": True,
        "message": "ChatGPT response saved successfully",
        "prompt_id": prompt.id,
        "quality_score": prompt.chatgpt_quality_score,
    }
