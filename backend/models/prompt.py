"""
Enhanced Database models for Prompt Optimizer Backend with ChatGPT Integration

This module defines all database models using SQLAlchemy ORM with comprehensive
ChatGPT integration, user feedback tracking, and optimization effectiveness metrics.
"""

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.core.database import Base


class Prompt(Base):
    """Enhanced model for storing prompts with ChatGPT integration and optimization tracking."""

    __tablename__ = "prompts"

    # Core identification
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        String(255), ForeignKey("sessions.id"), index=True, nullable=False
    )

    # Core prompt data
    original_prompt = Column(Text, nullable=False)
    optimized_prompt = Column(Text, nullable=True)

    # AI/ML data
    embedding = Column(Text, nullable=True)  # JSON string of vector embedding
    similarity_score = Column(Float, nullable=True)
    context_prompts = Column(Text, nullable=True)  # JSON array of similar prompt IDs

    # ChatGPT integration
    chatgpt_output = Column(Text, nullable=True)  # What ChatGPT actually returned
    chatgpt_quality_score = Column(
        Float, nullable=True
    )  # How good was ChatGPT's response
    chatgpt_tokens_used = Column(Integer, nullable=True)
    chatgpt_response_time = Column(Integer, nullable=True)  # milliseconds

    # User feedback system
    user_rating_optimization = Column(
        Integer, nullable=True
    )  # 1-5 rating of our optimization
    user_rating_chatgpt = Column(
        Integer, nullable=True
    )  # 1-5 rating of ChatGPT's output
    user_feedback_optimization = Column(
        Text, nullable=True
    )  # Feedback on our optimization
    user_feedback_chatgpt = Column(
        Text, nullable=True
    )  # Feedback on ChatGPT's response
    user_action = Column(
        String(50), nullable=True
    )  # "accepted", "modified", "rejected"
    final_prompt_used = Column(Text, nullable=True)  # What they actually used

    # Optimization effectiveness metrics
    optimization_effectiveness = Column(
        Float, nullable=True
    )  # Did our optimization help ChatGPT?
    improvement_over_original = Column(
        Float, nullable=True
    )  # How much better was ChatGPT's response?

    # Optimization metadata
    optimization_method = Column(
        String(50), nullable=True
    )  # "groq", "manual", "cached"
    optimization_time_ms = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    session = relationship("Session", back_populates="prompts")

    # Indexes for better query performance
    __table_args__ = (
        Index("idx_session_created", "session_id", "created_at"),
        Index("idx_created_at", "created_at"),
        Index("idx_user_ratings", "user_rating_optimization", "user_rating_chatgpt"),
        Index("idx_effectiveness", "optimization_effectiveness"),
        Index("idx_chatgpt_quality", "chatgpt_quality_score"),
    )

    def __repr__(self) -> str:
        """Return string representation of Prompt."""
        return f"<Prompt(id={self.id}, session_id='{self.session_id}', effectiveness={self.optimization_effectiveness})>"

    # Business logic methods
    def calculate_effectiveness_score(self) -> float:
        """Calculate overall effectiveness score based on user ratings and ChatGPT quality."""
        if not self.user_rating_optimization or not self.chatgpt_quality_score:
            return None

        # Weighted average: 60% user rating, 40% ChatGPT quality
        effectiveness = (self.user_rating_optimization * 0.6) + (
            self.chatgpt_quality_score * 0.4
        )
        return round(effectiveness, 2)

    def is_high_quality(self) -> bool:
        """Check if this prompt optimization was high quality."""
        return (
            self.user_rating_optimization
            and self.user_rating_optimization >= 4
            and self.chatgpt_quality_score
            and self.chatgpt_quality_score >= 0.8
        )

    def get_context_prompts_list(self) -> list:
        """Get context prompts as a Python list."""
        import json

        if self.context_prompts:
            try:
                return json.loads(self.context_prompts)
            except json.JSONDecodeError:
                return []
        return []

    def set_context_prompts_list(self, prompt_ids: list) -> None:
        """Set context prompts from a Python list."""
        import json

        self.context_prompts = json.dumps(prompt_ids) if prompt_ids else None


class Session(Base):
    """Enhanced model for tracking user sessions with learning and analytics."""

    __tablename__ = "sessions"

    # Core identification
    id = Column(String(255), primary_key=True, index=True)
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 max length

    # Session lifecycle
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive

    # Context metadata and analytics
    total_prompts = Column(Integer, default=0)
    total_optimizations = Column(Integer, default=0)
    total_chatgpt_interactions = Column(Integer, default=0)
    average_optimization_rating = Column(Float, nullable=True)
    average_chatgpt_rating = Column(Float, nullable=True)
    last_optimization_at = Column(DateTime(timezone=True), nullable=True)
    last_chatgpt_interaction_at = Column(DateTime(timezone=True), nullable=True)

    # User preferences (learned over time)
    preferred_style = Column(
        String(50), nullable=True
    )  # "concise", "detailed", "technical"
    common_feedback_patterns = Column(Text, nullable=True)  # JSON of learned patterns

    # Relationships
    prompts = relationship("Prompt", back_populates="session")

    # Indexes for better query performance
    __table_args__ = (
        Index("idx_session_active", "is_active", "last_accessed"),
        Index("idx_session_analytics", "total_prompts", "average_optimization_rating"),
    )

    def __repr__(self) -> str:
        """Return string representation of Session."""
        return f"<Session(id='{self.id}', prompts={self.total_prompts}, avg_rating={self.average_optimization_rating})>"

    # Business logic methods
    def update_analytics(self, prompt: "Prompt") -> None:
        """Update session analytics based on a new prompt."""
        self.total_prompts += 1

        if prompt.optimized_prompt:
            self.total_optimizations += 1
            self.last_optimization_at = func.now()

        if prompt.chatgpt_output:
            self.total_chatgpt_interactions += 1
            self.last_chatgpt_interaction_at = func.now()

        # Update average ratings
        self._update_average_ratings()

    def _update_average_ratings(self) -> None:
        """Update average ratings based on all prompts in this session."""
        if not self.prompts:
            return

        optimization_ratings = [
            p.user_rating_optimization
            for p in self.prompts
            if p.user_rating_optimization
        ]
        chatgpt_ratings = [
            p.user_rating_chatgpt for p in self.prompts if p.user_rating_chatgpt
        ]

        if optimization_ratings:
            self.average_optimization_rating = sum(optimization_ratings) / len(
                optimization_ratings
            )

        if chatgpt_ratings:
            self.average_chatgpt_rating = sum(chatgpt_ratings) / len(chatgpt_ratings)

    def get_feedback_patterns(self) -> dict:
        """Get common feedback patterns as a Python dict."""
        import json

        if self.common_feedback_patterns:
            try:
                return json.loads(self.common_feedback_patterns)
            except json.JSONDecodeError:
                return {}
        return {}

    def set_feedback_patterns(self, patterns: dict) -> None:
        """Set common feedback patterns from a Python dict."""
        import json

        self.common_feedback_patterns = json.dumps(patterns) if patterns else None

    def is_highly_active(self) -> bool:
        """Check if this session is highly active based on usage patterns."""
        return (
            self.total_prompts >= 10
            and self.average_optimization_rating
            and self.average_optimization_rating >= 4.0
        )


class CacheEntry(Base):
    """Enhanced model for caching optimization results with quality metrics."""

    __tablename__ = "cache_entries"

    # Cache identification
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(255), unique=True, index=True, nullable=False)
    cache_value = Column(Text, nullable=False)  # JSON string

    # Cache lifecycle
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Cache metadata and analytics
    cache_type = Column(
        String(50), nullable=True
    )  # "embedding", "optimization", "similarity", "chatgpt_response"
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), nullable=True)

    # ChatGPT integration and quality metrics
    chatgpt_output_quality = Column(
        Float, nullable=True
    )  # Quality of ChatGPT response for this cache
    optimization_effectiveness = Column(
        Float, nullable=True
    )  # How effective was this optimization

    # Indexes for better query performance
    __table_args__ = (
        Index("idx_cache_type_expires", "cache_type", "expires_at"),
        Index("idx_cache_hits", "hit_count", "last_accessed"),
        Index(
            "idx_cache_quality", "chatgpt_output_quality", "optimization_effectiveness"
        ),
    )

    def __repr__(self) -> str:
        """Return string representation of CacheEntry."""
        return f"<CacheEntry(key='{self.cache_key}', type='{self.cache_type}', hits={self.hit_count})>"

    # Business logic methods
    def increment_hit_count(self) -> None:
        """Increment hit count and update last accessed time."""
        self.hit_count += 1
        self.last_accessed = func.now()

    def is_expired(self) -> bool:
        """Check if this cache entry is expired."""
        from datetime import datetime

        return datetime.utcnow() > self.expires_at

    def is_high_quality(self) -> bool:
        """Check if this cache entry represents high-quality optimization."""
        return (
            self.chatgpt_output_quality
            and self.chatgpt_output_quality >= 0.8
            and self.optimization_effectiveness
            and self.optimization_effectiveness >= 0.7
        )

    def get_cache_value_dict(self) -> dict:
        """Get cache value as a Python dict."""
        import json

        if self.cache_value:
            try:
                return json.loads(self.cache_value)
            except json.JSONDecodeError:
                return {}
        return {}

    def set_cache_value_dict(self, value: dict) -> None:
        """Set cache value from a Python dict."""
        import json

        self.cache_value = json.dumps(value) if value else "{}"
