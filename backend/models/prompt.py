import re

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

    # User usage tracking
    final_prompt_used = Column(Text, nullable=True)  # What they actually used (if different from optimized_prompt, they modified it)

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
        Index("idx_effectiveness", "optimization_effectiveness"),
        Index("idx_chatgpt_quality", "chatgpt_quality_score"),
    )

    def __repr__(self) -> str:
        """Return string representation of Prompt."""
        return f"<Prompt(id={self.id}, session_id='{self.session_id}', effectiveness={self.optimization_effectiveness})>"

    # Business logic methods
    def calculate_effectiveness_score(self) -> float:
        """Calculate overall effectiveness score based on ChatGPT quality and usage."""
        if not self.chatgpt_quality_score:
            return None

        # Base score from ChatGPT quality
        effectiveness = self.chatgpt_quality_score

        # Bonus if user actually used the optimized prompt (or modified version)
        if self.final_prompt_used:
            # If they used it (even if modified), that's a positive signal
            effectiveness = min(effectiveness + 0.1, 1.0)

        return round(effectiveness, 2)

    def is_high_quality(self) -> bool:
        """Check if this prompt optimization was high quality."""
        # High quality if ChatGPT quality is good and user used it
        return (
            self.chatgpt_quality_score
            and self.chatgpt_quality_score >= 0.8
            and self.final_prompt_used is not None  # User actually used it
        )
    
    def was_used(self) -> bool:
        """Check if the user actually used this prompt."""
        return self.final_prompt_used is not None
    
    def _contains_placeholders(self, text: str) -> list:
        """
        Find all placeholder patterns in text.
        
        Looks for patterns like [something], [specific task], etc.
        Returns list of placeholder strings found.
        """
        if not text:
            return []
        # Match patterns like [something], [specific task], [your task], etc.
        pattern = r'\[[^\]]+\]'
        placeholders = re.findall(pattern, text)
        return placeholders
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Levenshtein similarity between two texts (0-1 scale).
        
        Returns similarity score where 1.0 is identical, 0.0 is completely different.
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Simple Levenshtein distance implementation
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(text1.lower(), text2.lower())
        max_length = max(len(text1), len(text2))
        if max_length == 0:
            return 1.0
        
        similarity = 1.0 - (distance / max_length)
        return similarity
    
    def _is_placeholder_filling(self) -> bool:
        """
        Check if the change is just placeholder filling (not a real modification).
        
        Returns True if the only difference is filling in placeholders.
        """
        optimized = self.optimized_prompt.strip()
        final = self.final_prompt_used.strip()
        
        # Find placeholders in optimized prompt
        placeholders = self._contains_placeholders(optimized)
        if not placeholders:
            return False  # No placeholders, so can't be placeholder filling
        
        # Split optimized text by placeholders (keeping the text segments)
        parts = re.split(r'(\[[^\]]+\])', optimized)
        text_segments = [part for part in parts if not part.startswith('[')]
        
        # Get non-empty segments only (skip empty segments)
        non_empty_segments = [seg for seg in text_segments if seg.strip()]
        
        # Handle case where placeholder is at start (first segment is empty)
        starts_with_placeholder = len(parts) > 0 and parts[0].startswith('[')
        ends_with_placeholder = len(parts) > 0 and (parts[-1].startswith('[') or 
                                                      (len(parts) >= 2 and parts[-2].startswith('[') and not parts[-1].strip()))
        
        # Check if all non-empty text segments appear in the final text in the same order
        # This ensures the structure is preserved
        current_pos = 0
        all_segments_found = True
        
        for segment in non_empty_segments:
            segment_lower = segment.lower().strip()
            final_lower = final.lower()
            
            # For placeholder at start, the segment might have leading space
            # Normalize by removing punctuation for more flexible matching
            segment_normalized = re.sub(r'[^\w\s]', '', segment_lower)
            final_normalized_seg = re.sub(r'[^\w\s]', '', final_lower[current_pos:])
            
            # Find segment starting from current position (normalized)
            found_pos_normalized = final_normalized_seg.find(segment_normalized)
            if found_pos_normalized == -1:
                # Segment not found - might be a real modification
                all_segments_found = False
                break
            # Update current_pos based on where we found it in the original text
            # We need to find the position in the original final text
            segment_words = segment_normalized.split()
            if segment_words:
                # Find the first word of the segment in final text
                first_word = segment_words[0]
                word_pos = final_lower.find(first_word, current_pos)
                if word_pos != -1:
                    current_pos = word_pos + len(segment_lower)
                else:
                    all_segments_found = False
                    break
            else:
                current_pos += len(segment_lower)
        
        if not all_segments_found:
            # If segments don't match but placeholder is at start, 
            # we might still be able to match via text_after_placeholder check
            # So don't return False immediately - let the starts_with_placeholder logic handle it
            if not starts_with_placeholder:
                return False  # Text segments don't match - real modification
        
        # Calculate word count difference
        optimized_no_placeholders = optimized
        for placeholder in placeholders:
            optimized_no_placeholders = optimized_no_placeholders.replace(placeholder, "")
        
        # Normalize both texts (remove punctuation, lowercase)
        optimized_normalized = re.sub(r'[^\w\s]', '', optimized_no_placeholders.lower())
        final_normalized = re.sub(r'[^\w\s]', '', final.lower())
        
        optimized_words = optimized_normalized.split()
        final_words = final_normalized.split()
        
        # Calculate how many words the placeholders were replaced with
        placeholder_word_count = len(final_words) - len(optimized_words)
        
        # Improved logic: Split into two limits
        # 1. Allow up to 10 words for placeholder filling
        # 2. Allow up to 3 extra words after structure preservation (e.g., "please," "now")
        # Anything more counts as a real modification
        
        if placeholder_word_count > 10:
            return False  # Too many words for placeholder - real modification
        
        # For placeholder at start: check if text after placeholder matches
        if starts_with_placeholder:
            # Get text after first placeholder
            if len(parts) > 2:
                text_after_placeholder = parts[2].strip().lower()
                if text_after_placeholder:
                    # Normalize text_after_placeholder (remove punctuation for comparison)
                    text_after_normalized = re.sub(r'[^\w\s]', '', text_after_placeholder)
                    final_normalized_check = re.sub(r'[^\w\s]', '', final.lower())
                    
                    # Check if the normalized text after placeholder appears in final
                    # (allowing for the placeholder text to be filled in between)
                    if text_after_normalized in final_normalized_check:
                        # Text after placeholder matches - check word count
                        if placeholder_word_count <= 10 and placeholder_word_count >= 0:
                            return True  # Placeholder filling
                    else:
                        return False  # Text after placeholder changed - real modification
            # If no text after placeholder, just check word count and that segments matched
            if placeholder_word_count <= 10 and placeholder_word_count >= 0:
                return True  # Placeholder filling
        
        # For placeholder at end: check for extra content beyond placeholder filling
        if ends_with_placeholder:
            # Find the last non-empty segment
            last_non_empty_segment = None
            for seg in reversed(non_empty_segments):
                if seg.strip():
                    last_non_empty_segment = seg.strip().lower()
                    break
            
            if last_non_empty_segment:
                # Find where this segment ends in final text
                last_segment_pos = final.lower().find(last_non_empty_segment)
                if last_segment_pos != -1:
                    last_segment_end = last_segment_pos + len(last_non_empty_segment)
                    remaining_text = final.lower()[last_segment_end:].strip()
                    
                    if remaining_text:
                        # Count words in remaining text (this is placeholder text + any extra)
                        remaining_words = len(re.findall(r'\b\w+\b', remaining_text))
                        
                        # Check for common qualifier words that indicate extra content
                        # Words like "but", "and", "also", "then", "now" often signal modifications
                        qualifier_pattern = r'\b(but|and|also|then|now|plus|additionally|furthermore|moreover)\b'
                        has_qualifier = bool(re.search(qualifier_pattern, remaining_text, re.IGNORECASE))
                        
                        # Logic: Allow up to 10 words for placeholder, plus 3 extra for structure preservation
                        # If there's a qualifier word, it's likely extra content beyond placeholder
                        if has_qualifier and remaining_words > 4:
                            # Qualifier + more than 4 words = likely modification
                            # (1 word placeholder + qualifier + 2+ words = modification)
                            return False
                        
                        # If remaining_words > 10, it's likely extra content
                        if remaining_words > 10:
                            return False  # Too much - real modification
                        
                        # If 4-10 words and no qualifier, might be just placeholder text
                        # If <= 3 words, likely just placeholder
                        if remaining_words <= 3:
                            return True  # Short placeholder text - placeholder filling
                        elif remaining_words <= 10 and not has_qualifier:
                            # Could be longer placeholder text, allow it
                            return True
                        else:
                            # Has qualifier or too many words
                            return False  # Extra content - real modification
        
        # Placeholder not at start or end, or in middle
        # Check first word matches and word count is reasonable
        if len(optimized_words) > 0 and len(final_words) > 0:
            first_match = optimized_words[0] == final_words[0]
            if first_match:
                # Allow up to 10 words for placeholder, plus 3 extra for structure preservation
                if placeholder_word_count <= 13:
                    return True  # Structure preserved, just placeholder filling
        
        return False
    
    def _has_same_structure(self) -> bool:
        """
        Check if final_prompt_used has the same structure as optimized_prompt.
        
        Compares punctuation positions, sentence boundaries, and word order.
        Returns True if structure is similar (suggesting placeholder filling).
        """
        optimized = self.optimized_prompt.strip()
        final = self.final_prompt_used.strip()
        
        # Extract punctuation positions
        optimized_punct_positions = [i for i, char in enumerate(optimized) if char in '.,!?;:']
        final_punct_positions = [i for i, char in enumerate(final) if char in '.,!?;:']
        
        # If punctuation count differs significantly, structure changed
        if abs(len(optimized_punct_positions) - len(final_punct_positions)) > 2:
            return False
        
        # Extract sentence boundaries (periods, exclamation, question marks)
        optimized_sentences = re.split(r'[.!?]+', optimized)
        final_sentences = re.split(r'[.!?]+', final)
        
        # If sentence count differs, structure changed
        if abs(len(optimized_sentences) - len(final_sentences)) > 1:
            return False
        
        # Check word order similarity (excluding the changed section)
        optimized_words = re.findall(r'\b\w+\b', optimized.lower())
        final_words = re.findall(r'\b\w+\b', final.lower())
        
        # If word count differs significantly, structure changed
        word_diff = abs(len(optimized_words) - len(final_words))
        if word_diff > max(5, len(optimized_words) * 0.3):
            return False
        
        # Check if first and last few words match (structure preserved)
        if len(optimized_words) >= 2 and len(final_words) >= 2:
            if optimized_words[:2] == final_words[:2] and optimized_words[-2:] == final_words[-2:]:
                return True
        
        return False
    
    def was_modified(self) -> bool:
        """
        Check if the user modified the optimized prompt.
        
        Distinguishes between:
        - Placeholder filling (expected behavior, returns False)
        - Real modifications (structural changes, returns True)
        
        Flow (as suggested by friend):
        1. exact match
        2. placeholder filling
        3. same structure
        4. high text similarity (fallback)
        5. else = modified
        """
        if not self.final_prompt_used or not self.optimized_prompt:
            return False
        
        optimized = self.optimized_prompt.strip()
        final = self.final_prompt_used.strip()
        
        # 1. Fast path: exact match
        if final == optimized:
            return False
        
        # 2. Check for placeholder filling (most common case)
        if self._is_placeholder_filling():
            return False
        
        # 3. Check structure similarity (catches edge cases)
        if self._has_same_structure():
            return False
        
        # 4. High text similarity fallback (prevents false positives from punctuation/small rephrasing)
        # Remove placeholders for similarity check
        optimized_no_placeholders = optimized
        placeholders = self._contains_placeholders(optimized)
        for placeholder in placeholders:
            optimized_no_placeholders = optimized_no_placeholders.replace(placeholder, "")
        
        # Normalize for similarity check (remove extra whitespace, lowercase)
        optimized_normalized = re.sub(r'\s+', ' ', optimized_no_placeholders.lower().strip())
        final_normalized = re.sub(r'\s+', ' ', final.lower().strip())
        
        similarity = self._text_similarity(optimized_normalized, final_normalized)
        if similarity > 0.85:
            return False  # High similarity - likely not modified (just punctuation/rephrasing)
        
        # 5. Otherwise, it's a real modification
        return True

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

        # Calculate average ChatGPT quality scores
        chatgpt_quality_scores = [
            p.chatgpt_quality_score
            for p in self.prompts
            if p.chatgpt_quality_score is not None
        ]

        if chatgpt_quality_scores:
            # Use ChatGPT quality as a proxy for optimization quality
            self.average_optimization_rating = sum(chatgpt_quality_scores) / len(
                chatgpt_quality_scores
            )
            self.average_chatgpt_rating = self.average_optimization_rating

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
        # Highly active if user has many prompts and uses them
        used_prompts = [p for p in self.prompts if p.final_prompt_used]
        return (
            self.total_prompts >= 10
            and len(used_prompts) >= 5  # At least 5 prompts were actually used
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
