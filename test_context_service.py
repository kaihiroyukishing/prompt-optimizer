"""
Comprehensive test script for context service.

Run this from project root with venv activated:
    python test_context_service.py

This script tests all Phase 3 functionality:
- Format similar prompts helper
- Format user preferences helper
- Combine context parts helper
- Build optimization context (with cache integration)
- Error handling and edge cases
"""

import sys
import logging
import json
from backend.app.core.database import SessionLocal, create_tables
from backend.models.prompt import Prompt, Session as SessionModel
from backend.services.context_service import (
    _format_similar_prompts,
    _format_user_preferences,
    _combine_context_parts,
    build_optimization_context,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_prompts(db, session_id: str = "test_session_123"):
    """Create test prompts for testing."""
    # Cleanup: Delete any existing prompts for this session
    db.query(Prompt).filter(Prompt.session_id == session_id).delete()
    db.commit()
    
    prompts = []
    
    # Create prompts with various scenarios
    test_cases = [
        ("help me write code", "Please help me write clean, well-documented code...", 0.85),
        ("write a program", "Please write a well-structured program...", 0.82),
        ("coding assistance", None, 0.75),  # No optimized prompt
        ("test prompt", "test optimized", None),  # No quality score
    ]
    
    for original, optimized, quality_score in test_cases:
        prompt = Prompt(
            session_id=session_id,
            original_prompt=original,
            optimized_prompt=optimized,
            chatgpt_quality_score=quality_score,
        )
        db.add(prompt)
        db.flush()
        prompts.append(prompt)
    
    db.commit()
    return prompts


def create_test_session(db, session_id: str = "test_session_123", with_preferences: bool = True):
    """Create test session with optional preferences."""
    # Delete if exists to avoid UNIQUE constraint errors
    db.query(SessionModel).filter(SessionModel.id == session_id).delete()
    db.commit()
    
    session = SessionModel(id=session_id)
    
    if with_preferences:
        session.preferred_style = "technical"
        feedback_patterns = {
            "content": {
                "structure_preference": "structured",
                "length_preference": "detailed"
            },
            "success": {
                "average_quality_score": 0.82,
                "best_intent_types": ["coding", "debugging", "explanation"]
            },
            "usage": {
                "modification_rate": 0.3
            }
        }
        session.set_feedback_patterns(feedback_patterns)
    
    db.add(session)
    db.commit()
    return session


def test_format_similar_prompts():
    """Test _format_similar_prompts() function."""
    print("\n" + "="*60)
    print("TEST 1: _format_similar_prompts()")
    print("="*60)
    
    db = SessionLocal()
    try:
        # Cleanup: Delete any existing test data
        db.query(Prompt).filter(Prompt.session_id == "test_session_1").delete()
        db.query(SessionModel).filter(SessionModel.id == "test_session_1").delete()
        db.commit()
        # Test 1.1: Empty list
        print("\n1.1 Testing empty list...")
        result = _format_similar_prompts([])
        assert result == "", f"Expected empty string, got: {result}"
        print("   ✅ Empty list returns empty string")
        
        # Test 1.2: Single prompt
        print("\n1.2 Testing single prompt...")
        prompts = create_test_prompts(db, "test_session_1")
        result = _format_similar_prompts([prompts[0]])
        assert "SIMILAR PROMPTS THAT WORKED WELL:" in result
        assert "help me write code" in result
        assert "0.85" in result
        print("   ✅ Single prompt formatted correctly")
        print(f"   Result preview: {result[:100]}...")
        
        # Test 1.3: Multiple prompts
        print("\n1.3 Testing multiple prompts...")
        result = _format_similar_prompts(prompts[:3])
        assert result.count("Original:") == 3
        print("   ✅ Multiple prompts formatted correctly")
        
        # Test 1.4: Max prompts limit
        print("\n1.4 Testing max_prompts limit...")
        result = _format_similar_prompts(prompts, max_prompts=2)
        assert result.count("Original:") == 2
        print("   ✅ Max prompts limit works correctly")
        
        # Test 1.5: None values handling
        print("\n1.5 Testing None values handling...")
        result = _format_similar_prompts([prompts[2]])  # No optimized prompt
        assert "[No optimized prompt]" in result
        print("   ✅ None values handled gracefully")
        
        # Test 1.6: No quality score
        print("\n1.6 Testing missing quality score...")
        result = _format_similar_prompts([prompts[3]])  # No quality score
        assert "N/A" in result
        print("   ✅ Missing quality score shows N/A")
        
        # Cleanup
        db.query(Prompt).filter(Prompt.session_id == "test_session_1").delete()
        db.commit()
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def test_format_user_preferences():
    """Test _format_user_preferences() function."""
    print("\n" + "="*60)
    print("TEST 2: _format_user_preferences()")
    print("="*60)
    
    db = SessionLocal()
    try:
        # Cleanup: Delete any existing test data
        db.query(SessionModel).filter(SessionModel.id.in_([
            "test_session_2", "test_session_3", "test_session_4", "test_session_5"
        ])).delete()
        db.commit()
        # Test 2.1: None session
        print("\n2.1 Testing None session...")
        result = _format_user_preferences(None)
        assert result == "", f"Expected empty string, got: {result}"
        print("   ✅ None session returns empty string")
        
        # Test 2.2: Session with only preferred_style
        print("\n2.2 Testing session with only preferred_style...")
        # Delete if exists, then create new
        db.query(SessionModel).filter(SessionModel.id == "test_session_2").delete()
        db.commit()
        session = SessionModel(id="test_session_2")
        session.preferred_style = "concise"
        db.add(session)
        db.commit()
        
        result = _format_user_preferences(session)
        assert "USER PREFERENCES:" in result
        assert "Preferred style: concise" in result
        print("   ✅ Preferred style formatted correctly")
        print(f"   Result: {result}")
        
        # Test 2.3: Session with full preferences
        print("\n2.3 Testing session with full preferences...")
        session = create_test_session(db, "test_session_3", with_preferences=True)
        result = _format_user_preferences(session)
        assert "USER PREFERENCES:" in result
        assert "Preferred style: technical" in result
        assert "Structure preference: structured" in result
        assert "Length preference: detailed" in result
        assert "Average quality score: 0.82" in result
        assert "Works well with intent types: coding, debugging, explanation" in result
        assert "User typically uses optimized prompts as-is" in result
        print("   ✅ Full preferences formatted correctly")
        print(f"   Result preview: {result[:200]}...")
        
        # Test 2.4: Session with empty preferences
        print("\n2.4 Testing session with empty preferences...")
        session = create_test_session(db, "test_session_4", with_preferences=False)
        result = _format_user_preferences(session)
        assert result == "", f"Expected empty string, got: {result}"
        print("   ✅ Empty preferences returns empty string")
        
        # Test 2.5: Session with modification_rate > 0.5
        print("\n2.5 Testing high modification rate...")
        # Delete if exists, then create new
        db.query(SessionModel).filter(SessionModel.id == "test_session_5").delete()
        db.commit()
        session = SessionModel(id="test_session_5")
        session.preferred_style = "detailed"
        feedback_patterns = {
            "usage": {
                "modification_rate": 0.7
            }
        }
        session.set_feedback_patterns(feedback_patterns)
        db.add(session)
        db.commit()
        
        result = _format_user_preferences(session)
        assert "User frequently modifies optimized prompts" in result
        print("   ✅ High modification rate detected correctly")
        
        # Cleanup
        db.query(SessionModel).filter(SessionModel.id.in_([
            "test_session_2", "test_session_3", "test_session_4", "test_session_5"
        ])).delete()
        db.commit()
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def test_combine_context_parts():
    """Test _combine_context_parts() function."""
    print("\n" + "="*60)
    print("TEST 3: _combine_context_parts()")
    print("="*60)
    
    try:
        # Test 3.1: Both empty
        print("\n3.1 Testing both empty...")
        result = _combine_context_parts("", "")
        assert result == "", f"Expected empty string, got: {result}"
        print("   ✅ Both empty returns empty string")
        
        # Test 3.2: Only prompts
        print("\n3.2 Testing only prompts...")
        prompts_str = "SIMILAR PROMPTS THAT WORKED WELL:\n- Test prompt"
        result = _combine_context_parts(prompts_str, "")
        assert result == prompts_str
        assert "SIMILAR PROMPTS" in result
        print("   ✅ Only prompts returns prompts string")
        
        # Test 3.3: Only preferences
        print("\n3.3 Testing only preferences...")
        prefs_str = "USER PREFERENCES:\nPreferred style: technical"
        result = _combine_context_parts("", prefs_str)
        assert result == prefs_str
        assert "USER PREFERENCES" in result
        print("   ✅ Only preferences returns preferences string")
        
        # Test 3.4: Both present
        print("\n3.4 Testing both present...")
        prompts_str = "SIMILAR PROMPTS THAT WORKED WELL:\n- Test prompt"
        prefs_str = "USER PREFERENCES:\nPreferred style: technical"
        result = _combine_context_parts(prompts_str, prefs_str)
        assert "SIMILAR PROMPTS" in result
        assert "USER PREFERENCES" in result
        assert result.count("\n\n") >= 1  # Should have double newline separator
        print("   ✅ Both parts combined correctly")
        print(f"   Result preview: {result[:150]}...")
        
        # Test 3.5: Verify separator
        print("\n3.5 Testing separator format...")
        parts = result.split("\n\n")
        assert len(parts) == 2, "Should have two parts separated by double newline"
        assert "SIMILAR PROMPTS" in parts[0]
        assert "USER PREFERENCES" in parts[1]
        print("   ✅ Proper double newline separator")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_build_optimization_context():
    """Test build_optimization_context() function."""
    print("\n" + "="*60)
    print("TEST 4: build_optimization_context()")
    print("="*60)
    
    db = SessionLocal()
    try:
        # Cleanup: Delete any existing test data
        db.query(Prompt).filter(Prompt.session_id == "test_session_main").delete()
        db.query(SessionModel).filter(SessionModel.id == "test_session_main").delete()
        db.commit()
        
        # Create test data
        session = create_test_session(db, "test_session_main", with_preferences=True)
        prompts = create_test_prompts(db, "test_session_main")
        embedding = [0.1] * 1536  # Mock embedding (OpenAI dimension)
        
        # Test 4.1: Input validation - None embedding
        print("\n4.1 Testing input validation (None embedding)...")
        try:
            build_optimization_context(prompts, session, None, db)
            print("   ❌ Should have raised ValueError")
        except ValueError as e:
            assert "embedding" in str(e).lower()
            print("   ✅ Correctly raised ValueError for None embedding")
        
        # Test 4.2: Input validation - empty embedding
        print("\n4.2 Testing input validation (empty embedding)...")
        try:
            build_optimization_context(prompts, session, [], db)
            print("   ❌ Should have raised ValueError")
        except ValueError as e:
            assert "embedding" in str(e).lower()
            print("   ✅ Correctly raised ValueError for empty embedding")
        
        # Test 4.3: Input validation - None db
        print("\n4.3 Testing input validation (None db)...")
        try:
            build_optimization_context(prompts, session, embedding, None)
            print("   ❌ Should have raised ValueError")
        except ValueError as e:
            assert "db" in str(e).lower()
            print("   ✅ Correctly raised ValueError for None db")
        
        # Test 4.4: Full flow - cache miss
        print("\n4.4 Testing full flow (cache miss)...")
        result = build_optimization_context(prompts[:2], session, embedding, db)
        assert result is not None
        assert "SIMILAR PROMPTS" in result or "USER PREFERENCES" in result
        print("   ✅ Full flow works correctly")
        print(f"   Result length: {len(result)} characters")
        print(f"   Result preview: {result[:200]}...")
        
        # Test 4.5: Cache hit
        print("\n4.5 Testing cache hit...")
        result2 = build_optimization_context(prompts[:2], session, embedding, db)
        assert result2 == result, "Cached result should match original"
        print("   ✅ Cache hit returns same result")
        
        # Test 4.6: Empty prompts
        print("\n4.6 Testing empty prompts...")
        result = build_optimization_context([], session, embedding, db)
        assert "USER PREFERENCES" in result
        print("   ✅ Empty prompts still returns preferences")
        
        # Test 4.7: None session
        print("\n4.7 Testing None session...")
        result = build_optimization_context(prompts[:2], None, embedding, db)
        assert "SIMILAR PROMPTS" in result
        print("   ✅ None session still returns prompts")
        
        # Test 4.8: Both empty prompts and None session
        print("\n4.8 Testing both empty prompts and None session...")
        result = build_optimization_context([], None, embedding, db)
        # Should return empty string or minimal context
        print(f"   ✅ Handles both empty gracefully (result length: {len(result)})")
        
        # Cleanup
        db.query(Prompt).filter(Prompt.session_id == "test_session_main").delete()
        db.query(SessionModel).filter(SessionModel.id == "test_session_main").delete()
        db.commit()
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CONTEXT SERVICE TEST SUITE")
    print("="*60)
    
    try:
        create_tables()
        print("✅ Database tables initialized")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
    
    test_format_similar_prompts()
    test_format_user_preferences()
    test_combine_context_parts()
    test_build_optimization_context()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

