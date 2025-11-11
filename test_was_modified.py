"""
Test script for was_modified() method in Prompt model.

Run this from project root with venv activated:
    python test_was_modified.py

This script tests the hybrid approach for distinguishing between:
- Placeholder filling (expected behavior, should return False)
- Real modifications (structural changes, should return True)
"""

import sys
import logging
from backend.app.core.database import SessionLocal, create_tables
from backend.models.prompt import Prompt, Session as SessionModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_exact_match():
    """Test that exact matches return False."""
    print("\n" + "="*60)
    print("TEST 1: Exact Match")
    print("="*60)
    
    db = SessionLocal()
    try:
        session = SessionModel(id="test_session_exact")
        db.add(session)
        db.flush()
        
        prompt = Prompt(
            session_id="test_session_exact",
            original_prompt="help me code",
            optimized_prompt="Please help me write clean, well-documented code",
            final_prompt_used="Please help me write clean, well-documented code",
        )
        
        result = prompt.was_modified()
        assert result == False, f"Expected False for exact match, got {result}"
        print("   ✅ Exact match correctly returns False")
        
        # Cleanup
        db.query(Prompt).filter(Prompt.session_id == "test_session_exact").delete()
        db.query(SessionModel).filter(SessionModel.id == "test_session_exact").delete()
        db.commit()
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def test_placeholder_filling():
    """Test that placeholder filling returns False."""
    print("\n" + "="*60)
    print("TEST 2: Placeholder Filling (Should Return False)")
    print("="*60)
    
    db = SessionLocal()
    try:
        session = SessionModel(id="test_session_placeholder")
        db.add(session)
        db.flush()
        
        test_cases = [
            (
                "Please help me with [specific task]",
                "Please help me with write Python code",
                "Single placeholder filled"
            ),
            (
                "Help me [action] using [language]",
                "Help me write code using Python",
                "Multiple placeholders filled"
            ),
            (
                "Please [action] for [task] and explain it",
                "Please write code for debugging and explain it",
                "Placeholders with text around them"
            ),
            (
                "I need help with [task].",
                "I need help with coding.",
                "Placeholder at end of sentence"
            ),
            (
                "[Task] is what I need help with.",
                "Debugging is what I need help with.",
                "Placeholder at start of sentence"
            ),
        ]
        
        for i, (optimized, final, description) in enumerate(test_cases):
            prompt = Prompt(
                session_id="test_session_placeholder",
                original_prompt="test",
                optimized_prompt=optimized,
                final_prompt_used=final,
            )
            
            result = prompt.was_modified()
            assert result == False, f"Expected False for placeholder filling: {description}"
            print(f"   ✅ {description}: Correctly returns False")
        
        # Cleanup
        db.query(Prompt).filter(Prompt.session_id == "test_session_placeholder").delete()
        db.query(SessionModel).filter(SessionModel.id == "test_session_placeholder").delete()
        db.commit()
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def test_real_modifications():
    """Test that real modifications return True."""
    print("\n" + "="*60)
    print("TEST 3: Real Modifications (Should Return True)")
    print("="*60)
    
    db = SessionLocal()
    try:
        session = SessionModel(id="test_session_modified")
        db.add(session)
        db.flush()
        
        test_cases = [
            (
                "Please help me with [task]",
                "Please help me with coding and make it detailed with comments",
                "Added extra content (real modification)"
            ),
            (
                "Help me write code",
                "Help me write code and also debug it",
                "Added sentence (real modification)"
            ),
            (
                "Please help me with [task]",
                "I need assistance with coding",
                "Changed structure completely (real modification)"
            ),
            (
                "Write code for me",
                "Write code for me. Also add tests.",
                "Added new sentence (real modification)"
            ),
            (
                "Help with [task]",
                "Help with coding but make it simple",
                "Added qualifier (real modification)"
            ),
        ]
        
        for i, (optimized, final, description) in enumerate(test_cases):
            prompt = Prompt(
                session_id="test_session_modified",
                original_prompt="test",
                optimized_prompt=optimized,
                final_prompt_used=final,
            )
            
            result = prompt.was_modified()
            assert result == True, f"Expected True for real modification: {description}"
            print(f"   ✅ {description}: Correctly returns True")
        
        # Cleanup
        db.query(Prompt).filter(Prompt.session_id == "test_session_modified").delete()
        db.query(SessionModel).filter(SessionModel.id == "test_session_modified").delete()
        db.commit()
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print("TEST 4: Edge Cases")
    print("="*60)
    
    db = SessionLocal()
    try:
        session = SessionModel(id="test_session_edge")
        db.add(session)
        db.flush()
        
        # Edge case 1: No placeholders but similar structure
        prompt1 = Prompt(
            session_id="test_session_edge",
            original_prompt="test",
            optimized_prompt="Please help me write code",
            final_prompt_used="Please help me write Python code",
        )
        result1 = prompt1.was_modified()
        # This should return True (no placeholders, so any change is modification)
        assert result1 == True, "Expected True when no placeholders but text changed"
        print("   ✅ No placeholders with text change: Correctly returns True")
        
        # Edge case 2: None values
        prompt2 = Prompt(
            session_id="test_session_edge",
            original_prompt="test",
            optimized_prompt="Please help me",
            final_prompt_used=None,
        )
        result2 = prompt2.was_modified()
        assert result2 == False, "Expected False when final_prompt_used is None"
        print("   ✅ None final_prompt_used: Correctly returns False")
        
        prompt3 = Prompt(
            session_id="test_session_edge",
            original_prompt="test",
            optimized_prompt=None,
            final_prompt_used="Please help me",
        )
        result3 = prompt3.was_modified()
        assert result3 == False, "Expected False when optimized_prompt is None"
        print("   ✅ None optimized_prompt: Correctly returns False")
        
        # Edge case 3: Whitespace differences
        prompt4 = Prompt(
            session_id="test_session_edge",
            original_prompt="test",
            optimized_prompt="Please help me with [task]",
            final_prompt_used="  Please help me with coding  ",
        )
        result4 = prompt4.was_modified()
        # Should return False (just whitespace + placeholder filling)
        assert result4 == False, "Expected False for whitespace differences with placeholder filling"
        print("   ✅ Whitespace differences with placeholder: Correctly returns False")
        
        # Cleanup
        db.query(Prompt).filter(Prompt.session_id == "test_session_edge").delete()
        db.query(SessionModel).filter(SessionModel.id == "test_session_edge").delete()
        db.commit()
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def test_placeholder_detection():
    """Test that placeholder detection works correctly."""
    print("\n" + "="*60)
    print("TEST 5: Placeholder Detection")
    print("="*60)
    
    db = SessionLocal()
    try:
        session = SessionModel(id="test_session_detect")
        db.add(session)
        db.flush()
        
        prompt = Prompt(
            session_id="test_session_detect",
            original_prompt="test",
            optimized_prompt="Test [placeholder] text",
            final_prompt_used="Test filled text",
        )
        
        # Test _contains_placeholders method
        placeholders = prompt._contains_placeholders("Test [placeholder] and [another] here")
        assert len(placeholders) == 2, f"Expected 2 placeholders, got {len(placeholders)}"
        assert "[placeholder]" in placeholders
        assert "[another]" in placeholders
        print("   ✅ Placeholder detection works correctly")
        
        # Test with no placeholders
        no_placeholders = prompt._contains_placeholders("Test without placeholders")
        assert len(no_placeholders) == 0, f"Expected 0 placeholders, got {len(no_placeholders)}"
        print("   ✅ No placeholder detection works correctly")
        
        # Cleanup
        db.query(Prompt).filter(Prompt.session_id == "test_session_detect").delete()
        db.query(SessionModel).filter(SessionModel.id == "test_session_detect").delete()
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
    print("WAS_MODIFIED() TEST SUITE")
    print("="*60)
    
    try:
        create_tables()
        print("✅ Database tables initialized")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
    
    test_exact_match()
    test_placeholder_filling()
    test_real_modifications()
    test_edge_cases()
    test_placeholder_detection()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

