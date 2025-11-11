"""
Comprehensive test script for Groq optimization service.

Run this from project root with venv activated:
    python test_groq_service.py

This script tests all Phase 4 functionality:
- Groq client initialization
- System prompt building helper
- User message building helper
- Groq API call (mocked)
- Response parsing and cleaning
- Input validation
- Full optimization flow
- Error handling and edge cases
"""

import sys
import logging
import os
from unittest.mock import Mock, patch, MagicMock
from backend.app.core.database import SessionLocal, create_tables
from backend.models.prompt import Session as SessionModel
from backend.services.groq_service import (
    _get_groq_client,
    _build_system_prompt,
    _build_user_message,
    _call_groq_api,
    _clean_optimized_prompt,
    optimize_with_groq,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_session(db, session_id: str = "test_session_123", with_preferences: bool = True):
    """Create test session with optional preferences."""
    from backend.models.prompt import Session as SessionModel
    
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


@patch('backend.services.groq_service.settings')
def test_groq_client_initialization(mock_settings):
    """Test Groq client initialization."""
    print("\n" + "="*60)
    print("TEST 1: Groq Client Initialization")
    print("="*60)
    
    # Get original API key from actual settings
    from backend.app.core.config import settings as real_settings
    original_key = real_settings.GROQ_API_KEY
    
    # Clear the module-level client cache
    import backend.services.groq_service as groq_module
    groq_module._groq_client = None
    
    try:
        # Test 1.1: Valid API key
        if original_key and original_key != "your_groq_api_key_here":
            print("\n✓ Test 1.1: Valid API key")
            mock_settings.GROQ_API_KEY = original_key
            groq_module._groq_client = None  # Clear cache
            
            client = _get_groq_client()
            assert client is not None, "Client should be initialized"
            print("  ✓ Client initialized successfully")
        else:
            print("\n⚠ Test 1.1: Skipped (GROQ_API_KEY not set or is placeholder)")
        
        # Test 1.2: Missing API key
        print("\n✓ Test 1.2: Missing API key")
        mock_settings.GROQ_API_KEY = ""
        groq_module._groq_client = None  # Clear cache
        
        try:
            _get_groq_client()
            assert False, "Should raise ValueError for missing API key"
        except ValueError as e:
            assert "not configured" in str(e).lower(), f"Error message should mention configuration: {e}"
            print(f"  ✓ Correctly raised ValueError: {e}")
        
        # Test 1.3: Placeholder API key
        print("\n✓ Test 1.3: Placeholder API key")
        mock_settings.GROQ_API_KEY = "your_groq_api_key_here"
        groq_module._groq_client = None  # Clear cache
        
        try:
            _get_groq_client()
            assert False, "Should raise ValueError for placeholder API key"
        except ValueError as e:
            assert "not configured" in str(e).lower(), f"Error message should mention configuration: {e}"
            print(f"  ✓ Correctly raised ValueError: {e}")
        
        # Test 1.4: None API key
        print("\n✓ Test 1.4: None API key")
        mock_settings.GROQ_API_KEY = None
        groq_module._groq_client = None  # Clear cache
        
        try:
            _get_groq_client()
            assert False, "Should raise ValueError for None API key"
        except ValueError as e:
            assert "not configured" in str(e).lower(), f"Error message should mention configuration: {e}"
            print(f"  ✓ Correctly raised ValueError: {e}")
        
    finally:
        # Clear client cache
        groq_module._groq_client = None
    
    print("\n✅ All client initialization tests passed!")


def test_build_system_prompt():
    """Test system prompt building helper."""
    print("\n" + "="*60)
    print("TEST 2: System Prompt Building")
    print("="*60)
    
    db = SessionLocal()
    try:
        # Test 2.1: With context and preferences
        print("\n✓ Test 2.1: With context and preferences")
        session = create_test_session(db, "test_session_sys1", with_preferences=True)
        context = "SIMILAR PROMPTS THAT WORKED WELL:\n- Original: 'help me code'\n  Optimized: 'Please help me...'"
        
        system_prompt = _build_system_prompt(context, session)
        assert "You are a prompt optimization expert" in system_prompt
        assert "CONTEXT FROM SIMILAR SUCCESSFUL OPTIMIZATIONS" in system_prompt
        assert "USER PREFERENCES" in system_prompt
        assert "technical" in system_prompt  # preferred_style
        assert "HANDLING MISSING INFORMATION" in system_prompt
        print(f"  ✓ System prompt built: {len(system_prompt)} chars")
        print(f"  ✓ Contains context: {context[:50] in system_prompt}")
        print(f"  ✓ Contains preferences: {'technical' in system_prompt}")
        
        # Test 2.2: Empty context
        print("\n✓ Test 2.2: Empty context")
        system_prompt = _build_system_prompt("", session)
        assert "You are a prompt optimization expert" in system_prompt
        assert "CONTEXT FROM SIMILAR SUCCESSFUL OPTIMIZATIONS" not in system_prompt
        assert "USER PREFERENCES" in system_prompt
        print("  ✓ System prompt built without context section")
        
        # Test 2.3: No preferences
        print("\n✓ Test 2.3: No preferences")
        session_no_prefs = create_test_session(db, "test_session_sys2", with_preferences=False)
        system_prompt = _build_system_prompt(context, session_no_prefs)
        assert "You are a prompt optimization expert" in system_prompt
        assert "CONTEXT FROM SIMILAR SUCCESSFUL OPTIMIZATIONS" in system_prompt
        assert "USER PREFERENCES" not in system_prompt
        print("  ✓ System prompt built without preferences section")
        
        # Test 2.4: None context
        print("\n✓ Test 2.4: None context")
        system_prompt = _build_system_prompt(None, session)
        assert "You are a prompt optimization expert" in system_prompt
        assert "CONTEXT FROM SIMILAR SUCCESSFUL OPTIMIZATIONS" not in system_prompt
        print("  ✓ System prompt handles None context")
        
        # Test 2.5: Output format instructions always included
        print("\n✓ Test 2.5: Output format instructions")
        system_prompt = _build_system_prompt("", session)
        assert "OUTPUT FORMAT:" in system_prompt
        assert "Return ONLY the optimized prompt text" in system_prompt
        assert "Do NOT include prefixes" in system_prompt
        assert "Do NOT include explanations" in system_prompt
        assert "Do NOT include phrases like \"Remember:\"" in system_prompt
        assert "Return the optimized prompt text directly" in system_prompt
        print("  ✓ Output format instructions included in system prompt")
        
    finally:
        db.close()
    
    print("\n✅ All system prompt building tests passed!")


def test_build_user_message():
    """Test user message building helper."""
    print("\n" + "="*60)
    print("TEST 3: User Message Building")
    print("="*60)
    
    # Test 3.1: Normal prompt
    print("\n✓ Test 3.1: Normal prompt")
    prompt = "help me write code"
    user_message = _build_user_message(prompt)
    assert "Now optimize:" in user_message
    assert prompt in user_message
    print(f"  ✓ User message: {user_message}")
    
    # Test 3.2: Empty prompt
    print("\n✓ Test 3.2: Empty prompt")
    try:
        _build_user_message("")
        assert False, "Should raise ValueError for empty prompt"
    except ValueError as e:
        assert "cannot be empty" in str(e).lower()
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test 3.3: Whitespace-only prompt
    print("\n✓ Test 3.3: Whitespace-only prompt")
    try:
        _build_user_message("   ")
        assert False, "Should raise ValueError for whitespace-only prompt"
    except ValueError as e:
        assert "cannot be empty" in str(e).lower()
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test 3.4: Very long prompt (should log warning)
    print("\n✓ Test 3.4: Very long prompt")
    long_prompt = "a" * 6000
    user_message = _build_user_message(long_prompt)
    assert long_prompt in user_message
    print(f"  ✓ User message built for long prompt: {len(user_message)} chars")
    
    print("\n✅ All user message building tests passed!")


def test_clean_optimized_prompt():
    """Test response cleaning helper."""
    print("\n" + "="*60)
    print("TEST 4: Response Cleaning")
    print("="*60)
    
    # Test 4.1: Clean response (no cleaning needed)
    print("\n✓ Test 4.1: Clean response")
    clean_response = "Please help me write clean, well-documented code for [specific task]."
    cleaned = _clean_optimized_prompt(clean_response)
    assert cleaned == clean_response
    print(f"  ✓ Clean response unchanged: {cleaned}")
    
    # Test 4.2: Response with meta-commentary (realistic scenario)
    # Groq might still add prefixes despite instructions, so we test cleaning
    print("\n✓ Test 4.2: Response with meta-commentary")
    # Realistic case: prefix on same line as content (should be handled by prefix removal)
    response_with_prefix = "Optimized: Please help me write clean, well-documented code for [specific task]."
    cleaned = _clean_optimized_prompt(response_with_prefix)
    assert "Optimized:" not in cleaned
    assert "Please help me write clean" in cleaned
    print(f"  ✓ Prefix removed from same line: {cleaned}")
    
    # Test 4.2b: Response with separate meta-commentary line
    print("\n✓ Test 4.2b: Response with separate meta line")
    response_with_meta_line = "Please help me write clean code.\nRemember: Use best practices."
    cleaned = _clean_optimized_prompt(response_with_meta_line)
    assert "Remember:" not in cleaned
    assert "Please help me write clean code" in cleaned
    print(f"  ✓ Meta-commentary line removed: {cleaned}")
    
    # Test 4.3: Response with different prefix pattern
    print("\n✓ Test 4.3: Response with different prefix pattern")
    response_with_prefix = "Here's the optimized prompt: Please help me write code."
    cleaned = _clean_optimized_prompt(response_with_prefix)
    assert "Here's the optimized prompt:" not in cleaned
    assert "Please help me write code" in cleaned
    print(f"  ✓ Different prefix pattern removed: {cleaned}")
    
    # Test 4.4: Empty response
    print("\n✓ Test 4.4: Empty response")
    try:
        _clean_optimized_prompt("")
        assert False, "Should raise ValueError for empty response"
    except ValueError as e:
        assert "empty" in str(e).lower()
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test 4.5: Response that's too short
    print("\n✓ Test 4.5: Response that's too short")
    try:
        _clean_optimized_prompt("abc")
        assert False, "Should raise ValueError for too short response"
    except ValueError as e:
        assert "too short" in str(e).lower() or "empty" in str(e).lower()
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test 4.6: Response that appears to be answering
    print("\n✓ Test 4.6: Response that appears to be answering")
    answering_response = "I can help you with that. Let me write some code for you."
    cleaned = _clean_optimized_prompt(answering_response)
    # Should still return it but log warning
    assert len(cleaned) > 0
    print(f"  ✓ Answering response handled: {cleaned[:50]}...")
    
    print("\n✅ All response cleaning tests passed!")


def test_input_validation():
    """Test input validation in optimize_with_groq."""
    print("\n" + "="*60)
    print("TEST 5: Input Validation")
    print("="*60)
    
    db = SessionLocal()
    try:
        session = create_test_session(db, "test_session_val")
        
        # Test 5.1: None prompt
        print("\n✓ Test 5.1: None prompt")
        try:
            optimize_with_groq(None, "", session)
            assert False, "Should raise ValueError for None prompt"
        except ValueError as e:
            assert "cannot be none" in str(e).lower() or "cannot be" in str(e).lower()
            print(f"  ✓ Correctly raised ValueError: {e}")
        
        # Test 5.2: Empty prompt
        print("\n✓ Test 5.2: Empty prompt")
        try:
            optimize_with_groq("", "", session)
            assert False, "Should raise ValueError for empty prompt"
        except ValueError as e:
            assert "cannot be empty" in str(e).lower()
            print(f"  ✓ Correctly raised ValueError: {e}")
        
        # Test 5.3: Too long prompt
        print("\n✓ Test 5.3: Too long prompt")
        long_prompt = "a" * 10001
        try:
            optimize_with_groq(long_prompt, "", session)
            assert False, "Should raise ValueError for too long prompt"
        except ValueError as e:
            assert "too long" in str(e).lower()
            print(f"  ✓ Correctly raised ValueError: {e}")
        
        # Test 5.4: None session
        print("\n✓ Test 5.4: None session")
        try:
            optimize_with_groq("test prompt", "", None)
            assert False, "Should raise ValueError for None session"
        except ValueError as e:
            assert "cannot be none" in str(e).lower() or "cannot be" in str(e).lower()
            print(f"  ✓ Correctly raised ValueError: {e}")
        
        # Test 5.5: Invalid context type
        print("\n✓ Test 5.5: Invalid context type")
        try:
            optimize_with_groq("test prompt", 123, session)  # context should be str
            assert False, "Should raise ValueError for invalid context type"
        except ValueError as e:
            assert "must be a string" in str(e).lower()
            print(f"  ✓ Correctly raised ValueError: {e}")
        
        # Test 5.6: Valid inputs (None context is OK)
        print("\n✓ Test 5.6: Valid inputs")
        # This will fail at API call, but validation should pass
        try:
            optimize_with_groq("test prompt", None, session)
            # If we get here, validation passed (API call will fail without mocking)
        except (ValueError, RuntimeError) as e:
            # ValueError = validation failed, RuntimeError = API call failed (expected)
            if isinstance(e, ValueError):
                assert False, f"Validation should pass but got: {e}"
            print(f"  ✓ Validation passed (API call failed as expected): {type(e).__name__}")
        
    finally:
        db.close()
    
    print("\n✅ All input validation tests passed!")


@patch('backend.services.groq_service._get_groq_client')
def test_call_groq_api(mock_get_client):
    """Test Groq API call (mocked)."""
    print("\n" + "="*60)
    print("TEST 6: Groq API Call (Mocked)")
    print("="*60)
    
    # Mock Groq client and response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Please help me write clean, well-documented code."
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 150
    
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client
    
    # Test 6.1: Successful API call
    print("\n✓ Test 6.1: Successful API call")
    result = _call_groq_api("System prompt", "User message")
    assert "optimized_prompt" in result
    assert "tokens_used" in result
    assert "optimization_time_ms" in result
    assert result["optimized_prompt"] == "Please help me write clean, well-documented code."
    assert result["tokens_used"] == 150
    assert result["optimization_time_ms"] >= 0  # Can be 0 for mocked instant calls
    print(f"  ✓ API call successful: {result['optimized_prompt'][:50]}...")
    print(f"  ✓ Tokens used: {result['tokens_used']}")
    print(f"  ✓ Time: {result['optimization_time_ms']}ms")
    
    # Test 6.2: Empty response
    print("\n✓ Test 6.2: Empty response")
    mock_response.choices = []
    try:
        _call_groq_api("System prompt", "User message")
        assert False, "Should raise RuntimeError for empty response"
    except RuntimeError as e:
        assert "empty response" in str(e).lower()
        print(f"  ✓ Correctly raised RuntimeError: {e}")
    finally:
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test"
    
    print("\n✅ All API call tests passed!")


@patch('backend.services.groq_service._call_groq_api')
def test_full_optimization_flow(mock_api_call):
    """Test full optimization flow."""
    print("\n" + "="*60)
    print("TEST 7: Full Optimization Flow")
    print("="*60)
    
    db = SessionLocal()
    try:
        session = create_test_session(db, "test_session_full", with_preferences=True)
        context = "SIMILAR PROMPTS THAT WORKED WELL:\n- Original: 'help me code'\n  Optimized: 'Please help me...'"
        
        # Mock API response
        mock_api_call.return_value = {
            "optimized_prompt": "Please help me write clean, well-documented code for [specific task].",
            "tokens_used": 150,
            "optimization_time_ms": 500
        }
        
        # Test 7.1: Full flow with context and preferences
        print("\n✓ Test 7.1: Full flow with context and preferences")
        result = optimize_with_groq("help me write code", context, session)
        assert "optimized_prompt" in result
        assert "tokens_used" in result
        assert "optimization_time_ms" in result
        assert len(result["optimized_prompt"]) > 0
        print(f"  ✓ Optimization successful: {result['optimized_prompt'][:50]}...")
        print(f"  ✓ Tokens: {result['tokens_used']}, Time: {result['optimization_time_ms']}ms")
        
        # Verify API was called
        assert mock_api_call.called, "API should have been called"
        call_args = mock_api_call.call_args
        system_prompt = call_args[0][0]
        user_message = call_args[0][1]
        assert "You are a prompt optimization expert" in system_prompt
        assert context in system_prompt
        assert "USER PREFERENCES" in system_prompt
        assert "help me write code" in user_message
        print("  ✓ System prompt contains context and preferences")
        print("  ✓ User message contains original prompt")
        
        # Test 7.2: Full flow with empty context
        print("\n✓ Test 7.2: Full flow with empty context")
        result = optimize_with_groq("test prompt", "", session)
        assert "optimized_prompt" in result
        call_args = mock_api_call.call_args
        system_prompt = call_args[0][0]
        assert "CONTEXT FROM SIMILAR SUCCESSFUL OPTIMIZATIONS" not in system_prompt
        assert "USER PREFERENCES" in system_prompt
        print("  ✓ Optimization works without context")
        
        # Test 7.3: Full flow with no preferences
        print("\n✓ Test 7.3: Full flow with no preferences")
        session_no_prefs = create_test_session(db, "test_session_full2", with_preferences=False)
        result = optimize_with_groq("test prompt", context, session_no_prefs)
        assert "optimized_prompt" in result
        call_args = mock_api_call.call_args
        system_prompt = call_args[0][0]
        assert "CONTEXT FROM SIMILAR SUCCESSFUL OPTIMIZATIONS" in system_prompt
        assert "USER PREFERENCES" not in system_prompt
        print("  ✓ Optimization works without preferences")
        
    finally:
        db.close()
    
    print("\n✅ All full flow tests passed!")


def main():
    """Run all tests."""
    print("="*60)
    print("GROQ SERVICE TEST SUITE")
    print("="*60)
    print("\nThis test suite covers all Phase 4 functionality:")
    print("- Client initialization")
    print("- System prompt building")
    print("- User message building")
    print("- Response cleaning")
    print("- Input validation")
    print("- API call (mocked)")
    print("- Full optimization flow")
    
    # Initialize database
    create_tables()
    
    try:
        test_groq_client_initialization()
        test_build_system_prompt()
        test_build_user_message()
        test_clean_optimized_prompt()
        test_input_validation()
        test_call_groq_api()
        test_full_optimization_flow()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

