"""
Quick test script for embedding service.

Run this from project root with venv activated:
    python test_embedding_service.py
"""

import sys
import logging
from backend.app.core.database import SessionLocal, create_tables
from backend.services.embedding_service import generate_embedding, parse_embedding_from_db

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_parse_embedding_from_db():
    """Test JSON parsing function."""
    print("\n" + "="*60)
    print("TEST 1: parse_embedding_from_db()")
    print("="*60)
    
    test_cases = [
        ('[0.1, 0.2, 0.3]', "Valid JSON list"),
        ('[0.123, -0.456, 0.789, 1.234]', "Valid JSON with negatives"),
        ('[]', "Empty list"),
    ]
    
    for json_str, description in test_cases:
        try:
            result = parse_embedding_from_db(json_str)
            print(f"✅ {description}: {len(result)} elements")
            print(f"   First 3: {result[:3] if len(result) >= 3 else result}")
        except Exception as e:
            print(f"❌ {description}: {e}")
    
    error_cases = [
        (None, "None input"),
        ('', "Empty string"),
        ('not json', "Invalid JSON"),
        ('{"not": "a list"}', "Not a list"),
        ('[1, 2, "not a number"]', "Non-numeric value"),
    ]
    
    for json_str, description in error_cases:
        try:
            result = parse_embedding_from_db(json_str)
            print(f"❌ {description}: Should have raised error but got: {result}")
        except ValueError as e:
            print(f"✅ {description}: Correctly raised ValueError: {str(e)[:50]}")


def test_generate_embedding():
    """Test embedding generation (requires API key and database)."""
    print("\n" + "="*60)
    print("TEST 2: generate_embedding()")
    print("="*60)
    
    db = SessionLocal()
    try:
        test_prompt = "help me write code"
        print(f"Testing with prompt: '{test_prompt}'")
        print("This will call OpenAI API (may take a few seconds)...")
        
        embedding = generate_embedding(test_prompt, db)
        
        print(f"✅ Success! Generated embedding:")
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Last 5 values: {embedding[-5:]}")
        
        print("\nTesting cache (should be instant)...")
        embedding2 = generate_embedding(test_prompt, db)
        
        if embedding == embedding2:
            print("✅ Cache working! Same embedding returned instantly.")
        else:
            print("⚠️  Cache returned different embedding (unexpected)")
        
        db.commit()
        
    except ValueError as e:
        print(f"❌ Validation error: {e}")
        print("   Check your .env file has OPENAI_API_KEY set")
    except RuntimeError as e:
        print(f"❌ API error: {e}")
        print("   Check your OpenAI API key is valid")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("EMBEDDING SERVICE TEST SUITE")
    print("="*60)
    
    try:
        create_tables()
        print("✅ Database tables initialized")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
    
    test_parse_embedding_from_db()
    
    print("\n" + "-"*60)
    response = input("\nTest generate_embedding()? This will call OpenAI API (costs money). [y/N]: ")
    if response.lower() == 'y':
        test_generate_embedding()
    else:
        print("Skipping generate_embedding() test")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

