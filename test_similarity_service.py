"""
Comprehensive test script for similarity service.

Run this from project root with venv activated:
    python test_similarity_service.py

This script tests all Phase 2 functionality:
- FAISS index creation and management
- Vector normalization
- Database embedding extraction
- Index building
- Similarity search
- Quality filtering
- Cache integration
- Index updates
"""

import sys
import logging
import json
import numpy as np
from backend.app.core.database import SessionLocal, create_tables
from backend.models.prompt import Prompt, Session as SessionModel
from backend.services.similarity_service import (
    _create_empty_index,
    _normalize_vector,
    _extract_embeddings_from_db,
    build_faiss_index,
    get_or_build_index,
    _search_index,
    _map_indices_to_prompts,
    _filter_by_quality,
    update_index_with_new_prompt,
    find_similar_prompts,
)
from backend.services.embedding_service import generate_embedding

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_data(db):
    """Create test prompts with embeddings in database."""
    print("\nCreating test data in database...")
    
    # Create a test session
    session = SessionModel(id="test_session_123")
    db.add(session)
    db.flush()
    
    # Create test prompts with embeddings
    test_prompts = [
        ("help me write code", "Please help me write clean, well-documented code...", 0.85),
        ("write a program", "Please write a well-structured program...", 0.82),
        ("coding assistance", "I need assistance with coding...", 0.75),
        ("how to code", "Can you explain how to code...", 0.60),  # Low quality
        ("programming help", "I need help with programming...", None),  # No quality score
    ]
    
    created_prompts = []
    for original, optimized, quality_score in test_prompts:
        try:
            # Generate embedding for the prompt
            embedding = generate_embedding(original, db)
            
            # Create prompt record
            prompt = Prompt(
                session_id="test_session_123",
                original_prompt=original,
                optimized_prompt=optimized,
                embedding=json.dumps(embedding),
                chatgpt_quality_score=quality_score,
            )
            db.add(prompt)
            db.flush()
            created_prompts.append(prompt)
            print(f"  ✅ Created prompt: '{original[:30]}...' (ID: {prompt.id}, Quality: {quality_score})")
        except Exception as e:
            print(f"  ⚠️  Failed to create prompt '{original}': {e}")
    
    db.commit()
    print(f"Created {len(created_prompts)} test prompts\n")
    return created_prompts


def test_create_empty_index():
    """Test FAISS index creation."""
    print("\n" + "="*60)
    print("TEST 1: _create_empty_index()")
    print("="*60)
    
    try:
        index = _create_empty_index()
        print(f"✅ Successfully created empty FAISS index")
        print(f"   Dimension: {index.d}")
        print(f"   Total vectors: {index.ntotal}")
        print(f"   Index type: {type(index).__name__}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_normalize_vector():
    """Test vector normalization."""
    print("\n" + "="*60)
    print("TEST 2: _normalize_vector()")
    print("="*60)
    
    # Test single vector (1D)
    try:
        vec1d = np.array([3.0, 4.0], dtype=np.float32)
        normalized = _normalize_vector(vec1d)
        norm = np.linalg.norm(normalized)
        print(f"✅ Single vector normalization:")
        print(f"   Input: {vec1d}")
        print(f"   Normalized: {normalized}")
        print(f"   Norm: {norm:.6f} (should be ~1.0)")
    except Exception as e:
        print(f"❌ Single vector failed: {e}")
    
    # Test batch (2D)
    try:
        vec2d = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        normalized = _normalize_vector(vec2d)
        norms = np.linalg.norm(normalized, axis=1)
        print(f"✅ Batch normalization:")
        print(f"   Input shape: {vec2d.shape}")
        print(f"   Normalized shape: {normalized.shape}")
        print(f"   Norms: {norms} (should be ~[1.0, 1.0])")
    except Exception as e:
        print(f"❌ Batch normalization failed: {e}")
    
    # Test zero vector
    try:
        zero_vec = np.array([0.0, 0.0], dtype=np.float32)
        normalized = _normalize_vector(zero_vec)
        print(f"✅ Zero vector handling:")
        print(f"   Result: {normalized}")
        print(f"   (Should handle without error)")
    except Exception as e:
        print(f"❌ Zero vector failed: {e}")


def test_extract_embeddings_from_db(db):
    """Test database embedding extraction."""
    print("\n" + "="*60)
    print("TEST 3: _extract_embeddings_from_db()")
    print("="*60)
    
    try:
        prompt_ids, embeddings_array = _extract_embeddings_from_db(db)
        print(f"✅ Successfully extracted embeddings:")
        print(f"   Number of prompts: {len(prompt_ids)}")
        print(f"   Embeddings shape: {embeddings_array.shape}")
        print(f"   First 3 prompt IDs: {prompt_ids[:3]}")
        print(f"   Embedding dtype: {embeddings_array.dtype}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_build_faiss_index(db):
    """Test FAISS index building."""
    print("\n" + "="*60)
    print("TEST 4: build_faiss_index()")
    print("="*60)
    
    try:
        index, mapping = build_faiss_index(db)
        print(f"✅ Successfully built FAISS index:")
        print(f"   Total vectors: {index.ntotal}")
        print(f"   Dimension: {index.d}")
        print(f"   Mapping size: {len(mapping)}")
        print(f"   Sample mapping: {dict(list(mapping.items())[:3])}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_get_or_build_index(db):
    """Test index caching."""
    print("\n" + "="*60)
    print("TEST 5: get_or_build_index() - Caching")
    print("="*60)
    
    try:
        # First call - should build
        print("First call (should build index)...")
        index1, mapping1 = get_or_build_index(db)
        print(f"✅ First call: {index1.ntotal} vectors")
        
        # Second call - should use cache
        print("Second call (should use cache)...")
        index2, mapping2 = get_or_build_index(db)
        print(f"✅ Second call: {index2.ntotal} vectors")
        
        if index1 is index2:
            print("✅ Cache working! Same index object returned.")
        else:
            print("⚠️  Different index objects (may still be correct)")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_search_index(db):
    """Test FAISS search."""
    print("\n" + "="*60)
    print("TEST 6: _search_index()")
    print("="*60)
    
    try:
        index, mapping = get_or_build_index(db)
        
        if index.ntotal == 0:
            print("⚠️  Index is empty, skipping search test")
            return
        
        # Get a query embedding (use first prompt's embedding)
        prompt_ids, embeddings = _extract_embeddings_from_db(db)
        if len(embeddings) == 0:
            print("⚠️  No embeddings found, skipping search test")
            return
        
        query_embedding = embeddings[0]  # Use first embedding as query
        
        # Search for top 3
        distances, indices = _search_index(query_embedding, index, k=3)
        
        print(f"✅ Search successful:")
        print(f"   Query embedding shape: {query_embedding.shape}")
        print(f"   Results: {len([i for i in indices if i != -1])} found")
        print(f"   Distances: {distances[:3]}")
        print(f"   Indices: {indices[:3]}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_map_indices_to_prompts(db):
    """Test mapping indices to prompts."""
    print("\n" + "="*60)
    print("TEST 7: _map_indices_to_prompts()")
    print("="*60)
    
    try:
        index, mapping = get_or_build_index(db)
        
        if index.ntotal == 0:
            print("⚠️  Index is empty, skipping mapping test")
            return
        
        # Get a query embedding
        prompt_ids, embeddings = _extract_embeddings_from_db(db)
        if len(embeddings) == 0:
            print("⚠️  No embeddings found, skipping mapping test")
            return
        
        query_embedding = embeddings[0]
        distances, indices = _search_index(query_embedding, index, k=3)
        
        # Map to prompts
        prompts = _map_indices_to_prompts(indices, mapping, db)
        
        print(f"✅ Mapping successful:")
        print(f"   Mapped {len(prompts)} prompts")
        for i, prompt in enumerate(prompts[:3]):
            print(f"   {i+1}. ID: {prompt.id}, Original: '{prompt.original_prompt[:40]}...'")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_filter_by_quality(db):
    """Test quality filtering."""
    print("\n" + "="*60)
    print("TEST 8: _filter_by_quality()")
    print("="*60)
    
    try:
        # Get all prompts
        prompts = db.query(Prompt).filter(Prompt.session_id == "test_session_123").all()
        
        print(f"Testing with {len(prompts)} prompts:")
        for p in prompts:
            print(f"   - ID {p.id}: quality={p.chatgpt_quality_score}")
        
        # Filter by quality (min 0.7)
        filtered = _filter_by_quality(prompts, min_quality=0.7)
        
        print(f"✅ Filtering successful:")
        print(f"   Original: {len(prompts)} prompts")
        print(f"   Filtered: {len(filtered)} prompts (quality >= 0.7)")
        for p in filtered:
            print(f"   - ID {p.id}: quality={p.chatgpt_quality_score}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_update_index_with_new_prompt(db):
    """Test index updates."""
    print("\n" + "="*60)
    print("TEST 9: update_index_with_new_prompt()")
    print("="*60)
    
    try:
        index, mapping = get_or_build_index(db)
        original_count = index.ntotal
        
        # Create a new embedding
        new_embedding = generate_embedding("new test prompt", db)
        
        # Create a new prompt
        new_prompt = Prompt(
            session_id="test_session_123",
            original_prompt="new test prompt",
            optimized_prompt="New optimized prompt",
            embedding=json.dumps(new_embedding),
            chatgpt_quality_score=0.9,
        )
        db.add(new_prompt)
        db.flush()
        
        # Update index
        update_index_with_new_prompt(new_embedding, new_prompt.id, index, mapping)
        
        print(f"✅ Index update successful:")
        print(f"   Original count: {original_count}")
        print(f"   New count: {index.ntotal}")
        print(f"   New prompt ID: {new_prompt.id}")
        print(f"   Mapping updated: {new_prompt.id in mapping.values()}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_find_similar_prompts(db):
    """Test main public function."""
    print("\n" + "="*60)
    print("TEST 10: find_similar_prompts() - Full Flow")
    print("="*60)
    
    try:
        # Generate a query embedding
        query_text = "I need help with programming"
        query_embedding = generate_embedding(query_text, db)
        
        print(f"Query: '{query_text}'")
        print("Searching for similar prompts...")
        
        # Find similar prompts
        similar_prompts = find_similar_prompts(
            embedding=query_embedding,
            session_id="test_session_123",
            limit=3,
            db=db
        )
        
        print(f"✅ Found {len(similar_prompts)} similar prompts:")
        for i, prompt in enumerate(similar_prompts, 1):
            print(f"   {i}. ID: {prompt.id}")
            print(f"      Original: '{prompt.original_prompt}'")
            print(f"      Quality: {prompt.chatgpt_quality_score}")
        
        # Test cache (second call should be faster)
        print("\nTesting cache (second call)...")
        similar_prompts2 = find_similar_prompts(
            embedding=query_embedding,
            session_id="test_session_123",
            limit=3,
            db=db
        )
        
        if len(similar_prompts) == len(similar_prompts2):
            print(f"✅ Cache working! Same {len(similar_prompts2)} results returned.")
        else:
            print(f"⚠️  Different number of results: {len(similar_prompts)} vs {len(similar_prompts2)}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_error_handling():
    """Test error handling."""
    print("\n" + "="*60)
    print("TEST 11: Error Handling")
    print("="*60)
    
    # Test invalid embedding
    try:
        find_similar_prompts(None, "test", 5, SessionLocal())
        print("❌ Should have raised ValueError for None embedding")
    except ValueError:
        print("✅ Correctly raised ValueError for None embedding")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")
    
    # Test invalid limit
    try:
        find_similar_prompts([0.1] * 1536, "test", -1, SessionLocal())
        print("❌ Should have raised ValueError for invalid limit")
    except ValueError:
        print("✅ Correctly raised ValueError for invalid limit")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")


def cleanup_test_data(db):
    """Clean up test data."""
    print("\nCleaning up test data...")
    try:
        db.query(Prompt).filter(Prompt.session_id == "test_session_123").delete()
        db.query(SessionModel).filter(SessionModel.id == "test_session_123").delete()
        db.commit()
        print("✅ Test data cleaned up")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
        db.rollback()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SIMILARITY SERVICE TEST SUITE")
    print("="*60)
    
    # Initialize database
    try:
        create_tables()
        print("✅ Database tables initialized")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
    
    db = SessionLocal()
    
    try:
        # Create test data
        test_prompts = create_test_data(db)
        
        if len(test_prompts) == 0:
            print("⚠️  No test prompts created. Some tests may be skipped.")
            print("   Make sure OPENAI_API_KEY is set in .env file")
        
        # Run tests
        test_create_empty_index()
        test_normalize_vector()
        
        if len(test_prompts) > 0:
            test_extract_embeddings_from_db(db)
            test_build_faiss_index(db)
            test_get_or_build_index(db)
            test_search_index(db)
            test_map_indices_to_prompts(db)
            test_filter_by_quality(db)
            test_update_index_with_new_prompt(db)
            test_find_similar_prompts(db)
        else:
            print("\n⚠️  Skipping database-dependent tests (no test data)")
        
        test_error_handling()
        
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cleanup_test_data(db)
        db.close()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

