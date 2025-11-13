#!/usr/bin/env python3
"""
Quick script to check database contents
Run: python check_database.py
"""

import sqlite3
import json
from pathlib import Path

db_path = Path("prompt_optimizer.db")

if not db_path.exists():
    print("❌ Database file not found!")
    print(f"Expected location: {db_path.absolute()}")
    exit(1)

print(f"✅ Database found: {db_path.absolute()}\n")

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print("=" * 80)
print("PROMPTS TABLE")
print("=" * 80)

cursor.execute("SELECT COUNT(*) FROM prompts")
prompt_count = cursor.fetchone()[0]
print(f"Total prompts: {prompt_count}\n")

if prompt_count > 0:
    cursor.execute("""
        SELECT id, session_id, original_prompt, optimized_prompt, 
               created_at, context_prompts, chatgpt_quality_score,
               optimization_method, optimization_time_ms, tokens_used,
               embedding IS NOT NULL as has_embedding
        FROM prompts 
        ORDER BY created_at DESC 
        LIMIT 10
    """)
    
    prompts = cursor.fetchall()
    
    for i, (pid, sid, original, optimized, created, context_prompts, quality, 
            opt_method, opt_time, tokens, has_embedding) in enumerate(prompts, 1):
        print(f"\n--- Prompt #{i} (ID: {pid}) ---")
        print(f"Session: {sid}")
        print(f"Created: {created}")
        print(f"Method: {opt_method or 'N/A'}")
        if opt_time:
            print(f"Optimization Time: {opt_time}ms")
        if tokens:
            print(f"Tokens Used: {tokens}")
        print(f"Has Embedding: {'✅' if has_embedding else '❌'}")
        print(f"\nOriginal ({len(original)} chars):")
        print(f"  {original[:150]}{'...' if len(original) > 150 else ''}")
        print(f"\nOptimized ({len(optimized) if optimized else 0} chars):")
        if optimized:
            print(f"  {optimized[:150]}{'...' if len(optimized) > 150 else ''}")
        else:
            print("  (None)")
        
        if context_prompts:
            try:
                context_ids = json.loads(context_prompts)
                print(f"\nContext: {len(context_ids)} similar prompt(s) used")
                if len(context_ids) > 0:
                    print(f"  Similar prompt IDs: {context_ids}")
            except:
                print(f"\nContext: {context_prompts}")
        else:
            print(f"\nContext: 0 similar prompts (new/unique prompt)")
else:
    print("No prompts found in database.")

print("\n" + "=" * 80)
print("SESSIONS TABLE")
print("=" * 80)

cursor.execute("SELECT COUNT(*) FROM sessions")
session_count = cursor.fetchone()[0]
print(f"Total sessions: {session_count}\n")

if session_count > 0:
    cursor.execute("""
        SELECT id, preferred_style, common_feedback_patterns, created_at
        FROM sessions 
        ORDER BY created_at DESC 
        LIMIT 5
    """)
    
    sessions = cursor.fetchall()
    
    for i, (sid, style, patterns, created) in enumerate(sessions, 1):
        print(f"\n--- Session #{i} (ID: {sid}) ---")
        print(f"Created: {created}")
        print(f"Preferred Style: {style if style else 'None'}")
        print(f"Feedback Patterns: {patterns if patterns else 'None'}")
        
        cursor.execute("SELECT COUNT(*) FROM prompts WHERE session_id = ?", (sid,))
        prompt_count = cursor.fetchone()[0]
        print(f"Prompts in this session: {prompt_count}")
else:
    print("No sessions found in database.")

print("\n" + "=" * 80)
print("EMBEDDINGS & STATISTICS")
print("=" * 80)

cursor.execute("SELECT COUNT(*) FROM prompts WHERE embedding IS NOT NULL")
embedding_count = cursor.fetchone()[0]
print(f"Prompts with embeddings: {embedding_count} / {prompt_count}")

cursor.execute("SELECT COUNT(*) FROM cache_entries")
cache_count = cursor.fetchone()[0]
print(f"Cache entries: {cache_count}")

if cache_count > 0:
    cursor.execute("""
        SELECT cache_type, COUNT(*) as count, SUM(hit_count) as total_hits
        FROM cache_entries 
        GROUP BY cache_type
    """)
    cache_stats = cursor.fetchall()
    print("\nCache breakdown:")
    for cache_type, count, hits in cache_stats:
        print(f"  {cache_type or 'unknown'}: {count} entries, {hits or 0} total hits")

cursor.execute("""
    SELECT optimization_method, COUNT(*) as count
    FROM prompts 
    WHERE optimization_method IS NOT NULL
    GROUP BY optimization_method
""")
method_stats = cursor.fetchall()
if method_stats:
    print("\nOptimization methods:")
    for method, count in method_stats:
        print(f"  {method}: {count} prompt(s)")

conn.close()
print("\n" + "=" * 80)
print("✅ Database check complete!")

