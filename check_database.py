#!/usr/bin/env python3
"""
Quick script to check database contents
Run: python check_database.py
"""

import sqlite3
import json
from pathlib import Path

# Database path
db_path = Path("prompt_optimizer.db")

if not db_path.exists():
    print("❌ Database file not found!")
    print(f"Expected location: {db_path.absolute()}")
    exit(1)

print(f"✅ Database found: {db_path.absolute()}\n")

# Connect to database
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Check prompts table
print("=" * 80)
print("PROMPTS TABLE")
print("=" * 80)

cursor.execute("SELECT COUNT(*) FROM prompts")
prompt_count = cursor.fetchone()[0]
print(f"Total prompts: {prompt_count}\n")

if prompt_count > 0:
    # Get all prompts
    cursor.execute("""
        SELECT id, session_id, original_prompt, optimized_prompt, 
               created_at, context_prompts, chatgpt_quality_score
        FROM prompts 
        ORDER BY created_at DESC 
        LIMIT 10
    """)
    
    prompts = cursor.fetchall()
    
    for i, (pid, sid, original, optimized, created, context_prompts, quality) in enumerate(prompts, 1):
        print(f"\n--- Prompt #{i} (ID: {pid}) ---")
        print(f"Session: {sid}")
        print(f"Created: {created}")
        print(f"Quality Score: {quality if quality else 'N/A'}")
        print(f"\nOriginal:")
        print(f"  {original[:100]}{'...' if len(original) > 100 else ''}")
        print(f"\nOptimized:")
        if optimized:
            print(f"  {optimized[:100]}{'...' if len(optimized) > 100 else ''}")
        else:
            print("  (None)")
        
        # Parse context_prompts
        if context_prompts:
            try:
                context_ids = json.loads(context_prompts)
                print(f"\nContext Prompts Used: {len(context_ids)} similar prompts")
                print(f"  IDs: {context_ids}")
            except:
                print(f"\nContext Prompts: {context_prompts}")
        else:
            print(f"\nContext Prompts Used: 0 (no similar prompts found)")
else:
    print("No prompts found in database.")

# Check sessions table
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
        
        # Count prompts for this session
        cursor.execute("SELECT COUNT(*) FROM prompts WHERE session_id = ?", (sid,))
        prompt_count = cursor.fetchone()[0]
        print(f"Prompts in this session: {prompt_count}")
else:
    print("No sessions found in database.")

# Check for embeddings
print("\n" + "=" * 80)
print("EMBEDDINGS")
print("=" * 80)

cursor.execute("SELECT COUNT(*) FROM prompts WHERE embedding IS NOT NULL")
embedding_count = cursor.fetchone()[0]
print(f"Prompts with embeddings: {embedding_count} / {prompt_count}")

conn.close()
print("\n" + "=" * 80)
print("✅ Database check complete!")

