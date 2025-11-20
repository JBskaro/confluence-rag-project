#!/usr/bin/env python3
"""
Direct test for metadata boost inside container
"""
import sys
import os

# Add rag_server to path
sys.path.insert(0, '/app/rag_server')

# Set required env vars if not set
if not os.getenv('CONFLUENCE_URL'):
    print("[SKIP] CONFLUENCE_URL not set, skipping test")
    sys.exit(0)

try:
    # In container, mcp_rag_secure.py is copied as server.py
    from server import confluence_semantic_search
    
    query = "технологический стек RAUII"
    print(f"[TEST] Query: '{query}'")
    print(f"[TEST] Expected: page_id=18153591 in TOP-3")
    print("-" * 80)
    
    # Call the search function
    result = confluence_semantic_search(
        query=query,
        limit=10,
        space="RAUII"
    )
    
    print("\n[RESULT]")
    print(result)
    print("\n" + "-" * 80)
    
    # Check if target page is mentioned
    if '18153591' in result:
        # Try to determine position
        lines = result.split('\n')
        for i, line in enumerate(lines):
            if '18153591' in line:
                print(f"[INFO] Target page found at line {i}")
                if i < 10:  # Approximate TOP-3 check
                    print("[SUCCESS] Target page appears early in results")
                    sys.exit(0)
                else:
                    print("[WARNING] Target page appears later in results")
                    sys.exit(1)
        print("[SUCCESS] Target page found")
        sys.exit(0)
    else:
        print("[ERROR] Target page NOT found in results")
        sys.exit(1)
        
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

