#!/usr/bin/env python3
"""
Test metadata boost directly by executing code in container context
"""
import sys
import os

# Change to /app directory where server.py is located
os.chdir('/app')
sys.path.insert(0, '/app')

print("[INFO] Working directory:", os.getcwd())
print("[INFO] sys.path:", sys.path[:3])
print("[INFO] Files in /app:", [f for f in os.listdir('/app') if f.endswith('.py')][:5])

# Now import should work
try:
    import server
    print("[OK] server module imported successfully")
    
    # Test the function
    query = "технологический стек RAUII"
    print(f"\n[TEST] Query: '{query}'")
    print(f"[TEST] Expected: page_id=18153591 in TOP-3")
    print("-" * 80)
    
    result = server.confluence_semantic_search(
        query=query,
        limit=10,
        space="RAUII"
    )
    
    print("\n[RESULT]")
    # Print first 2000 chars
    print(result[:2000] if len(result) > 2000 else result)
    print("\n" + "-" * 80)
    
    # Check for target page
    if '18153591' in result:
        print("[SUCCESS] Target page_id=18153591 found in results!")
        
        # Check position
        if result.find('18153591') < 500:  # Approximate TOP-3
            print("[SUCCESS] Target appears early (likely TOP-3)")
            sys.exit(0)
        else:
            print("[WARNING] Target appears later in results")
            sys.exit(1)
    else:
        print("[ERROR] Target page NOT found")
        sys.exit(1)
        
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

