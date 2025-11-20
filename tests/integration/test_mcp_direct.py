#!/usr/bin/env python3
"""
Прямой тест MCP сервера
"""
import requests
import json

url = "http://localhost:8012/mcp"

# Создаем сессию
session_req = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    }
}

print("[TEST] Testing MCP server...")
print(f"URL: {url}\n")

try:
    # Initialize
    print("1. Initializing session...")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream'
    }
    response = requests.post(url, json=session_req, headers=headers, timeout=10)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        print("   [OK] Session initialized\n")
        
        # Test search
        print("2. Testing search...")
        search_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "confluence_semantic_search",
                "arguments": {
                    "query": "технологический стек RAUII",
                    "limit": 5,
                    "space": "RAUII"
                }
            }
        }
        
        response = requests.post(url, json=search_req, headers=headers, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'result' in result:
                content = result['result'].get('content', [])
                if content:
                    text = content[0].get('text', '')
                    
                    # Check for success
                    if "Search Error" in text or "Error:" in text:
                        print("   [ERROR] Search failed!")
                        print(f"   Error: {text[:200]}")
                    else:
                        print("   [SUCCESS] Search successful!")
                        # Count results
                        result_count = text.count("Результат ") + text.count("Result ")
                        print(f"   Found {result_count} results")
                        
                        # Check for target page
                        if "18153591" in text:
                            print("   [TARGET] Target page '18153591' found!")
                        else:
                            print("   [WARNING] Target page '18153591' NOT found")
                        
                        # Show first 300 chars
                        print(f"\n   Preview:")
                        print(f"   {text[:300]}...")
                else:
                    print("   [ERROR] Empty response")
            else:
                print(f"   [ERROR] Unexpected response: {result}")
        else:
            print(f"   [ERROR] HTTP {response.status_code}")
            print(f"   {response.text[:200]}")
    else:
        print(f"   [ERROR] Failed to initialize: {response.text[:200]}")
        
except Exception as e:
    print(f"[ERROR] Exception: {e}")
    import traceback
    traceback.print_exc()

