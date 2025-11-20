#!/usr/bin/env python3
"""
Тестовый скрипт для проверки metadata boost.
Ожидаем что страница "Стек технологий" (page_id=18153591) будет в TOP-3.
"""
import requests
import json
import sys

def test_metadata_boost():
    """Тестируем metadata boost для запроса 'технологический стек RAUII'"""
    
    url = "http://localhost:8012/search"
    query = "технологический стек RAUII"
    
    print(f"[TEST] Query: '{query}'")
    print(f"[TEST] Expected: page_id=18153591 ('Stek tekhnologiy') in TOP-3")
    print("-" * 80)
    
    try:
        response = requests.post(
            url,
            json={
                "query": query,
                "limit": 10,
                "space": "RAUII"
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"[ERROR] HTTP {response.status_code}")
            print(response.text)
            return False
        
        data = response.json()
        
        if not data.get('results'):
            print("[ERROR] No results!")
            return False
        
        results = data['results']
        print(f"[OK] Got {len(results)} results\n")
        
        # Ищем целевую страницу
        target_found = False
        target_position = -1
        
        for i, result in enumerate(results, 1):
            page_id = result.get('metadata', {}).get('page_id')
            title = result.get('metadata', {}).get('title', 'N/A')
            score = result.get('score', 0)
            rerank_score = result.get('rerank_score', 0)
            metadata_boost = result.get('metadata_boost', 0)
            breadcrumb = result.get('breadcrumb', 'N/A')
            
            is_target = page_id == '18153591'
            marker = "[TARGET] " if is_target else "         "
            
            print(f"{marker}#{i}: {title[:50]}")
            print(f"       page_id={page_id}, score={score:.6f}, rerank_score={rerank_score:.6f}")
            print(f"       metadata_boost={metadata_boost:.6f}")
            print(f"       breadcrumb: {breadcrumb[:60]}")
            print()
            
            if is_target:
                target_found = True
                target_position = i
        
        print("-" * 80)
        
        if target_found:
            if target_position <= 3:
                print(f"[SUCCESS] Page 'Stek tekhnologiy' found at position #{target_position} (TOP-3)")
                return True
            else:
                print(f"[WARNING] Page 'Stek tekhnologiy' found at position #{target_position} (NOT in TOP-3)")
                return False
        else:
            print("[ERROR] Page 'Stek tekhnologiy' NOT found in results!")
            return False
            
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_metadata_boost()
    sys.exit(0 if success else 1)

