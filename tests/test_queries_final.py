#!/usr/bin/env python3
"""
Автоматическое тестирование запросов через HTTP MCP
Упрощенная версия с правильной обработкой сессий
"""
import requests
import json
import sys

MCP_URL = "http://localhost:8012/mcp"

def test_query(name, query, expected_page, limit=20, space="", session=None):
    """Тестирование одного запроса"""
    print(f"\n{'='*80}")
    print(f"ТЕСТ: {name}")
    print(f"{'='*80}")
    print(f"Запрос: {query[:80]}...")
    print(f"Ожидаем page_id: {expected_page}")
    print("-" * 80)
    
    payload = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "tools/call",
        "params": {
            "name": "confluence_semantic_search",
            "arguments": {
                "query": query,
                "limit": limit,
                "space": space
            }
        }
    }
    
    # Используем переданную сессию или создаём новую
    if session is None:
        session = requests.Session()
    
    # Убеждаемся, что заголовки установлены правильно
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    })
    
    try:
        response = session.post(
            MCP_URL,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            error_text = response.text[:500]
            print(f"[FAILED] HTTP {response.status_code}")
            print(f"Ответ: {error_text}")
            
            # Если ошибка про session ID, пробуем инициализировать
            if "session ID" in error_text or "sessionId" in error_text:
                print("[INFO] Пробуем инициализировать сессию...")
                init_response = session.post(
                    MCP_URL,
                    json={
                        "jsonrpc": "2.0",
                        "id": "init",
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1.0"}
                        }
                    },
                    timeout=10
                )
                print(f"[INFO] Initialize: {init_response.status_code}")
                
                # Повторяем запрос после инициализации
                response = session.post(MCP_URL, json=payload, timeout=60)
                if response.status_code != 200:
                    print(f"[FAILED] После инициализации: HTTP {response.status_code}")
                    return False
            
            if response.status_code != 200:
                return False
        
        # Обрабатываем ответ
        text = response.text.strip()
        
        # SSE формат
        if text.startswith("data: "):
            json_str = text.replace("data: ", "").strip()
            result = json.loads(json_str)
        else:
            result = response.json()
        
        content = result.get("result", {}).get("content", [])
        if not content:
            print("[FAILED] Пустой ответ")
            print(f"Полный ответ: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
            return False
        
        answer_text = content[0].get("text", "")
        
        if expected_page in answer_text:
            print(f"[PASSED] Найдена страница {expected_page}")
            print(f"Ответ ({len(answer_text)} символов):\n{answer_text[:500]}...")
            return True
        else:
            print(f"[FAILED] Страница {expected_page} не найдена")
            print(f"Ответ ({len(answer_text)} символов):\n{answer_text[:500]}...")
            return False
            
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*80)
    print("АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ")
    print("="*80)
    
    # Создаём сессию для сохранения cookies
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    })
    
    # Пробуем инициализировать сессию
    print("\nИнициализация MCP сессии...")
    try:
        init_response = session.post(
            MCP_URL,
            json={
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"}
                }
            },
            timeout=10
        )
        print(f"[INFO] Initialize статус: {init_response.status_code}")
        if session.cookies:
            print(f"[OK] Cookies установлены: {list(session.cookies.keys())}")
    except Exception as e:
        print(f"[WARN] Ошибка инициализации: {e}, продолжаем")
    
    results = []
    
    # Тест 1
    results.append(test_query(
        "Учет номенклатуры",
        "Провожу обследование компании по блоку Склад, а точнее Учет номенклатуры. Подготовь список вопросов.",
        "18153754",
        limit=20,
        session=session
    ))
    
    # Тест 2
    results.append(test_query(
        "Технологический стек",
        "Ккой тхнологиеский стек исользуется в проект рау ии.",
        "18153591",
        limit=10,
        space="RAUII",
        session=session
    ))
    
    # Итоги
    print(f"\n{'='*80}")
    print(f"ИТОГИ: {sum(results)}/{len(results)} тестов прошли")
    print("="*80)
    
    return 0 if sum(results) == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())

