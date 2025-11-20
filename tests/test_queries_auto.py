#!/usr/bin/env python3
"""
Автоматическое тестирование запросов через HTTP MCP
"""
import requests
import json
import sys

MCP_URL = "http://localhost:8012/mcp"

def init_mcp_session():
    """Инициализация MCP сессии через requests.Session для сохранения cookies"""
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    })
    
    payload = {
        "jsonrpc": "2.0",
        "id": "init-1",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }
    
    try:
        response = session.post(MCP_URL, json=payload, timeout=10)
        
        # Проверяем разные форматы ответа
        text = response.text.strip()
        
        if response.status_code == 200:
            # FastMCP может вернуть пустой ответ или SSE формат
            if text.startswith("data: "):
                json_str = text.replace("data: ", "").strip()
                try:
                    result = json.loads(json_str)
                    session_id = result.get("result", {}).get("sessionId")
                    if session_id:
                        print(f"[OK] Сессия инициализирована: {session_id[:20]}...")
                except:
                    pass
            elif text:
                try:
                    result = response.json()
                    session_id = result.get("result", {}).get("sessionId")
                    if session_id:
                        print(f"[OK] Сессия инициализирована: {session_id[:20]}...")
                except:
                    pass
            
            # Важно: cookies могут быть установлены даже при пустом ответе
            if session.cookies:
                print(f"[OK] Cookies установлены: {len(session.cookies)} cookie(s)")
            else:
                print(f"[INFO] Cookies не установлены, но продолжаем")
            
            return session
        else:
            print(f"[WARN] Initialize вернул {response.status_code}: {text[:200]}")
            # Продолжаем с сессией - возможно, cookies всё равно установлены
            return session
    except json.JSONDecodeError:
        # Пустой ответ или не JSON - это нормально для initialize
        print(f"[OK] Initialize выполнен (ответ: {len(text)} символов)")
        return session
    except Exception as e:
        print(f"[WARN] Ошибка инициализации: {e}, продолжаем с сессией")
        return session

def test_query(name, query, expected_page, limit=20, space="", session=None):
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
            print(f"[FAILED] HTTP {response.status_code}")
            print(f"Ответ сервера: {response.text[:500]}")
            return False
        
        text = response.text
        if text.startswith("data: "):
            json_str = text.replace("data: ", "").strip()
            result = json.loads(json_str)
        else:
            result = response.json()
        
        content = result.get("result", {}).get("content", [])
        if not content:
            print("[FAILED] Пустой ответ")
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
        return False

def main():
    print("\n" + "="*80)
    print("АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ")
    print("="*80)
    
    # Инициализация сессии
    print("\nИнициализация MCP сессии...")
    session = init_mcp_session()
    
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

