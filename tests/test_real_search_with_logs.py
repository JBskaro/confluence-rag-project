#!/usr/bin/env python3
"""
Тест реального поиска с проверкой логов reranking.
"""
import requests
import json
import time
import sys

MCP_URL = "http://localhost:8012/mcp"

def init_session():
    """Инициализирует MCP сессию."""
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }
    
    session = requests.Session()
    response = session.post(
        MCP_URL,
        json=init_payload,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    )
    
    if response.status_code != 200:
        print(f"❌ Ошибка инициализации: {response.status_code}")
        return None, None
    
    # Парсим SSE ответ
    text = response.text
    if "data:" in text:
        for line in text.split("\n"):
            if line.startswith("data:"):
                data = json.loads(line[5:].strip())
                session_id = data.get('result', {}).get('sessionId')
                if session_id:
                    return session, session_id
    
    return None, None

def test_search(session, session_id, query, space=""):
    """Выполняет поиск."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "confluence_semantic_search",
            "arguments": {
                "query": query,
                "limit": 10,
                "space": space
            }
        }
    }
    
    response = session.post(
        MCP_URL,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-Session-Id": session_id
        },
        timeout=120
    )
    
    if response.status_code != 200:
        return None
    
    # Парсим SSE
    result_text = ""
    for line in response.text.split("\n"):
        if line.startswith("data:"):
            try:
                data = json.loads(line[5:].strip())
                if 'result' in data:
                    content = data['result'].get('content', [])
                    if content and len(content) > 0:
                        result_text = content[0].get('text', '')
            except:
                pass
    
    return result_text

def main():
    print("=" * 80)
    print("ТЕСТ РЕАЛЬНОГО ПОИСКА С ПРОВЕРКОЙ ПОРОГОВ")
    print("=" * 80)
    
    # Инициализация
    print("\n1. Инициализация MCP сессии...")
    session, session_id = init_session()
    if not session:
        print("❌ Не удалось инициализировать сессию")
        return 1
    
    print(f"✅ Сессия создана: {session_id[:8]}...")
    
    # Тестовые запросы
    test_queries = [
        ("уточняющие вопросы для обследования по учету номенклатуры на складе", "Surveys"),
        ("как настроить API", "RAUII"),
    ]
    
    print(f"\n2. Тестирование {len(test_queries)} запросов...")
    print("-" * 80)
    
    for i, (query, space) in enumerate(test_queries, 1):
        print(f"\n📋 Запрос {i}: '{query}' (space: {space})")
        print("-" * 80)
        
        start_time = time.time()
        result = test_search(session, session_id, query, space)
        elapsed = time.time() - start_time
        
        if result:
            if "низкой релевантности" in result or "score <" in result:
                print(f"⚠️  Результат: {result[:150]}...")
            else:
                # Подсчитываем количество результатов
                lines = result.split('\n')
                result_count = sum(1 for line in lines if '✅' in line or 'Найдено' in line)
                print(f"✅ Найдено результатов (первые 200 символов):")
                print(f"   {result[:200]}...")
        else:
            print("❌ Ошибка при выполнении поиска")
        
        print(f"⏱  Время: {elapsed:.2f}с")
        time.sleep(2)
    
    print("\n" + "=" * 80)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("=" * 80)
    print("\nПроверьте логи для детальной диагностики:")
    print("  docker-compose logs confluence-rag | grep -i 'rerank\\|threshold\\|scores'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

