#!/usr/bin/env python3
"""
Комплексный тест pipeline reranking для диагностики проблемы с порогами.
Проверяет весь путь от поиска до фильтрации.
"""
import sys
import os
import requests
import json
import time

# Добавляем путь к модулям
sys.path.insert(0, '/app/rag_server')

def init_mcp_session():
    """Инициализирует MCP сессию и возвращает session_id."""
    url = "http://localhost:8012/mcp"
    
    # Инициализация
    init_payload = {
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
    
    session = requests.Session()
    response = session.post(
        url,
        json=init_payload,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    )
    
    if response.status_code != 200:
        print(f"❌ Ошибка инициализации: {response.status_code}")
        print(f"Response text: {response.text[:500]}")
        return None, None
    
    try:
        result = response.json()
    except Exception as e:
        print(f"❌ Ошибка парсинга JSON: {e}")
        print(f"Response text: {response.text[:500]}")
        return None, None
    
    session_id = result.get('result', {}).get('sessionId')
    
    if not session_id:
        print("❌ Не получен session_id")
        return None, None
    
    print(f"✅ MCP сессия инициализирована: {session_id[:8]}...")
    return session, session_id

def test_search(session, session_id, query, space=""):
    """Выполняет поиск и возвращает результаты."""
    url = "http://localhost:8012/mcp"
    
    call_payload = {
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
        url,
        json=call_payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-Session-Id": session_id
        }
    )
    
    if response.status_code != 200:
        print(f"❌ Ошибка поиска: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    return result.get('result', {}).get('content', [{}])[0].get('text', '')

def main():
    """Основная функция тестирования."""
    print("=" * 80)
    print("ТЕСТ PIPELINE RERANKING")
    print("=" * 80)
    
    # Проверка переменных окружения
    print("\n1. ПРОВЕРКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ:")
    print("-" * 80)
    rerank_technical = os.getenv('RERANK_THRESHOLD_TECHNICAL', 'НЕ УСТАНОВЛЕНО')
    rerank_general = os.getenv('RERANK_THRESHOLD_GENERAL', 'НЕ УСТАНОВЛЕНО')
    rerank_model = os.getenv('RE_RANKER_MODEL', 'НЕ УСТАНОВЛЕНО')
    
    print(f"  RERANK_THRESHOLD_TECHNICAL: {rerank_technical}")
    print(f"  RERANK_THRESHOLD_GENERAL: {rerank_general}")
    print(f"  RE_RANKER_MODEL: {rerank_model}")
    
    # Инициализация сессии
    print("\n2. ИНИЦИАЛИЗАЦИЯ MCP СЕССИИ:")
    print("-" * 80)
    session, session_id = init_mcp_session()
    if not session:
        print("❌ Не удалось инициализировать сессию")
        return 1
    
    # Тестовые запросы
    test_queries = [
        ("уточняющие вопросы для обследования по учету номенклатуры на складе", "Surveys"),
        ("как настроить API", "RAUII"),
        ("процессы складского учета", "Surveys"),
    ]
    
    print("\n3. ТЕСТИРОВАНИЕ ПОИСКА:")
    print("-" * 80)
    
    for i, (query, space) in enumerate(test_queries, 1):
        print(f"\nТест {i}: '{query}' (space: {space})")
        print("-" * 80)
        
        start_time = time.time()
        result = test_search(session, session_id, query, space)
        elapsed = time.time() - start_time
        
        if result:
            # Проверяем наличие предупреждения о низкой релевантности
            if "низкой релевантности" in result or "score <" in result:
                print(f"⚠️  Результат: {result[:200]}...")
            else:
                print(f"✅ Найдено результатов (первые 200 символов):")
                print(f"   {result[:200]}...")
        else:
            print("❌ Ошибка при выполнении поиска")
        
        print(f"⏱  Время выполнения: {elapsed:.2f}с")
        
        # Небольшая пауза между запросами
        time.sleep(2)
    
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    print("\nПроверьте логи контейнера для детальной диагностики:")
    print("  docker-compose logs -f confluence-rag | grep -E 'RERANKER|Scores|порог'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

