#!/usr/bin/env python3
"""
Тестирование реальных запросов пользователя
"""
import json
import requests
import sys
from typing import Dict, Any

MCP_URL = "http://localhost:8012/mcp"

def call_mcp_tool(query: str, limit: int = 20, space: str = "") -> Dict[str, Any]:
    """Вызов MCP инструмента confluence_semantic_search"""
    
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
    
    try:
        response = requests.post(
            MCP_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            timeout=60
        )
        
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        
        # MCP может возвращать SSE формат
        text = response.text
        if text.startswith("data: "):
            # SSE формат
            json_str = text.replace("data: ", "").strip()
            return json.loads(json_str)
        else:
            # Обычный JSON
            return response.json()
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "raw_response": response.text[:500]}

def test_query_1():
    """Тест 1: Учет номенклатуры"""
    print("=" * 80)
    print("ТЕСТ 1: Учет номенклатуры")
    print("=" * 80)
    print("\nЗапрос:")
    query = "Провожу обследование компании по блоку Склад, а точнее Учет номенклатуры. Подготовь список вопросов."
    print(f"   {query}")
    print("\nОжидаемый результат:")
    print("   - Страница: 18153754 (Учет номенклатуры)")
    print("   - ~30+ вопросов из всех разделов")
    print("\n" + "-" * 80)
    
    result = call_mcp_tool(query, limit=20)
    
    print("\nРезультат:")
    if "error" in result:
        print(f"   [ERROR] Ошибка: {result['error']}")
        return False
    
    if "result" in result:
        content = result["result"].get("content", [])
        if content and len(content) > 0:
            text = content[0].get("text", "")
            print(f"   ✅ Получен ответ (длина: {len(text)} символов)")
            print("\n   Первые 2000 символов ответа:")
            print("   " + "-" * 76)
            print("   " + text[:2000].replace("\n", "\n   "))
            if len(text) > 2000:
                print(f"\n   ... (еще {len(text) - 2000} символов)")
            
            # Проверяем наличие ключевых слов
            keywords = ["номенклатур", "классификац", "серии", "характеристик", "габарит", "штрихкод"]
            found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
            print(f"\n   Найдено ключевых слов: {len(found_keywords)}/{len(keywords)}")
            if found_keywords:
                print(f"      {', '.join(found_keywords)}")
            
            # Проверяем наличие page_id 18153754
            if "18153754" in text:
                print("\n   [OK] Найдена ожидаемая страница 18153754")
            else:
                print("\n   [WARN] Страница 18153754 не найдена в ответе")
            
            return True
        else:
            print("   [ERROR] Пустой ответ")
            return False
    else:
        print(f"   [ERROR] Неожиданный формат ответа: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return False

def test_query_2():
    """Тест 2: Технологический стек (с опечатками)"""
    print("\n" + "=" * 80)
    print("ТЕСТ 2: Технологический стек (с опечатками)")
    print("=" * 80)
    print("\nЗапрос:")
    query = "Ккой тхнологиеский стек исользуется в проект рау ии."
    print(f"   {query}")
    print("\nОжидаемый результат:")
    print("   - Query Rewriting исправляет опечатки")
    print("   - Страница: 18153591 (Общая информация о проекте RAUII)")
    print("   - Показывает стек технологий")
    print("\n" + "-" * 80)
    
    result = call_mcp_tool(query, limit=10, space="RAUII")
    
    print("\nРезультат:")
    if "error" in result:
        print(f"   [ERROR] Ошибка: {result['error']}")
        return False
    
    if "result" in result:
        content = result["result"].get("content", [])
        if content and len(content) > 0:
            text = content[0].get("text", "")
            print(f"   ✅ Получен ответ (длина: {len(text)} символов)")
            print("\n   Первые 2000 символов ответа:")
            print("   " + "-" * 76)
            print("   " + text[:2000].replace("\n", "\n   "))
            if len(text) > 2000:
                print(f"\n   ... (еще {len(text) - 2000} символов)")
            
            # Проверяем наличие ключевых слов
            keywords = ["Ollama", "OpenRouter", "LiteLLM", "MCP", "технологи", "стек"]
            found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
            print(f"\n   Найдено ключевых слов: {len(found_keywords)}/{len(keywords)}")
            if found_keywords:
                print(f"      {', '.join(found_keywords)}")
            
            # Проверяем наличие page_id 18153591
            if "18153591" in text:
                print("\n   [OK] Найдена ожидаемая страница 18153591")
            else:
                print("\n   [WARN] Страница 18153591 не найдена в ответе")
            
            return True
        else:
            print("   [ERROR] Пустой ответ")
            return False
    else:
        print(f"   [ERROR] Неожиданный формат ответа: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return False

def main():
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ РЕАЛЬНЫХ ЗАПРОСОВ")
    print("=" * 80)
    print(f"\nMCP URL: {MCP_URL}")
    
    # Проверка доступности сервера
    try:
        response = requests.get(MCP_URL.replace("/mcp", "/health"), timeout=5)
        print(f"[OK] Сервер доступен (health check: {response.status_code})")
    except:
        print("[WARN] Health check недоступен, но продолжаем...")
    
    print("\n")
    
    # Запуск тестов
    results = []
    
    results.append(("Тест 1: Учет номенклатуры", test_query_1()))
    results.append(("Тест 2: Технологический стек", test_query_2()))
    
    # Итоги
    print("\n" + "=" * 80)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 80)
    
    for name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{status} - {name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    print(f"\nРезультат: {passed_count}/{len(results)} тестов прошли")
    
    return 0 if passed_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())

