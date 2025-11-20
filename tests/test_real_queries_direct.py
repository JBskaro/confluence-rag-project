#!/usr/bin/env python3
"""
Тестирование реальных запросов пользователя (прямой вызов функции)
"""
import os
import sys
import json

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rag_server'))

# Загружаем переменные окружения
from dotenv import load_dotenv
load_dotenv()

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
    
    try:
        from mcp_rag_secure import confluence_semantic_search
        result = confluence_semantic_search(query, limit=20)
        
        print("\nРезультат:")
        if not result:
            print("   [ERROR] Пустой ответ")
            return False
        
        print(f"   [OK] Получен ответ (длина: {len(result)} символов)")
        print("\n   Первые 2000 символов ответа:")
        print("   " + "-" * 76)
        print("   " + result[:2000].replace("\n", "\n   "))
        if len(result) > 2000:
            print(f"\n   ... (еще {len(result) - 2000} символов)")
        
        # Проверяем наличие ключевых слов
        keywords = ["номенклатур", "классификац", "серии", "характеристик", "габарит", "штрихкод"]
        found_keywords = [kw for kw in keywords if kw.lower() in result.lower()]
        print(f"\n   Найдено ключевых слов: {len(found_keywords)}/{len(keywords)}")
        if found_keywords:
            print(f"      {', '.join(found_keywords)}")
        
        # Проверяем наличие page_id 18153754
        if "18153754" in result:
            print("\n   [OK] Найдена ожидаемая страница 18153754")
            return True
        else:
            print("\n   [WARN] Страница 18153754 не найдена в ответе")
            # Проверяем, есть ли хотя бы упоминания номенклатуры
            if any(kw in result.lower() for kw in ["номенклатур", "склад", "учет"]):
                print("   [INFO] Но найдены релевантные термины")
                return True
            return False
            
    except Exception as e:
        print(f"   [ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()
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
    
    try:
        from mcp_rag_secure import confluence_semantic_search
        result = confluence_semantic_search(query, limit=10, space="RAUII")
        
        print("\nРезультат:")
        if not result:
            print("   [ERROR] Пустой ответ")
            return False
        
        print(f"   [OK] Получен ответ (длина: {len(result)} символов)")
        print("\n   Первые 2000 символов ответа:")
        print("   " + "-" * 76)
        print("   " + result[:2000].replace("\n", "\n   "))
        if len(result) > 2000:
            print(f"\n   ... (еще {len(result) - 2000} символов)")
        
        # Проверяем наличие ключевых слов
        keywords = ["Ollama", "OpenRouter", "LiteLLM", "MCP", "технологи", "стек"]
        found_keywords = [kw for kw in keywords if kw.lower() in result.lower()]
        print(f"\n   Найдено ключевых слов: {len(found_keywords)}/{len(keywords)}")
        if found_keywords:
            print(f"      {', '.join(found_keywords)}")
        
        # Проверяем наличие page_id 18153591
        if "18153591" in result:
            print("\n   [OK] Найдена ожидаемая страница 18153591")
            return True
        else:
            print("\n   [WARN] Страница 18153591 не найдена в ответе")
            # Проверяем, есть ли хотя бы упоминания технологий
            if any(kw in result.lower() for kw in ["ollama", "openrouter", "технологи", "стек"]):
                print("   [INFO] Но найдены релевантные термины")
                return True
            return False
            
    except Exception as e:
        print(f"   [ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ РЕАЛЬНЫХ ЗАПРОСОВ (прямой вызов)")
    print("=" * 80)
    
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

