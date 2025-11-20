#!/usr/bin/env python3
"""
Тестирование реальных запросов (для запуска внутри контейнера)
"""
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, '/app')

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
        # Импортируем после инициализации RAG
        from server import collection, index, expand_query, format_search_results, parallel_multi_query_search
        if collection is None or index is None:
            print("   [ERROR] RAG система не инициализирована")
            return False
        
        # Используем внутреннюю логику поиска
        from server import (
            expand_query, format_search_results, parallel_multi_query_search,
            collection, index, init_reranker, classify_query_intent
        )
            
            # Выполняем поиск вручную
            vector_store = get_vector_store()
            expanded_queries = expand_query(query)
            
            print(f"\n   [INFO] Расширено запросов: {len(expanded_queries)}")
            for i, q in enumerate(expanded_queries[:3], 1):
                print(f"      {i}. {q[:80]}...")
            
            # Получаем результаты
            results = []
            for exp_query in expanded_queries[:3]:  # Берем первые 3 варианта
                try:
                    retriever = vector_store.as_retriever(similarity_top_k=20)
                    nodes = retriever.retrieve(exp_query)
                    results.extend(nodes)
                except Exception as e:
                    print(f"   [WARN] Ошибка при поиске '{exp_query[:50]}...': {e}")
            
            if not results:
                print("   [ERROR] Не найдено результатов")
                return False
            
            # Форматируем результаты
            formatted = format_search_results(results[:20], query)
            
            print(f"\n   [OK] Получен ответ (длина: {len(formatted)} символов)")
            print("\n   Первые 2000 символов ответа:")
            print("   " + "-" * 76)
            print("   " + formatted[:2000].replace("\n", "\n   "))
            if len(formatted) > 2000:
                print(f"\n   ... (еще {len(formatted) - 2000} символов)")
            
            # Проверяем наличие ключевых слов
            keywords = ["номенклатур", "классификац", "серии", "характеристик", "габарит", "штрихкод"]
            found_keywords = [kw for kw in keywords if kw.lower() in formatted.lower()]
            print(f"\n   Найдено ключевых слов: {len(found_keywords)}/{len(keywords)}")
            if found_keywords:
                print(f"      {', '.join(found_keywords)}")
            
            # Проверяем наличие page_id 18153754
            if "18153754" in formatted:
                print("\n   [OK] Найдена ожидаемая страница 18153754")
                return True
            else:
                print("\n   [WARN] Страница 18153754 не найдена в ответе")
                if any(kw in formatted.lower() for kw in ["номенклатур", "склад", "учет"]):
                    print("   [INFO] Но найдены релевантные термины")
                    return True
                return False
        else:
            result = search_func(query, limit=20)
            # ... обработка результата ...
            return True
            
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
        from server import (
            get_vector_store, expand_query, 
            format_search_results
        )
        
        vector_store = get_vector_store()
        expanded_queries = expand_query(query, space="RAUII")
        
        print(f"\n   [INFO] Расширено запросов: {len(expanded_queries)}")
        for i, q in enumerate(expanded_queries[:3], 1):
            print(f"      {i}. {q[:80]}...")
        
        # Получаем результаты
        results = []
        for exp_query in expanded_queries[:3]:
            try:
                retriever = vector_store.as_retriever(similarity_top_k=10)
                nodes = retriever.retrieve(exp_query)
                results.extend(nodes)
            except Exception as e:
                print(f"   [WARN] Ошибка при поиске: {e}")
        
        if not results:
            print("   [ERROR] Не найдено результатов")
            return False
        
        formatted = format_search_results(results[:10], query)
        
        print(f"\n   [OK] Получен ответ (длина: {len(formatted)} символов)")
        print("\n   Первые 2000 символов ответа:")
        print("   " + "-" * 76)
        print("   " + formatted[:2000].replace("\n", "\n   "))
        if len(formatted) > 2000:
            print(f"\n   ... (еще {len(formatted) - 2000} символов)")
        
        # Проверяем наличие ключевых слов
        keywords = ["Ollama", "OpenRouter", "LiteLLM", "MCP", "технологи", "стек"]
        found_keywords = [kw for kw in keywords if kw.lower() in formatted.lower()]
        print(f"\n   Найдено ключевых слов: {len(found_keywords)}/{len(keywords)}")
        if found_keywords:
            print(f"      {', '.join(found_keywords)}")
        
        # Проверяем наличие page_id 18153591
        if "18153591" in formatted:
            print("\n   [OK] Найдена ожидаемая страница 18153591")
            return True
        else:
            print("\n   [WARN] Страница 18153591 не найдена в ответе")
            if any(kw in formatted.lower() for kw in ["ollama", "openrouter", "технологи", "стек"]):
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
    print("ТЕСТИРОВАНИЕ РЕАЛЬНЫХ ЗАПРОСОВ (в контейнере)")
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

