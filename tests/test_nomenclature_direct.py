#!/usr/bin/env python3
"""
Тест с более точным запросом для поиска страницы "Учет номенклатуры"
"""
import sys

try:
    from server import collection
    import server
    
    # Получаем оригинальную функцию
    if hasattr(server.confluence_semantic_search, 'fn'):
        confluence_semantic_search = server.confluence_semantic_search.fn
    else:
        confluence_semantic_search = server.confluence_semantic_search.__wrapped__
    
    # Пробуем разные варианты запроса
    queries = [
        "Учет номенклатуры на складе список вопросов",
        "Классификация номенклатуры группы виды типы",
        "Учет движения и остатков серии характеристики",
        "список вопросов для обследования учета номенклатуры"
    ]
    
    for i, query in enumerate(queries, 1):
        print("=" * 80)
        print(f"ТЕСТ #{i}: {query}")
        print("=" * 80)
        
        result = confluence_semantic_search(query, limit=5, space="Surveys")
        
        # Проверяем наличие pageId=18153754
        if "18153754" in result:
            print(f"\n✅ НАЙДЕНА СТРАНИЦА pageId=18153754!")
            print("\n" + "=" * 80)
            print("РЕЗУЛЬТАТ:")
            print("=" * 80)
            print(result)
            break
        else:
            print(f"\n❌ Страница pageId=18153754 не найдена")
            # Показываем первые 500 символов результата
            print(f"\nПервые результаты:\n{result[:500]}...")
    
    print("\n" + "=" * 80)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ ОШИБКА: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

