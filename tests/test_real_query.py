#!/usr/bin/env python3
"""
Тест реального запроса: Обследование компании по блоку Склад
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
    
    query = "Провожу обследование компании по блоку Склад, а точнее Учет номенклатуры. Подготовь список вопросов."
    
    print("=" * 80)
    print("ТЕСТ РЕАЛЬНОГО ЗАПРОСА")
    print("=" * 80)
    print(f"\n📝 Запрос: {query}")
    print(f"\n📊 Документов в базе: {collection.count()}")
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ПОИСКА:")
    print("=" * 80 + "\n")
    
    result = confluence_semantic_search(query, limit=10)
    
    print(result)
    print("\n" + "=" * 80)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ ОШИБКА: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

