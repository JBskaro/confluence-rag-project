#!/usr/bin/env python3
"""
Быстрый тест поиска внутри контейнера
"""
import sys
import json

try:
    # Импорт из server.py
    from server import collection
    # Получаем оригинальную функцию из декоратора FastMCP
    import server
    # confluence_semantic_search - это FunctionTool, получаем оригинальную функцию
    if hasattr(server.confluence_semantic_search, '__wrapped__'):
        confluence_semantic_search = server.confluence_semantic_search.__wrapped__
    elif hasattr(server.confluence_semantic_search, 'fn'):
        confluence_semantic_search = server.confluence_semantic_search.fn
    else:
        # Пробуем вызвать напрямую через tool
        confluence_semantic_search = lambda *args, **kwargs: server.confluence_semantic_search.fn(*args, **kwargs)
    
    print("=" * 80)
    print("ТЕСТ ПОИСКА В КОНТЕЙНЕРЕ")
    print("=" * 80)
    
    # Проверка базы
    count = collection.count()
    print(f"\n✅ ChromaDB: {count} документов")
    
    if count == 0:
        print("⚠️  База пуста! Дождитесь синхронизации.")
        sys.exit(1)
    
    # Тест 1: Простой поиск
    print("\n" + "=" * 80)
    print("ТЕСТ 1: Поиск 'технологии'")
    print("=" * 80)
    result1 = confluence_semantic_search("технологии", limit=3)
    print(f"\nРезультат ({len(result1)} символов):")
    print(result1[:500] + "..." if len(result1) > 500 else result1)
    
    # Тест 2: Поиск с space
    print("\n" + "=" * 80)
    print("ТЕСТ 2: Поиск 'RAUII' в space 'RAUII'")
    print("=" * 80)
    result2 = confluence_semantic_search("RAUII", limit=3, space="RAUII")
    print(f"\nРезультат ({len(result2)} символов):")
    print(result2[:500] + "..." if len(result2) > 500 else result2)
    
    # Тест 3: Проверка re-ranker
    print("\n" + "=" * 80)
    print("ТЕСТ 3: Проверка Re-ranker")
    print("=" * 80)
    try:
        from server import reranker
        if reranker and hasattr(reranker, 'model') and reranker.model:
            print(f"✅ Re-ranker инициализирован")
            model_name = getattr(reranker.model, 'model_name', 'Unknown')
            print(f"   Модель: {model_name}")
        else:
            print("⚠️  Re-ranker не инициализирован")
    except Exception as e:
        print(f"⚠️  Ошибка проверки re-ranker: {e}")
    
    # Тест 4: Проверка Hybrid Search
    print("\n" + "=" * 80)
    print("ТЕСТ 4: Проверка Hybrid Search")
    print("=" * 80)
    try:
        from hybrid_search import get_hybrid_search_stats
        stats = get_hybrid_search_stats()
        print(f"✅ Hybrid Search статус:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"⚠️  Ошибка проверки Hybrid Search: {e}")
    
    print("\n" + "=" * 80)
    print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ ОШИБКА: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

