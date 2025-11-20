#!/usr/bin/env python3
"""
Скрипт для сбора статистики rerank scores на разных запросах
"""
import sys
import os

try:
    from server import collection
    import server
    
    # Убеждаемся, что пороги установлены в 0
    os.environ['RERANK_THRESHOLD_TECHNICAL'] = '0'
    os.environ['RERANK_THRESHOLD_GENERAL'] = '0'
    
    # Получаем оригинальную функцию
    if hasattr(server.confluence_semantic_search, 'fn'):
        confluence_semantic_search = server.confluence_semantic_search.fn
    else:
        confluence_semantic_search = server.confluence_semantic_search.__wrapped__
    
    # Разнообразные тестовые запросы
    test_queries = [
        "список вопросов для обследования учета номенклатуры на складе",
        "Учет номенклатуры классификация группы виды типы",
        "Как работает синхронизация с Confluence",
        "Какие технологии используются в проекте RAUII",
        "Процесс деплоя приложения",
        "Настройка безопасности API",
    ]
    
    all_scores = []
    
    print("=" * 80)
    print("СБОР СТАТИСТИКИ RERANK SCORES")
    print("=" * 80)
    print(f"Пороги установлены: TECHNICAL=0, GENERAL=0")
    print(f"Всего запросов для тестирования: {len(test_queries)}\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] Запрос: {query[:60]}...")
        
        try:
            result = confluence_semantic_search(query, limit=20, space="")
            
            # Извлекаем scores из результата (если они там есть)
            # В реальности scores не возвращаются в результате, только в логах
            # Но мы можем проверить, сколько результатов вернулось
            result_count = result.count("📍") if "📍" in result else 0
            print(f"  ✅ Найдено результатов: {result_count}")
            
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Сбор статистики завершен")
    print("=" * 80)
    print("\n📊 Проверьте логи для анализа scores:")
    print("   docker-compose logs --tail=500 | Select-String -Pattern 'Top score'")
    
except Exception as e:
    print(f"❌ ОШИБКА: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

