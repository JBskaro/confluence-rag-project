#!/usr/bin/env python3
"""
Финальный тест MCP запроса: список вопросов для обследования учета номенклатуры
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
    
    query = "список вопросов для обследования учета номенклатуры на складе"
    
    print("=" * 80)
    print("MCP ЗАПРОС: список вопросов для обследования учета номенклатуры на складе")
    print("=" * 80)
    print(f"\n📝 Запрос: {query}")
    print(f"🎯 Ожидаемая страница: pageId=18153754")
    print(f"🔗 URL: https://confluence.rauit.ru/pages/viewpage.action?pageId=18153754")
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ПОИСКА:")
    print("=" * 80 + "\n")
    
    # Пробуем с разными параметрами
    result = confluence_semantic_search(query, limit=10, space="Surveys")
    
    print(result)
    
    # Проверка
    print("\n" + "=" * 80)
    print("ПРОВЕРКА РЕЗУЛЬТАТОВ:")
    print("=" * 80)
    
    if "18153754" in result:
        print("✅ Страница pageId=18153754 найдена!")
    else:
        print("❌ Страница pageId=18153754 НЕ найдена")
    
    # Проверяем наличие ключевых разделов
    sections = [
        "1. Классификация номенклатуры",
        "2. Учет движения и остатков",
        "3. Логистические и финансовые параметры",
        "4. Классификация и хранение товаров",
        "5. Дополнительные вопросы"
    ]
    
    found_sections = sum(1 for s in sections if s in result)
    print(f"\n📊 Найдено разделов: {found_sections}/{len(sections)}")
    
    if found_sections == len(sections):
        print("🎉 ВСЕ РАЗДЕЛЫ НАЙДЕНЫ!")
    elif found_sections >= 3:
        print("⚠️ Найдены основные разделы")
    else:
        print("❌ Не все разделы найдены")
    
    print("\n" + "=" * 80)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ ОШИБКА: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

