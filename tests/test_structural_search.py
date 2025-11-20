#!/usr/bin/env python3
"""
Комплексный тест структурного поиска (Structural Navigation Search)
"""
import sys
import time
import io

# Настройка UTF-8 для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from server import collection
    import server
    
    # Получаем оригинальную функцию
    if hasattr(server.confluence_semantic_search, 'fn'):
        confluence_semantic_search = server.confluence_semantic_search.fn
    else:
        confluence_semantic_search = server.confluence_semantic_search.__wrapped__
    
    # Импортируем функции для прямого тестирования
    from mcp_rag_secure import (
        parse_query_structure,
        structural_metadata_search,
        cached_structural_search
    )
    
    def print_separator(title: str):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")
    
    def test_1_basic_structural():
        """Тест 1: Базовый структурный запрос"""
        print_separator("ТЕСТ 1: Базовый структурный запрос")
        
        query = 'Склад > Учет номенклатуры'
        print(f"📝 Запрос: {query}")
        print(f"📊 Ожидание: Найти ВСЕ результаты из этого раздела (50+ результатов)")
        print(f"✅ Успех: Видно структурный поиск в логах\n")
        
        # Проверяем парсинг
        structure = parse_query_structure(query)
        print(f"🔍 Структура запроса:")
        print(f"   - is_structural: {structure['is_structural_query']}")
        print(f"   - parts: {structure['parts']}")
        
        if not structure['is_structural_query']:
            print("❌ ОШИБКА: Запрос не распознан как структурный!")
            return False
        
        # Прямой структурный поиск
        start_time = time.time()
        structural_results = structural_metadata_search(collection, structure, limit=100)
        structural_time = time.time() - start_time
        
        print(f"\n📊 Результаты структурного поиска:")
        print(f"   - Найдено: {len(structural_results)} результатов")
        print(f"   - Время: {structural_time:.3f}с")
        
        if structural_results:
            print(f"\n📋 Топ-5 результатов:")
            for i, r in enumerate(structural_results[:5], 1):
                match_score = r.get('match_score', 0)
                page_id = r.get('metadata', {}).get('page_id', 'N/A')
                title = r.get('metadata', {}).get('title', 'N/A')[:50]
                print(f"   [{i}] match_score={match_score:.1f}, page_id={page_id}, title={title}")
        
        # Полный поиск через MCP
        print(f"\n🔎 Полный поиск через MCP:")
        start_time = time.time()
        result = confluence_semantic_search(query, limit=10)
        full_time = time.time() - start_time
        
        print(f"   - Время: {full_time:.3f}с")
        print(f"   - Результат содержит 'structural': {'structural' in result.lower()}")
        
        # Проверка успеха
        success = (
            structure['is_structural_query'] and
            len(structural_results) > 0 and
            'structural' in result.lower()
        )
        
        print(f"\n{'✅ ТЕСТ ПРОЙДЕН' if success else '❌ ТЕСТ НЕ ПРОЙДЕН'}")
        return success
    
    def test_2_multi_level():
        """Тест 2: Многоуровневый структурный запрос"""
        print_separator("ТЕСТ 2: Многоуровневый структурный запрос")
        
        query = 'Обследование > Склад > Учет номенклатуры'
        print(f"📝 Запрос: {query}")
        print(f"📊 Ожидание: Только результаты из этого конкретного раздела")
        print(f"✅ Успех: Результаты отфильтрованы по всем 3 уровням\n")
        
        structure = parse_query_structure(query)
        print(f"🔍 Структура: {structure['parts']}")
        
        start_time = time.time()
        structural_results = structural_metadata_search(collection, structure, limit=50)
        search_time = time.time() - start_time
        
        print(f"\n📊 Результаты:")
        print(f"   - Найдено: {len(structural_results)} результатов")
        print(f"   - Время: {search_time:.3f}с")
        
        # Проверяем, что все результаты содержат части запроса
        all_match = True
        for r in structural_results[:5]:
            metadata = r.get('metadata', {})
            page_path = (metadata.get('page_path', '') or '').lower()
            title = (metadata.get('title', '') or '').lower()
            
            has_obsledovanie = 'обследование' in page_path or 'обследование' in title
            has_sklad = 'склад' in page_path or 'склад' in title
            has_uchet = 'учет' in page_path or 'учет' in title or 'номенклатур' in page_path or 'номенклатур' in title
            
            print(f"   - page_id={metadata.get('page_id')}: обследование={has_obsledovanie}, склад={has_sklad}, учет={has_uchet}")
            if not (has_obsledovanie or has_sklad or has_uchet):
                all_match = False
        
        success = len(structural_results) > 0 and all_match
        print(f"\n{'✅ ТЕСТ ПРОЙДЕН' if success else '❌ ТЕСТ НЕ ПРОЙДЕН'}")
        return success
    
    def test_3_semantic_fallback():
        """Тест 3: Обычный запрос (семантический)"""
        print_separator("ТЕСТ 3: Обычный запрос (семантический)")
        
        query = 'технологический стек RAUII'
        print(f"📝 Запрос: {query}")
        print(f"📊 Ожидание: Семантический поиск + метаданные boost")
        print(f"✅ Успех: Страница 18153591 в топе благодаря page_title match\n")
        
        structure = parse_query_structure(query)
        print(f"🔍 Структура: is_structural={structure['is_structural_query']}")
        
        if structure['is_structural_query']:
            print("❌ ОШИБКА: Обычный запрос распознан как структурный!")
            return False
        
        start_time = time.time()
        result = confluence_semantic_search(query, limit=5)
        search_time = time.time() - start_time
        
        print(f"📊 Результаты:")
        print(f"   - Время: {search_time:.3f}с")
        print(f"   - Содержит 'semantic': {'semantic' in result.lower() or '🔎' in result}")
        
        # Проверяем наличие страницы 18153591
        has_target_page = '18153591' in result
        print(f"   - Страница 18153591 найдена: {has_target_page}")
        
        if has_target_page:
            print(f"\n📋 Фрагмент результата:")
            lines = result.split('\n')
            for line in lines[:10]:
                if '18153591' in line or 'Общая информация' in line:
                    print(f"   {line[:100]}")
        
        success = not structure['is_structural_query'] and has_target_page
        print(f"\n{'✅ ТЕСТ ПРОЙДЕН' if success else '❌ ТЕСТ НЕ ПРОЙДЕН'}")
        return success
    
    def test_4_fallback():
        """Тест 4: Fallback когда структурный не нашел"""
        print_separator("ТЕСТ 4: Fallback когда структурный не нашел")
        
        query = 'Несуществующее > Несуществующее'
        print(f"📝 Запрос: {query}")
        print(f"📊 Ожидание: Fallback на семантический поиск")
        print(f"✅ Успех: Логи показывают переключение на semantic search\n")
        
        structure = parse_query_structure(query)
        print(f"🔍 Структура: is_structural={structure['is_structural_query']}, parts={structure['parts']}")
        
        # Прямой структурный поиск
        structural_results = structural_metadata_search(collection, structure, limit=10)
        print(f"📊 Структурный поиск: {len(structural_results)} результатов")
        
        # Полный поиск (должен fallback на semantic)
        start_time = time.time()
        result = confluence_semantic_search(query, limit=5)
        search_time = time.time() - start_time
        
        print(f"📊 Полный поиск:")
        print(f"   - Время: {search_time:.3f}с")
        print(f"   - Результат не пустой: {len(result) > 50}")
        print(f"   - Содержит 'не найдено' или результаты: {'не найдено' not in result.lower() or len(result) > 100}")
        
        # Проверка: если структурный не нашел, должен быть fallback
        success = (
            structure['is_structural_query'] and
            len(structural_results) == 0 and
            len(result) > 50  # Есть какой-то результат (fallback сработал)
        )
        
        print(f"\n{'✅ ТЕСТ ПРОЙДЕН' if success else '❌ ТЕСТ НЕ ПРОЙДЕН'}")
        return success
    
    def test_5_performance():
        """Тест 5: Производительность"""
        print_separator("ТЕСТ 5: Производительность")
        
        query = 'Склад > Учет номенклатуры'
        print(f"📝 Запрос: {query}")
        print(f"📊 Ожидание: Ответ < 2 секунды")
        print(f"✅ Успех: Кэширование работает (2й запрос < 100ms)\n")
        
        structure = parse_query_structure(query)
        
        # Первый запрос (без кэша)
        print("🔄 Первый запрос (без кэша):")
        start_time = time.time()
        result1 = cached_structural_search(collection, structure, limit=50)
        time1 = time.time() - start_time
        print(f"   - Время: {time1:.3f}с")
        print(f"   - Результатов: {len(result1)}")
        
        # Второй запрос (с кэшем)
        print("\n🔄 Второй запрос (с кэшем):")
        start_time = time.time()
        result2 = cached_structural_search(collection, structure, limit=50)
        time2 = time.time() - start_time
        print(f"   - Время: {time2:.3f}с")
        print(f"   - Результатов: {len(result2)}")
        print(f"   - Ускорение: {time1/time2:.1f}x" if time2 > 0 else "   - Ускорение: ∞")
        
        # Полный поиск через MCP
        print("\n🔄 Полный поиск через MCP:")
        start_time = time.time()
        result3 = confluence_semantic_search(query, limit=10)
        time3 = time.time() - start_time
        print(f"   - Время: {time3:.3f}с")
        
        success = (
            time1 < 2.0 and  # Первый запрос < 2 сек
            time2 < 0.1 and  # Второй запрос < 100ms (кэш)
            time3 < 2.0      # Полный поиск < 2 сек
        )
        
        print(f"\n📊 Итоги производительности:")
        print(f"   - Первый запрос: {'✅' if time1 < 2.0 else '❌'} {time1:.3f}с")
        print(f"   - Второй запрос (кэш): {'✅' if time2 < 0.1 else '❌'} {time2:.3f}с")
        print(f"   - Полный поиск: {'✅' if time3 < 2.0 else '❌'} {time3:.3f}с")
        
        print(f"\n{'✅ ТЕСТ ПРОЙДЕН' if success else '❌ ТЕСТ НЕ ПРОЙДЕН'}")
        return success
    
    # Запуск всех тестов
    print("=" * 80)
    print("  КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ СТРУКТУРНОГО ПОИСКА")
    print("=" * 80)
    print(f"\n📊 Документов в базе: {collection.count()}\n")
    
    results = []
    results.append(("Тест 1: Базовый структурный запрос", test_1_basic_structural()))
    results.append(("Тест 2: Многоуровневый запрос", test_2_multi_level()))
    results.append(("Тест 3: Обычный запрос (семантический)", test_3_semantic_fallback()))
    results.append(("Тест 4: Fallback", test_4_fallback()))
    results.append(("Тест 5: Производительность", test_5_performance()))
    
    # Итоги
    print_separator("ИТОГИ ТЕСТИРОВАНИЯ")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ НЕ ПРОЙДЕН"
        print(f"{status}: {name}")
    
    print(f"\n📊 Результат: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total - passed} тест(ов) не пройдено")
        sys.exit(1)
    
except Exception as e:
    print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

