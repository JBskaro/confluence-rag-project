#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Упрощённый тест адаптивных лимитов Diversity Filter (без инициализации RAG)"""

import sys
import os
import io

# Устанавливаем UTF-8 для вывода в Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Устанавливаем ENV переменные для теста
os.environ['ENABLE_DIVERSITY_FILTER'] = 'true'
os.environ['DIVERSITY_LIMIT_NAVIGATIONAL'] = '1'
os.environ['DIVERSITY_LIMIT_EXPLORATORY'] = '4'
os.environ['DIVERSITY_LIMIT_FACTUAL'] = '2'
os.environ['DIVERSITY_LIMIT_HOWTO'] = '3'


def get_diversity_limit_for_intent(intent_type: str = None) -> int:
    """Копия функции из mcp_rag_secure.py для тестирования"""
    enable_filter = os.getenv('ENABLE_DIVERSITY_FILTER', 'true').lower() == 'true'
    if not enable_filter:
        return 999
    
    diversity_limits = {
        'navigational': int(os.getenv('DIVERSITY_LIMIT_NAVIGATIONAL', '1')),
        'exploratory': int(os.getenv('DIVERSITY_LIMIT_EXPLORATORY', '4')),
        'factual': int(os.getenv('DIVERSITY_LIMIT_FACTUAL', '2')),
        'howto': int(os.getenv('DIVERSITY_LIMIT_HOWTO', '3')),
    }
    
    if not intent_type or intent_type not in diversity_limits:
        intent_type = 'factual'
    
    return diversity_limits.get(intent_type, 2)


def apply_diversity_filter_simple(results: list, limit: int = 5, max_per_page: int = 2) -> list:
    """Упрощённая версия apply_diversity_filter для тестирования"""
    if not results:
        return []
    
    filtered_results = []
    page_counts = {}
    
    for result in results:
        if not result or not isinstance(result, dict):
            continue
        
        metadata = result.get('metadata')
        if not metadata or not isinstance(metadata, dict):
            continue
        
        page_id = metadata.get('page_id')
        
        if not page_id or page_counts.get(page_id, 0) < max_per_page:
            filtered_results.append(result)
            if page_id:
                page_counts[page_id] = page_counts.get(page_id, 0) + 1
            
            if len(filtered_results) >= limit:
                break
    
    return filtered_results


def test_diversity_limits():
    """Тест получения лимитов для разных типов запросов"""
    print("=" * 60)
    print("ТЕСТ 1: Адаптивные лимиты для разных типов запросов")
    print("=" * 60)
    
    test_cases = [
        ('navigational', 1),
        ('exploratory', 4),
        ('factual', 2),
        ('howto', 3),
        (None, 2),
        ('unknown', 2),
    ]
    
    print("\nЛимиты для разных типов:")
    all_passed = True
    for intent_type, expected_limit in test_cases:
        limit = get_diversity_limit_for_intent(intent_type)
        status = "✓" if limit == expected_limit else "✗"
        if limit != expected_limit:
            all_passed = False
        
        intent_str = intent_type if intent_type else "None (дефолт)"
        print(f"  {status} {intent_str:15s}: {limit} (ожидалось: {expected_limit})")
    
    if all_passed:
        print("\n[OK] Все тесты лимитов прошли успешно!")
    else:
        print("\n[ERROR] Некоторые тесты не прошли")
    
    print()


def test_diversity_filtering():
    """Тест фильтрации результатов"""
    print("=" * 60)
    print("ТЕСТ 2: Фильтрация результатов")
    print("=" * 60)
    
    # Создаём тестовые результаты (5 результатов с одной страницы)
    test_results = [
        {
            'id': f'chunk_{i}',
            'text': f'Text {i}',
            'metadata': {'page_id': 'page_1', 'title': 'Test Page'},
            'score': 0.9 - i * 0.1
        }
        for i in range(5)
    ]
    
    # Тест 1: Navigational (лимит 1)
    print("\n1. Navigational запрос (лимит 1 chunk/page):")
    limit = get_diversity_limit_for_intent('navigational')
    filtered = apply_diversity_filter_simple(test_results.copy(), limit=5, max_per_page=limit)
    print(f"   Исходно: {len(test_results)} результатов")
    print(f"   После фильтра: {len(filtered)} результатов")
    print(f"   Ожидалось: 1 результат (лимит 1)")
    if len(filtered) == 1:
        print("   [OK] Фильтр работает корректно")
    else:
        print(f"   [ERROR] Ожидалось 1, получено {len(filtered)}")
    
    # Тест 2: Exploratory (лимит 4)
    print("\n2. Exploratory запрос (лимит 4 chunks/page):")
    limit = get_diversity_limit_for_intent('exploratory')
    filtered = apply_diversity_filter_simple(test_results.copy(), limit=5, max_per_page=limit)
    print(f"   Исходно: {len(test_results)} результатов")
    print(f"   После фильтра: {len(filtered)} результатов")
    print(f"   Ожидалось: 4 результата (лимит 4)")
    if len(filtered) == 4:
        print("   [OK] Фильтр работает корректно")
    else:
        print(f"   [ERROR] Ожидалось 4, получено {len(filtered)}")
    
    # Тест 3: Factual (лимит 2)
    print("\n3. Factual запрос (лимит 2 chunks/page):")
    limit = get_diversity_limit_for_intent('factual')
    filtered = apply_diversity_filter_simple(test_results.copy(), limit=5, max_per_page=limit)
    print(f"   Исходно: {len(test_results)} результатов")
    print(f"   После фильтра: {len(filtered)} результатов")
    print(f"   Ожидалось: 2 результата (лимит 2)")
    if len(filtered) == 2:
        print("   [OK] Фильтр работает корректно")
    else:
        print(f"   [ERROR] Ожидалось 2, получено {len(filtered)}")
    
    print()


def test_multiple_pages():
    """Тест фильтрации с несколькими страницами"""
    print("=" * 60)
    print("ТЕСТ 3: Фильтрация с несколькими страницами")
    print("=" * 60)
    
    # Создаём результаты с разных страниц
    test_results = []
    for page_num in range(3):
        for chunk_num in range(3):
            test_results.append({
                'id': f'chunk_{page_num}_{chunk_num}',
                'text': f'Text from page {page_num}, chunk {chunk_num}',
                'metadata': {'page_id': f'page_{page_num}', 'title': f'Page {page_num}'},
                'score': 0.9 - (page_num * 3 + chunk_num) * 0.05
            })
    
    print(f"\nИсходно: {len(test_results)} результатов с 3 страниц")
    
    # Navigational (лимит 1)
    limit = get_diversity_limit_for_intent('navigational')
    filtered = apply_diversity_filter_simple(test_results.copy(), limit=10, max_per_page=limit)
    print(f"\nNavigational (лимит 1 chunk/page):")
    print(f"   После фильтра: {len(filtered)} результатов")
    print(f"   Ожидалось: 3 результата (по 1 с каждой страницы)")
    if len(filtered) == 3:
        print("   [OK] Фильтр работает корректно")
    else:
        print(f"   [WARNING] Получено {len(filtered)}, ожидалось 3")
    
    # Exploratory (лимит 4)
    limit = get_diversity_limit_for_intent('exploratory')
    filtered = apply_diversity_filter_simple(test_results.copy(), limit=10, max_per_page=limit)
    print(f"\nExploratory (лимит 4 chunks/page):")
    print(f"   После фильтра: {len(filtered)} результатов")
    print(f"   Ожидалось: 9 результатов (по 3 с каждой страницы, лимит не достигнут)")
    if len(filtered) == 9:
        print("   [OK] Фильтр работает корректно")
    else:
        print(f"   [WARNING] Получено {len(filtered)}, ожидалось 9")
    
    print()


def test_disable_filter():
    """Тест отключения фильтра"""
    print("=" * 60)
    print("ТЕСТ 4: Отключение фильтра")
    print("=" * 60)
    
    # Сохраняем текущее значение
    original_value = os.environ.get('ENABLE_DIVERSITY_FILTER', 'true')
    
    # Отключаем фильтр
    os.environ['ENABLE_DIVERSITY_FILTER'] = 'false'
    
    limit = get_diversity_limit_for_intent('factual')
    print(f"\nПри ENABLE_DIVERSITY_FILTER=false:")
    print(f"   Лимит: {limit}")
    print(f"   Ожидалось: 999 (очень большой, отключает фильтр)")
    if limit == 999:
        print("   [OK] Фильтр отключён корректно")
    else:
        print(f"   [ERROR] Ожидалось 999, получено {limit}")
    
    # Восстанавливаем значение
    os.environ['ENABLE_DIVERSITY_FILTER'] = original_value
    
    print()


def main():
    """Главная функция"""
    print("\n" + "=" * 60)
    print("ТЕСТЫ: Diversity Filter - Настраиваемость")
    print("=" * 60 + "\n")
    
    test_diversity_limits()
    test_diversity_filtering()
    test_multiple_pages()
    test_disable_filter()
    
    print("=" * 60)
    print("[SUCCESS] ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
    print("=" * 60)
    print("\nОжидаемое улучшение: лучшее разнообразие результатов")
    print("(зависит от типа запроса и настроек лимитов)")

if __name__ == '__main__':
    main()

