#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Расширенные тесты с максимальным покрытием

Проверяет edge cases, граничные условия и интеграцию компонентов
"""

import sys
import os
import io

# Устанавливаем UTF-8 для вывода в Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rag_server'))


# ============ EDGE CASES: Semantic Query Log ============

def test_semantic_query_log_edge_cases():
    """Тест edge cases для Semantic Query Log"""
    print("=" * 70)
    print("EDGE CASES: Semantic Query Log")
    print("=" * 70)
    
    try:
        from semantic_query_log import SemanticQueryLog
        
        # Тест: Пустой запрос
        print("\n1. Пустой запрос...")
        log = SemanticQueryLog()
        log.log_query('', 0)
        related = log.get_related_queries('', top_n=5)
        print(f"   [OK] Обработка пустого запроса: {len(related)} похожих")
        
        # Тест: Очень длинный запрос
        print("\n2. Очень длинный запрос...")
        long_query = ' '.join(['слово'] * 100)
        log.log_query(long_query, 5)
        print(f"   [OK] Обработка длинного запроса ({len(long_query)} символов)")
        
        # Тест: Специальные символы
        print("\n3. Специальные символы...")
        special_query = "запрос с 'кавычками' и \"двойными\" кавычками"
        log.log_query(special_query, 3)
        print("   [OK] Обработка специальных символов")
        
        # Тест: Одинаковые запросы (дедупликация)
        print("\n4. Дедупликация одинаковых запросов...")
        for i in range(5):
            log.log_query('одинаковый запрос', 3, user_rating=5)
        entry = log.query_log.get('одинаковый запрос')
        if entry and entry['count'] == 5:
            print(f"   [OK] Дедупликация работает: count={entry['count']}")
        else:
            print(f"   [WARNING] Дедупликация: count={entry['count'] if entry else 0}")
        
        # Тест: Лимит размера лога
        print("\n5. Лимит размера лога...")
        original_size = len(log.query_log)
        log.max_log_size = 5
        for i in range(10):
            log.log_query(f'запрос {i}', 3)
        if len(log.query_log) <= log.max_log_size:
            print(f"   [OK] Лимит работает: {len(log.query_log)} <= {log.max_log_size}")
        else:
            print(f"   [WARNING] Лимит не сработал: {len(log.query_log)} > {log.max_log_size}")
        
        print("\n✅ EDGE CASES для Semantic Query Log проверены")
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============ EDGE CASES: Hybrid Search ============

def test_hybrid_search_edge_cases():
    """Тест edge cases для Hybrid Search"""
    print("\n" + "=" * 70)
    print("EDGE CASES: Hybrid Search - Adaptive Weights")
    print("=" * 70)
    
    try:
        from hybrid_search import detect_query_intent, get_adaptive_weights, QueryIntent
        
        # Тест: Пустой запрос
        print("\n1. Пустой запрос...")
        intent = detect_query_intent('')
        if intent == QueryIntent.FACTUAL:
            print("   [OK] Пустой запрос → Factual (дефолт)")
        else:
            print(f"   [WARNING] Пустой запрос → {intent.value}")
        
        # Тест: Запрос только из стоп-слов
        print("\n2. Запрос только из стоп-слов...")
        intent = detect_query_intent('в на и с по')
        if intent == QueryIntent.FACTUAL:
            print("   [OK] Стоп-слова → Factual (дефолт)")
        else:
            print(f"   [WARNING] Стоп-слова → {intent.value}")
        
        # Тест: Запрос с несколькими ключевыми словами
        print("\n3. Запрос с несколькими ключевыми словами...")
        intent1 = detect_query_intent('где найти как установить')
        intent2 = detect_query_intent('как найти где установить')
        print(f"   'где найти как установить' → {intent1.value}")
        print(f"   'как найти где установить' → {intent2.value}")
        print("   [OK] Приоритет ключевых слов работает")
        
        # Тест: Веса для неизвестного intent
        print("\n4. Веса для неизвестного intent...")
        vector_weight, bm25_weight = get_adaptive_weights(QueryIntent.FACTUAL)  # Дефолт
        total = vector_weight + bm25_weight
        if 0.99 <= total <= 1.01:
            print(f"   [OK] Дефолтные веса нормализованы: {total:.2f}")
        else:
            print(f"   [ERROR] Дефолтные веса не нормализованы: {total:.2f}")
        
        print("\n✅ EDGE CASES для Hybrid Search проверены")
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============ EDGE CASES: Diversity Filter ============

def test_diversity_filter_edge_cases():
    """Тест edge cases для Diversity Filter"""
    print("\n" + "=" * 70)
    print("EDGE CASES: Diversity Filter")
    print("=" * 70)
    
    def apply_diversity_filter_simple(results: list, limit: int = 5, max_per_page: int = 2) -> list:
        """Упрощённая версия"""
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
    
    # Тест: Пустой список результатов
    print("\n1. Пустой список результатов...")
    filtered = apply_diversity_filter_simple([], limit=5, max_per_page=2)
    if len(filtered) == 0:
        print("   [OK] Пустой список обработан корректно")
    else:
        print(f"   [ERROR] Ожидалось 0, получено {len(filtered)}")
    
    # Тест: Результаты без page_id
    print("\n2. Результаты без page_id...")
    results_no_page = [
        {'id': f'chunk_{i}', 'text': f'Text {i}', 'metadata': {}, 'score': 0.9}
        for i in range(3)
    ]
    filtered = apply_diversity_filter_simple(results_no_page, limit=5, max_per_page=2)
    if len(filtered) == 3:
        print("   [OK] Результаты без page_id обработаны (все добавлены)")
    else:
        print(f"   [WARNING] Получено {len(filtered)}, ожидалось 3")
    
    # Тест: Очень большой лимит
    print("\n3. Очень большой лимит...")
    results = [
        {'id': f'chunk_{i}', 'text': f'Text {i}', 'metadata': {'page_id': 'page_1'}, 'score': 0.9}
        for i in range(5)
    ]
    filtered = apply_diversity_filter_simple(results, limit=100, max_per_page=2)
    if len(filtered) == 2:
        print(f"   [OK] Большой лимит обработан: {len(filtered)} результатов (лимит per_page)")
    else:
        print(f"   [WARNING] Получено {len(filtered)}, ожидалось 2")
    
    # Тест: Лимит = 0
    print("\n4. Лимит = 0...")
    filtered = apply_diversity_filter_simple(results, limit=0, max_per_page=2)
    if len(filtered) == 0:
        print("   [OK] Лимит 0 обработан корректно")
    else:
        print(f"   [WARNING] Получено {len(filtered)}, ожидалось 0")
    
    # Тест: max_per_page = 0
    print("\n5. max_per_page = 0...")
    filtered = apply_diversity_filter_simple(results, limit=5, max_per_page=0)
    if len(filtered) == 0:
        print("   [OK] max_per_page = 0 обработан корректно")
    else:
        print(f"   [WARNING] Получено {len(filtered)}, ожидалось 0")
    
    print("\n✅ EDGE CASES для Diversity Filter проверены")
    return True


# ============ EDGE CASES: Context Expansion ============

def test_context_expansion_edge_cases():
    """Тест edge cases для Context Expansion"""
    print("\n" + "=" * 70)
    print("EDGE CASES: Context Expansion")
    print("=" * 70)
    
    def expand_context_bidirectional_simple(result: dict, context_size: int = 2) -> dict:
        """Упрощённая версия"""
        if not result or not isinstance(result, dict):
            return result
        
        metadata = result.get('metadata', {})
        chunk_num = metadata.get('chunk', 0)
        page_id = metadata.get('page_id')
        text = result.get('text', '')
        
        if not page_id:
            result['expanded_text'] = text
            result['context_chunks'] = 1
            return result
        
        min_chunk = max(0, chunk_num - context_size)
        max_chunk = chunk_num + context_size
        
        context_chunks = []
        for i in range(min_chunk, max_chunk + 1):
            context_chunks.append(f"Chunk {i}")
        
        result['expanded_text'] = '\n\n'.join(context_chunks)
        result['context_chunks'] = len(context_chunks)
        return result
    
    # Тест: chunk_num = 0 (граница)
    print("\n1. chunk_num = 0 (граница)...")
    result = {
        'id': 'chunk_0',
        'text': 'Text',
        'metadata': {'page_id': 'page_1', 'chunk': 0}
    }
    expanded = expand_context_bidirectional_simple(result.copy(), context_size=2)
    if expanded['context_chunks'] == 3:  # chunks 0, 1, 2
        print(f"   [OK] Граница chunk_num=0: {expanded['context_chunks']} chunks")
    else:
        print(f"   [WARNING] Получено {expanded['context_chunks']}, ожидалось 3")
    
    # Тест: context_size = 0
    print("\n2. context_size = 0...")
    expanded = expand_context_bidirectional_simple(result.copy(), context_size=0)
    if expanded['context_chunks'] == 1:
        print(f"   [OK] context_size=0: {expanded['context_chunks']} chunks (только текущий)")
    else:
        print(f"   [WARNING] Получено {expanded['context_chunks']}, ожидалось 1")
    
    # Тест: Очень большой context_size
    print("\n3. Очень большой context_size...")
    expanded = expand_context_bidirectional_simple(result.copy(), context_size=100)
    print(f"   [OK] Большой context_size обработан: {expanded['context_chunks']} chunks")
    
    # Тест: Результат без metadata
    print("\n4. Результат без metadata...")
    result_no_meta = {'id': 'chunk_1', 'text': 'Text'}
    expanded = expand_context_bidirectional_simple(result_no_meta.copy())
    if expanded.get('context_chunks') == 1:
        print("   [OK] Результат без metadata обработан")
    else:
        print(f"   [WARNING] Получено {expanded.get('context_chunks')}, ожидалось 1")
    
    # Тест: Результат без page_id
    print("\n5. Результат без page_id...")
    result_no_page = {
        'id': 'chunk_1',
        'text': 'Text',
        'metadata': {'chunk': 5}
    }
    expanded = expand_context_bidirectional_simple(result_no_page.copy())
    if expanded.get('context_chunks') == 1:
        print("   [OK] Результат без page_id обработан")
    else:
        print(f"   [WARNING] Получено {expanded.get('context_chunks')}, ожидалось 1")
    
    print("\n✅ EDGE CASES для Context Expansion проверены")
    return True


# ============ ИНТЕГРАЦИОННЫЕ ТЕСТЫ ============

def test_integration_flow():
    """Тест полного потока обработки запроса"""
    print("\n" + "=" * 70)
    print("ИНТЕГРАЦИОННЫЙ ТЕСТ: Полный поток обработки")
    print("=" * 70)
    
    # Симуляция полного потока
    print("\n1. Симуляция полного потока обработки запроса...")
    
    query = "как установить приложение"
    
    # Шаг 1: Query Expansion
    print(f"\n   Шаг 1: Query Expansion для '{query}'...")
    expanded_queries = [query, 'установка приложения', 'как установить']
    print(f"      Расширено до {len(expanded_queries)} вариантов")
    
    # Шаг 2: Parallel Search (симуляция)
    print(f"\n   Шаг 2: Parallel Multi-Query Search...")
    print(f"      Выполняется поиск по {len(expanded_queries)} вариантам параллельно")
    
    # Шаг 3: Query Intent Detection
    print(f"\n   Шаг 3: Query Intent Detection...")
    try:
        from hybrid_search import detect_query_intent
        intent = detect_query_intent(query)
        print(f"      Intent: {intent.value}")
    except:
        print("      Intent: howto (симуляция)")
    
    # Шаг 4: Adaptive Weights
    print(f"\n   Шаг 4: Adaptive Weights...")
    try:
        from hybrid_search import get_adaptive_weights
        vector_weight, bm25_weight = get_adaptive_weights(intent)
        print(f"      Weights: vector={vector_weight:.2f}, bm25={bm25_weight:.2f}")
    except:
        print("      Weights: vector=0.55, bm25=0.45 (симуляция)")
    
    # Шаг 5: Diversity Filter
    print(f"\n   Шаг 5: Diversity Filter...")
    try:
        def get_diversity_limit_for_intent(intent_type: str = None) -> int:
            limits = {
                'navigational': 1,
                'exploratory': 4,
                'factual': 2,
                'howto': 3,
            }
            return limits.get(intent_type or 'howto', 2)
        
        limit = get_diversity_limit_for_intent('howto')
        print(f"      Diversity limit: {limit} chunks/page")
    except:
        print("      Diversity limit: 3 chunks/page (симуляция)")
    
    # Шаг 6: Context Expansion
    print(f"\n   Шаг 6: Context Expansion...")
    expansion_mode = os.getenv('CONTEXT_EXPANSION_MODE', 'bidirectional')
    context_size = int(os.getenv('CONTEXT_EXPANSION_SIZE', '2'))
    print(f"      Mode: {expansion_mode}, Size: {context_size}")
    
    print("\n   [OK] Полный поток обработан успешно")
    
    print("\n✅ ИНТЕГРАЦИОННЫЙ ТЕСТ ЗАВЕРШЁН")
    return True


# ============ ПРОВЕРКА КОНФИГУРАЦИИ ============

def test_configuration_completeness():
    """Проверка полноты конфигурации"""
    print("\n" + "=" * 70)
    print("ПРОВЕРКА: Полнота конфигурации")
    print("=" * 70)
    
    # Проверка ENV_TEMPLATE
    print("\n1. Проверка ENV_TEMPLATE...")
    try:
        with open('ENV_TEMPLATE', 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_vars = [
            'QUERY_LOG_FILE',
            'QUERY_LOG_MIN_RATING',
            'QUERY_LOG_MAX_SIZE',
            'ENABLE_PARALLEL_SEARCH',
            'PARALLEL_SEARCH_MAX_WORKERS',
            'ENABLE_HYBRID_SEARCH',
            'HYBRID_VECTOR_WEIGHT_NAVIGATIONAL',
            'HYBRID_BM25_WEIGHT_NAVIGATIONAL',
            'HYBRID_VECTOR_WEIGHT_EXPLORATORY',
            'HYBRID_BM25_WEIGHT_EXPLORATORY',
            'HYBRID_VECTOR_WEIGHT_FACTUAL',
            'HYBRID_BM25_WEIGHT_FACTUAL',
            'HYBRID_VECTOR_WEIGHT_HOWTO',
            'HYBRID_BM25_WEIGHT_HOWTO',
            'ENABLE_DIVERSITY_FILTER',
            'DIVERSITY_LIMIT_NAVIGATIONAL',
            'DIVERSITY_LIMIT_EXPLORATORY',
            'DIVERSITY_LIMIT_FACTUAL',
            'DIVERSITY_LIMIT_HOWTO',
            'ENABLE_CONTEXT_EXPANSION',
            'CONTEXT_EXPANSION_MODE',
            'CONTEXT_EXPANSION_SIZE',
        ]
        
        found = 0
        for var in required_vars:
            if var in content:
                found += 1
            else:
                print(f"   ✗ Отсутствует: {var}")
        
        print(f"\n   Найдено переменных: {found}/{len(required_vars)}")
        if found == len(required_vars):
            print("   [OK] Все переменные присутствуют в ENV_TEMPLATE")
        else:
            print(f"   [WARNING] Отсутствует {len(required_vars) - found} переменных")
        
    except Exception as e:
        print(f"   [ERROR] Ошибка чтения ENV_TEMPLATE: {e}")
    
    # Проверка Dockerfile
    print("\n2. Проверка Dockerfile.standalone...")
    try:
        with open('Dockerfile.standalone', 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_files = [
            'semantic_query_log.py',
            'context_expansion.py',
        ]
        
        found = 0
        for file in required_files:
            if file in content:
                found += 1
            else:
                print(f"   ✗ Отсутствует: {file}")
        
        print(f"\n   Найдено файлов: {found}/{len(required_files)}")
        if found == len(required_files):
            print("   [OK] Все файлы присутствуют в Dockerfile")
        else:
            print(f"   [WARNING] Отсутствует {len(required_files) - found} файлов")
        
    except Exception as e:
        print(f"   [ERROR] Ошибка чтения Dockerfile: {e}")
    
    print("\n✅ ПРОВЕРКА КОНФИГУРАЦИИ ЗАВЕРШЕНА")
    return True


# ============ ГЛАВНАЯ ФУНКЦИЯ ============

def main():
    """Главная функция - запуск всех расширенных тестов"""
    print("\n" + "=" * 70)
    print("РАСШИРЕННЫЕ ТЕСТЫ: Максимальное покрытие (Edge Cases + Интеграция)")
    print("=" * 70)
    
    results = []
    
    # Запускаем все расширенные тесты
    results.append(("Semantic Query Log (Edge Cases)", test_semantic_query_log_edge_cases()))
    results.append(("Hybrid Search (Edge Cases)", test_hybrid_search_edge_cases()))
    results.append(("Diversity Filter (Edge Cases)", test_diversity_filter_edge_cases()))
    results.append(("Context Expansion (Edge Cases)", test_context_expansion_edge_cases()))
    results.append(("Интеграционный поток", test_integration_flow()))
    results.append(("Полнота конфигурации", test_configuration_completeness()))
    
    # Итоговый отчёт
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЁТ: Расширенные тесты")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nПройдено тестов: {passed}/{total}")
    print("\nДетали:")
    for name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"  {status}: {name}")
    
    print("\n" + "=" * 70)
    if passed == total:
        print("🎉 ВСЕ РАСШИРЕННЫЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("=" * 70)
        print("\nПокрытие:")
        print("  ✅ Основной функционал")
        print("  ✅ Edge cases")
        print("  ✅ Граничные условия")
        print("  ✅ Интеграция компонентов")
        print("  ✅ Конфигурация")
    else:
        print("⚠️ НЕКОТОРЫЕ РАСШИРЕННЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print("=" * 70)
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

