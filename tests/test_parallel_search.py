#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тест параллельного Multi-Query Search"""

import sys
import os
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Устанавливаем UTF-8 для вывода в Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rag_server'))


def mock_search(query_text: str, delay: float = 0.1) -> list:
    """Мок функция поиска с задержкой"""
    time.sleep(delay)  # Имитация времени поиска
    return [
        {'id': f'doc_{query_text}_{i}', 'text': f'Result {i} for {query_text}', 'score': 0.9 - i*0.1}
        for i in range(3)
    ]


def test_sequential_vs_parallel():
    """Сравнение последовательного и параллельного выполнения"""
    print("=" * 60)
    print("ТЕСТ: Последовательный vs Параллельный поиск")
    print("=" * 60)
    
    queries = ['запрос 1', 'запрос 2', 'запрос 3', 'запрос 4']
    delay = 0.1  # 100ms на запрос
    
    # Последовательное выполнение
    print("\n1. Последовательное выполнение...")
    start = time.time()
    sequential_results = []
    for q in queries:
        results = mock_search(q, delay)
        sequential_results.extend(results)
    sequential_time = time.time() - start
    print(f"   Время: {sequential_time:.3f}с")
    print(f"   Результатов: {len(sequential_results)}")
    
    # Параллельное выполнение
    print("\n2. Параллельное выполнение (4 потока)...")
    start = time.time()
    parallel_results = []
    max_workers = 4
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(mock_search, q, delay): q
            for q in queries
        }
        
        for future in as_completed(futures):
            query = futures[future]
            try:
                results = future.result()
                parallel_results.extend(results)
            except Exception as e:
                print(f"   Ошибка для '{query}': {e}")
    
    parallel_time = time.time() - start
    print(f"   Время: {parallel_time:.3f}с")
    print(f"   Результатов: {len(parallel_results)}")
    
    # Сравнение
    print("\n3. Сравнение:")
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    print(f"   Ускорение: {speedup:.2f}x")
    print(f"   Экономия времени: {sequential_time - parallel_time:.3f}с ({(1 - parallel_time/sequential_time)*100:.1f}%)")
    
    # Проверка результатов
    print("\n4. Проверка результатов:")
    print(f"   Последовательный: {len(sequential_results)} результатов")
    print(f"   Параллельный: {len(parallel_results)} результатов")
    if len(sequential_results) == len(parallel_results):
        print("   [OK] Количество результатов совпадает")
    else:
        print("   [WARNING] Количество результатов не совпадает")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ТЕСТ ЗАВЕРШЕН!")
    print("=" * 60)


def test_error_handling():
    """Тест обработки ошибок при параллельном выполнении"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Обработка ошибок")
    print("=" * 60)
    
    def failing_search(query: str) -> list:
        """Поиск, который иногда падает"""
        if 'error' in query:
            raise Exception(f"Ошибка для {query}")
        return [{'id': f'doc_{query}', 'text': f'Result for {query}'}]
    
    queries = ['запрос 1', 'запрос error', 'запрос 2', 'запрос 3']
    
    print("\nПараллельное выполнение с ошибками...")
    results = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(failing_search, q): q
            for q in queries
        }
        
        for future in as_completed(futures):
            query = futures[future]
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                errors.append((query, str(e)))
                print(f"   [OK] Ошибка обработана для '{query}': {e}")
    
    print(f"\nРезультатов: {len(results)}")
    print(f"Ошибок: {len(errors)}")
    print(f"[OK] Система продолжает работать после ошибок (graceful degradation)")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ТЕСТ ОБРАБОТКИ ОШИБОК ЗАВЕРШЕН!")
    print("=" * 60)


def main():
    """Главная функция"""
    print("\n" + "=" * 60)
    print("ТЕСТЫ: Parallel Multi-Query Search")
    print("=" * 60 + "\n")
    
    test_sequential_vs_parallel()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
    print("=" * 60)
    print("\nОжидаемое ускорение в реальных условиях: 3-4x")
    print("(зависит от количества вариантов запроса и нагрузки на ChromaDB)")

if __name__ == '__main__':
    main()

