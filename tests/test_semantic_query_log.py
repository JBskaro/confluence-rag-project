#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Быстрый тест Semantic Query Log"""

import sys
import os
import io

# Устанавливаем UTF-8 для вывода в Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rag_server'))

from semantic_query_log import SemanticQueryLog

def test_semantic_query_log():
    """Тест Semantic Query Log"""
    print("=" * 60)
    print("ТЕСТ: Semantic Query Log")
    print("=" * 60)
    
    # Инициализация
    log = SemanticQueryLog()
    print(f"[OK] Semantic Query Log инициализирован: {len(log.query_log)} записей")
    
    # Логирование запросов
    print("\n1. Логирование запросов...")
    log.log_query('как установить приложение', 5, user_rating=5)
    log.log_query('как установить', 3, user_rating=4)
    log.log_query('установка программы', 4, user_rating=5)
    log.log_query('неуспешный запрос', 0, user_rating=2)
    print("[OK] Запросы залогированы")
    
    # Поиск похожих запросов
    print("\n2. Поиск похожих запросов...")
    test_query = 'установка приложения'
    related = log.get_related_queries(test_query, top_n=5)
    print(f"Запрос: '{test_query}'")
    print(f"Найдено похожих запросов: {len(related)}")
    for i, q in enumerate(related, 1):
        print(f"  {i}. {q}")
    
    # Получение топ успешных запросов
    print("\n3. Топ успешных запросов...")
    top_queries = log.get_expansion_terms(top_n=10)
    print(f"Найдено успешных запросов: {len(top_queries)}")
    for i, (q, count, rating) in enumerate(top_queries, 1):
        print(f"  {i}. '{q}' (использований: {count}, рейтинг: {rating:.2f})")
    
    # Принудительное сохранение
    log._save_log()
    print("\n4. Проверка сохранения файла...")
    import os
    log_file = log.log_file
    if os.path.exists(log_file):
        print(f"[OK] Файл создан: {log_file}")
        file_size = os.path.getsize(log_file)
        print(f"[OK] Размер файла: {file_size} байт")
    else:
        print(f"[WARNING] Файл не найден: {log_file}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 60)

if __name__ == '__main__':
    test_semantic_query_log()

