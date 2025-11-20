#!/usr/bin/env python3
"""Тест intent classification для запроса 'технологический стек проекта RAUII'"""
import re

def classify_query_intent(query: str) -> dict:
    """
    Копия функции из mcp_rag_secure.py для тестирования
    """
    query_lower = query.lower()
    
    print(f"Query: '{query}'")
    print(f"Query lower: '{query_lower}'")
    print()
    
    # 1. Навигационные запросы
    pattern1 = r'\b(где|найди|покажи|страница|документ)\b'
    match1 = re.search(pattern1, query_lower)
    print(f"1. Navigational pattern: {pattern1}")
    print(f"   Match: {match1.group() if match1 else 'НЕТ'}")
    if match1:
        print("   → RETURNING: navigational")
        return {'type': 'navigational'}
    print()
    
    # 2. How-to запросы
    pattern2 = r'\b(как|инструкция|настроить|установить|запустить|сделать)\b'
    match2 = re.search(pattern2, query_lower)
    print(f"2. How-to pattern: {pattern2}")
    print(f"   Match: {match2.group() if match2 else 'НЕТ'}")
    if match2:
        print("   → RETURNING: howto")
        return {'type': 'howto'}
    print()
    
    # 3. Фактические запросы
    factual_patterns = [
        r'\b(какой|какая|какие|что|когда|кто|сколько)\b',
        r'\b(стек|технолог|архитектур|компонент|версия|конфигурац|структур)\b',
        r'\b(api|сервис|модуль|база данн|бд|интеграц)\b'
    ]
    print(f"3. Factual patterns:")
    for i, pattern in enumerate(factual_patterns):
        match = re.search(pattern, query_lower)
        print(f"   Pattern {i+1}: {pattern}")
        print(f"   Match: {match.group() if match else 'НЕТ'}")
        if match:
            print(f"   → Pattern {i+1} MATCHED!")
    
    if any(re.search(pattern, query_lower) for pattern in factual_patterns):
        print("   → RETURNING: factual")
        return {'type': 'factual'}
    print("   → No factual match")
    print()
    
    # 4. Исследовательские запросы
    exploratory_keywords = [
        'перечень', 'подготовь', 'вопрос', 'уточняющ', 'список', 'перечисли',
        'какие', 'сравни', 'все', 'какой перечень', 'какие вопросы',
        'подготовь список', 'подготовь перечень', 'составь список'
    ]
    print(f"4. Exploratory keywords:")
    found_keywords = [kw for kw in exploratory_keywords if kw in query_lower]
    print(f"   Found: {found_keywords if found_keywords else 'НЕТ'}")
    if found_keywords:
        print("   → RETURNING: exploratory")
        return {'type': 'exploratory'}
    print()
    
    # 5. По умолчанию
    print("5. DEFAULT: exploratory")
    return {'type': 'exploratory'}

# Тест
query = "технологический стек проекта RAUII"
result = classify_query_intent(query)
print("\n" + "=" * 60)
print(f"FINAL RESULT: {result}")

