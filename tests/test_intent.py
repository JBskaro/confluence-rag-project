#!/usr/bin/env python3
"""Тест intent classification"""
import re

query = "технологический стек проекта RAUII"
query_lower = query.lower()

# Проверка factual patterns
factual_patterns = [
    r'\b(какой|какая|какие|что|когда|кто|сколько)\b',
    r'\b(стек|технолог|архитектур|компонент|версия|конфигурац|структур)\b',  # Технические термины
    r'\b(api|сервис|модуль|база данн|бд|интеграц)\b'  # Дополнительные термины
]

print(f"Query: {query}")
print(f"Query lower: {query_lower}")
print("\nПроверка factual patterns:")
for i, pattern in enumerate(factual_patterns):
    match = re.search(pattern, query_lower)
    print(f"  Pattern {i+1}: {pattern}")
    print(f"    Match: {match.group() if match else 'НЕТ'}")
    print(f"    Position: {match.span() if match else 'N/A'}")

# Проверка exploratory keywords
exploratory_keywords = [
    'перечень', 'подготовь', 'вопрос', 'уточняющ', 'список', 'перечисли',
    'какие', 'сравни', 'все', 'какой перечень', 'какие вопросы',
    'подготовь список', 'подготовь перечень', 'составь список'
]

print("\nПроверка exploratory keywords:")
for keyword in exploratory_keywords:
    if keyword in query_lower:
        print(f"  ✅ Найдено: '{keyword}'")

# Проверка как работает \b
print("\nПроверка границ слов:")
test_words = ['технологический', 'стек', 'технолог']
for word in test_words:
    pattern = rf'\b{word}\b'
    match = re.search(pattern, query_lower)
    print(f"  '{word}': pattern={pattern}, match={match.group() if match else 'НЕТ'}")

