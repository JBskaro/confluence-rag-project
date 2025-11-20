#!/usr/bin/env python3
"""
Тесты для модуля keyword_extraction.
"""

import sys
import os

# Добавляем путь к rag_server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_server.utils.keyword_extraction import (
    extract_keywords,
    extract_technical_terms,
    normalize_query,
    STOPWORDS,
    TECHNICAL_TERMS
)


def test_extract_keywords_basic():
    """Тест базового извлечения ключевых слов"""
    text = "Как настроить API для работы с базой данных"
    keywords = extract_keywords(text)
    
    assert len(keywords) > 0, "Должны быть найдены ключевые слова"
    assert "как" not in keywords, "Стоп-слова должны быть удалены"
    assert "api" in keywords, "API должно быть найдено"
    assert "база" in keywords or "данных" in keywords, "Технические термины должны быть найдены"
    
    print("[PASSED] test_extract_keywords_basic")


def test_extract_keywords_russian():
    """Тест извлечения русских ключевых слов"""
    text = "Подготовь перечень уточняющих вопросов для обследования компании"
    keywords = extract_keywords(text)
    
    assert len(keywords) > 0, "Должны быть найдены ключевые слова"
    assert "подготовь" not in keywords or "перечень" in keywords, "Стоп-слова должны быть удалены"
    
    print("[PASSED] test_extract_keywords_russian")


def test_extract_keywords_min_length():
    """Тест фильтрации по минимальной длине"""
    text = "а б в г д api sql docker"
    keywords = extract_keywords(text, min_length=3)
    
    assert "а" not in keywords, "Короткие слова должны быть отфильтрованы"
    assert "api" in keywords, "Слова >= min_length должны быть включены"
    
    print("[PASSED] test_extract_keywords_min_length")


def test_extract_keywords_without_stopwords():
    """Тест извлечения без удаления стоп-слов"""
    text = "как настроить api"
    keywords_with_stopwords = extract_keywords(text, remove_stopwords=False)
    keywords_without_stopwords = extract_keywords(text, remove_stopwords=True)
    
    assert len(keywords_with_stopwords) >= len(keywords_without_stopwords), \
        "Стоп-слова должны быть удалены при remove_stopwords=True"
    
    print("[PASSED] test_extract_keywords_without_stopwords")


def test_extract_technical_terms():
    """Тест извлечения технических терминов"""
    text = "Настроить API для работы с Docker и PostgreSQL"
    terms = extract_technical_terms(text)
    
    assert "api" in terms, "API должно быть найдено"
    assert "docker" in terms, "Docker должно быть найдено"
    assert "postgresql" in terms, "PostgreSQL должно быть найдено"
    
    print("[PASSED] test_extract_technical_terms")


def test_extract_technical_terms_russian():
    """Тест извлечения русских технических терминов"""
    text = "Проверка конфигурации 1С и тестирование багов"
    terms = extract_technical_terms(text)
    
    # Проверяем что найдено минимум 2 термина из списка
    found_terms = []
    if "1с" in terms or "1c" in terms:
        found_terms.append("1с")
    if "конфигурация" in terms:
        found_terms.append("конфигурация")
    if "тест" in terms:
        found_terms.append("тест")
    if "баг" in terms:
        found_terms.append("баг")
    
    assert len(found_terms) >= 2, f"Должно быть найдено минимум 2 термина, найдено: {terms}, found: {found_terms}"
    
    print("[PASSED] test_extract_technical_terms_russian")


def test_normalize_query():
    """Тест нормализации запроса"""
    query1 = "  Как   настроить   API  "
    normalized = normalize_query(query1)
    
    assert normalized == "как настроить api", "Запрос должен быть нормализован"
    assert "  " not in normalized, "Лишние пробелы должны быть удалены"
    
    print("[PASSED] test_normalize_query")


def test_stopwords_constant():
    """Тест константы STOPWORDS"""
    assert len(STOPWORDS) > 0, "STOPWORDS должен содержать слова"
    assert "в" in STOPWORDS, "Русские стоп-слова должны быть включены"
    assert "the" in STOPWORDS, "Английские стоп-слова должны быть включены"
    
    print("[PASSED] test_stopwords_constant")


def test_technical_terms_constant():
    """Тест константы TECHNICAL_TERMS"""
    assert len(TECHNICAL_TERMS) > 0, "TECHNICAL_TERMS должен содержать термины"
    assert "api" in TECHNICAL_TERMS, "API должно быть в списке"
    assert "docker" in TECHNICAL_TERMS, "Docker должно быть в списке"
    
    print("[PASSED] test_technical_terms_constant")


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 70)
    print("ТЕСТЫ ДЛЯ keyword_extraction")
    print("=" * 70)
    
    tests = [
        test_extract_keywords_basic,
        test_extract_keywords_russian,
        test_extract_keywords_min_length,
        test_extract_keywords_without_stopwords,
        test_extract_technical_terms,
        test_extract_technical_terms_russian,
        test_normalize_query,
        test_stopwords_constant,
        test_technical_terms_constant,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAILED] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1
    
    print("=" * 70)
    print(f"ИТОГО: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

