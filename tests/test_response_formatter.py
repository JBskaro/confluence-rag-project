#!/usr/bin/env python3
"""
Тесты для модуля response_formatter.
"""

import sys
import os

# Добавляем путь к rag_server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_server.response_formatter import ResponseFormatter


def test_format_success_basic():
    """Тест базового форматирования успешного ответа"""
    results = [
        {
            'title': 'Test Page',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/page', 'chunk': 1},
            'text': 'Sample text content',
            'rerank_score': 0.85,
            'final_score': 0.85
        }
    ]
    
    response = ResponseFormatter.format_success(
        query="test query",
        results=results
    )
    
    assert "test query" in response, "Запрос должен быть в ответе"
    assert "Test Page" in response, "Название страницы должно быть в ответе"
    assert "TEST" in response, "Пространство должно быть в ответе"
    assert "0.850" in response, "Score должен быть в ответе"
    
    print("[PASSED] test_format_success_basic")


def test_format_success_with_intent():
    """Тест форматирования с intent"""
    results = [
        {
            'title': 'Test Page',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/page', 'chunk': 1},
            'text': 'Sample text',
            'rerank_score': 0.75,
            'final_score': 0.75
        }
    ]
    
    intent = {'type': 'exploratory', 'diversity': 2}
    
    response = ResponseFormatter.format_success(
        query="test query",
        results=results,
        intent=intent
    )
    
    assert "exploratory" in response, "Тип intent должен быть в ответе"
    assert "Query Type: exploratory" in response, "Query Type должен быть указан"
    
    print("[PASSED] test_format_success_with_intent")


def test_format_success_with_latency():
    """Тест форматирования с latency"""
    results = [
        {
            'title': 'Test Page',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/page', 'chunk': 1},
            'text': 'Sample text',
            'rerank_score': 0.65,
            'final_score': 0.65
        }
    ]
    
    response = ResponseFormatter.format_success(
        query="test query",
        results=results,
        latency_ms=123
    )
    
    assert "123ms" in response or "Time: 123ms" in response, "Latency должен быть в ответе"
    
    print("[PASSED] test_format_success_with_latency")


def test_format_success_multiple_results():
    """Тест форматирования нескольких результатов"""
    results = [
        {
            'title': 'Page 1',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/page1', 'chunk': 1},
            'text': 'Text 1',
            'rerank_score': 0.9,
            'final_score': 0.9
        },
        {
            'title': 'Page 2',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/page2', 'chunk': 2},
            'text': 'Text 2',
            'rerank_score': 0.8,
            'final_score': 0.8
        }
    ]
    
    response = ResponseFormatter.format_success(
        query="test query",
        results=results
    )
    
    assert "Page 1" in response, "Первый результат должен быть в ответе"
    assert "Page 2" in response, "Второй результат должен быть в ответе"
    assert "Results: 2" in response, "Количество результатов должно быть указано"
    
    print("[PASSED] test_format_success_multiple_results")


def test_format_success_score_emojis():
    """Тест эмодзи для разных score"""
    # Высокий score (> 0.7)
    results_high = [
        {
            'title': 'High Score',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/high', 'chunk': 1},
            'text': 'Text',
            'rerank_score': 0.85,
            'final_score': 0.85
        }
    ]
    
    response_high = ResponseFormatter.format_success(
        query="test",
        results=results_high
    )
    
    assert "🟢" in response_high, "Высокий score должен иметь зеленый эмодзи"
    
    # Средний score (0.3-0.7)
    results_medium = [
        {
            'title': 'Medium Score',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/medium', 'chunk': 1},
            'text': 'Text',
            'rerank_score': 0.5,
            'final_score': 0.5
        }
    ]
    
    response_medium = ResponseFormatter.format_success(
        query="test",
        results=results_medium
    )
    
    assert "🟡" in response_medium, "Средний score должен иметь желтый эмодзи"
    
    # Низкий score (0.1-0.3)
    results_low = [
        {
            'title': 'Low Score',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/low', 'chunk': 1},
            'text': 'Text',
            'rerank_score': 0.2,
            'final_score': 0.2
        }
    ]
    
    response_low = ResponseFormatter.format_success(
        query="test",
        results=results_low
    )
    
    assert "🟠" in response_low, "Низкий score должен иметь оранжевый эмодзи"
    
    # Очень низкий score (< 0.1)
    results_very_low = [
        {
            'title': 'Very Low Score',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/vlow', 'chunk': 1},
            'text': 'Text',
            'rerank_score': 0.05,
            'final_score': 0.05
        }
    ]
    
    response_very_low = ResponseFormatter.format_success(
        query="test",
        results=results_very_low
    )
    
    assert "⚪" in response_very_low, "Очень низкий score должен иметь белый эмодзи"
    
    print("[PASSED] test_format_success_score_emojis")


def test_format_success_with_boosts():
    """Тест форматирования с hierarchy и breadcrumb boosts"""
    results = [
        {
            'title': 'Test Page',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/page', 'chunk': 1},
            'text': 'Sample text',
            'rerank_score': 0.7,
            'final_score': 0.85,
            'hierarchy_boost': 0.1,
            'breadcrumb_boost': 0.05
        }
    ]
    
    response = ResponseFormatter.format_success(
        query="test query",
        results=results
    )
    
    assert "0.850" in response, "Финальный score должен быть в ответе"
    # Проверяем что boosts отображаются (если есть)
    assert "hier:" in response or "path:" in response or "base:" in response, "Boosts должны быть в ответе"
    
    print("[PASSED] test_format_success_with_boosts")


def test_format_success_with_metadata():
    """Тест форматирования с дополнительными метаданными"""
    results = [
        {
            'title': 'Test Page',
            'metadata': {
                'space': 'TEST',
                'url': 'https://test.com/page',
                'chunk': 1,
                'labels': 'important,urgent',
                'created_by': 'admin',
                'attachments': 'file1.pdf,file2.docx'
            },
            'text': 'Sample text',
            'rerank_score': 0.75,
            'final_score': 0.75
        }
    ]
    
    response = ResponseFormatter.format_success(
        query="test query",
        results=results
    )
    
    # Проверяем что метаданные отображаются (если есть)
    assert "TEST" in response, "Пространство должно быть в ответе"
    
    print("[PASSED] test_format_success_with_metadata")


def test_format_success_empty_results():
    """Тест форматирования с пустым списком результатов"""
    response = ResponseFormatter.format_success(
        query="test query",
        results=[]
    )
    
    assert "test query" in response, "Запрос должен быть в ответе"
    assert "Results: 0" in response, "Количество результатов должно быть 0"
    
    print("[PASSED] test_format_success_empty_results")


def test_format_no_results_basic():
    """Тест базового форматирования пустого ответа"""
    response = ResponseFormatter.format_no_results(
        query="test query"
    )
    
    assert "test query" in response, "Запрос должен быть в ответе"
    assert "No Results Found" in response or "Ничего не найдено" in response or "No Results" in response, \
        "Должно быть сообщение об отсутствии результатов"
    
    print("[PASSED] test_format_no_results_basic")


def test_format_no_results_with_intent():
    """Тест форматирования пустого ответа с intent"""
    intent = {'type': 'exploratory', 'diversity': 2}
    
    response = ResponseFormatter.format_no_results(
        query="test query",
        intent=intent
    )
    
    assert "exploratory" in response or "Query Type: exploratory" in response, \
        "Тип intent должен быть в ответе"
    
    print("[PASSED] test_format_no_results_with_intent")


def test_format_no_results_with_suggestions():
    """Тест форматирования пустого ответа с предложениями"""
    suggestions = [
        "Попробуйте другой запрос",
        "Используйте другие ключевые слова"
    ]
    
    response = ResponseFormatter.format_no_results(
        query="test query",
        suggestions=suggestions
    )
    
    assert "Попробуйте другой запрос" in response or "Try" in response, \
        "Предложения должны быть в ответе"
    
    print("[PASSED] test_format_no_results_with_suggestions")


def test_format_no_results_with_threshold():
    """Тест форматирования пустого ответа с threshold"""
    response = ResponseFormatter.format_no_results(
        query="test query",
        threshold=0.001,
        vector_count=10,
        bm25_count=5
    )
    
    assert "0.001" in response or "threshold" in response.lower(), \
        "Threshold должен быть в ответе (если указан)"
    
    print("[PASSED] test_format_no_results_with_threshold")


def test_format_error_basic():
    """Тест базового форматирования ошибки"""
    error = ValueError("Test error message")
    
    response = ResponseFormatter.format_error(
        query="test query",
        error=error
    )
    
    assert "test query" in response, "Запрос должен быть в ответе"
    assert "Test error message" in response or "error" in response.lower(), \
        "Сообщение об ошибке должно быть в ответе"
    
    print("[PASSED] test_format_error_basic")


def test_format_error_with_suggestions():
    """Тест форматирования ошибки с предложениями"""
    error = RuntimeError("Connection failed")
    suggestions = [
        "Проверьте подключение",
        "Попробуйте еще раз"
    ]
    
    response = ResponseFormatter.format_error(
        query="test query",
        error=error,
        suggestions=suggestions
    )
    
    assert "Connection failed" in response or "error" in response.lower(), \
        "Сообщение об ошибке должно быть в ответе"
    assert "Проверьте подключение" in response or "Try" in response, \
        "Предложения должны быть в ответе"
    
    print("[PASSED] test_format_error_with_suggestions")


def test_format_low_relevance_basic():
    """Тест базового форматирования низкой релевантности"""
    response = ResponseFormatter.format_low_relevance(
        query="test query",
        threshold=0.001
    )
    
    assert "test query" in response, "Запрос должен быть в ответе"
    assert "0.001" in response, "Threshold должен быть в ответе"
    assert "Low Relevance" in response or "low relevance" in response.lower() or \
           "низкой релевантности" in response.lower(), \
        "Должно быть сообщение о низкой релевантности"
    
    print("[PASSED] test_format_low_relevance_basic")


def test_format_low_relevance_with_scores():
    """Тест форматирования низкой релевантности с scores"""
    response = ResponseFormatter.format_low_relevance(
        query="test query",
        threshold=0.001,
        min_score=0.0001,
        max_score=0.0005
    )
    
    assert "0.001" in response, "Threshold должен быть в ответе"
    assert "0.0001" in response or "0.0005" in response, \
        "Scores должны быть в ответе (если указаны)"
    
    print("[PASSED] test_format_low_relevance_with_scores")


def test_format_low_relevance_with_intent():
    """Тест форматирования низкой релевантности с intent"""
    intent = {'type': 'exploratory', 'diversity': 2}
    
    response = ResponseFormatter.format_low_relevance(
        query="test query",
        threshold=0.001,
        intent=intent
    )
    
    assert "exploratory" in response or "Query Type: exploratory" in response, \
        "Тип intent должен быть в ответе"
    
    print("[PASSED] test_format_low_relevance_with_intent")


def test_format_success_safe_getters():
    """Тест безопасных геттеров для отсутствующих полей"""
    # Результат с минимальными данными
    results = [
        {
            'text': 'Sample text',
            'rerank_score': 0.5,
            'final_score': 0.5
        }
    ]
    
    response = ResponseFormatter.format_success(
        query="test query",
        results=results
    )
    
    # Должен обработать без ошибок даже без metadata
    assert "test query" in response, "Запрос должен быть в ответе"
    assert "Sample text" in response, "Текст должен быть в ответе"
    
    print("[PASSED] test_format_success_safe_getters")


def test_format_success_with_breadcrumb():
    """Тест форматирования с breadcrumb"""
    results = [
        {
            'breadcrumb': 'Parent > Child > Page',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/page', 'chunk': 1},
            'text': 'Sample text',
            'rerank_score': 0.7,
            'final_score': 0.7
        }
    ]
    
    response = ResponseFormatter.format_success(
        query="test query",
        results=results
    )
    
    # Breadcrumb должен использоваться как title если title отсутствует
    assert "Parent" in response or "Child" in response or "Page" in response, \
        "Breadcrumb должен быть в ответе"
    
    print("[PASSED] test_format_success_with_breadcrumb")


def test_format_success_text_preview():
    """Тест обрезки длинного текста"""
    long_text = "A" * 1000  # Очень длинный текст
    results = [
        {
            'title': 'Test Page',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/page', 'chunk': 1},
            'text': long_text,
            'rerank_score': 0.7,
            'final_score': 0.7
        }
    ]
    
    response = ResponseFormatter.format_success(
        query="test query",
        results=results
    )
    
    # Текст должен быть обрезан до 500 символов
    assert len(response) < len(long_text) + 500, "Текст должен быть обрезан"
    
    print("[PASSED] test_format_success_text_preview")


def test_format_success_with_context_chunks():
    """Тест форматирования с context_chunks"""
    results = [
        {
            'title': 'Test Page',
            'metadata': {'space': 'TEST', 'url': 'https://test.com/page', 'chunk': 1},
            'text': 'Sample text',
            'rerank_score': 0.7,
            'final_score': 0.7,
            'context_chunks': 3
        }
    ]
    
    response = ResponseFormatter.format_success(
        query="test query",
        results=results
    )
    
    assert "3 chunks" in response or "chunks" in response, \
        "Количество chunks должно быть в ответе (если > 1)"
    
    print("[PASSED] test_format_success_with_context_chunks")


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 70)
    print("ТЕСТЫ ДЛЯ response_formatter")
    print("=" * 70)
    
    tests = [
        test_format_success_basic,
        test_format_success_with_intent,
        test_format_success_with_latency,
        test_format_success_multiple_results,
        test_format_success_score_emojis,
        test_format_success_with_boosts,
        test_format_success_with_metadata,
        test_format_success_empty_results,
        test_format_no_results_basic,
        test_format_no_results_with_intent,
        test_format_no_results_with_suggestions,
        test_format_no_results_with_threshold,
        test_format_error_basic,
        test_format_error_with_suggestions,
        test_format_low_relevance_basic,
        test_format_low_relevance_with_scores,
        test_format_low_relevance_with_intent,
        test_format_success_safe_getters,
        test_format_success_with_breadcrumb,
        test_format_success_text_preview,
        test_format_success_with_context_chunks,
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 70)
    print(f"ИТОГО: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

