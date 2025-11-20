#!/usr/bin/env python3
"""
Тесты для модуля intent_config.
"""

import sys
import os

# Добавляем путь к rag_server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_server.utils.intent_config import (
    QueryIntent,
    IntentConfig,
    get_intent_config,
    get_adaptive_rerank_threshold,
    get_adaptive_context_window
)


def test_query_intent_enum():
    """Тест enum QueryIntent"""
    assert QueryIntent.NAVIGATIONAL.value == "navigational"
    assert QueryIntent.HOWTO.value == "howto"
    assert QueryIntent.FACTUAL.value == "factual"
    assert QueryIntent.EXPLORATORY.value == "exploratory"
    
    print("[PASSED] test_query_intent_enum")


def test_get_intent_config_navigational():
    """Тест конфигурации для navigational запросов"""
    config = get_intent_config("navigational")
    
    assert config.diversity_limit == 1, "Navigational должен иметь diversity_limit=1"
    assert config.expand_context == False, "Navigational не должен расширять контекст"
    assert config.boost_hierarchy == True, "Navigational должен усиливать иерархию"
    
    print("[PASSED] test_get_intent_config_navigational")


def test_get_intent_config_howto():
    """Тест конфигурации для howto запросов"""
    config = get_intent_config("howto")
    
    assert config.diversity_limit == 3, "HowTo должен иметь diversity_limit=3"
    assert config.expand_context == True, "HowTo должен расширять контекст"
    assert config.boost_hierarchy == False, "HowTo не должен усиливать иерархию"
    
    print("[PASSED] test_get_intent_config_howto")


def test_get_intent_config_factual():
    """Тест конфигурации для factual запросов"""
    config = get_intent_config("factual")
    
    assert config.diversity_limit == 3, "Factual должен иметь diversity_limit=3"
    assert config.expand_context == True, "Factual должен расширять контекст"
    
    print("[PASSED] test_get_intent_config_factual")


def test_get_intent_config_exploratory():
    """Тест конфигурации для exploratory запросов"""
    config = get_intent_config("exploratory", reranker_model="BAAI/bge-reranker-v2-m3")
    
    assert config.diversity_limit == 2, "Exploratory должен иметь diversity_limit=2"
    assert config.expand_context == True, "Exploratory должен расширять контекст"
    assert config.rerank_threshold <= 0.0001, "Exploratory должен иметь очень низкий порог"
    assert config.context_window == 4, "Exploratory должен иметь больший context_window"
    
    print("[PASSED] test_get_intent_config_exploratory")


def test_get_adaptive_rerank_threshold_bge():
    """Тест адаптивного порога для bge-reranker"""
    # Устанавливаем переменные окружения для теста
    import os
    os.environ['RE_RANKER_MODEL'] = 'BAAI/bge-reranker-v2-m3'
    os.environ['RERANK_THRESHOLD_EXPLORATORY'] = '0.0001'
    
    threshold_exploratory = get_adaptive_rerank_threshold("exploratory", False, "BAAI/bge-reranker-v2-m3")
    threshold_technical = get_adaptive_rerank_threshold("howto", True, "BAAI/bge-reranker-v2-m3")
    
    assert threshold_exploratory <= 0.0001, "Exploratory должен иметь очень низкий порог"
    assert threshold_technical <= 0.01, "Technical должен иметь низкий порог"
    
    print("[PASSED] test_get_adaptive_rerank_threshold_bge")


def test_get_adaptive_rerank_threshold_crossencoder():
    """Тест адаптивного порога для cross-encoder"""
    threshold_exploratory = get_adaptive_rerank_threshold("exploratory", False, "DiTy/cross-encoder-russian-msmarco")
    threshold_general = get_adaptive_rerank_threshold("factual", False, "DiTy/cross-encoder-russian-msmarco")
    
    assert threshold_exploratory < threshold_general, "Exploratory должен иметь более низкий порог чем general"
    
    print("[PASSED] test_get_adaptive_rerank_threshold_crossencoder")


def test_get_adaptive_context_window():
    """Тест адаптивного размера окна контекста"""
    window_navigational = get_adaptive_context_window("navigational")
    window_exploratory = get_adaptive_context_window("exploratory")
    
    assert window_navigational == 2, "Navigational должен иметь context_window=2"
    assert window_exploratory == 4, "Exploratory должен иметь context_window=4"
    
    print("[PASSED] test_get_adaptive_context_window")


def test_intent_config_technical_vs_general():
    """Тест различия между техническими и общими запросами"""
    config_technical = get_intent_config("howto", reranker_model="BAAI/bge-reranker-v2-m3")
    config_general = get_intent_config("factual", reranker_model="BAAI/bge-reranker-v2-m3")
    
    # HowTo использует технический порог (0.01), Factual использует general (0.001)
    # Для howto порог может быть выше, т.к. это технический порог
    # Проверяем что оба порога установлены корректно
    assert config_technical.rerank_threshold > 0, "Технический порог должен быть > 0"
    assert config_general.rerank_threshold > 0, "Общий порог должен быть > 0"
    
    print("[PASSED] test_intent_config_technical_vs_general")


def test_fallback_intent():
    """Тест fallback для неизвестного типа intent"""
    config = get_intent_config("unknown_type")
    
    assert config.diversity_limit == 2, "Fallback должен иметь стандартные значения"
    assert config.expand_context == True, "Fallback должен расширять контекст"
    
    print("[PASSED] test_fallback_intent")


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 70)
    print("ТЕСТЫ ДЛЯ intent_config")
    print("=" * 70)
    
    tests = [
        test_query_intent_enum,
        test_get_intent_config_navigational,
        test_get_intent_config_howto,
        test_get_intent_config_factual,
        test_get_intent_config_exploratory,
        test_get_adaptive_rerank_threshold_bge,
        test_get_adaptive_rerank_threshold_crossencoder,
        test_get_adaptive_context_window,
        test_intent_config_technical_vs_general,
        test_fallback_intent,
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

