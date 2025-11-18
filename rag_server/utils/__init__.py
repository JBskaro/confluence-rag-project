"""
Утилиты для RAG системы.

Модули:
- keyword_extraction: Извлечение ключевых слов из запросов и текста
- intent_config: Конфигурация для классификации намерений запросов
"""

from .keyword_extraction import (
    extract_keywords,
    extract_technical_terms,
    normalize_query,
    STOPWORDS,
    TECHNICAL_TERMS
)

from .intent_config import (
    QueryIntent,
    IntentConfig,
    get_intent_config,
    get_adaptive_rerank_threshold,
    get_adaptive_context_window
)

__all__ = [
    # keyword_extraction
    'extract_keywords',
    'extract_technical_terms',
    'normalize_query',
    'STOPWORDS',
    'TECHNICAL_TERMS',
    # intent_config
    'QueryIntent',
    'IntentConfig',
    'get_intent_config',
    'get_adaptive_rerank_threshold',
    'get_adaptive_context_window',
]

