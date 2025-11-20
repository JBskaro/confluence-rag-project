#!/usr/bin/env python3
"""
Query Rewriting Module with Adaptive Fallback

Provides query rewriting with multiple backends:
1. Ollama (local, preferred)
2. OpenRouter (cloud, fallback)
3. No rewriting (graceful degradation)

Features:
- In-memory caching with TTL
- Automatic fallback between providers
- Examples from Semantic Query Log
- Detailed logging for debugging
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

# ========================================
# GLOBAL STATE
# ========================================

_rewrite_cache: Dict[str, tuple] = {}  # {query: (result, timestamp)}
_rewrite_stats: Dict[str, int] = {
    'total_requests': 0,
    'cache_hits': 0,
    'ollama_success': 0,
    'ollama_failed': 0,
    'openrouter_success': 0,
    'openrouter_failed': 0,
    'no_rewriting': 0,
}


# ========================================
# OLLAMA REWRITING
# ========================================

def rewrite_query_with_ollama(query: str) -> Optional[List[str]]:
    """
    Переписать запрос используя Ollama (локальный сервер)

    Args:
        query: Оригинальный запрос

    Returns:
        Список вариантов: [оригинал, вариант1, вариант2] или None если ошибка
    """

    if not os.getenv('USE_OLLAMA_FOR_QUERY_EXPANSION', 'false').lower() == 'true':
        logger.debug("Ollama rewriting is disabled")
        return None

    ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2')

    prompt = f"""Сгенерируй 2 альтернативных варианта этого поискового запроса,
используя синонимы и перефразирование.
Запросы должны быть на том же языке, что и исходный.

Исходный запрос: {query}

Варианты (только текст, без нумерации и пояснений):"""

    try:
        logger.debug(f"🔄 Ollama rewriting (model: {ollama_model})")

        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,
                "num_predict": 150,
            },
            timeout=10
        )

        if response.status_code != 200:
            logger.warning(f"⚠️ Ollama returned status {response.status_code}")
            _rewrite_stats['ollama_failed'] += 1
            return None

        result_text = response.json().get('response', '').strip()

        # Парсим результат
        variants = [query]  # Добавляем оригинальный запрос

        for line in result_text.split('\n'):
            line = line.strip()
            # Убираем нумерацию (1., 2., -, *, и т.д.)
            line = line.lstrip('0123456789.-) ')

            if line and len(line) > 5 and line not in variants:
                variants.append(line)

        logger.info(f"✅ Ollama rewriting: {len(variants)-1} variants generated")
        _rewrite_stats['ollama_success'] += 1

        return variants[:3]  # Возвращаем макс 3 варианта

    except requests.exceptions.Timeout:
        logger.warning("⚠️ Ollama timeout (10s)")
        _rewrite_stats['ollama_failed'] += 1
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("⚠️ Ollama connection error")
        _rewrite_stats['ollama_failed'] += 1
        return None
    except Exception as e:
        logger.warning(f"⚠️ Ollama rewriting failed: {e}")
        _rewrite_stats['ollama_failed'] += 1
        return None


# ========================================
# OPENROUTER REWRITING
# ========================================

def rewrite_query_with_openrouter(query: str, examples: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    Переписать запрос используя OpenRouter (облачный API)

    Args:
        query: Оригинальный запрос
        examples: Примеры успешных запросов из Semantic Query Log

    Returns:
        Список вариантов: [оригинал, вариант1, вариант2] или None если ошибка
    """

    if not os.getenv('USE_OPENROUTER_FOR_REWRITING', 'false').lower() == 'true':
        logger.debug("OpenRouter rewriting is disabled")
        return None

    api_base = os.getenv('OPENAI_API_BASE')
    api_key = os.getenv('OPENAI_API_KEY')
    # Используем отдельную модель для rewriting, если указана, иначе fallback на OPENAI_MODEL
    model = os.getenv('OPENAI_REWRITING_MODEL') or os.getenv('OPENAI_MODEL')

    if not api_base or not api_key or not model:
        logger.debug("⚠️ OpenRouter not configured (missing OPENAI_API_BASE/KEY/MODEL or OPENAI_REWRITING_MODEL)")
        return None

    # Формируем промпт с примерами если есть
    examples_text = ""
    if examples:
        examples_text = f"\n\nПримеры успешных запросов:\n" + "\n".join(f"- {ex}" for ex in examples[:3])

    prompt = f"""Сгенерируй 2 альтернативных варианта этого поискового запроса,
используя синонимы и перефразирование.

Исходный запрос: {query}{examples_text}

Варианты (только текст, без нумерации):"""

    try:
        logger.debug(f"🔄 OpenRouter rewriting (model: {model})")

        client = OpenAI(api_key=api_key, base_url=api_base)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )

        result_text = response.choices[0].message.content.strip()

        # Парсим результат
        variants = [query]

        for line in result_text.split('\n'):
            line = line.strip()
            line = line.lstrip('0123456789.-) ')

            if line and len(line) > 5 and line not in variants:
                variants.append(line)

        logger.info(f"✅ OpenRouter rewriting: {len(variants)-1} variants generated")
        _rewrite_stats['openrouter_success'] += 1

        return variants[:3]

    except Exception as e:
        logger.warning(f"⚠️ OpenRouter rewriting failed: {e}")
        _rewrite_stats['openrouter_failed'] += 1
        return None


# ========================================
# ADAPTIVE REWRITING WITH FALLBACK
# ========================================

def rewrite_query_adaptive(
    query: str,
    semantic_log: Optional[Any] = None
) -> List[str]:
    """
    Переписать запрос с выбором провайдера по QUERY_REWRITING_SOURCE

    Приоритет если QUERY_REWRITING_SOURCE не указан (legacy):
    1. Ollama (если USE_OLLAMA_FOR_QUERY_EXPANSION=true)
    2. OpenRouter (если USE_OPENROUTER_FOR_REWRITING=true)
    3. Без переписки (graceful degradation)

    Args:
        query: Оригинальный запрос
        semantic_log: SemanticQueryLog для примеров успешных запросов

    Returns:
        Список вариантов запроса
    """

    _rewrite_stats['total_requests'] += 1

    rewriting_source = os.getenv('QUERY_REWRITING_SOURCE', '').lower()

    logger.debug(f"🔄 Query rewriting: QUERY_REWRITING_SOURCE={rewriting_source or 'auto (legacy)'}")

    # Получаем примеры из Semantic Query Log если доступен
    examples = None
    if semantic_log and hasattr(semantic_log, 'get_successful_queries'):
        try:
            examples = semantic_log.get_successful_queries(limit=3)
        except Exception as e:
            logger.debug(f"Could not get examples from semantic log: {e}")

    # ========================================
    # ВАРИАНТ 1: OpenRouter (если QUERY_REWRITING_SOURCE=openrouter)
    # ========================================
    if rewriting_source == 'openrouter':
        logger.info("🔄 Query rewriting: trying OpenRouter (explicit)")
        result = rewrite_query_with_openrouter(query, examples=examples)
        if result:
            logger.info(f"✅ Query rewriting: used OpenRouter")
            return result
        else:
            logger.warning("❌ OpenRouter failed, no rewriting")
            _rewrite_stats['no_rewriting'] += 1
            return [query]

    # ========================================
    # ВАРИАНТ 2: Ollama (если QUERY_REWRITING_SOURCE=ollama)
    # ========================================
    elif rewriting_source == 'ollama':
        logger.info("🔄 Query rewriting: trying Ollama (explicit)")
        result = rewrite_query_with_ollama(query)
        if result:
            logger.info(f"✅ Query rewriting: used Ollama")
            return result
        else:
            logger.warning("❌ Ollama failed, no rewriting")
            _rewrite_stats['no_rewriting'] += 1
            return [query]

    # ========================================
    # LEGACY: Старая логика (если QUERY_REWRITING_SOURCE не указан)
    # ========================================
    elif rewriting_source == '':
        logger.info("ℹ️ QUERY_REWRITING_SOURCE not specified, using legacy logic")

        # Приоритет 1: Ollama (если USE_OLLAMA_FOR_QUERY_EXPANSION=true)
        logger.debug(f"🔄 Attempting query rewriting: '{query}'")
        result = rewrite_query_with_ollama(query)
        if result:
            logger.info(f"✅ Query rewriting: used Ollama (legacy)")
            return result

        logger.debug("⚠️ Ollama failed, trying OpenRouter")

        # Приоритет 2: OpenRouter (если USE_OPENROUTER_FOR_REWRITING=true)
        result = rewrite_query_with_openrouter(query, examples=examples)
        if result:
            logger.info(f"✅ Query rewriting: used OpenRouter (legacy)")
            return result

        logger.debug("⚠️ OpenRouter failed or disabled")

        # Приоритет 3: Без переписки
        logger.info(f"⚠️ Query rewriting: disabled (legacy, no providers available)")
        _rewrite_stats['no_rewriting'] += 1
        return [query]

    else:
        raise ValueError(
            f"Unknown QUERY_REWRITING_SOURCE: {rewriting_source}. "
            f"Use: 'openrouter', 'ollama' or leave empty for legacy logic"
        )


# ========================================
# CACHING LAYER
# ========================================

def cached_rewrite_query(
    query: str,
    semantic_log: Optional[Any] = None,
    ttl_seconds: Optional[int] = None
) -> List[str]:
    """
    Кэшированная переписка запроса

    Args:
        query: Оригинальный запрос
        semantic_log: SemanticQueryLog для примеров
        ttl_seconds: TTL кэша (по умолчанию из ENV или 3600)

    Returns:
        Список вариантов запроса
    """

    if ttl_seconds is None:
        ttl_seconds = int(os.getenv('REWRITE_CACHE_TTL', '3600'))

    current_time = time.time()

    # Проверяем кэш
    if query in _rewrite_cache:
        result, timestamp = _rewrite_cache[query]

        if current_time - timestamp < ttl_seconds:
            logger.debug(f"✅ Rewrite cache hit: {query}")
            _rewrite_stats['cache_hits'] += 1
            return result
        else:
            # Кэш устарел
            del _rewrite_cache[query]
            logger.debug(f"♻️ Rewrite cache expired: {query}")

    # Выполняем переписку
    result = rewrite_query_adaptive(query, semantic_log)

    # Сохраняем в кэш
    _rewrite_cache[query] = (result, current_time)
    logger.debug(f"📝 Rewrite cache updated: {query} (TTL: {ttl_seconds}s)")

    return result


# ========================================
# STATISTICS & DEBUGGING
# ========================================

def get_rewriter_stats() -> Dict[str, Any]:
    """
    Получить статистику Query Rewriting для отладки

    Returns:
        Словарь со статистикой
    """

    cache_size = len(_rewrite_cache)
    total = max(_rewrite_stats['total_requests'], 1)
    cache_hit_rate = (_rewrite_stats['cache_hits'] / total) * 100

    return {
        'total_requests': _rewrite_stats['total_requests'],
        'cache_hits': _rewrite_stats['cache_hits'],
        'cache_hit_rate': f"{cache_hit_rate:.1f}%",
        'cache_size': cache_size,
        'ollama_success': _rewrite_stats['ollama_success'],
        'ollama_failed': _rewrite_stats['ollama_failed'],
        'openrouter_success': _rewrite_stats['openrouter_success'],
        'openrouter_failed': _rewrite_stats['openrouter_failed'],
        'no_rewriting': _rewrite_stats['no_rewriting'],
    }


def clear_rewriter_cache() -> None:
    """Очистить кэш переписей"""
    global _rewrite_cache
    _rewrite_cache.clear()
    logger.info("✅ Rewrite cache cleared")

