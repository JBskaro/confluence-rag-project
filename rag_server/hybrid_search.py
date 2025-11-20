#!/usr/bin/env python3
"""
Hybrid Search: Vector + BM25 (full-text search)
Объединение результатов через Reciprocal Rank Fusion (RRF)

Это стандартный подход, используемый Google, OpenAI, Meta, Microsoft, AWS.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from enum import Enum

# Импортируем функцию для получения документов из Qdrant
try:
    from qdrant_storage import get_all_points
except ImportError:
    # Fallback для запуска из другой директории
    from rag_server.qdrant_storage import get_all_points

logger = logging.getLogger(__name__)

# Observability imports
from observability import tracer, BM25_LATENCY, RRF_LATENCY, timed_operation


class QueryIntent(Enum):
    """Тип запроса для адаптивных весов"""
    NAVIGATIONAL = "navigational"  # Поиск конкретной информации
    EXPLORATORY = "exploratory"    # Исследовательский поиск
    FACTUAL = "factual"            # Фактический поиск
    HOWTO = "howto"                # Инструкции (как сделать)


# Конфигурация через ENV
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))  # Вес векторного поиска (по умолчанию)
BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))      # Вес BM25 поиска (по умолчанию)
RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))                     # Параметр RRF (стандарт: 60)

# Адаптивные веса для разных типов запросов
VECTOR_WEIGHT_NAVIGATIONAL = float(os.getenv("HYBRID_VECTOR_WEIGHT_NAVIGATIONAL", "0.7"))
BM25_WEIGHT_NAVIGATIONAL = float(os.getenv("HYBRID_BM25_WEIGHT_NAVIGATIONAL", "0.3"))

VECTOR_WEIGHT_EXPLORATORY = float(os.getenv("HYBRID_VECTOR_WEIGHT_EXPLORATORY", "0.5"))
BM25_WEIGHT_EXPLORATORY = float(os.getenv("HYBRID_BM25_WEIGHT_EXPLORATORY", "0.5"))

VECTOR_WEIGHT_FACTUAL = float(os.getenv("HYBRID_VECTOR_WEIGHT_FACTUAL", "0.6"))
BM25_WEIGHT_FACTUAL = float(os.getenv("HYBRID_BM25_WEIGHT_FACTUAL", "0.4"))

VECTOR_WEIGHT_HOWTO = float(os.getenv("HYBRID_VECTOR_WEIGHT_HOWTO", "0.55"))
BM25_WEIGHT_HOWTO = float(os.getenv("HYBRID_BM25_WEIGHT_HOWTO", "0.45"))

# Глобальный BM25 индекс (ленивая инициализация)
bm25_index = None
bm25_corpus = []  # Список документов (текстов)
bm25_nodes = []   # Список соответствующих nodes (метаданные)


def simple_tokenize(text: str) -> List[str]:
    """Простая токенизация для BM25"""
    return text.lower().split()


# Ключевые слова для определения интента (в порядке приоритета)
INTENT_KEYWORDS = {
    QueryIntent.NAVIGATIONAL: ['где', 'url', 'ссылка', 'link', 'find', 'где найти', 'найди', 'покажи', 'страница', 'документ'],
    QueryIntent.HOWTO: ['как', 'инструкция', 'настроить', 'установить', 'запустить', 'сделать'],
    QueryIntent.FACTUAL: ['какой', 'какая', 'какие', 'что', 'когда', 'кто', 'сколько'],
    QueryIntent.EXPLORATORY: ['какие', 'сравни', 'список', 'все', 'перечисли']
}


def detect_query_intent(query: str) -> QueryIntent:
    """
    Определить тип запроса для адаптивных весов.

    Простые эвристики на основе ключевых слов.

    Args:
        query: Поисковый запрос

    Returns:
        QueryIntent enum
    """
    query_lower = query.lower()

    # Проверяем в порядке приоритета
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return intent

    # По умолчанию: Factual
    return QueryIntent.FACTUAL


def get_adaptive_weights(query_intent: QueryIntent) -> Tuple[float, float]:
    """
    Получить адаптивные веса для Hybrid Search на основе типа запроса.

    Args:
        query_intent: Тип запроса

    Returns:
        Tuple[vector_weight, bm25_weight]
    """
    weights = {
        QueryIntent.NAVIGATIONAL: (VECTOR_WEIGHT_NAVIGATIONAL, BM25_WEIGHT_NAVIGATIONAL),
        QueryIntent.EXPLORATORY: (VECTOR_WEIGHT_EXPLORATORY, BM25_WEIGHT_EXPLORATORY),
        QueryIntent.FACTUAL: (VECTOR_WEIGHT_FACTUAL, BM25_WEIGHT_FACTUAL),
        QueryIntent.HOWTO: (VECTOR_WEIGHT_HOWTO, BM25_WEIGHT_HOWTO),
    }

    vector_weight, bm25_weight = weights.get(query_intent, (VECTOR_WEIGHT, BM25_WEIGHT))

    # Нормализуем веса (сумма должна быть ~1.0)
    total = vector_weight + bm25_weight
    if total > 0:
        vector_weight = vector_weight / total
        bm25_weight = bm25_weight / total

    return vector_weight, bm25_weight


def _load_documents_from_qdrant(collection_name: str, limit: int) -> dict:
    """Загружает документы из Qdrant."""
    # Ограничиваем для предотвращения OOM
    MAX_DOCS = int(os.getenv("BM25_MAX_DOCS", "50000"))
    actual_limit = min(limit, MAX_DOCS)

    target_collection = collection_name or os.getenv("QDRANT_COLLECTION", "confluence_chunks")
    logger.info(f"Загрузка документов для BM25 из коллекции: {target_collection} (limit={actual_limit})")

    all_data = get_all_points(limit=actual_limit)

    if not all_data or not all_data.get('ids'):
        raise ValueError("Коллекция пуста или данные не получены")

    return all_data


def _prepare_bm25_corpus(all_data: dict) -> Tuple[List, List]:
    """Подготавливает корпус и nodes для BM25."""
    corpus_tokens = []
    nodes = []
    doc_count = len(all_data['ids'])

    logger.info(f"Подготовка {doc_count} документов для BM25...")

    for idx in range(doc_count):
        doc_id = all_data['ids'][idx]
        doc_text = all_data['documents'][idx]
        doc_metadata = all_data['metadatas'][idx]

        if doc_text and isinstance(doc_text, str):
            tokens = simple_tokenize(doc_text)
            corpus_tokens.append(tokens)
            # Сохраняем структуру, похожую на PointStruct для совместимости
            nodes.append({
                'id': doc_id,
                'payload': doc_metadata or {},
                'text': doc_text
            })

    if not corpus_tokens:
        raise ValueError("Не найдено текстового контента для индексации")

    return corpus_tokens, nodes


def _create_bm25_index(corpus_tokens: List) -> 'BM25Okapi':
    """Создает BM25 индекс."""
    from rank_bm25 import BM25Okapi
    with timed_operation(BM25_LATENCY):
        return BM25Okapi(corpus_tokens)


def init_bm25_retriever(collection_name: str = None) -> bool:
    """
    Инициализирует BM25 index из документов Qdrant используя rank_bm25.

    Args:
        collection_name: Имя коллекции Qdrant

    Returns:
        True если успешно, False если ошибка
    """
    global bm25_index, bm25_corpus, bm25_nodes

    if not ENABLE_HYBRID_SEARCH:
        logger.info("Hybrid Search отключен (ENABLE_HYBRID_SEARCH=false)")
        return False

    if bm25_index is not None:
        return True

    try:
        # 1. Загружаем документы
        all_data = _load_documents_from_qdrant(collection_name, 50000)

        # 2. Подготавливаем корпус
        corpus_tokens, nodes = _prepare_bm25_corpus(all_data)

        # 3. Создаем индекс
        bm25_index = _create_bm25_index(corpus_tokens)
        bm25_corpus = corpus_tokens
        bm25_nodes = nodes

        logger.info(f"✅ BM25 индекс создан. Индексировано {len(nodes)} документов.")
        return True

    except ImportError:
        logger.warning("rank_bm25 не установлен. Установите: pip install rank_bm25")
        return False
    except Exception as e:
        logger.error(f"Ошибка инициализации BM25: {e}", exc_info=True)
        return False


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = RRF_K,
    vector_weight: float = None,
    bm25_weight: float = None
) -> List[Dict[str, Any]]:
    """
    Объединяет результаты векторного и BM25 поиска через Reciprocal Rank Fusion (RRF).

    RRF Score = SUM(weight * (1 / (k + rank))) для каждого результата

    Args:
        vector_results: Результаты векторного поиска (с полем 'distance')
        bm25_results: Результаты BM25 поиска
        k: Параметр RRF (стандарт: 60)
        vector_weight: Вес векторного поиска (по умолчанию из ENV)
        bm25_weight: Вес BM25 поиска (по умолчанию из ENV)

    Returns:
        Объединенный и отсортированный список результатов
    """
    if not vector_results and not bm25_results:
        return []

    # Используем переданные веса или значения по умолчанию
    if vector_weight is None:
        vector_weight = VECTOR_WEIGHT
    if bm25_weight is None:
        bm25_weight = BM25_WEIGHT

    # Словарь для накопления RRF scores
    rrf_scores = defaultdict(lambda: {
        'id': None,
        'text': '',
        'metadata': {},
        'vector_rank': None,
        'bm25_rank': None,
        'rrf_score': 0.0
    })

    # Обрабатываем результаты векторного поиска
    for rank, result in enumerate(vector_results, start=1):
        doc_id = result.get('id')
        if doc_id:
            rrf_scores[doc_id]['id'] = doc_id
            rrf_scores[doc_id]['text'] = result.get('text', '')
            rrf_scores[doc_id]['metadata'] = result.get('metadata', {})
            rrf_scores[doc_id]['vector_rank'] = rank
            # RRF формула: weight * (1 / (k + rank))
            rrf_scores[doc_id]['rrf_score'] += vector_weight * (1.0 / (k + rank))

    # Обрабатываем результаты BM25 поиска
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result.get('id')
        if doc_id:
            if doc_id not in rrf_scores:
                # Новый результат из BM25
                rrf_scores[doc_id] = {
                    'id': doc_id,
                    'text': result.get('text', ''),
                    'metadata': result.get('payload', {}),  # payload = metadata
                    'vector_rank': None,
                    'bm25_rank': rank,
                    'rrf_score': 0.0
                }
            else:
                rrf_scores[doc_id]['bm25_rank'] = rank

            # RRF формула: weight * (1 / (k + rank))
            rrf_scores[doc_id]['rrf_score'] += bm25_weight * (1.0 / (k + rank))

    # Сортируем по RRF score (убывание)
    merged_results = sorted(
        rrf_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )

    # Преобразуем в формат, совместимый с существующим кодом
    formatted_results = []
    for result in merged_results:
        formatted_results.append({
            'id': result['id'],
            'text': result['text'],
            'metadata': result['metadata'],
            'distance': 1.0 - result['rrf_score'],  # Инвертируем для совместимости (меньше = лучше)
            'rrf_score': result['rrf_score'],
            'vector_rank': result['vector_rank'],
            'bm25_rank': result['bm25_rank']
        })

    return formatted_results


def hybrid_search(
    query: str,
    collection_name: str,
    vector_results: List[Dict[str, Any]],
    space_filter: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Выполняет Hybrid Search: объединяет Vector + BM25 результаты через RRF.
    """
    if not ENABLE_HYBRID_SEARCH:
        return vector_results[:limit]

    # Определяем интент запроса для весов
    query_intent = detect_query_intent(query)
    vector_weight, bm25_weight = get_adaptive_weights(query_intent)

    # Если BM25 вес 0, возвращаем только векторный поиск
    if bm25_weight <= 0.01:
        return vector_results[:limit]

    try:
        # Инициализируем BM25 если еще нет (ленивая загрузка)
        if bm25_index is None:
            success = init_bm25_retriever(collection_name)
            if not success:
                return vector_results[:limit]

        with tracer.start_as_current_span("hybrid_search_bm25"):
            # 1. BM25 Поиск
            tokenized_query = simple_tokenize(query)

            # Получаем топ-N документов по BM25
            # Берем больше кандидатов (limit * 2) для лучшего пересечения
            bm25_limit = limit * 3

            # get_top_n возвращает список документов, но нам нужны индексы или scores
            # rank_bm25 не имеет метода get_top_n_with_scores, делаем сами
            with timed_operation(BM25_LATENCY):
                doc_scores = bm25_index.get_scores(tokenized_query)

            # Сортируем индексы по убыванию score
            top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:bm25_limit]

            bm25_results = []
            for idx in top_indices:
                score = doc_scores[idx]
                if score <= 0:
                    continue

                point = bm25_nodes[idx]
                # Преобразуем point в формат результата (как vector_results)
                result = {
                    'id': point['id'],
                    'score': score,  # BM25 score
                    'payload': point['payload'],
                    'text': point['text']
                }
                bm25_results.append(result)

            # 2. Объединение через RRF
            with timed_operation(RRF_LATENCY):
                merged_results = reciprocal_rank_fusion(
                    vector_results,
                    bm25_results,
                    k=RRF_K,
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight
                )

        logger.info(f"Hybrid Search ({query_intent.value}): Vector={len(vector_results)}, BM25={len(bm25_results)}, Merged={len(merged_results)}")

        return merged_results[:limit]

    except Exception as e:
        logger.warning(f"Ошибка BM25 поиска: {e}, возвращаю только векторные результаты")
        return vector_results[:limit]


def get_hybrid_search_stats() -> Dict[str, Any]:
    """
    Возвращает статистику Hybrid Search.

    Returns:
        Словарь со статистикой
    """
    return {
        'enabled': ENABLE_HYBRID_SEARCH,
        'vector_weight': VECTOR_WEIGHT,
        'bm25_weight': BM25_WEIGHT,
        'rrf_k': RRF_K,
        'bm25_initialized': bm25_index is not None,
        'bm25_documents': len(bm25_corpus) if bm25_corpus else 0
    }
