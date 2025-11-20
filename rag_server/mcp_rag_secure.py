"""
MCP сервер для семантического поиска по Confluence.
Предоставляет инструменты для Open WebUI через Model Context Protocol.
"""
from typing import Any, List, Dict
import logging
import os
import re
import sys
import time

from fastmcp import FastMCP
from qdrant_client import QdrantClient

# Настройка логирования из ENV
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Отключаем избыточное логирование HTTP запросов от httpx/openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "confluence")
# Импортируем унифицированный модуль embeddings
from embeddings import (
    get_embedding_dimension,
    EMBED_MODEL,
    USE_OLLAMA
)

# Импортируем новые модули для продвинутого поиска
from synonyms_manager import get_synonyms_manager

from advanced_search import extract_keywords
from query_rewriter import cached_rewrite_query, get_rewriter_stats
from observability import setup_observability
from hybrid_search import init_bm25_retriever

# Глобальная переменная для reranker (ленивая инициализация)
reranker = None


def init_reranker():
    """
    Инициализация CrossEncoder для reranking.

    Модель инициализируется один раз при первом вызове и кэшируется глобально.
    Это предотвращает повторную загрузку модели (≈30-40 сек) при каждом запросе.

    По умолчанию используется Russian MS-MARCO для оптимальной работы с русским языком.
    Модель можно изменить через переменную окружения RE_RANKER_MODEL.
    """
    global reranker
    if reranker is None:
        try:
            start_time = time.time()
            from sentence_transformers import CrossEncoder

            # Получаем модель из ENV или используем дефолт (Russian MS-MARCO)
            model_name = os.getenv(
                'RE_RANKER_MODEL',
                'DiTy/cross-encoder-russian-msmarco'  # По умолчанию Russian MS-MARCO
            )

            logger.info(f"Инициализация CrossEncoder для reranking...")
            logger.info(f"  Модель: {model_name}")

            # Информация о популярных моделях
            model_info = {
                'BAAI/bge-reranker-v2-m3': {
                    'name': 'BAAI bge-reranker-v2-m3',
                    'language': '100+ языков (включая русский)',
                    'quality': '95%+ для многоязычных запросов'
                },
                'DiTy/cross-encoder-russian-msmarco': {
                    'name': 'Russian MS-MARCO',
                    'language': 'Русский',
                    'quality': '92% для русского'
                },
                'cross-encoder/ms-marco-MiniLM-L-6-v2': {
                    'name': 'MS-MARCO MiniLM',
                    'language': 'Английский (универсальная)',
                    'quality': '85% для русского'
                },
                'Qwen/Qwen3-Reranker-8B': {
                    'name': 'Qwen3 Reranker',
                    'language': '100+ языков',
                    'quality': '95% (требует больше ресурсов)'
                }
            }

            info = model_info.get(model_name, {})
            if info:
                logger.info(f"  Название: {info.get('name', model_name)}")
                logger.info(f"  Язык: {info.get('language', 'N/A')}")
                logger.info(f"  Качество: {info.get('quality', 'N/A')}")

            reranker = CrossEncoder(model_name)
            elapsed = time.time() - start_time
            logger.info(f"✅ CrossEncoder инициализирован за {elapsed:.1f}с. Модель кэширована.")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать reranker: {e}")
            reranker = None
    else:
        logger.debug("Переиспользование кэшированного CrossEncoder")
    return reranker

def _get_max_variants(query: str) -> int:
    """Определяет максимальное количество вариантов расширения."""
    query_length = len(query.split())
    if query_length <= 2:
        return 5
    elif query_length <= 4:
        return 3
    return 2

def _expand_with_semantic_log(query: str, current_queries: list, max_variants: int):
    """Источник 1: Semantic Query Log."""
    if len(current_queries) >= max_variants:
        return

    try:
        from semantic_query_log import get_semantic_query_log
        semantic_log = get_semantic_query_log()
        related_queries = semantic_log.get_related_queries(query, top_n=3)

        for related in related_queries:
            if related not in current_queries:
                current_queries.append(related)
                logger.debug(f"Semantic Query Log: добавлен похожий запрос '{related}'")
                if len(current_queries) >= max_variants:
                    break
    except Exception as e:
        logger.debug(f"Semantic Query Log недоступен: {e}")

def _expand_with_synonyms(query: str, current_queries: list, max_variants: int):
    """Источник 2-4: SynonymsManager."""
    if len(current_queries) >= max_variants:
        return

    try:
        synonyms_manager = get_synonyms_manager()
        from synonyms_manager import TERM_BLACKLIST

        keywords = extract_keywords(query)
        query_lower = query.lower().strip()

        for keyword in keywords[:3]:
            if len(current_queries) >= max_variants:
                break

            keyword_lower = keyword.lower()
            if keyword_lower in TERM_BLACKLIST:
                continue

            synonyms = synonyms_manager.get_synonyms(keyword, max_synonyms=2)
            if not synonyms:
                continue

            for synonym in synonyms:
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                expanded = re.sub(pattern, synonym.lower(), query_lower, flags=re.IGNORECASE)

                if expanded != query_lower and expanded not in current_queries:
                    current_queries.append(expanded)
                    if len(current_queries) >= max_variants:
                        break
    except Exception as e:
        logger.warning(f"Ошибка при расширении запроса через SynonymsManager: {e}")

def _expand_with_rewriting(query: str, current_queries: list, max_variants: int):
    """Источник 5: Query Rewriting."""
    if len(current_queries) >= max_variants:
        return

    try:
        # Передаем None для semantic_log, так как он используется внутри cached_rewrite_query опционально
        rewrite_variants = cached_rewrite_query(query, semantic_log=None)
        for variant in rewrite_variants[1:]:
            if variant not in current_queries:
                current_queries.append(variant)
                logger.debug(f"Query rewriting variant: {variant}")
                if len(current_queries) >= max_variants:
                    break
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}")

def expand_query(query: str, space: str = "") -> list[str]:
    """
    Умное расширение запроса с использованием множественных источников синонимов.
    """
    queries = [query]
    max_variants = _get_max_variants(query)

    # Источник 1: Semantic Query Log
    _expand_with_semantic_log(query, queries, max_variants)

    # Источник 2-4: SynonymsManager
    _expand_with_synonyms(query, queries, max_variants)

    # Источник 5: Query Rewriting
    _expand_with_rewriting(query, queries, max_variants)

    # Дополнительная обработка (стоп-слова, space, 1С)
    keywords = extract_keywords(query)
    if len(keywords) >= 2:
        clean_query = ' '.join(keywords)
        if clean_query not in queries:
            queries.append(clean_query)

    query_lower = query.lower()
    if space and len(query_lower.split()) <= 5:
        queries.append(f"{query} {space}")

    if any(term in query_lower for term in ['1с', '1c', 'конфигурация']):
        normalized = query.replace('1С', '1C').replace('1с', '1c')
        if normalized != query and normalized not in queries:
            queries.append(normalized)

    # Итоговая дедупликация и обрезка
    result = list(dict.fromkeys(queries))[:max_variants]

    if len(result) < len(queries):
        logger.debug(
            f"Query expansion ограничен: {len(queries)} -> {len(result)} "
            f"вариантов (len={len(query.split())})"
        )

    return result

def calculate_optimal_candidate_limit(query: str, limit: int) -> int:
    """
    Вычисляет оптимальное количество кандидатов для reranking.

    Args:
        query: Поисковый запрос
        limit: Желаемое количество финальных результатов

    Returns:
        Оптимальное количество кандидатов
    """
    query_words = len(query.split())

    if query_words <= 2:
        multiplier = 5  # Короткий запрос → больше шума, нужно больше кандидатов
    elif query_words <= 4:
        multiplier = 3  # Средний запрос
    else:
        multiplier = 2  # Длинный запрос → уже специфичный

    return min(limit * multiplier, 50)  # Максимум 50 кандидатов

def detect_content_type(text: str) -> str:
    """
    Определяет тип контента в тексте.

    Returns:
        'table' | 'list' | 'code' | 'plain'
    """
    import re

    # Таблицы: | col1 | col2 | или строки с табуляцией
    if re.search(r'\|.*\|.*\|', text) or text.count('\t') > 5:
        return 'table'

    # Списки: 3+ строк начинающихся с *, -, •, цифр
    list_lines = re.findall(r'^\s*[\*\-•][\s\)]|^\s*\d+[\.\)]', text, re.MULTILINE)
    if len(list_lines) >= 3:
        return 'list'

    # Код: ```code``` или 5+ строк с отступами
    if '```' in text or len(re.findall(r'^\s{4,}', text, re.MULTILINE)) >= 5:
        return 'code'

    return 'plain'

def format_search_results(results: List[Dict[str, Any]], query: str, limit: int) -> str:  # noqa: C901
    """
    Форматирует результаты поиска в читаемый текст.

    Args:
        results: Список результатов поиска
        query: Поисковый запрос
        limit: Максимальное количество результатов

    Returns:
        Форматированная строка с результатами
    """
    if not results:
        return f"❌ Ничего не найдено по запросу: '{query}'"

    response = [f"✅ Найдено {len(results)} результатов:\n"]

    for i, r in enumerate(results[:limit], 1):
        if not r or not isinstance(r, dict):
            continue

        m = r.get('metadata', {})
        if not isinstance(m, dict):
            m = {}

        page_space = m.get('space', 'Unknown')
        page_url = m.get('url', '')
        # Breadcrumb может быть в разных местах
        breadcrumb = (r.get('breadcrumb') or
                     m.get('page_path') or
                     m.get('title') or
                     'Без названия')
        chunk_num = m.get('chunk', 0)

        # Текст результата
        text = r.get('expanded_text', r.get('text', "[Текст недоступен]"))
        text_preview = extract_relevant_snippet(text, query, max_length=800)

        # Score информация
        final_score = r.get('boosted_score', r.get('rerank_score', r.get('final_score', 0)))
        score_emoji = "🔥" if final_score > 0.5 else "⭐" if final_score > 0.3 else "✓" if final_score > 0.1 else "·"
        score_str = f"{score_emoji} {final_score:.3f}"

        # Дополнительная информация
        extra_info = []
        labels = m.get('labels', '')
        if labels:
            extra_info.append(f"🏷️ {labels}")
        created_by = m.get('created_by', '')
        if created_by:
            extra_info.append(f"👤 {created_by}")

        extra_str = " | ".join(extra_info)
        if extra_str:
            extra_str = f" | {extra_str}"

        # Тип поиска
        search_type = r.get('search_type', 'semantic')
        search_type_str = "🔍 structural" if search_type == 'structural' else "🔎 semantic"

        response.append(
            f"[{i}] {search_type_str} 📍 {breadcrumb}\n"
            f"    📁 {page_space} | Chunk #{chunk_num} | {score_str}{extra_str}\n"
            f"    🔗 {page_url}\n"
            f"    💬 {text_preview}\n"
        )

    return "\n".join(response)

def extract_relevant_snippet(text: str, query: str, max_length: int = 400) -> str:  # noqa: C901
    """
    Извлекает наиболее релевантный фрагмент текста относительно запроса.
    Умеет распознавать списки, таблицы и показывать их полностью.

    Args:
        text: Исходный текст
        query: Поисковый запрос
        max_length: Максимальная длина фрагмента

    Returns:
        Релевантный фрагмент текста
    """
    if len(text) <= max_length:
        return text

    import re

    # НОВОЕ: Определяем тип контента
    content_type = detect_content_type(text)

    # Таблицы и списки НЕ обрезаем - показываем полностью (до разумного лимита)
    if content_type in ['table', 'list']:
        limit = max_length * 6  # До 2400 символов для структурированного контента
        if len(text) <= limit:
            return text
        else:
            # Обрезаем, но пытаемся сохранить структуру
            return text[:limit] + "\n... (обрезано)"

    # Для кода - увеличенный лимит
    if content_type == 'code':
        limit = max_length * 3  # До 1200 символов
        if len(text) <= limit:
            return text
        else:
            return text[:limit] + "\n... (обрезано)"

    # Для обычного текста - стандартная логика
    # Разбиваем на предложения (по точкам, вопросам, восклицаниям)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:max_length] + "..."

    # Ключевые слова из запроса
    query_words = set(extract_keywords(query))

    # Находим предложение с максимальным overlap
    best_idx = 0
    best_score = 0

    for idx, sent in enumerate(sentences):
        sent_words = set(extract_keywords(sent))
        overlap = len(query_words & sent_words)

        if overlap > best_score:
            best_score = overlap
            best_idx = idx

    # Берем предложение + контекст (1 до, 1 после)
    start = max(0, best_idx - 1)
    end = min(len(sentences), best_idx + 2)
    snippet = '. '.join(sentences[start:end]).strip()

    # Ограничиваем длину
    if len(snippet) > max_length:
        snippet = snippet[:max_length] + "..."

    return snippet

def deduplicate_results(results: list) -> list:
    """
    Удаляет дубликаты результатов на основе схожести текста.

    Args:
        results: Список результатов

    Returns:
        Список уникальных результатов
    """
    if len(results) <= 1:
        return results

    # Дедупликация по hash первых 200 символов
    seen_signatures = set()
    unique_results = []

    for r in results:
        # Берем первые 200 символов как сигнатуру
        text = r.get('text', '')
        signature = text[:200].strip()
        text_hash = hash(signature)

        if text_hash not in seen_signatures:
            seen_signatures.add(text_hash)
            unique_results.append(r)
        else:
            logger.debug(f"Удален дубликат: {r['metadata'].get('title', 'Unknown')}")

    return unique_results

def expand_context_window(result: dict, window_size: int = 1) -> dict:
    """
    Расширяет контекст найденного чанка соседними чанками.

    Context Window Retrieval - популярная техника из LangChain и LlamaIndex.
    Находим маленький релевантный чанк, но возвращаем больший контекст.

    Args:
        result: Найденный результат
        window_size: Количество чанков до/после (1 = ±1 чанк)

    Returns:
        Результат с расширенным контекстом
    """
    global qdrant_client

    if qdrant_client is None:
        return result

    try:
        if not result or not isinstance(result, dict):
            return result

        metadata = result.get('metadata')
        if not metadata or not isinstance(metadata, dict):
            return result

        chunk_num = metadata.get('chunk', 0)
        page_id = metadata.get('page_id')

        if not page_id:
            return result

        # Запрашиваем соседние чанки из той же страницы
        min_chunk = max(0, chunk_num - window_size)
        max_chunk = chunk_num + window_size

        # Получаем чанки в диапазоне
        from qdrant_storage import get_points_by_filter
        neighbors_raw = get_points_by_filter(
            where_filter={
                '$and': [
                    {'page_id': page_id},
                    {'chunk': {'$gte': min_chunk}},
                    {'chunk': {'$lte': max_chunk}}
                ]
            },
            limit=100,
            collection=QDRANT_COLLECTION
        )
        neighbors = {
            'documents': [r.get('text', '') for r in neighbors_raw],
            'metadatas': [r.get('metadata', {}) for r in neighbors_raw]
        }

        # Безопасная проверка: neighbors может быть None или содержать None поля
        if (neighbors and
            neighbors.get('documents') and
            neighbors.get('metadatas') and
            len(neighbors['documents']) == len(neighbors['metadatas'])):
            # Сортируем по chunk_num
            chunk_data = []
            for i, doc in enumerate(neighbors['documents']):
                if i < len(neighbors['metadatas']):
                    chunk_meta = neighbors['metadatas'][i]
                    if chunk_meta and isinstance(chunk_meta, dict):
                        chunk_data.append({
                            'chunk_num': chunk_meta.get('chunk', 0),
                            'text': doc if doc else ''
                        })

            chunk_data.sort(key=lambda x: x['chunk_num'])

            # Объединяем тексты
            expanded_text = '\n\n'.join([c['text'] for c in chunk_data])
            result['expanded_text'] = expanded_text
            result['context_chunks'] = len(chunk_data)

            logger.debug(f"Context expanded: chunk {chunk_num} + {len(chunk_data)-1} neighbors")
        else:
            result['expanded_text'] = result.get('text', '')
            result['context_chunks'] = 1

    except Exception as e:
        logger.warning(f"Context expansion failed: {e}")

def calculate_hierarchy_boost(metadata: dict) -> float:
    """
    Hierarchy Boost - техника из Elasticsearch и Pinecone для учета
    важности документов на основе их положения в структуре.

    Args:
        metadata: Метаданные результата

    Returns:
        Буст от -0.2 до +0.8
    """
    boost = 0.0

    # 1. Корневая страница в space (нет родителя) - самые важные
    if not metadata.get('parent_title'):
        boost += 0.5
        logger.debug(f"Root page boost: +0.5 for {metadata.get('title')}")

    # 2. Ключевые слова в названии страницы
    title = metadata.get('title', '').lower()
    important_keywords = {
        'общая информация': 0.3,
        'главная': 0.3,
        'readme': 0.3,
        'getting started': 0.3,
        'начало работы': 0.3,
        'обзор': 0.2,
        'документация': 0.2,
        'руководство': 0.2,
    }

    for keyword, value in important_keywords.items():
        if keyword in title:
            boost += value
            logger.debug(f"Title keyword boost: +{value} for '{keyword}'")
            break  # Только один буст за title

    # 3. Уровень заголовка (h1 важнее h4)
    heading_level = metadata.get('heading_level', 0)
    if heading_level == 1:
        boost += 0.2
    elif heading_level == 2:
        boost += 0.1

    # 4. Наличие меток (labeled pages обычно важнее)
    labels_str = metadata.get('labels', '').lower()
    if labels_str:
        # УЛУЧШЕНИЕ: Metadata Boosting - дополнительный буст для технических меток
        technical_labels = ['api', 'technical', 'архитектура', 'development',
                           'разработка', 'интеграция', 'configuration', 'настройка']

        has_technical_label = any(label in labels_str for label in technical_labels)
        if has_technical_label:
            boost += 0.3  # Увеличенный буст для технических страниц
            logger.debug(f"Technical label boost: +0.3 for labels '{labels_str}'")
        else:
            boost += 0.05  # Базовый буст за наличие меток

    return min(boost, 0.8)  # Ограничиваем максимум

def calculate_breadcrumb_match_score(query: str, breadcrumb: str) -> float:
    """
    Вычисляет совпадение запроса с breadcrumb (путем страницы).

    Path Matching - техника из semantic search для учета совпадения
    запроса с иерархией документа.

    Args:
        query: Поисковый запрос
        breadcrumb: Путь страницы (Space > Parent > Page > Section)

    Returns:
        Score от 0.0 до 1.0
    """
    if not breadcrumb:
        return 0.0

    query_words = set(extract_keywords(query))
    breadcrumb_words = set(extract_keywords(breadcrumb))

    if not query_words or not breadcrumb_words:
        return 0.0

    # Jaccard similarity
    intersection = len(query_words & breadcrumb_words)
    union = len(query_words | breadcrumb_words)

    score = intersection / union if union > 0 else 0.0

    if score > 0:
        logger.debug(f"Breadcrumb match: {score:.2f} ({intersection}/{union} words)")

    return score

# ========================================
# СТРУКТУРНЫЙ ПОИСК (Structural Navigation Search)
# ========================================

# Глобальный кэш для метаданных
_metadata_cache = {}
_cache_timestamp = 0
_cache_ttl = 3600  # 1 час

# Глобальный кэш для структурных результатов
_structural_cache = {}
_structural_cache_timestamp = {}


def get_all_metadata_cached(ttl_seconds: int = 3600) -> Dict[str, Any]:
    """
    Кэшировать метаданные для анализа запросов.

    Args:
        ttl_seconds: Время жизни кэша в секундах

    Returns:
        Словарь с метаданными всех документов
    """
    global _metadata_cache, _cache_timestamp, _cache_ttl

    current_time = time.time()

    # Проверяем кэш
    if (_metadata_cache and
        current_time - _cache_timestamp < ttl_seconds):
        logger.debug(f"✅ Metadata cache hit: {len(_metadata_cache.get('ids', []))} items")
        return _metadata_cache

    # Обновляем кэш
    logger.info("📊 Обновление metadata cache...")
    try:
        from qdrant_storage import get_all_points
        all_points = get_all_points(limit=10000, include_payload=True)
        # Преобразуем в ChromaDB формат
        all_data = {
            'ids': [p.get('id', '') for p in all_points.get('points', [])],
            'metadatas': [p.get('metadata', {}) for p in all_points.get('points', [])]
        }
        _metadata_cache = all_data
        _cache_timestamp = current_time
        _cache_ttl = ttl_seconds

        logger.info(f"✅ Metadata cache updated: {len(all_data.get('ids', []))} items")
        return all_data
    except Exception as e:
        logger.warning(f"Ошибка обновления metadata cache: {e}")
        return _metadata_cache if _metadata_cache else {}

STRUCTURAL_SEPARATORS = ['>', '→', '→', ' / ', ' | ']
STRUCTURAL_PATTERNS = [
    (r'по\s+блоку\s+(\w+)(?:\s*,\s*а\s+точнее\s+)?([^\.]+)?', True),
    (r'(\w+)\s*,\s*а\s+точнее\s+([^\.]+)', True),
    (r'по\s+блоку\s+(\w+)', False),
    (r'в\s+разделе\s+(\w+)', False),
    (r'на\s+странице\s+(\w+)', False),
]

def _parse_with_separators(query: str) -> list[str]:
    """Парсинг по разделителям."""
    for sep in STRUCTURAL_SEPARATORS:
        if sep in query:
            return [p.strip() for p in query.split(sep) if p.strip()]
    return []

def _parse_with_regex(query_lower: str) -> list[str]:
    """Парсинг по регулярным выражениям."""
    for pattern, extract_multi in STRUCTURAL_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            groups = [g.strip() for g in match.groups() if g and g.strip()]

            if extract_multi:
                 if len(groups) >= 2:
                     return groups
                 elif len(groups) == 1:
                     # Дополнительная проверка на "а точнее" если в основной регулярке не поймали
                     after_match = re.search(r'а\s+точнее\s+([^\.]+)', query_lower)
                     if after_match:
                         return [groups[0], after_match.group(1).strip()]
                     return groups
            else:
                 return groups
    return []

def parse_query_structure(query: str) -> Dict[str, Any]:
    """
    Парсит структурные компоненты запроса.

    Определяет является ли запрос структурным (с разделителями >, >)
    и извлекает части запроса.

    Примеры:
    - "Склад > Учет номенклатуры" > structural
    - "технологический стек RAUII" > обычный
    - "Обследование > Склад > Учет номенклатуры" > structural

    Args:
        query: Исходный запрос

    Returns:
        Словарь с информацией о структуре запроса
    """
    query_lower = query.lower().strip()

    # 1. Проверка разделителей
    parts = _parse_with_separators(query)
    is_structural = bool(parts)

    # 2. Проверка регулярок если не нашли разделители
    if not is_structural:
        regex_parts = _parse_with_regex(query_lower)
        if regex_parts:
            parts = regex_parts
            is_structural = True

    result = {
        'is_structural_query': is_structural,
        'parts': parts if parts else [query],
        'original_query': query,
        'query_lower': query_lower
    }

    logger.debug(f"🔍 Query structure: is_structural={is_structural}, parts={result['parts']}")

    return result


def _calculate_structural_match(parts: list, metadata: dict) -> tuple[float, list]:
    """Вычисляет скор совпадения структуры для одного документа."""
    # Конфигурация полей и их весов
    FIELD_WEIGHTS = [
        ('page_path', 3.0),
        ('title', 2.0),
        ('heading_path', 1.5),
        ('heading', 1.0),
    ]

    match_score = 0.0
    matches = []

    # Предварительная нормализация
    fields = {
        field: (metadata.get(field, '') or '').lower()
        for field, _ in FIELD_WEIGHTS
    }

    for part_idx, part in enumerate(parts):
        part_lower = part.lower()
        position_weight = len(parts) - part_idx
        matched = False

        for field_name, base_weight in FIELD_WEIGHTS:
            if part_lower in fields[field_name]:
                match_score += base_weight + position_weight
                matches.append({
                    'part': part,
                    'field': field_name,
                    'weight': position_weight
                })
                matched = True
                break

        if not matched:
            return 0.0, []  # Требуем совпадения всех частей

    return match_score, matches

def structural_metadata_search(
    collection: Any,
    structure: Dict[str, Any],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Поиск по метаданным на основе структуры запроса.

    Использует in-memory фильтрацию (ChromaDB не поддерживает $contains напрямую).
    Ищет совпадения в:
    - page_path (полный путь страницы)
    - title (заголовок страницы)
    - heading_path (путь заголовков)
    - heading (заголовок раздела)
    - parent_path (путь родителя)

    Args:
        collection: ChromaDB коллекция
        structure: Результат parse_query_structure()
        limit: Максимальное количество результатов

    Returns:
        Список результатов с match_score
    """
    if not structure['is_structural_query']:
        return []

    parts = structure['parts']

    # ============ УЛУЧШЕНИЕ: Детальное логирование с метриками ============
    search_start = time.time()
    logger.info(f"🔍 Начало структурного поиска: parts={parts}, limit={limit}")

    try:
        # Получаем ВСЕ результаты (с разумным лимитом)
        max_scan = min(limit * 10, 10000)  # Не более 10K для производительности

        fetch_start = time.time()

        # Проверяем кэш структурных запросов
        query_key = structure.get('original_query', '')
        current_time = time.time()

        if (query_key in _structural_cache and
            current_time - _structural_cache_timestamp.get(query_key, 0) < 3600):
            logger.debug(f"✅ Structural cache hit for '{query_key}'")
            return _structural_cache[query_key]

        # Получаем все метаданные (кэшированные)
        all_data = get_all_metadata_cached()

        if not all_data or not all_data.get('metadatas'):
            return []

        parts = structure.get('parts', [])
        if not parts:
            return []

        logger.info(f"🏗️ Structural search for parts: {parts}")

        formatted_results = []
        matched_count = 0

        # Проходим по всем документам
        for idx, metadata in enumerate(all_data['metadatas']):
            if matched_count >= limit:
                break

            if not metadata:
                continue

            match_score, matches = _calculate_structural_match(parts, metadata)

            if match_score > 0:
                matched_count += 1
                formatted_results.append({
                    'metadata': metadata,
                    'match_score': match_score,
                    'matches': matches,
                    'text': ''
                })

        # Сортируем по score
        formatted_results.sort(key=lambda x: x['match_score'], reverse=True)

        total_time = time.time() - search_start
        fetch_time = time.time() - fetch_start
        filter_time = total_time - fetch_time
        checked_count = len(all_data.get('metadatas', []))

        logger.info(
            f"✅ Structural search finished: {len(formatted_results)} results "
            f"(проверено {checked_count}, совпадений {matched_count}) "
            f"за {total_time:.3f}с "
            f"(fetch: {fetch_time:.3f}с, filter: {filter_time:.3f}с)"
        )

        if formatted_results and logger.isEnabledFor(logging.DEBUG):
            # Логируем топ-3 для отладки
            for i, r in enumerate(formatted_results[:3], 1):
                logger.debug(
                    f"  [{i}] match_score={r['match_score']}, "
                    f"page_id={r['metadata'].get('page_id')}, "
                    f"title={r['metadata'].get('title', '')[:50]}, "
                    f"matches={len(r.get('matches', []))}"
                )

        return formatted_results[:limit]

    except Exception as e:
        total_time = time.time() - search_start
        logger.error(f"Ошибка структурного поиска (за {total_time:.3f}с): {e}", exc_info=True)
        return []

def cached_structural_search(
    collection: Any,
    structure: Dict[str, Any],
    limit: int = 100,
    ttl_seconds: int = 300  # 5 минут
) -> List[Dict[str, Any]]:
    """
    Кэширование результатов структурного поиска.

    Args:
        collection: ChromaDB коллекция
        structure: Результат parse_query_structure()
        limit: Максимальное количество результатов
        ttl_seconds: Время жизни кэша в секундах

    Returns:
        Список результатов структурного поиска
    """
    cache_key = tuple(sorted(structure['parts']))
    current_time = time.time()

    # Проверяем кэш
    if (cache_key in _structural_cache and
        cache_key in _structural_cache_timestamp and
        current_time - _structural_cache_timestamp[cache_key] < ttl_seconds):

        cache_age = current_time - _structural_cache_timestamp[cache_key]
        logger.debug(
            f"✅ Structural cache hit: {cache_key} "
            f"(age: {cache_age:.1f}с, results: {len(_structural_cache[cache_key])})"
        )
        return _structural_cache[cache_key]

    # Выполняем поиск
    search_start = time.time()
    results = structural_metadata_search(collection, structure, limit)
    search_time = time.time() - search_start

    # Сохраняем в кэш
    _structural_cache[cache_key] = results
    _structural_cache_timestamp[cache_key] = current_time

    logger.info(
        f"📝 Structural cache updated: {cache_key} → {len(results)} results "
        f"(search time: {search_time:.3f}с)"
    )

    return results

def _find_best_keyword_match(text: str, keywords: list) -> tuple[str, float]:
    """Найти лучшее совпадение ключевого слова в тексте."""
    if not text:
        return "", 0.0

    text_lower = text.lower()
    for keyword in keywords:
        if len(keyword) > 3 and keyword in text_lower:
            score = len(keyword) / len(text_lower)
            return keyword, score
    return "", 0.0

def analyze_query_with_metadata(
    query: str
) -> Dict[str, Any]:
    """
    Анализирует запрос и находит совпадения в метаданных.

    Использует кэшированные метаданные для производительности.

    Args:
        query: Поисковый запрос

    Returns:
        Словарь с совпадениями в метаданных
    """
    keywords = extract_keywords(query)

    # Получаем кэшированные метаданные
    all_data = get_all_metadata_cached()

    if not all_data or not all_data.get('metadatas'):
        return {'page_title_matches': [], 'heading_path_matches': [], 'page_path_matches': []}

    matches = {
        'page_title_matches': [],
        'heading_path_matches': [],
        'page_path_matches': []
    }
    seen_pages = set()

    for idx, metadata in enumerate(all_data['metadatas']):
        if not metadata:
            continue

        page_id = metadata.get('page_id')
        if not page_id:
            continue

        # Проверка title (только уникальные страницы)
        if page_id not in seen_pages:
            title = metadata.get('title', '')
            kw, score = _find_best_keyword_match(title, keywords)
            if score > 0:
                matches['page_title_matches'].append({
                    'page_id': page_id,
                    'page_title': title,
                    'page_path': metadata.get('page_path', ''),
                    'match_keyword': kw,
                    'match_score': score
                })
                seen_pages.add(page_id)

        # Проверка page_path
        page_path = metadata.get('page_path', '')
        kw, score = _find_best_keyword_match(page_path, keywords)
        if score > 0:
            matches['page_path_matches'].append({
                'page_id': page_id,
                'page_path': page_path,
                'match_keyword': kw,
                'match_score': score
            })

        # Проверка heading_path
        heading_path = metadata.get('heading_path', '')
        kw, score = _find_best_keyword_match(heading_path, keywords)
        if score > 0:
            matches['heading_path_matches'].append({
                'page_id': page_id,
                'heading_path': heading_path,
                'match_keyword': kw,
                'match_score': score
            })

    # Сортируем по match_score и обрезаем
    for key in matches:
        matches[key].sort(key=lambda x: x['match_score'], reverse=True)
        matches[key] = matches[key][:10]

    return matches

def apply_metadata_boost(
    results: List[Dict[str, Any]],
    metadata_analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Применить boost с нормализацией (не более 30% от текущего score).

    Args:
        results: Список результатов поиска
        metadata_analysis: Результат analyze_query_with_metadata()

    Returns:
        Список результатов с примененным boost
    """
    if not results:
        return results

    for result in results:
        result['metadata_boost'] = 0.0
        page_id = result.get('metadata', {}).get('page_id')

        if not page_id:
            continue

        # Boost за совпадение page_title
        for match in metadata_analysis.get('page_title_matches', [])[:3]:
            if match['page_id'] == page_id:
                # Boost не более 30% от текущего score
                current_score = result.get('rerank_score', 0)
                boost = current_score * 0.3 * match['match_score']
                result['metadata_boost'] += boost
                logger.debug(f"Page title boost: +{boost:.3f} for {page_id}")
                break

        # Boost за совпадение heading_path
        heading_path = result.get('metadata', {}).get('heading_path', '')
        if heading_path:
            for match in metadata_analysis.get('heading_path_matches', [])[:3]:
                if (match['page_id'] == page_id and
                    match['heading_path'].lower() in heading_path.lower()):
                    current_score = result.get('rerank_score', 0)
                    boost = current_score * 0.2 * match['match_score']
                    result['metadata_boost'] += boost
                    logger.debug(f"Heading path boost: +{boost:.3f} for {page_id}")
                    break

        # Обновляем финальный score
        result['boosted_score'] = result.get('rerank_score', 0) + result['metadata_boost']

    return results

def get_diversity_limit_for_intent(intent_type: str = None) -> int:
    """
    Получить лимит diversity filter для типа запроса.

    Args:
        intent_type: Тип запроса ('navigational', 'exploratory', 'factual', 'howto')

    Returns:
        Максимальное количество chunks с одной страницы
    """
    # Проверяем включён ли фильтр
    enable_filter = os.getenv('ENABLE_DIVERSITY_FILTER', 'true').lower() == 'true'
    if not enable_filter:
        return 999  # Очень большой лимит (эффективно отключает фильтр)

    # Получаем лимиты из ENV или используем дефолты
    diversity_limits = {
        'navigational': int(os.getenv('DIVERSITY_LIMIT_NAVIGATIONAL', '1')),
        'exploratory': int(os.getenv('DIVERSITY_LIMIT_EXPLORATORY', '4')),
        'factual': int(os.getenv('DIVERSITY_LIMIT_FACTUAL', '2')),
        'howto': int(os.getenv('DIVERSITY_LIMIT_HOWTO', '3')),
    }

    # Если тип не указан или неизвестен, используем дефолт для factual
    if not intent_type or intent_type not in diversity_limits:
        intent_type = 'factual'

    return diversity_limits.get(intent_type, 2)

def _resolve_diversity_limit(max_per_page, query, intent) -> int:
    """Определяет лимит документов с одной страницы на основе интента."""
    if max_per_page is not None:
        return max_per_page

    intent_type = None
    if intent and isinstance(intent, dict):
        intent_type = intent.get('type')
    elif query:
        intent_dict = classify_query_intent(query)
        intent_type = intent_dict.get('type') if intent_dict else None

    limit = get_diversity_limit_for_intent(intent_type)
    logger.debug(f"Diversity filter: автоматический лимит {limit} для intent={intent_type}")
    return limit

def apply_diversity_filter(results: list, limit: int = 5, max_per_page: int = None, query: str = None, intent: dict = None) -> list:
    """
    Применяет diversity filter: ограничивает количество chunks с одной страницы.

    Это стандартная практика (Google, Perplexity): максимум 2-3 результата
    с одного источника в топ-5, остальные места - другие источники.

    НОВОЕ: Адаптивные лимиты на основе query intent через ENV переменные.

    Args:
        results: Список результатов отсортированных по score
        limit: Сколько результатов вернуть
        max_per_page: Максимум chunks с одной страницы (если None, определяется по intent)
        query: Поисковый запрос (для определения intent, опционально)
        intent: Словарь с информацией о типе запроса (опционально)

    Returns:
        Отфильтрованные результаты с разнообразием источников
    """
    if not results:
        return []

    limit_per_page = _resolve_diversity_limit(max_per_page, query, intent)

    filtered_results = []
    page_counts = {}

    for result in results:
        if not result or not isinstance(result, dict):
            continue

        metadata = result.get('metadata')
        if not metadata or not isinstance(metadata, dict):
            continue

        page_id = metadata.get('page_id')

        # Если страницы нет или лимит не превышен - добавляем
        if not page_id or page_counts.get(page_id, 0) < limit_per_page:
            filtered_results.append(result)
            if page_id:
                page_counts[page_id] = page_counts.get(page_id, 0) + 1

            # Достигли нужного количества результатов
            if len(filtered_results) >= limit:
                break

    # Логирование для анализа
    if page_counts:
        logger.debug(f"Diversity filter: {len(filtered_results)} results from {len(page_counts)} unique pages (max {limit_per_page}/page)")
        for page_id, count in page_counts.items():
            if count > 1:
                logger.debug(f"  Page {page_id}: {count} chunks")

    return filtered_results

def enrich_result_with_context(result: dict) -> dict:
    """
    Обогащает результат контекстом из родительских заголовков.

    Args:
        result: Результат поиска

    Returns:
        Обогащенный результат с breadcrumb
    """
    if not result or not isinstance(result, dict):
        return result

    metadata = result.get('metadata')
    if not metadata or not isinstance(metadata, dict):
        # Если metadata отсутствует, создаем пустой словарь
        metadata = {}
        result['metadata'] = metadata

    # Формируем breadcrumb (хлебные крошки)
    # НОВОЕ: Используем page_path если есть (полный путь)
    page_path = metadata.get('page_path', '')
    if page_path:
        # page_path уже содержит полный путь: "Parent 1 > Parent 2 > Page"
        breadcrumb_parts = page_path.split(' > ')
    else:
        # Fallback: формируем из parent_title (старый метод)
        breadcrumb_parts = []
        if metadata.get('parent_title'):
            breadcrumb_parts.append(metadata['parent_title'])
        if metadata.get('title') and metadata.get('title') != metadata.get('parent_title'):
            breadcrumb_parts.append(metadata['title'])

    # Добавляем заголовок секции, если есть
    if metadata.get('heading'):
        breadcrumb_parts.append(metadata['heading'])

    result['breadcrumb'] = ' → '.join(breadcrumb_parts) if breadcrumb_parts else metadata.get('title', 'Unknown')

    return result

def classify_query_intent(query: str) -> dict:
    """
    Классифицирует намерение пользователя для адаптации стратегии поиска.

    Returns:
        {
            'type': 'factual' | 'navigational' | 'howto' | 'exploratory',
            'boost_hierarchy': bool,  # Усилить буст иерархии
            'expand_context': bool,   # Расширить контекст
            'diversity': int          # Лимит чанков с одной страницы
        }
    """
    import re

    query_lower = query.lower()

    # 1. Навигационные запросы: "где", "найди страницу", "покажи"
    if re.search(r'\b(где|найди|покажи|страница|документ)\b', query_lower):
        return {
            'type': 'navigational',
            'boost_hierarchy': True,   # Важны корневые страницы
            'expand_context': False,   # Не нужен полный контекст
            'diversity': 1             # По 1 чанку с страницы (показать больше страниц)
        }

    # 2. How-to запросы: "как", "инструкция", "настроить"
    if re.search(r'\b(как|инструкция|настроить|установить|запустить|сделать)\b', query_lower):
        return {
            'type': 'howto',
            'boost_hierarchy': False,  # Не важна иерархия
            'expand_context': True,    # Нужен полный контекст
            'diversity': 3             # До 3 чанков с страницы (детальная инструкция)
        }

    # 3. Фактические запросы: "какой", "что", "когда", "кто"
    if re.search(r'\b(какой|какая|какие|что|когда|кто|сколько)\b', query_lower):
        return {
            'type': 'factual',
            'boost_hierarchy': False,  # Не важна иерархия
            'expand_context': True,    # Нужен контекст для ответа
            'diversity': 3             # До 3 чанков (может быть в разных местах)
        }

    # 4. Исследовательские запросы (по умолчанию)
    return {
        'type': 'exploratory',
        'boost_hierarchy': False,
        'expand_context': True,
        'diversity': 2  # Стандартный лимит
    }

def init_rag() -> QdrantClient:
    """
    Инициализация RAG системы: Qdrant.

    Returns:
        QdrantClient

    Raises:
        Exception: Если не удалось инициализировать компоненты
    """
    try:
        logger.info(f"Инициализация Qdrant: {QDRANT_HOST}:{QDRANT_PORT}, collection={QDRANT_COLLECTION}")

        # Импортируем функции из qdrant_storage
        from qdrant_storage import init_qdrant_client, init_qdrant_collection

        # Инициализируем клиент
        client = init_qdrant_client()

        # Получаем размерность embeddings
        embedding_dim = get_embedding_dimension()

        # Инициализируем коллекцию (создает если не существует)
        success = init_qdrant_collection(embedding_dim)
        if not success:
            raise ValueError(f"Failed to initialize Qdrant collection: {QDRANT_COLLECTION}")

        # Получаем количество документов
        from qdrant_storage import get_qdrant_count
        doc_count = get_qdrant_count()

        logger.info(f"✅ RAG система готова. Документов: {doc_count}, Размерность: {embedding_dim}D")
        return client

    except Exception as e:
        logger.error(f"Критическая ошибка инициализации RAG: {e}")
        sys.exit(1)

# Глобальная переменная для RAG компонента
qdrant_client = None

# Инициализация RAG при импорте модуля (до создания MCP сервера)
# Это гарантирует, что клиент инициализируется один раз при старте
logger.info("Инициализация RAG при старте MCP сервера...")
qdrant_client = init_rag()
from qdrant_storage import get_qdrant_count
doc_count = get_qdrant_count()
logger.info(f"RAG система готова. Документов: {doc_count}")

# Инициализация BM25 retriever для Hybrid Search (ленивая инициализация)
logger.info("Инициализация BM25 retriever для Hybrid Search...")
# BM25 работает напрямую с Qdrant через qdrant_storage
init_bm25_retriever(QDRANT_COLLECTION)

# Предзагрузка reranker модели при старте (чтобы первый запрос был быстрее)
logger.info("Предзагрузка reranker модели при старте...")
try:
    init_reranker()
    logger.info("✅ Reranker модель предзагружена и готова к использованию")
except Exception as e:
    logger.warning(f"⚠️ Не удалось предзагрузить reranker модель: {e}. Модель загрузится при первом запросе.")

# Initialize SearchPipeline
from search_pipeline import SearchPipeline, SearchParams
search_pipeline = SearchPipeline(qdrant_client, QDRANT_COLLECTION, reranker)
logger.info("✅ SearchPipeline initialized")

mcp = FastMCP("Confluence RAG")

def _extract_space_from_query(query: str, current_space: str) -> tuple[str, str]:
    """Извлечь название space из текста запроса."""
    if current_space:
        return query, current_space

    space_patterns = [
        r'\bspaces?\s+([A-Za-z0-9_-]+)\s*$',
        r'\bspaces?\s+([A-Za-z0-9_-]+)(?:\s|$)',
        r'\bв\s+пространстве\s+([A-Za-z0-9_-]+)\s*$',
        r'\bпространство\s+([A-Za-z0-9_-]+)\s*$',
    ]

    for pattern in space_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            space = match.group(1).strip()
            new_query = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
            logger.info(f"Извлечен space из запроса: '{space}'")
            return new_query, space

    return query, ""

def _validate_search_params(query: str, space: str, limit: int) -> tuple[bool, str]:
    """Валидация параметров поиска."""
    if not query or not isinstance(query, str):
        return False, "❌ Ошибка: Пустой или некорректный запрос"

    if len(query.strip()) < 2:
        return False, "❌ Ошибка: Запрос слишком короткий (минимум 2 символа)"

    if space and not re.match(r'^[a-zA-Z0-9_-]+$', space.strip()):
        return False, "❌ Ошибка: Параметр space содержит недопустимые символы"

    if qdrant_client is None:
        return False, "❌ Ошибка: RAG система не инициализирована. Проверьте логи сервера."

    return True, ""

@mcp.tool()
def confluence_semantic_search(query: str, limit: int = 5, space: str = "") -> str:
    """
    Выполняет семантический поиск по базе знаний Confluence.

    Args:
        query: Поисковый запрос
        limit: Максимальное количество результатов (по умолчанию 5)
        space: Опциональный фильтр по пространству (space key)

    Returns:
        Форматированный текст с результатами поиска
    """
    try:
        # 1. Извлечение space
        query, space = _extract_space_from_query(query, space)
        query = query.strip()

        # 2. Валидация
        is_valid, error_msg = _validate_search_params(query, space, limit)
        if not is_valid:
            return error_msg

        if len(query) > 1000:
            logger.warning(f"Очень длинный запрос ({len(query)} символов), обрезаю до 1000")
            query = query[:1000]

        limit = min(max(limit, 1), 20)
        if space:
            space = space.strip()

        # 3. Structural Navigation Search
        structure = parse_query_structure(query)
        if structure['is_structural_query']:
            logger.info(f"🔍 Структурный запрос обнаружен: {structure['parts']}")
            structural_results = cached_structural_search(
                QDRANT_COLLECTION, structure, limit=limit * 10
            )
            if structural_results and len(structural_results) >= limit:
                # Применяем легкий reranking
                for result in structural_results:
                    max_match = max([r['match_score'] for r in structural_results]) if structural_results else 1
                    result['rerank_score'] = (result['match_score'] / max_match) * 0.5 if max_match > 0 else 0.1
                    result['distance'] = 1.0 - result['rerank_score']

                metadata_analysis = analyze_query_with_metadata(query)
                structural_results = apply_metadata_boost(structural_results, metadata_analysis)
                structural_results.sort(key=lambda x: x.get('boosted_score', x.get('rerank_score', 0)), reverse=True)
                return format_search_results(structural_results[:limit], query, limit)

        # 4. Standard Semantic Search Pipeline
        expanded_queries = expand_query(query, space)
        params = SearchParams(
            query=query,
            space=space if space else None,
            limit=limit,
            use_reranking=True,
            expanded_queries=expanded_queries[1:] if len(expanded_queries) > 1 else []
        )

        results = search_pipeline.execute(params)

        if not results:
            return f"❌ Ничего не найдено по запросу: '{query}'"

        return format_search_results(results, query, limit)

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}", exc_info=True)
        return f"❌ Внутренняя ошибка поиска: {str(e)}"

@mcp.tool()
def confluence_list_spaces() -> str:
    """
    Возвращает список всех доступных пространств Confluence.

    Returns:
        Форматированный список пространств с количеством документов
    """

    try:
        if qdrant_client is None:
            return "❌ Ошибка: RAG система не инициализирована."

        # Получаем все уникальные пространства из метаданных
        # Используем разумный лимит для предотвращения OOM
        MAX_SCAN_LIMIT = 10000
        from qdrant_storage import get_all_points
        all_points = get_all_points(limit=MAX_SCAN_LIMIT, include_payload=True)
        all_data = {
            'metadatas': [p.get('metadata', {}) for p in all_points.get('points', [])]
        }
        spaces = {}

        for metadata in all_data.get('metadatas', []):
            space_name = metadata.get('space', 'Unknown')
            if space_name:
                spaces[space_name] = spaces.get(space_name, 0) + 1

        if not spaces:
            return "⚠️ Не найдено пространств в индексе."

        result = "📚 Доступные пространства Confluence:\n\n"
        for space_name, count in sorted(spaces.items(), key=lambda x: x[1], reverse=True):
            result += f"  • **{space_name}**: {count} документов\n"

        scanned_count = len(all_data.get('metadatas', []))
        if scanned_count >= MAX_SCAN_LIMIT:
            result += f"\n⚠️ Показаны пространства из первых {MAX_SCAN_LIMIT} документов."

        return result
    except Exception as e:
        logger.error(f"Ошибка при получении списка пространств: {e}")
        return f"❌ Ошибка: {e}"

@mcp.tool()
def confluence_health() -> str:
    """
    Проверяет состояние RAG системы и возвращает статистику.

    Returns:
        Информация о состоянии системы и статистика
    """

    try:
        # Проверка что RAG инициализирована при старте сервера
        if qdrant_client is None:
            return "❌ Ошибка: RAG система не инициализирована. Проверьте логи сервера."

        # Подсчёт документов (используем count() для эффективности)
        try:
            from qdrant_storage import get_qdrant_count
            total_docs = get_qdrant_count()
        except Exception:
            # Fallback: оценка через ограниченную выборку
            from qdrant_storage import get_all_points
            all_points = get_all_points(limit=10, include_payload=True)
            data = {'points': all_points.get('points', [])}
            total_docs = len(data.get("ids", []))
            if total_docs == 10:
                total_docs = "10+"  # Больше 10, точное число неизвестно

        status = "✅ Система работает"
        if total_docs == 0:
            status = "⚠️ Индекс пуст - ожидание синхронизации"

        # Статистика Query Rewriting
        rewrite_stats = get_rewriter_stats()
        rewrite_info = (
            f"📝 Query Rewriting: {rewrite_stats['total_requests']} запросов, "
            f"кэш: {rewrite_stats['cache_hit_rate']} ({rewrite_stats['cache_hits']}/{rewrite_stats['total_requests']})"
        )

        mode_str = 'Ollama' if USE_OLLAMA else 'HuggingFace'

        return (
            f"{status}\n"
            f"📊 Документов в индексе: {total_docs}\n"
            f"🔧 Модель эмбеддингов: {EMBED_MODEL}\n"
            f"💾 Qdrant: {QDRANT_HOST}:{QDRANT_PORT} (Collection: {QDRANT_COLLECTION})\n"
            f"🔄 Режим: {mode_str}\n"
            f"{rewrite_info}"
        )
    except Exception as e:
        logger.error(f"Ошибка health check: {e}", exc_info=True)
        return f"❌ Ошибка: {str(e)}"

if __name__ == "__main__":
    # Setup observability
    setup_observability("confluence-rag")

    # Initialize reranker in background
    init_reranker()

    # Start MCP server
    logger.info("MCP on 0.0.0.0:8012")
    try:
        mcp.run(transport="streamable-http", port=8012, host="0.0.0.0")
    except KeyboardInterrupt:
        pass
