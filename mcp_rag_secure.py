"""
MCP сервер для семантического поиска по Confluence.
Предоставляет инструменты для Open WebUI через Model Context Protocol.
"""
from typing import Optional, Tuple, Any, List, Dict
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    get_embed_model,
    generate_query_embedding,
    generate_query_embeddings_batch,
    get_embedding_dimension,
    EMBED_MODEL,
    USE_OLLAMA,
    OLLAMA_URL
)

# Импортируем новые модули для продвинутого поиска
from synonyms_manager import get_synonyms_manager
from semantic_cache import get_semantic_cache
from advanced_search import (
    pseudo_relevance_feedback,
    get_fallback_search,
    extract_keywords
)
from query_rewriter import cached_rewrite_query, get_rewriter_stats
from hybrid_search import hybrid_search, init_bm25_retriever
from context_expansion import expand_context_full

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

def extract_keywords(query: str) -> list[str]:
    """
    Извлекает ключевые слова из запроса (удаляет стоп-слова).
    
    Args:
        query: Исходный запрос
    
    Returns:
        Список ключевых слов
    """
    # Русские и английские стоп-слова
    stop_words = {
        'какой', 'какая', 'какие', 'где', 'как', 'что', 'это', 'в', 'на', 'по',
        'для', 'с', 'к', 'из', 'о', 'об', 'и', 'а', 'но', 'или', 'же',
        'the', 'is', 'at', 'which', 'on', 'in', 'a', 'an', 'and', 'or', 'but'
    }
    
    words = query.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    return keywords

def expand_query(query: str, space: str = "") -> list[str]:
    """
    Умное расширение запроса с использованием множественных источников синонимов.
    
    Источники синонимов (в порядке приоритета):
    1. НОВОЕ: Semantic Query Log (успешные запросы пользователей) ← ВЫСШИЙ ПРИОРИТЕТ!
    2. Базовый словарь (50 общих IT-терминов)
    3. Доменные термины (автоматически из Confluence)
    4. Выученные синонимы (Query Mining)
    5. Ollama (опционально, если включен)
    
    ОПТИМИЗАЦИЯ: Адаптивное количество вариантов в зависимости от длины запроса.
    
    Args:
        query: Исходный запрос
        space: Пространство для контекста (опционально)
        
    Returns:
        Список запросов (оригинал + варианты)
    """
    queries = [query]
    query_lower = query.lower().strip()
    
    # ОПТИМИЗАЦИЯ: Определяем максимальное количество вариантов по длине запроса
    query_length = len(query.split())
    if query_length <= 2:
        max_variants = 5  # Короткий запрос → больше вариантов для покрытия
    elif query_length <= 4:
        max_variants = 3  # Средний запрос → умеренное расширение
    else:
        max_variants = 2  # Длинный запрос → минимальное расширение (уже специфичен)
    
    # === ИСТОЧНИК 1 (ВЫСШИЙ ПРИОРИТЕТ): Semantic Query Log (успешные запросы пользователей) ===
    semantic_log = None  # Инициализируем для использования в Query Rewriting
    try:
        from semantic_query_log import get_semantic_query_log
        
        semantic_log = get_semantic_query_log()
        related_queries = semantic_log.get_related_queries(query, top_n=3)
        
        for related_query in related_queries:
            if related_query not in queries:
                queries.append(related_query)
                logger.debug(f"Semantic Query Log: добавлен похожий запрос '{related_query}'")
                
                if len(queries) >= max_variants:
                    break
        
        if related_queries:
            logger.debug(f"Semantic Query Log: найдено {len(related_queries)} похожих запросов")
    except Exception as e:
        logger.debug(f"Semantic Query Log недоступен: {e}")
    
    # === ИСТОЧНИК 2-4: SynonymsManager (базовый + доменные + выученные) ===
    try:
        synonyms_manager = get_synonyms_manager()
        from synonyms_manager import TERM_BLACKLIST
        
        # Извлекаем ключевые слова из запроса
        keywords = extract_keywords(query)
        
        # Для каждого ключевого слова получаем синонимы
        for keyword in keywords[:3]:  # Максимум 3 ключевых слова
            keyword_lower = keyword.lower()
            
            # ЗАЩИТА: Не заменяем blacklist термины (собственные имена, названия инструментов)
            if keyword_lower in TERM_BLACKLIST:
                logger.debug(f"Пропускаю blacklist термин: {keyword}")
                continue
            
            synonyms = synonyms_manager.get_synonyms(keyword, max_synonyms=2)
            
            if synonyms:
                # Заменяем ключевое слово на синоним с использованием word boundaries
                for synonym in synonyms:
                    # Используем регулярное выражение для точной замены целого слова
                    import re
                    pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                    expanded = re.sub(pattern, synonym.lower(), query_lower, flags=re.IGNORECASE)
                    
                    if expanded != query_lower and expanded not in queries:
                        queries.append(expanded)
                        
                        if len(queries) >= max_variants:
                            break
            
            if len(queries) >= max_variants:
                break
                
    except Exception as e:
        logger.warning(f"Ошибка при расширении запроса через SynonymsManager: {e}")
    
    # === ИСТОЧНИК 5: Query Rewriting (Ollama → OpenRouter) ===
    try:
        rewrite_variants = cached_rewrite_query(query, semantic_log=semantic_log)
        for variant in rewrite_variants[1:]:  # Пропускаем первый (оригинал)
            if variant not in queries and len(queries) < max_variants:
                queries.append(variant)
                logger.debug(f"Query rewriting variant: {variant}")
                
                if len(queries) >= max_variants:
                    break
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}")
    
    # Добавляем запрос без стоп-слов
    keywords = extract_keywords(query)
    if len(keywords) >= 2:
        clean_query = ' '.join(keywords)
        if clean_query not in queries:
            queries.append(clean_query)
    
    # Добавляем контекст пространства
    if space and len(query_lower.split()) <= 5:  # Только для коротких запросов
        queries.append(f"{query} {space}")
    
    # Специфичные для 1С/технологий термины
    if any(term in query_lower for term in ['1с', '1c', 'конфигурация']):
        normalized = query.replace('1С', '1C').replace('1с', '1c')
        if normalized != query and normalized not in queries:
            queries.append(normalized)
    
    # ОПТИМИЗАЦИЯ: Применяем адаптивный лимит
    result = list(dict.fromkeys(queries))[:max_variants]
    
    if len(result) < len(queries):
        logger.debug(f"Query expansion ограничен: {len(queries)} → {len(result)} вариантов (query_length={query_length})")
    
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


def format_search_results(results: List[Dict[str, Any]], query: str, limit: int) -> str:
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

def extract_relevant_snippet(text: str, query: str, max_length: int = 400) -> str:
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
        result['expanded_text'] = result.get('text', '')
        result['context_chunks'] = 1
    
    return result

def calculate_hierarchy_boost(metadata: dict) -> float:
    """
    Вычисляет буст на основе позиции страницы в иерархии Confluence.
    
    УЛУЧШЕНИЕ: Добавлен Metadata Boosting для технических страниц.
    
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
        breadcrumb: Путь страницы (Space → Parent → Page → Section)
        
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

def get_all_metadata_cached(collection: Any, ttl_seconds: int = 3600) -> Dict[str, Any]:
    """
    Кэшировать метаданные для анализа запросов.
    
    Args:
        collection: ChromaDB коллекция
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
        all_points = get_all_points(limit=10000, include_payload=True, collection=QDRANT_COLLECTION)
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

def parse_query_structure(query: str) -> Dict[str, Any]:
    """
    Парсит структурные компоненты запроса.
    
    Определяет является ли запрос структурным (с разделителями >, →)
    и извлекает части запроса.
    
    Примеры:
    - "Склад > Учет номенклатуры" → structural
    - "технологический стек RAUII" → обычный
    - "Обследование > Склад > Учет номенклатуры" → structural
    
    Args:
        query: Исходный запрос
    
    Returns:
        Словарь с информацией о структуре запроса
    """
    query_lower = query.lower().strip()
    
    # Проверяем наличие разделителей иерархии
    structural_separators = ['>', '→', '→', ' / ', ' | ']
    has_separator = any(sep in query for sep in structural_separators)
    
    # Проверяем паттерны структурных запросов
    # УЛУЧШЕНО: Добавлены паттерны для "по блоку X, а точнее Y"
    structural_patterns = [
        (r'по\s+блоку\s+(\w+)(?:\s*,\s*а\s+точнее\s+)?([^\.]+)?', True),  # "по блоку Склад, а точнее Учет номенклатуры"
        (r'(\w+)\s*,\s*а\s+точнее\s+([^\.]+)', True),  # "Склад, а точнее Учет номенклатуры"
        (r'по\s+блоку\s+(\w+)', False),  # "по блоку Склад"
        (r'в\s+разделе\s+(\w+)', False),  # "в разделе Учет"
        (r'на\s+странице\s+(\w+)', False),  # "на странице Склад"
    ]
    
    is_structural = has_separator
    parts = []
    
    if has_separator:
        # Разделяем по разделителям
        for sep in structural_separators:
            if sep in query:
                parts = [p.strip() for p in query.split(sep) if p.strip()]
                break
    else:
        # Проверяем паттерны (включая новые для "а точнее")
        for pattern, extract_parts in structural_patterns:
            match = re.search(pattern, query_lower)
            if match:
                is_structural = True
                if extract_parts:
                    # Извлекаем все группы из паттерна
                    groups = match.groups()
                    extracted_parts = [g.strip() for g in groups if g and g.strip()]
                    if len(extracted_parts) >= 2:
                        # Нашли паттерн "X, а точнее Y" - используем обе части
                        parts = extracted_parts
                    elif len(extracted_parts) == 1:
                        # Нашли первую часть, ищем вторую после "а точнее"
                        after_match = re.search(r'а\s+точнее\s+([^\.]+)', query_lower)
                        if after_match:
                            parts = [extracted_parts[0], after_match.group(1).strip()]
                        else:
                            parts = extracted_parts
                    break
                else:
                    # Старый паттерн - извлекаем найденную часть
                    groups = match.groups()
                    if groups:
                        parts = [g.strip() for g in groups if g and g.strip()]
                    else:
                        parts = [query]
                    break
    
    result = {
        'is_structural_query': is_structural,
        'parts': parts if parts else [query],
        'original_query': query,
        'query_lower': query_lower
    }
    
    logger.debug(f"🔍 Query structure: is_structural={is_structural}, parts={result['parts']}")
    
    return result

def structural_metadata_search(
    collection: Any,
    structure: Dict[str, Any],
    limit: int = 100
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
        from qdrant_storage import get_all_points
        all_points = get_all_points(limit=max_scan, include_payload=True, collection=QDRANT_COLLECTION)
        all_data = {
            'ids': [p.get('id', '') for p in all_points.get('points', [])],
            'documents': [p.get('text', '') for p in all_points.get('points', [])],
            'metadatas': [p.get('metadata', {}) for p in all_points.get('points', [])]
        }
        fetch_time = time.time() - fetch_start
        
        logger.debug(f"📊 Получено данных: {len(all_data.get('ids', []))} документов за {fetch_time:.3f}с")
        
        if not all_data or not all_data.get('ids'):
            logger.debug("Нет данных для структурного поиска")
            return []
        
        formatted_results = []
        
        # Фильтруем в памяти
        filter_start = time.time()
        checked_count = 0
        matched_count = 0
        
        for idx, doc_id in enumerate(all_data['ids']):
            checked_count += 1
            metadata = all_data['metadatas'][idx] if all_data.get('metadatas') else {}
            
            if not metadata:
                continue
            
            # Проверяем совпадение с частями запроса
            match_score = 0
            matches = []
            
            for part_idx, part in enumerate(parts):
                part_lower = part.lower().strip()
                if not part_lower or len(part_lower) < 2:
                    continue
                
                # Проверяем все метаданные поля
                page_path = (metadata.get('page_path', '') or '').lower()
                page_title = (metadata.get('title', '') or '').lower()
                heading_path = (metadata.get('heading_path', '') or '').lower()
                heading = (metadata.get('heading', '') or '').lower()
                parent_path = (metadata.get('parent_path', '') or '').lower()
                
                # Ищем совпадение (точное или частичное)
                matched_field = None
                if part_lower in page_path:
                    matched_field = 'page_path'
                elif part_lower in page_title:
                    matched_field = 'title'
                elif part_lower in heading_path:
                    matched_field = 'heading_path'
                elif part_lower in heading:
                    matched_field = 'heading'
                elif part_lower in parent_path:
                    matched_field = 'parent_path'
                
                if matched_field:
                    # Более высокий вес для ранних частей (space > page > section)
                    weight = len(parts) - part_idx
                    match_score += weight
                    matches.append({
                        'part': part,
                        'field': matched_field,
                        'weight': weight
                    })
            
            if match_score > 0:
                matched_count += 1
                formatted_results.append({
                    'id': doc_id,
                    'text': all_data['documents'][idx] if all_data.get('documents') else '',
                    'metadata': metadata,
                    'distance': 0.0,
                    'search_type': 'structural',
                    'match_score': match_score,
                    'matches': matches  # Для отладки
                })
        
        filter_time = time.time() - filter_start
        total_time = time.time() - search_start
        
        # Сортируем по match_score
        formatted_results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # ============ ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ============
        logger.info(
            f"✅ Структурный поиск завершен: "
            f"найдено {len(formatted_results)} результатов "
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

def analyze_query_with_metadata(
    query: str,
    collection: Any
) -> Dict[str, Any]:
    """
    Анализирует запрос и находит совпадения в метаданных.
    
    Использует кэшированные метаданные для производительности.
    
    Args:
        query: Поисковый запрос
        collection: ChromaDB коллекция
    
    Returns:
        Словарь с совпадениями в метаданных
    """
    query_lower = query.lower()
    keywords = extract_keywords(query)
    
    # Получаем кэшированные метаданные
    all_data = get_all_metadata_cached(collection)
    
    if not all_data or not all_data.get('metadatas'):
        return {'page_title_matches': [], 'heading_path_matches': [], 'page_path_matches': []}
    
    page_title_matches = []
    heading_path_matches = []
    page_path_matches = []
    
    seen_pages = set()
    
    for idx, metadata in enumerate(all_data['metadatas']):
        if not metadata:
            continue
        
        page_id = metadata.get('page_id')
        if not page_id or page_id in seen_pages:
            continue
        
        # Проверяем совпадения в page_title
        page_title = (metadata.get('title', '') or '').lower()
        if page_title:
            for keyword in keywords:
                if len(keyword) > 3 and keyword in page_title:
                    page_title_matches.append({
                        'page_id': page_id,
                        'page_title': metadata.get('title', ''),
                        'page_path': metadata.get('page_path', ''),
                        'match_keyword': keyword,
                        'match_score': len(keyword) / len(page_title) if page_title else 0
                    })
                    seen_pages.add(page_id)
                    break
        
        # Проверяем совпадения в page_path
        page_path = (metadata.get('page_path', '') or '').lower()
        if page_path:
            for keyword in keywords:
                if len(keyword) > 3 and keyword in page_path:
                    page_path_matches.append({
                        'page_id': page_id,
                        'page_path': metadata.get('page_path', ''),
                        'match_keyword': keyword,
                        'match_score': len(keyword) / len(page_path) if page_path else 0
                    })
                    break
        
        # Проверяем совпадения в heading_path
        heading_path = (metadata.get('heading_path', '') or '').lower()
        if heading_path:
            for keyword in keywords:
                if len(keyword) > 3 and keyword in heading_path:
                    heading_path_matches.append({
                        'page_id': page_id,
                        'heading_path': metadata.get('heading_path', ''),
                        'match_keyword': keyword,
                        'match_score': len(keyword) / len(heading_path) if heading_path else 0
                    })
                    break
    
    # Сортируем по match_score
    page_title_matches.sort(key=lambda x: x['match_score'], reverse=True)
    heading_path_matches.sort(key=lambda x: x['match_score'], reverse=True)
    page_path_matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    return {
        'page_title_matches': page_title_matches[:10],
        'heading_path_matches': heading_path_matches[:10],
        'page_path_matches': page_path_matches[:10]
    }

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
    
    # Определяем max_per_page на основе intent если не указан
    if max_per_page is None:
        intent_type = None
        if intent and isinstance(intent, dict):
            intent_type = intent.get('type')
        elif query:
            # Определяем intent из запроса если не передан
            intent_dict = classify_query_intent(query)
            intent_type = intent_dict.get('type') if intent_dict else None
        
        max_per_page = get_diversity_limit_for_intent(intent_type)
        logger.debug(f"Diversity filter: автоматический лимит {max_per_page} для intent={intent_type}")
    
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
        if not page_id or page_counts.get(page_id, 0) < max_per_page:
            filtered_results.append(result)
            if page_id:
                page_counts[page_id] = page_counts.get(page_id, 0) + 1
            
            # Достигли нужного количества результатов
            if len(filtered_results) >= limit:
                break
    
    # Логирование для анализа
    if page_counts:
        logger.debug(f"Diversity filter: {len(filtered_results)} results from {len(page_counts)} unique pages (max {max_per_page}/page)")
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
init_bm25_retriever(qdrant_client)

# Предзагрузка reranker модели при старте (чтобы первый запрос был быстрее)
logger.info("Предзагрузка reranker модели при старте...")
try:
    init_reranker()
    logger.info("✅ Reranker модель предзагружена и готова к использованию")
except Exception as e:
    logger.warning(f"⚠️ Не удалось предзагрузить reranker модель: {e}. Модель загрузится при первом запросе.")

mcp = FastMCP("Confluence RAG")

def execute_single_query_search(
    query_embedding: List[float],
    query_text: str,
    search_limit: int,
    where_filter: Optional[Dict],
    collection
) -> List[Dict]:
    """
    Выполнить поиск для одного варианта запроса.
    
    Args:
        query_embedding: Embedding запроса
        query_text: Текст запроса (для логирования)
        search_limit: Лимит результатов
        where_filter: Фильтр по метаданным
        collection: ChromaDB коллекция
        
    Returns:
        Список результатов поиска
    """
    try:
        # Используем search_in_qdrant напрямую
        from qdrant_storage import search_in_qdrant
        
        results_raw = search_in_qdrant(
            query_embedding=query_embedding,
            limit=search_limit,
            where_filter=where_filter,
            collection=QDRANT_COLLECTION
        )
        
        results = []
        for result in results_raw:
            doc_id = result.get('id', '')
            if doc_id:
                results.append({
                    'id': doc_id,
                    'text': result.get('text', ''),
                    'metadata': result.get('metadata', {}),
                    'distance': 1.0 - result.get('score', 0.0),
                    'query_variant': query_text  # Сохраняем вариант запроса для отладки
                })
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Ошибка Qdrant при поиске '{query_text}': {error_msg}")
        # Возвращаем пустой список вместо исключения для устойчивости
        return []


def parallel_multi_query_search(
    expanded_queries: List[str],
    query_embeddings: List[List[float]],
    search_limit: int,
    where_filter: Optional[Dict],
    collection,
    max_workers: int = None
) -> List[Dict]:
    """
    Параллельный поиск по всем вариантам запроса с использованием ThreadPoolExecutor.
    
    Args:
        expanded_queries: Список вариантов запроса
        query_embeddings: Список embeddings для каждого варианта
        search_limit: Лимит результатов для каждого запроса
        where_filter: Фильтр по метаданным
        collection: ChromaDB коллекция
        max_workers: Максимальное количество потоков (по умолчанию из ENV или количество запросов)
        
    Returns:
        Список уникальных результатов (дедуплицированных по id)
    """
    if not expanded_queries or not query_embeddings:
        return []
    
    if len(expanded_queries) != len(query_embeddings):
        logger.error(f"Несоответствие количества запросов и embeddings: {len(expanded_queries)} != {len(query_embeddings)}")
        return []
    
    # Определяем количество потоков
    if max_workers is None:
        max_workers = int(os.getenv('PARALLEL_SEARCH_MAX_WORKERS', '4'))
    
    # Ограничиваем количество потоков количеством запросов
    max_workers = min(max_workers, len(expanded_queries))
    
    # Если только один запрос или параллелизм отключен, выполняем последовательно
    enable_parallel = os.getenv('ENABLE_PARALLEL_SEARCH', 'true').lower() == 'true'
    if not enable_parallel or len(expanded_queries) == 1:
        logger.debug("Параллельный поиск отключен или только один запрос, выполняю последовательно")
        all_results = []
        seen_ids = set()
        for i, q in enumerate(expanded_queries):
            results = execute_single_query_search(
                query_embeddings[i], q, search_limit, where_filter, collection
            )
            for result in results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    all_results.append(result)
        return all_results
    
    # Параллельное выполнение
    logger.info(f"Параллельный поиск: {len(expanded_queries)} запросов в {max_workers} потоках")
    start_time = time.time()
    
    all_results = []
    seen_ids = set()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Создаём задачи для параллельного выполнения
        futures = {
            executor.submit(
                execute_single_query_search,
                query_embeddings[i],
                expanded_queries[i],
                search_limit,
                where_filter,
                collection
            ): expanded_queries[i]
            for i in range(len(expanded_queries))
        }
        
        # Собираем результаты по мере их готовности
        for future in as_completed(futures):
            query_variant = futures[future]
            try:
                results = future.result()
                for result in results:
                    if result['id'] not in seen_ids:
                        seen_ids.add(result['id'])
                        all_results.append(result)
            except Exception as e:
                logger.error(f"Ошибка при параллельном поиске для '{query_variant}': {e}")
                # Продолжаем с другими результатами (graceful degradation)
    
    elapsed = time.time() - start_time
    logger.info(f"Параллельный поиск завершён за {elapsed:.3f}с: {len(all_results)} уникальных результатов")
    
    return all_results


def get_text_for_reranking(text: str, query: str, max_len: int = 1200) -> str:
    """
    Умное извлечение текста для reranking.
    Максимизирует количество ключевых слов в фрагменте.
    
    Args:
        text: Полный текст документа
        query: Поисковый запрос
        max_len: Максимальная длина фрагмента (по умолчанию 1200 символов)
    
    Returns:
        Фрагмент текста с максимальным количеством ключевых слов
    """
    if not text or not query:
        return text[:max_len] if text else ""
    
    # Если текст короткий, возвращаем весь
    if len(text) <= max_len:
        return text
    
    # Извлекаем ключевые слова из запроса (убираем стоп-слова)
    stop_words = {'в', 'на', 'по', 'для', 'с', 'к', 'из', 'о', 'об', 'и', 'а', 'но', 'или', 'же', 'какой', 'какая', 'какие', 'где', 'как', 'что', 'это'}
    query_keywords = {kw.lower() for kw in query.split() if kw.lower() not in stop_words and len(kw) > 2}
    
    if not query_keywords:
        # Если нет ключевых слов, возвращаем первые max_len символов
        return text[:max_len]
    
    # Начальный фрагмент (первые max_len символов)
    best_snippet = text[:max_len]
    best_score = sum(1 for kw in query_keywords if kw in best_snippet.lower())
    
    # Ищем фрагмент с максимальным количеством ключевых слов
    # Шаг 50 символов для оптимизации производительности
    for start in range(0, len(text) - max_len, 50):
        snippet = text[start:start + max_len]
        score = sum(1 for kw in query_keywords if kw in snippet.lower())
        
        if score > best_score:
            best_score = score
            best_snippet = snippet
    
    # Если не нашли лучший фрагмент, пробуем последние max_len символов
    if best_score == 0:
        last_snippet = text[-max_len:] if len(text) > max_len else text
        last_score = sum(1 for kw in query_keywords if kw in last_snippet.lower())
        if last_score > best_score:
            best_snippet = last_snippet
    
    return best_snippet


@mcp.tool()
def confluence_semantic_search(query: str, limit: int = 5, space: str = "") -> str:
    """Семантический поиск по базе знаний Confluence с гибридным поиском.

    ВАЖНО: ВСЕГДА уточни пространство (space) перед поиском. Это критично для точности!

    Использует:
    - Hybrid Search (Vector 60% + BM25 40%) - объединение через RRF
    - Query expansion (синонимы, переформулировка, Semantic Query Log)
    - Metadata pre-filtering (по пространству, дате, типу)
    - CrossEncoder reranking (DiTy/russian-msmarco для точности)
    - Context expansion (соседние текст для полноты)
    - Hierarchy boost (иерархия для релевантности)

    Args:
        query (str): Поисковый запрос (вопрос пользователя)
        limit (int): Максимум результатов (1-20, по умолчанию 5). 
                     Для completeness используй 10+
        space (str): Пространство Confluence для фильтрации.
                     Примеры: "RAUII" (проект), "Surveys" (обследования), "DEVOPS" (девопс)
                     КРИТИЧНО: если не указан → результаты из всех пространств (менее точно)

    Returns:
        str: Форматированные результаты поиска с источниками или сообщение об ошибке

    Examples:
        - Если пользователь спрашивает "какой стек в RAUII" → space="RAUII"
        - Если пользователь спрашивает "вопросы про обследование" → space="Surveys"
        - Если space не ясен → спроси: "В каком пространстве искать: RAUII или Surveys?"

    Note:
        - Используй completeness search (limit=10 или больше) для полноты
        - Указывай источник в ответе (пространство, страница, ID)
        - Не сочиняй информацию вне Confluence
        - Если scores низкие (<0.0001) → retry с limit=20
    """
    try:
        # ============ БЕЗОПАСНОСТЬ: Валидация входных данных ============
        # Проверка query
        if not query or not isinstance(query, str):
            return "❌ Ошибка: Пустой или некорректный запрос"
        
        query = query.strip()
        if len(query) < 2:
            return "❌ Ошибка: Запрос слишком короткий (минимум 2 символа)"
        
        if len(query) > 1000:
            logger.warning(f"Очень длинный запрос ({len(query)} символов), обрезаю до 1000")
            query = query[:1000]
        
        # Сохраняем оригинальный запрос для использования в boost (до извлечения space)
        original_query = query
        
        # ============ НОВОЕ: Извлечение space из текста запроса ============
        # Если space не указан как отдельный параметр, пытаемся извлечь из текста
        if not space:
            # Паттерны: "spaces RAUII", "space RAUII", "в пространстве RAUII"
            space_patterns = [
                r'\bspaces?\s+([A-Za-z0-9_-]+)\s*$',  # "spaces RAUII" в конце
                r'\bspaces?\s+([A-Za-z0-9_-]+)(?:\s|$)',  # "spaces RAUII" в любом месте
                r'\bв\s+пространстве\s+([A-Za-z0-9_-]+)\s*$',  # "в пространстве RAUII"
                r'\bпространство\s+([A-Za-z0-9_-]+)\s*$',  # "пространство RAUII"
            ]
            
            for pattern in space_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    extracted_space = match.group(1).strip()
                    # Удаляем найденный паттерн из запроса
                    query = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
                    space = extracted_space
                    logger.info(f"Извлечен space из запроса: '{space}', обновленный запрос: '{query}'")
                    break
        
        # Проверка space
        if space and not isinstance(space, str):
            return "❌ Ошибка: Некорректный параметр space"
        
        if space:
            space = space.strip()
            # Защита от injection: только буквы, цифры, дефис, подчеркивание
            if not re.match(r'^[a-zA-Z0-9_-]+$', space):
                logger.warning(f"Подозрительный space параметр: {space}")
                return "❌ Ошибка: Параметр space содержит недопустимые символы"
        
        # Проверка что RAG инициализирована при старте сервера
        if qdrant_client is None:
            return "❌ Ошибка: RAG система не инициализирована. Проверьте логи сервера."
        
        # Нормализация лимита
        limit = min(max(limit, 1), 20)
        
        # Проверка наличия документов
        from qdrant_storage import get_qdrant_count
        doc_count = get_qdrant_count()
        if doc_count == 0:
            logger.warning("Попытка поиска по пустому индексу")
            return "⚠️ Индекс пуст. Дождитесь завершения первой синхронизации."
        
        # ============ НОВОЕ: Structural Navigation Search ============
        # Проверяем, является ли запрос структурным (с разделителями >, →)
        structure = parse_query_structure(query)
        logger.debug(f"🔍 Query structure analysis: is_structural={structure['is_structural_query']}, parts={structure['parts']}")
        
        # Переменная для хранения структурных результатов (для последующего объединения)
        structural_results = None
        
        if structure['is_structural_query']:
            logger.info(f"🔍 Структурный запрос обнаружен: {structure['parts']}")
            
            # Выполняем структурный поиск (с кэшированием)
            structural_search_start = time.time()
            structural_results = cached_structural_search(
                collection,
                structure,
                limit=limit * 10  # Получаем больше результатов для последующего reranking
            )
            structural_search_time = time.time() - structural_search_start
            
            if structural_results:
                logger.info(
                    f"✅ Найдено {len(structural_results)} результатов структурного поиска "
                    f"(за {structural_search_time:.3f}с)"
                )
                
                # Если достаточно результатов - применяем легкий reranking и возвращаем
                if len(structural_results) >= limit:
                    # Применяем легкий reranking для структурных результатов
                    # (используем match_score как базовый score)
                    for result in structural_results:
                        # Нормализуем match_score в диапазон 0-1
                        max_match = max([r['match_score'] for r in structural_results]) if structural_results else 1
                        result['rerank_score'] = (result['match_score'] / max_match) * 0.5 if max_match > 0 else 0.1
                        result['distance'] = 1.0 - result['rerank_score']  # Для совместимости
                    
                    # Анализируем метаданные для boost
                    metadata_analysis = analyze_query_with_metadata(query, collection)
                    structural_results = apply_metadata_boost(structural_results, metadata_analysis)
                    
                    # Сортируем по boosted_score
                    structural_results.sort(key=lambda x: x.get('boosted_score', x.get('rerank_score', 0)), reverse=True)
                    
                    # Форматируем результаты
                    format_start = time.time()
                    formatted = format_search_results(structural_results[:limit], query, limit)
                    format_time = time.time() - format_start
                    total_structural_time = time.time() - structural_search_start
                    
                    logger.info(
                        f"✅ Структурный поиск: возвращено {limit} результатов "
                        f"(total: {total_structural_time:.3f}с, format: {format_time:.3f}с)"
                    )
                    return formatted
                else:
                    logger.info(f"⚠️ Структурный поиск нашел только {len(structural_results)} результатов, объединяю с семантическим")
                    # Объединяем с семантическим поиском (продолжаем pipeline)
                    # structural_results будут объединены позже
            else:
                logger.info(f"⚠️ Структурный поиск не нашел результатов, fallback на семантический")
                # Продолжаем обычный семантический поиск
        
        # ============ НОВОЕ: Semantic Caching ============
        cache = get_semantic_cache()
        cached_results = cache.get(query, space, limit)
        
        if cached_results:
            logger.info(f"✅ Cache HIT: '{query[:50]}...'")
            return cached_results
        
        logger.debug(f"Cache MISS: '{query[:50]}...', выполняю поиск...")
        
        # ============ НОВОЕ: Query Intent Classification ============
        intent = classify_query_intent(query)
        if not intent or not isinstance(intent, dict):
            # Fallback: используем значения по умолчанию
            intent = {
                'type': 'exploratory',
                'boost_hierarchy': False,
                'expand_context': True,
                'diversity': 2
            }
        logger.info(f"Query intent: {intent.get('type', 'unknown')} (diversity={intent.get('diversity', 2)}, expand_context={intent.get('expand_context', True)})")
        
        # ============ УЛУЧШЕНИЕ 1: Query Expansion (расширенная версия) ============
        expanded_queries = expand_query(query, space)
        if len(expanded_queries) > 1:
            logger.info(f"Query expansion: {query} → {expanded_queries}")
        
        # ============ УЛУЧШЕНИЕ 2: Адаптивный лимит кандидатов ============
        search_limit = calculate_optimal_candidate_limit(query, limit)
        logger.debug(f"Оптимальное количество кандидатов: {search_limit}")
        
        # ============ УЛУЧШЕНИЕ 3: Metadata Pre-filtering ============
        # Поиск напрямую через ChromaDB с фильтрацией ДО векторного поиска
        
        where_filter = None
        if space:
            where_filter = {"space": space}
            logger.info(f"Pre-filtering by space: {space}")
        
        # ОПТИМИЗАЦИЯ: Генерируем все embeddings batch'ом (в 3-5 раз быстрее)
        embedding_start = time.time()
        query_embeddings = generate_query_embeddings_batch(expanded_queries)
        embedding_elapsed = time.time() - embedding_start
        logger.debug(f"Batch embeddings generated за {embedding_elapsed:.3f}с для {len(expanded_queries)} запросов")
        
        # ============ НОВОЕ: Параллельный Multi-Query Search ============
        # Выполняем поиск по всем вариантам запроса ПАРАЛЛЕЛЬНО
        try:
            all_results = parallel_multi_query_search(
                expanded_queries=expanded_queries,
                query_embeddings=query_embeddings,
                search_limit=search_limit,
                where_filter=where_filter,
                collection=QDRANT_COLLECTION
            )
        except Exception as e:
            logger.error(f"Ошибка при параллельном поиске: {e}, fallback на последовательный режим")
            # Fallback на последовательный режим
            all_results = []
            seen_ids = set()
            for i, q in enumerate(expanded_queries):
                results = execute_single_query_search(
                    query_embeddings[i], q, search_limit, where_filter, collection
                )
                for result in results:
                    if result['id'] not in seen_ids:
                        seen_ids.add(result['id'])
                        all_results.append(result)
        
        # ============ УЛУЧШЕНИЕ 4: Fallback стратегии (3 уровня) ============
        fallback_search = get_fallback_search(min_results=3)
        fallback_message = ""
        original_space = space
        
        # Fallback #1: Убираем space фильтр
        if fallback_search.should_apply_fallback(all_results, level=1) and space:
            logger.info(f"🔄 Fallback #1: Убираю space фильтр '{space}'...")
            # Повторяем поиск без фильтра по space (тоже параллельно)
            where_filter = None
            # ОПТИМИЗАЦИЯ: Переиспользуем уже сгенерированные embeddings
            try:
                fallback_results = parallel_multi_query_search(
                    expanded_queries=expanded_queries,
                    query_embeddings=query_embeddings,
                    search_limit=search_limit,
                    where_filter=where_filter,
                    collection=QDRANT_COLLECTION
                )
                # Добавляем только новые результаты
                seen_ids = {r['id'] for r in all_results}
                for result in fallback_results:
                    if result['id'] not in seen_ids:
                        all_results.append(result)
            except Exception as e:
                logger.warning(f"Ошибка при параллельном fallback поиске: {e}, используем последовательный режим")
                # Fallback на последовательный режим
                seen_ids = {r['id'] for r in all_results}
                for i, q in enumerate(expanded_queries):
                    results = execute_single_query_search(
                        query_embeddings[i], q, search_limit, where_filter, QDRANT_COLLECTION
                    )
                    for result in results:
                        if result['id'] not in seen_ids:
                            seen_ids.add(result['id'])
                            all_results.append(result)
            
            if all_results:
                fallback_message = fallback_search.get_fallback_message(1, original_space)
        
        # Fallback #2: PRF (Pseudo-Relevance Feedback)
        if fallback_search.should_apply_fallback(all_results, level=2):
            logger.info(f"🔄 Fallback #2: Применяю PRF (Pseudo-Relevance Feedback)...")
            
            try:
                # Применяем PRF к исходному запросу
                expanded_query_prf = pseudo_relevance_feedback(query, all_results, top_k=3, max_terms=5)
                
                if expanded_query_prf != query:
                    # Генерируем embedding для расширенного запроса
                    prf_embedding = generate_query_embedding(expanded_query_prf)
                    
                    # Повторный поиск через Qdrant
                    from qdrant_storage import search_in_qdrant
                    qdrant_results = search_in_qdrant(
                        query_embedding=prf_embedding,
                        limit=search_limit,
                        where_filter=None,  # Без фильтров
                        collection=QDRANT_COLLECTION
                    )
                    
                    # Обрабатываем результаты
                    for result in qdrant_results:
                        doc_id = result.get('id', '')
                        if doc_id and doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            doc_text = result.get('text', '')
                            doc_metadata = result.get('metadata', {})
                            
                            if doc_text:  # Проверяем, что текст не пустой
                                all_results.append({
                                    'id': doc_id,
                                    'text': doc_text,
                                    'metadata': doc_metadata,
                                    'distance': 1.0 - result.get('score', 0.0)
                                })
                    
                    if all_results:
                        fallback_message = fallback_search.get_fallback_message(2)
                        
            except Exception as e:
                logger.warning(f"PRF Fallback failed: {e}")
        
        # Fallback 2: Если все еще мало результатов, пробуем только ключевые слова
        if len(all_results) < 3:
            keywords = extract_keywords(query)
            if len(keywords) >= 2:
                keyword_query = ' '.join(keywords)
                logger.info(f"Мало результатов, пробую поиск по ключевым словам: {keyword_query}")
                
                # ОПТИМИЗАЦИЯ: Генерируем embedding только для keyword_query
                keyword_embedding = generate_query_embedding(keyword_query)
                
                # Поиск через Qdrant
                from qdrant_storage import search_in_qdrant
                qdrant_results = search_in_qdrant(
                    query_embedding=keyword_embedding,
                    limit=search_limit,
                    where_filter=None,  # Без фильтров для keyword search
                    collection=QDRANT_COLLECTION
                )
                
                # Обрабатываем результаты
                for result in qdrant_results:
                    doc_id = result.get('id', '')
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        doc_text = result.get('text', '')
                        doc_metadata = result.get('metadata', {})
                        
                        if doc_text:  # Проверяем, что текст не пустой
                            all_results.append({
                                'id': doc_id,
                                'text': doc_text,
                                'metadata': doc_metadata,
                                'distance': 1.0 - result.get('score', 0.0)
                            })
        
        if not all_results:
            return f"❌ Ничего не найдено по запросу: '{query}'"
        
        logger.info(f"Найдено {len(all_results)} уникальных результатов до обработки")
        
        # ============ НОВОЕ: Объединение структурных результатов с семантическими ============
        # Если структурный поиск нашел результаты, но их мало - объединяем
        if structural_results and len(structural_results) > 0:
            merge_start = time.time()
            logger.info(f"🔗 Объединяю {len(structural_results)} структурных результатов с {len(all_results)} семантическими")
            
            # Добавляем структурные результаты в начало списка (они более релевантны)
            seen_ids = {r.get('id') for r in all_results}
            for struct_result in structural_results:
                if struct_result.get('id') not in seen_ids:
                    # Нормализуем match_score для совместимости
                    max_match = max([r['match_score'] for r in structural_results]) if structural_results else 1
                    struct_result['rerank_score'] = (struct_result['match_score'] / max_match) * 0.5 if max_match > 0 else 0.1
                    struct_result['distance'] = 1.0 - struct_result['rerank_score']
                    all_results.insert(0, struct_result)  # Вставляем в начало
                    seen_ids.add(struct_result.get('id'))
            
            merge_time = time.time() - merge_start
            logger.info(
                f"✅ Объединено: {len(all_results)} результатов (структурные + семантические) "
                f"за {merge_time:.3f}с"
            )
        
        # ============ НОВОЕ: Hybrid Search (Vector + BM25) ============
        # Объединяем результаты векторного поиска с BM25 через RRF
        try:
            all_results = hybrid_search(
                query=query,
                collection=QDRANT_COLLECTION,
                vector_results=all_results,
                space_filter=space if space else None,
                limit=search_limit * 2  # Берем больше для лучшего объединения
            )
            logger.info(f"✅ Hybrid Search: объединено {len(all_results)} результатов (Vector + BM25)")
        except Exception as e:
            logger.warning(f"Ошибка Hybrid Search: {e}, используем только векторные результаты")
        
        # ============ УЛУЧШЕНИЕ 5: Дедупликация ============
        all_results = deduplicate_results(all_results)
        logger.info(f"После дедупликации: {len(all_results)} результатов")
        
        # ============ УЛУЧШЕНИЕ 6: Adaptive Reranking (умный выбор лимита) ============
        # ADAPTIVE: Выбираем количество документов для reranking в зависимости от запроса
        
        def get_adaptive_rerank_limit(query: str, candidate_count: int, has_space_filter: bool) -> int:
            """
            Умный выбор количества документов для reranking.
            
            Факторы:
            1. Длина запроса (короткие = меньше вариантов нужно)
            2. Наличие фильтра по space (уже отфильтровано = меньше нужно)
            3. Количество кандидатов
            
            Returns:
                Оптимальное количество документов для reranking (8-20)
            """
            query_words = len(query.split())
            
            # Базовый лимит в зависимости от длины запроса
            if query_words <= 3:
                # Короткий запрос: "стек RAUII", "контакты команды"
                base_limit = 8
            elif query_words <= 8:
                # Средний запрос: "какой стек технологий используется"
                base_limit = 12
            else:
                # Длинный/сложный запрос
                base_limit = 20
            
            # Если есть фильтр по space - уменьшаем лимит (уже отфильтровано)
            if has_space_filter:
                base_limit = max(8, int(base_limit * 0.8))
            
            # Не больше чем есть кандидатов
            return min(base_limit, candidate_count)
        
        # Вычисляем адаптивный лимит
        RERANK_LIMIT = get_adaptive_rerank_limit(query, len(all_results), bool(space))
        logger.info(f"Adaptive rerank limit: {RERANK_LIMIT} (query_words: {len(query.split())}, has_filter: {bool(space)})")
        
        if len(all_results) > RERANK_LIMIT:
            logger.info(f"Ограничение reranking: {len(all_results)} → {RERANK_LIMIT} документов")
            all_results = all_results[:RERANK_LIMIT]
        
        ranker = init_reranker()
        
        if ranker and len(all_results) > 1:
            try:
                start_time = time.time()
                
                # Подготовка пар (query, document) для reranking
                # ОПТИМИЗАЦИЯ: Уменьшаем длину документа для ускорения
                # Безопасная обработка: проверяем что text не None
                # Важно: создаем пары только для результатов с текстом и сохраняем индексы
                valid_indices = []
                pairs = []
                for i, r in enumerate(all_results):
                    text = r.get('text')
                    if text and isinstance(text, str) and len(text.strip()) > 0:
                        valid_indices.append(i)
                        pairs.append((query, get_text_for_reranking(text, query, max_len=1200)))
                
                if not pairs:
                    logger.warning("Нет валидных пар для reranking (все результаты имеют пустой текст)")
                else:
                    # Получаем scores от CrossEncoder
                    # CrossEncoder автоматически батчит запросы (batch_size=32 по умолчанию)
                    scores = ranker.predict(pairs)
                    
                    # Добавляем scores только к валидным результатам
                    for pair_idx, result_idx in enumerate(valid_indices):
                        if pair_idx < len(scores):
                            all_results[result_idx]['rerank_score'] = float(scores[pair_idx])
                        else:
                            all_results[result_idx]['rerank_score'] = 0.0
                    
                    # Для результатов без текста устанавливаем score = 0
                    for i, result in enumerate(all_results):
                        if i not in valid_indices:
                            result['rerank_score'] = 0.0
                
                # Сортируем по rerank score (descending)
                all_results.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                # ============ НОВОЕ: Metadata Boost ============
                # Увеличиваем score для результатов с ключевыми словами в title/breadcrumb/page_path
                query_keywords = set(query.lower().split())
                stop_words = {'в', 'на', 'по', 'для', 'с', 'к', 'из', 'о', 'об', 'и', 'а', 'но', 'или', 'же', 'какой', 'какая', 'какие', 'где', 'как', 'что', 'это', 'проекта', 'проект'}
                query_keywords = {kw for kw in query_keywords if kw.lower() not in stop_words and len(kw) > 2}
                
                boosted_count = 0
                for r in all_results:
                    if not r or not isinstance(r, dict):
                        continue
                    
                    metadata = r.get('metadata', {})
                    if not isinstance(metadata, dict):
                        continue
                    
                    title = metadata.get('title', '').lower()
                    breadcrumb = r.get('breadcrumb', '').lower()
                    page_path = metadata.get('page_path', '').lower()
                    
                    # Проверяем совпадение ключевых слов
                    title_match = any(kw in title for kw in query_keywords)
                    breadcrumb_match = any(kw in breadcrumb for kw in query_keywords)
                    path_match = any(kw in page_path for kw in query_keywords)
                    
                    if title_match or breadcrumb_match or path_match:
                        current_score = r.get('rerank_score', 0)
                        if current_score >= 0:  # Boost даже для score 0
                            # Сильный boost если ключевые слова найдены
                            r['rerank_score'] = max(current_score * 2.0, 0.1)  # Минимум 0.1
                            r['metadata_boost'] = True
                            boosted_count += 1
                            logger.debug(
                                f"Metadata boost для '{metadata.get('title', 'Unknown')}': "
                                f"{current_score:.3f} → {r['rerank_score']:.3f}"
                            )
                    
                    # ============ НОВОЕ: Exact Phrase Boost ============
                    # Дополнительный boost для точных совпадений важных фраз
                    # Используем оригинальный запрос (до извлечения space)
                    original_query_lower = original_query.lower()
                    important_phrases = [
                        'учет номенклатуры',
                        'номенклатура',
                        'склад',
                        'обследование',
                        'классификация',
                    ]
                    
                    # Проверяем точные совпадения фраз (более сильный boost)
                    for phrase in important_phrases:
                        if phrase in original_query_lower:
                            combined_metadata = f"{title} {breadcrumb} {page_path}"
                            if phrase in combined_metadata:
                                # Очень сильный boost для точных совпадений важных фраз
                                current_score = r.get('rerank_score', 0)
                                r['rerank_score'] = max(current_score * 2.5, 0.2)
                                r['exact_phrase_boost'] = True
                                boosted_count += 1
                                logger.debug(
                                    f"Exact phrase boost для '{phrase}' в '{metadata.get('title', 'Unknown')}': "
                                    f"{current_score:.3f} → {r['rerank_score']:.3f}"
                                )
                                break
                
                if boosted_count > 0:
                    logger.info(f"Metadata boost применен к {boosted_count} результатам")
                
                # Пересортируем после boost
                all_results.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                elapsed = time.time() - start_time
                if all_results:
                    top_score = max((r.get('rerank_score', 0) for r in all_results), default=0)
                    logger.info(f"Reranking completed за {elapsed:.2f}с. Top score: {top_score:.3f}")
                else:
                    logger.warning("Reranking completed, но нет результатов")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using original order")
        
        # ============ НОВОЕ: Score Threshold Filtering (адаптивный) ============
        # УЛУЧШЕНИЕ: Адаптивный порог в зависимости от типа запроса
        # Технические запросы требуют более мягкого порога
        
        # Определяем технические термины в запросе
        technical_terms = ['api', 'http', 'rest', 'json', 'xml', 'sql', 'docker', 
                          'git', '1с', '1c', 'endpoint', 'webhook', 'oauth', 
                          'deployment', 'ssl', 'тест', 'баг', 'конфигурация']
        
        is_technical_query = any(term in query.lower() for term in technical_terms)
        
        # Адаптивный порог через ENV или дефолт
        RERANK_THRESHOLD_TECHNICAL = float(os.getenv('RERANK_THRESHOLD_TECHNICAL', '1.5'))
        RERANK_THRESHOLD_GENERAL = float(os.getenv('RERANK_THRESHOLD_GENERAL', '2.0'))
        
        if is_technical_query:
            MIN_RERANK_SCORE = RERANK_THRESHOLD_TECHNICAL
            logger.debug(f"Технический запрос обнаружен, порог: {MIN_RERANK_SCORE}")
        else:
            MIN_RERANK_SCORE = RERANK_THRESHOLD_GENERAL
            logger.debug(f"Общий запрос, порог: {MIN_RERANK_SCORE}")
        
        original_count = len(all_results)
        all_results = [r for r in all_results if r.get('rerank_score', 0) >= MIN_RERANK_SCORE]
        
        if len(all_results) < original_count:
            filtered_count = original_count - len(all_results)
            logger.info(f"Score threshold filtering: убрано {filtered_count} низкорелевантных результатов (score < {MIN_RERANK_SCORE})")
        
        if not all_results:
            logger.warning(f"Все результаты отфильтрованы по score threshold ({MIN_RERANK_SCORE})")
            return f"⚠️ Найденные результаты имеют низкую релевантность (score < {MIN_RERANK_SCORE}). Попробуйте переформулировать запрос."
        
        # Фильтрация результатов с пустым текстом (до финальной обработки)
        all_results = [r for r in all_results if r['text'] and len(str(r['text']).strip()) > 0]
        
        if not all_results:
            return f"❌ Ничего не найдено по запросу: '{query}' (все результаты имели пустой текст)"
        
        # ============ УЛУЧШЕНИЕ 7: Context Enrichment ============
        # Безопасная обработка: проверяем что результат не None
        enriched_results = []
        for r in all_results:
            if r and isinstance(r, dict):
                enriched = enrich_result_with_context(r)
                if enriched:
                    enriched_results.append(enriched)
        all_results = enriched_results
        
        if not all_results:
            return f"❌ Ничего не найдено по запросу: '{query}' (ошибка при обогащении контекста)"
        
        # ============ УЛУЧШЕНИЕ 8: Context Expansion (Bidirectional + Related) ============
        # АДАПТИВНО: Расширяем контекст только если intent требует
        if intent and intent.get('expand_context', True):
            expansion_mode = os.getenv('CONTEXT_EXPANSION_MODE', 'bidirectional').lower()
            context_size = int(os.getenv('CONTEXT_EXPANSION_SIZE', '2'))
            logger.info(f"Расширяю контекст (mode={expansion_mode}, size={context_size}, intent: expand_context=True)...")
            
            # Получаем модель embeddings для related expansion
            embeddings_model = None
            if expansion_mode in ['related', 'all']:
                try:
                    embeddings_model = get_embed_model()
                except Exception as e:
                    logger.debug(f"Не удалось получить embeddings model для related expansion: {e}")
            
            # Безопасная обработка: проверяем что результат не None
            expanded_results = []
            for r in all_results:
                if r and isinstance(r, dict):
                    expanded = expand_context_full(
                        r,
                        collection=QDRANT_COLLECTION,
                        embeddings_model=embeddings_model,
                        expansion_mode=expansion_mode,
                        context_size=context_size
                    )
                    if expanded:
                        expanded_results.append(expanded)
            all_results = expanded_results
            
            if not all_results:
                logger.warning("Нет результатов после context expansion")
        else:
            logger.info("Пропускаю context expansion (intent: expand_context=False)")
            # Заполняем expanded_text оригинальным текстом
            for r in all_results:
                r['expanded_text'] = r.get('text', '')
                r['context_chunks'] = 1
                r['expansion_mode'] = 'disabled'
        
        # ============ УЛУЧШЕНИЕ 9: Hierarchy & Path Boost ============
        logger.info("Вычисляю буст на основе иерархии и пути...")
        for r in all_results:
            if not r or not isinstance(r, dict):
                continue
            
            # Базовый score от reranker
            base_score = r.get('rerank_score', 0.0)
            
            # Hierarchy boost (важность страницы в структуре)
            # АДАПТИВНО: Усиливаем для навигационных запросов
            metadata = r.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
                r['metadata'] = metadata
            
            hierarchy_boost = calculate_hierarchy_boost(metadata)
            if intent and intent.get('boost_hierarchy', False):
                hierarchy_boost *= 1.5  # Усиливаем для навигационных запросов
                logger.debug(f"Hierarchy boost усилен для навигационного запроса: {hierarchy_boost:.2f}")
            
            # Breadcrumb match (совпадение пути с запросом)
            breadcrumb_boost = calculate_breadcrumb_match_score(query, r.get('breadcrumb', ''))
            
            # Комбинированный финальный score
            # Веса: rerank (основной) + hierarchy (30%) + breadcrumb (20%)
            final_score = (
                base_score * 1.0 +           # Основной score от CrossEncoder
                hierarchy_boost * 0.3 +      # Буст за важность в иерархии
                breadcrumb_boost * 0.2       # Буст за совпадение пути
            )
            
            r['final_score'] = final_score
            r['hierarchy_boost'] = hierarchy_boost
            r['breadcrumb_boost'] = breadcrumb_boost
            
            logger.debug(
                f"Scores for {r['metadata'].get('title', 'Unknown')}: "
                f"rerank={base_score:.2f}, hierarchy=+{hierarchy_boost:.2f}, "
                f"breadcrumb=+{breadcrumb_boost:.2f}, final={final_score:.2f}"
            )
        
        # Пересортируем по финальному score
        all_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        if all_results:
            logger.info(f"Результаты пересортированы. Top score: {all_results[0].get('final_score', 0):.2f}")
        else:
            logger.warning("Нет результатов после пересортировки")
        
        # ============ НОВОЕ: Diversity Filter (адаптивный с ENV) ============
        # АДАПТИВНО: Лимит чанков/страницу зависит от intent и настраивается через ENV
        intent_type = intent.get('type', 'unknown') if intent else 'unknown'
        diversity_limit = get_diversity_limit_for_intent(intent_type)
        logger.info(f"Применяю diversity filter (max {diversity_limit} chunks/page, intent={intent_type})...")
        filtered_results = apply_diversity_filter(
            all_results, 
            limit=limit, 
            max_per_page=diversity_limit,
            query=query,
            intent=intent
        )
        logger.info(f"После diversity filter: {len(filtered_results)} результатов")
        
        results = filtered_results
        
        # Формирование ответа
        response = [f"✅ Найдено {len(results)} результатов (intent={intent_type}, diversity: max {diversity_limit}/page)"
                   f" | pipeline: expansion → filter → reranking → hierarchy → diversity:\n"]
        for i, r in enumerate(results, 1):
            if not r or not isinstance(r, dict):
                continue
            
            m = r.get('metadata')
            if not m or not isinstance(m, dict):
                m = {}
                r['metadata'] = m
            page_space = m.get('space', 'Unknown')
            page_url = m.get('url', '')
            
            # Используем breadcrumb вместо отдельных полей
            breadcrumb = r.get('breadcrumb', m.get('title', 'Без названия'))
            
            # Новые метаданные
            labels = m.get('labels', '')
            created_by = m.get('created_by', '')
            attachments = m.get('attachments', '')
            chunk_num = m.get('chunk', 0)
            
            # ============ Snippet Extraction из расширенного контекста ============
            # Используем expanded_text (с context expansion)
            text = r.get('expanded_text', r.get('text', "[Текст недоступен]"))
            text_preview = extract_relevant_snippet(text, query, max_length=800)
            
            # Показываем финальный score с декомпозицией
            final_score = r.get('final_score', 0)
            rerank_score = r.get('rerank_score', 0)
            hierarchy_boost = r.get('hierarchy_boost', 0)
            breadcrumb_boost = r.get('breadcrumb_boost', 0)
            context_chunks = r.get('context_chunks', 1)
            
            # Эмодзи в зависимости от score
            if final_score > 7.0:
                score_emoji = "🔥"
            elif final_score > 5.0:
                score_emoji = "⭐"
            elif final_score > 3.0:
                score_emoji = "✓"
            else:
                score_emoji = "·"
            
            # Формируем строку со score
            score_parts = [f"{score_emoji} {final_score:.2f}"]
            
            # Добавляем декомпозицию если есть бусты
            if hierarchy_boost > 0 or breadcrumb_boost > 0:
                score_details = []
                if rerank_score > 0:
                    score_details.append(f"base:{rerank_score:.1f}")
                if hierarchy_boost > 0:
                    score_details.append(f"+hier:{hierarchy_boost:.1f}")
                if breadcrumb_boost > 0:
                    score_details.append(f"+path:{breadcrumb_boost:.1f}")
                score_parts.append(f"({', '.join(score_details)})")
            
            score_str = " | ".join(score_parts)
            
            # Информация о контексте
            context_str = ""
            if context_chunks > 1:
                context_str = f" | 📚 {context_chunks} chunks"
            
            # Дополнительная информация
            extra_info = []
            if labels:
                extra_info.append(f"🏷️ {labels}")
            if created_by:
                extra_info.append(f"👤 {created_by}")
            if attachments:
                # Показываем до 3 первых вложений
                att_list = attachments.split(',')[:3]
                att_preview = ', '.join(att_list)
                if len(attachments.split(',')) > 3:
                    att_preview += f" (+{len(attachments.split(',')) - 3})"
                extra_info.append(f"📎 {att_preview}")
            
            extra_str = " | ".join(extra_info)
            if extra_str:
                extra_str = f" | {extra_str}"
            
            response.append(
                f"[{i}] 📍 {breadcrumb}\n"
                f"    📁 {page_space} | Chunk #{chunk_num} | {score_str}{context_str}{extra_str}\n"
                f"    🔗 {page_url}\n"
                f"    💬 {text_preview}\n"
            )
        
        # Добавляем fallback_message если есть
        if fallback_message:
            response.insert(1, f"\n{fallback_message}\n")
        
        final_response = "\n".join(response)
        
        # ============ НОВОЕ: Сохраняем в кэш ============
        try:
            cache.set(query, final_response, space, limit)
        except Exception as e:
            logger.warning(f"Не удалось сохранить в кэш: {e}")
        
        # ============ НОВОЕ: Логируем запрос для обучения ============
        try:
            synonyms_manager = get_synonyms_manager()
            synonyms_manager.log_query(query, results)
        except Exception as e:
            logger.warning(f"Не удалось залогировать запрос: {e}")
        
        # ============ НОВОЕ: Логирование для Semantic Query Log (5-й источник) ============
        expansion_source = 'other'
        try:
            from semantic_query_log import get_semantic_query_log
            
            semantic_log = get_semantic_query_log()
            
            # Проверяем, использовался ли Semantic Query Log для расширения
            related_queries = semantic_log.get_related_queries(query, top_n=1)
            if related_queries:
                expansion_source = 'semantic_query_log'
            
            # Логируем запрос
            semantic_log.log_query(query, len(results))
            logger.debug(f"Semantic Query Log: запрос '{query}' залогирован ({len(results)} результатов, источник расширения: {expansion_source})")
        except Exception as e:
            logger.debug(f"Ошибка логирования в Semantic Query Log: {e}")
        
        return final_response
        
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}", exc_info=True)
        return f"❌ Ошибка при выполнении поиска: {str(e)}"

@mcp.tool()
def confluence_search_by_label(label: str, limit: int = 5) -> str:
    """
    Поиск страниц Confluence по метке (label/tag).
    
    Args:
        label: Название метки для поиска
        limit: Максимальное количество результатов (1-20)
    
    Returns:
        Список страниц с данной меткой
    """
    try:
        # Проверка что RAG инициализирована при старте сервера
        if qdrant_client is None:
            return "❌ Ошибка: RAG система не инициализирована. Проверьте логи сервера."
        
        limit = min(max(limit, 1), 20)
        
        # Проверка наличия документов
        from qdrant_storage import get_qdrant_count, get_all_points
        doc_count = get_qdrant_count(QDRANT_COLLECTION)
        if doc_count == 0:
            return "⚠️ Индекс пуст. Дождитесь завершения первой синхронизации."
        
        # Получаем документы с разумным лимитом для предотвращения OOM
        MAX_SCAN_LIMIT = 10000  # Максимум документов для сканирования
        all_points = get_all_points(limit=MAX_SCAN_LIMIT, include_payload=True, collection=QDRANT_COLLECTION)
        all_data = {
            'documents': [p.get('text', '') for p in all_points.get('points', [])],
            'metadatas': [p.get('metadata', {}) for p in all_points.get('points', [])]
        }
        scanned_count = len(all_data.get('metadatas', []))
        
        # Фильтруем по метке
        matching_results = []
        seen_pages = set()  # Для удаления дубликатов по page_id
        
        for idx, metadata in enumerate(all_data.get('metadatas', [])):
            labels_str = metadata.get('labels', '')
            page_id = metadata.get('page_id', '')
            
            # Проверяем что метка присутствует и страница еще не добавлена
            if label.lower() in labels_str.lower() and page_id not in seen_pages:
                matching_results.append({
                    'title': metadata.get('title', 'Без названия'),
                    'space': metadata.get('space', 'Unknown'),
                    'url': metadata.get('url', ''),
                    'labels': labels_str,
                    'parent_title': metadata.get('parent_title', ''),
                    'page_id': page_id
                })
                seen_pages.add(page_id)
                
                if len(matching_results) >= limit:
                    break
        
        if not matching_results:
            warning = f" (проверено {scanned_count} документов)" if scanned_count >= MAX_SCAN_LIMIT else ""
            return f"❌ Страниц с меткой '{label}' не найдено{warning}"
        
        # Формируем ответ
        response = [f"✅ Найдено {len(matching_results)} страниц с меткой '{label}':\n"]
        for i, page in enumerate(matching_results, 1):
            parent_str = f" (← {page['parent_title']})" if page['parent_title'] else ""
            response.append(
                f"[{i}] 📄 {page['title']}{parent_str}\n"
                f"    📁 Space: {page['space']}\n"
                f"    🏷️ Метки: {page['labels']}\n"
                f"    🔗 {page['url']}\n"
            )
        
        return "\n".join(response)
        
    except Exception as e:
        logger.error(f"Ошибка поиска по метке: {e}", exc_info=True)
        return f"❌ Ошибка при поиске по метке: {str(e)}"

@mcp.tool()
def confluence_list_spaces() -> str:
    """
    Получить список доступных пространств Confluence для помощи в выборе.
    
    Используйте этот инструмент, если пользователь не знает, в каком пространстве искать,
    или если нужно показать доступные пространства перед вызовом confluence_semantic_search.
    
    Returns:
        Список доступных пространств с количеством документов в каждом
    """
    try:
        if qdrant_client is None:
            return "❌ Ошибка: RAG система не инициализирована."
        
        # Получаем все уникальные пространства из метаданных
        # Используем разумный лимит для предотвращения OOM
        MAX_SCAN_LIMIT = 10000
        from qdrant_storage import get_all_points
        all_points = get_all_points(limit=MAX_SCAN_LIMIT, include_payload=True, collection=QDRANT_COLLECTION)
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
    Проверка статуса индекса и доступности системы.
    
    Returns:
        Информация о статусе системы
    """
    try:
        # Проверка что RAG инициализирована при старте сервера
        if qdrant_client is None:
            return "❌ Ошибка: RAG система не инициализирована. Проверьте логи сервера."
        
        # Подсчёт документов (используем count() для эффективности)
        try:
            from qdrant_storage import get_qdrant_count
            total_docs = get_qdrant_count(QDRANT_COLLECTION)
        except Exception:
            # Fallback: оценка через ограниченную выборку
            from qdrant_storage import get_all_points
            all_points = get_all_points(limit=10, include_payload=True, collection=QDRANT_COLLECTION)
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
        
        return (
            f"{status}\n"
            f"📊 Документов в индексе: {total_docs}\n"
            f"🔧 Модель эмбеддингов: {EMBED_MODEL}\n"
            f"💾 Путь к БД: {CHROMA_PATH}\n"
            f"🔄 Режим: {'Ollama' if USE_OLLAMA else 'HuggingFace'}\n"
            f"{rewrite_info}"
        )
    except Exception as e:
        logger.error(f"Ошибка health check: {e}", exc_info=True)
        return f"❌ Ошибка: {str(e)}"

if __name__ == "__main__":
    logger.info("MCP on 0.0.0.0:8012")
    try:
        mcp.run(transport="streamable-http", port=8012, host="0.0.0.0")
    except KeyboardInterrupt:
        pass
