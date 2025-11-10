#!/usr/bin/env python3
"""
Hybrid Search: Vector + BM25 (full-text search)
Объединение результатов через Reciprocal Rank Fusion (RRF)

Это стандартный подход, используемый Google, OpenAI, Meta, Microsoft, AWS.
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


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

# Глобальный BM25 retriever (ленивая инициализация)
bm25_retriever = None
bm25_documents = None


def detect_query_intent(query: str) -> QueryIntent:
    """
    Определить тип запроса для адаптивных весов.
    
    Простые эвристики:
    - Navigational: "где", "как найти", "url", "link", "страница"
    - Exploratory: "какие", "сравни", "список", "все"
    - Factual: "какой", "что", "когда", "кто"
    - HowTo: "как", "инструкция", "настроить"
    
    Args:
        query: Поисковый запрос
        
    Returns:
        QueryIntent enum
    """
    query_lower = query.lower()
    
    # Navigational: поиск конкретной информации
    navigational_keywords = ['где', 'url', 'ссылка', 'link', 'find', 'где найти', 'найди', 'покажи', 'страница', 'документ']
    if any(kw in query_lower for kw in navigational_keywords):
        return QueryIntent.NAVIGATIONAL
    
    # HowTo: инструкции
    howto_keywords = ['как', 'инструкция', 'настроить', 'установить', 'запустить', 'сделать']
    if any(kw in query_lower for kw in howto_keywords):
        return QueryIntent.HOWTO
    
    # Factual: факты
    factual_keywords = ['какой', 'какая', 'какие', 'что', 'когда', 'кто', 'сколько']
    if any(kw in query_lower for kw in factual_keywords):
        return QueryIntent.FACTUAL
    
    # Exploratory: исследовательский поиск
    exploratory_keywords = ['какие', 'сравни', 'список', 'все', 'перечисли']
    if any(kw in query_lower for kw in exploratory_keywords):
        return QueryIntent.EXPLORATORY
    
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


def init_bm25_retriever(collection: Any) -> Optional[Any]:
    """
    Инициализирует BM25 retriever из документов ChromaDB.
    
    Args:
        collection: ChromaDB коллекция
        
    Returns:
        BM25Retriever или None если не удалось инициализировать
    """
    global bm25_retriever, bm25_documents
    
    if not ENABLE_HYBRID_SEARCH:
        logger.info("Hybrid Search отключен (ENABLE_HYBRID_SEARCH=false)")
        return None
    
    if bm25_retriever is not None:
        return bm25_retriever
    
    try:
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core import Document
        from llama_index.core.node_parser import SimpleNodeParser
        
        logger.info("Инициализация BM25 retriever...")
        
        # Загружаем все документы из ChromaDB
        # Ограничиваем для предотвращения OOM (можно увеличить при необходимости)
        MAX_DOCS_FOR_BM25 = int(os.getenv("BM25_MAX_DOCS", "50000"))
        
        all_data = collection.get(limit=MAX_DOCS_FOR_BM25)
        doc_count = len(all_data.get('ids', []))
        
        if doc_count == 0:
            logger.warning("Нет документов для BM25 индексации")
            return None
        
        logger.info(f"Загрузка {doc_count} документов для BM25 индексации...")
        
        # Создаем Document объекты из ChromaDB данных
        documents = []
        for idx in range(doc_count):
            doc_id = all_data['ids'][idx] if idx < len(all_data.get('ids', [])) else None
            doc_text = all_data['documents'][idx] if idx < len(all_data.get('documents', [])) else None
            doc_metadata = all_data['metadatas'][idx] if idx < len(all_data.get('metadatas', [])) else None
            
            if doc_id and doc_text:
                # Создаем Document с текстом и метаданными
                doc = Document(
                    text=doc_text,
                    metadata=doc_metadata if doc_metadata else {},
                    id_=doc_id
                )
                documents.append(doc)
        
        if not documents:
            logger.warning("Не удалось создать документы для BM25")
            return None
        
        # Конвертируем Document в Node для BM25Retriever
        # Используем большой chunk_size и отключаем metadata-aware splitting
        # для обработки документов с большими метаданными
        from llama_index.core.node_parser import SentenceSplitter
        node_parser = SentenceSplitter(
            chunk_size=4096,
            chunk_overlap=200,
            include_metadata=False  # Отключаем metadata-aware для избежания ошибок
        )
        nodes = node_parser.get_nodes_from_documents(documents)
        
        # Создаем BM25 retriever используя from_defaults с nodes
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, language='ru')
        bm25_documents = documents
        
        logger.info(f"✅ BM25 retriever готов. Индексировано {len(nodes)} nodes из {len(documents)} документов")
        return bm25_retriever
        
    except ImportError as e:
        logger.warning(f"BM25 retriever недоступен: {e}. Установите llama-index-retrievers-bm25")
        return None
    except Exception as e:
        logger.error(f"Ошибка инициализации BM25 retriever: {e}", exc_info=True)
        return None


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
        # Безопасное извлечение doc_id
        # result - это словарь из hybrid_search, но может содержать объект NodeWithScore в ключе 'node'
        doc_id = result.get('id')
        
        if not doc_id:
            # Пытаемся извлечь из node объекта (NodeWithScore)
            node = result.get('node')
            if node:
                # NodeWithScore - это объект, используем атрибуты, а не .get()
                if hasattr(node, 'node_id'):
                    doc_id = node.node_id
                elif hasattr(node, 'id_'):
                    doc_id = node.id_
        
        if not doc_id:
            # Пытаемся извлечь из metadata
            metadata = result.get('metadata', {})
            if not metadata:
                node = result.get('node')
                if node and hasattr(node, 'metadata'):
                    node_metadata = node.metadata
                    # metadata может быть словарем или объектом
                    if isinstance(node_metadata, dict):
                        metadata = node_metadata
                    elif hasattr(node_metadata, '__dict__'):
                        metadata = node_metadata.__dict__
                    else:
                        metadata = {}
            
            if metadata and isinstance(metadata, dict):
                doc_id = metadata.get('page_id')
        
        if doc_id:
            if doc_id not in rrf_scores:
                # Новый результат из BM25
                text = result.get('text', '')
                metadata = result.get('metadata', {})
                
                # Если text или metadata отсутствуют, пытаемся извлечь из node
                if not text or not metadata:
                    node = result.get('node')
                    if node:
                        if not text:
                            if hasattr(node, 'text'):
                                text = node.text
                            elif hasattr(node, 'get_content'):
                                text = node.get_content()
                        
                        if not metadata:
                            if hasattr(node, 'metadata'):
                                node_metadata = node.metadata
                                if isinstance(node_metadata, dict):
                                    metadata = node_metadata
                                elif hasattr(node_metadata, '__dict__'):
                                    metadata = node_metadata.__dict__
                                else:
                                    metadata = {}
                
                # Убеждаемся, что metadata - это словарь
                if not isinstance(metadata, dict):
                    metadata = {}
                
                rrf_scores[doc_id] = {
                    'id': doc_id,
                    'text': text,
                    'metadata': metadata,
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
    collection: Any,
    vector_results: List[Dict[str, Any]],
    space_filter: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Выполняет Hybrid Search: объединяет Vector + BM25 результаты через RRF.
    
    Использует адаптивные веса на основе типа запроса (navigational/exploratory/factual/howto).
    
    Args:
        query: Поисковый запрос
        collection: ChromaDB коллекция (для инициализации BM25)
        vector_results: Результаты векторного поиска
        space_filter: Фильтр по пространству (применяется к BM25)
        limit: Максимальное количество результатов
        
    Returns:
        Объединенные результаты, отсортированные по RRF score
    """
    if not ENABLE_HYBRID_SEARCH:
        logger.debug("Hybrid Search отключен, возвращаю только векторные результаты")
        return vector_results[:limit]
    
    # ============ НОВОЕ: Определение query intent для адаптивных весов ============
    query_intent = detect_query_intent(query)
    vector_weight, bm25_weight = get_adaptive_weights(query_intent)
    logger.debug(f"Query intent: {query_intent.value}, adaptive weights: vector={vector_weight:.2f}, bm25={bm25_weight:.2f}")
    
    # Инициализируем BM25 retriever если еще не инициализирован
    bm25 = init_bm25_retriever(collection)
    if not bm25:
        logger.debug("BM25 недоступен, возвращаю только векторные результаты")
        return vector_results[:limit]
    
    try:
        # Выполняем BM25 поиск
        logger.debug(f"Выполняю BM25 поиск для: '{query[:50]}...'")
        bm25_nodes = bm25.retrieve(query)
        
        # Преобразуем результаты BM25 в формат словарей
        bm25_results = []
        for node in bm25_nodes:
            doc_id = node.node_id if hasattr(node, 'node_id') else node.id_
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            text = node.text if hasattr(node, 'text') else node.get_content()
            
            # Применяем фильтр по space если указан
            if space_filter:
                node_space = metadata.get('space', '')
                if node_space != space_filter:
                    continue
            
            bm25_results.append({
                'id': doc_id,
                'text': text,
                'metadata': metadata,
                'node': node,
                'score': node.score if hasattr(node, 'score') else 0.0
            })
        
        logger.debug(f"BM25 нашел {len(bm25_results)} результатов")
        
        # Объединяем через RRF с адаптивными весами
        merged_results = reciprocal_rank_fusion(
            vector_results, 
            bm25_results, 
            k=RRF_K,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        
        logger.info(f"Hybrid Search ({query_intent.value}): Vector={len(vector_results)}, BM25={len(bm25_results)}, Merged={len(merged_results)}, weights=[{vector_weight:.2f}, {bm25_weight:.2f}]")
        
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
        'bm25_initialized': bm25_retriever is not None,
        'bm25_documents': len(bm25_documents) if bm25_documents else 0
    }

