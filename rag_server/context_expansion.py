#!/usr/bin/env python3
"""
Context Expansion: Bidirectional + Related chunks

Расширение контекста найденных результатов для улучшения релевантности.
Поддерживает несколько режимов:
- bidirectional: ±N соседних чанков
- related: похожие чанки на основе семантического сходства
- parent: родительские разделы
- all: все режимы вместе
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple

# Pydantic config
from rag_server.config import settings

logger = logging.getLogger(__name__)


def _validate_result_and_collection(
    result: Dict[str, Any], 
    collection: Any
) -> Optional[Tuple[str, str, str]]:
    """
    Общая валидация для всех функций.
    Returns: (page_id, text, current_id) or None
    """
    if not result or not isinstance(result, dict):
        return None
    
    if collection is None:
        logger.debug("Collection недоступна")
        return None
    
    metadata = result.get('metadata')
    if not metadata or not isinstance(metadata, dict):
        return None
    
    page_id = metadata.get('page_id')
    text = result.get('text', '')
    current_id = str(result.get('id', ''))
    
    if not page_id or not text:
        return None
    
    return (page_id, text, current_id)


def _default_result(result: Dict[str, Any], mode: str = 'none') -> Dict[str, Any]:
    """Возвращает дефолтный результат без расширения."""
    if not isinstance(result, dict):
        return {}
    result['expanded_text'] = result.get('text', '')
    result['context_chunks'] = 1
    result['expansion_mode'] = mode
    return result


async def _get_page_chunks_async(collection: Any, page_id: str) -> Optional[Dict]:
    """Получает все чанки страницы (Async)."""
    try:
        # Проверяем, является ли collection AsyncQdrantClient
        # В Qdrant API методы называются scroll/search, а не get
        # Но в qdrant_storage есть обертки.
        # Если collection - это raw QdrantClient (Async), используем scroll
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http import models

        if isinstance(collection, AsyncQdrantClient):
            # Используем scroll для получения всех точек с фильтром
            points, _ = await collection.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.page_id",
                            match=models.MatchValue(value=page_id)
                        )
                    ]
                ),
                limit=100, # Ограничение на количество чанков страницы
                with_payload=True,
                with_vectors=False # Векторы не нужны, только текст
            )
            
            return {
                'documents': [p.payload.get('text', '') for p in points],
                'metadatas': [p.payload for p in points],
                'ids': [p.id for p in points]
            }
        else:
            # Fallback for sync client or Chroma-like interface (deprecated but kept for safety)
            # В новой архитектуре мы используем QdrantClient напрямую или через qdrant_storage helpers
            # Но qdrant_storage helpers пока не экспортированы сюда.
            # Предполагаем, что collection - это AsyncQdrantClient
            logger.warning(f"Unknown collection type in context expansion: {type(collection)}")
            return None

    except Exception as e:
        logger.warning(f"Error getting page chunks: {e}")
        return None


def _compute_similarity(emb1: List[float], emb2: List[float]) -> float:
    """Cosine similarity с fallback."""
    try:
        import numpy as np
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    except ImportError:
        # Fallback без numpy
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm_a = sum(a * a for a in emb1) ** 0.5
        norm_b = sum(b * b for b in emb2) ** 0.5
        return dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0
    except Exception:
        return 0.0


async def _find_similar_chunks_with_embeddings_async(
    current_text: str,
    current_id: str,
    page_chunks: Dict,
    embeddings_model: Any
) -> List[Dict]:
    """Находит похожие чанки через embeddings (Async)."""
    similar_chunks = []
    
    try:
        current_embedding = await embeddings_model.get_query_embedding_async(current_text)
        
        # Оптимизация: Batch embedding generation
        chunk_texts = []
        chunk_indices = []
        
        for i, text in enumerate(page_chunks['documents']):
            chunk_id = str(page_chunks['ids'][i]) if i < len(page_chunks['ids']) else ''
            if chunk_id == current_id or not text:
                continue
            chunk_texts.append(text)
            chunk_indices.append(i)
            
        if not chunk_texts:
            return []

        # Пытаемся получить batch embeddings
        try:
            chunk_embeddings = await embeddings_model.get_text_embeddings_async(chunk_texts)
        except Exception:
            # Fallback to sequential
            chunk_embeddings = []
            for t in chunk_texts:
                emb = await embeddings_model.get_query_embedding_async(t)
                chunk_embeddings.append(emb)
            
        for idx, embedding in enumerate(chunk_embeddings):
            original_idx = chunk_indices[idx]
            similarity = _compute_similarity(current_embedding, embedding)
            
            chunk_meta = page_chunks['metadatas'][original_idx] if original_idx < len(page_chunks['metadatas']) else {}
            
            similar_chunks.append({
                'text': chunk_texts[idx],
                'similarity': similarity,
                'chunk_num': chunk_meta.get('chunk', 0) if chunk_meta else 0
            })
            
    except Exception as e:
        logger.debug(f"Semantic similarity calculation failed: {e}")
        return [] # Вернет пустой список, сработает fallback
        
    return similar_chunks


def _find_similar_chunks_simple(
    current_id: str,
    page_chunks: Dict
) -> List[Dict]:
    """Fallback: просто возвращает другие чанки."""
    similar_chunks = []
    
    for i, text in enumerate(page_chunks['documents']):
        chunk_id = str(page_chunks['ids'][i]) if i < len(page_chunks['ids']) else ''
        
        if chunk_id == current_id or not text:
            continue
            
        chunk_meta = page_chunks['metadatas'][i] if i < len(page_chunks['metadatas']) else {}
        
        similar_chunks.append({
            'text': text,
            'similarity': 0.5,
            'chunk_num': chunk_meta.get('chunk', 0) if chunk_meta else 0
        })
        
    return similar_chunks


async def _get_bidirectional_chunks_async(
    collection: Any,
    page_id: str,
    chunk_num: int,
    context_size: int
) -> List[Dict]:
    """Получает соседние чанки (±N) (Async)."""
    min_chunk = max(0, chunk_num - context_size)
    max_chunk = chunk_num + context_size

    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http import models

        if isinstance(collection, AsyncQdrantClient):
            # Используем scroll с фильтром по диапазону chunk
            points, _ = await collection.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.page_id",
                            match=models.MatchValue(value=page_id)
                        ),
                        models.FieldCondition(
                            key="metadata.chunk",
                            range=models.Range(gte=min_chunk, lte=max_chunk)
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            chunk_data = []
            for p in points:
                chunk_meta = p.payload
                if chunk_meta:
                    chunk_data.append({
                        'chunk_num': chunk_meta.get('chunk', 0),
                        'text': chunk_meta.get('text', '') # В Qdrant текст часто дублируется в payload для удобства
                    })
            
            chunk_data.sort(key=lambda x: x['chunk_num'])
            return chunk_data
            
        else:
            logger.warning(f"Unknown collection type: {type(collection)}")
            return []

    except Exception as e:
        logger.warning(f"Error getting bidirectional chunks: {e}")
        return []


async def expand_context_bidirectional_async(
    result: Dict[str, Any],
    collection: Any,
    context_size: int = None
) -> Dict[str, Any]:
    """
    Расширить контекст bidirectionally (±N chunks) (Async).
    """
    if context_size is None:
        context_size = settings.context_expansion_size

    validation = _validate_result_and_collection(result, collection)
    if not validation:
        return _default_result(result)

    page_id, _, _ = validation
    chunk_num = result.get('metadata', {}).get('chunk', 0)

    try:
        chunk_data = await _get_bidirectional_chunks_async(collection, page_id, chunk_num, context_size)
        
        if chunk_data:
            # Фильтруем пустые тексты
            expanded_text = '\n\n'.join([c['text'] for c in chunk_data if c['text']])
            
            # Если текст пустой (например, не было в payload), пробуем взять исходный result text
            if not expanded_text and result.get('text'):
                 expanded_text = result.get('text')

            result['expanded_text'] = expanded_text
            result['context_chunks'] = len(chunk_data)
            result['expansion_mode'] = 'bidirectional'
            result['context_size'] = context_size
            logger.debug(f"Bidirectional: chunk {chunk_num} ±{context_size} -> {len(chunk_data)} chunks")
        else:
            return _default_result(result)

    except Exception as e:
        logger.warning(f"Bidirectional expansion failed: {e}")
        return _default_result(result, 'error')

    return result


async def expand_context_with_related_async(
    result: Dict[str, Any],
    collection: Any,
    embeddings_model: Any = None,
    top_k: int = 2
) -> Dict[str, Any]:
    """
    Добавить похожие чанки на той же странице (Async).
    """
    validation = _validate_result_and_collection(result, collection)
    if not validation:
        return _default_result(result)

    page_id, text, current_id = validation

    try:
        page_chunks = await _get_page_chunks_async(collection, page_id)
        if not page_chunks or not page_chunks.get('documents'):
            return _default_result(result)

        if embeddings_model:
            similar_chunks = await _find_similar_chunks_with_embeddings_async(
                text, current_id, page_chunks, embeddings_model
            )
        else:
            similar_chunks = _find_similar_chunks_simple(current_id, page_chunks)

        # Сортировка и выбор топ-K
        similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        top_similar = similar_chunks[:top_k]

        if top_similar:
            related_texts = [chunk['text'] for chunk in top_similar]
            expanded_text = text + '\n\n--- Related chunks ---\n\n' + '\n\n'.join(related_texts)
            
            result['expanded_text'] = expanded_text
            result['context_chunks'] = 1 + len(top_similar)
            result['related_chunks_count'] = len(top_similar)
            result['expansion_mode'] = 'related'
            logger.debug(f"Related expansion: {len(top_similar)} chunks added")
        else:
            return _default_result(result)

    except Exception as e:
        logger.warning(f"Related expansion failed: {e}")
        return _default_result(result, 'error')

    return result


async def expand_context_full_async(
    result: Dict[str, Any],
    collection: Any,
    embeddings_model: Any = None,
    expansion_mode: str = None,
    context_size: int = None
) -> Dict[str, Any]:
    """
    Полная context expansion с выбором режима (Async).

    Args:
        result: Результат поиска
        collection: QdrantClient (Async)
        embeddings_model: Модель для вычисления embeddings (опционально)
        expansion_mode: Режим расширения ('bidirectional', 'related', 'parent', 'all')
        context_size: Размер контекста для bidirectional режима

    Returns:
        Результат с расширенным контекстом
    """
    if not settings.enable_context_expansion:
        result['expanded_text'] = result.get('text', '')
        result['context_chunks'] = 1
        result['expansion_mode'] = 'disabled'
        return result

    if expansion_mode is None:
        expansion_mode = settings.context_expansion_mode

    if context_size is None:
        context_size = settings.context_expansion_size

    try:
        if expansion_mode == 'bidirectional':
            result = await expand_context_bidirectional_async(result, collection, context_size)
        elif expansion_mode == 'related':
            result = await expand_context_with_related_async(result, collection, embeddings_model, top_k=context_size)
        elif expansion_mode == 'parent':
            # Режим parent: используем bidirectional как основу
            result = await expand_context_bidirectional_async(result, collection, context_size)
            # TODO: Добавить логику для родительских разделов
        elif expansion_mode == 'all':
            # Все режимы вместе: сначала bidirectional, потом related
            result = await expand_context_bidirectional_async(result, collection, context_size)
            result = await expand_context_with_related_async(result, collection, embeddings_model, top_k=context_size)
            result['expansion_mode'] = 'all'
        else:
            logger.warning(f"Неизвестный режим expansion: {expansion_mode}, используем bidirectional")
            result = await expand_context_bidirectional_async(result, collection, context_size)

        logger.debug(f"Context expansion ({expansion_mode}): применён к результату")

    except Exception as e:
        logger.error(f"Ошибка в context expansion: {e}")
        result['expanded_text'] = result.get('text', '')
        result['context_chunks'] = 1
        result['expansion_mode'] = 'error'

    return result
