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

import os
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Конфигурация через ENV
ENABLE_CONTEXT_EXPANSION = os.getenv("ENABLE_CONTEXT_EXPANSION", "true").lower() == "true"
CONTEXT_EXPANSION_MODE = os.getenv("CONTEXT_EXPANSION_MODE", "bidirectional").lower()
CONTEXT_EXPANSION_SIZE = int(os.getenv("CONTEXT_EXPANSION_SIZE", "2"))  # ±N chunks


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


def _get_page_chunks(collection: Any, page_id: str) -> Optional[Dict]:
    """Получает все чанки страницы."""
    try:
        return collection.get(
            where={'page_id': page_id},
            include=['documents', 'metadatas', 'ids']
        )
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


def _find_similar_chunks_with_embeddings(
    current_text: str,
    current_id: str,
    page_chunks: Dict,
    embeddings_model: Any
) -> List[Dict]:
    """Находит похожие чанки через embeddings."""
    similar_chunks = []
    
    try:
        current_embedding = embeddings_model.get_query_embedding(current_text)
        
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
            chunk_embeddings = embeddings_model.get_text_embeddings(chunk_texts)
        except Exception:
            # Fallback to sequential
            chunk_embeddings = [embeddings_model.get_query_embedding(t) for t in chunk_texts]
            
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


def _get_bidirectional_chunks(
    collection: Any,
    page_id: str,
    chunk_num: int,
    context_size: int
) -> List[Dict]:
    """Получает соседние чанки (±N)."""
    min_chunk = max(0, chunk_num - context_size)
    max_chunk = chunk_num + context_size

    neighbors = collection.get(
        where={
            '$and': [
                {'page_id': page_id},
                {'chunk': {'$gte': min_chunk}},
                {'chunk': {'$lte': max_chunk}}
            ]
        },
        include=['documents', 'metadatas']
    )

    chunk_data = []
    if (neighbors and neighbors.get('documents') and 
        len(neighbors['documents']) == len(neighbors.get('metadatas', []))):
        
        for i, doc in enumerate(neighbors['documents']):
            chunk_meta = neighbors['metadatas'][i]
            if chunk_meta:
                chunk_data.append({
                    'chunk_num': chunk_meta.get('chunk', 0),
                    'text': doc if doc else ''
                })
                
        chunk_data.sort(key=lambda x: x['chunk_num'])
        
    return chunk_data


def expand_context_bidirectional(
    result: Dict[str, Any],
    collection: Any,
    context_size: int = None
) -> Dict[str, Any]:
    """
    Расширить контекст bidirectionally (±N chunks).
    """
    if context_size is None:
        context_size = CONTEXT_EXPANSION_SIZE

    validation = _validate_result_and_collection(result, collection)
    if not validation:
        return _default_result(result)

    page_id, _, _ = validation
    chunk_num = result.get('metadata', {}).get('chunk', 0)

    try:
        chunk_data = _get_bidirectional_chunks(collection, page_id, chunk_num, context_size)
        
        if chunk_data:
            expanded_text = '\n\n'.join([c['text'] for c in chunk_data if c['text']])
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


def expand_context_with_related(
    result: Dict[str, Any],
    collection: Any,
    embeddings_model: Any = None,
    top_k: int = 2
) -> Dict[str, Any]:
    """
    Добавить похожие чанки на той же странице.
    """
    validation = _validate_result_and_collection(result, collection)
    if not validation:
        return _default_result(result)

    page_id, text, current_id = validation

    try:
        page_chunks = _get_page_chunks(collection, page_id)
        if not page_chunks or not page_chunks.get('documents'):
            return _default_result(result)

        if embeddings_model:
            similar_chunks = _find_similar_chunks_with_embeddings(
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


def expand_context_full(
    result: Dict[str, Any],
    collection: Any,
    embeddings_model: Any = None,
    expansion_mode: str = None,
    context_size: int = None
) -> Dict[str, Any]:
    """
    Полная context expansion с выбором режима.

    Args:
        result: Результат поиска
        collection: ChromaDB коллекция
        embeddings_model: Модель для вычисления embeddings (опционально)
        expansion_mode: Режим расширения ('bidirectional', 'related', 'parent', 'all')
        context_size: Размер контекста для bidirectional режима

    Returns:
        Результат с расширенным контекстом
    """
    if not ENABLE_CONTEXT_EXPANSION:
        result['expanded_text'] = result.get('text', '')
        result['context_chunks'] = 1
        result['expansion_mode'] = 'disabled'
        return result

    if expansion_mode is None:
        expansion_mode = CONTEXT_EXPANSION_MODE

    if context_size is None:
        context_size = CONTEXT_EXPANSION_SIZE

    try:
        if expansion_mode == 'bidirectional':
            result = expand_context_bidirectional(result, collection, context_size)
        elif expansion_mode == 'related':
            result = expand_context_with_related(result, collection, embeddings_model, top_k=context_size)
        elif expansion_mode == 'parent':
            # Режим parent: используем bidirectional как основу
            result = expand_context_bidirectional(result, collection, context_size)
            # TODO: Добавить логику для родительских разделов
        elif expansion_mode == 'all':
            # Все режимы вместе: сначала bidirectional, потом related
            result = expand_context_bidirectional(result, collection, context_size)
            result = expand_context_with_related(result, collection, embeddings_model, top_k=context_size)
            result['expansion_mode'] = 'all'
        else:
            logger.warning(f"Неизвестный режим expansion: {expansion_mode}, используем bidirectional")
            result = expand_context_bidirectional(result, collection, context_size)

        logger.debug(f"Context expansion ({expansion_mode}): применён к результату")

    except Exception as e:
        logger.error(f"Ошибка в context expansion: {e}")
        result['expanded_text'] = result.get('text', '')
        result['context_chunks'] = 1
        result['expansion_mode'] = 'error'

    return result

