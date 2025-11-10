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
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Конфигурация через ENV
ENABLE_CONTEXT_EXPANSION = os.getenv("ENABLE_CONTEXT_EXPANSION", "true").lower() == "true"
CONTEXT_EXPANSION_MODE = os.getenv("CONTEXT_EXPANSION_MODE", "bidirectional").lower()
CONTEXT_EXPANSION_SIZE = int(os.getenv("CONTEXT_EXPANSION_SIZE", "2"))  # ±N chunks


def expand_context_bidirectional(
    result: Dict[str, Any],
    collection: Any,
    context_size: int = None
) -> Dict[str, Any]:
    """
    Расширить контекст bidirectionally (±N chunks).
    
    Args:
        result: Результат поиска
        collection: ChromaDB коллекция
        context_size: Размер контекста (±N chunks, по умолчанию из ENV)
        
    Returns:
        Результат с расширенным контекстом
    """
    if context_size is None:
        context_size = CONTEXT_EXPANSION_SIZE
    
    if not result or not isinstance(result, dict):
        return result
    
    if collection is None:
        logger.debug("Collection недоступна для context expansion")
        result['expanded_text'] = result.get('text', '')
        result['context_chunks'] = 1
        return result
    
    try:
        metadata = result.get('metadata')
        if not metadata or not isinstance(metadata, dict):
            result['expanded_text'] = result.get('text', '')
            result['context_chunks'] = 1
            return result
        
        chunk_num = metadata.get('chunk', 0)
        page_id = metadata.get('page_id')
        
        if not page_id:
            result['expanded_text'] = result.get('text', '')
            result['context_chunks'] = 1
            return result
        
        # Запрашиваем соседние чанки из той же страницы (±context_size)
        min_chunk = max(0, chunk_num - context_size)
        max_chunk = chunk_num + context_size
        
        # Получаем чанки в диапазоне
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
        
        # Безопасная проверка
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
            expanded_text = '\n\n'.join([c['text'] for c in chunk_data if c['text']])
            result['expanded_text'] = expanded_text
            result['context_chunks'] = len(chunk_data)
            result['expansion_mode'] = 'bidirectional'
            result['context_size'] = context_size
            
            logger.debug(f"Bidirectional expansion: chunk {chunk_num} ±{context_size} = {len(chunk_data)} chunks")
        else:
            result['expanded_text'] = result.get('text', '')
            result['context_chunks'] = 1
            result['expansion_mode'] = 'none'
            
    except Exception as e:
        logger.warning(f"Bidirectional context expansion failed: {e}")
        result['expanded_text'] = result.get('text', '')
        result['context_chunks'] = 1
        result['expansion_mode'] = 'error'
    
    return result


def expand_context_with_related(
    result: Dict[str, Any],
    collection: Any,
    embeddings_model: Any = None,
    top_k: int = 2
) -> Dict[str, Any]:
    """
    Добавить похожие чанки на той же странице на основе семантического сходства.
    
    Args:
        result: Результат поиска
        collection: ChromaDB коллекция
        embeddings_model: Модель для вычисления embeddings (опционально)
        top_k: Количество похожих чанков
        
    Returns:
        Результат с расширенным контекстом
    """
    if not result or not isinstance(result, dict):
        return result
    
    if collection is None:
        logger.debug("Collection недоступна для related context expansion")
        result['expanded_text'] = result.get('text', '')
        result['context_chunks'] = 1
        return result
    
    try:
        metadata = result.get('metadata')
        if not metadata or not isinstance(metadata, dict):
            result['expanded_text'] = result.get('text', '')
            result['context_chunks'] = 1
            return result
        
        chunk_num = metadata.get('chunk', 0)
        page_id = metadata.get('page_id')
        text = result.get('text', '')
        
        if not page_id or not text:
            result['expanded_text'] = text
            result['context_chunks'] = 1
            return result
        
        # Получаем все чанки с той же страницы
        page_chunks = collection.get(
            where={'page_id': page_id},
            include=['documents', 'metadatas', 'ids']
        )
        
        if not page_chunks or not page_chunks.get('documents'):
            result['expanded_text'] = text
            result['context_chunks'] = 1
            return result
        
        # Находим похожие чанки (исключая текущий)
        similar_chunks = []
        current_chunk_id = result.get('id')
        
        if embeddings_model:
            # Используем семантическое сходство
            try:
                current_embedding = embeddings_model.get_query_embedding(text)
                
                for i, chunk_text in enumerate(page_chunks['documents']):
                    chunk_id = page_chunks['ids'][i] if i < len(page_chunks.get('ids', [])) else None
                    chunk_meta = page_chunks['metadatas'][i] if i < len(page_chunks.get('metadatas', [])) else {}
                    
                    # Пропускаем текущий чанк
                    if chunk_id == current_chunk_id or not chunk_text:
                        continue
                    
                    # Вычисляем сходство
                    chunk_embedding = embeddings_model.get_query_embedding(chunk_text)
                    
                    # Cosine similarity
                    try:
                        import numpy as np
                        similarity = np.dot(current_embedding, chunk_embedding) / (
                            np.linalg.norm(current_embedding) * np.linalg.norm(chunk_embedding)
                        )
                    except ImportError:
                        # Fallback без numpy (простая dot product нормализация)
                        dot_product = sum(a * b for a, b in zip(current_embedding, chunk_embedding))
                        norm_a = sum(a * a for a in current_embedding) ** 0.5
                        norm_b = sum(b * b for b in chunk_embedding) ** 0.5
                        similarity = dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0
                    
                    similar_chunks.append({
                        'text': chunk_text,
                        'similarity': float(similarity),
                        'chunk_num': chunk_meta.get('chunk', 0) if chunk_meta else 0
                    })
            except Exception as e:
                logger.debug(f"Semantic similarity failed: {e}, using simple text matching")
                # Fallback: просто берём другие чанки с той же страницы
                for i, chunk_text in enumerate(page_chunks['documents']):
                    chunk_id = page_chunks['ids'][i] if i < len(page_chunks.get('ids', [])) else None
                    if chunk_id != current_chunk_id and chunk_text:
                        chunk_meta = page_chunks['metadatas'][i] if i < len(page_chunks.get('metadatas', [])) else {}
                        similar_chunks.append({
                            'text': chunk_text,
                            'similarity': 0.5,  # Дефолтное значение
                            'chunk_num': chunk_meta.get('chunk', 0) if chunk_meta else 0
                        })
        else:
            # Fallback: просто берём другие чанки с той же страницы
            for i, chunk_text in enumerate(page_chunks['documents']):
                chunk_id = page_chunks['ids'][i] if i < len(page_chunks.get('ids', [])) else None
                if chunk_id != current_chunk_id and chunk_text:
                    chunk_meta = page_chunks['metadatas'][i] if i < len(page_chunks.get('metadatas', [])) else {}
                    similar_chunks.append({
                        'text': chunk_text,
                        'similarity': 0.5,
                        'chunk_num': chunk_meta.get('chunk', 0) if chunk_meta else 0
                    })
        
        # Сортируем по сходству и берём топ-K
        similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        top_similar = similar_chunks[:top_k]
        
        # Объединяем тексты
        related_texts = [chunk['text'] for chunk in top_similar]
        expanded_text = text + '\n\n--- Related chunks ---\n\n' + '\n\n'.join(related_texts)
        
        result['expanded_text'] = expanded_text
        result['context_chunks'] = 1 + len(top_similar)
        result['related_chunks_count'] = len(top_similar)
        result['expansion_mode'] = 'related'
        
        logger.debug(f"Related expansion: {len(top_similar)} похожих чанков добавлено")
        
    except Exception as e:
        logger.warning(f"Related context expansion failed: {e}")
        result['expanded_text'] = result.get('text', '')
        result['context_chunks'] = 1
        result['expansion_mode'] = 'error'
    
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

