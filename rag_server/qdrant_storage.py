"""
Qdrant vector store для хранения embeddings.
Замена ChromaDB для лучшей масштабируемости.
"""
import os
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range, PayloadSchemaType

# Инициализация logger (должен быть до использования)
logger = logging.getLogger(__name__)

def extract_text_from_payload(payload: Dict[str, Any]) -> str:
    """
    Извлечь текст из payload Qdrant.
    """
    # Сначала проверяем прямое поле 'text'
    text = payload.get('text', '')
    if text:
        return text

    # Если text пуст, пытаемся извлечь из _node_content (LlamaIndex формат)
    node_content = payload.get('_node_content', '')
    if node_content:
        try:
            node_data = json.loads(node_content)
            text = node_data.get('text', '') or node_data.get('text_', '')
            if text:
                return text
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    return ''

# Импорт MMR reranker (опционально, чтобы не ломать если модуль недоступен)
try:
    from mmr_reranker import mmr_rerank
    HAS_MMR = True
except ImportError:
    try:
        from rag_server.mmr_reranker import mmr_rerank
        HAS_MMR = True
    except ImportError:
        HAS_MMR = False
        logger.warning("MMR reranker not available (mmr_reranker module not found)")

# Qdrant settings
from rag_server.config import settings

qdrant_client = None
async_qdrant_client = None

def init_qdrant_client() -> QdrantClient:
    """Инициализировать синхронный Qdrant клиент."""
    global qdrant_client
    if qdrant_client is None:
        try:
            qdrant_client = QdrantClient(
                host=settings.qdrant_host, 
                port=settings.qdrant_port, 
                timeout=30,
                api_key=settings.qdrant_api_key
            )
            logger.info(f"✅ Qdrant client initialized: {settings.qdrant_host}:{settings.qdrant_port}")
        except Exception as e:
            logger.error(f"Ошибка инициализации Qdrant client: {e}")
            raise
    return qdrant_client

def init_async_qdrant_client() -> AsyncQdrantClient:
    """Инициализировать асинхронный Qdrant клиент."""
    global async_qdrant_client
    if async_qdrant_client is None:
        try:
            async_qdrant_client = AsyncQdrantClient(
                host=settings.qdrant_host, 
                port=settings.qdrant_port, 
                timeout=30,
                api_key=settings.qdrant_api_key
            )
            logger.info(f"✅ AsyncQdrant client initialized: {settings.qdrant_host}:{settings.qdrant_port}")
        except Exception as e:
            logger.error(f"Ошибка инициализации AsyncQdrant client: {e}")
            raise
    return async_qdrant_client

def init_qdrant_collection(embedding_dim: int) -> bool:
    """
    Инициализировать коллекцию Qdrant с индексами для метаданных.
    """
    client = init_qdrant_client()

    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        collection_created = False
        if settings.qdrant_collection not in collection_names:
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Created Qdrant collection: {settings.qdrant_collection} (dim={embedding_dim})")
            collection_created = True
        else:
            # Проверяем размерность существующей коллекции
            collection_info = client.get_collection(settings.qdrant_collection)
            existing_dim = collection_info.config.params.vectors.size
            if existing_dim != embedding_dim:
                logger.error(
                    f"Несовпадение размерности: Qdrant={existing_dim}D, Model={embedding_dim}D. "
                    f"Удалите коллекцию {settings.qdrant_collection} и перезапустите."
                )
                return False
            logger.info(f"✅ Qdrant collection exists: {settings.qdrant_collection} (dim={embedding_dim})")

        # Создаем payload индексы
        try:
            # Индексы для строковых полей (KEYWORD)
            keyword_fields = ['space', 'status', 'type', 'content_type', 'created_by', 'modified_by', 'page_path', 'author', 'page_id']
            for field in keyword_fields:
                try:
                    client.create_payload_index(
                        collection_name=settings.qdrant_collection,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                except Exception:
                    pass

            # Индекс для labels (TEXT)
            try:
                client.create_payload_index(
                    collection_name=settings.qdrant_collection,
                    field_name="labels",
                    field_schema=PayloadSchemaType.TEXT
                )
            except Exception:
                pass

            # Индекс для headings (TEXT)
            try:
                client.create_payload_index(
                    collection_name=settings.qdrant_collection,
                    field_name="headings",
                    field_schema=PayloadSchemaType.TEXT
                )
            except Exception:
                pass

            # Индекс для title (TEXT)
            try:
                client.create_payload_index(
                    collection_name=settings.qdrant_collection,
                    field_name="title",
                    field_schema=PayloadSchemaType.TEXT
                )
            except Exception:
                pass

            # Индексы для числовых полей (INTEGER)
            integer_fields = ['hierarchy_depth', 'version', 'children_count', 'heading_count', 'created', 'modified']
            for field in integer_fields:
                try:
                    client.create_payload_index(
                        collection_name=settings.qdrant_collection,
                        field_name=field,
                        field_schema=PayloadSchemaType.INTEGER
                    )
                except Exception:
                    pass

            if collection_created:
                logger.info("✅ Created payload indexes for metadata filtering")
        except Exception as e:
            logger.warning(f"Could not create some indexes (may already exist): {e}")

        return True
    except Exception as e:
        logger.error(f"Ошибка инициализации Qdrant collection: {e}")
        return False

def clear_qdrant_collection() -> int:
    """
    Полностью удалить коллекцию Qdrant.
    Возвращает количество удаленных точек (примерно).
    """
    client = init_qdrant_client()
    count = get_qdrant_count()
    
    try:
        client.delete_collection(settings.qdrant_collection)
        logger.info(f"✅ Коллекция {settings.qdrant_collection} удалена")
        return count
    except Exception as e:
        logger.error(f"Ошибка удаления коллекции: {e}")
        return 0

# ... insert functions can remain sync for now as sync logic is heavy, 
# or can be ported if needed. Focus on search first.

def _parse_where_filter(where_filter: Dict) -> List[FieldCondition]:
    """Парсит where_filter в список условий Qdrant."""
    conditions = []
    
    if 'must' in where_filter:
        for condition in where_filter['must']:
            if isinstance(condition, dict):
                key = condition.get('key')
                if not key: continue
                
                if condition.get('match'):
                    val = condition['match'].get('value') or condition['match'].get('text')
                    if val: conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
                elif condition.get('range'):
                    conditions.append(FieldCondition(key=key, range=Range(**condition['range'])))
                    
    elif '$and' in where_filter:
        for condition in where_filter['$and']:
            if isinstance(condition, dict):
                for key, value in condition.items():
                    if isinstance(value, dict):
                        # range operators
                        kwargs = {}
                        if '$gte' in value: kwargs['gte'] = value['$gte']
                        if '$lte' in value: kwargs['lte'] = value['$lte']
                        if '$gt' in value: kwargs['gt'] = value['$gt']
                        if '$lt' in value: kwargs['lt'] = value['$lt']
                        if kwargs: conditions.append(FieldCondition(key=key, range=Range(**kwargs)))
                    else:
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    else:
        for key, value in where_filter.items():
            if isinstance(value, dict):
                kwargs = {}
                if '$gte' in value: kwargs['gte'] = value['$gte']
                if '$lte' in value: kwargs['lte'] = value['$lte']
                if kwargs: conditions.append(FieldCondition(key=key, range=Range(**kwargs)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                
    return conditions

def _build_metadata_conditions(
    space: Optional[str] = None,
    author: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    status: Optional[str] = None,
    content_type: Optional[str] = None,
    labels: Optional[List[str]] = None,
    page_path: Optional[str] = None,
    search_headings: Optional[str] = None
) -> List[FieldCondition]:
    """Строит условия фильтрации из прямых параметров метаданных."""
    conditions = []
    
    if space:
        conditions.append(FieldCondition(key="space", match=MatchValue(value=space)))
    if author:
        conditions.append(FieldCondition(key="created_by", match=MatchValue(value=author)))
    if status:
        conditions.append(FieldCondition(key="status", match=MatchValue(value=status)))
    if content_type:
        conditions.append(FieldCondition(key="content_type", match=MatchValue(value=content_type)))
    if from_date:
        conditions.append(FieldCondition(key="created", range=Range(gte=from_date)))
    if to_date:
        conditions.append(FieldCondition(key="created", range=Range(lte=to_date)))
    if labels and len(labels) > 0:
        conditions.append(FieldCondition(key="labels", match=MatchValue(value=labels[0])))
    if page_path:
        conditions.append(FieldCondition(key="page_path", match=MatchValue(value=page_path)))
    if search_headings:
        conditions.append(FieldCondition(key="headings", match=MatchValue(value=search_headings)))
        
    return conditions

def _format_search_results(results, with_vectors: bool = False, query_embedding=None) -> List[Dict]:
    """Форматирует результаты Qdrant в стандартный формат."""
    formatted = []
    for result in results:
        result_dict = {
            'id': str(result.id),
            'score': result.score,
            'payload': result.payload or {}
        }
        if with_vectors:
            if hasattr(result, 'vector') and result.vector:
                result_dict['embedding'] = result.vector
            else:
                result_dict['embedding'] = query_embedding
        formatted.append(result_dict)
    return formatted

def _apply_mmr_diversification(
    results: List[Dict],
    query_embedding: List[float],
    diversity_weight: float,
    limit: int
) -> List[Dict]:
    """Применяет MMR диверсификацию к результатам."""
    if not HAS_MMR or len(results) <= limit:
        return results[:limit]
    
    try:
        if all('embedding' in r for r in results):
            diversified = mmr_rerank(
                query_embedding=np.array(query_embedding, dtype=np.float32),
                results=results,
                diversity_weight=diversity_weight,
                top_k=limit
            )
            return diversified
        else:
            return results[:limit]
    except Exception as e:
        logger.warning(f"MMR failed: {e}")
        return results[:limit]

def search_in_qdrant(
    query_embedding: List[float],
    limit: int = 10,
    where_filter: Optional[Dict] = None,
    space: Optional[str] = None,
    author: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    status: Optional[str] = None,
    content_type: Optional[str] = None,
    labels: Optional[List[str]] = None,
    page_path: Optional[str] = None,
    search_headings: Optional[str] = None,
    use_mmr: bool = False,
    mmr_diversity_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Поиск в Qdrant с поддержкой фильтрации по метаданным (Sync).
    """
    client = init_qdrant_client()
    return _search_common(client, query_embedding, limit, where_filter, space, author, from_date, to_date, status, content_type, labels, page_path, search_headings, use_mmr, mmr_diversity_weight)

async def search_in_qdrant_async(
    query_embedding: List[float],
    limit: int = 10,
    where_filter: Optional[Dict] = None,
    space: Optional[str] = None,
    author: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    status: Optional[str] = None,
    content_type: Optional[str] = None,
    labels: Optional[List[str]] = None,
    page_path: Optional[str] = None,
    search_headings: Optional[str] = None,
    use_mmr: bool = False,
    mmr_diversity_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Поиск в Qdrant с поддержкой фильтрации по метаданным (Async).
    """
    client = init_async_qdrant_client()
    # Для async клиента методы такие же, но с await
    return await _search_common_async(client, query_embedding, limit, where_filter, space, author, from_date, to_date, status, content_type, labels, page_path, search_headings, use_mmr, mmr_diversity_weight)


def _search_common(client, query_embedding, limit, where_filter, space, author, from_date, to_date, status, content_type, labels, page_path, search_headings, use_mmr, mmr_diversity_weight):
    """Общая логика поиска (Sync)."""
    # 1. Строим фильтр
    conditions = []
    if where_filter:
        conditions.extend(_parse_where_filter(where_filter))
    
    conditions.extend(_build_metadata_conditions(
        space, author, from_date, to_date, status, 
        content_type, labels, page_path, search_headings
    ))

    qdrant_filter = Filter(must=conditions) if conditions else None

    try:
        # 2. Поиск
        with_vectors = use_mmr and HAS_MMR
        search_limit = limit * 3 if with_vectors else limit

        results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_embedding,
            limit=search_limit,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=with_vectors
        )

        # 3. Форматирование
        formatted_results = _format_search_results(results, with_vectors, query_embedding)

        # 4. MMR диверсификация
        if with_vectors:
            return _apply_mmr_diversification(
                formatted_results,
                query_embedding,
                mmr_diversity_weight,
                limit
            )

        return formatted_results[:limit]
    except Exception as e:
        logger.error(f"Ошибка поиска в Qdrant: {e}")
        return []

async def _search_common_async(client, query_embedding, limit, where_filter, space, author, from_date, to_date, status, content_type, labels, page_path, search_headings, use_mmr, mmr_diversity_weight):
    """Общая логика поиска (Async)."""
    # 1. Строим фильтр
    conditions = []
    if where_filter:
        conditions.extend(_parse_where_filter(where_filter))
    
    conditions.extend(_build_metadata_conditions(
        space, author, from_date, to_date, status, 
        content_type, labels, page_path, search_headings
    ))

    qdrant_filter = Filter(must=conditions) if conditions else None

    try:
        # 2. Поиск
        with_vectors = use_mmr and HAS_MMR
        search_limit = limit * 3 if with_vectors else limit

        results = await client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_embedding,
            limit=search_limit,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=with_vectors
        )

        # 3. Форматирование
        formatted_results = _format_search_results(results, with_vectors, query_embedding)

        # 4. MMR диверсификация
        # MMR выполняется на CPU (numpy), поэтому можно синхронно или в треде
        if with_vectors:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            return await loop.run_in_executor(None, _apply_mmr_diversification, formatted_results, query_embedding, mmr_diversity_weight, limit)

        return formatted_results[:limit]
    except Exception as e:
        logger.error(f"Ошибка асинхронного поиска в Qdrant: {e}")
        return []

def get_qdrant_count() -> int:
    """Получить количество документов в Qdrant."""
    client = init_qdrant_client()
    try:
        collection_info = client.get_collection(settings.qdrant_collection)
        return collection_info.points_count
    except Exception as e:
        logger.error(f"Ошибка получения количества документов: {e}")
        return 0

def get_all_points(limit: int = 10000, include_payload: bool = True) -> Dict[str, Any]:
    """Получить все точки из Qdrant."""
    client = init_qdrant_client()
    try:
        scroll_result = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=limit,
            with_payload=include_payload,
            with_vectors=False
        )
        points, _ = scroll_result

        ids = []
        documents = []
        metadatas = []

        for point in points:
            ids.append(str(point.id))
            if include_payload and point.payload:
                text = extract_text_from_payload(point.payload)
                documents.append(text)
                meta = {k: v for k, v in point.payload.items() if k not in ['text', '_node_content']}
                metadatas.append(meta)
            else:
                documents.append("")
                metadatas.append({})

        return {
            'ids': ids,
            'documents': documents,
            'metadatas': metadatas
        }
    except Exception as e:
        logger.error(f"Ошибка получения всех точек: {e}")
        return {'ids': [], 'documents': [], 'metadatas': []}

# Для совместимости с sync_confluence (оставляем старые функции)
def insert_chunks_batch_to_qdrant(client: QdrantClient, chunks_data: List[Dict[str, Any]], batch_size: int = 100) -> Tuple[int, int]:
    """Вставить chunks батчами в Qdrant (Sync)."""
    success_count = 0
    error_count = 0

    for i in range(0, len(chunks_data), batch_size):
        batch = chunks_data[i:i + batch_size]
        points = []

        for chunk in batch:
            try:
                payload = {**chunk['metadata'], "text": chunk['text']}
                point = PointStruct(
                    id=chunk['point_id'],
                    vector=chunk['embedding'],
                    payload=payload
                )
                points.append(point)
            except Exception as e:
                logger.warning(f"Error preparing point: {e}")
                error_count += 1
                continue

        if points:
            try:
                client.upsert(
                    collection_name=settings.qdrant_collection,
                    points=points
                )
                success_count += len(points)
            except Exception as e:
                logger.error(f"Error inserting batch: {e}")
                # Не увеличиваем error_count - точки не были обработаны

    return success_count, error_count

def delete_points_by_page_ids(page_ids: List[str], chunk_size: int = 500) -> int:
    """Удалить все точки для списка страниц."""
    if not page_ids:
        return 0

    client = init_qdrant_client()
    total_deleted = 0

    for i in range(0, len(page_ids), chunk_size):
        chunk = page_ids[i:i+chunk_size]
        try:
            scroll_result = client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(
                    should=[
                        FieldCondition(key="page_id", match=MatchValue(value=pid))
                        for pid in chunk
                    ]
                ),
                limit=10000
            )
            point_ids = [point.id for point in scroll_result[0]]
            if point_ids:
                client.delete(
                    collection_name=settings.qdrant_collection,
                    points_selector=point_ids
                )
                total_deleted += len(point_ids)
        except Exception as e:
            logger.error(f"Ошибка batch deletion: {e}")
            continue

    return total_deleted

def get_points_by_filter(
    where_filter: Dict,
    limit: int = 100,
    collection: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Получить точки из Qdrant по фильтру метаданных.
    
    Args:
        where_filter: Фильтр в формате {'$and': [{'page_id': 'xxx'}, {'chunk': {'$gte': 1}}]}
        limit: Максимальное количество результатов
        collection: Имя коллекции (по умолчанию из settings)
    
    Returns:
        Список документов с text и metadata
    """
    client = init_qdrant_client()
    collection_name = collection or settings.qdrant_collection
    
    # Парсим фильтр
    must_conditions = []
    if '$and' in where_filter:
        for condition in where_filter['$and']:
            for key, value in condition.items():
                if isinstance(value, dict):
                    # Range фильтр ($gte, $lte, $gt, $lt)
                    range_params = {}
                    if '$gte' in value:
                        range_params['gte'] = value['$gte']
                    if '$lte' in value:
                        range_params['lte'] = value['$lte']
                    if '$gt' in value:
                        range_params['gt'] = value['$gt']
                    if '$lt' in value:
                        range_params['lt'] = value['$lt']
                    must_conditions.append(
                        FieldCondition(key=key, range=Range(**range_params))
                    )
                else:
                    # Exact match
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
    else:
        # Простой фильтр без $and
        for key, value in where_filter.items():
            if isinstance(value, dict) and any(k in value for k in ['$gte', '$lte', '$gt', '$lt']):
                range_params = {}
                if '$gte' in value:
                    range_params['gte'] = value['$gte']
                if '$lte' in value:
                    range_params['lte'] = value['$lte']
                if '$gt' in value:
                    range_params['gt'] = value['$gt']
                if '$lt' in value:
                    range_params['lt'] = value['$lt']
                must_conditions.append(
                    FieldCondition(key=key, range=Range(**range_params))
                )
            else:
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
    
    try:
        scroll_result = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        points, _ = scroll_result
        
        return [
            {
                'text': extract_text_from_payload(point.payload),
                'metadata': {k: v for k, v in point.payload.items() if k not in ['text', '_node_content']}
            }
            for point in points
        ]
    except Exception as e:
        logger.error(f"Ошибка get_points_by_filter: {e}")
        return []
