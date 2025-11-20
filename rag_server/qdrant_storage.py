"""
Qdrant vector store для хранения embeddings.
Замена ChromaDB для лучшей масштабируемости.
"""
import os
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range, PayloadSchemaType

# Инициализация logger (должен быть до использования)
logger = logging.getLogger(__name__)

def extract_text_from_payload(payload: Dict[str, Any]) -> str:
    """
    Извлечь текст из payload Qdrant.

    КРИТИЧНО: LlamaIndex QdrantVectorStore сохраняет текст в _node_content (JSON),
    а не в поле 'text'. Эта функция проверяет оба варианта.

    Args:
        payload: Payload из Qdrant точки

    Returns:
        Текст документа или пустая строка
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
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "confluence")

qdrant_client = None

def init_qdrant_client() -> QdrantClient:
    """Инициализировать Qdrant клиент."""
    global qdrant_client
    if qdrant_client is None:
        try:
            qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
            logger.info(f"✅ Qdrant client initialized: {QDRANT_HOST}:{QDRANT_PORT}")
        except Exception as e:
            logger.error(f"Ошибка инициализации Qdrant client: {e}")
            raise
    return qdrant_client

def init_qdrant_collection(embedding_dim: int) -> bool:
    """
    Инициализировать коллекцию Qdrant с индексами для метаданных.

    ИСПРАВЛЕНО: Добавлено создание payload индексов для быстрой фильтрации по метаданным.
    """
    client = init_qdrant_client()

    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        collection_created = False
        if QDRANT_COLLECTION not in collection_names:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Created Qdrant collection: {QDRANT_COLLECTION} (dim={embedding_dim})")
            collection_created = True
        else:
            # Проверяем размерность существующей коллекции
            collection_info = client.get_collection(QDRANT_COLLECTION)
            existing_dim = collection_info.config.params.vectors.size
            if existing_dim != embedding_dim:
                logger.error(
                    f"Несовпадение размерности: Qdrant={existing_dim}D, Model={embedding_dim}D. "
                    f"Удалите коллекцию {QDRANT_COLLECTION} и перезапустите."
                )
                return False
            logger.info(f"✅ Qdrant collection exists: {QDRANT_COLLECTION} (dim={embedding_dim})")

        # Создаем payload индексы для быстрой фильтрации (если коллекция новая или индексов нет)
        # Пытаемся создать индексы даже для существующей коллекции (если их еще нет)
        try:
            # Индексы для строковых полей (KEYWORD)
            # ✅ ДОБАВЛЕНО: author, page_id для быстрой фильтрации
            keyword_fields = ['space', 'status', 'type', 'content_type', 'created_by', 'modified_by', 'page_path', 'author', 'page_id']
            for field in keyword_fields:
                try:
                    client.create_payload_index(
                        collection_name=QDRANT_COLLECTION,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    logger.debug(f"✅ Created keyword index for {field}")
                except Exception as e:
                    # Индекс может уже существовать - это нормально
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        logger.debug(f"Index for {field} already exists")
                    else:
                        logger.warning(f"Could not create index for {field}: {e}")

            # Индекс для labels (TEXT для поиска по подстроке)
            try:
                client.create_payload_index(
                    collection_name=QDRANT_COLLECTION,
                    field_name="labels",
                    field_schema=PayloadSchemaType.TEXT
                )
                logger.debug("✅ Created text index for labels")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    logger.debug("Index for labels already exists")
                else:
                    logger.warning(f"Could not create index for labels: {e}")

            # Индекс для headings (TEXT для поиска по заголовкам)
            try:
                client.create_payload_index(
                    collection_name=QDRANT_COLLECTION,
                    field_name="headings",
                    field_schema=PayloadSchemaType.TEXT
                )
                logger.debug("✅ Created text index for headings")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    logger.debug("Index for headings already exists")
                else:
                    logger.warning(f"Could not create index for headings: {e}")

            # ✅ ДОБАВЛЕНО: Индекс для title (TEXT для full-text search)
            try:
                client.create_payload_index(
                    collection_name=QDRANT_COLLECTION,
                    field_name="title",
                    field_schema=PayloadSchemaType.TEXT
                )
                logger.debug("✅ Created text index for title")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    logger.debug("Index for title already exists")
                else:
                    logger.warning(f"Could not create index for title: {e}")

            # Индексы для числовых полей (INTEGER)
            # ✅ ДОБАВЛЕНО: created, modified для range queries
            integer_fields = ['hierarchy_depth', 'version', 'children_count', 'heading_count', 'created', 'modified']
            for field in integer_fields:
                try:
                    client.create_payload_index(
                        collection_name=QDRANT_COLLECTION,
                        field_name=field,
                        field_schema=PayloadSchemaType.INTEGER
                    )
                    logger.debug(f"✅ Created integer index for {field}")
                except Exception as e:
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        logger.debug(f"Index for {field} already exists")
                    else:
                        logger.warning(f"Could not create index for {field}: {e}")

            if collection_created:
                logger.info("✅ Created payload indexes for metadata filtering")
            else:
                logger.debug("✅ Verified payload indexes for metadata filtering")
        except Exception as e:
            logger.warning(f"Could not create some indexes (may already exist): {e}")

        return True
    except Exception as e:
        logger.error(f"Ошибка инициализации Qdrant collection: {e}")
        return False

def insert_chunk_to_qdrant(
    client: QdrantClient,
    chunk_text: str,
    metadata: dict,
    embedding: List[float],
    point_id: str
) -> bool:
    """
    Вставить один chunk в Qdrant напрямую (без llama-index).

    Args:
        client: QdrantClient
        chunk_text: Текст chunk
        metadata: Метаданные chunk
        embedding: Векторное представление текста
        point_id: Уникальный ID точки (например: f"{page_id}_{chunk_idx}")

    Returns:
        True если успешно, False если ошибка
    """
    try:
        payload = {**metadata, "text": chunk_text}
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[point]
        )
        return True
    except Exception as e:
        logger.error(f"Failed to insert chunk {point_id}: {e}")
        return False

def insert_chunks_batch_to_qdrant(
    client: QdrantClient,
    chunks_data: List[Dict[str, Any]],
    batch_size: int = 100
) -> Tuple[int, int]:
    """
    Вставить chunks батчами в Qdrant.

    Args:
        client: QdrantClient
        chunks_data: Список словарей с ключами: text, metadata, embedding, point_id
        batch_size: Размер батча

    Returns:
        Tuple[success_count, error_count]
    """
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
                logger.warning(f"Error preparing point {chunk.get('point_id', 'unknown')}: {e}")
                error_count += 1
                continue

        if points:
            try:
                client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points
                )
                success_count += len(points)
            except Exception as e:
                logger.error(f"Error inserting batch {i//batch_size + 1}: {e}")
                # Не увеличиваем error_count - точки не были обработаны
                # Они останутся в chunks_data и могут быть обработаны позже при retry

    return success_count, error_count

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
    
    logger.debug(f"🔀 Applying MMR diversification (weight={diversity_weight}, {len(results)} → {limit} results)")
    
    try:
        if all('embedding' in r for r in results):
            diversified = mmr_rerank(
                query_embedding=np.array(query_embedding, dtype=np.float32),
                results=results,
                diversity_weight=diversity_weight,
                top_k=limit
            )
            logger.debug(f"✅ MMR completed: {len(diversified)} results")
            return diversified
        else:
            logger.warning("⚠️ Some results missing embeddings, skipping MMR")
            return results[:limit]
    except Exception as e:
        logger.warning(f"MMR failed: {e}")
        return results[:limit]

def search_in_qdrant(
    query_embedding: List[float],
    limit: int = 10,
    where_filter: Optional[Dict] = None,
    # НОВЫЕ ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ для metadata filtering:
    space: Optional[str] = None,
    author: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    status: Optional[str] = None,
    content_type: Optional[str] = None,
    labels: Optional[List[str]] = None,
    # === НОВОЕ: ФИЛЬТР ПО ПУТИ ===
    page_path: Optional[str] = None,  # "RAUII/Development/API"
    # === НОВОЕ: ПОИСК В ЗАГОЛОВКАХ ===
    search_headings: Optional[str] = None,  # Поиск query в заголовках
    # === НОВОЕ: MMR DIVERSIFICATION ===
    use_mmr: bool = False,  # Использовать MMR (default: false для обратной совместимости)
    mmr_diversity_weight: float = 0.3  # Вес diversity (30%)
) -> List[Dict[str, Any]]:
    """
    Поиск в Qdrant с поддержкой фильтрации по метаданным.

    ИСПРАВЛЕНО: Добавлены опциональные параметры для удобной фильтрации.
    Если указаны параметры фильтрации, они автоматически преобразуются в where_filter.

    Args:
        query_embedding: Vector embedding запроса
        limit: Максимальное количество результатов
        where_filter: Прямой фильтр Qdrant (если указан, остальные параметры игнорируются)
        space: Фильтр по пространству (например, "RAUII")
        author: Фильтр по автору (created_by)
        from_date: Фильтр по дате создания >= (ISO format: "2025-01-01T00:00:00Z")
        to_date: Фильтр по дате создания <= (ISO format)
        status: Фильтр по статусу ("current", "archived", "draft")
        content_type: Фильтр по типу ("page", "blogpost", "attachment")
        labels: Фильтр по меткам (список строк, любая должна совпадать)
        page_path: Фильтр по пути (например, "RAUII/Development/API")
        search_headings: Поиск query в заголовках (текст для поиска)
        use_mmr: Использовать MMR для диверсификации результатов
        mmr_diversity_weight: Вес diversity для MMR (0-1), default 0.3

    Returns:
        Список результатов поиска
    """
    client = init_qdrant_client()

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
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=search_limit,
            query_filter=qdrant_filter,
            with_payload=True,  # КРИТИЧНО: Получаем payload с metadata!
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

def get_qdrant_count() -> int:
    """Получить количество документов в Qdrant."""
    client = init_qdrant_client()
    try:
        collection_info = client.get_collection(QDRANT_COLLECTION)
        return collection_info.points_count
    except Exception as e:
        logger.error(f"Ошибка получения количества документов: {e}")
        return 0

def delete_points_by_page_id(page_id: str) -> bool:
    """Удалить все точки (chunks) для страницы по page_id."""
    client = init_qdrant_client()
    try:
        # Ищем все точки с данным page_id в payload
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="page_id",
                        match=MatchValue(value=page_id)
                    )
                ]
            ),
            limit=10000  # Максимум для удаления
        )

        point_ids = [point.id for point in scroll_result[0]]
        if point_ids:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=point_ids
            )
            logger.debug(f"Удалено {len(point_ids)} точек для страницы {page_id}")

        return True
    except Exception as e:
        logger.error(f"Ошибка удаления точек для страницы {page_id}: {e}")
        return False

def delete_points_by_page_ids(page_ids: List[str], chunk_size: int = 500) -> int:
    """
    Удалить все точки (chunks) для списка страниц (batch operation с chunking).

    Оптимизация: один scroll запрос для chunk страниц вместо N запросов.
    Для больших списков (>chunk_size) разбивает на chunks для избежания timeout.

    Args:
        page_ids: Список page_id для удаления
        chunk_size: Размер chunk для batch операции (по умолчанию 500)

    Returns:
        Количество удаленных точек
    """
    if not page_ids:
        return 0

    client = init_qdrant_client()
    total_deleted = 0

    # Chunking для больших batch operations
    for i in range(0, len(page_ids), chunk_size):
        chunk = page_ids[i:i+chunk_size]

        try:
            # Один scroll для chunk page_ids (OR условие)
            scroll_result = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(
                    should=[  # OR условие для множественных page_ids
                        FieldCondition(
                            key="page_id",
                            match=MatchValue(value=pid)
                        )
                        for pid in chunk
                    ]
                ),
                limit=10000  # Максимум для удаления
            )

            point_ids = [point.id for point in scroll_result[0]]

            if point_ids:
                # Batch deletion
                client.delete(
                    collection_name=QDRANT_COLLECTION,
                    points_selector=point_ids
                )
                total_deleted += len(point_ids)
                logger.debug(
                    f"Batch deletion chunk {i//chunk_size + 1}: "
                    f"удалено {len(point_ids)} точек для {len(chunk)} страниц"
                )
        except Exception as e:
            logger.error(
                f"Ошибка batch deletion для chunk {i//chunk_size + 1} "
                f"({len(chunk)} страниц): {e}"
            )
            continue

    if total_deleted > 0:
        logger.info(
            f"✅ Batch deletion завершено: удалено {total_deleted} точек "
            f"для {len(page_ids)} страниц ({len(page_ids)//chunk_size + 1} chunks)"
        )

    return total_deleted

def clear_qdrant_collection() -> int:
    """
    Полностью очистить коллекцию Qdrant (удалить все точки).

    Returns:
        Количество удаленных точек
    """
    client = init_qdrant_client()
    try:
        # Получаем все точки батчами
        total_deleted = 0
        offset = None

        while True:
            scroll_result = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=10000,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )

            points, next_offset = scroll_result

            if not points:
                break

            point_ids = [point.id for point in points]
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=point_ids
            )
            total_deleted += len(point_ids)
            logger.info(f"Удалено {len(point_ids)} точек (всего: {total_deleted})")

            if next_offset is None:
                break
            offset = next_offset

        logger.info(f"✅ Qdrant коллекция очищена: удалено {total_deleted} точек")
        return total_deleted
    except Exception as e:
        logger.error(f"Ошибка очистки Qdrant коллекции: {e}")
        return 0

def get_all_points(limit: int = 10000, include_payload: bool = True) -> Dict[str, Any]:
    """
    Получить все точки из Qdrant (аналог collection.get() для ChromaDB).

    Args:
        limit: Максимальное количество точек
        include_payload: Включать ли payload (текст и метаданные)

    Returns:
        Словарь в формате {'ids': [...], 'documents': [...], 'metadatas': [...]}
    """
    client = init_qdrant_client()
    try:
        # Используем scroll для получения всех точек
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION,
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
                # Извлекаем текст
                text = extract_text_from_payload(point.payload)
                documents.append(text)

                # Метаданные (все кроме текста)
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
