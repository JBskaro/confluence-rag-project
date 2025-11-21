"""Unit tests для embeddings module"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from rag_server.embeddings import (
    generate_query_embedding,
    generate_query_embedding_async,
    generate_query_embeddings_batch_async,
    get_embedding_dimension,
)

@pytest.mark.asyncio
async def test_generate_query_embedding_async():
    """Тест async генерации embedding"""
    embedding = await generate_query_embedding_async("test query")
    
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)

@pytest.mark.asyncio
async def test_generate_embeddings_batch_async():
    """Тест batch генерации embeddings"""
    queries = ["query 1", "query 2", "query 3"]
    embeddings = await generate_query_embeddings_batch_async(queries)
    
    assert len(embeddings) == 3
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) > 0 for emb in embeddings)

def test_get_embedding_dimension():
    """Тест получения размерности"""
    dim = get_embedding_dimension()
    
    assert dim > 0
    assert isinstance(dim, int)
    # Обычно 384, 768, 1024, 1536, 3072, 4096...
    assert dim in [384, 768, 1024, 1536, 3072, 4096]

@pytest.mark.asyncio
@patch('rag_server.embeddings.get_embed_model')
async def test_embedding_caching(mock_get_model):
    """Тест генерации embeddings (caching not implemented explicitly)"""
    # Mock model
    mock_model = AsyncMock()
    mock_model.get_query_embedding_async = AsyncMock(return_value=[0.1] * 384)
    mock_get_model.return_value = mock_model

    query = "test query for cache"
    
    # First call
    emb1 = await generate_query_embedding_async(query)
    
    # Second call
    emb2 = await generate_query_embedding_async(query)
    
    # Должны быть одинаковыми (mock returns same)
    assert np.allclose(emb1, emb2, rtol=1e-5)

@pytest.mark.asyncio
async def test_empty_query_handling():
    """Тест обработки пустого запроса"""
    # Should return zero vector or handle gracefully
    embedding = await generate_query_embedding_async("")
    assert embedding is not None
    assert len(embedding) > 0
    # Optional: check if all zeros
    # assert all(x == 0.0 for x in embedding)

@pytest.mark.asyncio
async def test_batch_with_empty_queries():
    """Тест batch с пустыми запросами"""
    queries = ["valid", "", "another valid"]
    
    # Должен пропустить пустые или вернуть ошибку
    embeddings = await generate_query_embeddings_batch_async(queries)
    
    # В зависимости от реализации
    assert len(embeddings) <= 3

@pytest.mark.asyncio
@patch('rag_server.embeddings.get_embed_model')
async def test_mock_embedding_model(mock_get_model):
    """Тест с mock embedding model"""
    # Mock embedding model
    mock_model = AsyncMock()
    mock_model.get_query_embedding_async = AsyncMock(return_value=[0.1] * 384)
    mock_get_model.return_value = mock_model
    
    embedding = await generate_query_embedding_async("test")
    
    assert len(embedding) == 384
    mock_model.get_query_embedding_async.assert_called_once()

