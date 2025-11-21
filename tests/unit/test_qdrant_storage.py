"""Unit tests для Qdrant storage"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from rag_server.qdrant_storage import (
    init_qdrant_client,
    init_async_qdrant_client,
    search_in_qdrant,
    search_in_qdrant_async,
)

def test_init_qdrant_client():
    """Тест инициализации sync клиента"""
    client = init_qdrant_client()
    assert client is not None

def test_init_async_qdrant_client():
    """Тест инициализации async клиента"""
    client = init_async_qdrant_client()
    assert client is not None

@pytest.mark.asyncio
@patch('rag_server.qdrant_storage.AsyncQdrantClient')
async def test_search_in_qdrant_async(mock_client_class):
    """Тест async поиска в Qdrant"""
    # Mock async client
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value=[
        Mock(id='1', score=0.9, payload={'text': 'doc1'}),
        Mock(id='2', score=0.8, payload={'text': 'doc2'}),
    ])
    
    mock_client_class.return_value = mock_client
    
    # Mock init
    with patch('rag_server.qdrant_storage.async_qdrant_client', mock_client):
        results = await search_in_qdrant_async(
            query_embedding=[0.1] * 384,
            limit=10
        )
    
    assert len(results) == 2
    assert results[0]['id'] == '1'
    assert results[0]['score'] == 0.9

@pytest.mark.asyncio
async def test_search_with_filters():
    """Тест поиска с фильтрами"""
    pytest.skip("Requires Qdrant test instance")

@pytest.mark.asyncio
async def test_search_with_mmr():
    """Тест поиска с MMR диверсификацией"""
    pytest.skip("Requires Qdrant test instance")

