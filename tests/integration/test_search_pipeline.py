"""Integration tests для Search Pipeline"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from rag_server.search_pipeline import SearchPipeline, SearchParams

@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client"""
    client = Mock()
    client.search = Mock(return_value=[
        Mock(id='1', score=0.9, payload={'text': 'Doc 1'}),
        Mock(id='2', score=0.8, payload={'text': 'Doc 2'}),
    ])
    return client

@pytest.fixture
def mock_reranker():
    """Mock reranker"""
    reranker = Mock()
    reranker.predict = Mock(return_value=[0.95, 0.85])
    return reranker

@pytest.mark.asyncio
async def test_search_pipeline_basic(mock_qdrant, mock_reranker):
    """Тест базового pipeline"""
    pipeline = SearchPipeline(
        mock_qdrant,
        collection_name="test",
        reranker=mock_reranker
    )
    # Manually inject async client mock
    pipeline.async_qdrant_client = AsyncMock()
    pipeline.async_qdrant_client.search = AsyncMock(return_value=[
        Mock(id='1', score=0.9, payload={'text': 'Doc 1'}),
        Mock(id='2', score=0.8, payload={'text': 'Doc 2'}),
    ])
    
    params = SearchParams(
        query="test query",
        limit=5,
        use_reranking=True
    )
    
    with patch('rag_server.search_pipeline.generate_query_embeddings_batch_async') as mock_emb:
        mock_emb.return_value = [[0.1] * 384]
        
        results = await pipeline.execute_async(params)
    
    assert len(results) > 0
    assert all('text' in r for r in results)

@pytest.mark.asyncio
async def test_search_pipeline_with_expansion(mock_qdrant, mock_reranker):
    """Тест pipeline с query expansion"""
    pipeline = SearchPipeline(mock_qdrant, "test", mock_reranker)
    pipeline.async_qdrant_client = AsyncMock()
    # Returns empty results to simulate expansion logic processing
    pipeline.async_qdrant_client.search = AsyncMock(return_value=[])
    
    params = SearchParams(
        query="main query",
        expanded_queries=["variant 1", "variant 2"],
        limit=10
    )
    
    with patch('rag_server.search_pipeline.generate_query_embeddings_batch_async') as mock_emb:
        mock_emb.return_value = [[0.1] * 384] * 3  # main + 2 variants
        
        results = await pipeline.execute_async(params)
    
    assert len(results) <= 10

@pytest.mark.asyncio
async def test_search_pipeline_deduplication(mock_qdrant):
    """Тест дедупликации результатов"""
    # Mock возвращает дубликаты
    mock_qdrant.search = Mock(return_value=[
        Mock(id='1', score=0.9, payload={'text': 'Doc 1'}),
        Mock(id='1', score=0.85, payload={'text': 'Doc 1'}),  # дубль
        Mock(id='2', score=0.8, payload={'text': 'Doc 2'}),
    ])
    
    pipeline = SearchPipeline(mock_qdrant, "test")
    pipeline.async_qdrant_client = AsyncMock()
    pipeline.async_qdrant_client.search = AsyncMock(return_value=[
        Mock(id='1', score=0.9, payload={'text': 'Doc 1'}),
        Mock(id='1', score=0.85, payload={'text': 'Doc 1'}),  # дубль
        Mock(id='2', score=0.8, payload={'text': 'Doc 2'}),
    ])
    
    params = SearchParams(query="test", limit=10)
    
    with patch('rag_server.search_pipeline.generate_query_embeddings_batch_async') as mock_emb:
        mock_emb.return_value = [[0.1] * 384]
        results = await pipeline.execute_async(params)
    
    # Должно быть 2 уникальных документа
    ids = [r['id'] for r in results]
    assert len(ids) == len(set(ids))  # Все уникальные

@pytest.mark.asyncio
async def test_search_pipeline_error_handling(mock_qdrant):
    """Тест обработки ошибок"""
    mock_qdrant.search = Mock(side_effect=Exception("Connection error"))
    
    pipeline = SearchPipeline(mock_qdrant, "test")
    params = SearchParams(query="test", limit=5)
    
    with pytest.raises(Exception):
        with patch('rag_server.search_pipeline.generate_query_embeddings_batch_async') as mock_emb:
            mock_emb.side_effect = Exception("Embedding error")
            await pipeline.execute_async(params)

