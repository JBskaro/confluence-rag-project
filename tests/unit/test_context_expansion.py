"""Unit tests для Context Expansion"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from rag_server.context_expansion import (
    expand_context_bidirectional_async,
    expand_context_with_related_async,
    expand_context_full_async,
    _validate_result_and_collection,
    _default_result,
)

@pytest.fixture
def sample_result():
    """Пример результата поиска"""
    return {
        'id': 'doc-1',
        'text': 'Original text',
        'metadata': {
            'page_id': 'page-123',
            'chunk': 5,
            'space': 'DOCS',
        },
        'score': 0.9
    }

def test_validate_result_and_collection_valid(sample_result):
    """Тест валидации валидного результата"""
    collection = Mock()
    
    validation = _validate_result_and_collection(sample_result, collection)
    
    assert validation is not None
    page_id, text, current_id = validation
    assert page_id == 'page-123'
    assert text == 'Original text'
    assert current_id == 'doc-1'

def test_validate_result_and_collection_invalid():
    """Тест валидации невалидного результата"""
    result = {'text': 'No metadata'}
    collection = Mock()
    
    validation = _validate_result_and_collection(result, collection)
    
    assert validation is None

def test_validate_result_no_collection(sample_result):
    """Тест валидации без collection"""
    validation = _validate_result_and_collection(sample_result, None)
    assert validation is None

def test_default_result(sample_result):
    """Тест дефолтного результата"""
    result = _default_result(sample_result, mode='none')
    
    assert result['expanded_text'] == 'Original text'
    assert result['context_chunks'] == 1
    assert result['expansion_mode'] == 'none'

@pytest.mark.asyncio
async def test_expand_context_bidirectional_async(sample_result, mock_async_qdrant_client):
    """Тест bidirectional расширения"""
    # Mock scroll для получения соседних чанков
    # Note: scroll returns (points, next_page_offset)
    mock_async_qdrant_client.scroll = AsyncMock(return_value=(
        [
            Mock(payload={'chunk': 4, 'text': 'Chunk 4'}),
            Mock(payload={'chunk': 5, 'text': 'Original text'}),
            Mock(payload={'chunk': 6, 'text': 'Chunk 6'}),
        ],
        None
    ))
    
    result = await expand_context_bidirectional_async(
        sample_result,
        mock_async_qdrant_client,
        context_size=1
    )
    
    assert 'expanded_text' in result
    assert result['context_chunks'] == 3
    assert result['expansion_mode'] == 'bidirectional'
    assert 'Chunk 4' in result['expanded_text']
    assert 'Chunk 6' in result['expanded_text']

@pytest.mark.asyncio
async def test_expand_context_with_related_async(sample_result, mock_async_qdrant_client):
    """Тест related расширения"""
    # Mock scroll для получения всех чанков страницы
    mock_async_qdrant_client.scroll = AsyncMock(return_value=(
        [
            Mock(id='doc-1', payload={'text': 'Original text'}),
            Mock(id='doc-2', payload={'text': 'Related text 1'}),
            Mock(id='doc-3', payload={'text': 'Related text 2'}),
        ],
        None
    ))
    
    # Mock embeddings model
    mock_embeddings = AsyncMock()
    mock_embeddings.get_query_embedding_async = AsyncMock(
        return_value=[0.1] * 384
    )
    mock_embeddings.get_text_embeddings_async = AsyncMock(
        return_value=[[0.2] * 384, [0.15] * 384]
    )
    
    result = await expand_context_with_related_async(
        sample_result,
        mock_async_qdrant_client,
        embeddings_model=mock_embeddings,
        top_k=2
    )
    
    assert 'expanded_text' in result
    assert result['expansion_mode'] == 'related'
    assert 'Related chunks' in result['expanded_text']

@pytest.mark.asyncio
async def test_expand_context_full_async_disabled():
    """Тест когда expansion отключен"""
    from rag_server.config import settings
    original = settings.enable_context_expansion
    settings.enable_context_expansion = False
    
    try:
        result = {'text': 'Original', 'metadata': {}}
        collection = Mock()
        
        expanded = await expand_context_full_async(result, collection)
        
        assert expanded['expanded_text'] == 'Original'
        # Should be 'none' if disabled, or 'disabled' depending on implementation
        # If fails with 'none', implementation uses 'none'
        assert expanded['expansion_mode'] == 'none'
    finally:
        settings.enable_context_expansion = original

@pytest.mark.asyncio
async def test_expand_context_full_async_bidirectional(sample_result, mock_async_qdrant_client):
    """Тест full expansion в режиме bidirectional"""
    from rag_server.config import settings
    original_enable = settings.enable_context_expansion
    original_mode = settings.context_expansion_mode
    
    settings.enable_context_expansion = True
    settings.context_expansion_mode = 'bidirectional'
    
    try:
        mock_async_qdrant_client.scroll = AsyncMock(return_value=(
            [Mock(payload={'chunk': 5, 'text': 'Text'})],
            None
        ))
        
        result = await expand_context_full_async(
            sample_result,
            mock_async_qdrant_client,
            expansion_mode='bidirectional',
            context_size=1
        )
        
        assert result['expansion_mode'] == 'bidirectional'
    finally:
        settings.enable_context_expansion = original_enable
        settings.context_expansion_mode = original_mode

@pytest.mark.asyncio
async def test_expand_context_error_handling(sample_result):
    """Тест обработки ошибок"""
    # Collection который бросает исключение
    mock_client = AsyncMock()
    mock_client.scroll = AsyncMock(side_effect=Exception("Connection error"))
    
    result = await expand_context_bidirectional_async(
        sample_result,
        mock_client
    )
    
    # Должен вернуть дефолтный результат или ошибку
    assert 'expanded_text' in result or 'error' in result

