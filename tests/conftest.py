"""Shared pytest fixtures"""
import pytest
import sys
import os
from unittest.mock import Mock, AsyncMock

# Add rag_server to sys.path to allow imports like 'from embeddings import ...'
# which are used inside the source files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag_server')))
# Also add the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_server.config import Settings

@pytest.fixture
def test_settings():
    """Тестовые настройки"""
    return Settings(
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection="test_collection",
        enable_hybrid_search=True,
        enable_context_expansion=True,
        enable_metrics=False,
        enable_tracing=False,
    )

@pytest.fixture
def mock_embedding_384():
    """Mock embedding 384D"""
    return [0.1] * 384

@pytest.fixture
def mock_embedding_768():
    """Mock embedding 768D"""
    return [0.1] * 768

@pytest.fixture
def sample_documents():
    """Тестовые документы"""
    return [
        {
            'id': 'doc-1',
            'text': 'This is document 1 about Python programming.',
            'metadata': {'space': 'DOCS', 'page_id': 'page-1'}
        },
        {
            'id': 'doc-2',
            'text': 'This is document 2 about machine learning.',
            'metadata': {'space': 'DOCS', 'page_id': 'page-2'}
        },
        {
            'id': 'doc-3',
            'text': 'This is document 3 about data science.',
            'metadata': {'space': 'TECH', 'page_id': 'page-3'}
        },
    ]

@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient"""
    client = Mock()
    client.search = Mock(return_value=[])
    client.scroll = Mock(return_value=([], None))
    return client

from qdrant_client import AsyncQdrantClient

@pytest.fixture
def mock_async_qdrant_client():
    """Mock AsyncQdrantClient"""
    client = AsyncMock(spec=AsyncQdrantClient)
    client.search = AsyncMock(return_value=[])
    client.scroll = AsyncMock(return_value=([], None))
    return client

@pytest.fixture
def mock_embeddings_model():
    """Mock embeddings model"""
    model = AsyncMock()
    model.get_query_embedding_async = AsyncMock(return_value=[0.1] * 384)
    model.get_text_embeddings_async = AsyncMock(return_value=[[0.1] * 384])
    return model

from rag_server.config import settings as global_settings

@pytest.fixture(autouse=True)
def setup_test_settings():
    """Override global settings for tests"""
    # Save original values
    original_qdrant_host = global_settings.qdrant_host
    original_expansion = global_settings.enable_context_expansion
    original_hybrid = global_settings.enable_hybrid_search
    
    # Set test values
    global_settings.qdrant_host = "localhost"
    global_settings.qdrant_port = 6333
    global_settings.qdrant_collection = "test_collection"
    global_settings.enable_hybrid_search = True
    global_settings.enable_context_expansion = True
    global_settings.enable_metrics = False
    global_settings.enable_tracing = False
    
    yield global_settings
    
    # Restore (optional, but good practice)
    global_settings.qdrant_host = original_qdrant_host
    global_settings.enable_context_expansion = original_expansion
    global_settings.enable_hybrid_search = original_hybrid
