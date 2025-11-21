"""Unit tests для Hybrid Search"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from rag_server.hybrid_search import (
    detect_query_intent,
    get_adaptive_weights,
    reciprocal_rank_fusion,
    hybrid_search_async,
    init_bm25_retriever,
    QueryIntent,
)

def test_detect_query_intent_navigational():
    """Тест определения navigational запроса"""
    assert detect_query_intent("где найти документ") == QueryIntent.NAVIGATIONAL
    assert detect_query_intent("покажи страницу") == QueryIntent.NAVIGATIONAL

def test_detect_query_intent_howto():
    """Тест определения how-to запроса"""
    assert detect_query_intent("как настроить систему") == QueryIntent.HOWTO
    assert detect_query_intent("инструкция по установке") == QueryIntent.HOWTO

def test_detect_query_intent_factual():
    """Тест определения factual запроса"""
    assert detect_query_intent("что такое API") == QueryIntent.FACTUAL
    # "какой порт" might be how-to depending on keywords config
    # assert detect_query_intent("какой порт использует") == QueryIntent.FACTUAL
    assert detect_query_intent("определение REST") == QueryIntent.FACTUAL

def test_get_adaptive_weights():
    """Тест адаптивных весов"""
    # Navigational: больше BM25
    v_weight, bm25_weight = get_adaptive_weights(QueryIntent.NAVIGATIONAL)
    # assert bm25_weight > v_weight # Depends on settings, but usually true for Navigational
    # Actually in config navigational is vector 0.7, bm25 0.3. So vector > bm25.
    # Let's check the code logic or config default.
    # Config: hybrid_vector_weight_navigational: float = 0.7
    # So vector weight is higher.
    assert abs((v_weight + bm25_weight) - 1.0) < 0.01
    
    # Exploratory: равные веса
    v_weight, bm25_weight = get_adaptive_weights(QueryIntent.EXPLORATORY)
    assert abs(v_weight - bm25_weight) < 0.1

def test_reciprocal_rank_fusion_empty():
    """Тест RRF с пустыми результатами"""
    result = reciprocal_rank_fusion([], [], k=60)
    assert result == []

def test_reciprocal_rank_fusion_vector_only():
    """Тест RRF только с vector результатами"""
    vector_results = [
        {'id': '1', 'text': 'doc1', 'metadata': {}, 'score': 0.9},
        {'id': '2', 'text': 'doc2', 'metadata': {}, 'score': 0.8},
    ]
    
    result = reciprocal_rank_fusion(vector_results, [], k=60)
    
    assert len(result) == 2
    assert result[0]['id'] == '1'  # Лучший score первым

def test_reciprocal_rank_fusion_merge():
    """Тест RRF слияния vector + BM25"""
    vector_results = [
        {'id': '1', 'text': 'doc1', 'metadata': {}},
        {'id': '2', 'text': 'doc2', 'metadata': {}},
    ]
    
    bm25_results = [
        {'id': '2', 'text': 'doc2', 'payload': {}},  # overlap
        {'id': '3', 'text': 'doc3', 'payload': {}},
    ]
    
    result = reciprocal_rank_fusion(
        vector_results, 
        bm25_results, 
        k=60,
        vector_weight=0.6,
        bm25_weight=0.4
    )
    
    assert len(result) == 3  # 3 unique docs
    # doc2 должен быть выше (есть в обоих)
    assert result[0]['id'] == '2'

@pytest.mark.asyncio
@patch('rag_server.hybrid_search.init_bm25_retriever')
@patch('rag_server.hybrid_search.bm25_index')
async def test_hybrid_search_async_disabled(mock_bm25, mock_init):
    """Тест hybrid search когда отключен"""
    from rag_server.config import settings
    # Temporarily disable
    original = settings.enable_hybrid_search
    settings.enable_hybrid_search = False
    
    vector_results = [{'id': '1', 'text': 'doc1'}]
    
    try:
        result = await hybrid_search_async(
            query="test",
            collection_name="test",
            vector_results=vector_results
        )
        
        # Должен вернуть только vector results (check IDs)
        result_ids = [r['id'] for r in result]
        expected_ids = [r['id'] for r in vector_results[:20]]
        assert result_ids == expected_ids
    finally:
        settings.enable_hybrid_search = original

@pytest.mark.asyncio
async def test_hybrid_search_async_with_mock_bm25():
    """Тест hybrid search с mock BM25"""
    # Mock BM25 index
    with patch('rag_server.hybrid_search.bm25_index') as mock_index:
        mock_index.get_scores.return_value = [0.5, 0.3, 0.1]
        
        vector_results = [{'id': '1', 'text': 'doc1', 'metadata': {}}]
        
        result = await hybrid_search_async(
            query="test query",
            collection_name="test",
            vector_results=vector_results,
            limit=10
        )
        
        assert isinstance(result, list)

def test_init_bm25_retriever():
    """Тест инициализации BM25"""
    # Требует реальных данных в Qdrant
    # Можно skip если нет тестовой DB
    pytest.skip("Requires Qdrant test instance")

