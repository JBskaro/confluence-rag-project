#!/usr/bin/env python3
"""
Unit tests для rag_server/mmr_reranker.py
"""
import pytest
import numpy as np
import sys
import os

# Добавляем путь к rag_server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_server.mmr_reranker import mmr_rerank, maximal_marginal_relevance


def test_mmr_diversification():
    """Test that MMR selects diverse results"""
    query_emb = np.array([1.0, 0.0, 0.0])
    
    results = [
        {'id': '1', 'embedding': [0.9, 0.1, 0.0], 'score': 0.9},
        {'id': '2', 'embedding': [0.9, 0.05, 0.0], 'score': 0.89},
        {'id': '3', 'embedding': [0.5, 0.5, 0.0], 'score': 0.7},
        {'id': '4', 'embedding': [0.0, 0.0, 1.0], 'score': 0.5},
    ]
    
    reranked = mmr_rerank(query_emb, results, diversity_weight=0.5, top_k=2)
    
    assert len(reranked) == 2
    # MMR должен выбрать результаты, которые не слишком похожи друг на друга
    # Первый результат должен быть одним из наиболее релевантных (1 или 2)
    assert reranked[0]['id'] in ['1', '2']
    # Второй результат должен быть более разнообразным (3 или 4)
    assert reranked[1]['id'] in ['3', '4']


def test_mmr_empty_results():
    """Test MMR with empty results"""
    query_emb = np.array([1.0, 0.0, 0.0])
    results = []
    
    reranked = mmr_rerank(query_emb, results, diversity_weight=0.3, top_k=5)
    assert len(reranked) == 0


def test_mmr_less_than_k():
    """Test MMR when results < k"""
    query_emb = np.array([1.0, 0.0, 0.0])
    results = [
        {'id': '1', 'embedding': [0.9, 0.1, 0.0], 'score': 0.9},
    ]
    
    reranked = mmr_rerank(query_emb, results, diversity_weight=0.3, top_k=5)
    assert len(reranked) == 1
    assert reranked[0]['id'] == '1'


def test_mmr_zero_diversity():
    """Test MMR with zero diversity (pure relevance)"""
    query_emb = np.array([1.0, 0.0, 0.0])
    results = [
        {'id': '1', 'embedding': [0.9, 0.1, 0.0], 'score': 0.9},
        {'id': '2', 'embedding': [0.8, 0.2, 0.0], 'score': 0.8},
        {'id': '3', 'embedding': [0.7, 0.3, 0.0], 'score': 0.7},
    ]
    
    reranked = mmr_rerank(query_emb, results, diversity_weight=0.0, top_k=2)
    assert len(reranked) == 2
    # With zero diversity, should select most relevant
    assert reranked[0]['id'] == '1'


def test_mmr_max_diversity():
    """Test MMR with maximum diversity"""
    query_emb = np.array([1.0, 0.0, 0.0])
    results = [
        {'id': '1', 'embedding': [0.9, 0.1, 0.0], 'score': 0.9},
        {'id': '2', 'embedding': [0.9, 0.05, 0.0], 'score': 0.89},
        {'id': '3', 'embedding': [0.0, 0.0, 1.0], 'score': 0.5},
    ]
    
    reranked = mmr_rerank(query_emb, results, diversity_weight=1.0, top_k=2)
    assert len(reranked) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

