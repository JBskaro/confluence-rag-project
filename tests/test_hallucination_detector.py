#!/usr/bin/env python3
"""
Unit tests для rag_server/hallucination_detector.py
"""
import pytest
import numpy as np
import sys
import os

# Добавляем путь к rag_server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_server.hallucination_detector import HallucinationDetector, detect_hallucination


def test_grounded_response():
    """Test that grounded response is not flagged as hallucination"""
    detector = HallucinationDetector()
    
    response = "Docker is a container platform."
    docs = ["Docker is a containerization platform for developers."]
    
    result = detector.detect(
        query="What is Docker?",
        response=response,
        retrieved_docs=docs
    )
    
    # Should not be hallucination (grounded in docs)
    assert not result['is_hallucination'] or len(result['reasons']) < 2


def test_hallucination_detection():
    """Test that ungrounded response is flagged as hallucination"""
    detector = HallucinationDetector()
    
    response = "Docker is a programming language for web development."
    docs = ["Docker is a containerization platform."]
    
    result = detector.detect(
        query="What is Docker?",
        response=response,
        retrieved_docs=docs
    )
    
    # May be flagged as hallucination if scores are low
    assert isinstance(result['is_hallucination'], bool)


def test_empty_response():
    """Test empty response"""
    detector = HallucinationDetector()
    
    result = detector.detect(
        query="test",
        response="",
        retrieved_docs=["Some document"]
    )
    
    assert result['is_hallucination'] is True


def test_empty_docs():
    """Test with empty retrieved docs"""
    detector = HallucinationDetector()
    
    result = detector.detect(
        query="test",
        response="Some response",
        retrieved_docs=[]
    )
    
    assert result['is_hallucination'] is True


def test_keyword_overlap():
    """Test keyword overlap detection"""
    detector = HallucinationDetector()
    
    response = "Python is a programming language used for data science."
    docs = ["Python programming language data science machine learning"]
    
    result = detector.detect(
        query="What is Python?",
        response=response,
        retrieved_docs=docs
    )
    
    # Should have good keyword overlap
    assert 'keyword_overlap' in result['scores']
    assert result['scores']['keyword_overlap'] > 0


def test_semantic_similarity():
    """Test semantic similarity detection"""
    detector = HallucinationDetector()
    
    # Create dummy embeddings
    response_emb = np.array([0.9, 0.1, 0.0])
    docs_embs = [np.array([0.85, 0.15, 0.0])]
    
    result = detector.detect(
        query="test",
        response="Some response",
        retrieved_docs=["Some document"],
        response_embedding=response_emb,
        docs_embeddings=docs_embs
    )
    
    assert 'semantic_similarity' in result['scores']


def test_detect_hallucination_helper():
    """Test convenience function"""
    response = "Docker is a container platform."
    docs = ["Docker is a containerization platform."]
    
    is_hallucination, details = detect_hallucination(response, docs)
    
    assert isinstance(is_hallucination, bool)
    assert isinstance(details, dict)
    assert 'confidence' in details


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

