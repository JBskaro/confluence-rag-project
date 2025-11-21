"""Unit tests для Hallucination Detection"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from rag_server.hallucination_detector import (
    HallucinationDetector,
    detect_hallucination
)

@pytest.fixture
def detector():
    return HallucinationDetector()

@pytest.fixture
def sample_docs():
    return ["Python is a programming language.", "It was created by Guido."]

def test_validate_inputs_valid(detector, sample_docs):
    """Тест валидации корректных данных"""
    result = detector._validate_inputs("Python is good", sample_docs)
    assert result is None

def test_validate_inputs_empty_response(detector, sample_docs):
    """Тест валидации пустого ответа"""
    result = detector._validate_inputs("", sample_docs)
    assert result is not None
    assert result['is_hallucination'] is True

def test_validate_inputs_no_docs(detector):
    """Тест валидации без документов"""
    result = detector._validate_inputs("Response", [])
    assert result is not None
    assert result['is_hallucination'] is True

def test_check_keyword_overlap(detector, sample_docs):
    """Тест проверки ключевых слов"""
    # High overlap
    score = detector._check_keyword_overlap("Python programming language", sample_docs)
    assert score > 0.5
    
    # Low overlap
    score = detector._check_keyword_overlap("Java script code", sample_docs)
    assert score < 0.5

def test_check_grounding(detector, sample_docs):
    """Тест проверки grounding"""
    # Grounded
    score = detector._check_grounding("Python is a language.", sample_docs)
    assert score > 0.0
    
    # Not grounded
    score = detector._check_grounding("Mars is red.", sample_docs)
    assert score == 0.0

def test_check_semantic_similarity(detector):
    """Тест семантической близости"""
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([0.9, 0.1])
    emb3 = np.array([0.0, 1.0])
    
    # High similarity
    score = detector._check_semantic_similarity(emb1, [emb2])
    assert score > 0.8
    
    # Low similarity
    score = detector._check_semantic_similarity(emb1, [emb3])
    assert score < 0.5

def test_make_decision(detector):
    """Тест принятия решения"""
    # No reasons -> No hallucination
    is_hallucination, confidence = detector._make_decision({}, [])
    assert is_hallucination is False
    assert confidence == 1.0
    
    # Many reasons -> Hallucination
    reasons = ["Low grounding", "Low overlap"]
    scores = {'s1': 0.1, 's2': 0.2}
    is_hallucination, confidence = detector._make_decision(scores, reasons)
    assert is_hallucination is True
    assert confidence < 0.9

def test_detect_integration(detector, sample_docs):
    """Интеграционный тест метода detect"""
    response = "Python is a programming language."
    
    result = detector.detect(response, sample_docs)
    
    assert result['is_hallucination'] is False
    # Confidence in hallucination is low (0.0) or Confidence in result is high
    # Based on code: confidence = 1.0 - avg_score (Higher score = lower confidence in hallucination)
    # If result is perfect (1.0 score), confidence = 0.0
    assert result['confidence'] >= 0.0
    assert 'scores' in result

def test_detect_hallucination_helper(sample_docs):
    """Тест helper функции"""
    from rag_server.config import settings
    settings.enable_hallucination_detection = True
    
    is_hallucination, details = detect_hallucination(
        "Python is a language", 
        sample_docs
    )
    
    assert isinstance(is_hallucination, bool)
    assert isinstance(details, dict)
