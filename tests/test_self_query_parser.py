#!/usr/bin/env python3
"""
Unit tests для rag_server/self_query_parser.py
"""
import pytest
import sys
import os

# Добавляем путь к rag_server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_server.self_query_parser import parse_self_query


def test_space_extraction():
    """Test space extraction from query"""
    result = parse_self_query("docs from DevOps space")
    assert result['filters']['space'] == 'DevOps'
    assert 'docs' in result['clean_query']
    assert 'DevOps' not in result['clean_query']


def test_date_extraction():
    """Test date extraction from query"""
    result = parse_self_query("latest API documentation")
    assert 'date_from' in result['filters']
    assert result['intent'] == 'api'


def test_author_extraction():
    """Test author extraction from query"""
    result = parse_self_query("installation guide by John Doe")
    assert result['filters']['author'] == 'John Doe'
    assert 'installation' in result['clean_query']


def test_content_type_extraction():
    """Test content type extraction"""
    result = parse_self_query("tutorial about docker")
    assert result['filters']['content_type'] == 'tutorial'
    assert result['intent'] == 'guide'


def test_status_extraction():
    """Test status extraction"""
    result = parse_self_query("published documentation")
    assert result['filters']['status'] == 'published'


def test_multiple_filters():
    """Test extraction of multiple filters"""
    result = parse_self_query("latest docs from DevOps by John")
    assert 'space' in result['filters']
    assert 'author' in result['filters']
    assert 'date_from' in result['filters']


def test_empty_query():
    """Test empty query"""
    result = parse_self_query("")
    assert result['clean_query'] == ""
    assert len(result['filters']) == 0
    assert result['intent'] is None


def test_no_filters():
    """Test query with no filters"""
    result = parse_self_query("how to install docker")
    assert len(result['filters']) == 0
    assert result['clean_query'] == "how to install docker"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

