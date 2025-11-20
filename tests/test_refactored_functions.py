import sys
import os
from unittest.mock import MagicMock, patch

# Mock missing dependencies BEFORE importing ANY module
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.models'] = MagicMock()
sys.modules['qdrant_storage'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['fastmcp'] = MagicMock()

# Добавляем пути
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag_server')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
# Импортируем тестируемый модуль
from rag_server.mcp_rag_secure import expand_query, structural_metadata_search

class TestExpandQuery:
    """Тесты для проверки что рефакторинг не сломал функциональность"""

    @patch('rag_server.mcp_rag_secure.get_semantic_query_log')
    @patch('rag_server.mcp_rag_secure.get_synonyms_manager')
    @patch('rag_server.mcp_rag_secure.cached_rewrite_query')
    def test_basic_expansion(self, mock_rewrite, mock_synonyms, mock_semantic):
        """Базовый тест расширения запроса"""
        # Настраиваем моки
        mock_semantic.return_value.get_related_queries.return_value = []
        mock_synonyms.return_value.get_synonyms.return_value = ["хранилище"]
        mock_rewrite.return_value = ["склад номенклатура", "учет товаров"]

        result = expand_query("склад номенклатура")

        assert isinstance(result, list)
        assert len(result) > 0
        assert "склад номенклатура" in result

    def test_max_variants_respected(self):
        """Проверка что соблюдается лимит вариантов"""
        with patch('rag_server.mcp_rag_secure.cached_rewrite_query', return_value=["v1", "v2", "v3", "v4", "v5", "v6"]):
            with patch('rag_server.mcp_rag_secure.get_semantic_query_log') as mock_sem:
                mock_sem.return_value.get_related_queries.return_value = []
                with patch('rag_server.mcp_rag_secure.get_synonyms_manager') as mock_syn:
                    mock_syn.return_value.get_synonyms.return_value = []
                    
                    result = expand_query("тест")  # короткий запрос -> max 5
                    assert len(result) <= 5

    def test_1c_normalization(self):
        """Проверка нормализации 1С/1C"""
        with patch('rag_server.mcp_rag_secure.cached_rewrite_query', return_value=[]):
             with patch('rag_server.mcp_rag_secure.get_semantic_query_log') as mock_sem:
                 mock_sem.return_value.get_related_queries.return_value = []
                 with patch('rag_server.mcp_rag_secure.get_synonyms_manager') as mock_syn:
                    mock_syn.return_value.get_synonyms.return_value = []
                    result = expand_query("1С конфигурация")
                    assert any('1c' in q.lower() for q in result)

class TestStructuralSearch:
    """Тесты структурного поиска"""

    @patch('rag_server.mcp_rag_secure.get_all_metadata_cached')
    def test_structural_search_logic(self, mock_get_metadata):
        """Проверка логики структурного поиска"""
        # Подготовка данных
        mock_metadata = {
            'ids': ['1', '2'],
            'metadatas': [
                {'page_path': 'Склад > Учет', 'title': 'Номенклатура', 'page_id': '1', 'heading_path': '', 'heading': ''},
                {'page_path': 'Продажи > Отчеты', 'title': 'Выручка', 'page_id': '2', 'heading_path': '', 'heading': ''}
            ]
        }
        mock_get_metadata.return_value = mock_metadata

        structure = {
            'is_structural_query': True,
            'parts': ['Склад', 'Учет'],
            'original_query': 'Склад > Учет'
        }

        result = structural_metadata_search(None, structure, limit=10)

        assert len(result) == 1
        assert result[0]['metadata']['page_id'] == '1'
        assert result[0]['match_score'] > 0

    def test_non_structural_query(self):
        """Не структурный запрос возвращает пустой список"""
        result = structural_metadata_search(
            collection=None,
            structure={'is_structural_query': False, 'parts': []},
            limit=10
        )
        assert result == []

    def test_empty_parts(self):
        """Пустые части возвращают пустой список"""
        with patch('rag_server.mcp_rag_secure.get_all_metadata_cached') as mock_get:
             mock_get.return_value = {'ids': [], 'metadatas': []}
             result = structural_metadata_search(
                collection=None,
                structure={'is_structural_query': True, 'parts': []},
                limit=10
            )
             assert result == []
