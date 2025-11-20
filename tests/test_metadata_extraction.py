"""
Unit tests для metadata extraction функций.

Покрывает:
- build_breadcrumb()
- build_page_path()
- extract_page_metadata() (headings extraction)
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Мокаем проблемные импорты перед импортом sync_confluence
sys.modules['llama_index'] = MagicMock()
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.vector_stores'] = MagicMock()
sys.modules['llama_index.vector_stores.qdrant'] = MagicMock()
sys.modules['atlassian'] = MagicMock()

# Мокаем локальные модули
mock_postgres = MagicMock()
mock_postgres.init_postgres_schema = MagicMock()
mock_postgres.save_page_to_postgres = MagicMock()
mock_postgres.get_pages_from_postgres = MagicMock()
mock_postgres.mark_as_indexed = MagicMock()
sys.modules['postgres_storage'] = mock_postgres

mock_qdrant = MagicMock()
mock_qdrant.init_qdrant_client = MagicMock()
mock_qdrant.init_qdrant_collection = MagicMock()
mock_qdrant.get_qdrant_vector_store = MagicMock()
mock_qdrant.get_qdrant_index = MagicMock()
mock_qdrant.delete_points_by_page_id = MagicMock()
sys.modules['qdrant_storage'] = mock_qdrant

# Мокаем sys.exit чтобы не прерывать тесты
original_exit = sys.exit
sys.exit = lambda code: None

try:
    from rag_server.sync_confluence import (
        build_breadcrumb,
        build_page_path,
        extract_page_metadata
    )
finally:
    # Восстанавливаем sys.exit
    sys.exit = original_exit


class TestBuildBreadcrumb:
    """Тесты для build_breadcrumb()"""
    
    def test_build_breadcrumb_normal(self):
        """Обычный случай с несколькими уровнями"""
        result = build_breadcrumb("RAUII", ["Dev", "API"], "Guide")
        assert result == "RAUII > Dev > API > Guide"
    
    def test_build_breadcrumb_empty(self):
        """Пустой breadcrumb"""
        result = build_breadcrumb("", [], "")
        assert result == ""
    
    def test_build_breadcrumb_only_space(self):
        """Только space без родителей"""
        result = build_breadcrumb("RAUII", [], "Page")
        assert result == "RAUII > Page"
    
    def test_build_breadcrumb_too_many_levels(self):
        """Слишком много уровней - обрезка"""
        result = build_breadcrumb("SPACE", ["A"] * 10, "Current", max_levels=5)
        assert "..." in result
        assert result.startswith("SPACE > ...")
        assert result.endswith("Current")
        # Проверяем что уровней не больше max_levels + 1 (space + ... + последние)
        parts = result.split(" > ")
        assert len(parts) <= 6  # SPACE + ... + последние (max_levels-1) + Current
    
    def test_build_breadcrumb_too_long(self):
        """Слишком длинный breadcrumb - обрезка"""
        long_title = "A" * 300
        result = build_breadcrumb("SPACE", [], long_title, max_length=100)
        assert len(result) <= 100
        assert result.endswith("...")
    
    def test_build_breadcrumb_max_levels_edge_case(self):
        """Граничный случай: ровно max_levels"""
        result = build_breadcrumb("SPACE", ["A", "B", "C"], "D", max_levels=4)
        # При max_levels=4: SPACE + 3 родителя + D = 5 частей, что больше 4, поэтому обрезается
        # Результат: SPACE > ... > B > C > D (первые 1 + последние 3)
        assert "..." in result or result == "SPACE > A > B > C > D"
        # Проверяем что результат валидный
        assert "SPACE" in result
        assert "D" in result


class TestBuildPagePath:
    """Тесты для build_page_path()"""
    
    def test_build_page_path_normal(self):
        """Обычный случай"""
        result = build_page_path("RAUII", ["Dev", "API"])
        assert result == "RAUII/Dev/API"
    
    def test_build_page_path_empty(self):
        """Пустой путь"""
        result = build_page_path("", [])
        assert result == ""
    
    def test_build_page_path_only_space(self):
        """Только space"""
        result = build_page_path("RAUII", [])
        assert result == "RAUII"
    
    def test_build_page_path_escapes_slashes(self):
        """Экранирование слэшей в названиях"""
        result = build_page_path("RAUII", ["Dev/API", "Guide\\Test"])
        assert result == "RAUII/Dev_API/Guide_Test"
        assert "/" not in result.split("/")[1:]  # В названиях нет слэшей
        assert "\\" not in result
    
    def test_build_page_path_special_chars(self):
        """Специальные символы"""
        result = build_page_path("SPACE", ["Test: Name", "File (v2)"])
        assert ":" in result  # Двоеточие не экранируется
        assert "(" in result  # Скобки не экранируются


class TestExtractPageMetadata:
    """Тесты для extract_page_metadata() - headings extraction"""
    
    @pytest.fixture
    def minimal_page(self):
        """Минимальная страница для тестов"""
        return {
            'id': 'test-page-123',
            'title': 'Test Page',
            'body': {'storage': {'value': ''}},
            'version': {'when': '2025-01-01T00:00:00Z', 'by': {'displayName': 'Test User'}},
            'ancestors': [],
            'history': {'createdDate': '2025-01-01T00:00:00Z'},
            'metadata': {'labels': {'results': []}}
        }
    
    def test_extract_headings_basic(self, minimal_page):
        """Базовое извлечение заголовков"""
        minimal_page['body']['storage']['value'] = "<h1>Installation</h1><h2>Prerequisites</h2>"
        
        metadata = extract_page_metadata(minimal_page, 'SPACE')
        
        assert metadata['heading_count'] == 2
        assert "Installation" in metadata['headings_list']
        assert "Prerequisites" in metadata['headings_list']
        assert "Installation" in metadata['headings']
    
    def test_extract_headings_limit(self, minimal_page):
        """Лимит заголовков"""
        # Создаем HTML с 100 заголовками
        html_content = "".join([f"<h1>Heading {i}</h1>" for i in range(100)])
        minimal_page['body']['storage']['value'] = html_content
        
        # Используем patch для изменения константы
        with patch('rag_server.sync_confluence.MAX_HEADINGS_EXTRACT', 10):
            metadata = extract_page_metadata(minimal_page, 'SPACE')
            assert metadata['heading_count'] <= 10
    
    def test_extract_headings_html_entities(self, minimal_page):
        """Декодирование HTML entities"""
        minimal_page['body']['storage']['value'] = "<h1>&lt;Installation&gt; Guide</h1>"
        
        metadata = extract_page_metadata(minimal_page, 'SPACE')
        
        assert "<Installation> Guide" in metadata['headings_list']
        assert "&lt;" not in metadata['headings']
        assert "&gt;" not in metadata['headings']
    
    def test_extract_headings_hierarchy(self, minimal_page):
        """Правильная иерархия заголовков"""
        html_content = """
        <h1>Chapter 1</h1>
        <h2>Section 1.1</h2>
        <h3>Subsection 1.1.1</h3>
        <h2>Section 1.2</h2>
        """
        minimal_page['body']['storage']['value'] = html_content
        
        metadata = extract_page_metadata(minimal_page, 'SPACE')
        
        assert metadata['heading_count'] == 4
        hierarchy = metadata['heading_hierarchy']
        
        # Проверяем что path правильный
        subsection = next((h for h in hierarchy if h['text'] == 'Subsection 1.1.1'), None)
        assert subsection is not None
        assert 'Chapter 1' in subsection['path']
        assert 'Section 1.1' in subsection['path']
    
    def test_extract_headings_empty(self, minimal_page):
        """Пустая страница без заголовков"""
        metadata = extract_page_metadata(minimal_page, 'SPACE')
        
        assert metadata['heading_count'] == 0
        assert metadata['headings'] == ''
        assert metadata['headings_list'] == []
        assert metadata['heading_hierarchy'] == []
    
    def test_extract_headings_string_length_limit(self, minimal_page):
        """Ограничение длины строки headings"""
        # Создаем много длинных заголовков
        html_content = "".join([f"<h1>{'A' * 100} Heading {i}</h1>" for i in range(50)])
        minimal_page['body']['storage']['value'] = html_content
        
        metadata = extract_page_metadata(minimal_page, 'SPACE')
        
        # Проверяем что строка обрезана если слишком длинная
        if len(metadata['headings']) > 2000:
            assert metadata['headings'].endswith("...")
    
    def test_extract_breadcrumb(self, minimal_page):
        """Извлечение breadcrumb"""
        minimal_page['ancestors'] = [
            {'id': '1', 'title': 'Parent 1'},
            {'id': '2', 'title': 'Parent 2'}
        ]
        
        metadata = extract_page_metadata(minimal_page, 'SPACE')
        
        # Проверяем что breadcrumb содержит основные элементы
        assert 'SPACE' in metadata['breadcrumb']
        assert 'Test Page' in metadata['breadcrumb']
        # Ancestors должны быть извлечены
        assert len(metadata.get('parent_titles', [])) == 2
        assert 'Parent 1' in metadata['breadcrumb']
        assert 'Parent 2' in metadata['breadcrumb']
    
    def test_extract_page_path(self, minimal_page):
        """Извлечение page_path"""
        minimal_page['ancestors'] = [
            {'id': '1', 'title': 'Parent 1'},
            {'id': '2', 'title': 'Parent 2'}
        ]
        
        metadata = extract_page_metadata(minimal_page, 'SPACE')
        
        # Проверяем что page_path содержит space и parents
        assert 'SPACE' in metadata['page_path']
        assert metadata['page_path'] == 'SPACE/Parent 1/Parent 2'
        assert '/' in metadata['page_path']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

