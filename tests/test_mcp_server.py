"""
Unit tests для MCP сервера.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag_server'))


class TestMCPServer:
    """Тесты для MCP сервера."""
    
    @patch('mcp_rag_secure.chromadb')
    @patch('mcp_rag_secure.init_embeddings')
    def test_init_rag_success(self, mock_embeddings, mock_chromadb):
        """Тест успешной инициализации RAG."""
        from mcp_rag_secure import init_rag
        
        # Mock ChromaDB
        mock_collection = Mock()
        mock_collection.get.return_value = {'ids': ['id1', 'id2']}
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Mock embeddings
        mock_embeddings.return_value = Mock()
        
        collection, storage_context, index = init_rag()
        
        assert collection is not None
        assert storage_context is not None
        assert index is not None
    
    @patch('mcp_rag_secure.collection', None)
    def test_ensure_rag_initialized_when_none(self):
        """Тест ленивой инициализации когда RAG = None."""
        from mcp_rag_secure import ensure_rag_initialized
        
        with patch('mcp_rag_secure.init_rag') as mock_init:
            mock_init.return_value = (Mock(), Mock(), Mock())
            ensure_rag_initialized()
            mock_init.assert_called_once()
    
    @patch('mcp_rag_secure.collection', Mock())
    def test_ensure_rag_initialized_when_exists(self):
        """Тест что RAG не реинициализируется если уже есть."""
        from mcp_rag_secure import ensure_rag_initialized
        
        with patch('mcp_rag_secure.init_rag') as mock_init:
            ensure_rag_initialized()
            mock_init.assert_not_called()


class TestConfluenceSemanticSearch:
    """Тесты для confluence_semantic_search tool."""
    
    @patch('mcp_rag_secure.ensure_rag_initialized')
    @patch('mcp_rag_secure.collection')
    @patch('mcp_rag_secure.index')
    def test_search_with_results(self, mock_index, mock_collection, mock_ensure):
        """Тест поиска с результатами."""
        from mcp_rag_secure import confluence_semantic_search
        
        # Mock collection
        mock_collection.get.return_value = {'ids': ['id1']}
        
        # Mock retriever results
        mock_result = Mock()
        mock_result.text = "Test content"
        mock_result.metadata = {
            'title': 'Test Page',
            'space': 'TEST',
            'chunk': 0,
            'url': 'http://test.com',
            'heading': 'Section 1',
            'heading_level': 2,
            'labels': 'tag1,tag2',
            'parent_title': 'Parent',
            'created_by': 'User',
            'attachments': 'file.pdf'
        }
        
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [mock_result]
        mock_index.as_retriever.return_value = mock_retriever
        
        result = confluence_semantic_search("test query", limit=5)
        
        assert "✅" in result or "Найдено" in result
        assert "Test Page" in result
    
    @patch('mcp_rag_secure.ensure_rag_initialized')
    @patch('mcp_rag_secure.collection')
    def test_search_empty_index(self, mock_collection, mock_ensure):
        """Тест поиска по пустому индексу."""
        from mcp_rag_secure import confluence_semantic_search
        
        mock_collection.get.return_value = {'ids': []}
        
        result = confluence_semantic_search("test query")
        
        assert "пуст" in result.lower() or "empty" in result.lower()
    
    @patch('mcp_rag_secure.ensure_rag_initialized')
    @patch('mcp_rag_secure.collection', None)
    @patch('mcp_rag_secure.index', None)
    def test_search_not_initialized(self, mock_ensure):
        """Тест поиска когда RAG не инициализирован."""
        from mcp_rag_secure import confluence_semantic_search
        
        result = confluence_semantic_search("test query")
        
        assert "ошибка" in result.lower() or "error" in result.lower()
    
    @patch('mcp_rag_secure.ensure_rag_initialized')
    @patch('mcp_rag_secure.collection')
    @patch('mcp_rag_secure.index')
    def test_search_limit_normalization(self, mock_index, mock_collection, mock_ensure):
        """Тест нормализации лимита результатов."""
        from mcp_rag_secure import confluence_semantic_search
        
        mock_collection.get.return_value = {'ids': ['id1']}
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        mock_index.as_retriever.return_value = mock_retriever
        
        # Тест отрицательного лимита
        result = confluence_semantic_search("test", limit=-5)
        # Должен нормализоваться до 1
        mock_index.as_retriever.assert_called()
        
        # Тест слишком большого лимита
        result = confluence_semantic_search("test", limit=100)
        # Должен нормализоваться до 20


class TestConfluenceSearchByLabel:
    """Тесты для confluence_search_by_label tool."""
    
    @patch('mcp_rag_secure.ensure_rag_initialized')
    @patch('mcp_rag_secure.collection')
    def test_search_by_label_found(self, mock_collection, mock_ensure):
        """Тест поиска по метке с результатами."""
        from mcp_rag_secure import confluence_search_by_label
        
        mock_collection.get.side_effect = [
            {'ids': ['id1']},  # Первый вызов для проверки пустоты
            {  # Второй вызов для получения данных
                'ids': ['id1', 'id2'],
                'metadatas': [
                    {
                        'page_id': 'page1',
                        'title': 'Page 1',
                        'space': 'TEST',
                        'url': 'http://test.com/1',
                        'labels': 'important,test',
                        'parent_title': 'Parent'
                    },
                    {
                        'page_id': 'page2',
                        'title': 'Page 2',
                        'space': 'TEST',
                        'url': 'http://test.com/2',
                        'labels': 'test',
                        'parent_title': ''
                    }
                ]
            }
        ]
        
        result = confluence_search_by_label("test", limit=5)
        
        assert "✅" in result or "Найдено" in result
        assert "Page 1" in result or "Page 2" in result
    
    @patch('mcp_rag_secure.ensure_rag_initialized')
    @patch('mcp_rag_secure.collection')
    def test_search_by_label_not_found(self, mock_collection, mock_ensure):
        """Тест поиска по несуществующей метке."""
        from mcp_rag_secure import confluence_search_by_label
        
        mock_collection.get.side_effect = [
            {'ids': ['id1']},
            {'ids': ['id1'], 'metadatas': [{'labels': 'other', 'page_id': 'p1'}]}
        ]
        
        result = confluence_search_by_label("nonexistent")
        
        assert "не найдено" in result.lower() or "not found" in result.lower()


class TestConfluenceHealth:
    """Тесты для confluence_health tool."""
    
    @patch('mcp_rag_secure.ensure_rag_initialized')
    @patch('mcp_rag_secure.collection')
    def test_health_with_documents(self, mock_collection, mock_ensure):
        """Тест health check с документами."""
        from mcp_rag_secure import confluence_health
        
        mock_collection.count.return_value = 100
        
        result = confluence_health()
        
        assert "✅" in result or "работает" in result.lower()
        assert "100" in result
    
    @patch('mcp_rag_secure.ensure_rag_initialized')
    @patch('mcp_rag_secure.collection')
    def test_health_empty_index(self, mock_collection, mock_ensure):
        """Тест health check с пустым индексом."""
        from mcp_rag_secure import confluence_health
        
        mock_collection.count.return_value = 0
        
        result = confluence_health()
        
        assert "пуст" in result.lower() or "empty" in result.lower()
    
    @patch('mcp_rag_secure.ensure_rag_initialized')
    @patch('mcp_rag_secure.collection')
    def test_health_count_fallback(self, mock_collection, mock_ensure):
        """Тест health check с fallback если count() не работает."""
        from mcp_rag_secure import confluence_health
        
        # count() выбрасывает exception, должен использовать get()
        mock_collection.count.side_effect = Exception("count not available")
        mock_collection.get.return_value = {'ids': ['id1', 'id2']}
        
        result = confluence_health()
        
        # Должен вернуть что-то разумное
        assert isinstance(result, str)
        assert len(result) > 0


class TestInitEmbeddings:
    """Тесты для init_embeddings."""
    
    @patch.dict(os.environ, {'USE_OLLAMA': 'false'})
    @patch('mcp_rag_secure.HuggingFaceEmbedding')
    def test_huggingface_embeddings(self, mock_hf):
        """Тест инициализации HuggingFace embeddings."""
        from mcp_rag_secure import init_embeddings
        
        mock_hf.return_value = Mock()
        
        result = init_embeddings()
        
        assert result is not None
        mock_hf.assert_called_once()
    
    @patch.dict(os.environ, {'USE_OLLAMA': 'true'})
    @patch('mcp_rag_secure.OllamaEmbedding')
    def test_ollama_embeddings(self, mock_ollama):
        """Тест инициализации Ollama embeddings."""
        from mcp_rag_secure import init_embeddings
        
        mock_ollama.return_value = Mock()
        
        result = init_embeddings()
        
        assert result is not None
        mock_ollama.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

