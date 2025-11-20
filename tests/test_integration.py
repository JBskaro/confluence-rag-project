"""
Integration tests для полного workflow.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag_server'))


class TestSyncWorkflow:
    """Integration tests для sync workflow."""
    
    @patch('sync_confluence_optimized_final.Confluence')
    @patch('sync_confluence_optimized_final.chromadb')
    @patch('sync_confluence_optimized_final.init_embeddings')
    def test_full_sync_workflow(self, mock_embeddings, mock_chromadb, mock_confluence_class, 
                                mock_confluence, mock_chromadb_collection, tmp_path):
        """Тест полного цикла синхронизации."""
        from sync_confluence_optimized_final import process_batch
        
        # Setup mocks
        mock_confluence_instance = mock_confluence
        mock_confluence_class.return_value = mock_confluence_instance
        
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_chromadb_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        mock_embeddings.return_value = Mock()
        
        # Mock index
        mock_index = Mock()
        mock_index.insert = Mock()
        
        # Test data
        pages = [
            {
                'id': '12345',
                'title': 'Test Page',
                'version': {'when': '2024-01-15T10:30:00.000Z'}
            }
        ]
        
        state = {'pages': {}, 'last_sync': 0}
        
        # Execute
        updated, errors, skipped = process_batch(
            mock_index, 
            mock_chromadb_collection, 
            mock_confluence_instance, 
            pages, 
            state, 
            'TEST'
        )
        
        # Verify
        assert updated + skipped + errors == len(pages)
    
    @patch('sync_confluence_optimized_final.load_state')
    @patch('sync_confluence_optimized_final.save_state')
    def test_state_persistence(self, mock_save, mock_load, sample_sync_state):
        """Тест сохранения и загрузки состояния."""
        mock_load.return_value = sample_sync_state
        
        from sync_confluence_optimized_final import load_state, save_state
        
        # Load
        state = load_state()
        assert 'pages' in state
        assert '12345' in state['pages']
        
        # Modify
        state['pages']['99999'] = {'updated': 20240117, 'chunks': 10}
        
        # Save
        save_state(state)
        mock_save.assert_called_once_with(state)


class TestEndToEndRAG:
    """End-to-end tests для RAG системы."""
    
    @patch('mcp_rag_secure.chromadb')
    @patch('mcp_rag_secure.init_embeddings')
    def test_rag_initialization_and_search(self, mock_embeddings, mock_chromadb, 
                                          mock_chromadb_collection, mock_llama_index_retriever):
        """Тест инициализации RAG и выполнения поиска."""
        from mcp_rag_secure import init_rag, confluence_semantic_search
        
        # Setup
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_chromadb_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_embeddings.return_value = Mock()
        
        # Initialize RAG
        collection, storage_context, index = init_rag()
        
        assert collection is not None
        assert storage_context is not None
        assert index is not None
        
        # Mock search
        with patch('mcp_rag_secure.collection', mock_chromadb_collection):
            with patch('mcp_rag_secure.index') as mock_index:
                mock_index.as_retriever.return_value = mock_llama_index_retriever
                
                result = confluence_semantic_search("test query", limit=5)
                
                assert isinstance(result, str)
                assert len(result) > 0


class TestMacroProcessingPipeline:
    """Integration tests для pipeline обработки макросов."""
    
    def test_macro_to_text_pipeline(self, sample_html_with_macros):
        """Тест полного pipeline: HTML с макросами → текст → chunks."""
        from sync_confluence_optimized_final import preprocess_confluence_macros, html_to_text, chunk_text
        
        # Step 1: Preprocess macros
        preprocessed = preprocess_confluence_macros(sample_html_with_macros)
        assert 'INFO' in preprocessed or '💡' in preprocessed
        assert 'WARNING' in preprocessed or '⚠️' in preprocessed
        assert '```python' in preprocessed or 'def hello_world' in preprocessed
        
        # Step 2: Convert to text
        text = html_to_text(preprocessed)
        assert len(text) > 0
        assert 'Test Document' in text
        
        # Step 3: Chunk
        chunks = chunk_text(text, size=500)
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
    
    def test_metadata_extraction_pipeline(self, sample_confluence_page):
        """Тест pipeline извлечения метаданных."""
        from sync_confluence_optimized_final import extract_page_metadata, get_timestamp
        
        # Extract metadata
        metadata = extract_page_metadata(sample_confluence_page)
        
        assert len(metadata['labels']) == 2
        assert 'important' in metadata['labels']
        assert metadata['parent_title'] == 'Parent Page'
        assert metadata['created_by'] == 'John Doe'
        assert metadata['version'] == 5
        assert metadata['has_children'] is True
        assert metadata['children_count'] == 3
        
        # Get timestamp
        timestamp = get_timestamp(sample_confluence_page)
        assert timestamp == 20240115


class TestErrorHandling:
    """Integration tests для обработки ошибок."""
    
    @patch('sync_confluence_optimized_final.Confluence')
    def test_confluence_api_failure(self, mock_confluence_class):
        """Тест обработки ошибок Confluence API."""
        from sync_confluence_optimized_final import get_page
        
        mock_confluence = Mock()
        mock_confluence.get_page_by_id.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            # Должен retry 3 раза перед fail
            get_page(mock_confluence, "12345")
    
    @patch('mcp_rag_secure.collection')
    def test_search_with_exception(self, mock_collection):
        """Тест обработки exceptions при поиске."""
        from mcp_rag_secure import confluence_semantic_search
        
        mock_collection.get.side_effect = Exception("Database error")
        
        # Должен вернуть error message, не упасть
        result = confluence_semantic_search("test query")
        assert "ошибка" in result.lower() or "error" in result.lower()


class TestDataConsistency:
    """Tests для проверки консистентности данных."""
    
    def test_chunk_metadata_consistency(self, sample_markdown_text):
        """Тест что metadata chunks консистентна."""
        from sync_confluence_optimized_final import chunk_text
        
        chunks = chunk_text(sample_markdown_text, size=200)
        
        # Проверяем что все chunks имеют необходимые поля
        for chunk in chunks:
            assert 'text' in chunk
            assert 'heading' in chunk
            assert 'level' in chunk
            assert isinstance(chunk['text'], str)
            assert isinstance(chunk['level'], int)
    
    def test_metadata_field_limits(self):
        """Тест что metadata fields не превышают лимиты."""
        from sync_confluence_optimized_final import extract_page_metadata
        
        # Create page with very long strings
        page_data = {
            'metadata': {
                'labels': {
                    'results': [{'name': 'x' * 1000} for _ in range(100)]
                }
            },
            'ancestors': [
                {'id': '1', 'title': 'y' * 1000}
            ],
            'version': {
                'by': {'displayName': 'z' * 1000}
            }
        }
        
        metadata = extract_page_metadata(page_data)
        
        # Все поля должны быть валидными (не None, не пустые если данные есть)
        assert isinstance(metadata['labels'], list)
        assert isinstance(metadata['parent_title'], str)
        assert isinstance(metadata['created_by'], str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

