import sys
import os
from unittest.mock import MagicMock

# Mocking dependencies
sys.modules['fastmcp'] = MagicMock()
# Mock qdrant and its submodules
qdrant_mock = MagicMock()
sys.modules['qdrant_client'] = qdrant_mock
sys.modules['qdrant_client.models'] = MagicMock()
sys.modules['qdrant_client.http'] = MagicMock()
sys.modules['qdrant_client.http.models'] = MagicMock()

sys.modules['sentence_transformers'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
rag_server_path = os.path.join(project_root, 'rag_server')
sys.path.insert(0, rag_server_path)

# Mock internal modules
sys.modules['embeddings'] = MagicMock()
sys.modules['synonyms_manager'] = MagicMock()
sys.modules['advanced_search'] = MagicMock()
sys.modules['query_rewriter'] = MagicMock()
sys.modules['observability'] = MagicMock()
sys.modules['hybrid_search'] = MagicMock()

from rag_server.mcp_rag_secure import parse_query_structure

def test_parse_query_structure_simple():
    query = "simple query"
    result = parse_query_structure(query)
    assert result['is_structural_query'] is False
    # For simple non-structural queries, it returns the original query in list
    # Based on code: parts = [query]
    assert result['parts'] == [query]

def test_parse_query_structure_with_separator():
    query = "Space > Page > Section"
    result = parse_query_structure(query)
    assert result['is_structural_query'] is True
    # Separator splitting is done on original query?
    # Code: parts = [p.strip() for p in query.split(sep) if p.strip()]
    # So it preserves case
    assert result['parts'] == ["Space", "Page", "Section"]

def test_parse_query_structure_with_regex_po_bloku():
    query = "по блоку Склад"
    result = parse_query_structure(query)
    assert result['is_structural_query'] is True
    # Regex matching is done on query_lower
    # So parts will be lowercase
    assert result['parts'] == ["склад"]

def test_parse_query_structure_with_regex_a_tochnee():
    query = "Склад, а точнее Учет"
    result = parse_query_structure(query)
    assert result['is_structural_query'] is True
    assert result['parts'] == ["склад", "учет"]

def test_parse_query_structure_complex_regex():
    query = "по блоку Продажи, а точнее Отчеты"
    result = parse_query_structure(query)
    assert result['is_structural_query'] is True
    assert result['parts'] == ["продажи", "отчеты"]

if __name__ == "__main__":
    try:
        test_parse_query_structure_simple()
        test_parse_query_structure_with_separator()
        test_parse_query_structure_with_regex_po_bloku()
        test_parse_query_structure_with_regex_a_tochnee()
        test_parse_query_structure_complex_regex()
        print("All structure parsing tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
