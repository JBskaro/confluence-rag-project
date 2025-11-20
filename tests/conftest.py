"""
Pytest configuration и общие fixtures.
"""
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Добавляем rag_server в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag_server'))


@pytest.fixture
def mock_confluence():
    """Mock Confluence API client."""
    mock = Mock()
    mock.get_page_by_id.return_value = {
        'id': '12345',
        'title': 'Test Page',
        'version': {
            'number': 1,
            'when': '2024-01-15T10:30:00.000Z',
            'by': {'displayName': 'Test User'}
        },
        'body': {
            'storage': {
                'value': '<p>Test content</p>'
            }
        },
        'metadata': {
            'labels': {
                'results': [{'name': 'test'}]
            }
        },
        'ancestors': [],
        'children': {'page': {'size': 0}}
    }
    mock.get_all_spaces.return_value = {
        'results': [
            {'key': 'TEST', 'name': 'Test Space'}
        ]
    }
    mock.get_all_pages_from_space.return_value = [
        {
            'id': '12345',
            'title': 'Test Page',
            'version': {'when': '2024-01-15T10:30:00.000Z'}
        }
    ]
    return mock


@pytest.fixture
def mock_chromadb_collection():
    """Mock ChromaDB collection."""
    mock = Mock()
    mock.get.return_value = {
        'ids': ['id1', 'id2'],
        'metadatas': [
            {
                'page_id': 'page1',
                'title': 'Page 1',
                'space': 'TEST',
                'chunk': 0,
                'url': 'http://test.com/1',
                'labels': 'test',
                'parent_title': ''
            }
        ]
    }
    mock.count.return_value = 100
    mock.delete.return_value = None
    return mock


@pytest.fixture
def mock_chroma_client(mock_chromadb_collection):
    """Mock ChromaDB client."""
    mock = Mock()
    mock.get_or_create_collection.return_value = mock_chromadb_collection
    return mock


@pytest.fixture
def sample_confluence_page():
    """Sample Confluence page data."""
    return {
        'id': '123456',
        'title': 'Sample Page',
        'type': 'page',
        'space': {'key': 'TEST'},
        'version': {
            'number': 5,
            'when': '2024-01-15T10:30:00.000Z',
            'by': {'displayName': 'John Doe'}
        },
        'body': {
            'storage': {
                'value': '''
                <h1>Main Heading</h1>
                <p>Some content</p>
                <ac:structured-macro ac:name="info">
                    <ac:rich-text-body>Important information</ac:rich-text-body>
                </ac:structured-macro>
                '''
            }
        },
        'metadata': {
            'labels': {
                'results': [
                    {'name': 'important'},
                    {'name': 'documentation'}
                ]
            }
        },
        'ancestors': [
            {'id': '111', 'title': 'Parent Page'}
        ],
        'children': {
            'page': {'size': 3}
        }
    }


@pytest.fixture
def sample_html_with_macros():
    """Sample HTML с Confluence макросами."""
    return '''
    <h1>Test Document</h1>
    <p>Introduction text</p>
    
    <ac:structured-macro ac:name="info">
        <ac:rich-text-body>
            <p>This is important information</p>
        </ac:rich-text-body>
    </ac:structured-macro>
    
    <ac:structured-macro ac:name="warning">
        <ac:rich-text-body>
            <p>This is a warning</p>
        </ac:rich-text-body>
    </ac:structured-macro>
    
    <ac:structured-macro ac:name="code">
        <ac:parameter ac:name="language">python</ac:parameter>
        <ac:plain-text-body><![CDATA[
def hello_world():
    print("Hello, World!")
        ]]></ac:plain-text-body>
    </ac:structured-macro>
    
    <h2>Section 1</h2>
    <p>Section content</p>
    '''


@pytest.fixture
def sample_markdown_text():
    """Sample markdown text с заголовками."""
    return '''# Main Heading

Some introduction text here.

## Section 1

Content of section 1 with multiple paragraphs.

This is the second paragraph in section 1.

### Subsection 1.1

Detailed content in subsection.

## Section 2

Content of section 2.

### Subsection 2.1

More detailed content.

#### Sub-subsection 2.1.1

Even more detailed content.
'''


@pytest.fixture
def mock_llama_index_retriever():
    """Mock LlamaIndex retriever."""
    mock = Mock()
    mock_result = Mock()
    mock_result.text = "Sample retrieved text"
    mock_result.metadata = {
        'title': 'Test Page',
        'space': 'TEST',
        'chunk': 0,
        'url': 'http://test.com',
        'heading': 'Section',
        'heading_level': 2,
        'labels': 'test',
        'parent_title': '',
        'created_by': 'User',
        'attachments': ''
    }
    mock.retrieve.return_value = [mock_result]
    return mock


@pytest.fixture
def mock_vector_index(mock_llama_index_retriever):
    """Mock VectorStoreIndex."""
    mock = Mock()
    mock.as_retriever.return_value = mock_llama_index_retriever
    mock.insert.return_value = None
    return mock


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_state_file(tmp_path):
    """Временный файл состояния для тестов."""
    state_file = tmp_path / "sync_state.json"
    return str(state_file)


@pytest.fixture
def sample_sync_state():
    """Sample sync state."""
    return {
        'last_sync': 1234567890,
        'pages': {
            '12345': {
                'updated': 20240115,
                'chunks': 5
            },
            '67890': {
                'updated': 20240116,
                'chunks': 3
            }
        }
    }

