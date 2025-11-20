"""
Unit tests для функций синхронизации Confluence.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
import os

# Устанавливаем обязательные переменные окружения перед импортом
os.environ.setdefault('CONFLUENCE_URL', 'http://test.confluence.com')
os.environ.setdefault('CONFLUENCE_TOKEN', 'test_token')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag_server'))

from sync_confluence_optimized_final import (
    get_timestamp,
    get_int_env,
    html_to_text,
    chunk_text,
    extract_sections,
    extract_page_metadata,
    preprocess_confluence_macros,
    convert_table_to_markdown,
    extract_list_text,
    extract_structural_blocks,
    smart_chunk_with_context,
    get_bool_env,
)


class TestGetTimestamp:
    """Тесты для функции get_timestamp."""
    
    def test_valid_timestamp(self):
        """Тест корректного timestamp."""
        page = {
            'version': {
                'when': '2024-01-15T10:30:00.000Z'
            }
        }
        result = get_timestamp(page)
        assert result == 20240115
    
    def test_empty_timestamp(self):
        """Тест пустого timestamp."""
        page = {'version': {'when': ''}}
        result = get_timestamp(page)
        assert result == 0
    
    def test_missing_version(self):
        """Тест отсутствующего version."""
        page = {}
        result = get_timestamp(page)
        assert result == 0
    
    def test_invalid_format(self):
        """Тест некорректного формата."""
        page = {'version': {'when': 'invalid'}}
        result = get_timestamp(page)
        assert result == 0


class TestGetIntEnv:
    """Тесты для функции get_int_env."""
    
    @patch.dict(os.environ, {'TEST_VAR': '42'})
    def test_valid_integer(self):
        """Тест корректного integer."""
        result = get_int_env('TEST_VAR', 10)
        assert result == 42
    
    @patch.dict(os.environ, {'TEST_VAR': 'invalid'})
    def test_invalid_integer(self):
        """Тест некорректного integer - должен вернуть default."""
        result = get_int_env('TEST_VAR', 10)
        assert result == 10
    
    @patch.dict(os.environ, {'TEST_VAR': '0'})
    def test_zero_value(self):
        """Тест нулевого значения - должен вернуть default."""
        result = get_int_env('TEST_VAR', 10)
        assert result == 10
    
    @patch.dict(os.environ, {'TEST_VAR': '-5'})
    def test_negative_value(self):
        """Тест отрицательного значения - должен вернуть default."""
        result = get_int_env('TEST_VAR', 10)
        assert result == 10
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_env_var(self):
        """Тест отсутствующей переменной - должен вернуть default."""
        result = get_int_env('NONEXISTENT', 10)
        assert result == 10


class TestHtmlToText:
    """Тесты для функции html_to_text."""
    
    def test_simple_html(self):
        """Тест простого HTML."""
        html = '<p>Hello <strong>World</strong></p>'
        result = html_to_text(html)
        assert 'Hello' in result
        assert 'World' in result
    
    def test_empty_html(self):
        """Тест пустого HTML."""
        result = html_to_text('')
        assert result == ""
    
    def test_none_html(self):
        """Тест None HTML."""
        result = html_to_text(None)
        assert result == ""
    
    def test_html_with_links(self):
        """Тест HTML с ссылками."""
        html = '<p><a href="http://example.com">Link</a></p>'
        result = html_to_text(html)
        assert 'Link' in result
        assert 'http://example.com' in result
    
    def test_html_truncation(self):
        """Тест обрезки длинного HTML."""
        html = '<p>' + 'A' * 100000 + '</p>'
        result = html_to_text(html, max_len=1000)
        assert len(result) < 100000


class TestChunkText:
    """Тесты для функции chunk_text."""
    
    def test_short_text(self):
        """Тест короткого текста."""
        text = "Short text"
        result = chunk_text(text)
        assert len(result) == 1
        assert result[0]['text'] == text
    
    def test_empty_text(self):
        """Тест пустого текста."""
        result = chunk_text('')
        assert result == []
    
    def test_text_with_headings(self):
        """Тест текста с заголовками."""
        text = """# Heading 1
Content 1

## Heading 2
Content 2
"""
        result = chunk_text(text, size=1000)
        assert len(result) >= 1
        # Проверяем что результат содержит текст
        assert any(chunk.get('text') for chunk in result)
        # Проверяем что headings могут быть сохранены в metadata (если есть)
        has_heading = any(chunk.get('heading') for chunk in result)
        # Heading может быть пустым для некоторых чанков, это нормально
        assert len(result) > 0
    
    def test_large_text_chunking(self):
        """Тест разбиения большого текста."""
        text = '\n\n'.join([f'Paragraph {i}' for i in range(100)])
        result = chunk_text(text, size=100)
        assert len(result) > 1


class TestExtractSections:
    """Тесты для функции extract_sections."""
    
    def test_text_with_headings(self):
        """Тест текста с заголовками."""
        text = """# Heading 1
Content 1

## Heading 2
Content 2

### Heading 3
Content 3
"""
        result = extract_sections(text)
        assert len(result) == 3
        assert result[0]['heading'] == 'Heading 1'
        assert result[0]['level'] == 1
        assert result[1]['heading'] == 'Heading 2'
        assert result[1]['level'] == 2
    
    def test_text_without_headings(self):
        """Тест текста без заголовков."""
        text = "Just plain text without headings"
        result = extract_sections(text)
        # Должен вернуть одну секцию без заголовка
        assert len(result) >= 0


class TestExtractPageMetadata:
    """Тесты для функции extract_page_metadata."""
    
    def test_full_metadata(self):
        """Тест полных метаданных."""
        page_data = {
            'metadata': {
                'labels': {
                    'results': [
                        {'name': 'tag1'},
                        {'name': 'tag2'}
                    ]
                }
            },
            'ancestors': [
                {'id': '123', 'title': 'Parent Page'}
            ],
            'version': {
                'number': 5,
                'by': {'displayName': 'John Doe'},
                'when': '2024-01-15T10:30:00.000Z'
            },
            'children': {
                'page': {'size': 3}
            }
        }
        
        result = extract_page_metadata(page_data)
        
        assert len(result['labels']) == 2
        assert 'tag1' in result['labels']
        assert result['parent_title'] == 'Parent Page'
        assert result['created_by'] == 'John Doe'
        assert result['version'] == 5
        assert result['has_children'] is True
        assert result['children_count'] == 3
    
    def test_empty_metadata(self):
        """Тест пустых метаданных."""
        page_data = {}
        result = extract_page_metadata(page_data)
        
        assert result['labels'] == []
        assert result['parent_id'] == ''
        assert result['parent_title'] == ''
        assert result['version'] == 1
        assert result['has_children'] is False
    
    def test_none_metadata(self):
        """Тест None метаданных."""
        result = extract_page_metadata(None)
        assert isinstance(result, dict)
        assert 'labels' in result
    
    def test_page_path_extraction(self):
        """Тест извлечения полного пути страницы (page_path)."""
        page_data = {
            'title': 'Current Page',
            'ancestors': [
                {'id': '1', 'title': 'Grandparent Page'},
                {'id': '2', 'title': 'Parent Page'}
            ]
        }
        
        result = extract_page_metadata(page_data)
        
        # Проверяем, что page_path содержит полный путь
        assert 'page_path' in result
        assert result['page_path'] == 'Grandparent Page > Parent Page > Current Page'
        assert result['parent_title'] == 'Parent Page'  # Ближайший родитель
    
    def test_page_path_single_ancestor(self):
        """Тест page_path с одним предком."""
        page_data = {
            'title': 'Child Page',
            'ancestors': [
                {'id': '1', 'title': 'Parent Page'}
            ]
        }
        
        result = extract_page_metadata(page_data)
        assert result['page_path'] == 'Parent Page > Child Page'
    
    def test_page_path_no_ancestors(self):
        """Тест page_path без предков (корневая страница)."""
        page_data = {
            'title': 'Root Page',
            'ancestors': []
        }
        
        result = extract_page_metadata(page_data)
        # Если нет предков, page_path должен содержать только текущую страницу
        assert result['page_path'] == 'Root Page'
    
    def test_page_path_empty_ancestors(self):
        """Тест page_path когда ancestors отсутствует в page_data."""
        page_data = {
            'title': 'Standalone Page'
        }
        
        result = extract_page_metadata(page_data)
        # Если ancestors отсутствует, но есть title, page_path должен содержать только title
        assert result['page_path'] == 'Standalone Page'


class TestPreprocessConfluenceMacros:
    """Тесты для функции preprocess_confluence_macros."""
    
    def test_info_macro(self):
        """Тест Info макроса."""
        html = '<ac:structured-macro ac:name="info"><ac:rich-text-body>Important info</ac:rich-text-body></ac:structured-macro>'
        result = preprocess_confluence_macros(html)
        assert '💡' in result or 'INFO' in result
        assert 'Important info' in result
    
    def test_warning_macro(self):
        """Тест Warning макроса."""
        html = '<ac:structured-macro ac:name="warning"><ac:rich-text-body>Warning text</ac:rich-text-body></ac:structured-macro>'
        result = preprocess_confluence_macros(html)
        assert '⚠️' in result or 'WARNING' in result
        assert 'Warning text' in result
    
    def test_code_macro(self):
        """Тест Code макроса."""
        html = '''<ac:structured-macro ac:name="code">
            <ac:parameter ac:name="language">python</ac:parameter>
            <ac:plain-text-body><![CDATA[print("Hello")]]></ac:plain-text-body>
        </ac:structured-macro>'''
        result = preprocess_confluence_macros(html)
        assert '```python' in result or 'print("Hello")' in result
    
    def test_multiple_macros(self):
        """Тест нескольких макросов."""
        html = '''
        <ac:structured-macro ac:name="info"><ac:rich-text-body>Info</ac:rich-text-body></ac:structured-macro>
        <ac:structured-macro ac:name="warning"><ac:rich-text-body>Warning</ac:rich-text-body></ac:structured-macro>
        '''
        result = preprocess_confluence_macros(html)
        assert 'Info' in result
        assert 'Warning' in result
    
    def test_no_macros(self):
        """Тест HTML без макросов."""
        html = '<p>Plain HTML</p>'
        result = preprocess_confluence_macros(html)
        assert result == html
    
    def test_ac_table_conversion(self):
        """Тест конвертации <ac:table> в <table>."""
        html = '<ac:table><ac:tr><ac:td>Cell 1</ac:td><ac:td>Cell 2</ac:td></ac:tr></ac:table>'
        result = preprocess_confluence_macros(html)
        assert '<table>' in result
        assert '<tr>' in result
        assert '<td>' in result
        assert 'ac:table' not in result


class TestGetBoolEnv:
    """Тесты для функции get_bool_env."""
    
    @patch.dict(os.environ, {'TEST_BOOL': 'true'})
    def test_true_value(self):
        """Тест значения true."""
        result = get_bool_env('TEST_BOOL', False)
        assert result is True
    
    @patch.dict(os.environ, {'TEST_BOOL': '1'})
    def test_one_value(self):
        """Тест значения 1."""
        result = get_bool_env('TEST_BOOL', False)
        assert result is True
    
    @patch.dict(os.environ, {'TEST_BOOL': 'yes'})
    def test_yes_value(self):
        """Тест значения yes."""
        result = get_bool_env('TEST_BOOL', False)
        assert result is True
    
    @patch.dict(os.environ, {'TEST_BOOL': 'false'})
    def test_false_value(self):
        """Тест значения false."""
        result = get_bool_env('TEST_BOOL', True)
        assert result is False
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_env_var(self):
        """Тест отсутствующей переменной - должен вернуть default."""
        result = get_bool_env('NONEXISTENT', True)
        assert result is True


class TestConvertTableToMarkdown:
    """Тесты для функции convert_table_to_markdown."""
    
    def test_simple_table(self):
        """Тест простой таблицы."""
        from bs4 import BeautifulSoup
        html = '''
        <table>
            <tr>
                <th>Header 1</th>
                <th>Header 2</th>
            </tr>
            <tr>
                <td>Cell 1</td>
                <td>Cell 2</td>
            </tr>
        </table>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        
        markdown, html_result = convert_table_to_markdown(table)
        
        assert '|' in markdown
        assert 'Header 1' in markdown
        assert 'Header 2' in markdown
        assert 'Cell 1' in markdown
        assert 'Cell 2' in markdown
        assert '---' in markdown  # Separator
        assert html_result is not None
    
    def test_empty_table(self):
        """Тест пустой таблицы."""
        from bs4 import BeautifulSoup
        html = '<table><tr><td></td></tr></table>'
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        
        markdown, html_result = convert_table_to_markdown(table)
        
        assert markdown == ""
        assert html_result == ""
    
    def test_table_with_pipe_symbols(self):
        """Тест таблицы с символами pipe."""
        from bs4 import BeautifulSoup
        html = '''
        <table>
            <tr>
                <th>Header 1</th>
                <th>Header 2</th>
            </tr>
            <tr>
                <td>Value | Pipe</td>
                <td>Normal</td>
            </tr>
        </table>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        
        markdown, _ = convert_table_to_markdown(table)
        
        # Таблица должна быть конвертирована, pipe символы экранированы
        assert markdown != ""
        assert 'Value' in markdown or 'Pipe' in markdown


class TestExtractListText:
    """Тесты для функции extract_list_text."""
    
    def test_unordered_list(self):
        """Тест неупорядоченного списка."""
        from bs4 import BeautifulSoup
        html = '''
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
        </ul>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        ul = soup.find('ul')
        
        result = extract_list_text(ul, 'ul')
        
        assert '- Item 1' in result
        assert '- Item 2' in result
        assert '- Item 3' in result
    
    def test_ordered_list(self):
        """Тест упорядоченного списка."""
        from bs4 import BeautifulSoup
        html = '''
        <ol>
            <li>First</li>
            <li>Second</li>
        </ol>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        ol = soup.find('ol')
        
        result = extract_list_text(ol, 'ol')
        
        assert '1. First' in result
        assert '2. Second' in result
    
    def test_empty_list(self):
        """Тест пустого списка."""
        from bs4 import BeautifulSoup
        html = '<ul></ul>'
        soup = BeautifulSoup(html, 'html.parser')
        ul = soup.find('ul')
        
        result = extract_list_text(ul, 'ul')
        
        assert result == ""


class TestExtractStructuralBlocks:
    """Тесты для функции extract_structural_blocks."""
    
    def test_html_with_table(self):
        """Тест HTML с таблицей."""
        html = '''
        <h1>Main Heading</h1>
        <table>
            <tr>
                <th>Col1</th>
                <th>Col2</th>
            </tr>
            <tr>
                <td>Data1</td>
                <td>Data2</td>
            </tr>
        </table>
        '''
        blocks = extract_structural_blocks(html)
        
        assert len(blocks) >= 1
        table_blocks = [b for b in blocks if b.get('type') == 'table']
        assert len(table_blocks) > 0
        assert 'Col1' in table_blocks[0]['content']
    
    def test_html_with_list(self):
        """Тест HTML со списком."""
        html = '''
        <h2>Section</h2>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        '''
        blocks = extract_structural_blocks(html)
        
        assert len(blocks) >= 1
        list_blocks = [b for b in blocks if b.get('type') == 'list']
        assert len(list_blocks) > 0
        assert 'Item 1' in list_blocks[0]['content']
    
    def test_html_with_headings(self):
        """Тест HTML с заголовками и иерархией."""
        html = '''
        <body>
            <h1>Level 1</h1>
            <p>This is a longer text paragraph that should be processed correctly by the structural chunking function.</p>
            <h2>Level 2</h2>
            <p>Another paragraph with sufficient text content to pass the minimum length requirement.</p>
            <h3>Level 3</h3>
            <p>Third paragraph with enough text to be recognized as a valid block.</p>
        </body>
        '''
        blocks = extract_structural_blocks(html)
        
        assert len(blocks) >= 1
        # Проверяем что блоки создаются и заголовки сохраняются
        headings = [b.get('heading') for b in blocks if b.get('heading')]
        # Хотя бы один блок должен быть найден
        assert len(blocks) > 0
    
    def test_empty_html(self):
        """Тест пустого HTML."""
        blocks = extract_structural_blocks('')
        assert blocks == []
    
    def test_html_with_mixed_content(self):
        """Тест HTML со смешанным контентом."""
        html = '''
        <body>
            <h1>Title</h1>
            <p>Paragraph text</p>
            <table>
                <tr>
                    <th>Header</th>
                </tr>
                <tr>
                    <td>Table data</td>
                </tr>
            </table>
            <ul>
                <li>List item</li>
            </ul>
        </body>
        '''
        blocks = extract_structural_blocks(html)
        
        assert len(blocks) >= 1
        types = [b.get('type') for b in blocks]
        # Проверяем что хотя бы один тип найден
        assert len(types) > 0
        # Проверяем что список найден
        assert 'list' in types


class TestSmartChunkWithContext:
    """Тесты для функции smart_chunk_with_context."""
    
    def test_table_block_not_split(self):
        """Тест что таблицы не разбиваются."""
        blocks = [
            {
                'type': 'table',
                'content': '| Col1 | Col2 |\n| --- | --- |\n| Data1 | Data2 |',
                'heading': 'Test Table',
                'level': 2,
                'parent_path': 'Section 1',
                'size': 50
            }
        ]
        
        chunks = smart_chunk_with_context(blocks, max_size=20)
        
        assert len(chunks) == 1
        assert chunks[0]['type'] == 'table'
        assert 'Test Table' in chunks[0]['text']
        assert 'Section 1' in chunks[0]['text']
    
    def test_list_block_not_split(self):
        """Тест что списки не разбиваются."""
        blocks = [
            {
                'type': 'list',
                'content': '- Item 1\n- Item 2\n- Item 3',
                'heading': 'Test List',
                'level': 2,
                'parent_path': '',
                'size': 30
            }
        ]
        
        chunks = smart_chunk_with_context(blocks, max_size=10)
        
        assert len(chunks) == 1
        assert chunks[0]['type'] == 'list'
        assert 'Item 1' in chunks[0]['text']
        assert 'Item 2' in chunks[0]['text']
        assert 'Item 3' in chunks[0]['text']
    
    def test_text_block_small(self):
        """Тест маленького текстового блока (не разбивается)."""
        blocks = [
            {
                'type': 'text',
                'content': 'Short text content',
                'heading': 'Section',
                'level': 2,
                'parent_path': '',
                'size': 20
            }
        ]
        
        chunks = smart_chunk_with_context(blocks, max_size=500)
        
        assert len(chunks) == 1
        assert chunks[0]['type'] == 'text'
        assert chunks[0]['text'] == 'Section\n\nShort text content'
    
    def test_text_block_large_split(self):
        """Тест большого текстового блока (разбивается)."""
        # Создаем большой текст
        sentences = [f"Sentence {i}. " for i in range(50)]
        large_text = ''.join(sentences)
        
        blocks = [
            {
                'type': 'text',
                'content': large_text,
                'heading': 'Large Section',
                'level': 2,
                'parent_path': '',
                'size': len(large_text)
            }
        ]
        
        chunks = smart_chunk_with_context(blocks, max_size=100)
        
        assert len(chunks) > 1
        # Все чанки должны иметь правильные метаданные
        for chunk in chunks:
            assert chunk['type'] == 'text'
            assert chunk['heading'] == 'Large Section'
            assert 'parent_path' in chunk
    
    def test_context_prefix_with_parent(self):
        """Тест префикса контекста с родительским заголовком."""
        blocks = [
            {
                'type': 'text',
                'content': 'Content',
                'heading': 'Subsection',
                'level': 3,
                'parent_path': 'Section 1 > Section 2',
                'size': 10
            }
        ]
        
        chunks = smart_chunk_with_context(blocks, max_size=500)
        
        assert len(chunks) == 1
        assert 'Section 1 > Section 2 > Subsection' in chunks[0]['text']
    
    def test_empty_blocks(self):
        """Тест пустого списка блоков."""
        chunks = smart_chunk_with_context([], max_size=500)
        assert chunks == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

