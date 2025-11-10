# Тестирование Confluence RAG

Этот документ описывает структуру тестов и как их запускать.

## Структура тестов

```
tests/
├── __init__.py              # Инициализация пакета тестов
├── conftest.py              # Общие fixtures и конфигурация pytest
├── test_simple.py           # Простые standalone тесты
├── test_embeddings.py       # Тесты модуля embeddings.py
├── test_dimension_validation.py  # Тесты проверки размерности
├── test_sync_functions.py   # Тесты функций синхронизации
├── test_mcp_server.py       # Тесты MCP сервера
└── test_integration.py      # Интеграционные тесты
```

## Запуск тестов

### Базовые команды

```bash
# Запустить все тесты
pytest

# Запустить с подробным выводом
pytest -v

# Запустить с coverage отчетом
pytest --cov=rag_server --cov-report=html

# Запустить только unit тесты
pytest -m unit

# Запустить только integration тесты
pytest -m integration
```

### Использование Makefile

```bash
# Все тесты
make test

# Только unit тесты
make test-unit

# Только integration тесты
make test-integration

# Тесты с coverage
make test-cov

# Быстрые тесты (без coverage)
make test-fast

# Только последние failed тесты
make test-failed

# Тесты с отладкой
make test-debug
```

## Покрытие кода

Целевое покрытие: **70%**

Проверить текущее покрытие:

```bash
pytest --cov=rag_server --cov-report=html --cov-report=term-missing
```

Отчет будет сохранен в `htmlcov/index.html`.

## Категории тестов

### Unit тесты (`@pytest.mark.unit`)

Тестируют отдельные функции и модули без внешних зависимостей.

**Примеры:**
- `test_embeddings.py` - тесты загрузки моделей
- `test_dimension_validation.py` - тесты проверки размерности
- `test_simple.py` - базовые функции

### Integration тесты (`@pytest.mark.integration`)

Тестируют взаимодействие между компонентами.

**Примеры:**
- `test_integration.py` - полный цикл синхронизации
- `test_mcp_server.py` - работа MCP сервера

### Медленные тесты (`@pytest.mark.slow`)

Тесты, которые выполняются долго (например, с реальными API).

Запуск без медленных тестов:

```bash
pytest -m "not slow"
```

### Тесты требующие сеть (`@pytest.mark.requires_network`)

Тесты, которые требуют доступ к сети.

Запуск без сетевых тестов:

```bash
pytest -m "not requires_network"
```

## Fixtures

Общие fixtures определены в `conftest.py`:

- `mock_confluence` - Mock Confluence API client
- `mock_chromadb_collection` - Mock ChromaDB collection
- `mock_chroma_client` - Mock ChromaDB client
- `sample_confluence_page` - Пример данных страницы Confluence
- `sample_html_with_macros` - Пример HTML с макросами
- `sample_markdown_text` - Пример markdown текста
- `mock_llama_index_retriever` - Mock LlamaIndex retriever
- `mock_vector_index` - Mock VectorStoreIndex
- `temp_state_file` - Временный файл состояния
- `sample_sync_state` - Пример состояния синхронизации

## Написание новых тестов

### Пример unit теста

```python
import pytest
from unittest.mock import Mock, patch

def test_my_function():
    """Тест моей функции."""
    # Arrange
    input_data = "test"
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result == "expected"
```

### Пример теста с mock

```python
@pytest.mark.unit
def test_function_with_external_dependency(mock_confluence):
    """Тест функции с внешней зависимостью."""
    with patch('module.external_function') as mock_external:
        mock_external.return_value = "mocked"
        
        result = my_function()
        
        assert result == "expected"
        mock_external.assert_called_once()
```

### Пример теста с параметризацией

```python
@pytest.mark.parametrize("input,expected", [
    ("test1", "result1"),
    ("test2", "result2"),
])
def test_parametrized(input, expected):
    """Параметризованный тест."""
    assert my_function(input) == expected
```

## Проверка качества кода

Перед коммитом рекомендуется запустить:

```bash
# Полная проверка качества
make quality

# Или по отдельности:
make format        # Форматирование кода
make sort-imports  # Сортировка импортов
make lint          # Все проверки
make test-cov      # Тесты с coverage
```

## Troubleshooting

### Тесты падают с ImportError

Убедитесь, что все зависимости установлены:

```bash
pip install -r requirements.txt
```

### Тесты падают из-за переменных окружения

Используйте `monkeypatch` fixture для управления переменными окружения:

```python
def test_with_env(monkeypatch):
    monkeypatch.setenv("MY_VAR", "value")
    # ваш тест
```

### Тесты требуют реальные данные

Используйте моки и фикстуры из `conftest.py` вместо реальных API.

## CI/CD

Тесты автоматически запускаются в CI/CD при каждом коммите.

Требования:
- Все тесты должны проходить
- Покрытие кода должно быть >= 70%
- Все проверки качества должны проходить (black, isort, flake8, mypy)
