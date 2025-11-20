# Refactoring Plan - High Complexity Functions

## Текущее состояние

**Файл:** `rag_server/mcp_rag_secure.py`
**Размер:** 2637 строк, 108KB
**Проблем:** 500 (430 trailing spaces + 49 длинных строк + 17 сложных функций + 4 минорных)

## Priority 1: Критические функции (Complexity > 40)

### 1. `get_adaptive_rerank_limit` (Line 2008, Complexity: 86) 🔴

**Текущая проблема:** Монолитная функция с 86 ветвлениями
**Целевая сложность:** < 15

**Стратегия декомпозиции:**
```python
def get_adaptive_rerank_limit(query: str, limit: int) -> int:
    base_limit = _calculate_base_limit(query, limit)
    context_boost = _apply_query_context_boost(query)
    quality_factor = _apply_result_quality_factors()
    return min(base_limit + context_boost + quality_factor, MAX_RERANK_LIMIT)

def _calculate_base_limit(query: str, limit: int) -> int:
    """Базовый лимит на основе длины запроса"""
    query_words = len(query.split())
    if query_words <= 2:
        return limit * 5
    elif query_words <= 4:
        return limit * 3
    return limit * 2

def _apply_query_context_boost(query: str) -> int:
    """Буст на основе контекста запроса"""
    # Логика буста
    pass

def _apply_result_quality_factors() -> int:
    """Факторы качества результатов"""
    # Логика факторов
    pass
```

**Timeline:** Week 1 (3-4 дня)

---

### 2. `confluence_semantic_search` (Line 1612, Complexity: 84) 🔴

**Текущая проблема:** Монолитная функция с 84 ветвлениями
**Целевая сложность:** < 15

**Стратегия: Pipeline Pattern**
```python
def confluence_semantic_search(query: str, limit: int = 5, space: str = "") -> str:
    # Валидация
    params = _validate_search_params(query, space, limit)
    
    # Получение embedding
    embedding = _get_query_embedding(params['query'])
    
    # Поиск
    results = _execute_vector_search(embedding, params)
    
    # Reranking (опционально)
    if USE_RERANKING:
        results = _apply_reranking(results, params)
    
    # Форматирование
    return _format_search_results(results, params)

def _validate_search_params(query: str, space: str, limit: int) -> dict:
    """Валидация и нормализация параметров"""
    if not query or len(query) < 2:
        raise ValueError("Query too short")
    return {'query': query.strip(), 'space': space, 'limit': limit}

def _get_query_embedding(query: str) -> list:
    """Получение embedding для запроса"""
    # Используем SearchPipeline
    pass

def _execute_vector_search(embedding: list, params: dict) -> list:
    """Выполнение векторного поиска"""
    # Логика поиска
    pass

def _apply_reranking(results: list, params: dict) -> list:
    """Применение reranking к результатам"""
    # Логика reranking
    pass

def _format_search_results(results: list, params: dict) -> str:
    """Форматирование результатов"""
    # Используем format_search_results
    pass
```

**Timeline:** Week 1 (4-5 дней)

---

### 3. `structural_metadata_search` (Line 821, Complexity: 40) 🔴

**Текущая проблема:** Смешанная логика поиска
**Целевая сложность:** < 15

**Стратегия: Strategy Pattern**
```python
def structural_metadata_search(collection, structure, limit=10) -> list:
    if not structure['is_structural_query']:
        return []
    
    # Выбор стратегии поиска
    if _is_hierarchical_query(structure):
        return _hierarchical_search(collection, structure, limit)
    elif _has_metadata_filters(structure):
        return _metadata_filter_search(collection, structure, limit)
    else:
        return _combined_search(collection, structure, limit)

def _is_hierarchical_query(structure: dict) -> bool:
    """Проверка на иерархический запрос"""
    return len(structure.get('parts', [])) > 1

def _hierarchical_search(collection, structure, limit) -> list:
    """Поиск по иерархии страниц"""
    # Логика иерархического поиска
    pass

def _metadata_filter_search(collection, structure, limit) -> list:
    """Поиск по метаданным"""
    # Логика поиска по метаданным
    pass

def _combined_search(collection, structure, limit) -> list:
    """Комбинированный поиск"""
    results_hierarchical = _hierarchical_search(collection, structure, limit)
    results_metadata = _metadata_filter_search(collection, structure, limit)
    return _merge_results(results_hierarchical, results_metadata, limit)
```

**Timeline:** Week 2 (3 дня)

---

## Priority 2: Высокая сложность (25-40)

### 4. `expand_query` (Line 146, Complexity: 37)

**Стратегия: Extract Method**
```python
def expand_query(query: str, space: str = "") -> list[str]:
    queries = [query]
    
    # Semantic variants
    queries.extend(_get_semantic_variants(query))
    
    # Synonym variants
    queries.extend(_get_synonym_variants(query))
    
    # Space context variants
    if space:
        queries.extend(_get_space_context_variants(query, space))
    
    return _deduplicate_and_limit(queries, max_variants=5)
```

**Timeline:** Week 2 (2 дня)

---

### 5. `parse_query_structure` (Line 737, Complexity: 29)

**Стратегия: Pattern Matching System**
```python
def parse_query_structure(query: str) -> dict:
    patterns = _get_structural_patterns()
    
    for pattern_type, pattern in patterns:
        if match := _try_pattern(query, pattern):
            return _build_structure(query, match, pattern_type)
    
    return _build_default_structure(query)
```

**Timeline:** Week 3 (2 дня)

---

## Метрики успеха

**До рефакторинга:**
- Средняя сложность топ-5 функций: **55.2**
- Максимальная сложность: **86**
- Функций с complexity > 15: **17**

**После рефакторинга (цель):**
- Средняя сложность топ-5 функций: **< 12**
- Максимальная сложность: **< 15**
- Функций с complexity > 15: **0**

---

## Общий timeline

**Week 1 (5 дней):**
- Day 1-2: `get_adaptive_rerank_limit` (86 → <15)
- Day 3-5: `confluence_semantic_search` (84 → <15)

**Week 2 (5 дней):**
- Day 1-3: `structural_metadata_search` (40 → <15)
- Day 4-5: `expand_query` (37 → <15)

**Week 3 (2 дня):**
- Day 1-2: `parse_query_structure` (29 → <15)

**Итого:** 12 рабочих дней (~2.5 недели)

---

## Следующие шаги

1. ✅ Создать `.flake8` и `pyproject.toml` конфиги
2. ✅ Создать этот план рефакторинга
3. ⏳ Исправить 434 trailing spaces (вручную через Find & Replace)
4. ⏳ Удалить импорт `Tuple`
5. ⏳ Начать рефакторинг с `get_adaptive_rerank_limit`
