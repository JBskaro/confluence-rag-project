# Отчет о тестировании

## Дата: $(Get-Date -Format "yyyy-MM-dd HH:mm")

## Новые тесты для новых модулей

### ✅ test_keyword_extraction.py
- **Статус**: Все тесты пройдены (9/9)
- **Покрытие**: 
  - `extract_keywords()` - базовое извлечение, русский язык, фильтрация по длине
  - `extract_technical_terms()` - английские и русские технические термины
  - `normalize_query()` - нормализация запросов
  - Константы `STOPWORDS` и `TECHNICAL_TERMS`

### ✅ test_intent_config.py
- **Статус**: Все тесты пройдены (10/10)
- **Покрытие**:
  - `QueryIntent` enum - все типы запросов
  - `get_intent_config()` - конфигурация для всех типов (navigational, howto, factual, exploratory)
  - `get_adaptive_rerank_threshold()` - адаптивные пороги для bge-reranker и cross-encoder
  - `get_adaptive_context_window()` - адаптивные размеры окна контекста
  - Fallback для неизвестных типов

## Существующие тесты

### ✅ test_simple.py
- **Статус**: Все тесты пройдены (11/11)
- **Исправлено**: Обновлен тест для проверки `sync_confluence.py` вместо `sync_confluence_optimized_final.py`
- **Добавлено**: Проверка новых файлов `utils/keyword_extraction.py` и `utils/intent_config.py`

## Интеграционные тесты

### test_queries.sh
- **Требует**: Работающий контейнер confluence-rag на порту 8012
- **Тесты**:
  1. Тест 1: Учет номенклатуры - проверка поиска страницы 18153754
  2. Тест 2: Технологический стек - проверка поиска страницы 18153591

## Итоговая статистика

| Категория | Пройдено | Всего | Процент |
|-----------|----------|-------|---------|
| Новые unit тесты | 19 | 19 | 100% |
| Существующие unit тесты | 11 | 11 | 100% |
| **ИТОГО** | **30** | **30** | **100%** |

## Запуск тестов

### Unit тесты (без контейнера)
```bash
# Windows PowerShell
pwsh -File tests/run_all_unit_tests.ps1

# Linux/Mac
bash tests/run_all_unit_tests.sh

# Или отдельно
python tests/test_keyword_extraction.py
python tests/test_intent_config.py
python -m pytest tests/test_simple.py -v
```

### Интеграционные тесты (требуют контейнер)
```bash
# Убедитесь что контейнер запущен
docker-compose ps

# Запустите тесты
bash tests/test_queries.sh
```

## Примечания

1. Все новые модули (`keyword_extraction`, `intent_config`) полностью покрыты тестами
2. Тесты проверяют как базовую функциональность, так и edge cases
3. Адаптивные пороги для разных типов запросов протестированы для обеих моделей reranker
4. Все тесты совместимы с Windows (исправлены проблемы с кодировкой)

