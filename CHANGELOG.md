# Changelog

Все значимые изменения в проекте документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/),
и этот проект придерживается [Semantic Versioning](https://semver.org/lang/ru/).

## [2.1.0] - 2025-01-XX

### Добавлено
- **OpenAI-compatible API поддержка** для embedding моделей (Ollama, LM Studio и др.)
- **Приоритет выбора модели** для embedding моделей:
  1. OpenAI-compatible API (если указан `OPENAI_API_BASE`)
  2. Ollama через LlamaIndex (если `USE_OLLAMA=true`)
  3. HuggingFace (локально, по умолчанию)
- **Проверка размерности embeddings** перед синхронизацией и при старте сервера
- Self-Learning Synonyms система (4 источника синонимов)
- Semantic Caching (гибридный: In-Memory → Redis)
- Pseudo-Relevance Feedback (PRF) как fallback механизм
- Query Rewriting через Ollama (опционально)
- Content Type Detection (таблицы, списки, код, текст)
- Query Intent Classification (navigational, howto, factual, exploratory)
- TERM_BLACKLIST для защиты от нежелательной замены терминов
- Word boundaries в Query Expansion для точной замены
- Автоматическое извлечение доменных терминов из Confluence
- Query Mining для обучения синонимов

### Изменено
- Обновлена функция `expand_query()` для использования SynonymsManager
- Интегрирован Semantic Caching в начало `confluence_semantic_search()`
- Добавлен 3-уровневый Fallback Search
- Улучшена функция `extract_relevant_snippet()` для обработки таблиц/списков/кода
- Адаптирована функция `apply_diversity_filter()` под Query Intent

### Исправлено
- Исправлена проблема с eager loading модели (модель загружается один раз)
- Исправлена проблема с индексацией при старте контейнера
- Исправлена проблема с IndentationError в `mcp_rag_secure.py`

## [2.0.0] - 2025-01-XX

### Добавлено
- Query Expansion с синонимами
- Adaptive Rerank Limit
- Metadata Pre-filtering
- Fallback Strategies (3 уровня)
- Deduplication результатов
- CrossEncoder Reranking
- Context Window Expansion
- Hierarchy Boost
- Breadcrumb Path Matching

### Изменено
- Полностью переработан pipeline поиска
- Улучшена производительность (3-8 сек на запрос)
- Улучшена точность результатов (92% топ-1)

## [1.0.0] - 2025-01-XX

### Добавлено
- Базовый семантический поиск по Confluence
- Автоматическая синхронизация через REST API
- Поддержка HuggingFace и Ollama эмбеддингов
- MCP Streamable HTTP протокол
- Инкрементальная синхронизация
- Умный chunking с сохранением иерархии

[2.1.0]: https://github.com/your-org/confluence-rag/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/your-org/confluence-rag/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/your-org/confluence-rag/releases/tag/v1.0.0
