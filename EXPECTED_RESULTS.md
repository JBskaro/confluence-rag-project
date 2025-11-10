# Ожидаемые результаты тестирования

## Тест 1: Учет номенклатуры

**Запрос:**
```
Провожу обследование компании по блоку Склад, а точнее Учет номенклатуры. Подготовь список вопросов.
```

**Ожидаемый результат:**
- ✅ **Страница:** `18153754` (должна быть в ответе)
- ✅ **Минимум 30+ вопросов** из разделов:
  1. Классификация номенклатуры
  2. Учет движения и остатков
  3. Логистические и финансовые параметры
  4. Классификация и хранение товаров
  5. Дополнительные вопросы

**Ключевые слова, которые должны быть:**
- номенклатур
- классификац
- серии
- характеристик
- габарит
- штрихкод
- Классификация номенклатуры
- Учет движения и остатков
- Логистические и финансовые параметры

---

## Тест 2: Технологический стек (с опечатками)

**Запрос:**
```
Ккой тхнологиеский стек исользуется в проект рау ии.
```

**Ожидаемый результат:**
- ✅ **Страница:** `18153591` (должна быть в ответе)
- ✅ **Query Rewriting должен исправить опечатки:**
  - "Ккой" → "Какой"
  - "тхнологиеский" → "технологический"
  - "исользуется" → "используется"
  - "рау ии" → "РАУ ИИ"
- ✅ **Должен содержать информацию о стеке:**
  - Ollama
  - OpenRouter
  - LiteLLM
  - MCP
  - Syntaxcheck, Docsearch, Codesearch, Templatesearch
  - Open WebUI

**Ключевые слова, которые должны быть:**
- Ollama
- OpenRouter
- LiteLLM
- MCP
- технологи
- стек
- Syntaxcheck
- Docsearch
- Codesearch
- Templatesearch
- Open WebUI

---

## Проблемы, обнаруженные в логах

### 1. ❌ Scores после reranking слишком низкие

**Проблема:**
- Top score: `0.001` и `0.018` (очень низкие!)
- Пороги: `RERANK_THRESHOLD_TECHNICAL=0.01`, `RERANK_THRESHOLD_GENERAL=0.005`
- Результаты отфильтровываются, т.к. scores < порогов

**Решение:**
1. Снизить пороги до `0.001` и `0.0005` (временно для тестирования)
2. Или проверить, почему модель `DiTy/cross-encoder-russian-msmarco` возвращает такие низкие scores
3. Возможно, нужно нормализовать scores или использовать другую модель

### 2. ❌ Query Rewriting не работает

**Проблема:**
```
⚠️ OpenRouter rewriting failed: Error code: 403 - 
{'error': {'message': 'Provider returned error', 'code': 403, 
'metadata': {'raw': '{"error":{"code":"unsupported_country_region_territory",
"message":"Country, region, or territory not supported"}}'}}}
```

**Решение:**
1. Использовать Ollama как fallback (должен работать автоматически)
2. Или настроить прокси для OpenRouter
3. Или использовать другой сервис для Query Rewriting

### 3. ⚠️ Неправильный space

**Проблема:**
- Поиск идет в space `"Surveys"`
- Но нужные страницы находятся в space `"RAUII"`

**Решение:**
- Указать правильный space в запросе: `space="RAUII"`

### 4. ❌ Индекс может быть пустым

**Проблема:**
- Ранее видели "Documents: 0"
- Синхронизация показала "0 updated, 69 skipped"

**Решение:**
1. Проверить количество документов: `docker-compose exec confluence-rag python -c "from server import get_vector_store; vs = get_vector_store(); print(f'Documents: {vs._collection.count()}')"`
2. Если 0, то переиндексировать:
   - Удалить `chroma_data` и `data/sync_state.json`
   - Перезапустить контейнер

---

## Рекомендации по исправлению

### Шаг 1: Проверить индекс
```bash
docker-compose exec confluence-rag python -c "from server import get_vector_store; vs = get_vector_store(); print(f'Documents: {vs._collection.count()}')"
```

### Шаг 2: Снизить пороги reranking (временно)
В `.env`:
```bash
RERANK_THRESHOLD_TECHNICAL=0.001
RERANK_THRESHOLD_GENERAL=0.0005
```

### Шаг 3: Включить Ollama для Query Rewriting
В `.env`:
```bash
QUERY_REWRITING_SOURCE=ollama
# Или оставить пустым для автоматического fallback
```

### Шаг 4: Указать правильный space
В запросах использовать `space="RAUII"` вместо `space="Surveys"`

---

## Использование скрипта анализа

Запустите `analyze_search_results.py` для сравнения фактических результатов с ожидаемыми:

```bash
python analyze_search_results.py
```

Скрипт попросит вставить результаты из Open WebUI и покажет:
- ✅/❌ Найдена ли ожидаемая страница
- Количество найденных ключевых слов
- Оценку релевантности (0-100)
- Рекомендации по улучшению

