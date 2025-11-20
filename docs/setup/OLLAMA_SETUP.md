# Настройка Embedding Моделей

Этот документ описывает, как настроить различные источники embedding моделей: OpenRouter, Ollama и HuggingFace.

## OpenRouter (Облачный сервис) ⭐⭐

OpenRouter предоставляет доступ к различным embedding моделям через OpenAI-compatible API, включая Qwen3-Embedding-8B, 4B и другие мощные модели.

**Преимущества:**
- ✅ Не требует локальных ресурсов (GPU/RAM)
- ✅ Быстрая работа (облачные сервисы)
- ✅ Доступ к мощным моделям (8B, 4B и др.)
- ✅ Автоматическое масштабирование
- ✅ Единый API для разных моделей

**Настройка:**

1. Зарегистрируйтесь на [OpenRouter.ai](https://openrouter.ai/) и получите API ключ

2. Настройте `.env`:
```env
# OpenRouter
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-ваш-ключ-здесь
OPENAI_MODEL=qwen/qwen3-embedding-8b
```

3. Запустите сервер:
```bash
python rag_server/mcp_rag_secure.py
```

**Рекомендуемые модели OpenRouter:**
- `qwen/qwen3-embedding-8b` — мощная модель, отличное качество
- `qwen/qwen3-embedding-4b` — баланс качества и скорости
- `qwen/qwen3-embedding-1.7b` — быстрая модель

**Примечание:** Проверьте тарифы на [OpenRouter.ai/pricing](https://openrouter.ai/pricing)

**Для Docker:**

```yaml
services:
  confluence-rag:
    environment:
      - OPENAI_API_BASE=https://openrouter.ai/api/v1
      - OPENAI_API_KEY=${OPENAI_API_KEY}  # Из .env
      - OPENAI_MODEL=qwen/qwen3-embedding-8b
```

**Проверка подключения:**

```bash
docker logs confluence-rag | grep -i "openai\|embedding\|qwen"
```

Ожидаемый вывод:
```
[INFO] 🔌 Попытка подключения к OpenAI-compatible API: https://openrouter.ai/api/v1
[INFO]    Модель: qwen/qwen3-embedding-8b
[INFO] ✅ OpenAI-compatible API подключен за X.X сек
[INFO]    Модель: qwen/qwen3-embedding-8b, Размерность: XXXD
```

---

## Ollama (Локальный сервер)

### Вариант 1: OpenAI-compatible API (рекомендуется) ⭐

Ollama поддерживает OpenAI-compatible API, что позволяет использовать стандартный клиент OpenAI.

**Преимущества:**
- ✅ Более стабильное подключение
- ✅ Поддержка batch запросов (быстрее)
- ✅ Единый интерфейс для разных сервисов (Ollama, LM Studio и др.)

**Настройка:**

1. Убедитесь, что Ollama запущен:
```bash
ollama serve
```

2. Установите embedding модель:
```bash
ollama pull nomic-embed-text
# или
ollama pull all-minilm
```

3. Настройте `.env`:
```env
# OpenAI-compatible API (Ollama)
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=nomic-embed-text
```

4. Запустите сервер:
```bash
python rag_server/mcp_rag_secure.py
```

**Для Docker:**

```yaml
services:
  confluence-rag:
    environment:
      - OPENAI_API_BASE=http://ollama:11434/v1
      - OPENAI_API_KEY=ollama
      - OPENAI_MODEL=nomic-embed-text
    networks:
      - your-network
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - your-network

volumes:
  ollama_data:
```

### Вариант 2: LlamaIndex Ollama (альтернатива)

Прямое подключение через LlamaIndex OllamaEmbedding.

**Настройка:**

1. Убедитесь, что Ollama запущен и модель установлена (см. выше)

2. Настройте `.env`:
```env
USE_OLLAMA=true
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
```

3. Запустите сервер:
```bash
python rag_server/mcp_rag_secure.py
```

## Рекомендуемые модели

### Для русского языка:
- `nomic-embed-text` — универсальная модель, поддерживает русский
- `all-minilm` — компактная модель, быстрая

### Для английского:
- `nomic-embed-text` — отличное качество
- `all-minilm` — быстрая альтернатива

## Проверка подключения

После запуска сервера проверьте логи:

```bash
docker logs confluence-rag | grep -i "ollama\|openai\|embedding"
```

Ожидаемый вывод:
```
[INFO] 🔌 Попытка подключения к OpenAI-compatible API: http://localhost:11434/v1
[INFO]    Модель: nomic-embed-text
[INFO] ✅ OpenAI-compatible API подключен за 0.5 сек
[INFO]    Модель: nomic-embed-text, Размерность: 768D
```

## Troubleshooting

### "Connection refused"
- Убедитесь, что Ollama запущен: `ollama serve`
- Проверьте URL: `http://localhost:11434` (или `http://ollama:11434` в Docker)

### "Model not found"
- Установите модель: `ollama pull nomic-embed-text`
- Проверьте имя модели в `OPENAI_MODEL` или `EMBED_MODEL`

### "КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к Ollama"
- Убедитесь, что Ollama запущен: `ollama serve`
- Проверьте URL в `OPENAI_API_BASE` или `OLLAMA_URL`
- Проверьте, что модель установлена: `ollama list`
- **ВАЖНО:** Если указана модель Ollama, она должна быть доступна. Нет автоматического fallback на HuggingFace.

## Производительность

| Модель | Размерность | Скорость | Качество |
|--------|-------------|----------|----------|
| `nomic-embed-text` | 768 | Средняя | Высокое |
| `all-minilm` | 384 | Быстрая | Хорошее |
| `ai-forever/FRIDA` (HuggingFace) | 1024 | Медленная | Очень высокое |

## Примеры использования

### Локальная разработка
```bash
# Запуск Ollama
ollama serve

# В другом терминале
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_MODEL=nomic-embed-text
python rag_server/mcp_rag_secure.py
```

### Docker Compose
```yaml
services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"

  confluence-rag:
    environment:
      - OPENAI_API_BASE=http://ollama:11434/v1
      - OPENAI_MODEL=nomic-embed-text
    depends_on:
      - ollama
```

## Дополнительная информация

- [Ollama Documentation](https://ollama.ai/docs)
- [Ollama Models](https://ollama.ai/library)
- [OpenAI API Compatibility](https://github.com/ollama/ollama/blob/main/docs/openai.md)

