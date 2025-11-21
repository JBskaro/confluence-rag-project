# 📦 Установка Confluence RAG Server

## 🎯 Рекомендуемая установка

### Вариант 1: Минимальная установка (Core только)

Быстрая установка без ML моделей (для тестирования, CI/CD):

```bash
pip install -r requirements-core.txt
```

**Размер:** ~200MB  
**Время:** ~2-3 минуты  
**Поддерживает:**
- ✅ Hybrid search (BM25 + Vector)
- ✅ Qdrant vector database
- ✅ Confluence sync
- ✅ Observability (metrics, tracing)
- ❌ Embeddings (нужен внешний API: Ollama/OpenRouter)

---

### Вариант 2: Full ML (рекомендуется для production)

Полная установка с локальными ML моделями:

```bash
pip install -r requirements-core.txt -r requirements-ml.txt
```

**Размер:** ~3-5GB (зависит от torch)  
**Время:** ~10-20 минут  
**Поддерживает:**
- ✅ Все из Core
- ✅ Локальные embeddings (SentenceTransformers)
- ✅ Ragas evaluation
- ✅ OpenAI/OpenRouter API (опционально)

---

### Вариант 3: Development

Установка для разработки (с тестами и линтерами):

```bash
pip install -r requirements-core.txt -r requirements-dev.txt
```

**Размер:** ~300MB  
**Время:** ~3-5 минут  
**Поддерживает:**
- ✅ Все из Core
- ✅ pytest, mypy, flake8, black, isort

---

## 🚀 Установка через pyproject.toml (рекомендуется)

### Editable mode (для разработки):

```bash
# Core only
pip install -e .

# С ML моделями
pip install -e .[ml]

# С OpenAI
pip install -e .[openai]

# С RAG evaluation
pip install -e .[eval]

# Для разработки
pip install -e .[dev]

# Все вместе
pip install -e .[all]
```

---

## 📊 Сравнение вариантов

| Вариант | Размер | Время установки | Use case |
|---------|--------|-----------------|----------|
| **Core** | ~200MB | 2-3 мин | Тестирование, CI/CD, с внешним API |
| **ML** | ~3-5GB | 10-20 мин | Production с локальными моделями |
| **Dev** | ~300MB | 3-5 мин | Разработка, код ревью |
| **All** | ~5GB | 15-25 мин | Полная среда разработки |

---

## 🎯 Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/yourusername/confluence-rag-project.git
cd confluence-rag-project
```

### 2. Создать виртуальное окружение

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate  # Windows
```

### 3. Установить зависимости

**Option A: Минимальная (быстро, без ML)**
```bash
pip install -r requirements-core.txt
```

**Option B: Full (production)**
```bash
pip install -r requirements.txt
```

**Option C: С extras (рекомендуется)**
```bash
pip install -e .[ml,openai,eval]
```

### 4. Настроить .env

```bash
cp ENV_TEMPLATE .env
# Отредактировать .env (см. QUICKSTART.md)
```

### 5. Запустить сервер

```bash
# С локальными embeddings (нужен requirements-ml.txt)
python -m rag_server.mcp_rag_secure

# С Ollama
EMBEDDING_SOURCE=ollama python -m rag_server.mcp_rag_secure

# С OpenRouter
EMBEDDING_SOURCE=openai python -m rag_server.mcp_rag_secure
```

---

## 🐳 Docker (альтернатива)

### Multi-stage build для оптимизации

```dockerfile
# Stage 1: Core
FROM python:3.11-slim as core
WORKDIR /app
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# Stage 2: ML (опционально)
FROM core as ml
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Stage 3: App
FROM ml as app
COPY rag_server ./rag_server
COPY .env .
CMD ["python", "-m", "rag_server.mcp_rag_secure"]
```

**Билд:**
```bash
# Core only (200MB)
docker build --target core -t rag-server:core .

# Full ML (3GB)
docker build --target app -t rag-server:latest .
```

---

## 🔧 Проверка установки

```bash
# 1. Проверить импорты
python -c "from rag_server.config import settings; print('✅ Config OK')"
python -c "from qdrant_client import QdrantClient; print('✅ Qdrant OK')"
python -c "import numpy; print('✅ NumPy OK')"

# 2. Проверить embeddings (если установлен requirements-ml.txt)
python -c "from sentence_transformers import SentenceTransformer; print('✅ SentenceTransformers OK')"

# 3. Запустить health check
python -c "from rag_server.mcp_rag_secure import confluence_health; print(confluence_health())"
```

---

## ⚠️ Troubleshooting

### Проблема: "ModuleNotFoundError: No module named 'qdrant_client'"

**Решение:**
```bash
pip install qdrant-client>=1.11.0
```

### Проблема: "torch слишком долго устанавливается"

**Решение:** Установить CPU-only версию:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Проблема: "numpy compatibility error"

**Решение:** Установить совместимую версию:
```bash
pip install "numpy>=1.24.0,<2.0.0"
```

### Проблема: "IDE не видит импорты"

**Решение:**
1. Убедитесь, что виртуальное окружение активировано
2. В VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → выбрать `.venv`
3. Перезапустить IDE

---

## 📝 Дополнительные ресурсы

- [QUICKSTART.md](../../QUICKSTART.md) - Быстрый старт
- [ENV_TEMPLATE](../../ENV_TEMPLATE) - Шаблон конфигурации
- [PROJECT_STRUCTURE.md](../../PROJECT_STRUCTURE.md) - Структура проекта

---

## 🎯 Рекомендации по выбору

| Сценарий | Установка | Embedding Source |
|----------|-----------|------------------|
| **CI/CD тесты** | `requirements-core.txt` | mock/test fixtures |
| **Dev машина** | `pip install -e .[dev]` | Ollama (локально) |
| **Production (мощный сервер)** | `requirements.txt` | HuggingFace (локально) |
| **Production (слабый сервер)** | `requirements-core.txt` | OpenRouter API |
| **Kubernetes** | Docker multi-stage | OpenRouter API или Ollama sidecar |

---

**Готово!** 🚀 Теперь у вас есть гибкая система установки зависимостей.

