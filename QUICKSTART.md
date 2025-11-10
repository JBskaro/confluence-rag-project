# Быстрый старт

Этот документ поможет вам быстро запустить Confluence RAG MCP сервер.

## Предварительные требования

- Docker и Docker Compose
- Confluence API Token
- 2GB+ RAM для эмбеддингов

## Шаг 1: Получите Confluence API Token

1. Перейдите: https://id.atlassian.com/manage-profile/security/api-tokens
2. Создайте новый токен
3. Скопируйте токен (он понадобится на следующем шаге)

## Шаг 2: Создайте файл `.env`

Создайте файл `.env` в корне проекта:

```bash
# Обязательные параметры
CONFLUENCE_URL=https://your-confluence.atlassian.net/
CONFLUENCE_TOKEN=your_api_token_here

# Опциональные параметры (можно оставить по умолчанию)
CHROMA_PATH=./chroma_data
SYNC_INTERVAL=3600
USE_OLLAMA=false
EMBED_MODEL=ai-forever/FRIDA
```

## Шаг 3: Запустите контейнер

```bash
# Соберите образ
docker build -f Dockerfile.standalone -t confluence-rag:latest .

# Запустите сервис
docker compose up -d confluence-rag

# Проверьте логи
docker logs -f confluence-rag
```

## Шаг 4: Подождите синхронизацию

Первая синхронизация может занять:
- 100 страниц: ~2-5 минут
- 500 страниц: ~10-20 минут
- 1000+ страниц: ~30-60 минут

В логах вы увидите:
```
✅ Синхронизация завершена. Документов: N
```

## Шаг 5: Подключите к Open WebUI

1. Откройте Open WebUI
2. Перейдите: **Settings → Tools → Add MCP Server**
3. Настройте:
   - **Name**: Confluence RAG
   - **URL**: `http://confluence-rag:8012/mcp` (внутри Docker сети)
4. Сохраните

## Шаг 6: Протестируйте

Задайте вопрос в Open WebUI:
```
Найди информацию о процессе деплоя в space DEVOPS
```

## Проверка статуса

```bash
# Проверка health
curl http://localhost:8012/health

# Проверка количества документов
docker exec confluence-rag python -c "from mcp_rag_secure import collection; print(f'Documents: {collection.count()}')"
```

## Проблемы?

### "Index empty"
Синхронизация ещё не завершена, подождите.

### "Connection refused"
Проверьте Docker сеть и порты.

### "SSL Error"
Установите `VERIFY_SSL=false` в `.env` для тестирования.

## Дополнительная информация

- Полная документация: [README.md](README.md)
- Техническое задание: [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md)
- Конфигурация: [ENV_CONFIGURATION.md](ENV_CONFIGURATION.md)
