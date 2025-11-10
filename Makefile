.PHONY: help build up down restart logs shell test test-unit test-integration test-cov test-fast clean backup restore health status

# Переменные
COMPOSE=docker compose
CONTAINER=confluence-rag
BACKUP_DIR=./backups

help: ## Показать эту справку
	@echo "Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Собрать Docker образ
	$(COMPOSE) build confluence-rag

up: ## Запустить сервис
	$(COMPOSE) up -d confluence-rag
	@echo "✅ Сервис запущен. Проверьте логи: make logs"

down: ## Остановить сервис
	$(COMPOSE) down

restart: ## Перезапустить сервис
	$(COMPOSE) restart confluence-rag

logs: ## Показать логи (follow mode)
	$(COMPOSE) logs -f confluence-rag

logs-tail: ## Показать последние 100 строк логов
	$(COMPOSE) logs --tail=100 confluence-rag

shell: ## Открыть shell в контейнере
	docker exec -it $(CONTAINER) bash

# ========== Тестирование ==========

test: ## Запустить все тесты
	pytest -v

test-unit: ## Запустить только unit тесты
	pytest tests/test_sync_functions.py tests/test_mcp_server.py -v

test-integration: ## Запустить только integration тесты
	pytest tests/test_integration.py -v

test-cov: ## Запустить тесты с coverage report
	pytest --cov=rag_server --cov-report=html --cov-report=term-missing

test-fast: ## Запустить тесты без coverage (быстро)
	pytest -v --tb=short

test-failed: ## Запустить только последние failed тесты
	pytest --lf -v

test-watch: ## Запустить тесты в watch mode (при изменении файлов)
	pytest-watch -v

test-debug: ## Запустить тесты с подробным выводом и остановкой на ошибках
	pytest -vv -s --pdb -x

# ========== Code Quality ==========

lint: ## Запустить все проверки качества кода
	@echo "🔍 Запуск проверок качества кода..."
	@echo ""
	@echo "1. Black (форматирование)..."
	@black --check rag_server tests || echo "⚠️  Black: требуется форматирование (запустите: make format)"
	@echo ""
	@echo "2. isort (сортировка импортов)..."
	@isort --check-only rag_server tests || echo "⚠️  isort: требуется сортировка (запустите: make sort-imports)"
	@echo ""
	@echo "3. Flake8 (стиль кода)..."
	@flake8 rag_server tests || echo "⚠️  Flake8: найдены проблемы стиля"
	@echo ""
	@echo "4. MyPy (проверка типов)..."
	@mypy rag_server || echo "⚠️  MyPy: найдены проблемы типов"
	@echo ""
	@echo "✅ Проверки завершены"

format: ## Форматировать код с помощью Black
	@echo "🎨 Форматирование кода..."
	@black rag_server tests
	@echo "✅ Форматирование завершено"

sort-imports: ## Сортировать импорты с помощью isort
	@echo "📦 Сортировка импортов..."
	@isort rag_server tests
	@echo "✅ Сортировка завершена"

type-check: ## Проверить типы с помощью MyPy
	@echo "🔍 Проверка типов..."
	@mypy rag_server
	@echo "✅ Проверка типов завершена"

style-check: ## Проверить стиль кода с помощью Flake8
	@echo "🔍 Проверка стиля кода..."
	@flake8 rag_server tests
	@echo "✅ Проверка стиля завершена"

quality: format sort-imports lint test-cov ## Полная проверка качества (форматирование + проверки + тесты)
	@echo "✅ Полная проверка качества завершена"

python-shell: ## Открыть Python shell в контейнере
	docker exec -it $(CONTAINER) python

health: ## Проверить health status
	@echo "Проверка health status..."
	@curl -s http://localhost:8012/mcp || echo "❌ Сервис недоступен"

status: ## Показать статус контейнера
	@docker ps -a --filter name=$(CONTAINER) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

stats: ## Показать использование ресурсов
	docker stats $(CONTAINER) --no-stream

# Backup и восстановление
backup: ## Создать backup ChromaDB и state
	@echo "Создание backup..."
	@mkdir -p $(BACKUP_DIR)
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf $(BACKUP_DIR)/confluence-rag-backup-$$timestamp.tar.gz \
		chroma_data/ sync_state.json 2>/dev/null || true
	@echo "✅ Backup создан: $(BACKUP_DIR)/confluence-rag-backup-$$(date +%Y%m%d_)*.tar.gz"
	@ls -lh $(BACKUP_DIR)/ | tail -n 1

restore: ## Восстановить из последнего backup (restore BACKUP=filename.tar.gz)
	@if [ -z "$(BACKUP)" ]; then \
		echo "❌ Укажите файл backup: make restore BACKUP=filename.tar.gz"; \
		exit 1; \
	fi
	@echo "Восстановление из $(BACKUP)..."
	@$(COMPOSE) down
	@tar -xzf $(BACKUP_DIR)/$(BACKUP)
	@echo "✅ Восстановление завершено"
	@$(COMPOSE) up -d

list-backups: ## Показать список backups
	@echo "Доступные backups:"
	@ls -lh $(BACKUP_DIR)/ 2>/dev/null || echo "Нет backups"

# Очистка
clean: ## Удалить все данные (ChromaDB + state)
	@echo "⚠️  ВНИМАНИЕ: Это удалит все данные!"
	@read -p "Продолжить? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(COMPOSE) down; \
		rm -rf chroma_data/ sync_state.json; \
		echo "✅ Данные удалены"; \
	else \
		echo "❌ Отменено"; \
	fi

clean-logs: ## Очистить логи контейнера
	docker logs $(CONTAINER) 2>&1 | tail -n 0

rebuild: down clean build up ## Полный пересбор (down + clean + build + up)

# Разработка
dev-logs: ## Логи в режиме разработки (с отладкой)
	LOG_LEVEL=DEBUG $(COMPOSE) up

test-search: ## Тестовый поиск
	@echo "Тестовый поиск..."
	@curl -s -X POST http://localhost:8012/mcp/tools/confluence_semantic_search \
		-H "Content-Type: application/json" \
		-d '{"query": "test", "limit": 3}' | jq -r '.result' || echo "❌ Ошибка"

test-health: ## Тест health check
	@echo "Health check..."
	@curl -s -X POST http://localhost:8012/mcp/tools/confluence_health | jq -r '.result' || echo "❌ Ошибка"

# Мониторинг
watch-logs: ## Watch логи синхронизации
	watch -n 2 'docker logs $(CONTAINER) 2>&1 | tail -n 30'

watch-status: ## Watch статус ресурсов
	watch -n 2 'docker stats $(CONTAINER) --no-stream'

# Информация
info: ## Показать информацию о проекте
	@echo "Confluence RAG Project Info"
	@echo "============================"
	@echo "Container: $(CONTAINER)"
	@echo "Compose file: docker-compose.yml"
	@echo ""
	@echo "Volumes:"
	@docker inspect $(CONTAINER) --format='{{range .Mounts}}  {{.Source}} → {{.Destination}}{{"\n"}}{{end}}' 2>/dev/null || echo "  Container not running"
	@echo ""
	@echo "Environment:"
	@docker exec $(CONTAINER) env | grep -E "(CONFLUENCE_URL|EMBED_MODEL|MAX_SPACES|SYNC_INTERVAL)" 2>/dev/null || echo "  Container not running"

version: ## Показать версии
	@echo "Python: $$(docker exec $(CONTAINER) python --version 2>/dev/null || echo 'N/A')"
	@echo "Docker: $$(docker --version)"
	@echo "Docker Compose: $$(docker compose version)"

# Troubleshooting
debug: ## Отладочная информация
	@echo "=== Debug Information ==="
	@echo ""
	@echo "Container status:"
	@docker ps -a --filter name=$(CONTAINER) --format "  {{.Status}}"
	@echo ""
	@echo "Recent errors:"
	@docker logs $(CONTAINER) 2>&1 | grep -i error | tail -n 5 || echo "  No errors"
	@echo ""
	@echo "Disk usage:"
	@du -sh chroma_data/ 2>/dev/null || echo "  No data"
	@echo ""
	@echo "State file:"
	@ls -lh sync_state.json 2>/dev/null || echo "  No state file"

fix-permissions: ## Исправить права на файлы
	@echo "Исправление прав доступа..."
	@chmod -R 755 chroma_data/ 2>/dev/null || true
	@chmod 644 sync_state.json 2>/dev/null || true
	@echo "✅ Готово"

# Установка
setup: ## Первоначальная настройка
	@echo "Настройка Confluence RAG..."
	@if [ ! -f .env ]; then \
		echo "Создание .env из ENV_TEMPLATE..."; \
		cp ENV_TEMPLATE .env; \
		echo "⚠️  Отредактируйте .env и укажите CONFLUENCE_URL и CONFLUENCE_TOKEN"; \
	else \
		echo "✅ .env уже существует"; \
	fi
	@mkdir -p chroma_data $(BACKUP_DIR)
	@echo "✅ Настройка завершена"

# Production
prod-deploy: setup build up ## Развёртывание в production
	@echo "✅ Production развёртывание завершено"
	@echo "Проверьте логи: make logs"

prod-update: backup down build up ## Обновление в production (с backup)
	@echo "✅ Production обновление завершено"

