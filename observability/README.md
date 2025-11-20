# Observability Stack для Confluence RAG

Полный стек мониторинга и трейсинга для Confluence RAG системы.

## 📊 Компоненты

### 1. **OpenTelemetry** - Distributed Tracing
- Автоматический трейсинг всех запросов
- Waterfall view всех операций в pipeline
- Span attributes для детального анализа

### 2. **Tempo** - Trace Storage
- Хранилище traces
- Query interface для поиска traces
- Интеграция с Grafana

### 3. **Prometheus** - Metrics Collection
- Сбор метрик производительности
- Histograms для latency
- Counters для cache hits/misses
- Gauges для active requests

### 4. **Grafana** - Visualization
- 3 готовых dashboard'а:
  - **Search Performance** - общая производительность
  - **Pipeline Breakdown** - детальный breakdown по этапам
  - **Reranking Analysis** - анализ reranking производительности

## 🚀 Быстрый старт

### 1. Запуск observability stack

```bash
# Запустить Tempo + Prometheus + Grafana
docker-compose -f docker-compose.observability.yml up -d

# Проверить статус
docker-compose -f docker-compose.observability.yml ps
```

### 2. Установка зависимостей

```bash
# Установить Python зависимости
pip install -r requirements.txt

# Или в Docker контейнере
docker-compose exec confluence-rag pip install -r requirements.txt
```

### 3. Перезапуск Confluence RAG с observability

```bash
# Rebuild контейнер (если нужно)
docker-compose build confluence-rag

# Перезапустить
docker-compose restart confluence-rag

# Проверить логи
docker-compose logs -f confluence-rag | grep -i observability
```

### 4. Открыть Grafana

```
URL: http://localhost:3001
Username: admin (или anonymous если настроено)
Password: admin (или не требуется)
```

## 📈 Dashboards

### Search Performance Overview
- **Total Latency (p50, p95, p99)** - общая задержка поиска
- **Requests Per Second** - throughput
- **Active Requests** - текущие активные запросы
- **Cache Hit Rate** - процент попаданий в кэш

### Pipeline Breakdown
- **Stacked Bar Chart** - время по этапам:
  - Embedding Generation
  - Vector Search
  - BM25 Search
  - RRF Merge
  - Reranking
- **Individual Gauges** - p95 latency для каждого этапа
- **Total Latency** - общая задержка

### Reranking Analysis
- **Reranking Latency Percentiles** - p50, p95, p99
- **Documents Reranked** - количество документов
- **Latency per Document** - ms/doc
- **Throughput** - docs/sec
- **Score Distribution** - распределение rerank scores

## 🔍 Метрики

### Prometheus Metrics

#### Latency Metrics
- `confluence_search_latency_seconds` - общая задержка поиска
- `confluence_rerank_latency_seconds` - задержка reranking
- `confluence_bm25_latency_seconds` - задержка BM25 поиска
- `confluence_vector_latency_seconds` - задержка vector поиска
- `confluence_embedding_latency_seconds` - задержка генерации embeddings
- `confluence_rrf_latency_seconds` - задержка RRF merge

#### Cache Metrics
- `confluence_cache_hits_total` - количество cache hits
- `confluence_cache_misses_total` - количество cache misses

#### Request Metrics
- `confluence_active_requests` - активные запросы
- `confluence_requests_total{query_type, status}` - общее количество запросов

#### Reranking Metrics
- `confluence_rerank_docs_count` - количество документов для reranking
- `confluence_rerank_scores` - распределение rerank scores

#### Results Metrics
- `confluence_results_count` - количество возвращенных результатов

### Prometheus Endpoint

```
http://localhost:8001/metrics
```

## 🔗 Endpoints

- **Grafana:** http://localhost:3001
- **Prometheus:** http://localhost:9090
- **Tempo:** http://localhost:3200
- **Prometheus Metrics:** http://localhost:8001/metrics (Confluence RAG)

## 📝 Конфигурация

### Environment Variables

```bash
# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317

# Prometheus
PROMETHEUS_PORT=8001

# Service Info
APP_VERSION=2.2.0
ENV=production
```

### Tempo Configuration
`observability/tempo-config.yaml`

### Prometheus Configuration
`observability/prometheus.yml`

### Grafana Configuration
- Datasources: `observability/grafana/datasources.yml`
- Dashboards: `observability/grafana/dashboards.yml`
- Dashboard JSONs: `observability/grafana/dashboards/*.json`

## 🐛 Troubleshooting

### Метрики не появляются в Prometheus

1. Проверьте что Prometheus scraper работает:
```bash
curl http://localhost:9090/api/v1/targets
```

2. Проверьте что Confluence RAG экспортирует метрики:
```bash
curl http://localhost:8001/metrics
```

3. Проверьте логи:
```bash
docker-compose logs prometheus
docker-compose logs confluence-rag | grep -i prometheus
```

### Traces не появляются в Tempo

1. Проверьте что Tempo работает:
```bash
curl http://localhost:3200/ready
```

2. Проверьте что OpenTelemetry настроен:
```bash
docker-compose logs confluence-rag | grep -i observability
```

3. Проверьте OTLP endpoint:
```bash
# В .env должно быть:
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
```

### Grafana не показывает данные

1. Проверьте datasources:
   - Prometheus: http://prometheus:9090
   - Tempo: http://tempo:3200

2. Проверьте что dashboards загружены:
   - Settings → Dashboards → Confluence RAG

3. Проверьте time range (должен быть "Last 1 hour" или больше)

## 📚 Дополнительные ресурсы

- [OpenTelemetry Python Docs](https://opentelemetry.io/docs/instrumentation/python/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- [Tempo Documentation](https://grafana.com/docs/tempo/latest/)

## 🎯 Следующие шаги

1. **Найти bottlenecks** - используйте Pipeline Breakdown dashboard
2. **Оптимизировать** - на основе данных из Grafana
3. **Настроить alerts** - для критичных метрик (latency > 5s, error rate > 1%)
4. **Добавить custom metrics** - для специфичных use cases

