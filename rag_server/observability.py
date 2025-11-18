"""
Observability setup для Confluence RAG.
Автоматический трейсинг всех запросов через OpenTelemetry.

Usage:
    from observability import setup_observability, tracer, SEARCH_LATENCY, CACHE_HITS

    # В начале приложения:
    setup_observability("confluence-rag")

    # В функциях:
    with tracer.start_as_current_span("my_operation"):
        SEARCH_LATENCY.observe(latency_seconds)
        CACHE_HITS.inc()
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================
# PROMETHEUS METRICS
# ============================================

try:
    from prometheus_client import (
        start_http_server, 
        Histogram, 
        Counter, 
        Gauge,
        Info
    )
    
    HAS_PROMETHEUS = True
    
    # Latency metrics
    SEARCH_LATENCY = Histogram(
        'confluence_search_latency_seconds',
        'Total search latency in seconds',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    )
    
    RERANK_LATENCY = Histogram(
        'confluence_rerank_latency_seconds',
        'Reranking latency in seconds',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    
    BM25_LATENCY = Histogram(
        'confluence_bm25_latency_seconds',
        'BM25 search latency in seconds',
        buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    VECTOR_LATENCY = Histogram(
        'confluence_vector_latency_seconds',
        'Vector search latency in seconds',
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    )
    
    EMBEDDING_LATENCY = Histogram(
        'confluence_embedding_latency_seconds',
        'Embedding generation latency in seconds',
        buckets=[0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    )
    
    RRF_LATENCY = Histogram(
        'confluence_rrf_latency_seconds',
        'RRF merge latency in seconds',
        buckets=[0.001, 0.01, 0.05, 0.1, 0.5]
    )
    
    # Cache metrics
    CACHE_HITS = Counter(
        'confluence_cache_hits_total',
        'Total cache hits'
    )
    
    CACHE_MISSES = Counter(
        'confluence_cache_misses_total',
        'Total cache misses'
    )
    
    # Request metrics
    ACTIVE_REQUESTS = Gauge(
        'confluence_active_requests',
        'Active search requests'
    )
    
    REQUESTS_TOTAL = Counter(
        'confluence_requests_total',
        'Total search requests',
        ['query_type', 'status']
    )
    
    # Reranking metrics
    RERANK_DOCS = Histogram(
        'confluence_rerank_docs_count',
        'Number of docs reranked',
        buckets=[5, 10, 20, 50, 100, 200]
    )
    
    RERANK_SCORES = Histogram(
        'confluence_rerank_scores',
        'Distribution of rerank scores',
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    
    # Results metrics
    RESULTS_COUNT = Histogram(
        'confluence_results_count',
        'Number of results returned',
        buckets=[0, 1, 5, 10, 20, 50, 100]
    )
    
    # System info
    SYSTEM_INFO = Info(
        'confluence_rag_info',
        'Confluence RAG system information'
    )
    
except ImportError:
    HAS_PROMETHEUS = False
    logger.warning("prometheus_client not available, metrics disabled")
    
    # Dummy metrics
    class DummyMetric:
        def observe(self, value): pass
        def inc(self): pass
        def dec(self): pass
        def labels(self, **kwargs): return self
        def info(self, info): pass
    
    SEARCH_LATENCY = DummyMetric()
    RERANK_LATENCY = DummyMetric()
    BM25_LATENCY = DummyMetric()
    VECTOR_LATENCY = DummyMetric()
    EMBEDDING_LATENCY = DummyMetric()
    RRF_LATENCY = DummyMetric()
    CACHE_HITS = DummyMetric()
    CACHE_MISSES = DummyMetric()
    ACTIVE_REQUESTS = DummyMetric()
    REQUESTS_TOTAL = DummyMetric()
    RERANK_DOCS = DummyMetric()
    RERANK_SCORES = DummyMetric()
    RESULTS_COUNT = DummyMetric()
    SYSTEM_INFO = DummyMetric()

# ============================================
# OPENTELEMETRY SETUP
# ============================================

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.resources import Resource
    
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    logger.warning("OpenTelemetry not available, distributed tracing disabled")


def setup_observability(service_name: str = "confluence-rag") -> bool:
    """
    Настройка Observability stack (OpenTelemetry + Prometheus).
    
    Вызывать один раз при старте приложения.
    
    Args:
        service_name: Имя сервиса для идентификации в traces/metrics
    
    Returns:
        True если настройка успешна, False если observability недоступен
    """
    success = False
    
    # ============================================
    # OPENTELEMETRY SETUP
    # ============================================
    
    if HAS_OPENTELEMETRY:
        try:
            # OTLP endpoint (Tempo)
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")
            
            # Resource (service info)
            resource = Resource.create({
                "service.name": service_name,
                "service.version": os.getenv("APP_VERSION", "2.2.0"),
                "deployment.environment": os.getenv("ENV", "production")
            })
            
            # Trace provider
            trace_provider = TracerProvider(resource=resource)
            trace_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
                )
            )
            trace.set_tracer_provider(trace_provider)
            
            # Metrics provider
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True),
                export_interval_millis=15000  # 15 seconds
            )
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_reader]
            )
            metrics.set_meter_provider(meter_provider)
            
            logger.info(f"✅ OpenTelemetry настроен:")
            logger.info(f"  - Service: {service_name}")
            logger.info(f"  - OTLP endpoint: {otlp_endpoint}")
            success = True
            
        except Exception as e:
            logger.warning(f"⚠️ OpenTelemetry setup failed: {e}")
            logger.warning("   Продолжаем без distributed tracing")
    else:
        logger.info("ℹ️  OpenTelemetry не установлен, distributed tracing disabled")
    
    # ============================================
    # PROMETHEUS SETUP
    # ============================================
    
    if HAS_PROMETHEUS:
        try:
            prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8001"))
            start_http_server(prometheus_port)
            
            # Set system info
            SYSTEM_INFO.info({
                'version': os.getenv("APP_VERSION", "2.2.0"),
                'python_version': os.sys.version.split()[0],
                'service': service_name
            })
            
            logger.info(f"✅ Prometheus metrics доступны:")
            logger.info(f"  - Endpoint: http://0.0.0.0:{prometheus_port}/metrics")
            success = True
            
        except Exception as e:
            logger.warning(f"⚠️ Prometheus setup failed: {e}")
            logger.warning("   Продолжаем без Prometheus metrics")
    else:
        logger.info("ℹ️  Prometheus client не установлен, metrics disabled")
    
    return success


# ============================================
# TRACER & METER INSTANCES
# ============================================

if HAS_OPENTELEMETRY:
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
else:
    # Dummy tracer
    class DummySpan:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_attribute(self, key, value):
            pass
    
    class DummyTracer:
        def start_as_current_span(self, name, **kwargs):
            return DummySpan()
    
    tracer = DummyTracer()
    meter = None


# ============================================
# CONTEXT MANAGER FOR METRICS
# ============================================

class timed_operation:
    """
    Context manager для автоматического измерения времени операции.
    
    Usage:
        with timed_operation(SEARCH_LATENCY, "my_search"):
            # your code
            pass
    """
    def __init__(self, metric: Optional[Histogram], operation_name: str = ""):
        self.metric = metric
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        if self.metric and self.start_time:
            elapsed = time.time() - self.start_time
            self.metric.observe(elapsed)
            if self.operation_name:
                logger.debug(f"⏱️  {self.operation_name}: {elapsed:.3f}s")

