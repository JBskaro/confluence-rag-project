import os
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Централизованная конфигурация RAG системы.
    Использует pydantic-settings для валидации и загрузки из .env
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Игнорировать лишние переменные в .env
    )
    
    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "confluence"
    qdrant_api_key: Optional[str] = None
    
    # --- Embeddings ---
    # huggingface, ollama, openai, openrouter
    embedding_source: str = "huggingface"
    # Модель по умолчанию
    embed_model: str = "ai-forever/FRIDA"
    embedding_dimension: int = 1024
    
    # --- Ollama ---
    use_ollama: bool = False
    ollama_url: str = "http://ollama:11434"
    # Модели для генерации и эмбеддингов
    ollama_model: str = "llama3.2"
    ollama_embedding_model: Optional[str] = None
    
    # --- OpenAI / OpenRouter ---
    openai_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    
    # --- Search Configuration ---
    enable_hybrid_search: bool = True
    enable_mmr: bool = True
    mmr_diversity_weight: float = 0.3
    reranker_model: str = "DiTy/cross-encoder-russian-msmarco"
    
    # === Hybrid Search Weights ===
    hybrid_vector_weight: float = 0.6
    hybrid_bm25_weight: float = 0.4
    hybrid_rrf_k: int = 60
    bm25_max_docs: int = 50000
    
    # Adaptive weights (Navigational)
    hybrid_vector_weight_navigational: float = 0.7
    hybrid_bm25_weight_navigational: float = 0.3
    
    # Adaptive weights (Exploratory)
    hybrid_vector_weight_exploratory: float = 0.5
    hybrid_bm25_weight_exploratory: float = 0.5
    
    # Adaptive weights (Factual)
    hybrid_vector_weight_factual: float = 0.6
    hybrid_bm25_weight_factual: float = 0.4
    
    # Adaptive weights (How-to)
    hybrid_vector_weight_howto: float = 0.55
    hybrid_bm25_weight_howto: float = 0.45
    
    # --- Context Expansion ---
    enable_context_expansion: bool = True
    context_expansion_mode: str = "bidirectional" # window, bidirectional, hierarchy
    context_expansion_size: int = 2
    
    # --- Advanced Search ---
    use_ollama_for_query_expansion: bool = False
    enable_prf_fallback: bool = True
    
    # --- Observability ---
    enable_tracing: bool = True
    enable_metrics: bool = True
    metrics_port: int = 9090
    jaeger_endpoint: Optional[str] = "http://tempo:4317"
    otlp_export_interval_millis: int = 15000  # 15 seconds
    
    # --- App Info ---
    app_version: str = "2.2.0"
    app_env: str = "production"
    
    # --- Confluence ---
    confluence_url: Optional[str] = None
    confluence_username: Optional[str] = None
    confluence_api_token: Optional[str] = None
    
    # --- Hallucination Detection ---
    enable_hallucination_detection: bool = False
    hallucination_threshold: float = 0.7
    hallucination_keyword_overlap: float = 0.3

# Singleton instance
settings = Settings()
