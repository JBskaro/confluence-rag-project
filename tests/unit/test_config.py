"""Unit tests для Pydantic Settings"""
import pytest
from rag_server.config import Settings, settings

def test_settings_load():
    """Тест загрузки настроек"""
    assert settings is not None
    assert isinstance(settings, Settings)

def test_settings_qdrant():
    """Тест Qdrant настроек"""
    assert settings.qdrant_host is not None
    assert settings.qdrant_port > 0
    assert settings.qdrant_collection is not None

def test_settings_hybrid_search():
    """Тест Hybrid Search настроек"""
    assert settings.enable_hybrid_search in [True, False]
    assert 0 <= settings.hybrid_vector_weight <= 1
    assert 0 <= settings.hybrid_bm25_weight <= 1

def test_settings_context_expansion():
    """Тест Context Expansion настроек"""
    assert settings.enable_context_expansion in [True, False]
    assert settings.context_expansion_mode in [
        'bidirectional', 'related', 'parent', 'all'
    ]
    assert settings.context_expansion_size > 0

def test_settings_observability():
    """Тест Observability настроек"""
    assert settings.enable_metrics in [True, False]
    assert settings.enable_tracing in [True, False]
    assert settings.metrics_port > 0

def test_settings_from_env(monkeypatch):
    """Тест загрузки из ENV"""
    monkeypatch.setenv("QDRANT_HOST", "test-host")
    monkeypatch.setenv("QDRANT_PORT", "9999")
    
    # Reload settings
    from importlib import reload
    from rag_server import config
    reload(config)
    
    assert config.settings.qdrant_host == "test-host"
    assert config.settings.qdrant_port == 9999

# def test_settings_validation():
#     """Тест валидации настроек"""
#     # Pydantic validation depends on field constraints which might not be set
#     pass

