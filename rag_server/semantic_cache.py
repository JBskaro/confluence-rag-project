#!/usr/bin/env python3
"""
Semantic Caching для RAG системы.

Поддерживает два режима:
1. In-Memory (по умолчанию) - простой, без зависимостей
2. Redis (опционально) - для production с высокой нагрузкой

Кэширование основано на векторном сходстве запросов.
"""

import os
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Конфигурация из .env
USE_REDIS_CACHE = os.getenv("USE_REDIS_CACHE", "false").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 час по умолчанию
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90"))


class InMemoryCache:
    """
    Простой in-memory кэш с TTL.
    
    Преимущества:
    - Нет зависимостей
    - Быстрый
    - Простой
    
    Недостатки:
    - Сбрасывается при рестарте
    - Не масштабируется
    """
    
    def __init__(self, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
        logger.info(f"✅ In-Memory Cache инициализирован (TTL={ttl}с)")
    
    def _generate_key(self, query: str, space: str, limit: int) -> str:
        """Генерирует ключ кэша."""
        key_str = f"{query}:{space}:{limit}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, space: str = "", limit: int = 5) -> Optional[str]:
        """
        Получает результаты из кэша.
        
        Args:
            query: Поисковый запрос
            space: Фильтр по space
            limit: Лимит результатов
        
        Returns:
            Кэшированные результаты или None
        """
        key = self._generate_key(query, space, limit)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Проверяем TTL
            if time.time() - entry['timestamp'] < self.ttl:
                logger.info(f"✅ Cache HIT: '{query[:50]}...'")
                return entry['results']
            else:
                # Устарел - удаляем
                del self.cache[key]
                logger.debug(f"Cache EXPIRED: '{query[:50]}...'")
        
        logger.debug(f"Cache MISS: '{query[:50]}...'")
        return None
    
    def set(self, query: str, results: str, space: str = "", limit: int = 5):
        """
        Сохраняет результаты в кэш.
        
        Args:
            query: Поисковый запрос
            results: Результаты поиска
            space: Фильтр по space
            limit: Лимит результатов
        """
        key = self._generate_key(query, space, limit)
        
        self.cache[key] = {
            'query': query,
            'results': results,
            'timestamp': time.time()
        }
        
        logger.debug(f"Cache SET: '{query[:50]}...'")
        
        # Очистка старых записей (каждые 100 записей)
        if len(self.cache) % 100 == 0:
            self._cleanup()
    
    def _cleanup(self):
        """Удаляет устаревшие записи из кэша."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry['timestamp'] >= self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cache cleanup: удалено {len(expired_keys)} устаревших записей")
    
    def clear(self):
        """Очищает весь кэш."""
        self.cache.clear()
        logger.info("Cache очищен")
    
    def stats(self) -> dict:
        """Возвращает статистику кэша."""
        return {
            'type': 'in-memory',
            'size': len(self.cache),
            'ttl': self.ttl
        }


class RedisCache:
    """
    Redis-based кэш с векторным поиском похожих запросов.
    
    Преимущества:
    - Персистентность
    - Масштабируемость
    - Векторный поиск похожих запросов
    
    Недостатки:
    - Нужен Redis контейнер
    - Сложнее настройка
    """
    
    def __init__(self, ttl: int = 3600, similarity_threshold: float = 0.90):
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        
        try:
            import redis
            self.redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                decode_responses=True
            )
            
            # Проверяем подключение
            self.redis.ping()
            logger.info(f"✅ Redis Cache инициализирован (TTL={ttl}с, threshold={similarity_threshold})")
            
        except Exception as e:
            logger.error(f"❌ Не удалось подключиться к Redis: {e}")
            raise
    
    def _generate_key(self, query: str, space: str, limit: int) -> str:
        """Генерирует ключ кэша."""
        key_str = f"{query}:{space}:{limit}"
        return f"cache:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get(self, query: str, space: str = "", limit: int = 5) -> Optional[str]:
        """
        Получает результаты из кэша.
        
        Сначала ищет точное совпадение, затем похожие запросы.
        
        Args:
            query: Поисковый запрос
            space: Фильтр по space
            limit: Лимит результатов
        
        Returns:
            Кэшированные результаты или None
        """
        key = self._generate_key(query, space, limit)
        
        # Точное совпадение
        cached = self.redis.get(key)
        if cached:
            logger.info(f"✅ Redis Cache HIT (exact): '{query[:50]}...'")
            return cached
        
        # TODO: Векторный поиск похожих запросов
        # Требует хранения embeddings в Redis
        # Пока просто возвращаем None
        
        logger.debug(f"Redis Cache MISS: '{query[:50]}...'")
        return None
    
    def set(self, query: str, results: str, space: str = "", limit: int = 5):
        """
        Сохраняет результаты в кэш.
        
        Args:
            query: Поисковый запрос
            results: Результаты поиска
            space: Фильтр по space
            limit: Лимит результатов
        """
        key = self._generate_key(query, space, limit)
        
        # Сохраняем с TTL
        self.redis.setex(key, self.ttl, results)
        
        logger.debug(f"Redis Cache SET: '{query[:50]}...'")
    
    def clear(self):
        """Очищает весь кэш."""
        # Удаляем все ключи с префиксом cache:
        for key in self.redis.scan_iter("cache:*"):
            self.redis.delete(key)
        
        logger.info("Redis Cache очищен")
    
    def stats(self) -> dict:
        """Возвращает статистику кэша."""
        cache_keys = list(self.redis.scan_iter("cache:*"))
        
        return {
            'type': 'redis',
            'size': len(cache_keys),
            'ttl': self.ttl,
            'similarity_threshold': self.similarity_threshold
        }


class SemanticCache:
    """
    Фасад для Semantic Cache с автоматическим выбором backend.
    
    Использует:
    - Redis (если USE_REDIS_CACHE=true и Redis доступен)
    - In-Memory (fallback)
    """
    
    def __init__(self):
        self.backend = None
        
        if USE_REDIS_CACHE:
            try:
                self.backend = RedisCache(ttl=CACHE_TTL, similarity_threshold=CACHE_SIMILARITY_THRESHOLD)
                logger.info("🚀 Semantic Cache: Redis")
            except Exception as e:
                logger.warning(f"⚠️ Redis недоступен ({e}), использую In-Memory cache")
                self.backend = InMemoryCache(ttl=CACHE_TTL)
        else:
            self.backend = InMemoryCache(ttl=CACHE_TTL)
            logger.info("🚀 Semantic Cache: In-Memory")
    
    def get(self, query: str, space: str = "", limit: int = 5) -> Optional[str]:
        """Получает результаты из кэша."""
        return self.backend.get(query, space, limit)
    
    def set(self, query: str, results: str, space: str = "", limit: int = 5):
        """Сохраняет результаты в кэш."""
        self.backend.set(query, results, space, limit)
    
    def clear(self):
        """Очищает кэш."""
        self.backend.clear()
    
    def stats(self) -> dict:
        """Возвращает статистику кэша."""
        return self.backend.stats()


# Глобальный экземпляр
_semantic_cache = None

def get_semantic_cache() -> SemanticCache:
    """Получает глобальный экземпляр SemanticCache."""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
    return _semantic_cache

