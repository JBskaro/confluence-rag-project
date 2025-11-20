#!/usr/bin/env python3
"""
Скрипт для полной очистки базы данных (Qdrant + PostgreSQL).

Использование:
    docker-compose exec confluence-rag python scripts/clear_database.py
"""
import sys
import os
import logging

# Добавляем путь к модулям
sys.path.insert(0, '/app')

from qdrant_storage import clear_qdrant_collection, init_qdrant_collection
from postgres_storage import clear_all_pages_postgres, init_postgres_schema
from embeddings import get_embedding_dimension

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Очистить всю базу данных."""
    logger.info("=" * 60)
    logger.info("ОЧИСТКА БАЗЫ ДАННЫХ")
    logger.info("=" * 60)
    
    # 1. Очистка Qdrant
    logger.info("\n📦 Шаг 1: Очистка Qdrant коллекции...")
    try:
        qdrant_deleted = clear_qdrant_collection()
        logger.info(f"✅ Qdrant: удалено {qdrant_deleted} точек")
    except Exception as e:
        logger.error(f"❌ Ошибка очистки Qdrant: {e}")
        return 1
    
    # 2. Очистка PostgreSQL
    logger.info("\n🗄️  Шаг 2: Очистка PostgreSQL...")
    try:
        postgres_deleted = clear_all_pages_postgres()
        logger.info(f"✅ PostgreSQL: удалено {postgres_deleted} страниц")
    except Exception as e:
        logger.error(f"❌ Ошибка очистки PostgreSQL: {e}")
        return 1
    
    # 3. Проверка размерности для переинициализации коллекции
    logger.info("\n🔧 Шаг 3: Проверка размерности модели...")
    try:
        model_dim = get_embedding_dimension()
        logger.info(f"✅ Размерность embeddings: {model_dim}D")
        
        # Переинициализируем коллекцию (на случай если структура изменилась)
        logger.info("\n🔄 Шаг 4: Переинициализация Qdrant коллекции...")
        if init_qdrant_collection(model_dim):
            logger.info("✅ Qdrant коллекция переинициализирована")
        else:
            logger.warning("⚠️  Не удалось переинициализировать коллекцию")
    except Exception as e:
        logger.error(f"❌ Ошибка проверки размерности: {e}")
        return 1
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ ОЧИСТКА ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info("\n📝 Следующие шаги:")
    logger.info("   1. Перезапустите контейнер: docker-compose restart confluence-rag")
    logger.info("   2. Мониторьте логи: docker-compose logs -f confluence-rag")
    logger.info("   3. Дождитесь полной синхронизации")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

