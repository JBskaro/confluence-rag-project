"""
PostgreSQL промежуточное хранилище для страниц Confluence.
Хранит страницы перед индексацией в векторную БД.

✅ WITH CONNECTION POOLING
"""
import os
import json
import logging
import psycopg2
from psycopg2 import pool  # ✅ NEW
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# PostgreSQL connection settings
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "confluence_rag")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

# ✅ NEW: Connection pool settings
POSTGRES_POOL_MIN = int(os.getenv("POSTGRES_POOL_MIN", "2"))
POSTGRES_POOL_MAX = int(os.getenv("POSTGRES_POOL_MAX", "10"))

# ✅ NEW: Global connection pool
_postgres_pool = None


def get_postgres_pool():
    """
    Получить connection pool для PostgreSQL.
    
    ✅ Создается один раз и переиспользуется.
    """
    global _postgres_pool
    
    if _postgres_pool is None:
        try:
            _postgres_pool = pool.ThreadedConnectionPool(
                minconn=POSTGRES_POOL_MIN,
                maxconn=POSTGRES_POOL_MAX,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                connect_timeout=10
            )
            logger.info(f"✅ PostgreSQL connection pool created: min={POSTGRES_POOL_MIN}, max={POSTGRES_POOL_MAX}")
        except Exception as e:
            logger.error(f"❌ Failed to create PostgreSQL connection pool: {e}")
            raise
    
    return _postgres_pool


@contextmanager
def get_postgres_connection():
    """
    Context manager для получения соединения из пула.
    
    ✅ Автоматически возвращает соединение в пул после использования.
    
    Usage:
        with get_postgres_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM pages")
    """
    pool_obj = get_postgres_pool()
    conn = pool_obj.getconn()
    
    try:
        yield conn
    finally:
        pool_obj.putconn(conn)


def close_postgres_pool():
    """
    Закрыть connection pool (для graceful shutdown).
    """
    global _postgres_pool
    
    if _postgres_pool is not None:
        _postgres_pool.closeall()
        _postgres_pool = None
        logger.info("✅ PostgreSQL connection pool closed")


# ✅ LEGACY: Оставляем старую функцию для обратной совместимости
def get_postgres_connection_legacy():
    """Получить соединение с PostgreSQL (legacy, без pool)."""
    try:
        return psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=10
        )
    except Exception as e:
        logger.error(f"Ошибка подключения к PostgreSQL: {e}")
        raise

def init_postgres_schema():
    """Инициализировать схему PostgreSQL."""
    try:
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS confluence_pages (
                        page_id TEXT PRIMARY KEY,
                        space_key TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content_html TEXT,
                        content_markdown TEXT,
                        version INTEGER NOT NULL,
                        metadata JSONB,
                        updated_at TIMESTAMP NOT NULL,
                        indexed_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_space_key ON confluence_pages(space_key);
                    CREATE INDEX IF NOT EXISTS idx_indexed_at ON confluence_pages(indexed_at);
                    CREATE INDEX IF NOT EXISTS idx_updated_at ON confluence_pages(updated_at);
                    CREATE INDEX IF NOT EXISTS idx_version ON confluence_pages(version);
                """)
                conn.commit()
                logger.info("✅ PostgreSQL schema initialized")
    except Exception as e:
        logger.error(f"Ошибка инициализации PostgreSQL schema: {e}")
        raise

def save_page_to_postgres(
    page_id: str,
    space_key: str,
    title: str,
    content_html: str,
    content_markdown: str,
    version: int,
    metadata: Dict[str, Any],
    updated_at: datetime
) -> bool:
    """Сохранить страницу в PostgreSQL."""
    try:
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO confluence_pages 
                    (page_id, space_key, title, content_html, content_markdown, version, metadata, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (page_id) 
                    DO UPDATE SET
                        space_key = EXCLUDED.space_key,
                        title = EXCLUDED.title,
                        content_html = EXCLUDED.content_html,
                        content_markdown = EXCLUDED.content_markdown,
                        version = EXCLUDED.version,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at,
                        indexed_at = NULL
                """, (
                    page_id, space_key, title, content_html, content_markdown,
                    version, json.dumps(metadata), updated_at
                ))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Ошибка сохранения страницы {page_id} в PostgreSQL: {e}")
        return False

def get_pages_from_postgres(
    space_key: Optional[str] = None,
    not_indexed: bool = False,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Получить страницы из PostgreSQL."""
    try:
        with get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT * FROM confluence_pages WHERE 1=1"
                params = []
                
                if space_key:
                    query += " AND space_key = %s"
                    params.append(space_key)
                
                if not_indexed:
                    query += " AND indexed_at IS NULL"
                
                query += " ORDER BY updated_at DESC"
                
                if limit:
                    query += " LIMIT %s"
                    params.append(limit)
                
                cur.execute(query, params)
                rows = cur.fetchall()
                result = []
                for row in rows:
                    page_dict = dict(row)
                    # Парсим JSONB metadata
                    if page_dict.get('metadata') and isinstance(page_dict['metadata'], str):
                        try:
                            page_dict['metadata'] = json.loads(page_dict['metadata'])
                        except:
                            page_dict['metadata'] = {}
                    elif page_dict.get('metadata') is None:
                        page_dict['metadata'] = {}
                    result.append(page_dict)
                return result
    except Exception as e:
        logger.error(f"Ошибка чтения страниц из PostgreSQL: {e}")
        return []

def mark_as_indexed(page_id: str):
    """Пометить страницу как проиндексированную."""
    try:
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE confluence_pages SET indexed_at = CURRENT_TIMESTAMP WHERE page_id = %s",
                    (page_id,)
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Ошибка пометки страницы {page_id} как проиндексированной: {e}")

def cleanup_deleted_pages_postgres(current_page_ids: set) -> int:
    """Удалить страницы, которых нет в current_page_ids."""
    try:
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                # Получаем все page_id из PostgreSQL
                cur.execute("SELECT page_id FROM confluence_pages")
                all_page_ids = {row[0] for row in cur.fetchall()}
                
                # Находим удаленные
                deleted_page_ids = all_page_ids - current_page_ids
                
                if not deleted_page_ids:
                    return 0
                
                # Удаляем
                cur.execute(
                    "DELETE FROM confluence_pages WHERE page_id = ANY(%s)",
                    (list(deleted_page_ids),)
                )
                conn.commit()
                logger.info(f"Удалено {len(deleted_page_ids)} страниц из PostgreSQL")
                return len(deleted_page_ids)
    except Exception as e:
        logger.error(f"Ошибка очистки удаленных страниц: {e}")
        return 0

def clear_all_pages_postgres() -> int:
    """
    Полностью очистить таблицу confluence_pages (удалить все страницы).
    
    Returns:
        Количество удаленных страниц
    """
    try:
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                # Получаем количество перед удалением
                cur.execute("SELECT COUNT(*) FROM confluence_pages")
                count = cur.fetchone()[0]
                
                # Удаляем все
                cur.execute("TRUNCATE TABLE confluence_pages RESTART IDENTITY CASCADE")
                conn.commit()
                logger.info(f"✅ PostgreSQL очищен: удалено {count} страниц")
                return count
    except Exception as e:
        logger.error(f"Ошибка очистки PostgreSQL: {e}")
        return 0

def get_postgres_stats() -> Dict[str, Any]:
    """Получить статистику PostgreSQL."""
    try:
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_pages,
                        COUNT(CASE WHEN indexed_at IS NULL THEN 1 END) as not_indexed,
                        COUNT(DISTINCT space_key) as spaces_count
                    FROM confluence_pages
                """)
                row = cur.fetchone()
                return {
                    'total_pages': row[0] if row else 0,
                    'not_indexed': row[1] if row else 0,
                    'spaces_count': row[2] if row else 0
                }
    except Exception as e:
        logger.error(f"Ошибка получения статистики PostgreSQL: {e}")
        return {'total_pages': 0, 'not_indexed': 0, 'spaces_count': 0}

