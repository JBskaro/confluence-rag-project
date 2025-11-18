"""
Сервис синхронизации Confluence в PostgreSQL + Qdrant.
Выполняет инкрементальную синхронизацию документов с умным chunking.
Архитектура: Confluence → PostgreSQL → Qdrant
"""
import os
import sys
import json
import time
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import OrderedDict
from threading import RLock

import html2text
import requests
import urllib3
from atlassian import Confluence
# llama-index импорты удалены - используем прямую работу с Qdrant
# QdrantVectorStore используется только в qdrant_storage.py для поиска (get_qdrant_vector_store)
from qdrant_storage import (
    insert_chunk_to_qdrant,
    insert_chunks_batch_to_qdrant,
    init_qdrant_client
)
from embeddings import generate_query_embedding
from tenacity import retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup, NavigableString
import html  # ИСПРАВЛЕНО: Для декодирования HTML entities

# Инициализация logger (должен быть до использования)
logger = logging.getLogger(__name__)

# === SEMANTIC CHUNKING (LangChain) ===
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    logger.warning("langchain not available, using basic chunking")

# === METADATA EXTRACTION LIMITS ===
# Константы для ограничения размера извлекаемых метаданных
MAX_HEADINGS_EXTRACT = int(os.getenv('MAX_HEADINGS_EXTRACT', '50'))
MAX_BREADCRUMB_LENGTH = int(os.getenv('MAX_BREADCRUMB_LENGTH', '200'))
MAX_BREADCRUMB_LEVELS = int(os.getenv('MAX_BREADCRUMB_LEVELS', '5'))
MAX_HEADINGS_STRING_LENGTH = int(os.getenv('MAX_HEADINGS_STRING_LENGTH', '2000'))

# === METADATA SANITIZATION ===
MAX_METADATA_SIZE = 1000  # Максимальный размер строкового поля метаданных
MAX_METADATA_LIST_SIZE = 10  # Максимальный размер списка в метаданных

def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обрезает большие поля метаданных для предотвращения избыточного хранения.
    
    КРИТИЧНО: Поле 'text' НЕ обрезается, т.к. это основной контент чанка.
    
    Args:
        metadata: Исходные метаданные
    
    Returns:
        Очищенные метаданные с ограниченным размером полей
    """
    sanitized = {}
    trimmed_fields = []
    trimmed_lists = []
    
    for key, value in metadata.items():
        if isinstance(value, str):
            # КРИТИЧНО: Поле 'text' не обрезаем - это основной контент чанка
            if key == 'text':
                sanitized[key] = value
            # Обрезаем длинные строки для остальных полей
            elif len(value) > MAX_METADATA_SIZE:
                trimmed_fields.append(f"{key}:{len(value)}→{MAX_METADATA_SIZE}")
                sanitized[key] = value[:MAX_METADATA_SIZE]
            else:
                sanitized[key] = value
        elif isinstance(value, list):
            # Ограничиваем размер списков
            if len(value) > MAX_METADATA_LIST_SIZE:
                trimmed_lists.append(f"{key}:{len(value)}→{MAX_METADATA_LIST_SIZE}")
                sanitized[key] = value[:MAX_METADATA_LIST_SIZE]
            else:
                sanitized[key] = value
        elif isinstance(value, dict):
            # Рекурсивно обрабатываем вложенные словари
            sanitized[key] = sanitize_metadata(value)
        else:
            sanitized[key] = value
    
    # Логирование для отладки
    if trimmed_fields or trimmed_lists:
        logger.debug(
            f"Trimmed metadata: fields={trimmed_fields}, lists={trimmed_lists}"
        )
    
    return sanitized

# Импортируем модули для PostgreSQL и Qdrant
sys.path.insert(0, os.path.dirname(__file__))
from postgres_storage import (
    init_postgres_schema,
    save_page_to_postgres,
    get_pages_from_postgres,
    mark_as_indexed,
    cleanup_deleted_pages_postgres,
    get_postgres_stats
)
from qdrant_storage import (
    init_qdrant_client,
    init_qdrant_collection,
    get_qdrant_vector_store,
    delete_points_by_page_id,
    get_qdrant_count
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Настройка логирования из ENV
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO), 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# logger уже определен выше (строка 30)

# Отключаем избыточное логирование HTTP запросов от httpx/openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN")

if not CONFLUENCE_URL or not CONFLUENCE_TOKEN:
    logger.error("CONFLUENCE_URL and CONFLUENCE_TOKEN required")
    sys.exit(1)

def get_int_env(name: str, default: int) -> int:
    """Безопасное получение integer ENV переменной с валидацией."""
    try:
        value = int(os.getenv(name, str(default)))
        if value <= 0:
            logger.warning(f"{name}={value} некорректно, использую default={default}")
            return default
        return value
    except (ValueError, TypeError):
        logger.warning(f"{name} невалидно, использую default={default}")
        return default

STATE_FILE = os.getenv("STATE_FILE", "./data/sync_state.json")
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
# Импортируем унифицированный модуль embeddings
# sys уже импортирован в начале файла (строка 7)
sys.path.insert(0, os.path.dirname(__file__))
from embeddings import get_embed_model, EMBED_MODEL, USE_OLLAMA, OLLAMA_URL
MAX_SPACES = get_int_env("MAX_SPACES", 10)
# Фильтр пространств: если указан CONFLUENCE_SPACES, используем его вместо MAX_SPACES
CONFLUENCE_SPACES = os.getenv("CONFLUENCE_SPACES", "").strip()
MAX_CHUNK_SIZE = get_int_env("MAX_CHUNK_SIZE", 1200)  # Оптимально для Qwen3-8B
MIN_TEXT_LEN = get_int_env("MIN_TEXT_LEN", 50)
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
BATCH_SIZE = get_int_env("BATCH_SIZE", 50)
SYNC_INTERVAL = get_int_env("SYNC_INTERVAL", 3600)

def get_bool_env(name: str, default: bool = False) -> bool:
    """Безопасное получение boolean ENV переменной."""
    value = os.getenv(name, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

# Константы для структурной нарезки
MAX_TABLE_SIZE = get_int_env("MAX_TABLE_SIZE", 2048)
CHUNK_OVERLAP = get_int_env("CHUNK_OVERLAP", 100)

# === SEMANTIC CHUNKING CONFIGURATION ===
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', str(MAX_CHUNK_SIZE)))
CHUNK_OVERLAP_SIZE = int(os.getenv('CHUNK_OVERLAP', str(CHUNK_OVERLAP)))
MIN_CHUNK_SIZE = int(os.getenv('MIN_CHUNK_SIZE', '100'))
PRESERVE_STRUCTURE = os.getenv('PRESERVE_STRUCTURE', 'true').lower() == 'true'

# Initialize semantic chunker if available
SEMANTIC_SPLITTER = None
if HAS_LANGCHAIN:
    try:
        SEMANTIC_SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP_SIZE,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        logger.info(f"✅ Semantic chunker initialized: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP_SIZE}")
    except Exception as e:
        logger.warning(f"Failed to initialize semantic chunker: {e}")
        SEMANTIC_SPLITTER = None
        HAS_LANGCHAIN = False

# Тестовый режим
TEST_MODE = get_bool_env("TEST_MODE", False)
TEST_MAX_PAGES = get_int_env("TEST_MAX_PAGES", 10)

# Проверка доступности tqdm для progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available, progress bars disabled")

logger.info("Starting Confluence RAG sync (optimized for large instances)")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_page(confluence: Confluence, page_id: str) -> Dict[str, Any]:
    """
    Получение страницы из Confluence с retry логикой и расширенными метаданными.
    
    Args:
        confluence: Confluence API client
        page_id: ID страницы
    
    Returns:
        Данные страницы с body, version, ancestors, labels, children
    """
    # Расширенный запрос для получения всех метаданных
    return confluence.get_page_by_id(
        page_id, 
        expand='body.storage,version,ancestors,metadata.labels,children.page,space'
    )

def get_timestamp(page: Dict[str, Any]) -> int:
    """
    Извлечение timestamp обновления страницы для инкрементальной синхронизации.
    
    Args:
        page: Объект страницы Confluence
    
    Returns:
        Timestamp в формате YYYYMMDD или 0 при ошибке
    """
    try:
        ts = page.get('version', {}).get('when', '')
        return int(ts[:10].replace('-', '')) if ts else 0
    except Exception as e:
        logger.debug(f"Ошибка парсинга timestamp: {e}")
        return 0

def get_page_attachments(confluence: Confluence, page_id: str) -> List[str]:
    """
    Получение списка вложений страницы.
    
    Args:
        confluence: Confluence API client
        page_id: ID страницы
    
    Returns:
        Список имён файлов вложений
    """
    try:
        url = f"{confluence.url}/rest/api/content/{page_id}/child/attachment"
        response = requests.get(url, headers=confluence.default_headers, verify=VERIFY_SSL)
        response.raise_for_status()
        data = response.json()
        attachments = data.get('results', [])
        return [att.get('title', '') for att in attachments if att.get('title')]
    except Exception as e:
        logger.debug(f"Ошибка получения attachments для {page_id}: {e}")
        return []

# ============ Smart Caching для get_page() ============
_page_cache_lock = RLock()
_page_cache = OrderedDict()  # LRU через OrderedDict
_cache_stats = {"hits": 0, "misses": 0}
_cache_max_size = 1000  # Максимальный размер кэша

def get_page_cached(confluence: Confluence, page_id: str, expand: str = "body.storage,version,ancestors,metadata.labels,children.page,space") -> Dict[str, Any]:
    """
    Получить страницу с кэшированием (LRU eviction + double-checked locking).
    
    Кэширует результаты get_page для сокращения API запросов.
    Использует LRU (Least Recently Used) eviction для предотвращения memory leaks.
    Thread-safe с защитой от race conditions.
    
    Args:
        confluence: Confluence API client
        page_id: ID страницы
        expand: Параметры расширения
    
    Returns:
        Данные страницы
    """
    cache_key = f"{page_id}:{expand}"
    
    # Первая проверка кэша (thread-safe)
    with _page_cache_lock:
        if cache_key in _page_cache:
            _cache_stats["hits"] += 1
            # LRU: move_to_end при hit
            _page_cache.move_to_end(cache_key)
            logger.debug(f"Cache HIT for page {page_id}")
            return _page_cache[cache_key]
        
        _cache_stats["misses"] += 1
    
    logger.debug(f"Cache MISS for page {page_id}, fetching from Confluence...")
    
    # Получаем страницу с retry (вне lock для избежания блокировки)
    page = get_page(confluence, page_id)
    
    # Сохраняем в кэш с double-checked locking (thread-safe)
    with _page_cache_lock:
        # Double-checked locking: проверяем ещё раз (другой поток мог уже добавить)
        if cache_key in _page_cache:
            logger.debug(f"Cache populated by another thread for page {page_id}")
            return _page_cache[cache_key]
        
        # Evict oldest if full (FIFO через popitem(last=False))
        if len(_page_cache) >= _cache_max_size:
            oldest_key, _ = _page_cache.popitem(last=False)  # Remove oldest
            logger.debug(f"Cache full, removed oldest entry: {oldest_key}")
        
        _page_cache[cache_key] = page
        # Move to end (most recently used)
        _page_cache.move_to_end(cache_key)
    
    return page

def clear_page_cache():
    """Очистить кэш страниц (thread-safe)"""
    global _page_cache, _cache_stats
    with _page_cache_lock:
        logger.info(f"Clearing page cache. Stats: hits={_cache_stats['hits']}, misses={_cache_stats['misses']}")
        _page_cache.clear()
        _cache_stats = {"hits": 0, "misses": 0}

def get_cache_stats() -> Dict[str, Any]:
    """Получить статистику кэша (thread-safe)"""
    with _page_cache_lock:
        total = _cache_stats["hits"] + _cache_stats["misses"]
        hit_rate = (_cache_stats["hits"] / total * 100) if total > 0 else 0.0
        return {
            "hits": _cache_stats["hits"],
            "misses": _cache_stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(_page_cache),
            "cache_max_size": _cache_max_size
        }

def build_breadcrumb(space_key: str, parent_titles: List[str], current_title: str, 
                     max_levels: int = None, max_length: int = None) -> str:
    """
    Построить breadcrumb путь с валидацией.
    
    Создает иерархический путь вида "Space > Parent1 > Parent2 > Current".
    Автоматически обрезает если слишком много уровней или слишком длинный.
    
    Args:
        space_key: Ключ пространства Confluence (например, "RAUII")
        parent_titles: Список родительских страниц (от корня к текущей)
        current_title: Текущий заголовок страницы
        max_levels: Максимальное количество уровней (default: из env или 5)
        max_length: Максимальная длина строки в символах (default: из env или 200)
    
    Returns:
        Строка breadcrumb path или пустая строка если нет данных
    
    Examples:
        >>> build_breadcrumb("RAUII", ["Dev", "API"], "Guide")
        'RAUII > Dev > API > Guide'
        
        >>> build_breadcrumb("RAUII", ["A"]*10, "B", max_levels=3)
        'RAUII > ... > A > A > B'
        
        >>> build_breadcrumb("SPACE", [], "Page", max_length=10)
        'SPACE > ...'
    """
    if max_levels is None:
        max_levels = MAX_BREADCRUMB_LEVELS
    if max_length is None:
        max_length = MAX_BREADCRUMB_LENGTH
    parts = []
    if space_key:
        parts.append(space_key)
    parts.extend(parent_titles)
    if current_title:
        parts.append(current_title)
    
    # Ограничение по уровням
    if len(parts) > max_levels:
        parts = parts[:1] + ['...'] + parts[-(max_levels-1):]
    
    breadcrumb = ' > '.join(parts) if parts else ''
    
    # Ограничение по длине
    if len(breadcrumb) > max_length:
        breadcrumb = breadcrumb[:max_length-3] + "..."
    
    return breadcrumb


def build_page_path(space_key: str, parent_titles: List[str]) -> str:
    """
    Построить URL-friendly путь для фильтрации.
    
    Создает путь вида "Space/Parent1/Parent2" для использования в фильтрах Qdrant.
    Автоматически экранирует символы "/" и "\\" в названиях.
    
    Args:
        space_key: Ключ пространства Confluence (например, "RAUII")
        parent_titles: Список родительских страниц (от корня к текущей)
    
    Returns:
        Путь в формате "Space/Parent1/Parent2" или space_key если нет родителей
    
    Examples:
        >>> build_page_path("RAUII", ["Dev", "API"])
        'RAUII/Dev/API'
        
        >>> build_page_path("RAUII", ["Dev/API", "Guide\\Test"])
        'RAUII/Dev_API/Guide_Test'
        
        >>> build_page_path("SPACE", [])
        'SPACE'
    """
    path_parts = []
    if space_key:
        path_parts.append(space_key)
    
    # Экранирование "/" в названиях
    safe_parent_titles = [str(t).replace('/', '_').replace('\\', '_') for t in parent_titles]
    path_parts.extend(safe_parent_titles)
    
    return '/'.join(path_parts) if path_parts else space_key

def extract_page_metadata(page_data: Dict[str, Any], space_key: str = '') -> Dict[str, Any]:
    """
    Извлечение ПОЛНЫХ метаданных из Confluence страницы с защитой от ошибок.
    
    ИСПРАВЛЕНО: Добавлены status, type, hierarchy_depth, created, modified для metadata indexing.
    ДОБАВЛЕНО: breadcrumb, page_path, headings для навигации и фильтрации.
    
    Args:
        page_data: Полные данные страницы из API
        space_key: Ключ пространства Confluence (для breadcrumb)
    
    Returns:
        Словарь с полными метаданными (всегда возвращает все ключи)
    """
    metadata = {
        'labels': [],
        'parent_id': '',
        'parent_title': '',
        'page_path': '',  # НОВОЕ: полный путь страницы (URL-friendly)
        'breadcrumb': '',  # НОВОЕ: полный путь с разделителями >
        'version': 1,
        'created_by': '',
        'modified_date': '',
        'has_children': False,
        'children_count': 0,
        'attachments': [],
        # НОВЫЕ ПОЛЯ для metadata indexing:
        'status': 'current',  # current, archived, draft
        'type': 'page',      # page, blogpost, attachment
        'hierarchy_depth': 0,
        'created': '',       # ISO format для фильтрации
        'modified': '',      # ISO format для фильтрации
        'modified_by': '',   # Кто последний раз редактировал
        # НОВЫЕ ПОЛЯ для заголовков:
        'headings': '',
        'headings_list': [],
        'heading_hierarchy': [],
        'heading_count': 0,
        'parent_titles': [],
    }
    
    if not page_data or not isinstance(page_data, dict):
        logger.debug("Invalid page_data structure")
        return metadata
    
    # Status (current, archived, draft)
    try:
        # Confluence API может возвращать status в разных местах
        status = page_data.get('status', 'current')
        if isinstance(status, str):
            metadata['status'] = status.lower()
        else:
            # Проверяем через _expandable
            expandable = page_data.get('_expandable', {})
            if 'status' in expandable:
                # Status доступен через отдельный запрос, но обычно 'current'
                metadata['status'] = 'current'
    except Exception as e:
        logger.debug(f"Error extracting status: {e}")
    
    # Type (page, blogpost, attachment)
    try:
        page_type = page_data.get('type', 'page')
        if isinstance(page_type, str):
            metadata['type'] = page_type.lower()
    except Exception as e:
        logger.debug(f"Error extracting type: {e}")
    
    # Labels (метки) - улучшенная обработка
    try:
        labels_data = page_data.get('metadata', {}).get('labels', {})
        if isinstance(labels_data, dict):
            labels = labels_data.get('results', [])
        elif isinstance(labels_data, list):
            labels = labels_data
        else:
            labels = []
        label_names = [
            label.get('name', '') for label in labels 
            if isinstance(label, dict) and label.get('name')
        ]
        metadata['labels'] = label_names
    except Exception as e:
        logger.debug(f"Error extracting labels: {e}")
    
    # Ancestors (hierarchy) - улучшенная обработка
    try:
        ancestors = page_data.get('ancestors', [])
        parent_titles = []  # ИСПРАВЛЕНО: Инициализируем список
        
        if ancestors and isinstance(ancestors, list):
            if len(ancestors) > 0:
                parent = ancestors[-1]  # Ближайший родитель
                if isinstance(parent, dict):
                    metadata['parent_id'] = str(parent.get('id', ''))
                    metadata['parent_title'] = str(parent.get('title', ''))
            
            # Hierarchy depth
            metadata['hierarchy_depth'] = len(ancestors)
            
            # Извлекаем все parent_titles
            for ancestor in ancestors:
                if isinstance(ancestor, dict):
                    ancestor_title = ancestor.get('title', '')
                    if ancestor_title:
                        parent_titles.append(ancestor_title)
        else:
            metadata['hierarchy_depth'] = 0
        
        # Сохраняем parent_titles для удобства
        metadata['parent_titles'] = parent_titles
    except Exception as e:
        logger.debug(f"Error extracting ancestors: {e}")
        metadata['hierarchy_depth'] = 0
        metadata['parent_titles'] = []
    
    # === НОВОЕ: ПОЛНЫЙ ПУТЬ С SPACE (BREADCRUMB) ===
    try:
        current_title = page_data.get('title', '')
        parent_titles = metadata.get('parent_titles', [])
        
        # ИСПРАВЛЕНО: Используем вспомогательные функции (константы уже определены в начале файла)
        metadata['breadcrumb'] = build_breadcrumb(
            space_key, 
            parent_titles, 
            current_title
        )
        
        metadata['page_path'] = build_page_path(space_key, parent_titles)
    except Exception as e:
        logger.debug(f"Error building breadcrumb: {e}")
        current_title = page_data.get('title', '')
        metadata['breadcrumb'] = f"{space_key} > {current_title}" if current_title else space_key
        metadata['page_path'] = space_key
    
    # === НОВОЕ: ИЗВЛЕЧЕНИЕ ВСЕХ ЗАГОЛОВКОВ ИЗ HTML ===
    try:
        body = page_data.get('body', {})
        storage = body.get('storage', {})
        content_html = storage.get('value', '')
        
        if content_html:
            # ИСПРАВЛЕНО: Логирование производительности
            headings_start = time.time()
            
            soup = BeautifulSoup(content_html, 'html.parser')
            
            # Извлечь все заголовки (h1-h6) с лимитом
            headings = []
            heading_hierarchy = []
            current_path = []
            
            # ИСПРАВЛЕНО: Используем константу из начала файла
            MAX_HEADINGS = MAX_HEADINGS_EXTRACT
            
            for i, heading_tag in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
                # ИСПРАВЛЕНО: Лимит заголовков
                if i >= MAX_HEADINGS:
                    logger.debug(f"Truncated headings extraction at {MAX_HEADINGS} for page {page_data.get('id', 'unknown')}")
                    break
                
                # ИСПРАВЛЕНО: HTML entities декодирование
                heading_text = heading_tag.get_text(strip=True)
                heading_text = html.unescape(heading_text)  # Декодировать &lt; в <
                
                if not heading_text:
                    continue
                
                heading_level = int(heading_tag.name[1])  # h1 -> 1, h2 -> 2, etc.
                
                headings.append({
                    'text': heading_text,
                    'level': heading_level
                })
                
                # ИСПРАВЛЕНО: Улучшенная логика иерархии
                # Обрезать path до текущего уровня
                while len(current_path) > 0 and len(current_path) >= heading_level:
                    current_path.pop()
                
                # ИСПРАВЛЕНО: Дополнить path если пропущены уровни
                # Если был h1, потом сразу h3, это нормально - просто продолжаем
                # Не добавляем placeholder, чтобы не искажать структуру
                
                current_path.append(heading_text)
                heading_hierarchy.append({
                    'text': heading_text,
                    'level': heading_level,
                    'path': ' > '.join(current_path)
                })
            
            # Список всех заголовков (для поиска)
            all_headings = [h['text'] for h in headings]
            
            # ИСПРАВЛЕНО: Ограничение длины для headings строки (используем константу)
            headings_string = ' | '.join(all_headings)
            if len(headings_string) > MAX_HEADINGS_STRING_LENGTH:
                # Обрезаем и добавляем "..."
                truncated = headings_string[:MAX_HEADINGS_STRING_LENGTH]
                last_pipe = truncated.rfind(' | ')
                if last_pipe > 0:
                    headings_string = truncated[:last_pipe] + " | ..."
                else:
                    headings_string = truncated + "..."
            
            metadata['headings'] = headings_string
            metadata['headings_list'] = all_headings
            metadata['heading_hierarchy'] = heading_hierarchy
            metadata['heading_count'] = len(all_headings)
            
            # ИСПРАВЛЕНО: Логирование производительности
            headings_time = (time.time() - headings_start) * 1000  # в миллисекундах
            if headings_time > 100:  # Медленнее 100ms
                logger.warning(
                    f"⚠️ Slow headings extraction: {headings_time:.0f}ms "
                    f"for page {page_data.get('id', 'unknown')} "
                    f"({metadata['heading_count']} headings, "
                    f"{len(content_html)} chars HTML)"
                )
            elif headings_time > 50:  # Средняя скорость
                logger.debug(
                    f"Headings extraction: {headings_time:.0f}ms "
                    f"for {metadata['heading_count']} headings"
                )
        else:
            metadata['headings'] = ''
            metadata['headings_list'] = []
            metadata['heading_hierarchy'] = []
            metadata['heading_count'] = 0
    except Exception as e:
        logger.debug(f"Error extracting headings: {e}")
        metadata['headings'] = ''
        metadata['headings_list'] = []
        metadata['heading_hierarchy'] = []
        metadata['heading_count'] = 0
    
    # Version info (created, modified, authors) - улучшенная обработка
    try:
        version = page_data.get('version', {})
        if isinstance(version, dict):
            metadata['version'] = int(version.get('number', 1))
            
            # Modified date
            modified_when = version.get('when', '')
            if modified_when:
                metadata['modified_date'] = str(modified_when)
                metadata['modified'] = modified_when  # ISO format для фильтрации
            
            # Modified by
            by_info = version.get('by', {})
            if isinstance(by_info, dict):
                metadata['modified_by'] = str(by_info.get('displayName', ''))
        
        # History для created date
        history = page_data.get('history', {})
        if isinstance(history, dict):
            created_date = history.get('createdDate', '')
            if created_date:
                metadata['created'] = created_date
            else:
                # Fallback на version.when если history нет
                metadata['created'] = metadata.get('modified', '')
        
        # Created by из history
        if isinstance(history, dict):
            created_by_info = history.get('createdBy', {})
            if isinstance(created_by_info, dict):
                metadata['created_by'] = str(created_by_info.get('displayName', ''))
            else:
                # Fallback на version.by
                by_info = version.get('by', {})
                if isinstance(by_info, dict):
                    metadata['created_by'] = str(by_info.get('displayName', ''))
    except Exception as e:
        logger.debug(f"Error extracting version/history info: {e}")
    
    # Child pages count
    try:
        children_data = page_data.get('children', {})
        if isinstance(children_data, dict):
            page_info = children_data.get('page', {})
            if isinstance(page_info, dict):
                children = int(page_info.get('size', 0))
                metadata['has_children'] = children > 0
                metadata['children_count'] = children
    except Exception as e:
        logger.debug(f"Error extracting children info: {e}")
    
    return metadata

def extract_macro_body(macro_html: str) -> str:
    """
    Извлечение текста из тела Confluence макроса.
    
    Args:
        macro_html: HTML макроса
    
    Returns:
        Текст из rich-text-body макроса
    """
    # Извлекаем содержимое <ac:rich-text-body>...</ac:rich-text-body>
    body_match = re.search(r'<ac:rich-text-body>(.*?)</ac:rich-text-body>', macro_html, re.DOTALL)
    if body_match:
        return body_match.group(1)
    return macro_html

def preprocess_confluence_macros(html: str) -> str:
    """
    Предобработка Confluence макросов для лучшей конвертации в текст.
    
    Args:
        html: HTML с Confluence макросами
    
    Returns:
        HTML с обработанными макросами
    """
    
    # Info макрос: <ac:structured-macro ac:name="info">
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="info"[^>]*>(.*?)</ac:structured-macro>',
        lambda m: f'\n\n💡 **INFO:** {extract_macro_body(m.group(1))}\n\n',
        html,
        flags=re.DOTALL
    )
    
    # Warning макрос
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="warning"[^>]*>(.*?)</ac:structured-macro>',
        lambda m: f'\n\n⚠️ **WARNING:** {extract_macro_body(m.group(1))}\n\n',
        html,
        flags=re.DOTALL
    )
    
    # Note макрос
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="note"[^>]*>(.*?)</ac:structured-macro>',
        lambda m: f'\n\n📝 **NOTE:** {extract_macro_body(m.group(1))}\n\n',
        html,
        flags=re.DOTALL
    )
    
    # Tip макрос
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="tip"[^>]*>(.*?)</ac:structured-macro>',
        lambda m: f'\n\n💡 **TIP:** {extract_macro_body(m.group(1))}\n\n',
        html,
        flags=re.DOTALL
    )
    
    # Expand макрос (скрываемый контент)
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="expand"[^>]*>(.*?)</ac:structured-macro>',
        lambda m: f'\n\n🔽 **EXPAND:** {extract_macro_body(m.group(1))}\n\n',
        html,
        flags=re.DOTALL
    )
    
    # Code макрос с языком
    def replace_code_macro(match):
        full_macro = match.group(0)
        # Извлекаем язык
        lang_match = re.search(r'<ac:parameter[^>]*ac:name="language"[^>]*>([^<]*)</ac:parameter>', full_macro)
        language = lang_match.group(1) if lang_match else ''
        # Извлекаем код
        code_match = re.search(r'<ac:plain-text-body><!\[CDATA\[(.*?)\]\]></ac:plain-text-body>', full_macro, re.DOTALL)
        code = code_match.group(1) if code_match else extract_macro_body(full_macro)
        return f'\n\n```{language}\n{code}\n```\n\n'
    
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="code"[^>]*>.*?</ac:structured-macro>',
        replace_code_macro,
        html,
        flags=re.DOTALL
    )
    
    # Panel макрос
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="panel"[^>]*>(.*?)</ac:structured-macro>',
        lambda m: f'\n\n📋 **PANEL:** {extract_macro_body(m.group(1))}\n\n',
        html,
        flags=re.DOTALL
    )
    
    # Status макрос
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="status"[^>]*>.*?<ac:parameter[^>]*ac:name="title"[^>]*>([^<]*)</ac:parameter>.*?</ac:structured-macro>',
        lambda m: f'[STATUS: {m.group(1)}]',
        html,
        flags=re.DOTALL
    )
    
    # TOC (Table of Contents) - удаляем, т.к. это автогенерируемое оглавление
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="toc"[^>]*>.*?</ac:structured-macro>',
        '',
        html,
        flags=re.DOTALL
    )
    
    # Excerpt макрос (краткое содержание)
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="excerpt"[^>]*>(.*?)</ac:structured-macro>',
        lambda m: f'\n\n📌 **EXCERPT:** {extract_macro_body(m.group(1))}\n\n',
        html,
        flags=re.DOTALL
    )
    
    # Quote макрос
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="quote"[^>]*>(.*?)</ac:structured-macro>',
        lambda m: f'\n\n> {extract_macro_body(m.group(1))}\n\n',
        html,
        flags=re.DOTALL
    )
    
    # Page Properties макрос (структурированные данные)
    def extract_page_properties(match):
        full_macro = match.group(0)
        # Ищем все параметры
        props = []
        prop_pattern = re.findall(r'<ac:parameter[^>]*ac:name="([^"]*)"[^>]*>([^<]*)</ac:parameter>', full_macro)
        for key, value in prop_pattern:
            if key and value:
                props.append(f"{key}: {value}")
        if props:
            return f'\n\n📊 **PAGE PROPERTIES:**\n' + '\n'.join([f'  • {p}' for p in props]) + '\n\n'
        return ''
    
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="details"[^>]*>.*?</ac:structured-macro>',
        extract_page_properties,
        html,
        flags=re.DOTALL
    )
    
    # Include Page макрос (транслюдированный контент)
    def extract_include_page(match):
        full_macro = match.group(0)
        # Ищем ссылку на страницу
        page_match = re.search(r'<ri:page[^>]*ri:content-title="([^"]*)"', full_macro)
        if page_match:
            page_title = page_match.group(1)
            return f'\n\n🔗 **INCLUDES PAGE:** "{page_title}"\n\n'
        return ''
    
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="include"[^>]*>.*?</ac:structured-macro>',
        extract_include_page,
        html,
        flags=re.DOTALL
    )
    
    # Children Display макрос (список дочерних страниц)
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="children"[^>]*>.*?</ac:structured-macro>',
        '\n\n📑 **CHILD PAGES LIST**\n\n',
        html,
        flags=re.DOTALL
    )
    
    # Recently Updated макрос
    html = re.sub(
        r'<ac:structured-macro[^>]*ac:name="recently-updated"[^>]*>.*?</ac:structured-macro>',
        '',
        html,
        flags=re.DOTALL
    )
    
    # Confluence таблицы: конвертируем <ac:table> в <table>
    html = re.sub(r'<ac:table>', '<table>', html)
    html = re.sub(r'</ac:table>', '</table>', html)
    html = re.sub(r'<ac:tr>', '<tr>', html)
    html = re.sub(r'</ac:tr>', '</tr>', html)
    html = re.sub(r'<ac:td>', '<td>', html)
    html = re.sub(r'</ac:td>', '</td>', html)
    html = re.sub(r'<ac:th>', '<th>', html)
    html = re.sub(r'</ac:th>', '</th>', html)
    
    return html

def convert_table_to_markdown(table_element) -> tuple[str, str]:
    """Конвертирует HTML таблицу в markdown формат. Возвращает (markdown, html)."""
    try:
        table_html = str(table_element)
        rows = []
        for tr in table_element.find_all('tr'):
            cells = []
            for td in tr.find_all(['td', 'th']):
                cell_text = td.get_text(separator=' ', strip=True)
                cell_text = cell_text.replace('|', '\\|')
                cells.append(cell_text)
            if cells:
                rows.append('| ' + ' | '.join(cells) + ' |')
        
        if not rows or len(rows) < 2:
            return "", ""
        
        num_cols = len(rows[0].split('|')) - 2
        if num_cols > 0 and len(rows) > 1:
            separator = '| ' + ' | '.join(['---'] * num_cols) + ' |'
            rows.insert(1, separator)
        
        markdown = '\n'.join(rows) if rows else ""
        return markdown, table_html
    except Exception as e:
        logger.warning(f"Ошибка конвертации таблицы: {e}")
        plain = table_element.get_text(separator=' ', strip=True)
        return plain, str(table_element)

def extract_list_text(list_element, tag: str) -> str:
    """Извлекает текст из списка с правильными маркерами."""
    try:
        items = []
        for li in list_element.find_all('li', recursive=False):
            item_text = li.get_text(separator=' ', strip=True)
            if item_text:
                items.append(item_text)
        
        if not items:
            return ""
        
        if tag == 'ul':
            return '\n'.join([f"- {item}" for item in items])
        else:
            return '\n'.join([f"{i+1}. {item}" for i, item in enumerate(items)])
    except Exception as e:
        logger.warning(f"Ошибка извлечения списка: {e}")
        return list_element.get_text(separator='\n', strip=True)

def extract_structural_blocks(html_content: str) -> List[Dict[str, Any]]:
    """Структурная нарезка HTML на логические блоки (таблицы, списки, текст)."""
    if not html_content:
        return []
    
    try:
        html_content = preprocess_confluence_macros(html_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        blocks = []
        heading_stack = []
        
        def create_block(block_type: str, content: str, heading_stack: list, html: Optional[str] = None) -> Dict[str, Any]:
            parent_path = " > ".join([h['text'] for h in heading_stack[:-1]]) if len(heading_stack) > 1 else ""
            current_h = heading_stack[-1]['text'] if heading_stack else ""
            block = {
                "type": block_type,
                "content": content,
                "heading": current_h,
                "level": heading_stack[-1]['level'] if heading_stack else 0,
                "parent_path": parent_path,
                "size": len(content)
            }
            if html:
                block["html"] = html
            return block
        
        def walk_tree(element):
            if isinstance(element, NavigableString):
                return
            tag = element.name
            if not tag:
                return
            
            if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(tag[1])
                heading_text = element.get_text(strip=True)
                if heading_text:
                    heading_stack[:] = [h for h in heading_stack if h['level'] < level]
                    heading_stack.append({"text": heading_text, "level": level})
                for child in element.children:
                    walk_tree(child)
                return
            
            if tag == 'table':
                table_md, table_html = convert_table_to_markdown(element)
                if table_md:
                    blocks.append(create_block("table", table_md, heading_stack, table_html))
                    logger.debug(f"✓ Table block (size={len(table_md)} chars): '{heading_stack[-1]['text'] if heading_stack else 'no heading'}'")
                return
            
            if tag in ['ul', 'ol']:
                list_text = extract_list_text(element, tag)
                if list_text:
                    blocks.append(create_block("list", list_text, heading_stack))
                return
            
            if tag in ['p', 'div', 'section', 'article']:
                text = element.get_text(separator=' ', strip=True)
                if text and len(text) > 20:
                    blocks.append(create_block("text", text, heading_stack))
                return
            
            for child in element.children:
                walk_tree(child)
        
        root = soup.body if soup.body else soup
        for child in root.children:
            walk_tree(child)
        
        return blocks
    except Exception as e:
        logger.error(f"Ошибка структурной нарезки: {e}", exc_info=True)
        text = html_to_text(html_content)
        return [{"type": "text", "content": text, "heading": "", "level": 0, "parent_path": "", "size": len(text)}]

def smart_chunk_with_context(blocks: List[Dict[str, Any]], max_size: int = CHUNK_SIZE) -> List[Dict[str, Any]]:
    """Умная нарезка: таблицы и списки целиком, текст по предложениям."""
    chunks = []
    
    for block in blocks:
        block_type = block['type']
        heading = block['heading']
        level = block['level']
        content = block['content']
        size = block['size']
        parent_path = block.get('parent_path', '')
        
        context_prefix = ""
        if parent_path:
            context_prefix = f"{parent_path} > {heading}\n\n" if heading else f"{parent_path}\n\n"
        elif heading:
            context_prefix = f"{heading}\n\n"
        
        if block_type in ['table', 'list']:
            chunk = {
                "text": context_prefix + content if context_prefix else content,
                "heading": heading,
                "level": level,
                "type": block_type,
                "parent_path": parent_path,
                "size": size
            }
            if block_type == 'table' and 'html' in block:
                chunk['html'] = block['html']
            chunks.append(chunk)
            logger.info(f"✓ {block_type.capitalize()} block (size={size} chars): '{heading}' in {parent_path or 'root'}")
            continue
        
        if block_type == 'text':
            if size <= max_size:
                # КРИТИЧНО: Объединяем маленькие blocks под одним heading
                if chunks and chunks[-1].get('heading') == heading and chunks[-1].get('type') == 'text':
                    last_chunk = chunks[-1]
                    last_size = last_chunk.get('size', 0)
                    
                    # Если предыдущий chunk маленький (< 600 chars) и вместе они < max_size - объединяем
                    combined_size = last_size + size + len(context_prefix)
                    if last_size < 600 and combined_size <= max_size:
                        # Объединяем chunks
                        new_content = context_prefix + content if context_prefix else content
                        combined_text = last_chunk['text'] + "\n\n" + new_content
                        last_chunk['text'] = combined_text
                        last_chunk['size'] = len(combined_text)
                        logger.debug(f"📦 Объединены blocks: {last_size} + {size} = {len(combined_text)} chars под heading '{heading}'")
                        continue  # Пропускаем создание нового chunk
                
                # Создаём новый chunk
                chunk = {
                    "text": context_prefix + content if context_prefix else content,
                    "heading": heading,
                    "level": level,
                    "type": block_type,
                    "parent_path": parent_path,
                    "size": size
                }
                chunks.append(chunk)
            else:
                logger.info(f"⚠ Text block too large ({size} > {max_size}), splitting: '{heading}'")
                
                # === НОВОЕ: Использовать RecursiveCharacterTextSplitter если доступен ===
                if SEMANTIC_SPLITTER and PRESERVE_STRUCTURE:
                    try:
                        # Semantic chunking с сохранением структуры
                        text_chunks = SEMANTIC_SPLITTER.split_text(content)
                        
                        for i, chunk_text in enumerate(text_chunks):
                            if len(chunk_text.strip()) < MIN_CHUNK_SIZE:
                                continue
                            
                            chunk = {
                                "text": context_prefix + chunk_text.strip() if context_prefix else chunk_text.strip(),
                                "heading": heading,
                                "level": level,
                                "type": block_type,
                                "parent_path": parent_path,
                                "size": len(chunk_text)
                            }
                            chunks.append(chunk)
                        
                        logger.debug(f"✅ Semantic chunking: {size} chars → {len(text_chunks)} chunks")
                        continue
                    except Exception as e:
                        logger.warning(f"Semantic chunking failed, using fallback: {e}")
                
                # Fallback: existing sentence-based splitting
                import re
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current = ""
                overlap_buffer = ""
                
                for sent in sentences:
                    if len(current) + len(sent) + 1 < max_size:
                        current += sent + " "
                    else:
                        if current.strip():
                            chunk_text = context_prefix + (overlap_buffer + current).strip() if context_prefix else (overlap_buffer + current).strip()
                            chunk = {
                                "text": chunk_text,
                                "heading": heading,
                                "level": level,
                                "type": block_type,
                                "parent_path": parent_path,
                                "size": len(chunk_text)
                            }
                            chunks.append(chunk)
                            overlap_buffer = current[-CHUNK_OVERLAP_SIZE:] if len(current) > CHUNK_OVERLAP_SIZE else current
                        current = sent + " "
                
                if current.strip():
                    chunk_text = context_prefix + (overlap_buffer + current).strip() if context_prefix else (overlap_buffer + current).strip()
                    chunk = {
                        "text": chunk_text,
                        "heading": heading,
                        "level": level,
                        "type": block_type,
                        "parent_path": parent_path,
                        "size": len(chunk_text)
                    }
                    chunks.append(chunk)
    
    return chunks if chunks else []

def html_to_text(html: str, max_len: int = 50000) -> str:
    """
    Конвертация HTML Confluence в plain text с сохранением структуры и макросов.
    
    Args:
        html: HTML контент
        max_len: Максимальная длина для обработки
    
    Returns:
        Plain text или пустая строка при ошибке
    """
    if not html:
        return ""
    try:
        if len(html) > max_len:
            html = html[:max_len]
            logger.warning(f"HTML обрезан до {max_len} символов")
        
        # Предобработка Confluence макросов
        html = preprocess_confluence_macros(html)
        
        h = html2text.HTML2Text()
        # Улучшенные настройки для Confluence
        h.ignore_links = False
        h.body_width = 0
        h.unicode_snob = True
        h.ignore_images = False  # Сохранить ссылки на изображения
        h.ignore_emphasis = False  # Сохранить форматирование (жирный, курсив)
        h.skip_internal_links = False  # Сохранить внутренние ссылки
        h.inline_links = False  # Ссылки в виде [text](url)
        h.mark_code = True  # Отмечать код блоки
        h.wrap_links = False  # Не переносить ссылки
        h.default_image_alt = "[Изображение]"  # Альт для изображений
        
        text = h.handle(html).strip()
        
        # Очистка множественных пустых строк (больше 2 подряд)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    except Exception as e:
        logger.error(f"Ошибка конвертации HTML: {e}")
        return ""

def extract_sections(text: str) -> List[Dict[str, Any]]:
    """
    Извлекает секции документа по заголовкам (markdown формат) с сохранением иерархии.
    
    Args:
        text: Текст в markdown формате (после html2text)
    
    Returns:
        Список секций с заголовками, контентом и родительскими заголовками
    """
    lines = text.split('\n')
    sections = []
    current_section = {"heading": "", "level": 0, "content": [], "parent_headings": []}
    heading_stack = []  # Стек заголовков для отслеживания иерархии
    
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    
    for line in lines:
        match = heading_pattern.match(line)
        if match:
            # Сохраняем предыдущую секцию
            if current_section["content"]:
                sections.append(current_section)
            
            # Начинаем новую секцию
            level = len(match.group(1))
            heading = match.group(2).strip()
            
            # Обновляем стек заголовков: удаляем все заголовки того же или более низкого уровня
            heading_stack = [h for h in heading_stack if h['level'] < level]
            
            # Создаем новую секцию с родительскими заголовками
            parent_headings = [h['text'] for h in heading_stack]
            current_section = {
                "heading": heading, 
                "level": level, 
                "content": [line],
                "parent_headings": parent_headings
            }
            
            # Добавляем текущий заголовок в стек
            heading_stack.append({'level': level, 'text': heading})
        else:
            current_section["content"].append(line)
    
    # Сохраняем последнюю секцию
    if current_section["content"]:
        sections.append(current_section)
    
    return sections

def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[Dict[str, Any]]:
    """
    Умное разбиение текста на семантические чанки с учётом заголовков.
    
    Args:
        text: Исходный текст (markdown после html2text)
        size: Максимальный размер чанка
    
    Returns:
        Список чанков с metadata (heading, content)
    """
    if not text or len(text) < 100:
        return [{"text": text, "heading": "", "level": 0}] if text else []
    
    # Извлекаем секции по заголовкам
    sections = extract_sections(text)
    
    if not sections:
        # Fallback: разбиение по параграфам
        paras = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 5]
        chunks = []
        current = ""
        for para in paras:
            if len(current) + len(para) + 2 < size:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append({"text": current.strip(), "heading": "", "level": 0})
                current = para + "\n\n"
        if current.strip():
            chunks.append({"text": current.strip(), "heading": "", "level": 0})
        return chunks if chunks else [{"text": text, "heading": "", "level": 0}]
    
    # Разбиваем секции на чанки
    chunks = []
    for section in sections:
        heading = section["heading"]
        level = section["level"]
        parent_headings = section.get("parent_headings", [])
        content = '\n'.join(section["content"])
        
        # Формируем префикс из родительских заголовков (только для уровня 3+)
        context_prefix = ""
        if level >= 3 and parent_headings:
            # Добавляем родительские заголовки для контекста
            context_prefix = " > ".join(parent_headings) + "\n\n"
        
        # Если секция целиком влезает в чанк
        if len(content) <= size:
            chunk_text = context_prefix + content.strip() if context_prefix else content.strip()
            chunks.append({
                "text": chunk_text,
                "heading": heading,
                "level": level
            })
        else:
            # Разбиваем большую секцию по параграфам, сохраняя заголовок
            paras = [p.strip() for p in content.split('\n\n') if p.strip()]
            current = ""
            for para in paras:
                if len(current) + len(para) + 2 < size:
                    current += para + "\n\n"
                else:
                    if current.strip():
                        chunk_text = context_prefix + current.strip() if context_prefix else current.strip()
                        chunks.append({
                            "text": chunk_text,
                            "heading": heading,
                            "level": level
                        })
                    current = para + "\n\n"
            if current.strip():
                chunk_text = context_prefix + current.strip() if context_prefix else current.strip()
                chunks.append({
                    "text": chunk_text,
                    "heading": heading,
                    "level": level
                })
    
    return chunks if chunks else [{"text": text, "heading": "", "level": 0}]

def load_state() -> Dict[str, Any]:
    """
    Загрузка состояния синхронизации из файла.
    
    Returns:
        Словарь с last_sync и pages
    """
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
                logger.info(f"Загружено состояние: {len(state.get('pages', {}))} страниц")
                return state
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга state file: {e}")
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {e}")
    
    logger.info("Создание нового состояния")
    return {"last_sync": 0, "pages": {}}

def save_state(state: Dict[str, Any]) -> None:
    """
    Сохранение состояния синхронизации в файл.
    
    Args:
        state: Словарь с данными состояния
    """
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.debug(f"Состояние сохранено: {len(state.get('pages', {}))} страниц")
    except Exception as e:
        logger.error(f"Ошибка сохранения состояния: {e}")

# init_embeddings() теперь импортируется из embeddings.py как get_embed_model()

def get_all_pages_generator(confluence: Confluence, space_key: str, batch_size: int = 50):
    """
    Генератор для потокового получения всех страниц пространства.
    
    Вместо загрузки ВСЕ страницы в памяти, загружает по батчам.
    Экономит RAM на больших пространствах (1000+ страниц).
    
    Args:
        confluence: Confluence клиент
        space_key: Ключ пространства (RAUII, Surveys)
        batch_size: Размер батча для получения
        
    Yields:
        Dict: Информация о странице
    """
    start = 0
    total_yielded = 0
    
    while True:
        try:
            logger.debug(f"Fetching pages from {space_key} starting at {start}")
            
            # Получаем батч
            batch = list(confluence.get_all_pages_from_space(
                space_key,
                start=start,
                limit=batch_size,
                expand='history.lastUpdated,version.number'
            ))
            
            if not batch:
                logger.info(f"No more pages for {space_key}. Total yielded: {total_yielded}")
                break
            
            # Выдаем по одной странице
            for page in batch:
                yield page
                total_yielded += 1
            
            start += batch_size
            
        except Exception as e:
            logger.error(f"Error fetching pages for {space_key}: {e}")
            break

# ✅ NEW: Simple Bloom Filter для duplicate detection (опционально)
USE_BLOOM_FILTER = os.getenv("USE_BLOOM_FILTER", "false").lower() == "true"
BLOOM_FILTER_SIZE = int(os.getenv("BLOOM_FILTER_SIZE", "100000"))  # Ожидаемое количество элементов

try:
    from pybloom_live import BloomFilter
    HAS_BLOOM_FILTER = True
except ImportError:
    HAS_BLOOM_FILTER = False
    if USE_BLOOM_FILTER:
        logger.warning("pybloom_live not available, falling back to set() for duplicate detection")


class BatchProcessor:
    """Обработчик батчей с recovery механизмом"""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        # ✅ ИСПРАВЛЕНИЕ: Используем Bloom Filter для больших объемов данных
        if USE_BLOOM_FILTER and HAS_BLOOM_FILTER:
            self.processed_ids = BloomFilter(capacity=BLOOM_FILTER_SIZE, error_rate=0.001)
            self._use_bloom = True
            logger.debug(f"Using Bloom Filter for duplicate detection (capacity={BLOOM_FILTER_SIZE})")
        else:
            self.processed_ids = set()  # Fallback на set() для малых объемов
            self._use_bloom = False
        self.failed_ids = {}  # Неудачные с причиной
    
    def process_batch_safe(self, qdrant_client: Any, confluence: Confluence, 
                          batch: List[Dict[str, Any]], state: Dict[str, Any], space_key: str) -> tuple[int, int, int, list]:
        """
        Процессировать батч с гарантией целостности.
        
        Если ошибка → откатывает только эту страницу, не весь батч.
        """
        updated, errors, skipped = 0, 0, 0
        error_details = []
        
        for page in batch:
            page_id = str(page.get('id', ''))
            if not page_id:
                skipped += 1
                continue
            
            title = page.get('title', 'Unknown')
            ts = get_timestamp(page)
            
            # ✅ ИСПРАВЛЕНИЕ: Пропускаем если уже обработана (Bloom Filter или set())
            if page_id in self.processed_ids:
                skipped += 1
                continue
            
            # Пропускаем если не изменилась
            if page_id in state['pages'] and state['pages'][page_id].get('updated') == ts:
                skipped += 1
                continue
            
            try:
                # Пытаемся обработать с retry
                for attempt in range(self.max_retries):
                    try:
                        # Получаем полную страницу (с кэшированием)
                        page_data = get_page_cached(confluence, page_id)
                        
                        # Извлечение расширенных метаданных
                        page_metadata = extract_page_metadata(page_data, space_key=space_key)
                        page_metadata['attachments'] = get_page_attachments(confluence, page_id)
                        
                        html = page_data.get('body', {}).get('storage', {}).get('value', '')
                        if not html or len(html) < MIN_TEXT_LEN:
                            skipped += 1
                            break
                        
                        # Структурная нарезка
                        blocks = extract_structural_blocks(html)
                        if not blocks:
                            skipped += 1
                            break
                        
                        chunks = smart_chunk_with_context(blocks, max_size=CHUNK_SIZE)
                        
                        # Конвертируем HTML в markdown
                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        h.ignore_images = False
                        content_markdown = h.handle(html)
                        
                        # Сохраняем в PostgreSQL
                        try:
                            version_when = page_data.get('version', {}).get('when', '')
                            if version_when:
                                if version_when.endswith('Z'):
                                    version_when = version_when[:-1] + '+00:00'
                                updated_at = datetime.fromisoformat(version_when)
                            else:
                                updated_at = datetime.now()
                        except Exception as e:
                            logger.debug(f"Ошибка парсинга даты для {page_id}: {e}")
                            updated_at = datetime.now()
                        
                        if not save_page_to_postgres(
                            page_id=page_id,
                            space_key=space_key,
                            title=title,
                            content_html=html,
                            content_markdown=content_markdown,
                            version=page_data.get('version', {}).get('number', 1),
                            metadata=page_metadata,
                            updated_at=updated_at
                        ):
                            logger.warning(f"Не удалось сохранить страницу {page_id} в PostgreSQL")
                            skipped += 1
                            break
                        
                        # Удаляем старые чанки из Qdrant
                        try:
                            delete_points_by_page_id(page_id)
                        except Exception as e:
                            logger.warning(f"Ошибка удаления старых чанков для {page_id}: {e}")
                        
                        # Индексируем чанки в Qdrant
                        for chunk_idx, chunk in enumerate(chunks):
                            chunk_text = chunk.get('text', '')
                            if not chunk_text:
                                continue
                            
                            # Санитизация метаданных перед сохранением
                            # КРИТИЧНО: Добавляем text в metadata, т.к. LlamaIndex QdrantVectorStore 
                            # не сохраняет Document.text в payload автоматически
                            sanitized_metadata = sanitize_metadata({
                                'text': chunk_text,  # КРИТИЧНО: текст должен быть в metadata для сохранения в payload
                                'page_id': page_id,
                                'chunk': chunk_idx,
                                'chunk_type': chunk.get('type', 'text'),
                                'heading': chunk.get('heading', '')[:200],  # Обрезаем заголовки
                                'heading_level': chunk.get('level', 0),
                                'parent_path': chunk.get('parent_path', '')[:200],
                                'space': space_key,
                                'title': title[:200] if title else '',  # Обрезаем title
                                'url': f"{CONFLUENCE_URL}pages/viewpage.action?pageId={page_id}",
                                # Включаем только необходимые поля из page_metadata
                                'created': page_metadata.get('created', ''),
                                'modified': page_metadata.get('modified', ''),
                                'author': page_metadata.get('author', ''),
                                'space_key': page_metadata.get('space_key', space_key),
                                # НЕ включаем полный HTML, breadcrumb, headings_list и т.д.!
                            })
                            
                            # Генерируем embedding для chunk
                            embedding = generate_query_embedding(chunk_text)
                            point_id = f"{page_id}_{chunk_idx}"
                            
                            # Вставляем напрямую в Qdrant
                            success = insert_chunk_to_qdrant(
                                client=qdrant_client,
                                chunk_text=chunk_text,
                                metadata=sanitized_metadata,
                                embedding=embedding,
                                point_id=point_id
                            )
                            if not success:
                                logger.warning(f"Failed to insert chunk {point_id} for page {page_id}")
                        
                        # Обновляем состояние
                        state['pages'][page_id] = {'updated': ts, 'title': title}
                        
                        # Помечаем как проиндексированную в PostgreSQL
                        mark_as_indexed(page_id)
                        
                        # ✅ Добавляем в Bloom Filter или set()
                        if self._use_bloom:
                            self.processed_ids.add(page_id)
                        else:
                            self.processed_ids.add(page_id)
                        updated += 1
                        break  # Успешно обработано
                        
                    except Exception as retry_error:
                        if attempt < self.max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.warning(f"Retry {attempt+1}/{self.max_retries} for page {page_id}: {retry_error}. Waiting {wait_time}s...")
                            time.sleep(wait_time)  # Exponential backoff
                        else:
                            raise
                
            except Exception as e:
                # Ошибка на странице, но продолжаем батч
                error_msg = f"Не удалось обработать страницу {page_id} ({title}): {e}"
                logger.error(error_msg)
                self.failed_ids[page_id] = str(e)
                errors += 1
                error_details.append({
                    "page_id": page_id,
                    "title": title,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return updated, errors, skipped, error_details
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику обработки"""
        # ✅ Для Bloom Filter используем приблизительный размер
        if self._use_bloom:
            processed_count = self.processed_ids.count  # Приблизительное количество
        else:
            processed_count = len(self.processed_ids)
        
        return {
            "processed": processed_count,
            "failed": len(self.failed_ids),
            "failed_pages": self.failed_ids,
            "using_bloom_filter": self._use_bloom
        }

def process_batch(qdrant_client: Any, confluence: Confluence, 
                  pages: List[Dict[str, Any]], state: Dict[str, Any], space_key: str) -> tuple[int, int, int, list]:
    """
    Обработка batch страниц Confluence.
    Сохраняет страницы в PostgreSQL, затем индексирует в Qdrant.
    
    Args:
        index: Qdrant VectorStoreIndex
        qdrant_client: Qdrant клиент
        confluence: Confluence API client
        pages: Список страниц для обработки
        state: Состояние синхронизации
        space_key: Ключ пространства
    
    Returns:
        Tuple[updated, errors, skipped, error_details] - статистика обработки
    """
    # Тестовый режим
    if TEST_MODE:
        pages = pages[:TEST_MAX_PAGES]
        logger.info(f"🧪 TEST MODE ENABLED - Processing only first {len(pages)} pages")
    
    updated, errors, skipped = 0, 0, 0
    error_details = []  # Список деталей ошибок
    for page in pages:
        pid = str(page.get('id', ''))
        if not pid:
            skipped += 1
            continue
        title = page.get('title', 'Unknown')
        ts = get_timestamp(page)
        try:
            if pid in state['pages'] and state['pages'][pid].get('updated') == ts:
                skipped += 1
                continue
            try:
                page_data = get_page(confluence, pid)
            except Exception as e:
                error_msg = f"Не удалось получить страницу {pid} ({title}): {e}"
                logger.warning(error_msg)
                error_details.append(error_msg)
                skipped += 1
                continue
            
            # Извлечение расширенных метаданных
            page_metadata = extract_page_metadata(page_data, space_key=space_key)
            
            # Получаем список вложений
            page_metadata['attachments'] = get_page_attachments(confluence, pid)
            
            html = page_data.get('body', {}).get('storage', {}).get('value', '')
            if not html or len(html) < MIN_TEXT_LEN:
                skipped += 1
                continue
            
            # Структурная нарезка
            blocks = extract_structural_blocks(html)
            if not blocks:
                skipped += 1
                continue
            
            chunks = smart_chunk_with_context(blocks, max_size=CHUNK_SIZE)
            
            # Конвертируем HTML в markdown для хранения
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            content_markdown = h.handle(html)
            
            # Сохраняем страницу в PostgreSQL
            try:
                version_when = page_data.get('version', {}).get('when', '')
                if version_when:
                    # Парсим ISO формат даты
                    if version_when.endswith('Z'):
                        version_when = version_when[:-1] + '+00:00'
                    updated_at = datetime.fromisoformat(version_when)
                else:
                    updated_at = datetime.now()
            except Exception as e:
                logger.debug(f"Ошибка парсинга даты для {pid}: {e}, использую текущую дату")
                updated_at = datetime.now()
            
            if not save_page_to_postgres(
                page_id=pid,
                space_key=space_key,
                title=title,
                content_html=html,
                content_markdown=content_markdown,
                version=page_data.get('version', {}).get('number', 1),
                metadata=page_metadata,
                updated_at=updated_at
            ):
                logger.warning(f"Не удалось сохранить страницу {pid} в PostgreSQL")
                skipped += 1
                continue
            
            # Удаляем старые чанки из Qdrant
            try:
                delete_points_by_page_id(pid)
            except Exception as e:
                logger.debug(f"Не удалось удалить старые чанки для {pid}: {e}")
            
            # Собираем все chunks для batch вставки
            chunks_to_insert = []
            page_url = f"{CONFLUENCE_URL.rstrip('/')}/wiki/spaces/{space_key}/pages/{pid}"
            
            # Полные метаданные: базовые + заголовки + Confluence метаданные
            labels_list = page_metadata.get('labels', [])
            labels_str = ",".join(labels_list) if labels_list else ""
            
            attachments_list = page_metadata.get('attachments', [])
            attachments_str = ",".join(attachments_list) if attachments_list else ""
            
            # Ограничиваем длину строковых полей (для совместимости с Qdrant)
            max_str_len = 500
            title_safe = title[:max_str_len] if title else "Unknown"
            parent_safe = page_metadata.get('parent_title', '')[:max_str_len]
            author_safe = page_metadata.get('created_by', '')[:max_str_len]
            attachments_safe = attachments_str[:max_str_len]
            page_path_safe = page_metadata.get('page_path', '')[:max_str_len]
            breadcrumb_safe = page_metadata.get('breadcrumb', '')[:max_str_len]
            
            for i, chunk_data in enumerate(chunks):
                # chunk_data теперь словарь с ключами: text, heading, level
                if not isinstance(chunk_data, dict):
                    logger.warning(f"Unexpected chunk_data type: {type(chunk_data)}")
                    continue
                
                chunk_content = chunk_data.get("text", "")
                if not chunk_content or len(chunk_content) < 20:
                    continue
                
                heading_safe = chunk_data.get("heading", "")[:max_str_len]
                labels_safe = labels_str[:max_str_len]
                parent_path_safe = chunk_data.get("parent_path", "")[:max_str_len]
                
                # Обогащенные метаданные
                block_type = chunk_data.get("type", "text")
                block_size = chunk_data.get("size", 0)
                is_complete = block_type in ["table", "list"]
                heading_path = (parent_path_safe + " > " + heading_safe if parent_path_safe else heading_safe)[:max_str_len]
                
                metadata = {
                    # Базовые
                    "page_id": pid,
                    "chunk": i,
                    "title": title_safe,
                    "space": space_key,
                    "url": page_url,
                    # Структура документа
                    "heading": heading_safe,
                    "heading_level": chunk_data.get("level", 0),
                    # НОВЫЕ ПОЛЯ для структурной нарезки
                    "type": block_type,                          # table|list|text (тип блока)
                    "parent_path": parent_path_safe,             # Иерархия заголовков
                    "block_size": block_size,                    # Размер блока
                    "is_complete_block": is_complete,            # Целый блок или часть
                    "has_table": block_type == "table",          # Содержит таблицу
                    "heading_path": heading_path,                # Полный путь
                    # Confluence метаданные
                    "labels": labels_safe,
                    "parent_title": parent_safe,                 # Родительская страница (ближайший)
                    "page_path": page_path_safe[:200] if page_path_safe else '',  # URL-friendly путь (ограниченный)
                    "breadcrumb": breadcrumb_safe[:200] if breadcrumb_safe else '',  # Полный путь с разделителями (ограниченный)
                    "created_by": author_safe,
                    "has_children": page_metadata.get('has_children', False),
                    "version": page_metadata.get('version', 1),
                    "attachments": attachments_safe[:10] if attachments_safe else [],  # Ограничиваем список вложений
                    # НОВЫЕ ПОЛЯ для metadata filtering (из extract_page_metadata):
                    "status": page_metadata.get('status', 'current'),  # current, archived, draft
                    "content_type": page_metadata.get('type', 'page'),  # page, blogpost, attachment
                    "hierarchy_depth": page_metadata.get('hierarchy_depth', 0),
                    "created": page_metadata.get('created', ''),
                    "modified": page_metadata.get('modified', ''),
                    "modified_by": page_metadata.get('modified_by', ''),
                    "children_count": page_metadata.get('children_count', 0),
                    # === ПУТЬ И ЗАГОЛОВКИ (ограниченные) ===
                    "headings": (page_metadata.get('headings', '') or '')[:500],  # Обрезаем headings
                    "headings_list": (page_metadata.get('headings_list', []) or [])[:10],  # Ограничиваем список
                    "heading_count": page_metadata.get('heading_count', 0)
                }
                
                # Санитизация метаданных перед сохранением
                # КРИТИЧНО: Добавляем text в metadata, т.к. LlamaIndex QdrantVectorStore 
                # не сохраняет Document.text в payload автоматически
                metadata['text'] = chunk_content  # КРИТИЧНО: текст должен быть в metadata для сохранения в payload
                sanitized_metadata = sanitize_metadata(metadata)
                
                # Генерируем embedding для chunk
                embedding = generate_query_embedding(chunk_content)
                point_id = f"{pid}_{i}"
                
                chunks_to_insert.append({
                    'text': chunk_content,
                    'metadata': sanitized_metadata,
                    'embedding': embedding,
                    'point_id': point_id
                })
            
            # Batch вставка (если chunks > 10) или single
            inserted = 0
            if len(chunks_to_insert) > 10:
                success_count, error_count = insert_chunks_batch_to_qdrant(
                    client=qdrant_client,
                    chunks_data=chunks_to_insert,
                    batch_size=100
                )
                inserted = success_count
                if error_count > 0:
                    logger.warning(f"Failed to insert {error_count} chunks for page {pid}")
            else:
                for chunk_data in chunks_to_insert:
                    success = insert_chunk_to_qdrant(
                        client=qdrant_client,
                        chunk_text=chunk_data['text'],
                        metadata=chunk_data['metadata'],
                        embedding=chunk_data['embedding'],
                        point_id=chunk_data['point_id']
                    )
                    if success:
                        inserted += 1
            if inserted > 0:
                state['pages'][pid] = {'updated': ts, 'chunks': inserted}
                # Помечаем страницу как проиндексированную в PostgreSQL
                mark_as_indexed(pid)
                updated += 1
            else:
                skipped += 1
        except Exception as e:
            import traceback
            error_msg = f"Error processing page {pid} ({title}): {e}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            error_details.append(error_msg)
            errors += 1
    return updated, errors, skipped, error_details

def get_blogposts_from_space(confluence: Confluence, space_key: str, start: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Получение blog posts из space (обёртка для API).
    
    Args:
        confluence: Confluence API client
        space_key: Ключ пространства
        start: Начальная позиция
        limit: Количество постов
    
    Returns:
        Список blog posts
    """
    try:
        # API запрос для блогов
        url = f"{confluence.url}/rest/api/content?type=blogpost&spaceKey={space_key}&start={start}&limit={limit}&expand=version"
        response = requests.get(url, headers=confluence.default_headers, verify=VERIFY_SSL)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except Exception as e:
        logger.debug(f"Ошибка получения blogposts для {space_key}: {e}")
        return []

def cleanup_deleted_pages(qdrant_client: Any, state: Dict[str, Any], current_page_ids: set) -> int:
    """
    Удаление документов из индекса для страниц, которые были удалены в Confluence.
    
    Args:
        qdrant_client: Qdrant клиент
        state: Текущее состояние синхронизации
        current_page_ids: Набор ID страниц, которые существуют в Confluence
    
    Returns:
        Количество удалённых страниц
    """
    # Используем функцию из postgres_storage для очистки PostgreSQL
    deleted_count = cleanup_deleted_pages_postgres(current_page_ids)
    
    # Удаляем из Qdrant через delete_points_by_page_id
    state_page_ids = set(state.get('pages', {}).keys())
    deleted_page_ids = state_page_ids - current_page_ids
    
    if not deleted_page_ids:
        logger.debug("Нет удалённых страниц для очистки")
        return deleted_count
    
    logger.info(f"Обнаружено {len(deleted_page_ids)} удалённых страниц в Confluence")
    
    for page_id in deleted_page_ids:
        try:
            # Удаление всех чанков этой страницы из Qdrant
            if delete_points_by_page_id(page_id):
                logger.info(f"  Удалена страница {page_id} из Qdrant")
            
            # Удаление из state
            if page_id in state['pages']:
                del state['pages'][page_id]
        except Exception as e:
            logger.error(f"Ошибка удаления страницы {page_id}: {e}")
    
    return deleted_count

def sync() -> None:
    """Основной процесс синхронизации Confluence с PostgreSQL + Qdrant."""
    logger.info("Sync started")
    state = load_state()
    start_time = time.time()
    try:
        # Примечание: параметр verify_ssl не поддерживается в текущей версии atlassian-python-api
        confluence = Confluence(url=CONFLUENCE_URL, token=CONFLUENCE_TOKEN)
        logger.info("Connected to Confluence")
    except Exception as e:
        logger.error(f"Confluence error: {e}")
        return
    try:
        from embeddings import get_embedding_dimension
        
        logger.info("Шаг 1: Инициализация PostgreSQL...")
        init_postgres_schema()
        logger.info("✅ PostgreSQL schema initialized")
        
        logger.info("Шаг 2: Загрузка embedding модели (может занять ~60 сек)...")
        embed_model = get_embed_model()
        logger.info(f"✅ Модель загружена: {type(embed_model)}")
        
        model_dim = get_embedding_dimension()
        
        logger.info("Шаг 3: Инициализация Qdrant...")
        qdrant_client = init_qdrant_client()
        
        # Инициализируем Qdrant коллекцию с проверкой размерности
        if not init_qdrant_collection(model_dim):
            raise ValueError(
                f"Не удалось инициализировать Qdrant коллекцию. "
                f"Проверьте размерность модели: {model_dim}D"
            )
        
        logger.info(f"✅ Размерность embeddings: {model_dim}D")
        logger.info("✅ Qdrant client ready")
        
        # Проверяем количество документов
        doc_count = get_qdrant_count()
        logger.info(f"Текущее количество документов в Qdrant: {doc_count}")
        
    except ValueError as ve:
        # Это ошибка несовпадения размерности - не продолжаем
        logger.error(f"Sync остановлен: {ve}")
        return
    except Exception as e:
        logger.error(f"Init error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
    total_updated, total_errors, total_skipped = 0, 0, 0
    current_page_ids = set()  # Для отслеживания существующих страниц
    
    # Словарь для статистики по пространствам
    space_stats = {}
    
    try:
        all_spaces = confluence.get_all_spaces().get('results', [])
        
        # Фильтрация пространств
        if CONFLUENCE_SPACES:
            # Парсим список пространств из ENV (через запятую)
            target_spaces = [s.strip().upper() for s in CONFLUENCE_SPACES.split(',') if s.strip()]
            spaces = [s for s in all_spaces if s.get('key', '').upper() in target_spaces]
            
            # Предупреждение, если оба параметра указаны
            if MAX_SPACES != 10:  # Если MAX_SPACES изменен от значения по умолчанию
                logger.warning(f"⚠️  Указаны оба параметра: CONFLUENCE_SPACES и MAX_SPACES={MAX_SPACES}")
                logger.warning(f"   Используется CONFLUENCE_SPACES (MAX_SPACES игнорируется)")
            
            logger.info(f"Фильтр пространств: {len(spaces)} из {len(all_spaces)} (указаны: {CONFLUENCE_SPACES})")
            
            # Предупреждение, если некоторые указанные пространства не найдены
            found_keys = {s.get('key', '').upper() for s in spaces}
            not_found = [t for t in target_spaces if t not in found_keys]
            if not_found:
                logger.warning(f"⚠️  Пространства не найдены в Confluence: {', '.join(not_found)}")
        else:
            # Старое поведение: MAX_SPACES
            spaces = all_spaces[:MAX_SPACES]
            logger.info(f"Processing {len(spaces)} spaces (MAX_SPACES={MAX_SPACES})")
        
        # Используем tqdm если доступен
        if TQDM_AVAILABLE:
            spaces_iter = tqdm(spaces, desc="Syncing spaces", unit="space")
        else:
            spaces_iter = spaces
        
        for space in spaces_iter:
            key = space.get('key', '')
            if not key:
                continue
            
            space_name = space.get('name', key)
            
            # Обновляем описание progress bar
            if TQDM_AVAILABLE:
                spaces_iter.set_description(f"Syncing {space_name}")
            
            # Инициализация статистики для пространства
            space_stats[key] = {
                'total_pages': 0,
                'total_blogs': 0,
                'processed': 0,
                'updated': 0,
                'skipped': 0,
                'errors': 0,
                'chunks_created': 0,
                'error_details': []
            }
            
            logger.info(f"📂 {space_name}:")
            try:
                # Используем генератор вместо загрузки всех страниц
                all_pages = []
                for page in get_all_pages_generator(confluence, key, batch_size=BATCH_SIZE):
                    all_pages.append(page)
                    page_id = str(page.get('id', ''))
                    if page_id:
                        current_page_ids.add(page_id)
                
                space_stats[key]['total_pages'] = len(all_pages)
                logger.info(f"   Страниц найдено: {len(all_pages)} (блогов: 0)")
                
                if not all_pages:
                    logger.info(f"   Обработано: 0")
                    logger.info(f"   Обновлено: 0 | Пропущено: 0 | Ошибок: 0")
                    logger.info(f"   Chunks создано: 0")
                    continue
                
                # Разбиваем на батчи
                batches = [all_pages[i:i+BATCH_SIZE] for i in range(0, len(all_pages), BATCH_SIZE)]
                
                # Параллельный процессинг с ThreadPoolExecutor
                max_workers = get_int_env("PARALLEL_SYNC_MAX_WORKERS", 4)
                batch_processor = BatchProcessor(max_retries=3)
                
                if TQDM_AVAILABLE:
                    batches_iter = tqdm(batches, desc=f"  Processing {space_name}", unit="batch", leave=False)
                else:
                    batches_iter = batches
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Отправляем все батчи на обработку
                    future_to_batch = {
                        executor.submit(
                            batch_processor.process_batch_safe,
                            qdrant_client, confluence, batch, state, key
                        ): batch
                        for batch in batches
                    }
                    
                    # Собираем результаты
                    for future in as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        try:
                            batch_updated, batch_errors, batch_skipped, batch_error_details = future.result()
                            total_updated += batch_updated
                            total_errors += batch_errors
                            total_skipped += batch_skipped
                            space_stats[key]['updated'] += batch_updated
                            space_stats[key]['errors'] += batch_errors
                            space_stats[key]['skipped'] += batch_skipped
                            space_stats[key]['processed'] += len(batch)
                            space_stats[key]['error_details'].extend(batch_error_details)
                            
                            if TQDM_AVAILABLE:
                                batches_iter.update(1)
                                batches_iter.set_postfix({
                                    "updated": space_stats[key]['updated'],
                                    "errors": space_stats[key]['errors']
                                })
                        except Exception as e:
                            logger.error(f"Batch processing failed: {e}")
                            total_errors += 1
                            space_stats[key]['errors'] += 1
                            if TQDM_AVAILABLE:
                                batches_iter.update(1)
                
                # Логируем статистику пространства
                logger.info(f"   Обработано: {space_stats[key]['processed']}")
                logger.info(f"   Обновлено: {space_stats[key]['updated']} | Пропущено: {space_stats[key]['skipped']} | Ошибок: {space_stats[key]['errors']}")
                
                # Подсчет chunks (упрощенная версия)
                chunks_count = 0
                # Можно добавить более точный подсчет из Qdrant если нужно
                space_stats[key]['chunks_created'] = chunks_count
                logger.info(f"   Chunks создано: {chunks_count}")
                
                # Логируем статистику кэша
                cache_stats = get_cache_stats()
                logger.debug(f"   Cache stats: {cache_stats}")
                
                # Обработка blog posts (аналогично, но без параллельности для простоты)
                try:
                    all_blogs = []
                    blog_start = 0
                    while True:
                        batch_blogs = get_blogposts_from_space(confluence, key, start=blog_start, limit=BATCH_SIZE)
                        if not batch_blogs:
                            break
                        all_blogs.extend(batch_blogs)
                        blog_start += BATCH_SIZE
                    
                    for blog in all_blogs:
                        blog_id = str(blog.get('id', ''))
                        if blog_id:
                            current_page_ids.add(blog_id)
                    
                    space_stats[key]['total_blogs'] = len(all_blogs)
                    if all_blogs:
                        logger.info(f"   Блогов найдено: {len(all_blogs)}")
                        # Обработка блогов (можно добавить параллельность позже)
                        for i in range(0, len(all_blogs), BATCH_SIZE):
                            batch = all_blogs[i:i+BATCH_SIZE]
                            updated, errors, skipped, error_details = batch_processor.process_batch_safe(
                                qdrant_client, confluence, batch, state, key
                            )
                            total_updated += updated
                            total_errors += errors
                            total_skipped += skipped
                            space_stats[key]['updated'] += updated
                            space_stats[key]['errors'] += errors
                            space_stats[key]['skipped'] += skipped
                            space_stats[key]['processed'] += len(batch)
                            space_stats[key]['error_details'].extend(error_details)
                except Exception as blog_err:
                    logger.warning(f"Error processing blogs for {key}: {blog_err}")
                    space_stats[key]['error_details'].append(f"Blog processing error: {str(blog_err)}")
                    
            except Exception as e:
                logger.error(f"Space error for {key}: {e}")
                total_errors += 1
                space_stats[key]['errors'] += 1
                space_stats[key]['error_details'].append(f"Space processing error: {str(e)}")
            
            # Обновляем postfix для progress bar
            if TQDM_AVAILABLE:
                spaces_iter.set_postfix({
                    "updated": total_updated,
                    "errors": total_errors
                })
    except Exception as e:
        logger.error(f"Critical: {e}")
        return
    
    # Подсчет chunks для каждого пространства
    try:
        from qdrant_storage import get_points_by_filter
        for space_key in space_stats.keys():
            space_data = get_points_by_filter(filter_dict={"space": space_key}, limit=10000)
            space_stats[space_key]['chunks_created'] = len(space_data.get('ids', []))
    except Exception as e:
        logger.warning(f"Не удалось подсчитать chunks по пространствам: {e}")
    
    # Очистка удалённых страниц
    deleted_count = cleanup_deleted_pages(qdrant_client, state, current_page_ids)
    
    # TODO: Реализовать извлечение доменных терминов из Qdrant
    # (ранее было для ChromaDB, требует адаптации для Qdrant)
    
    state['last_sync'] = int(time.time())
    save_state(state)
    elapsed = time.time() - start_time
    
    # ============ ДЕТАЛЬНАЯ СТАТИСТИКА ПО ПРОСТРАНСТВАМ ============
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 ИТОГИ СИНХРОНИЗАЦИИ")
    logger.info("=" * 80)
    logger.info(f"⏱  Время выполнения: {elapsed:.1f}с ({elapsed/60:.1f} мин)")
    logger.info("")
    logger.info("📁 СТАТИСТИКА ПО ПРОСТРАНСТВАМ:")
    logger.info("-" * 80)
    
    for space_key, stats in sorted(space_stats.items()):
        logger.info(f"  📂 {space_key}:")
        logger.info(f"     Страниц найдено: {stats['total_pages']} (блогов: {stats['total_blogs']})")
        logger.info(f"     Обработано: {stats['processed']}")
        logger.info(f"     Обновлено: {stats['updated']} | Пропущено: {stats['skipped']} | Ошибок: {stats['errors']}")
        logger.info(f"     Chunks создано: {stats['chunks_created']}")
        if stats['error_details']:
            logger.warning(f"     ⚠️  Ошибки ({len(stats['error_details'])}):")
            for err in stats['error_details'][:5]:  # Показываем первые 5 ошибок
                logger.warning(f"        - {err}")
            if len(stats['error_details']) > 5:
                logger.warning(f"        ... и еще {len(stats['error_details']) - 5} ошибок")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("📈 ОБЩАЯ СТАТИСТИКА:")
    logger.info(f"   ✅ Обновлено: {total_updated}")
    logger.info(f"   ⏭  Пропущено: {total_skipped}")
    logger.info(f"   ❌ Ошибок: {total_errors}")
    logger.info(f"   🗑  Удалено: {deleted_count}")
    logger.info(f"   ⏱  Время: {elapsed:.1f}с")
    
    # Статистика PostgreSQL и Qdrant
    try:
        pg_stats = get_postgres_stats()
        qdrant_count = get_qdrant_count()
        logger.info("")
        logger.info("📊 СТАТИСТИКА ХРАНИЛИЩ:")
        logger.info(f"   PostgreSQL: {pg_stats['total_pages']} страниц ({pg_stats['not_indexed']} не проиндексировано)")
        logger.info(f"   Qdrant: {qdrant_count} документов")
    except Exception as e:
        logger.warning(f"Не удалось получить статистику хранилищ: {e}")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    sync()
    logger.info(f"Синхронизация будет повторяться каждые {SYNC_INTERVAL} секунд ({SYNC_INTERVAL/3600:.1f} часов)")
    while True:
        try:
            time.sleep(SYNC_INTERVAL)
            sync()
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки, завершение работы...")
            break
        except Exception as e:
            logger.error(f"Критическая ошибка в главном цикле: {e}")
            time.sleep(60)
