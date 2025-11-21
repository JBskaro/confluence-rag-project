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

from collections import OrderedDict
from threading import RLock

import html2text
import requests
import urllib3
from atlassian import Confluence
# llama-index импорты удалены - используем прямую работу с Qdrant
# QdrantVectorStore используется только в qdrant_storage.py для поиска (get_qdrant_vector_store)
from qdrant_storage import (
    insert_chunks_batch_to_qdrant,
    init_qdrant_client
)
from embeddings import generate_query_embedding, generate_query_embeddings_batch
from tenacity import retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup, NavigableString
import html  # ИСПРАВЛЕНО: Для декодирования HTML entities

# Инициализация logger (должен быть до использования)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===


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


def get_bool_env(name: str, default: bool = False) -> bool:
    """Безопасное получение boolean ENV переменной."""
    value = os.getenv(name, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


MAX_SPACES = get_int_env("MAX_SPACES", 10)
CONFLUENCE_SPACES = os.getenv("CONFLUENCE_SPACES", "").strip()
MAX_CHUNK_SIZE = get_int_env("MAX_CHUNK_SIZE", 1200)
MIN_TEXT_LEN = get_int_env("MIN_TEXT_LEN", 50)
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
BATCH_SIZE = get_int_env("BATCH_SIZE", 50)
BATCH_INSERT_THRESHOLD = get_int_env("BATCH_INSERT_THRESHOLD", 10)
SYNC_INTERVAL = get_int_env("SYNC_INTERVAL", 3600)
MAX_TABLE_SIZE = get_int_env("MAX_TABLE_SIZE", 2048)
CHUNK_OVERLAP = get_int_env("CHUNK_OVERLAP", 100)

# === SEMANTIC CHUNKING CONFIGURATION ===
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', str(MAX_CHUNK_SIZE)))
CHUNK_OVERLAP_SIZE = int(os.getenv('CHUNK_OVERLAP', str(CHUNK_OVERLAP)))
MIN_CHUNK_SIZE = int(os.getenv('MIN_CHUNK_SIZE', '100'))
PRESERVE_STRUCTURE = os.getenv('PRESERVE_STRUCTURE', 'true').lower() == 'true'

# Initialize semantic chunker if available
SEMANTIC_SPLITTER = None
HAS_LANGCHAIN = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True

    SEMANTIC_SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP_SIZE,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    logger.info(f"✅ Semantic chunker initialized (LangChain): size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP_SIZE}")
except ImportError:
    logger.warning("langchain-text-splitters не установлен. Установите: pip install langchain-text-splitters")
    SEMANTIC_SPLITTER = None
except Exception as e:
    logger.warning(f"Failed to initialize semantic chunker: {e}")
    SEMANTIC_SPLITTER = None

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
    mark_as_indexed,
    cleanup_deleted_pages_postgres,
    get_postgres_stats
)
from qdrant_storage import (
    init_qdrant_collection,
    delete_points_by_page_ids,
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

STATE_FILE = os.getenv("STATE_FILE", "./data/sync_state.json")
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
# Импортируем унифицированный модуль embeddings
# sys уже импортирован в начале файла (строка 7)
sys.path.insert(0, os.path.dirname(__file__))
from embeddings import get_embed_model


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
        parts = parts[:1] + ['...'] + parts[-(max_levels - 1):]

    breadcrumb = ' > '.join(parts) if parts else ''

    # Ограничение по длине
    if len(breadcrumb) > max_length:
        breadcrumb = breadcrumb[:max_length - 3] + "..."

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


def _extract_basic_fields(page_data: Dict[str, Any]) -> Dict[str, Any]:
    """Извлечение базовых полей: status, type."""
    metadata = {'status': 'current', 'type': 'page'}

    # Status
    try:
        status = page_data.get('status', 'current')
        if isinstance(status, str):
            metadata['status'] = status.lower()
        else:
            expandable = page_data.get('_expandable', {})
            if 'status' in expandable:
                metadata['status'] = 'current'
    except Exception as e:
        logger.debug(f"Error extracting status: {e}")

    # Type
    try:
        page_type = page_data.get('type', 'page')
        if isinstance(page_type, str):
            metadata['type'] = page_type.lower()
    except Exception as e:
        logger.debug(f"Error extracting type: {e}")

    return metadata

def _extract_labels(page_data: Dict[str, Any]) -> List[str]:
    """Извлечение меток страницы."""
    try:
        labels_data = page_data.get('metadata', {}).get('labels', {})
        if isinstance(labels_data, dict):
            labels = labels_data.get('results', [])
        elif isinstance(labels_data, list):
            labels = labels_data
        else:
            labels = []

        return [
            label.get('name', '') for label in labels
            if isinstance(label, dict) and label.get('name')
        ]
    except Exception as e:
        logger.debug(f"Error extracting labels: {e}")
        return []

def _extract_hierarchy(page_data: Dict[str, Any]) -> Dict[str, Any]:
    """Извлечение информации об иерархии (родители, глубина)."""
    result = {
        'parent_id': '',
        'parent_title': '',
        'hierarchy_depth': 0,
        'parent_titles': []
    }
    try:
        ancestors = page_data.get('ancestors', [])
        if ancestors and isinstance(ancestors, list):
            if len(ancestors) > 0:
                parent = ancestors[-1]
                if isinstance(parent, dict):
                    result['parent_id'] = str(parent.get('id', ''))
                    result['parent_title'] = str(parent.get('title', ''))

            result['hierarchy_depth'] = len(ancestors)

            for ancestor in ancestors:
                if isinstance(ancestor, dict):
                    title = ancestor.get('title', '')
                    if title:
                        result['parent_titles'].append(title)
    except Exception as e:
        logger.debug(f"Error extracting ancestors: {e}")

    return result

def _extract_version_info(page_data: Dict[str, Any]) -> Dict[str, Any]:
    """Извлечение информации о версии, датах и авторах."""
    result = {
        'version': 1,
        'created': '',
        'modified': '',
        'created_by': '',
        'modified_by': '',
        'modified_date': ''
    }
    try:
        version = page_data.get('version', {})
        if isinstance(version, dict):
            result['version'] = int(version.get('number', 1))

            modified_when = version.get('when', '')
            if modified_when:
                result['modified_date'] = str(modified_when)
                result['modified'] = modified_when

            by_info = version.get('by', {})
            if isinstance(by_info, dict):
                result['modified_by'] = str(by_info.get('displayName', ''))

        history = page_data.get('history', {})
        if isinstance(history, dict):
            result['created'] = history.get('createdDate', '') or result['modified']

            created_by = history.get('createdBy', {})
            if isinstance(created_by, dict):
                result['created_by'] = str(created_by.get('displayName', ''))
            elif not result['created_by']: # Fallback
                 result['created_by'] = result['modified_by']

    except Exception as e:
        logger.debug(f"Error extracting version info: {e}")

    return result

def _extract_headings_from_html(content_html: str, page_id: str) -> Dict[str, Any]:
    """Извлечение заголовков из HTML."""
    result = {
        'headings': '',
        'headings_list': [],
        'heading_hierarchy': [],
        'heading_count': 0
    }

    if not content_html:
        return result

    try:
        headings_start = time.time()
        soup = BeautifulSoup(content_html, 'html.parser')

        headings = []
        heading_hierarchy = []
        current_path = []

        MAX_HEADINGS = MAX_HEADINGS_EXTRACT

        for i, heading_tag in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
            if i >= MAX_HEADINGS:
                logger.debug(f"Truncated headings at {MAX_HEADINGS} for page {page_id}")
                break

            heading_text = heading_tag.get_text(strip=True)
            heading_text = html.unescape(heading_text)

            if not heading_text:
                continue

            heading_level = int(heading_tag.name[1])

            headings.append({'text': heading_text, 'level': heading_level})

            # Hierarchy logic
            while len(current_path) > 0 and len(current_path) >= heading_level:
                current_path.pop()
            current_path.append(heading_text)

            heading_hierarchy.append({
                'text': heading_text,
                'level': heading_level,
                'path': ' > '.join(current_path)
            })

        all_headings = [h['text'] for h in headings]

        # Form string
        headings_string = ' | '.join(all_headings)
        if len(headings_string) > MAX_HEADINGS_STRING_LENGTH:
            headings_string = headings_string[:MAX_HEADINGS_STRING_LENGTH] + "..."

        result = {
            'headings': headings_string,
            'headings_list': all_headings,
            'heading_hierarchy': heading_hierarchy,
            'heading_count': len(all_headings)
        }

        # Performance check
        headings_time = (time.time() - headings_start) * 1000
        if headings_time > 100:
             logger.warning(f"⚠️ Slow headings extraction: {headings_time:.0f}ms for {page_id}")

    except Exception as e:
        logger.debug(f"Error extracting headings: {e}")

    return result

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
    if not page_data or not isinstance(page_data, dict):
        logger.debug("Invalid page_data structure")
        return {
            'labels': [], 'parent_id': '', 'parent_title': '', 'page_path': '',
            'breadcrumb': '', 'version': 1, 'created_by': '', 'modified_date': '',
            'has_children': False, 'children_count': 0, 'attachments': [],
            'status': 'current', 'type': 'page', 'hierarchy_depth': 0,
            'created': '', 'modified': '', 'modified_by': '',
            'headings': '', 'headings_list': [], 'heading_hierarchy': [],
            'heading_count': 0, 'parent_titles': []
        }

    # 1. Базовые поля (status, type)
    metadata = _extract_basic_fields(page_data)

    # 2. Метки
    metadata['labels'] = _extract_labels(page_data)

    # 3. Иерархия (parents)
    hierarchy = _extract_hierarchy(page_data)
    metadata.update(hierarchy)

    # 4. Пути и хлебные крошки
    try:
        current_title = page_data.get('title', '')
        parent_titles = metadata.get('parent_titles', [])

        metadata['breadcrumb'] = build_breadcrumb(space_key, parent_titles, current_title)
        metadata['page_path'] = build_page_path(space_key, parent_titles)
    except Exception as e:
        logger.debug(f"Error building paths: {e}")
        metadata['breadcrumb'] = space_key
        metadata['page_path'] = space_key

    # 5. Заголовки из HTML
    body = page_data.get('body', {})
    storage = body.get('storage', {})
    content_html = storage.get('value', '')
    page_id = page_data.get('id', 'unknown')

    headings_info = _extract_headings_from_html(content_html, page_id)
    metadata.update(headings_info)

    # 6. Версии и даты
    version_info = _extract_version_info(page_data)
    metadata.update(version_info)

    # 7. Дочерние страницы
    try:
        children_data = page_data.get('children', {})
        if isinstance(children_data, dict):
            page_info = children_data.get('page', {})
            children = int(page_info.get('size', 0) if isinstance(page_info, dict) else 0)
            metadata['has_children'] = children > 0
            metadata['children_count'] = children
        else:
            metadata['has_children'] = False
            metadata['children_count'] = 0
    except Exception:
        metadata['has_children'] = False
        metadata['children_count'] = 0

    # Init missing fields
    metadata.setdefault('attachments', [])

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
            return '\n\n📊 **PAGE PROPERTIES:**\n' + '\n'.join([f'  • {p}' for p in props]) + '\n\n'
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
            return '\n'.join([f"{i + 1}. {item}" for i, item in enumerate(items)])
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

def _create_chunk(block_content, heading, level, block_type, parent_path, size, context_prefix):
    """Создание структуры чанка."""
    return {
        "text": context_prefix + block_content if context_prefix else block_content,
        "heading": heading,
        "level": level,
        "type": block_type,
        "parent_path": parent_path,
        "size": size
    }

def _try_merge_with_last_chunk(chunks, content, heading, max_size, context_prefix):
    """Попытка объединения с предыдущим чанком."""
    if not chunks: return False
    last = chunks[-1]
    if last.get('heading') != heading or last.get('type') != 'text': return False

    current_len = len(content) + (len(context_prefix) if context_prefix else 0)
    if last['size'] < 600 and (last['size'] + current_len) <= max_size:
        new_text = context_prefix + content if context_prefix else content
        last['text'] += "\n\n" + new_text
        last['size'] += len("\n\n" + new_text)
        logger.debug(f"📦 Объединены blocks: {last['size']} chars")
        return True
    return False

def _split_large_text_block(content, heading, level, block_type, parent_path, max_size, context_prefix):
    """Разбиение большого текстового блока."""
    chunks = []

    # 1. Semantic Splitter
    if SEMANTIC_SPLITTER and PRESERVE_STRUCTURE:
        try:
            text_chunks = SEMANTIC_SPLITTER.split_text(content)
            for t in text_chunks:
                if len(t.strip()) < MIN_CHUNK_SIZE: continue
                chunks.append(_create_chunk(t.strip(), heading, level, block_type, parent_path, len(t), context_prefix))
            logger.debug(f"✅ Semantic chunking: {len(content)} chars → {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}")

    # 2. Fallback (sentence based)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', content)
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 < max_size:
            current += sent + " "
        else:
            if current.strip():
                chunks.append(_create_chunk(current.strip(), heading, level, block_type, parent_path, len(current), context_prefix))
            current = sent + " "

    if current.strip():
        chunks.append(_create_chunk(current.strip(), heading, level, block_type, parent_path, len(current), context_prefix))

    return chunks

def smart_chunk_with_context(blocks: List[Dict[str, Any]], max_size: int = CHUNK_SIZE) -> List[Dict[str, Any]]:
    """Умная нарезка: таблицы и списки целиком, текст по предложениям."""
    chunks = []

    for block in blocks:
        b_type = block['type']
        heading = block['heading']
        content = block['content']
        parent = block.get('parent_path', '')

        prefix = f"{parent} > {heading}\n\n" if parent and heading else (f"{heading}\n\n" if heading else "")

        if b_type in ['table', 'list']:
            chunk = _create_chunk(content, heading, block['level'], b_type, parent, block['size'], prefix)
            if b_type == 'table' and 'html' in block: chunk['html'] = block['html']
            chunks.append(chunk)
            logger.info(f"✓ {b_type.capitalize()} block: '{heading}'")
            continue

        if b_type == 'text':
            if block['size'] <= max_size:
                if not _try_merge_with_last_chunk(chunks, content, heading, max_size, prefix):
                    chunks.append(_create_chunk(content, heading, block['level'], b_type, parent, block['size'], prefix))
            else:
                logger.info(f"⚠ Splitting large text block ({block['size']} chars)")
                sub_chunks = _split_large_text_block(content, heading, block['level'], b_type, parent, max_size, prefix)
                chunks.extend(sub_chunks)

    return chunks

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
    Izvlekaet sekcii dokumenta po zagolovkam (markdown format) s sohraneniem ierarhii.

    Args:
        text: Tekst v markdown formate (posle html2text)

    Returns:
        Spisok sekcij s zagolovkami, kontentom i roditelskimi zagolovkami
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

def _chunk_from_sections(sections, size):
    """Разбиение секций на чанки."""
    chunks = []
    for section in sections:
        heading = section["heading"]
        level = section["level"]
        parent_headings = section.get("parent_headings", [])
        content = '\n'.join(section["content"])

        context_prefix = ""
        if level >= 3 and parent_headings:
            context_prefix = " > ".join(parent_headings) + "\n\n"

        if len(content) <= size:
            text = context_prefix + content.strip() if context_prefix else content.strip()
            chunks.append({"text": text, "heading": heading, "level": level})
        else:
            paras = [p.strip() for p in content.split('\n\n') if p.strip()]
            current = ""
            for para in paras:
                if len(current) + len(para) + 2 < size:
                    current += para + "\n\n"
                else:
                    if current.strip():
                        text = context_prefix + current.strip() if context_prefix else current.strip()
                        chunks.append({"text": text, "heading": heading, "level": level})
                    current = para + "\n\n"
            if current.strip():
                text = context_prefix + current.strip() if context_prefix else current.strip()
                chunks.append({"text": text, "heading": heading, "level": level})
    return chunks

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

    if sections:
        return _chunk_from_sections(sections, size) or [{"text": text, "heading": "", "level": 0}]

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
        if USE_BLOOM_FILTER and HAS_BLOOM_FILTER:
            self.processed_ids = BloomFilter(capacity=BLOOM_FILTER_SIZE, error_rate=0.001)
            self._use_bloom = True
            logger.debug(f"Using Bloom Filter for duplicate detection (capacity={BLOOM_FILTER_SIZE})")
        else:
            self.processed_ids = set()
            self._use_bloom = False
        self.failed_ids = {}

    def _parse_updated_at(self, page_data: Dict[str, Any]) -> datetime:
        """Парсинг даты обновления."""
        try:
            version_when = page_data.get('version', {}).get('when', '')
            if version_when:
                if version_when.endswith('Z'):
                    version_when = version_when[:-1] + '+00:00'
                return datetime.fromisoformat(version_when)
        except Exception as e:
            logger.debug(f"Ошибка парсинга даты: {e}")
        return datetime.now()

    def _prepare_chunk_metadata(self, chunk: dict, chunk_idx: int, page_id: str, title: str,
                              page_metadata: dict, space_key: str) -> dict:
        """Подготовка метаданных для чанка."""
        chunk_content = chunk.get('text', '')

        # Ограничиваем длину полей
        max_str_len = 500
        labels_list = page_metadata.get('labels', [])
        labels_safe = ",".join(labels_list)[:max_str_len]

        # Базовые метаданные
        metadata = {
            "page_id": page_id,
            "chunk": chunk_idx,
            "title": title[:max_str_len] if title else "Unknown",
            "space": space_key,
            "url": f"{CONFLUENCE_URL}pages/viewpage.action?pageId={page_id}",
            "text": chunk_content,  # КРИТИЧНО для payload

            # Структура
            "heading": chunk.get('heading', '')[:max_str_len],
            "heading_level": chunk.get('level', 0),
            "type": chunk.get('type', 'text'),
            "parent_path": chunk.get('parent_path', '')[:max_str_len],
            "block_size": chunk.get('size', 0),

            # Confluence
            "labels": labels_safe,
            "parent_title": page_metadata.get('parent_title', '')[:max_str_len],
            "page_path": page_metadata.get('page_path', '')[:200],
            "breadcrumb": page_metadata.get('breadcrumb', '')[:200],
            "created_by": page_metadata.get('created_by', '')[:max_str_len],
            "version": page_metadata.get('version', 1),

            # Фильтрация
            "status": page_metadata.get('status', 'current'),
            "content_type": page_metadata.get('type', 'page'),
            "hierarchy_depth": page_metadata.get('hierarchy_depth', 0),
            "created": page_metadata.get('created', ''),
            "modified": page_metadata.get('modified', ''),
        }

        return sanitize_metadata(metadata)

    def _index_chunks(self, page_id: str, title: str, chunks: list, page_metadata: dict,
                     space_key: str, qdrant_client: Any):
        """Индексация чанков в Qdrant."""
        # Удаляем старые
        try:
            delete_points_by_page_ids([page_id])
        except Exception as e:
            logger.warning(f"Ошибка удаления старых чанков для {page_id}: {e}")

        # Подготавливаем чанки
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if not chunk.get('text') or len(chunk['text']) < 20:
                continue

            metadata = self._prepare_chunk_metadata(chunk, i, page_id, title, page_metadata, space_key)
            valid_chunks.append({
                'text': chunk['text'],
                'metadata': metadata,
                'point_id': f"{page_id}_{i}"
            })

        if not valid_chunks:
            return

        # Генерируем эмбеддинги батчем
        texts = [c['text'] for c in valid_chunks]
        try:
            embeddings = generate_query_embeddings_batch(texts)
            for i, chunk in enumerate(valid_chunks):
                chunk['embedding'] = embeddings[i]
        except Exception:
            # Fallback: по одному
            for chunk in valid_chunks:
                chunk['embedding'] = generate_query_embedding(chunk['text'])

        # Вставка в Qdrant
        chunks_to_insert = valid_chunks
        if len(chunks_to_insert) > BATCH_INSERT_THRESHOLD:
            success, failed = insert_chunks_batch_to_qdrant(qdrant_client, chunks_to_insert, batch_size=100)
            if failed > 0:
                 logger.warning(f"Failed to insert {failed} chunks for {page_id}")

    def _process_page_logic(self, page_id: str, title: str, qdrant_client: Any,
                          confluence: Confluence, space_key: str) -> bool:
        """Логика обработки одной страницы (без retry и проверок состояния)."""
        # 1. Получение и метаданные
        page_data = get_page_cached(confluence, page_id)
        page_metadata = extract_page_metadata(page_data, space_key=space_key)
        page_metadata['attachments'] = get_page_attachments(confluence, page_id)

        html = page_data.get('body', {}).get('storage', {}).get('value', '')
        if not html or len(html) < MIN_TEXT_LEN:
            return False

        # 2. Нарезка
        blocks = extract_structural_blocks(html)
        if not blocks:
            return False
        chunks = smart_chunk_with_context(blocks, max_size=CHUNK_SIZE)

        # 3. Postgres
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        content_markdown = h.handle(html)

        updated_at = self._parse_updated_at(page_data)

        if not save_page_to_postgres(
            page_id=page_id, space_key=space_key, title=title,
            content_html=html, content_markdown=content_markdown,
            version=page_data.get('version', {}).get('number', 1),
            metadata=page_metadata, updated_at=updated_at
        ):
            logger.warning(f"Не удалось сохранить страницу {page_id} в PostgreSQL")
            return False

        # 4. Qdrant
        self._index_chunks(page_id, title, chunks, page_metadata, space_key, qdrant_client)

        return True

    def process_batch_safe(
        self,
        qdrant_client: Any,
        confluence: Confluence,
        batch: List[Dict[str, Any]],
        state: Dict[str, Any],
        space_key: str
    ) -> tuple[int, int, int, list]:
        """
        Процессировать батч с гарантией целостности.
        """
        updated, errors, skipped = 0, 0, 0
        error_details = []

        for page in batch:
            page_id = str(page.get('id', ''))
            if not page_id:
                skipped += 1; continue

            title = page.get('title', 'Unknown')
            ts = get_timestamp(page)

            # Проверка дубликатов и состояния
            if page_id in self.processed_ids:
                skipped += 1; continue

            if page_id in state['pages'] and state['pages'][page_id].get('updated') == ts:
                skipped += 1; continue

            try:
                # Retry logic
                success = False
                for attempt in range(self.max_retries):
                    try:
                        if self._process_page_logic(page_id, title, qdrant_client, confluence, space_key):
                            state['pages'][page_id] = {'updated': ts, 'title': title}
                            mark_as_indexed(page_id)
                            self.processed_ids.add(page_id)
                            updated += 1
                            success = True
                        else:
                            skipped += 1
                        break # Success
                    except Exception as retry_err:
                        if attempt < self.max_retries - 1:
                            wait = 2 ** attempt
                            logger.warning(f"Retry {attempt+1} for {page_id}: {retry_err}. Wait {wait}s")
                            time.sleep(wait)
                        else:
                            raise

            except Exception as e:
                error_msg = f"Error processing {page_id} ({title}): {e}"
                logger.error(error_msg)
                self.failed_ids[page_id] = str(e)
                errors += 1
                error_details.append({
                    "page_id": page_id, "title": title,
                    "error": str(e), "timestamp": datetime.now().isoformat()
                })

        return updated, errors, skipped, error_details

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику обработки"""
        if self._use_bloom:
            processed_count = self.processed_ids.count
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
    Legacy wrapper for backward compatibility.
    Использует BatchProcessor для обработки.
    """
    processor = BatchProcessor(max_retries=1)
    return processor.process_batch_safe(qdrant_client, confluence, pages, state, space_key)


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
            deleted_count = delete_points_by_page_ids([page_id])
            if deleted_count > 0:
                logger.info(f"  Удалена страница {page_id} из Qdrant ({deleted_count} чанков)")

            # Удаление из state
            if page_id in state['pages']:
                del state['pages'][page_id]
        except Exception as e:
            logger.error(f"Ошибка удаления страницы {page_id}: {e}")

    return deleted_count

def _init_services() -> Optional[tuple]:
    """Инициализация сервисов (Confluence, Postgres, Qdrant)."""
    try:
        confluence = Confluence(url=CONFLUENCE_URL, token=CONFLUENCE_TOKEN)
        logger.info("Connected to Confluence")

        logger.info("Шаг 1: Инициализация PostgreSQL...")
        init_postgres_schema()

        logger.info("Шаг 2: Загрузка embedding модели...")
        get_embed_model()

        logger.info("Шаг 3: Инициализация Qdrant...")
        from embeddings import get_embedding_dimension
        model_dim = get_embedding_dimension()

        qdrant_client = init_qdrant_client()
        if not init_qdrant_collection(model_dim):
            raise ValueError(f"Failed to init Qdrant collection {model_dim}D")

        return confluence, qdrant_client
    except Exception as e:
        logger.error(f"Init error: {e}")
        return None

def _get_target_spaces(confluence: Confluence) -> List[Dict[str, Any]]:
    """Получение целевых пространств."""
    try:
        all_spaces = confluence.get_all_spaces().get('results', [])
        if CONFLUENCE_SPACES:
            targets = [s.strip().upper() for s in CONFLUENCE_SPACES.split(',') if s.strip()]
            spaces = [s for s in all_spaces if s.get('key', '').upper() in targets]
            logger.info(f"Фильтр пространств: {len(spaces)} из {len(all_spaces)}")
            return spaces
        logger.info(f"Processing {len(all_spaces[:MAX_SPACES])} spaces")
        return all_spaces[:MAX_SPACES]
    except Exception as e:
        logger.error(f"Error fetching spaces: {e}")
        return []

def _process_items_parallel(processor: BatchProcessor, items: list, qdrant_client: Any,
                          confluence: Confluence, state: Dict, key: str, stats: Dict):
    """Параллельная обработка списка элементов."""
    if not items: return

    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    max_workers = get_int_env("PARALLEL_SYNC_MAX_WORKERS", 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(processor.process_batch_safe, qdrant_client, confluence, b, state, key): b
            for b in batches
        }
        for f in as_completed(futures):
            try:
                u, e, s, d = f.result()
                stats['updated'] += u
                stats['errors'] += e
                stats['skipped'] += s
                stats['processed'] += len(futures[f])
                stats['error_details'].extend(d)
            except Exception as err:
                logger.error(f"Batch failed: {err}")
                stats['errors'] += 1

def sync() -> None:
    """Основной процесс синхронизации Confluence с PostgreSQL + Qdrant."""
    logger.info("Sync started")
    state = load_state()
    start_time = time.time()

    # 1. Инициализация
    services = _init_services()
    if not services: return
    confluence, qdrant_client = services

    # 2. Получение пространств
    spaces = _get_target_spaces(confluence)

    space_stats = {}
    current_page_ids = set()
    total_updated, total_errors, total_skipped = 0, 0, 0

    batch_processor = BatchProcessor(max_retries=3)

    if TQDM_AVAILABLE:
        spaces_iter = tqdm(spaces, desc="Syncing spaces", unit="space")
    else:
        spaces_iter = spaces

    # 3. Обработка пространств
    for space in spaces_iter:
        key = space.get('key', '')
        if not key: continue

        if TQDM_AVAILABLE: spaces_iter.set_description(f"Syncing {space.get('name', key)}")

        logger.info(f"📂 {space.get('name', key)}:")
        stats = {
            'total_pages': 0, 'total_blogs': 0, 'processed': 0,
            'updated': 0, 'skipped': 0, 'errors': 0, 'chunks_created': 0, 'error_details': []
        }
        space_stats[key] = stats

        # Обработка страниц
        pages = []
        for page in get_all_pages_generator(confluence, key, batch_size=BATCH_SIZE):
            pages.append(page)
            if page.get('id'): current_page_ids.add(str(page['id']))

        stats['total_pages'] = len(pages)
        logger.info(f"   Страниц: {len(pages)}")

        _process_items_parallel(batch_processor, pages, qdrant_client, confluence, state, key, stats)

        # Обработка блогов
        blogs = []
        start = 0
        while True:
            b = get_blogposts_from_space(confluence, key, start=start, limit=BATCH_SIZE)
            if not b: break
            blogs.extend(b)
            start += BATCH_SIZE

        for blog in blogs:
            if blog.get('id'): current_page_ids.add(str(blog['id']))

        stats['total_blogs'] = len(blogs)
        if blogs:
            logger.info(f"   Блогов: {len(blogs)}")
            _process_items_parallel(batch_processor, blogs, qdrant_client, confluence, state, key, stats)

        # Агрегация статистики
        total_updated += stats['updated']
        total_errors += stats['errors']
        total_skipped += stats['skipped']

        logger.info(f"   Итог: Upd={stats['updated']} Skip={stats['skipped']} Err={stats['errors']}")

    # 4. Очистка и сохранение
    deleted_count = cleanup_deleted_pages(qdrant_client, state, current_page_ids)
    state['last_sync'] = int(time.time())
    save_state(state)

    elapsed = time.time() - start_time

    # Итоговый отчет
    logger.info("=" * 80)
    logger.info(f"🏁 Sync finished in {elapsed:.1f}s")
    logger.info(f"✅ Updated: {total_updated} | ⏭ Skipped: {total_skipped} | ❌ Errors: {total_errors} | 🗑 Deleted: {deleted_count}")
    logger.info("=" * 80)


if __name__ == "__main__":
    sync()
    logger.info(f"Синхронизация будет повторяться каждые {SYNC_INTERVAL} секунд ({SYNC_INTERVAL / 3600:.1f} часов)")
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
