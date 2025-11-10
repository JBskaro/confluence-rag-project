"""
Единый модуль для работы с embeddings.
Поддерживает HuggingFace, Ollama и OpenAI-compatible API (LM Studio, Ollama и др.).
Thread-safe кэширование моделей.
"""
import os
import logging
import threading
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Отключаем избыточное логирование HTTP запросов от httpx/openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Конфигурация из ENV
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "ai-forever/FRIDA")

# OpenAI-compatible API (для Ollama, LM Studio и др.)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")  # Например: http://localhost:11434/v1
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")  # Для Ollama обычно не нужен
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")  # Имя модели для API

# Thread-safe кэширование модели с использованием Lock
_model_lock = threading.Lock()
_embed_model = None
_embed_model_type = None  # 'openai', 'ollama', 'huggingface'

def get_embed_model():
    """
    Получить инициализированную модель embeddings.
    
    Приоритет выбора модели (в порядке приоритета):
    1. OpenAI-compatible API (если указан OPENAI_API_BASE) - для Ollama, LM Studio и др.
    2. LlamaIndex Ollama (если USE_OLLAMA=true)
    3. HuggingFace (локальная модель, по умолчанию)
    
    ВАЖНО: Если указана модель, она должна быть доступна. Нет автоматического fallback.
    
    Модель кэшируется глобально и создается только один раз.
    
    Returns:
        Embedding model instance
    
    Raises:
        RuntimeError: Если указанная модель недоступна
        ImportError: Если не установлены необходимые пакеты
    """
    global _embed_model, _embed_model_type
    
    # ========================================
    # НОВАЯ ЛОГИКА: Выбор источника по EMBEDDING_SOURCE
    # ========================================
    embedding_source = os.getenv('EMBEDDING_SOURCE', '').lower()
    
    if embedding_source:
        logger.info(f"🔄 EMBEDDING_SOURCE={embedding_source} (explicit selection)")
        
        # Если модель уже загружена с другим источником - нужно перезагрузить
        if _embed_model is not None and _embed_model_type != embedding_source:
            logger.warning(f"⚠️ Model already loaded with type {_embed_model_type}, reloading for {embedding_source}")
            _embed_model = None
            _embed_model_type = None
    else:
        logger.debug("ℹ️ EMBEDDING_SOURCE not specified, using legacy logic (priority order)")
    
    # Double-check locking pattern для thread-safety
    if _embed_model is None:
        with _model_lock:
            # Повторная проверка внутри lock (другой поток мог уже загрузить)
            if _embed_model is None:
                import time
                start_time = time.time()
                
                # ========== ВАРИАНТ 1: OpenRouter (если EMBEDDING_SOURCE=openrouter) ==========
                if embedding_source == 'openrouter':
                    if not OPENAI_API_BASE or not OPENAI_API_KEY or not OPENAI_MODEL:
                        raise ValueError(
                            "EMBEDDING_SOURCE=openrouter requires: "
                            "OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL"
                        )
                    # Используем существующий код для OpenAI-compatible API
                    try:
                        from openai import OpenAI
                        from llama_index.core.embeddings import BaseEmbedding
                        
                        api_base = OPENAI_API_BASE.rstrip('/')
                        if not api_base.endswith('/v1'):
                            api_base = f"{api_base}/v1"
                        
                        model_name = OPENAI_MODEL or EMBED_MODEL
                        
                        logger.info(f"🔌 Попытка подключения к OpenAI-compatible API: {api_base}")
                        logger.info(f"   Модель: {model_name}")
                        
                        client = OpenAI(base_url=api_base, api_key=OPENAI_API_KEY or None)
                        
                        # Тестовая проверка подключения
                        test_response = client.embeddings.create(
                            model=model_name,
                            input=["test"]
                        )
                        
                        # Определяем размерность
                        test_dim = len(test_response.data[0].embedding)
                        
                        # Создаем кастомный класс, наследующийся от BaseEmbedding
                        class CustomOpenAIEmbedding(BaseEmbedding):
                            def __init__(self, client, model_name, dimension):
                                # BaseEmbedding использует Pydantic, нужно вызывать super().__init__() без параметров
                                super().__init__()
                                # Сохраняем параметры как приватные атрибуты (не Pydantic поля)
                                self._client = client
                                self._model_name = model_name
                                self._dimension = dimension
                            
                            def _get_query_embedding(self, query: str) -> List[float]:
                                response = self._client.embeddings.create(
                                    model=self._model_name,
                                    input=[query]
                                )
                                return response.data[0].embedding
                            
                            async def _aget_query_embedding(self, query: str) -> List[float]:
                                # Асинхронная версия (используем синхронный клиент)
                                return self._get_query_embedding(query)
                            
                            def _get_text_embedding(self, text: str) -> List[float]:
                                return self._get_query_embedding(text)
                            
                            async def _aget_text_embedding(self, text: str) -> List[float]:
                                return self._get_text_embedding(text)
                            
                            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                                response = self._client.embeddings.create(
                                    model=self._model_name,
                                    input=texts
                                )
                                return [item.embedding for item in response.data]
                            
                            async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                                return self._get_text_embeddings(texts)
                            
                            @property
                            def dimension(self) -> int:
                                return self._dimension
                            
                            def get_embedding_dimension(self) -> int:
                                """Метод для совместимости с get_embedding_dimension()."""
                                return self._dimension
                        
                        _embed_model = CustomOpenAIEmbedding(client, model_name, test_dim)
                        _embed_model_type = 'openai'
                        
                        elapsed = time.time() - start_time
                        logger.info(f"✅ OpenAI-compatible API подключен за {elapsed:.1f} сек")
                        logger.info(f"   Модель: {model_name}, Размерность: {test_dim}D")
                        return _embed_model
                        
                    except ImportError as import_err:
                        error_msg = (
                            f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать необходимые модули\n"
                            f"   Ошибка: {import_err}\n"
                            f"   Установите: pip install openai llama-index"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    except Exception as api_error:
                        error_msg = (
                            f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к OpenAI-compatible API\n"
                            f"   URL: {OPENAI_API_BASE}\n"
                            f"   Модель: {model_name}\n"
                            f"   Ошибка: {api_error}\n\n"
                            f"   РЕШЕНИЕ:\n"
                            f"   1. Проверьте URL: {OPENAI_API_BASE}\n"
                            f"   2. Проверьте API ключ: {OPENAI_API_KEY[:10] if OPENAI_API_KEY else 'не указан'}...\n"
                            f"   3. Проверьте имя модели: {model_name}"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                
                # ========== ВАРИАНТ 2: Ollama (если EMBEDDING_SOURCE=ollama) ==========
                elif embedding_source == 'ollama':
                    ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
                    ollama_model = os.getenv('OLLAMA_EMBEDDING_MODEL') or os.getenv('OLLAMA_MODEL') or EMBED_MODEL
                    
                    if not ollama_url:
                        raise ValueError("EMBEDDING_SOURCE=ollama requires: OLLAMA_URL")
                    
                    # Используем существующий код для Ollama
                    # ========== ПРИОРИТЕТ 2: LlamaIndex Ollama ==========
                    try:
                        from llama_index.embeddings.ollama import OllamaEmbedding
                        logger.info(f"🔌 Попытка подключения к Ollama: {ollama_model} @ {ollama_url}")
                        _embed_model = OllamaEmbedding(model_name=ollama_model, base_url=ollama_url)
                        
                        # Тестовая проверка
                        test_embedding = _embed_model.get_text_embedding("test")
                        _embed_model_type = 'ollama'
                        
                        elapsed = time.time() - start_time
                        logger.info(f"✅ Ollama подключен за {elapsed:.1f} сек")
                        logger.info(f"   Модель: {ollama_model}, Размерность: {len(test_embedding)}D")
                        return _embed_model
                        
                    except ImportError:
                        error_msg = (
                            f"❌ КРИТИЧЕСКАЯ ОШИБКА: llama-index-embeddings-ollama не установлен\n"
                            f"   Установите: pip install llama-index-embeddings-ollama"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    except Exception as ollama_error:
                        error_msg = (
                            f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к Ollama\n"
                            f"   URL: {ollama_url}\n"
                            f"   Модель: {ollama_model}\n"
                            f"   Ошибка: {ollama_error}\n\n"
                            f"   РЕШЕНИЕ:\n"
                            f"   1. Убедитесь, что Ollama запущен: ollama serve\n"
                            f"   2. Проверьте URL: {ollama_url}\n"
                            f"   3. Установите модель: ollama pull {ollama_model}"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                
                # ========== ВАРИАНТ 3: HuggingFace (если EMBEDDING_SOURCE=huggingface) ==========
                elif embedding_source == 'huggingface':
                    # Используем существующий код для HuggingFace
                    # ========== ПРИОРИТЕТ 3: HuggingFace (по умолчанию, если ничего не указано) ==========
                    try:
                        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                        logger.info(f"📦 Загрузка HuggingFace embeddings: {EMBED_MODEL}")
                        logger.info("   (~1.5GB, может занять 30-90 сек)")
                        _embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
                        _embed_model_type = 'huggingface'
                        
                        elapsed = time.time() - start_time
                        logger.info(f"✅ HuggingFace модель загружена за {elapsed:.1f} сек")
                        logger.info(f"   Модель: {EMBED_MODEL}")
                        return _embed_model
                        
                    except ImportError as e:
                        error_msg = (
                            f"❌ КРИТИЧЕСКАЯ ОШИБКА: llama-index-embeddings-huggingface не установлен\n"
                            f"   Установите: pip install llama-index-embeddings-huggingface"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    except Exception as e:
                        error_msg = (
                            f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить HuggingFace модель\n"
                            f"   Модель: {EMBED_MODEL}\n"
                            f"   Ошибка: {e}"
                        )
                        logger.error(error_msg)
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        raise RuntimeError(error_msg)
                
                # ========== LEGACY: Старая логика (если EMBEDDING_SOURCE не указан) ==========
                elif embedding_source == '':
                    logger.info("ℹ️ EMBEDDING_SOURCE not specified, using legacy logic (priority order)")
                    
                    # ========== ПРИОРИТЕТ 1: OpenAI-compatible API ==========
                    if OPENAI_API_BASE:
                        try:
                            from openai import OpenAI
                            from llama_index.core.embeddings import BaseEmbedding
                            
                            api_base = OPENAI_API_BASE.rstrip('/')
                            if not api_base.endswith('/v1'):
                                api_base = f"{api_base}/v1"
                            
                            model_name = OPENAI_MODEL or EMBED_MODEL
                            
                            logger.info(f"🔌 Попытка подключения к OpenAI-compatible API: {api_base}")
                            logger.info(f"   Модель: {model_name}")
                            
                            client = OpenAI(base_url=api_base, api_key=OPENAI_API_KEY or None)
                            
                            # Тестовая проверка подключения
                            test_response = client.embeddings.create(
                                model=model_name,
                                input=["test"]
                            )
                            
                            # Определяем размерность
                            test_dim = len(test_response.data[0].embedding)
                            
                            # Создаем кастомный класс, наследующийся от BaseEmbedding
                            class CustomOpenAIEmbedding(BaseEmbedding):
                                def __init__(self, client, model_name, dimension):
                                    super().__init__()
                                    self._client = client
                                    self._model_name = model_name
                                    self._dimension = dimension
                                
                                def _get_query_embedding(self, query: str) -> List[float]:
                                    response = self._client.embeddings.create(
                                        model=self._model_name,
                                        input=[query]
                                    )
                                    return response.data[0].embedding
                                
                                async def _aget_query_embedding(self, query: str) -> List[float]:
                                    return self._get_query_embedding(query)
                                
                                def _get_text_embedding(self, text: str) -> List[float]:
                                    return self._get_query_embedding(text)
                                
                                async def _aget_text_embedding(self, text: str) -> List[float]:
                                    return self._get_text_embedding(text)
                                
                                def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                                    response = self._client.embeddings.create(
                                        model=self._model_name,
                                        input=texts
                                    )
                                    return [item.embedding for item in response.data]
                                
                                async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                                    return self._get_text_embeddings(texts)
                                
                                @property
                                def dimension(self) -> int:
                                    return self._dimension
                                
                                def get_embedding_dimension(self) -> int:
                                    return self._dimension
                            
                            _embed_model = CustomOpenAIEmbedding(client, model_name, test_dim)
                            _embed_model_type = 'openai'
                            
                            elapsed = time.time() - start_time
                            logger.info(f"✅ OpenAI-compatible API подключен за {elapsed:.1f} сек")
                            logger.info(f"   Модель: {model_name}, Размерность: {test_dim}D")
                            return _embed_model
                            
                        except ImportError as import_err:
                            error_msg = (
                                f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать необходимые модули\n"
                                f"   Ошибка: {import_err}\n"
                                f"   Установите: pip install openai llama-index"
                            )
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                        except Exception as api_error:
                            error_msg = (
                                f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к OpenAI-compatible API\n"
                                f"   URL: {OPENAI_API_BASE}\n"
                                f"   Модель: {model_name}\n"
                                f"   Ошибка: {api_error}\n\n"
                                f"   РЕШЕНИЕ:\n"
                                f"   1. Проверьте URL: {OPENAI_API_BASE}\n"
                                f"   2. Проверьте API ключ: {OPENAI_API_KEY[:10] if OPENAI_API_KEY else 'не указан'}...\n"
                                f"   3. Проверьте имя модели: {model_name}"
                            )
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                    
                    # ========== ПРИОРИТЕТ 2: LlamaIndex Ollama ==========
                    if USE_OLLAMA:
                        try:
                            from llama_index.embeddings.ollama import OllamaEmbedding
                            logger.info(f"🔌 Попытка подключения к Ollama: {EMBED_MODEL} @ {OLLAMA_URL}")
                            _embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)
                            
                            # Тестовая проверка
                            test_embedding = _embed_model.get_text_embedding("test")
                            _embed_model_type = 'ollama'
                            
                            elapsed = time.time() - start_time
                            logger.info(f"✅ Ollama подключен за {elapsed:.1f} сек")
                            logger.info(f"   Модель: {EMBED_MODEL}, Размерность: {len(test_embedding)}D")
                            return _embed_model
                            
                        except ImportError:
                            error_msg = (
                                f"❌ КРИТИЧЕСКАЯ ОШИБКА: llama-index-embeddings-ollama не установлен\n"
                                f"   Установите: pip install llama-index-embeddings-ollama"
                            )
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                        except Exception as ollama_error:
                            error_msg = (
                                f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к Ollama\n"
                                f"   URL: {OLLAMA_URL}\n"
                                f"   Модель: {EMBED_MODEL}\n"
                                f"   Ошибка: {ollama_error}\n\n"
                                f"   РЕШЕНИЕ:\n"
                                f"   1. Убедитесь, что Ollama запущен: ollama serve\n"
                                f"   2. Проверьте URL: {OLLAMA_URL}\n"
                                f"   3. Установите модель: ollama pull {EMBED_MODEL}"
                            )
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                    
                    # ========== ПРИОРИТЕТ 3: HuggingFace (по умолчанию, если ничего не указано) ==========
                    try:
                        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                        logger.info(f"📦 Загрузка HuggingFace embeddings: {EMBED_MODEL}")
                        logger.info("   (~1.5GB, может занять 30-90 сек)")
                        _embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
                        _embed_model_type = 'huggingface'
                        
                        elapsed = time.time() - start_time
                        logger.info(f"✅ HuggingFace модель загружена за {elapsed:.1f} сек")
                        logger.info(f"   Модель: {EMBED_MODEL}")
                        return _embed_model
                        
                    except ImportError as e:
                        error_msg = (
                            f"❌ КРИТИЧЕСКАЯ ОШИБКА: llama-index-embeddings-huggingface не установлен\n"
                            f"   Установите: pip install llama-index-embeddings-huggingface"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    except Exception as e:
                        error_msg = (
                            f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить HuggingFace модель\n"
                            f"   Модель: {EMBED_MODEL}\n"
                            f"   Ошибка: {e}"
                        )
                        logger.error(error_msg)
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        raise RuntimeError(error_msg)
                
                else:
                    raise ValueError(
                        f"Unknown EMBEDDING_SOURCE: {embedding_source}. "
                        f"Use: 'openrouter', 'ollama', 'huggingface' or leave empty for legacy logic"
                    )
    
    return _embed_model

def get_embedding_dimension() -> int:
    """
    Получить размерность embeddings текущей модели.
    
    Returns:
        int: Размерность вектора embeddings
    """
    global _embed_model_type
    
    model = get_embed_model()
    
    if _embed_model_type == 'openai':
        # OpenAI-compatible API: используем сохраненную размерность
        return model.get_embedding_dimension()
    elif _embed_model_type == 'ollama':
        # Для Ollama нужно сделать тестовый запрос
        test_embedding = model.get_text_embedding("test")
        return len(test_embedding)
    else:
        # Для HuggingFace можем получить из _model
        try:
            return model._model.get_sentence_embedding_dimension()
        except AttributeError:
            # Fallback: тестовый запрос
            test_embedding = model.get_text_embedding("test")
            return len(test_embedding)

def generate_query_embedding(query: str) -> List[float]:
    """
    Генерирует embedding для одного запроса.
    
    Поддерживает как HuggingFace, так и Ollama.
    
    Args:
        query: Текст запроса
        
    Returns:
        Список float значений (embedding вектор)
    """
    model = get_embed_model()
    
    # LlamaIndex embeddings имеют унифицированный метод get_query_embedding()
    embedding = model.get_query_embedding(query)
    
    return embedding

def generate_query_embeddings_batch(queries: List[str]) -> List[List[float]]:
    """
    Генерирует embeddings для списка запросов.
    
    ОПТИМИЗАЦИЯ для HuggingFace: 
    - Batch encoding в 3-5 раз быстрее через внутренний SentenceTransformer
    
    ОПТИМИЗАЦИЯ для OpenAI-compatible API:
    - Batch encoding через API (если поддерживается)
    
    ОПТИМИЗАЦИЯ для Ollama:
    - Итерация по запросам (Ollama API не поддерживает batch)
    
    Args:
        queries: Список текстов запросов
        
    Returns:
        Список embedding векторов
    """
    global _embed_model_type
    
    model = get_embed_model()
    
    if _embed_model_type == 'openai':
        # OpenAI-compatible API: пробуем batch, fallback на последовательные запросы
        try:
            # Используем приватные атрибуты для CustomOpenAIEmbedding
            if hasattr(model, '_client'):
                # CustomOpenAIEmbedding использует приватные атрибуты
                response = model._client.embeddings.create(
                    model=model._model_name,
                    input=queries
                )
                return [item.embedding for item in response.data]
            else:
                # Стандартный OpenAIEmbedding из LlamaIndex
                response = model.client.embeddings.create(
                    model=model.model_name,
                    input=queries
                )
                return [item.embedding for item in response.data]
        except Exception as e:
            logger.debug(f"Batch не поддерживается, используем последовательные запросы: {e}")
            return [model.get_query_embedding(q) for q in queries]
    elif _embed_model_type == 'huggingface':
        # HuggingFace: используем batch encoding через внутренний SentenceTransformer
        try:
            sentence_model = model._model
            embeddings = sentence_model.encode(queries, normalize_embeddings=False)
            return [emb.tolist() for emb in embeddings]
        except AttributeError:
            # Fallback если _model недоступен
            logger.warning("Не удалось использовать batch encoding, fallback на последовательную генерацию")
            return [model.get_query_embedding(q) for q in queries]
    else:
        # Ollama: по одному запросу (нет batch API)
        return [model.get_query_embedding(q) for q in queries]

def validate_collection_dimension(collection, expected_dim: int = None) -> tuple[bool, int, int]:
    """
    Проверяет совпадение размерности embeddings в ChromaDB с текущей моделью.
    
    Args:
        collection: ChromaDB collection
        expected_dim: Ожидаемая размерность (если None - берется из модели)
        
    Returns:
        tuple: (is_valid, collection_dim, model_dim)
    """
    try:
        # Получаем размерность из collection
        data = collection.get(limit=1, include=['embeddings'])
        embeddings = data.get('embeddings', [])
        
        if len(embeddings) == 0 or embeddings[0] is None:
            logger.warning("⚠️ ChromaDB пустая, размерность не определена")
            return (True, 0, 0)  # Пустая база - OK
        
        collection_dim = len(embeddings[0])
        
        # Получаем размерность модели
        if expected_dim is None:
            model_dim = get_embedding_dimension()
        else:
            model_dim = expected_dim
        
        is_valid = (collection_dim == model_dim)
        
        if not is_valid:
            logger.error(
                f"❌ НЕСОВПАДЕНИЕ РАЗМЕРНОСТИ!\n"
                f"   ChromaDB: {collection_dim} dimensions\n"
                f"   Модель {EMBED_MODEL}: {model_dim} dimensions\n"
                f"   → Необходима пересинхронизация базы!"
            )
        else:
            logger.info(f"✅ Размерность embeddings корректна: {model_dim}D")
        
        return (is_valid, collection_dim, model_dim)
        
    except Exception as e:
        logger.error(f"Ошибка проверки размерности: {e}")
        return (False, 0, 0)

