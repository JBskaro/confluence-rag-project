"""
Единый модуль для работы с embeddings.
Поддерживает HuggingFace, Ollama и OpenAI-compatible API (LM Studio, Ollama и др.).
Thread-safe кэширование моделей.
"""
import os
import logging
import threading
import time
import asyncio
import atexit
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor

# Отключаем избыточное логирование
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Конфигурация из Pydantic
from rag_server.config import settings

# Thread-safe кэширование
_model_lock = threading.Lock()
_embed_model = None

# Executor для синхронных операций (HuggingFace)
_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

def _shutdown_executor():
    """Graceful shutdown for executor."""
    logger.info("Shutting down ThreadPoolExecutor...")
    _executor.shutdown(wait=True)
    logger.info("✅ Executor shut down")

atexit.register(_shutdown_executor)


class UnifiedEmbeddingModel:
    """
    Унифицированный класс для работы с различными моделями embeddings.
    Заменяет LlamaIndex BaseEmbedding.
    """
    def __init__(self, source: str, model_name: str, dimension: int, client: Any = None, async_client: Any = None):
        self.source = source
        self.model_name = model_name
        self._dimension = dimension
        self.client = client
        self.async_client = async_client

    def get_query_embedding(self, query: str) -> List[float]:
        """Генерация embedding для запроса."""
        return self._generate_embedding(query)

    async def get_query_embedding_async(self, query: str) -> List[float]:
        """Асинхронная генерация embedding для запроса."""
        return await self._generate_embedding_async(query)

    def get_text_embedding(self, text: str) -> List[float]:
        """Генерация embedding для текста."""
        return self._generate_embedding(text)

    async def get_text_embedding_async(self, text: str) -> List[float]:
        """Асинхронная генерация embedding для текста."""
        return await self._generate_embedding_async(text)

    def _generate_embedding(self, text: str) -> List[float]:
        """Внутренний метод генерации."""
        if not text:
            return [0.0] * self._dimension

        try:
            if self.source in ('openai', 'openrouter', 'ollama'):
                # Используем OpenAI client
                text = text.replace("\n", " ")
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=[text]
                )
                return response.data[0].embedding

            elif self.source == 'huggingface':
                # Используем SentenceTransformer
                embedding = self.client.encode(text, normalize_embeddings=False)
                return embedding.tolist()

            else:
                raise ValueError(f"Unknown source: {self.source}")

        except Exception as e:
            logger.error(f"Error generating embedding ({self.source}): {e}")
            raise e

    async def _generate_embedding_async(self, text: str) -> List[float]:
        """Внутренний асинхронный метод генерации."""
        if not text:
            return [0.0] * self._dimension

        try:
            if self.source in ('openai', 'openrouter', 'ollama'):
                # Используем AsyncOpenAI client
                text = text.replace("\n", " ")
                if self.async_client:
                    response = await self.async_client.embeddings.create(
                        model=self.model_name,
                        input=[text]
                    )
                    return response.data[0].embedding
                else:
                    # Fallback to sync in thread
                    logger.warning("Async client not available for OpenAI, falling back to sync")
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    return await loop.run_in_executor(None, self._generate_embedding, text)

            elif self.source == 'huggingface':
                # SentenceTransformer is sync, run in executor
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return await loop.run_in_executor(_executor, self._generate_embedding, text)

            else:
                raise ValueError(f"Unknown source: {self.source}")

        except Exception as e:
            logger.error(f"Error generating async embedding ({self.source}): {e}")
            raise e

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch генерация embeddings."""
        if not texts:
            return []

        try:
            if self.source in ('openai', 'openrouter', 'ollama'):
                # OpenAI поддерживает batch
                cleaned_texts = [t.replace("\n", " ") for t in texts]
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=cleaned_texts
                )
                # Гарантируем порядок
                return [item.embedding for item in response.data]

            elif self.source == 'huggingface':
                # SentenceTransformer поддерживает batch
                embeddings = self.client.encode(texts, normalize_embeddings=False)
                return [emb.tolist() for emb in embeddings]

            else:
                return [self._generate_embedding(t) for t in texts]

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Fallback to sequential
            return [self._generate_embedding(t) for t in texts]

    async def get_text_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Batch асинхронная генерация embeddings."""
        if not texts:
            return []

        try:
            if self.source in ('openai', 'openrouter', 'ollama'):
                # OpenAI поддерживает batch
                cleaned_texts = [t.replace("\n", " ") for t in texts]
                if self.async_client:
                    response = await self.async_client.embeddings.create(
                        model=self.model_name,
                        input=cleaned_texts
                    )
                    return [item.embedding for item in response.data]
                else:
                     try:
                         loop = asyncio.get_running_loop()
                     except RuntimeError:
                         loop = asyncio.new_event_loop()
                         asyncio.set_event_loop(loop)
                     return await loop.run_in_executor(None, self.get_text_embeddings, texts)

            elif self.source == 'huggingface':
                # SentenceTransformer supports batch but is CPU bound
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                # Используем executor для CPU-bound задачи
                return await loop.run_in_executor(_executor, self.get_text_embeddings, texts)

            else:
                # Fallback
                results = []
                for t in texts:
                    results.append(await self._generate_embedding_async(t))
                return results

        except Exception as e:
            logger.error(f"Error generating async batch embeddings: {e}")
            return [await self._generate_embedding_async(t) for t in texts]


    def get_embedding_dimension(self) -> int:
        return self._dimension

    @property
    def dimension(self) -> int:
        return self._dimension


def _determine_source() -> str:
    """Определяет источник embedding."""
    return settings.embedding_source


def _init_openai_embedding(
    api_base: str,
    api_key: str,
    model_name: str
) -> UnifiedEmbeddingModel:
    """Инициализация OpenAI/OpenRouter embedding."""
    # Validation for API Key
    if not api_key or api_key.strip() == "sk-..." or api_key.strip() == "":
        raise ValueError(
            "OPENAI_API_KEY is required for OpenAI/OpenRouter source. "
            "Set it in .env or environment variables."
        )

    from openai import OpenAI, AsyncOpenAI

    if api_base and not api_base.endswith('/v1'):
        api_base = f"{api_base}/v1"

    client = OpenAI(base_url=api_base, api_key=api_key)
    async_client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    logger.info(f"Testing connection to {api_base} with model {model_name}...")
    try:
        resp = client.embeddings.create(model=model_name, input=["test"])
        dim = len(resp.data[0].embedding)
    except Exception as e:
        logger.warning(f"Failed to test connection, using default dimension 1536: {e}")
        dim = 1536  # Fallback default

    return UnifiedEmbeddingModel('openai', model_name, dim, client, async_client)


def _init_ollama_embedding(
    api_base: str,
    model_name: str
) -> UnifiedEmbeddingModel:
    """Инициализация Ollama embedding."""
    from openai import OpenAI, AsyncOpenAI

    if not api_base.endswith('/v1'):
        api_base = f"{api_base}/v1"

    client = OpenAI(base_url=api_base, api_key="ollama")
    async_client = AsyncOpenAI(base_url=api_base, api_key="ollama")

    logger.info(f"Testing Ollama at {api_base} with model {model_name}...")
    try:
        resp = client.embeddings.create(model=model_name, input=["test"])
        dim = len(resp.data[0].embedding)
    except Exception as e:
        logger.warning(f"Failed to test Ollama connection, using default dimension 1024: {e}")
        dim = 1024  # Fallback default

    return UnifiedEmbeddingModel('ollama', model_name, dim, client, async_client)


def _init_huggingface_embedding(model_name: str) -> UnifiedEmbeddingModel:
    """Инициализация HuggingFace embedding."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading HuggingFace model: {model_name}...")
    client = SentenceTransformer(model_name)
    dim = client.get_sentence_embedding_dimension()

    # HuggingFace is sync-only, async_client is None
    return UnifiedEmbeddingModel('huggingface', model_name, dim, client, None)


def get_embed_model() -> UnifiedEmbeddingModel:
    """
    Фабрика для получения модели embeddings.
    """
    global _embed_model

    if _embed_model:
        return _embed_model

    with _model_lock:
        if _embed_model:
            return _embed_model

        start_time = time.time()
        source = _determine_source()

        logger.info(f"Initializing embedding model. Source: {source}")

        try:
            if source in ('openrouter', 'openai'):
                api_base = settings.openai_api_base
                if not api_base and source == 'openrouter':
                    api_base = "https://openrouter.ai/api/v1"
                
                model_name = settings.openai_model or settings.embed_model
                api_key = settings.openai_api_key
                _embed_model = _init_openai_embedding(api_base, api_key, model_name)

            elif source == 'ollama':
                model_name = settings.ollama_embedding_model or settings.ollama_model or settings.embed_model
                _embed_model = _init_ollama_embedding(settings.ollama_url, model_name)

            elif source == 'huggingface':
                _embed_model = _init_huggingface_embedding(settings.embed_model)

            else:
                raise ValueError(f"Unsupported embedding source: {source}")

            elapsed = time.time() - start_time
            logger.info(
                f"✅ Embedding model initialized in {elapsed:.2f}s. "
                f"Dimension: {_embed_model.dimension}"
            )
            return _embed_model

        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise RuntimeError(f"Failed to initialize embedding model: {e}")


def get_embedding_dimension() -> int:
    """Helper to get dimension."""
    model = get_embed_model()
    return model.get_embedding_dimension()


def generate_query_embedding(query: str) -> List[float]:
    """Helper to generate single embedding."""
    model = get_embed_model()
    return model.get_query_embedding(query)

async def generate_query_embedding_async(query: str) -> List[float]:
    """Helper to generate single embedding async."""
    model = get_embed_model()
    return await model.get_query_embedding_async(query)


def generate_query_embeddings_batch(queries: List[str]) -> List[List[float]]:
    """Helper to generate batch embeddings."""
    model = get_embed_model()
    return model.get_text_embeddings(queries)

async def generate_query_embeddings_batch_async(queries: List[str]) -> List[List[float]]:
    """Helper to generate batch embeddings async."""
    model = get_embed_model()
    return await model.get_text_embeddings_async(queries)
