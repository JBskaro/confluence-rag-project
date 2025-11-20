"""
Единый модуль для работы с embeddings.
Поддерживает HuggingFace, Ollama и OpenAI-compatible API (LM Studio, Ollama и др.).
Thread-safe кэширование моделей.
"""
import os
import logging
import threading
import time
from typing import List, Optional, Union, Any

# Отключаем избыточное логирование
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Конфигурация из ENV
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "ai-forever/FRIDA")

# OpenAI-compatible API
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")

# Источник embeddings
EMBEDDING_SOURCE = os.getenv('EMBEDDING_SOURCE', '').lower()

# Thread-safe кэширование
_model_lock = threading.Lock()
_embed_model = None


class UnifiedEmbeddingModel:
    """
    Унифицированный класс для работы с различными моделями embeddings.
    Заменяет LlamaIndex BaseEmbedding.
    """
    def __init__(self, source: str, model_name: str, dimension: int, client: Any = None):
        self.source = source
        self.model_name = model_name
        self._dimension = dimension
        self.client = client

    def get_query_embedding(self, query: str) -> List[float]:
        """Генерация embedding для запроса."""
        return self._generate_embedding(query)

    def get_text_embedding(self, text: str) -> List[float]:
        """Генерация embedding для текста."""
        return self._generate_embedding(text)

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
            # Возвращаем нулевой вектор в случае ошибки, чтобы не ронять процесс
            # (хотя лучше бы рейзить, но для надежности оставим так или добавим retry)
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

    def get_embedding_dimension(self) -> int:
        return self._dimension
        
    @property
    def dimension(self) -> int:
        return self._dimension


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
        source = EMBEDDING_SOURCE
        
        # Логика выбора источника (совместимость с legacy)
        if not source:
            if OPENAI_API_BASE:
                source = 'openai'
            elif USE_OLLAMA:
                source = 'ollama'
            else:
                source = 'huggingface'
        
        logger.info(f"Initializing embedding model. Source: {source}")

        try:
            if source == 'openrouter' or source == 'openai':
                from openai import OpenAI
                
                api_base = OPENAI_API_BASE
                if not api_base and source == 'openrouter':
                    api_base = "https://openrouter.ai/api/v1"
                
                if api_base and not api_base.endswith('/v1'):
                    api_base = f"{api_base}/v1"
                    
                api_key = OPENAI_API_KEY
                model_name = OPENAI_MODEL or EMBED_MODEL
                
                client = OpenAI(base_url=api_base, api_key=api_key)
                
                # Test & get dimension
                logger.info(f"Testing connection to {api_base} with model {model_name}...")
                resp = client.embeddings.create(model=model_name, input=["test"])
                dim = len(resp.data[0].embedding)
                
                _embed_model = UnifiedEmbeddingModel(source, model_name, dim, client)
                
            elif source == 'ollama':
                # Ollama совместима с OpenAI API
                from openai import OpenAI
                
                api_base = OLLAMA_URL
                if not api_base.endswith('/v1'):
                    api_base = f"{api_base}/v1"
                    
                model_name = os.getenv('OLLAMA_EMBEDDING_MODEL') or os.getenv('OLLAMA_MODEL') or EMBED_MODEL
                
                client = OpenAI(base_url=api_base, api_key="ollama")
                
                logger.info(f"Testing connection to Ollama at {api_base} with model {model_name}...")
                resp = client.embeddings.create(model=model_name, input=["test"])
                dim = len(resp.data[0].embedding)
                
                _embed_model = UnifiedEmbeddingModel(source, model_name, dim, client)
                
            elif source == 'huggingface':
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"Loading HuggingFace model: {EMBED_MODEL}...")
                client = SentenceTransformer(EMBED_MODEL)
                dim = client.get_sentence_embedding_dimension()
                
                _embed_model = UnifiedEmbeddingModel(source, EMBED_MODEL, dim, client)
                
            else:
                raise ValueError(f"Unsupported embedding source: {source}")
                
            elapsed = time.time() - start_time
            logger.info(f"✅ Embedding model initialized in {elapsed:.2f}s. Dimension: {_embed_model.dimension}")
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


def generate_query_embeddings_batch(queries: List[str]) -> List[List[float]]:
    """Helper to generate batch embeddings."""
    model = get_embed_model()
    return model.get_text_embeddings(queries)
