import os
import logging
import time
import asyncio
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models

from embeddings import generate_query_embeddings_batch, generate_query_embeddings_batch_async
from observability import tracer
# Metrics will be imported from observability once added there
try:
    from observability import (
        SEARCH_LATENCY,
        VECTOR_SEARCH_LATENCY,
        RERANK_LATENCY,
        SEARCH_REQUESTS,
        QDRANT_CONNECTION_ERRORS
    )
except ImportError:
    # Fallback if metrics are not yet defined
    SEARCH_LATENCY = None
    VECTOR_SEARCH_LATENCY = None
    RERANK_LATENCY = None
    SEARCH_REQUESTS = None
    QDRANT_CONNECTION_ERRORS = None

logger = logging.getLogger(__name__)

@dataclass
class SearchParams:
    """Валидированные параметры поиска"""
    query: str
    space: Optional[str] = None
    limit: int = 5
    use_reranking: bool = True
    threshold: float = 0.6
    diversity_limit: int = 999
    expanded_queries: List[str] = field(default_factory=list)
    enable_parallel: bool = True

@dataclass
class SearchResult:
    """Структурированный результат поиска"""
    text: str
    metadata: Dict[str, Any]
    score: float
    rerank_score: Optional[float] = None
    expanded_text: Optional[str] = None

class SearchPipeline:
    """Orchestrator для поискового пайплайна (Async)"""

    def __init__(self, qdrant_client: QdrantClient, collection_name: str, reranker=None):
        self.qdrant_client = qdrant_client
        # Try to init async client if possible, or assume it's passed or available globally
        # For simplicity, we will use the one from qdrant_storage if not provided
        self.async_qdrant_client = None 
        try:
            from qdrant_storage import init_async_qdrant_client
            self.async_qdrant_client = init_async_qdrant_client()
        except ImportError:
            pass

        self.collection_name = collection_name
        self.reranker = reranker

    def execute(self, params: SearchParams) -> List[Dict[str, Any]]:
        """Sync wrapper for backward compatibility"""
        return asyncio.run(self.execute_async(params))

    async def execute_async(self, params: SearchParams) -> List[Dict[str, Any]]:
        """
        Основной pipeline выполнения поиска (Async).
        
        Steps:
        1. Validation
        2. Embedding Generation (Batch Async)
        3. Parallel Vector Search (Async)
        4. Deduplication
        5. Reranking (optional)
        6. Formatting
        """
        start_time = time.time()

        with tracer.start_as_current_span("search_pipeline.execute_async") as span:
            span.set_attribute("query", params.query)
            span.set_attribute("space", params.space or "all")
            span.set_attribute("limit", params.limit)
            span.set_attribute("expanded_queries_count", len(params.expanded_queries))

            if SEARCH_REQUESTS:
                SEARCH_REQUESTS.inc()

            try:
                # 1. Prepare queries (original + expanded)
                queries = [params.query]
                if params.expanded_queries:
                    queries.extend(params.expanded_queries)

                # Remove duplicates and empty strings
                queries = list(dict.fromkeys([q for q in queries if q and q.strip()]))

                # 2. Generate Embeddings (Batch Async)
                embeddings = await self._get_embeddings_batch_async(queries)

                # 3. Parallel Vector Search (Async)
                raw_results = await self._parallel_search_async(queries, embeddings, params)

                # 4. Deduplication
                unique_results = self._deduplicate(raw_results)

                # 5. Reranking (Async/Threaded)
                if params.use_reranking and self.reranker and unique_results:
                    unique_results = await self._rerank_async(params.query, unique_results)
                else:
                    # Sort by vector score if no reranking
                    unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)

                # 6. Final Filtering
                final_results = unique_results[:params.limit]

                if SEARCH_LATENCY:
                    SEARCH_LATENCY.observe(time.time() - start_time)

                return final_results

            except Exception as e:
                logger.error(f"Search pipeline failed: {e}", exc_info=True)
                span.record_exception(e)
                raise

    async def _get_embeddings_batch_async(self, queries: List[str]) -> List[List[float]]:
        """Batch generation of embeddings (Async)"""
        with tracer.start_as_current_span("generate_embeddings_batch_async"):
            return await generate_query_embeddings_batch_async(queries)

    async def _parallel_search_async(self, queries: List[str], embeddings: List[List[float]], params: SearchParams) -> List[Dict[str, Any]]:
        """Execute search for multiple queries in parallel (Async)"""
        if not params.enable_parallel or len(queries) <= 1:
            results = []
            for i, query in enumerate(queries):
                results.extend(await self._single_search_async(query, embeddings[i], params))
            return results

        start_time = time.time()
        
        with tracer.start_as_current_span("parallel_search_async") as span:
            # Create tasks for all queries
            tasks = [
                self._single_search_async(queries[i], embeddings[i], params)
                for i in range(len(queries))
            ]
            
            # Wait for all tasks
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_results = []
            for res in results_list:
                if isinstance(res, Exception):
                    logger.error(f"Search failed for query variant: {res}")
                else:
                    all_results.extend(res)

            if VECTOR_SEARCH_LATENCY:
                VECTOR_SEARCH_LATENCY.observe(time.time() - start_time)

            return all_results

    async def _single_search_async(self, query_text: str, embedding: List[float], params: SearchParams) -> List[Dict[str, Any]]:
        """Single vector search execution (Async)"""
        # Build filter
        query_filter = None
        if params.space:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="space",
                        match=models.MatchValue(value=params.space)
                    )
                ]
            )

        # Search limit (fetch more for reranking)
        search_limit = params.limit * 3 if params.use_reranking else params.limit

        try:
            if not self.async_qdrant_client:
                 logger.warning("Async Qdrant client not available, using sync in thread")
                 loop = asyncio.get_event_loop()
                 # Fallback to sync client in thread
                 points = await loop.run_in_executor(
                     None, 
                     lambda: self.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=embedding,
                        query_filter=query_filter,
                        limit=search_limit,
                        score_threshold=params.threshold if not params.use_reranking else 0.0
                     )
                 )
            else:
                points = await self.async_qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=embedding,
                    query_filter=query_filter,
                    limit=search_limit,
                    score_threshold=params.threshold if not params.use_reranking else 0.0
                )

            results = []
            for point in points:
                payload = point.payload or {}
                results.append({
                    "text": payload.get("text", ""),
                    "metadata": payload,
                    "score": point.score,
                    "id": point.id,
                    "query_variant": query_text
                })
            return results

        except Exception as e:
            if QDRANT_CONNECTION_ERRORS:
                QDRANT_CONNECTION_ERRORS.inc()
            logger.error(f"Qdrant search error: {e}")
            return []

    def _deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate results by ID"""
        seen_ids = set()
        unique_results = []
        for r in results:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                unique_results.append(r)
        return unique_results

    async def _rerank_async(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using CrossEncoder (Async wrapper for CPU bound task)"""
        if not results:
            return []

        start_time = time.time()
        with tracer.start_as_current_span("rerank_results_async") as span:
            try:
                pairs = [(query, r["text"]) for r in results]
                
                # CrossEncoder is CPU bound, run in executor
                loop = asyncio.get_event_loop()
                scores = await loop.run_in_executor(None, self.reranker.predict, pairs)

                for i, score in enumerate(scores):
                    results[i]["rerank_score"] = float(score)
                    results[i]["boosted_score"] = float(score) # Alias

                results.sort(key=lambda x: x["rerank_score"], reverse=True)

                if RERANK_LATENCY:
                    RERANK_LATENCY.observe(time.time() - start_time)

                return results

            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                return results
