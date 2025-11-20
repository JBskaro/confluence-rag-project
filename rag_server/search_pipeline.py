import os
import logging
import time
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from qdrant_client import QdrantClient
from qdrant_client.http import models

from embeddings import generate_query_embedding, generate_query_embeddings_batch
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
    """Orchestrator для поискового пайплайна"""

    def __init__(self, qdrant_client: QdrantClient, collection_name: str, reranker=None):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.reranker = reranker

    def execute(self, params: SearchParams) -> List[Dict[str, Any]]:
        """
        Основной pipeline выполнения поиска.

        Steps:
        1. Validation
        2. Embedding Generation (Batch)
        3. Parallel Vector Search
        4. Deduplication
        5. Reranking (optional)
        6. Formatting
        """
        start_time = time.time()

        with tracer.start_as_current_span("search_pipeline.execute") as span:
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

                # 2. Generate Embeddings (Batch)
                embeddings = self._get_embeddings_batch(queries)

                # 3. Parallel Vector Search
                raw_results = self._parallel_search(queries, embeddings, params)

                # 4. Deduplication
                unique_results = self._deduplicate(raw_results)

                # 5. Reranking
                if params.use_reranking and self.reranker and unique_results:
                    unique_results = self._rerank(params.query, unique_results)
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

    def _get_embeddings_batch(self, queries: List[str]) -> List[List[float]]:
        """Batch generation of embeddings"""
        with tracer.start_as_current_span("generate_embeddings_batch"):
            return generate_query_embeddings_batch(queries)

    def _parallel_search(self, queries: List[str], embeddings: List[List[float]], params: SearchParams) -> List[Dict[str, Any]]:
        """Execute search for multiple queries in parallel"""
        if not params.enable_parallel or len(queries) <= 1:
            results = []
            for i, query in enumerate(queries):
                results.extend(self._single_search(query, embeddings[i], params))
            return results

        start_time = time.time()
        all_results = []
        max_workers = min(len(queries), int(os.getenv('PARALLEL_SEARCH_MAX_WORKERS', '4')))

        with tracer.start_as_current_span("parallel_search") as span:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._single_search, queries[i], embeddings[i], params): queries[i]
                    for i in range(len(queries))
                }

                for future in as_completed(futures):
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        logger.error(f"Search failed for query variant: {e}")

            if VECTOR_SEARCH_LATENCY:
                VECTOR_SEARCH_LATENCY.observe(time.time() - start_time)

            return all_results

    def _single_search(self, query_text: str, embedding: List[float], params: SearchParams) -> List[Dict[str, Any]]:
        """Single vector search execution"""
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
            points = self.qdrant_client.search(
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

    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using CrossEncoder"""
        if not results:
            return []

        start_time = time.time()
        with tracer.start_as_current_span("rerank_results") as span:
            try:
                pairs = [(query, r["text"]) for r in results]
                scores = self.reranker.predict(pairs)

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
