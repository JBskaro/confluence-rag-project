#!/usr/bin/env python3
"""
MMR (Maximal Marginal Relevance) для диверсификации результатов.

Балансирует relevance vs diversity для уменьшения redundancy.
"""

import os
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    results: List[Dict[str, Any]],
    lambda_mult: float = 0.5,
    k: int = 10,
    embedding_key: str = 'embedding'
) -> List[Dict[str, Any]]:
    """
    Применить MMR для диверсификации результатов (векторизованная версия).

    Формула: MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected))

    Оптимизация: O(k×n×m) вместо O(k×n²) через матричные операции.

    Args:
        query_embedding: Embedding query
        results: Список результатов с embeddings
        lambda_mult: Balance relevance vs diversity (0-1)
                    0 = max diversity, 1 = max relevance
        k: Количество результатов
        embedding_key: Ключ для embedding в result dict

    Returns:
        Топ-k результатов с максимальной MMR score
    """
    import time
    start_time = time.time()

    if not results or k <= 0:
        return []

    if len(results) <= k:
        return results  # Не нужна фильтрация

    # Защита от memory exhaustion
    MAX_MMR_ITEMS = int(os.getenv('MAX_MMR_ITEMS', '100'))
    if k > MAX_MMR_ITEMS:
        logger.warning(f"MMR k={k} too large, limiting to {MAX_MMR_ITEMS}")
        k = MAX_MMR_ITEMS

    try:
        # Проверяем что есть embeddings
        if not all(embedding_key in r for r in results):
            logger.warning("⚠️ Some results missing embeddings, skipping MMR")
            return results[:k]

        # Предвычисляем все embeddings в матрицу
        embeddings_list = []
        valid_results = []
        for r in results:
            try:
                emb = np.array(r[embedding_key], dtype=np.float32)
                emb_norm = np.linalg.norm(emb)
                if emb_norm > 0:
                    embeddings_list.append(emb / emb_norm)
                    valid_results.append(r)
            except (KeyError, ValueError, TypeError):
                continue

        if not embeddings_list:
            logger.warning("⚠️ No valid embeddings found, skipping MMR")
            return results[:k]

        # Нормализуем query embedding один раз
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            logger.warning("⚠️ Zero query embedding, skipping MMR")
            return results[:k]
        query_embedding = query_embedding / query_norm

        # Векторизованная матрица embeddings
        embeddings_matrix = np.vstack(embeddings_list)

        selected = []
        remaining_indices = list(range(len(valid_results)))
        remaining_embeddings = embeddings_matrix.copy()

        while len(selected) < k and len(remaining_indices) > 0:
            # Векторизованное вычисление relevance (O(n))
            relevance = remaining_embeddings @ query_embedding

            if selected:
                # Матричное вычисление max_sim (O(n×m))
                selected_embeddings = np.vstack([
                    np.array(valid_results[s][embedding_key], dtype=np.float32)
                    for s in selected
                ])
                # Нормализуем selected embeddings
                selected_norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
                selected_embeddings = selected_embeddings / (selected_norms + 1e-8)

                # Матричное умножение для similarity
                similarity = remaining_embeddings @ selected_embeddings.T  # (n, m)
                max_sim = similarity.max(axis=1)  # O(n)
            else:
                max_sim = np.zeros(len(remaining_indices))

            # Векторизованный MMR score
            mmr_scores = lambda_mult * relevance - (1 - lambda_mult) * max_sim

            # Выбираем лучший
            best_idx = mmr_scores.argmax()
            selected_idx = remaining_indices[best_idx]
            selected.append(selected_idx)

            # Удаляем из remaining
            remaining_indices.pop(best_idx)
            remaining_embeddings = np.delete(remaining_embeddings, best_idx, axis=0)

        # Возвращаем результаты в порядке выбора
        selected_results = [valid_results[s] for s in selected]

        # Профилирование для мониторинга
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"✅ MMR diversification (vectorized): "
            f"{len(results)} → {len(selected_results)} results "
            f"(lambda={lambda_mult}, latency={elapsed_ms:.1f}ms)"
        )

        return selected_results

    except Exception as e:
        logger.error(f"❌ MMR failed: {e}", exc_info=True)
        return results[:k]  # Fallback


def mmr_rerank(
    query_embedding: np.ndarray,
    results: List[Dict[str, Any]],
    diversity_weight: float = 0.3,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper для MMR reranking.

    Args:
        query_embedding: Query embedding
        results: Search results
        diversity_weight: Вес diversity (0-1), default 0.3 = 30% diversity
        top_k: Количество результатов

    Returns:
        Reranked results
    """
    # lambda_mult = 1 - diversity_weight
    # Если diversity_weight=0.3, то lambda_mult=0.7 (70% relevance, 30% diversity)
    lambda_mult = max(0.0, min(1.0, 1.0 - diversity_weight))

    return maximal_marginal_relevance(
        query_embedding=query_embedding,
        results=results,
        lambda_mult=lambda_mult,
        k=top_k
    )

