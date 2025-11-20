#!/usr/bin/env python3
"""
Hallucination Detection для проверки ответов на соответствие источникам.

Методы:
1. Semantic similarity - проверка близости ответа к источникам
2. Keyword overlap - проверка наличия ключевых слов из источников
3. Confidence scoring - анализ уверенности модели
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Detector для проверки ответов на hallucinations.
    """

    # Константы для grounding check
    MIN_GROUNDED_RATIO = 0.5  # Минимальный ratio для grounded sentence
    MIN_WORD_LENGTH = 3  # Минимальная длина слова для проверки

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        keyword_overlap_threshold: float = 0.3
    ):
        """
        Initialize detector.

        Args:
            similarity_threshold: Минимальная similarity между ответом и источниками
            keyword_overlap_threshold: Минимальный overlap ключевых слов
        """
        self.similarity_threshold = similarity_threshold
        self.keyword_overlap_threshold = keyword_overlap_threshold

    def _validate_inputs(
        self,
        response: str,
        retrieved_docs: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Валидация входных данных."""
        if not response or not isinstance(response, str) or not response.strip():
            return {
                'is_hallucination': True,
                'confidence': 1.0,
                'reasons': ['Empty response'],
                'scores': {}
            }

        if not retrieved_docs:
            return {
                'is_hallucination': True,
                'confidence': 1.0,
                'reasons': ['No retrieved docs'],
                'scores': {}
            }
        return None

    def _compute_detection_scores(
        self,
        response: str,
        retrieved_docs: List[str],
        response_embedding: Optional[np.ndarray],
        docs_embeddings: Optional[List[np.ndarray]]
    ) -> Tuple[Dict[str, float], List[str]]:
        """Вычисление всех detection scores."""
        scores = {}
        reasons = []

        # 1. Semantic similarity
        if response_embedding is not None and docs_embeddings:
            try:
                semantic_score = self._check_semantic_similarity(
                    response_embedding,
                    docs_embeddings
                )
                scores['semantic_similarity'] = semantic_score

                if semantic_score < self.similarity_threshold:
                    reasons.append(
                        f"Low semantic similarity: {semantic_score:.2f} < {self.similarity_threshold}"
                    )
            except Exception as e:
                logger.debug(f"Semantic similarity check failed: {e}")

        # 2. Keyword overlap
        try:
            keyword_score = self._check_keyword_overlap(response, retrieved_docs)
            scores['keyword_overlap'] = keyword_score

            if keyword_score < self.keyword_overlap_threshold:
                reasons.append(
                    f"Low keyword overlap: {keyword_score:.2f} < {self.keyword_overlap_threshold}"
                )
        except Exception as e:
            logger.debug(f"Keyword overlap check failed: {e}")

        # 3. Grounding check
        try:
            grounded_ratio = self._check_grounding(response, retrieved_docs)
            scores['grounded_ratio'] = grounded_ratio

            if grounded_ratio < 0.5:  # Меньше половины контента grounded
                reasons.append(
                    f"Low grounding: {grounded_ratio:.2f} < 0.5"
                )
        except Exception as e:
            logger.debug(f"Grounding check failed: {e}")

        return scores, reasons

    def _make_decision(
        self,
        scores: Dict[str, float],
        reasons: List[str]
    ) -> Tuple[bool, float]:
        """Принимает решение о hallucination."""
        is_hallucination = len(reasons) >= 2  # Если 2+ красных флага

        # Confidence: среднее значение всех scores
        all_scores = list(scores.values())
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            confidence = 1.0 - avg_score  # Higher score = lower confidence (hallucination)
        else:
            confidence = 1.0  # No scores = high confidence in hallucination

        return is_hallucination, confidence

    def detect(
        self,
        query: str,
        response: str,
        retrieved_docs: List[str],
        response_embedding: Optional[np.ndarray] = None,
        docs_embeddings: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Detect hallucinations в response.

        Args:
            query: Исходный query
            response: Ответ модели
            retrieved_docs: Список документов из retrieval
            response_embedding: Embedding ответа (optional)
            docs_embeddings: Embeddings документов (optional)

        Returns:
            Dict с результатами проверки
        """
        # 1. Валидация
        validation_result = self._validate_inputs(response, retrieved_docs)
        if validation_result:
            return validation_result

        # 2. Вычисление scores
        scores, reasons = self._compute_detection_scores(
            response, retrieved_docs, response_embedding, docs_embeddings
        )

        # 3. Решение
        is_hallucination, confidence = self._make_decision(scores, reasons)

        # 4. Логирование и результат
        result = {
            'is_hallucination': is_hallucination,
            'confidence': confidence,
            'reasons': reasons,
            'scores': scores
        }

        if is_hallucination:
            logger.warning(f"⚠️ Hallucination detected: {reasons}")
        else:
            logger.debug(f"✅ Response seems grounded (scores: {scores})")

        return result

    def _check_semantic_similarity(
        self,
        response_embedding: np.ndarray,
        docs_embeddings: List[np.ndarray]
    ) -> float:
        """Check semantic similarity между response и docs."""
        try:
            # Normalize embeddings
            response_norm = np.linalg.norm(response_embedding)
            if response_norm == 0:
                return 0.0
            response_norm = response_embedding / response_norm

            similarities = []
            for doc_emb in docs_embeddings:
                doc_emb = np.array(doc_emb)
                doc_norm = np.linalg.norm(doc_emb)
                if doc_norm == 0:
                    continue
                doc_norm = doc_emb / doc_norm
                sim = float(np.dot(response_norm, doc_norm))
                similarities.append(sim)

            # Return maximum similarity (best match)
            return max(similarities) if similarities else 0.0

        except Exception as e:
            logger.error(f"Error in semantic similarity check: {e}")
            return 0.0

    def _check_keyword_overlap(
        self,
        response: str,
        retrieved_docs: List[str]
    ) -> float:
        """Check keyword overlap между response и docs."""
        try:
            # Extract keywords (simple: words > 3 chars)
            response_words = set(
                word.lower() for word in response.split()
                if len(word) > 3 and word.isalnum()
            )

            docs_text = ' '.join(retrieved_docs)
            docs_words = set(
                word.lower() for word in docs_text.split()
                if len(word) > 3 and word.isalnum()
            )

            if not response_words:
                return 0.0

            # Calculate overlap
            overlap = response_words & docs_words
            overlap_ratio = len(overlap) / len(response_words)

            return overlap_ratio

        except Exception as e:
            logger.error(f"Error in keyword overlap check: {e}")
            return 0.0

    def _check_grounding(
        self,
        response: str,
        retrieved_docs: List[str]
    ) -> float:
        """Check сколько контента из response присутствует в docs."""
        try:
            # Split response на sentences
            sentences = [s.strip() for s in response.split('.') if s.strip()]

            if not sentences:
                return 0.0

            docs_text = ' '.join(retrieved_docs).lower()

            # Предварительно строим set для O(1) lookup (оптимизация)
            docs_words_set = set(
                w for w in docs_text.split()
                if len(w) > HallucinationDetector.MIN_WORD_LENGTH
            )

            # Check сколько sentences grounded
            grounded_count = 0
            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Простая проверка: есть ли 50%+ слов sentence в docs
                words = [
                    w for w in sentence_lower.split()
                    if len(w) > HallucinationDetector.MIN_WORD_LENGTH
                ]
                if not words:
                    continue

                found_words = sum(1 for w in words if w in docs_words_set)
                if found_words / len(words) >= HallucinationDetector.MIN_GROUNDED_RATIO:
                    grounded_count += 1

            grounded_ratio = grounded_count / len(sentences)

            return grounded_ratio

        except Exception as e:
            logger.error(f"Error in grounding check: {e}")
            return 0.0


# === HELPER FUNCTION ===

def detect_hallucination(
    response: str,
    retrieved_docs: List[str],
    response_embedding: Optional[np.ndarray] = None,
    docs_embeddings: Optional[List[np.ndarray]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function для hallucination detection.

    Returns:
        (is_hallucination, details)
    """
    detector = HallucinationDetector()

    result = detector.detect(
        query="",  # Not used in basic detection
        response=response,
        retrieved_docs=retrieved_docs,
        response_embedding=response_embedding,
        docs_embeddings=docs_embeddings
    )

    return result['is_hallucination'], result

