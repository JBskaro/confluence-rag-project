#!/usr/bin/env python3
"""
Semantic Query Log: 5-й источник для Query Expansion

Логирует успешные поисковые запросы пользователей и их результаты
для использования в расширении запросов.
"""

import json
import os
import logging
from typing import List, Dict, Tuple, Set
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class SemanticQueryLog:
    """
    Логирует успешные запросы и их результаты для Query Expansion.

    Отличается от QueryMiner:
    - QueryMiner: обучение на документах (статическое)
    - SemanticQueryLog: обучение на пользовательском поведении (динамическое)
    """

    def __init__(self, log_file: str = None):
        """
        Инициализация Semantic Query Log.

        Args:
            log_file: Путь к файлу лога (по умолчанию из ENV или ./data/query_log_semantic.json)
        """
        if log_file is None:
            data_dir = Path("./data")
            data_dir.mkdir(exist_ok=True)
            log_file = os.getenv('QUERY_LOG_FILE', str(data_dir / "query_log_semantic.json"))

        self.log_file = Path(log_file)
        self.min_rating = float(os.getenv('QUERY_LOG_MIN_RATING', '4.0'))
        self.max_log_size = int(os.getenv('QUERY_LOG_MAX_SIZE', '10000'))  # Макс 10K записей
        self.query_log = self._load_log()

        # Очистка при инициализации если лог слишком большой
        if len(self.query_log) > self.max_log_size:
            self._cleanup_old_entries()

        logger.info(f"Semantic Query Log инициализирован: {len(self.query_log)} записей (лимит: {self.max_log_size})")

    def _load_log(self) -> Dict:
        """Загрузить лог из файла."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить semantic query log: {e}")
                return {}
        return {}

    def _save_log(self):
        """Сохранить лог в файл."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.query_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения semantic query log: {e}")

    def _cleanup_old_entries(self):
        """Удалить старые/неуспешные записи если лог слишком большой."""
        original_size = len(self.query_log)

        # Удаляем записи с низким рейтингом и низкой популярностью
        self.query_log = {
            q: data for q, data in self.query_log.items()
            if (data.get('avg_rating', 0) > 2.0 or data.get('count', 0) > 5) and data.get('success', False)
        }

        # Если всё ещё слишком много, оставляем только топ по популярности
        if len(self.query_log) > self.max_log_size:
            sorted_entries = sorted(
                self.query_log.items(),
                key=lambda x: (x[1].get('avg_rating', 0), x[1].get('count', 0)),
                reverse=True
            )
            self.query_log = dict(sorted_entries[:self.max_log_size])

        cleaned = original_size - len(self.query_log)
        if cleaned > 0:
            logger.info(f"Semantic Query Log: очищено {cleaned} старых записей ({original_size} → {len(self.query_log)})")
            self._save_log()

    def log_query(self, query: str, results_count: int, user_rating: int = None):
        """
        Логировать запрос и его результаты.

        Args:
            query: Поисковый запрос
            results_count: Количество найденных результатов
            user_rating: Рейтинг пользователя (1-5 звёзд, опционально)
        """
        query_normalized = query.lower().strip()

        if query_normalized not in self.query_log:
            self.query_log[query_normalized] = {
                'count': 0,
                'ratings': [],
                'avg_rating': 0.0,
                'success': False,
                'results_count': 0
            }

        log_entry = self.query_log[query_normalized]
        log_entry['count'] += 1
        log_entry['results_count'] = max(log_entry['results_count'], results_count)

        if user_rating is not None and 1 <= user_rating <= 5:
            log_entry['ratings'].append(user_rating)
            log_entry['avg_rating'] = sum(log_entry['ratings']) / len(log_entry['ratings'])

        # Успешный запрос: есть результаты И (рейтинг >= min_rating ИЛИ нет рейтинга)
        log_entry['success'] = (
            results_count > 0 and
            (log_entry['avg_rating'] >= self.min_rating if log_entry['ratings'] else True)
        )

        # Сохраняем каждые 5 запросов
        if log_entry['count'] % 5 == 0:
            self._save_log()

        # Очистка если лог слишком большой
        if len(self.query_log) > self.max_log_size:
            self._cleanup_old_entries()

    def get_expansion_terms(self, top_n: int = 50) -> List[Tuple[str, int, float]]:
        """
        Получить успешные запросы для Query Expansion.

        Args:
            top_n: Максимальное количество запросов

        Returns:
            Список кортежей: [(query, count, avg_rating), ...]
        """
        successful = [
            (q, data['count'], data['avg_rating'])
            for q, data in self.query_log.items()
            if data['success']
        ]

        # Сортируем по количеству успешных использований и рейтингу
        successful.sort(key=lambda x: (x[2], x[1]), reverse=True)

        return successful[:top_n]

    def get_related_queries(self, query: str, top_n: int = 5) -> List[str]:
        """
        Найти похожие успешные запросы.

        Args:
            query: Текущий запрос
            top_n: Количество похожих запросов

        Returns:
            Список похожих запросов
        """
        query_words = set(query.lower().split())
        related = []

        for logged_query, data in self.query_log.items():
            if not data['success']:
                continue

            logged_words = set(logged_query.split())

            # Jaccard similarity
            intersection = query_words & logged_words
            union = query_words | logged_words

            if union:
                similarity = len(intersection) / len(union)

                # Порог похожести > 30%
                if similarity > 0.3:
                    related.append((logged_query, similarity, data['count'], data['avg_rating']))

        # Сортируем по похожести, затем по популярности и рейтингу
        related.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)

        return [q for q, _, _, _ in related[:top_n]]


# Глобальный экземпляр (ленивая инициализация)
_semantic_query_log_instance = None


def get_semantic_query_log() -> SemanticQueryLog:
    """Получить глобальный экземпляр SemanticQueryLog."""
    global _semantic_query_log_instance
    if _semantic_query_log_instance is None:
        _semantic_query_log_instance = SemanticQueryLog()
    return _semantic_query_log_instance

