#!/usr/bin/env python3
"""
Продвинутые техники поиска:
1. Pseudo-Relevance Feedback (PRF)
2. Query Rewriting с Ollama
3. Fallback Search (многоуровневый)
"""

import os
import logging
import re
from typing import List, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)

# Конфигурация
USE_OLLAMA_FOR_QUERY_EXPANSION = os.getenv("USE_OLLAMA_FOR_QUERY_EXPANSION", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
ENABLE_PRF_FALLBACK = os.getenv("ENABLE_PRF_FALLBACK", "true").lower() == "true"


def extract_keywords(text: str, min_length: int = 3) -> list:
    """
    Извлекает ключевые слова из текста.
    
    Args:
        text: Исходный текст
        min_length: Минимальная длина слова
    
    Returns:
        Список ключевых слов
    """
    # Приводим к нижнему регистру
    text = text.lower()
    
    # Стоп-слова (русские и английские)
    stop_words = {
        # Русские
        'в', 'на', 'и', 'с', 'по', 'для', 'как', 'что', 'это', 'или', 'а', 'но',
        'из', 'к', 'о', 'от', 'до', 'за', 'под', 'над', 'при', 'про', 'через',
        'без', 'у', 'об', 'не', 'ни', 'то', 'же', 'бы', 'ли', 'уже', 'еще',
        # Английские
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'
    }
    
    # Извлекаем слова (кириллица и латиница)
    words = re.findall(r'[а-яёa-z0-9]+', text)
    
    # Фильтруем стоп-слова и короткие слова
    keywords = [w for w in words if w not in stop_words and len(w) >= min_length]
    
    return keywords


def pseudo_relevance_feedback(
    query: str,
    initial_results: list,
    top_k: int = 3,
    max_terms: int = 5
) -> str:
    """
    Pseudo-Relevance Feedback (PRF).
    
    Извлекает ключевые термины из топ-K результатов и расширяет запрос.
    
    Техника из Information Retrieval:
    1. Выполняем первый поиск
    2. Извлекаем термины из топ-K результатов
    3. Расширяем запрос этими терминами
    4. Выполняем второй поиск
    
    Args:
        query: Исходный запрос
        initial_results: Результаты первого поиска
        top_k: Количество топ результатов для анализа
        max_terms: Максимальное количество терминов для добавления
    
    Returns:
        Расширенный запрос
    """
    if not initial_results or len(initial_results) < 1:
        logger.debug("PRF: Недостаточно результатов для анализа")
        return query
    
    logger.info(f"🔍 PRF: Анализирую топ-{min(top_k, len(initial_results))} результатов...")
    
    # Извлекаем текст из топ-K результатов
    top_results = initial_results[:top_k]
    combined_text = ' '.join([r.get('text', '') for r in top_results])
    
    # Извлекаем ключевые слова
    keywords = extract_keywords(combined_text)
    
    # Подсчитываем частоту
    word_freq = Counter(keywords)
    
    # Убираем слова, которые уже есть в запросе
    query_words = set(extract_keywords(query))
    new_terms = [
        word for word, count in word_freq.most_common(max_terms * 2)
        if word not in query_words
    ]
    
    # Берем топ-N новых терминов
    new_terms = new_terms[:max_terms]
    
    if new_terms:
        expanded_query = f"{query} {' '.join(new_terms)}"
        logger.info(f"✅ PRF: Добавлены термины: {new_terms}")
        logger.info(f"   Исходный запрос: '{query}'")
        logger.info(f"   Расширенный: '{expanded_query}'")
        return expanded_query
    else:
        logger.debug("PRF: Новых терминов не найдено")
        return query


def rewrite_query_with_ollama(query: str) -> List[str]:
    """
    Переписывает запрос с помощью Ollama для генерации альтернативных формулировок.
    
    Args:
        query: Исходный запрос
    
    Returns:
        Список альтернативных формулировок (включая исходный запрос)
    """
    if not USE_OLLAMA_FOR_QUERY_EXPANSION:
        return [query]
    
    try:
        import requests
        
        prompt = f"""Сгенерируй 2 альтернативных варианта этого поискового запроса, используя синонимы и перефразирование. Запросы должны быть на том же языке, что и исходный.

Исходный запрос: {query}

Варианты (только текст, без нумерации и пояснений):"""
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 100
                }
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            # Парсим ответ (разделяем по переносам строк)
            variants = [line.strip() for line in generated_text.split('\n') if line.strip()]
            
            # Убираем нумерацию если есть (1., 2., -, *, etc.)
            variants = [re.sub(r'^[\d\.\-\*\)]+\s*', '', v) for v in variants]
            
            # Фильтруем пустые и слишком короткие
            variants = [v for v in variants if len(v) > 5]
            
            if variants:
                logger.info(f"✅ Ollama сгенерировал {len(variants)} вариантов запроса")
                logger.debug(f"   Варианты: {variants}")
                return [query] + variants[:2]  # Исходный + 2 варианта
            else:
                logger.warning("Ollama вернул пустой ответ")
                return [query]
        else:
            logger.warning(f"Ollama вернул ошибку: {response.status_code}")
            return [query]
            
    except requests.exceptions.Timeout:
        logger.warning("Ollama timeout (10 сек)")
        return [query]
    except Exception as e:
        logger.warning(f"Ошибка при обращении к Ollama: {e}")
        return [query]


class FallbackSearch:
    """
    Многоуровневая стратегия поиска с постепенным ослаблением критериев.
    
    Уровни:
    1. Поиск с фильтром space (если указан)
    2. Поиск без фильтра space
    3. Поиск с PRF (Pseudo-Relevance Feedback)
    4. Поиск с пониженным порогом релевантности
    """
    
    def __init__(self, min_results: int = 3):
        """
        Args:
            min_results: Минимальное количество результатов для успешного поиска
        """
        self.min_results = min_results
        logger.info(f"✅ FallbackSearch инициализирован (min_results={min_results})")
    
    def should_apply_fallback(self, results: list, level: int = 1) -> bool:
        """
        Определяет, нужно ли применять fallback.
        
        Args:
            results: Текущие результаты
            level: Уровень fallback (1, 2, 3, 4)
        
        Returns:
            True если нужен fallback
        """
        if not results:
            return True
        
        if len(results) < self.min_results:
            return True
        
        return False
    
    def get_fallback_message(self, level: int, original_space: str = "") -> str:
        """
        Генерирует сообщение о примененном fallback.
        
        Args:
            level: Уровень fallback
            original_space: Исходный space фильтр
        
        Returns:
            Сообщение для пользователя
        """
        messages = {
            1: f"⚠️ В space '{original_space}' найдено мало результатов. Показываю из всех spaces.",
            2: f"⚠️ Применен Pseudo-Relevance Feedback для улучшения результатов.",
            3: f"⚠️ Применен пониженный порог релевантности для расширения результатов."
        }
        
        return messages.get(level, "")


# Глобальный экземпляр
_fallback_search = None

def get_fallback_search(min_results: int = 3) -> FallbackSearch:
    """Получает глобальный экземпляр FallbackSearch."""
    global _fallback_search
    if _fallback_search is None:
        _fallback_search = FallbackSearch(min_results=min_results)
    return _fallback_search

