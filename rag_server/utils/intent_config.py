#!/usr/bin/env python3
"""
Конфигурация для классификации намерений запросов и адаптивных параметров поиска.

Определяет типы запросов и адаптивные пороги reranking для каждого типа.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class QueryIntent(Enum):
    """Тип намерения запроса"""
    NAVIGATIONAL = "navigational"  # Поиск конкретной страницы/документа
    HOWTO = "howto"                # Инструкции (как сделать)
    FACTUAL = "factual"            # Фактический поиск (что, когда, кто)
    EXPLORATORY = "exploratory"    # Исследовательский поиск (список, сравнение)


@dataclass
class IntentConfig:
    """Конфигурация параметров для типа запроса"""
    rerank_threshold: float      # Порог reranking score
    diversity_limit: int         # Лимит чанков с одной страницы
    expand_context: bool         # Расширять ли контекст
    boost_hierarchy: bool       # Усиливать ли иерархию
    context_window: int         # Размер окна контекста


def get_intent_config(intent_type: str, reranker_model: str = None) -> IntentConfig:
    """
    Получить конфигурацию для типа запроса с адаптивными порогами.
    
    Args:
        intent_type: Тип запроса ('navigational', 'howto', 'factual', 'exploratory')
        reranker_model: Модель reranker (для определения диапазона scores)
        
    Returns:
        IntentConfig с адаптивными параметрами
    """
    if reranker_model is None:
        reranker_model = os.getenv('RE_RANKER_MODEL', 'DiTy/cross-encoder-russian-msmarco')
    
    # Определяем базовые пороги в зависимости от модели
    is_bge_reranker = 'bge-reranker' in reranker_model.lower()
    
    # Базовые пороги из ENV или дефолты
    if is_bge_reranker:
        # BAAI/bge-reranker-v2-m3: scores в диапазоне 0.0-1.0 (обычно 0.5-1.0 для релевантных)
        base_technical = float(os.getenv('RERANK_THRESHOLD_TECHNICAL', '0.01'))
        base_general = float(os.getenv('RERANK_THRESHOLD_GENERAL', '0.001'))
    else:
        # DiTy/cross-encoder-russian-msmarco: scores в диапазоне 0.001-0.295
        base_technical = float(os.getenv('RERANK_THRESHOLD_TECHNICAL', '0.01'))
        base_general = float(os.getenv('RERANK_THRESHOLD_GENERAL', '0.005'))
    
    # Адаптивные пороги для каждого типа запроса
    intent_lower = intent_type.lower()
    
    if intent_lower == 'navigational':
        # Навигационные: нужны точные совпадения, можно использовать более высокий порог
        return IntentConfig(
            rerank_threshold=base_general * 1.5,  # Чуть выше для точности
            diversity_limit=1,                    # По 1 чанку (показать больше страниц)
            expand_context=False,                  # Не нужен полный контекст
            boost_hierarchy=True,                  # Важны корневые страницы
            context_window=2
        )
    
    elif intent_lower == 'howto':
        # How-to: нужен детальный контекст, более мягкий порог
        return IntentConfig(
            rerank_threshold=base_technical,       # Мягкий порог (технические термины)
            diversity_limit=3,                     # До 3 чанков (детальная инструкция)
            expand_context=True,                   # Нужен полный контекст
            boost_hierarchy=False,
            context_window=3
        )
    
    elif intent_lower == 'factual':
        # Фактические: нужен контекст, стандартный порог
        return IntentConfig(
            rerank_threshold=base_general,         # Стандартный порог
            diversity_limit=3,                     # До 3 чанков (может быть в разных местах)
            expand_context=True,                   # Нужен контекст для ответа
            boost_hierarchy=False,
            context_window=3
        )
    
    elif intent_lower == 'exploratory':
        # Исследовательские: нужен широкий охват, ОЧЕНЬ мягкий порог
        # КРИТИЧНО: Для exploratory используем минимальный порог!
        if is_bge_reranker:
            # Для bge-reranker: используем очень низкий порог (0.0001 или даже ниже)
            exploratory_threshold = float(os.getenv('RERANK_THRESHOLD_EXPLORATORY', '0.0001'))
        else:
            # Для других моделей: используем базовый general или еще ниже
            exploratory_threshold = base_general * 0.5  # В 2 раза ниже general
        
        return IntentConfig(
            rerank_threshold=exploratory_threshold, # ОЧЕНЬ мягкий порог для широкого охвата
            diversity_limit=2,                     # Стандартный лимит
            expand_context=True,                   # Нужен контекст для исследования
            boost_hierarchy=False,
            context_window=4                       # Больше контекста для исследования
        )
    
    else:
        # Fallback: используем стандартные значения
        return IntentConfig(
            rerank_threshold=base_general,
            diversity_limit=2,
            expand_context=True,
            boost_hierarchy=False,
            context_window=2
        )


def get_adaptive_rerank_threshold(intent_type: str, is_technical: bool = False, reranker_model: str = None) -> float:
    """
    Получить адаптивный порог reranking на основе типа запроса.
    
    Args:
        intent_type: Тип запроса
        is_technical: Является ли запрос техническим
        reranker_model: Модель reranker
        
    Returns:
        Порог reranking score
    """
    config = get_intent_config(intent_type, reranker_model)
    
    # Если технический запрос, используем технический порог из конфига
    if is_technical:
        # Для технических запросов используем более мягкий порог из конфига
        # но не ниже чем exploratory threshold
        base_technical = float(os.getenv('RERANK_THRESHOLD_TECHNICAL', '0.01'))
        return min(config.rerank_threshold, base_technical)
    
    return config.rerank_threshold


def get_adaptive_context_window(intent_type: str) -> int:
    """
    Получить адаптивный размер окна контекста для типа запроса.
    
    Args:
        intent_type: Тип запроса
        
    Returns:
        Размер окна контекста
    """
    config = get_intent_config(intent_type)
    return config.context_window

