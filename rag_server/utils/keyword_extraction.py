#!/usr/bin/env python3
"""
Централизованный модуль для извлечения ключевых слов из запросов и текста.

Унифицирует логику извлечения ключевых слов, используемую в различных частях системы.
"""

import re
from typing import List, Set

# Стоп-слова (русские и английские)
STOPWORDS: Set[str] = {
    # Русские
    'в', 'на', 'и', 'с', 'по', 'для', 'как', 'что', 'это', 'или', 'а', 'но',
    'из', 'к', 'о', 'от', 'до', 'за', 'под', 'над', 'при', 'про', 'через',
    'без', 'у', 'об', 'не', 'ни', 'то', 'же', 'бы', 'ли', 'уже', 'еще',
    'какой', 'какая', 'какие', 'где', 'когда', 'кто', 'сколько',
    # Английские
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
    'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'
}

# Технические термины (для определения технических запросов)
TECHNICAL_TERMS: Set[str] = {
    'api', 'http', 'rest', 'json', 'xml', 'sql', 'docker', 'git', '1с', '1c',
    'endpoint', 'webhook', 'oauth', 'deployment', 'ssl', 'тест', 'баг',
    'конфигурация', 'python', 'javascript', 'typescript', 'java', 'c++',
    'react', 'vue', 'angular', 'node', 'npm', 'yarn', 'pip', 'conda',
    'kubernetes', 'k8s', 'terraform', 'ansible', 'jenkins', 'ci', 'cd',
    'aws', 'azure', 'gcp', 's3', 'ec2', 'lambda', 'database', 'db', 'mysql',
    'postgresql', 'mongodb', 'redis', 'elasticsearch', 'kafka', 'rabbitmq'
}


def extract_keywords(text: str, min_length: int = 3, remove_stopwords: bool = True) -> List[str]:
    """
    Извлекает ключевые слова из текста.
    
    Args:
        text: Исходный текст
        min_length: Минимальная длина слова
        remove_stopwords: Удалять ли стоп-слова
        
    Returns:
        Список ключевых слов
    """
    # Приводим к нижнему регистру
    text = text.lower()
    
    # Извлекаем слова (кириллица и латиница)
    words = re.findall(r'[а-яёa-z0-9]+', text)
    
    # Фильтруем стоп-слова и короткие слова
    if remove_stopwords:
        keywords = [w for w in words if w not in STOPWORDS and len(w) >= min_length]
    else:
        keywords = [w for w in words if len(w) >= min_length]
    
    return keywords


def extract_technical_terms(text: str) -> List[str]:
    """
    Извлекает технические термины из текста.
    
    Args:
        text: Исходный текст
        
    Returns:
        Список найденных технических терминов
    """
    text_lower = text.lower()
    found_terms = [term for term in TECHNICAL_TERMS if term in text_lower]
    return found_terms


def normalize_query(query: str) -> str:
    """
    Нормализует запрос: удаляет лишние пробелы, приводит к нижнему регистру.
    
    Args:
        query: Исходный запрос
        
    Returns:
        Нормализованный запрос
    """
    return ' '.join(query.lower().split())

