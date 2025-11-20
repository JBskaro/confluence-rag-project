#!/usr/bin/env python3
"""
Менеджер синонимов с самообучением.

Источники синонимов:
1. Базовый словарь (50 общих IT-терминов)
2. Доменные термины (автоматически из Confluence)
3. Выученные синонимы (Query Mining)
4. Ollama (опционально)
"""

import json
import re
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Базовый словарь (50 общих IT-терминов)
# Blacklist: термины, которые НЕ следует заменять (собственные имена, названия инструментов)
TERM_BLACKLIST = {
    'syntaxcheck', 'codesearch', 'docsearch', 'metadatasearch', 'templatesearch',
    'ollama', 'openrouter', 'litellm', 'confluence', 'jira', 'bitbucket',
    'github', 'gitlab', 'docker', 'kubernetes', 'postgres', 'mysql', 'redis',
    'mcp', 'rag', 'llm', 'gpt', 'claude', 'chatgpt',
    'rauii', 'map', 'md', 'mdo', 'mi'  # Названия ваших spaces
}

BASE_SYNONYMS = {
    # === Технологии ===
    'стек': ['технологии', 'инструменты', 'frameworks', 'tech stack', 'tools'],
    'технологий': ['стек', 'инструментов', 'tools', 'tech stack'],
    'framework': ['фреймворк', 'библиотека', 'library', 'фреймворки'],

    # === Разработка ===
    'разработка': ['development', 'dev', 'coding', 'программирование'],
    'баг': ['bug', 'ошибка', 'error', 'дефект', 'issue'],
    'тест': ['test', 'testing', 'проверка', 'тестирование'],

    # === Инфраструктура ===
    'сервер': ['server', 'backend', 'бэкенд', 'хост', 'host'],
    'база данных': ['БД', 'database', 'DB', 'хранилище', 'storage'],
    'бд': ['база данных', 'database', 'DB', 'хранилище'],
    'контейнер': ['container', 'докер'],

    # === API ===
    'api': ['интерфейс', 'endpoint', 'метод', 'веб-сервис', 'rest'],
    'endpoint': ['api', 'метод', 'точка входа', 'route', 'эндпоинт'],
    'rest': ['api', 'restful', 'веб-сервис'],

    # === Confluence ===
    'страница': ['page', 'документ', 'doc', 'страничка'],
    'пространство': ['space', 'спейс', 'область'],
    'документация': ['docs', 'documentation', 'руководство', 'мануал'],

    # === Общие IT-термины ===
    'настройка': ['конфигурация', 'config', 'configuration', 'setup'],
    'установка': ['инсталляция', 'install', 'installation', 'setup'],
    'запуск': ['старт', 'start', 'run', 'launch'],
    'проблема': ['issue', 'баг', 'ошибка', 'problem'],
    'решение': ['solution', 'fix', 'исправление'],
    'инструкция': ['руководство', 'guide', 'мануал', 'howto'],
    'команда': ['team', 'группа', 'отдел'],
    'проект': ['project', 'система', 'приложение', 'сервис'],
    'версия': ['version', 'релиз', 'release'],
    'обновление': ['update', 'апдейт', 'upgrade'],
}


class QueryMiner:
    """
    Анализирует запросы пользователей и автоматически строит граф синонимов.
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.query_log_file = self.data_dir / "query_log.json"
        self.learned_synonyms_file = self.data_dir / "learned_synonyms.json"

        self.query_log = self._load_query_log()
        self.co_occurrence = {}

        # Восстанавливаем граф из логов
        self._rebuild_co_occurrence()

    def _load_query_log(self) -> list:
        """Загружает историю запросов."""
        if self.query_log_file.exists():
            try:
                with open(self.query_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить query_log: {e}")
        return []

    def _rebuild_co_occurrence(self):
        """Восстанавливает граф совместной встречаемости из логов."""
        for entry in self.query_log:
            query = entry.get('query', '')
            result_pages = entry.get('result_pages', [])

            query_terms = self._extract_keywords(query)

            for term in query_terms:
                if term not in self.co_occurrence:
                    self.co_occurrence[term] = {'pages': set(), 'count': 0}

                self.co_occurrence[term]['pages'].update(result_pages)
                self.co_occurrence[term]['count'] += 1

    def _extract_keywords(self, text: str) -> list:
        """Извлекает ключевые слова из текста."""
        # Приводим к нижнему регистру
        text = text.lower()

        # Убираем стоп-слова
        stop_words = {'в', 'на', 'и', 'с', 'по', 'для', 'как', 'что', 'это', 'или', 'а', 'но'}

        # Извлекаем слова (кириллица и латиница)
        words = re.findall(r'[а-яёa-z0-9]+', text)

        # Фильтруем стоп-слова и короткие слова
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def log_query(self, query: str, results: list):
        """
        Логирует запрос и результаты для обучения.

        Args:
            query: Поисковый запрос
            results: Список результатов поиска
        """
        entry = {
            'query': query,
            'timestamp': time.time(),
            'result_pages': [r['metadata'].get('page_id') for r in results if 'metadata' in r]
        }

        self.query_log.append(entry)

        # Обновляем граф
        query_terms = self._extract_keywords(query)
        result_pages = set(entry['result_pages'])

        for term in query_terms:
            if term not in self.co_occurrence:
                self.co_occurrence[term] = {'pages': set(), 'count': 0}

            self.co_occurrence[term]['pages'].update(result_pages)
            self.co_occurrence[term]['count'] += 1

        # Сохраняем каждые 10 запросов
        if len(self.query_log) % 10 == 0:
            self._save_query_log()

        # Экспортируем выученные синонимы каждые 50 запросов
        if len(self.query_log) % 50 == 0:
            self.export_learned_synonyms()

    def _save_query_log(self):
        """Сохраняет историю запросов."""
        try:
            # Конвертируем sets в lists для JSON
            log_to_save = []
            for entry in self.query_log[-1000:]:  # Храним последние 1000 запросов
                log_to_save.append(entry)

            with open(self.query_log_file, 'w', encoding='utf-8') as f:
                json.dump(log_to_save, f, ensure_ascii=False, indent=2)

            logger.debug(f"Query log сохранен: {len(log_to_save)} записей")
        except Exception as e:
            logger.warning(f"Не удалось сохранить query_log: {e}")

    def find_synonyms(self, term: str, threshold: float = 0.5) -> list:
        """
        Находит синонимы на основе совместной встречаемости.

        Логика: Если два термина приводят к одним и тем же страницам,
        они, вероятно, синонимы.

        Args:
            term: Термин для поиска синонимов
            threshold: Порог Jaccard similarity (0.0-1.0)

        Returns:
            Список синонимов
        """
        term = term.lower()

        if term not in self.co_occurrence:
            return []

        term_pages = self.co_occurrence[term]['pages']

        if len(term_pages) < 2:  # Недостаточно данных
            return []

        synonyms = []

        for other_term, data in self.co_occurrence.items():
            if other_term == term:
                continue

            other_pages = data['pages']

            if len(other_pages) < 2:  # Недостаточно данных
                continue

            # Вычисляем Jaccard similarity
            intersection = len(term_pages & other_pages)
            union = len(term_pages | other_pages)

            if union > 0:
                similarity = intersection / union

                if similarity >= threshold:
                    synonyms.append((other_term, similarity))

        # Сортируем по similarity
        synonyms.sort(key=lambda x: x[1], reverse=True)

        return [syn for syn, _ in synonyms[:5]]

    def export_learned_synonyms(self) -> dict:
        """
        Экспортирует выученные синонимы в формат словаря.

        Returns:
            Словарь выученных синонимов
        """
        learned_synonyms = {}

        # Только для терминов с достаточным количеством данных
        for term, data in self.co_occurrence.items():
            if data['count'] >= 3:  # Минимум 3 запроса
                synonyms = self.find_synonyms(term)
                if synonyms:
                    learned_synonyms[term] = synonyms

        # Сохраняем в файл
        try:
            with open(self.learned_synonyms_file, 'w', encoding='utf-8') as f:
                json.dump(learned_synonyms, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Экспортировано {len(learned_synonyms)} выученных синонимов")
        except Exception as e:
            logger.warning(f"Не удалось сохранить learned_synonyms: {e}")

        return learned_synonyms

    def get_learned_synonyms(self) -> dict:
        """Загружает выученные синонимы из файла."""
        if self.learned_synonyms_file.exists():
            try:
                with open(self.learned_synonyms_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить learned_synonyms: {e}")
        return {}


class SynonymsManager:
    """
    Менеджер синонимов с поддержкой множественных источников.
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.domain_terms_file = self.data_dir / "domain_terms.json"

        # Query Miner для обучения
        self.query_miner = QueryMiner(data_dir)

        # Загружаем доменные термины
        self.domain_terms = self._load_domain_terms()

        logger.info(f"✅ SynonymsManager инициализирован")
        logger.info(f"  - Базовый словарь: {len(BASE_SYNONYMS)} терминов")
        logger.info(f"  - Доменные термины: {len(self.domain_terms)} терминов")

    def _load_domain_terms(self) -> dict:
        """Загружает доменные термины из файла."""
        if self.domain_terms_file.exists():
            try:
                with open(self.domain_terms_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить domain_terms: {e}")
        return {}

    def extract_domain_terms_from_confluence(self, collection) -> dict:
        """
        Автоматически извлекает специфичные термины из Confluence.

        Извлекает:
        - Названия spaces
        - Аббревиатуры (ЗАГЛАВНЫЕ СЛОВА)
        - Часто встречающиеся термины

        Args:
            collection: ChromaDB collection

        Returns:
            Словарь доменных терминов
        """
        logger.info("🔍 Анализирую Confluence для извлечения доменных терминов...")

        try:
            # Получаем все документы
            all_docs = collection.get(limit=10000, include=['documents', 'metadatas'])

            if not all_docs or not all_docs.get('metadatas'):
                logger.warning("Нет документов для анализа")
                return {}

            domain_terms = {}

            # 1. Извлекаем названия spaces
            spaces = set()
            for metadata in all_docs['metadatas']:
                if metadata and 'space' in metadata:
                    spaces.add(metadata['space'])

            logger.info(f"  Найдено spaces: {list(spaces)}")

            # 2. Извлекаем аббревиатуры (ЗАГЛАВНЫЕ слова 2-6 букв)
            abbreviations = set()
            for doc in all_docs['documents']:
                if doc:
                    # Кириллица
                    abbrs_ru = re.findall(r'\b[А-ЯЁ]{2,6}\b', doc)
                    abbreviations.update(abbrs_ru)

                    # Латиница
                    abbrs_en = re.findall(r'\b[A-Z]{2,6}\b', doc)
                    abbreviations.update(abbrs_en)

            # Фильтруем частые стоп-слова
            stop_abbrs = {'HTTP', 'HTTPS', 'HTML', 'CSS', 'JSON', 'XML', 'URL', 'API', 'SQL'}
            abbreviations = abbreviations - stop_abbrs

            logger.info(f"  Найдено аббревиатур: {len(abbreviations)} (показываю первые 20)")
            logger.info(f"  {list(abbreviations)[:20]}")

            # 3. Формируем словарь
            for space in spaces:
                domain_terms[space.lower()] = [space, space.upper(), space.lower()]

            for abbr in abbreviations:
                key = abbr.lower()
                if key not in domain_terms:
                    domain_terms[key] = [abbr, abbr.upper(), abbr.lower()]

            # Сохраняем в файл
            with open(self.domain_terms_file, 'w', encoding='utf-8') as f:
                json.dump(domain_terms, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Извлечено {len(domain_terms)} доменных терминов")

            self.domain_terms = domain_terms
            return domain_terms

        except Exception as e:
            logger.error(f"Ошибка при извлечении доменных терминов: {e}")
            return {}

    def get_synonyms(self, word: str, max_synonyms: int = 5) -> list:
        """
        Получает синонимы из всех источников.

        Приоритет:
        1. Базовый словарь
        2. Доменные термины
        3. Выученные синонимы

        Args:
            word: Слово для поиска синонимов
            max_synonyms: Максимальное количество синонимов

        Returns:
            Список синонимов
        """
        word_lower = word.lower()
        synonyms = []

        # 1. Базовый словарь (приоритет)
        if word_lower in BASE_SYNONYMS:
            synonyms.extend(BASE_SYNONYMS[word_lower])

        # 2. Доменные термины
        if word_lower in self.domain_terms:
            synonyms.extend(self.domain_terms[word_lower])

        # 3. Выученные синонимы
        learned = self.query_miner.get_learned_synonyms()
        if word_lower in learned:
            synonyms.extend(learned[word_lower])

        # Дедупликация
        seen = set()
        unique_synonyms = []
        for syn in synonyms:
            syn_lower = syn.lower()
            if syn_lower not in seen and syn_lower != word_lower:
                seen.add(syn_lower)
                unique_synonyms.append(syn)

        return unique_synonyms[:max_synonyms]

    def log_query(self, query: str, results: list):
        """
        Логирует запрос для обучения Query Miner.

        Args:
            query: Поисковый запрос
            results: Список результатов поиска
        """
        self.query_miner.log_query(query, results)


# Глобальный экземпляр
_synonyms_manager = None

def get_synonyms_manager(data_dir: str = "./data") -> SynonymsManager:
    """Получает глобальный экземпляр SynonymsManager."""
    global _synonyms_manager
    if _synonyms_manager is None:
        _synonyms_manager = SynonymsManager(data_dir)
    return _synonyms_manager

