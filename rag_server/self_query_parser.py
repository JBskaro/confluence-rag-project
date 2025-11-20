#!/usr/bin/env python3
"""
Self-Query Parser для автоматического извлечения фильтров из query.

Примеры:
- "latest docs from DevOps" → {space: "DevOps", date: recent}
- "installation guide by John" → {author: "John", content_contains: "installation"}
"""

import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SelfQueryParser:
    """
    Парсер для извлечения metadata filters из natural language query.
    """

    def __init__(self):
        """Initialize parser with patterns"""

        # Patterns для извлечения фильтров
        self.patterns = {
            'space': [
                r'(?:from|in)\s+(?:space\s+)?["\']?(\w+)["\']?(?:\s+space)?',
                r'(?:space|проект|пространство)[:\s]+["\']?(\w+)["\']?'
            ],
            'author': [
                r'(?:by|from|автор|от)\s+["\']?([A-Za-zА-Яа-я\s\-\']+)["\']?',
                r'(?:written|created)\s+by\s+["\']?([A-Za-zА-Яа-я\s\-\']+)["\']?'
            ],
            'date': [
                r'(?:latest|recent|новые|последние)',
                r'(?:this|past|за последние?)\s+(week|month|year|неделю|месяц|год)',
                r'(?:after|since|с|после)\s+(\d{4}-\d{2}-\d{2})',
                r'(?:before|до)\s+(\d{4}-\d{2}-\d{2})'
            ],
            'type': [
                r'(guide|tutorial|documentation|api|troubleshooting|гайд|документация)',
                r'(?:type|тип)[:\s]+(\w+)'
            ],
            'status': [
                r'(draft|published|archived|черновик|опубликовано)',
                r'(?:status|статус)[:\s]+(\w+)'
            ]
        }

        # Keywords для определения intent
        self.keywords = {
            'installation': ['install', 'setup', 'configure', 'установка', 'настройка'],
            'troubleshooting': ['error', 'problem', 'issue', 'fix', 'ошибка', 'проблема'],
            'api': ['api', 'endpoint', 'request', 'response', 'АПИ'],
            'guide': ['guide', 'tutorial', 'how to', 'гайд', 'руководство']
        }

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse query и извлечь filters.

        Args:
            query: Natural language query

        Returns:
            {
                'clean_query': 'installation guide',  # Query без filter keywords
                'filters': {
                    'space': 'DevOps',
                    'author': 'John Doe',
                    'date_from': '2025-01-01',
                    'content_type': 'guide'
                },
                'intent': 'installation'  # Определенный intent
            }
        """
        result = {
            'clean_query': query,
            'filters': {},
            'intent': None
        }

        if not query or not isinstance(query, str):
            return result

        query_lower = query.lower()
        clean_query = query

        # === ИЗВЛЕЧЕНИЕ ФИЛЬТРОВ ===

        # 1. Space
        for pattern in self.patterns['space']:
            match = re.search(pattern, clean_query, re.IGNORECASE)
            if match:
                space_value = match.group(1).strip()
                if space_value:
                    result['filters']['space'] = space_value
                    # Удаляем из query
                    clean_query = re.sub(pattern, '', clean_query, flags=re.IGNORECASE)
                    break

        # 2. Author
        for pattern in self.patterns['author']:
            match = re.search(pattern, clean_query, re.IGNORECASE)
            if match:
                author_value = match.group(1).strip()
                if author_value and len(author_value) > 1:
                    result['filters']['author'] = author_value
                    clean_query = re.sub(pattern, '', clean_query, flags=re.IGNORECASE)
                    break

        # 3. Date
        date_filter = self._parse_date_filter(clean_query)
        if date_filter:
            result['filters'].update(date_filter)
            # Удаляем date keywords
            for pattern in self.patterns['date']:
                clean_query = re.sub(pattern, '', clean_query, flags=re.IGNORECASE)

        # 4. Type/Content Type
        for pattern in self.patterns['type']:
            match = re.search(pattern, clean_query, re.IGNORECASE)
            if match:
                type_value = match.group(1).strip().lower() if match.lastindex else match.group(0).strip().lower()
                if type_value:
                    result['filters']['content_type'] = type_value
                    break

        # 5. Status
        for pattern in self.patterns['status']:
            match = re.search(pattern, clean_query, re.IGNORECASE)
            if match:
                status_value = match.group(1).strip().lower() if match.lastindex else match.group(0).strip().lower()
                if status_value:
                    result['filters']['status'] = status_value
                    clean_query = re.sub(pattern, '', clean_query, flags=re.IGNORECASE)
                    break

        # === ОПРЕДЕЛЕНИЕ INTENT ===
        for intent, keywords in self.keywords.items():
            if any(kw in query_lower for kw in keywords):
                result['intent'] = intent
                break

        # Cleanup clean_query
        result['clean_query'] = ' '.join(clean_query.split()).strip()

        if result['filters'] or result['intent']:
            logger.debug(f"🔍 Self-query parsed: filters={result['filters']}, intent={result['intent']}, clean_query='{result['clean_query']}'")

        return result

    def _parse_date_filter(self, query: str) -> Optional[Dict[str, str]]:
        """
        Parse date-related filters.

        Returns:
            {'date_from': '2025-01-01', 'date_to': '2025-12-31'} or None
        """
        query_lower = query.lower()

        # Latest/Recent → last 30 days
        if any(kw in query_lower for kw in ['latest', 'recent', 'новые', 'последние']):
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT00:00:00Z')
            return {'date_from': from_date}

        # This/Past week/month/year
        time_periods = {
            'week': 7,
            'неделю': 7,
            'month': 30,
            'месяц': 30,
            'year': 365,
            'год': 365
        }

        for period, days in time_periods.items():
            pattern = r'(?:this|past|за последние?)\s+' + period
            if re.search(pattern, query_lower):
                from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%dT00:00:00Z')
                return {'date_from': from_date}

        # Specific date: "after 2025-01-01"
        match_after = re.search(r'(?:after|since|с|после)\s+(\d{4}-\d{2}-\d{2})', query, re.IGNORECASE)
        if match_after:
            date_str = match_after.group(1)
            # Преобразуем в ISO формат с timezone
            return {'date_from': f"{date_str}T00:00:00Z"}

        # Specific date: "before 2025-12-31"
        match_before = re.search(r'(?:before|до)\s+(\d{4}-\d{2}-\d{2})', query, re.IGNORECASE)
        if match_before:
            date_str = match_before.group(1)
            # Преобразуем в ISO формат с timezone
            return {'date_to': f"{date_str}T23:59:59Z"}

        return None


# === HELPER FUNCTION ===

def parse_self_query(query: str) -> Dict[str, Any]:
    """
    Convenience function для self-query parsing.

    Args:
        query: Natural language query

    Returns:
        Parsed result with clean_query and filters
    """
    parser = SelfQueryParser()
    return parser.parse(query)

