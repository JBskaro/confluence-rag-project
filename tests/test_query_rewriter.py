#!/usr/bin/env python3
"""
Unit tests для Query Rewriter модуля.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys
import time

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag_server'))

# Мокируем openai перед импортом query_rewriter
import sys
from unittest.mock import MagicMock
openai_mock = MagicMock()
sys.modules['openai'] = openai_mock
sys.modules['openai'].OpenAI = MagicMock()

from query_rewriter import (
    rewrite_query_with_ollama,
    rewrite_query_with_openrouter,
    rewrite_query_adaptive,
    cached_rewrite_query,
    get_rewriter_stats,
    clear_rewriter_cache
)


class TestOllamaRewriting:
    """Тесты для Ollama rewriting."""
    
    def test_ollama_disabled(self, monkeypatch):
        """Тест когда Ollama отключен."""
        monkeypatch.setenv('USE_OLLAMA_FOR_QUERY_EXPANSION', 'false')
        
        result = rewrite_query_with_ollama("тестовый запрос")
        assert result is None
    
    @patch('query_rewriter.requests.post')
    def test_ollama_success(self, mock_post, monkeypatch):
        """Тест успешного ответа от Ollama."""
        monkeypatch.setenv('USE_OLLAMA_FOR_QUERY_EXPANSION', 'true')
        monkeypatch.setenv('OLLAMA_URL', 'http://localhost:11434')
        monkeypatch.setenv('OLLAMA_MODEL', 'llama3.2')
        
        # Мокируем успешный ответ
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'альтернативный вариант 1\nальтернативный вариант 2'
        }
        mock_post.return_value = mock_response
        
        result = rewrite_query_with_ollama("тестовый запрос")
        
        assert result is not None
        assert len(result) >= 1
        assert result[0] == "тестовый запрос"  # Первый - оригинал
        assert len(result) <= 3  # Макс 3 варианта
    
    @patch('query_rewriter.requests.post')
    def test_ollama_timeout(self, mock_post, monkeypatch):
        """Тест timeout от Ollama."""
        monkeypatch.setenv('USE_OLLAMA_FOR_QUERY_EXPANSION', 'true')
        
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        result = rewrite_query_with_ollama("тестовый запрос")
        assert result is None
    
    @patch('query_rewriter.requests.post')
    def test_ollama_connection_error(self, mock_post, monkeypatch):
        """Тест ошибки подключения к Ollama."""
        monkeypatch.setenv('USE_OLLAMA_FOR_QUERY_EXPANSION', 'true')
        
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        result = rewrite_query_with_ollama("тестовый запрос")
        assert result is None
    
    @patch('query_rewriter.requests.post')
    def test_ollama_bad_status(self, mock_post, monkeypatch):
        """Тест плохого статуса от Ollama."""
        monkeypatch.setenv('USE_OLLAMA_FOR_QUERY_EXPANSION', 'true')
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        result = rewrite_query_with_ollama("тестовый запрос")
        assert result is None


class TestOpenRouterRewriting:
    """Тесты для OpenRouter rewriting."""
    
    def test_openrouter_disabled(self, monkeypatch):
        """Тест когда OpenRouter отключен."""
        monkeypatch.setenv('USE_OPENROUTER_FOR_REWRITING', 'false')
        
        result = rewrite_query_with_openrouter("тестовый запрос")
        assert result is None
    
    def test_openrouter_not_configured(self, monkeypatch):
        """Тест когда OpenRouter не настроен."""
        monkeypatch.setenv('USE_OPENROUTER_FOR_REWRITING', 'true')
        monkeypatch.delenv('OPENAI_API_BASE', raising=False)
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        monkeypatch.delenv('OPENAI_MODEL', raising=False)
        monkeypatch.delenv('OPENAI_REWRITING_MODEL', raising=False)
        
        result = rewrite_query_with_openrouter("тестовый запрос")
        assert result is None
    
    @patch('query_rewriter.OpenAI')
    def test_openrouter_uses_rewriting_model_priority(self, mock_openai_class, monkeypatch):
        """Тест приоритета OPENAI_REWRITING_MODEL над OPENAI_MODEL."""
        monkeypatch.setenv('USE_OPENROUTER_FOR_REWRITING', 'true')
        monkeypatch.setenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1')
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
        monkeypatch.setenv('OPENAI_MODEL', 'qwen/qwen3-embedding-8b')  # Embedding модель
        monkeypatch.setenv('OPENAI_REWRITING_MODEL', 'openai/gpt-3.5-turbo')  # Генеративная модель
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'вариант 1'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = rewrite_query_with_openrouter("тестовый запрос")
        
        assert result is not None
        # Проверяем что использовалась OPENAI_REWRITING_MODEL, а не OPENAI_MODEL
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'openai/gpt-3.5-turbo'
    
    @patch('query_rewriter.OpenAI')
    def test_openrouter_fallback_to_openai_model(self, mock_openai_class, monkeypatch):
        """Тест fallback на OPENAI_MODEL если OPENAI_REWRITING_MODEL не указана."""
        monkeypatch.setenv('USE_OPENROUTER_FOR_REWRITING', 'true')
        monkeypatch.setenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1')
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
        monkeypatch.setenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        monkeypatch.delenv('OPENAI_REWRITING_MODEL', raising=False)
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'вариант 1'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = rewrite_query_with_openrouter("тестовый запрос")
        
        assert result is not None
        # Проверяем что использовался OPENAI_MODEL как fallback
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-3.5-turbo'
    
    @patch('query_rewriter.OpenAI')
    def test_openrouter_success(self, mock_openai_class, monkeypatch):
        """Тест успешного ответа от OpenRouter."""
        monkeypatch.setenv('USE_OPENROUTER_FOR_REWRITING', 'true')
        monkeypatch.setenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1')
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
        monkeypatch.setenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
        # Мокируем OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'вариант 1\nвариант 2'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = rewrite_query_with_openrouter("тестовый запрос")
        
        assert result is not None
        assert len(result) >= 1
        assert result[0] == "тестовый запрос"  # Первый - оригинал
        assert len(result) <= 3  # Макс 3 варианта
    
    @patch('query_rewriter.OpenAI')
    def test_openrouter_with_examples(self, mock_openai_class, monkeypatch):
        """Тест OpenRouter с примерами из Semantic Query Log."""
        monkeypatch.setenv('USE_OPENROUTER_FOR_REWRITING', 'true')
        monkeypatch.setenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1')
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
        monkeypatch.setenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'вариант 1'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        examples = ["пример 1", "пример 2"]
        result = rewrite_query_with_openrouter("тестовый запрос", examples=examples)
        
        assert result is not None
        # Проверяем что примеры были переданы в промпт
        call_args = mock_client.chat.completions.create.call_args
        assert call_args is not None
        messages = call_args[1]['messages']
        assert 'пример' in messages[0]['content'].lower()


class TestAdaptiveRewriting:
    """Тесты для адаптивного rewriting с fallback."""
    
    @patch('query_rewriter.rewrite_query_with_ollama')
    def test_ollama_priority(self, mock_ollama, monkeypatch):
        """Тест приоритета Ollama."""
        mock_ollama.return_value = ["оригинал", "вариант 1", "вариант 2"]
        
        result = rewrite_query_adaptive("тестовый запрос")
        
        assert result == ["оригинал", "вариант 1", "вариант 2"]
        mock_ollama.assert_called_once()
    
    @patch('query_rewriter.rewrite_query_with_ollama')
    @patch('query_rewriter.rewrite_query_with_openrouter')
    def test_openrouter_fallback(self, mock_openrouter, mock_ollama, monkeypatch):
        """Тест fallback на OpenRouter когда Ollama недоступен."""
        mock_ollama.return_value = None
        mock_openrouter.return_value = ["оригинал", "вариант 1"]
        
        result = rewrite_query_adaptive("тестовый запрос")
        
        assert result == ["оригинал", "вариант 1"]
        mock_ollama.assert_called_once()
        mock_openrouter.assert_called_once()
    
    @patch('query_rewriter.rewrite_query_with_ollama')
    @patch('query_rewriter.rewrite_query_with_openrouter')
    def test_graceful_degradation(self, mock_openrouter, mock_ollama, monkeypatch):
        """Тест graceful degradation когда оба провайдера недоступны."""
        mock_ollama.return_value = None
        mock_openrouter.return_value = None
        
        result = rewrite_query_adaptive("тестовый запрос")
        
        assert result == ["тестовый запрос"]  # Только оригинал
    
    @patch('query_rewriter.rewrite_query_with_ollama')
    def test_semantic_log_examples(self, mock_ollama, monkeypatch):
        """Тест передачи примеров из Semantic Query Log."""
        mock_ollama.return_value = None
        
        mock_semantic_log = Mock()
        mock_semantic_log.get_successful_queries.return_value = ["пример 1", "пример 2"]
        
        with patch('query_rewriter.rewrite_query_with_openrouter') as mock_openrouter:
            mock_openrouter.return_value = ["оригинал", "вариант"]
            
            result = rewrite_query_adaptive("тестовый запрос", semantic_log=mock_semantic_log)
            
            # Проверяем что примеры были переданы в OpenRouter
            mock_openrouter.assert_called_once()
            call_args = mock_openrouter.call_args
            assert call_args[1]['examples'] == ["пример 1", "пример 2"]


class TestCaching:
    """Тесты для кэширования."""
    
    def test_cache_hit(self, monkeypatch):
        """Тест попадания в кэш."""
        monkeypatch.setenv('REWRITE_CACHE_TTL', '3600')
        clear_rewriter_cache()
        
        with patch('query_rewriter.rewrite_query_adaptive') as mock_adaptive:
            mock_adaptive.return_value = ["оригинал", "вариант 1"]
            
            # Первый вызов - должен вызвать rewrite_query_adaptive
            result1 = cached_rewrite_query("тестовый запрос")
            assert result1 == ["оригинал", "вариант 1"]
            assert mock_adaptive.call_count == 1
            
            # Второй вызов - должен использовать кэш
            result2 = cached_rewrite_query("тестовый запрос")
            assert result2 == ["оригинал", "вариант 1"]
            assert mock_adaptive.call_count == 1  # Не должен вызваться снова
    
    def test_cache_expiry(self, monkeypatch):
        """Тест истечения кэша."""
        monkeypatch.setenv('REWRITE_CACHE_TTL', '1')  # 1 секунда
        clear_rewriter_cache()
        
        with patch('query_rewriter.rewrite_query_adaptive') as mock_adaptive:
            mock_adaptive.return_value = ["оригинал", "вариант 1"]
            
            # Первый вызов
            result1 = cached_rewrite_query("тестовый запрос")
            assert mock_adaptive.call_count == 1
            
            # Ждем истечения TTL
            time.sleep(1.1)
            
            # Второй вызов после истечения TTL
            result2 = cached_rewrite_query("тестовый запрос")
            assert mock_adaptive.call_count == 2  # Должен вызваться снова
    
    def test_cache_different_queries(self, monkeypatch):
        """Тест что разные запросы кэшируются отдельно."""
        clear_rewriter_cache()
        
        with patch('query_rewriter.rewrite_query_adaptive') as mock_adaptive:
            mock_adaptive.side_effect = [
                ["оригинал 1", "вариант 1"],
                ["оригинал 2", "вариант 2"]
            ]
            
            result1 = cached_rewrite_query("запрос 1")
            result2 = cached_rewrite_query("запрос 2")
            
            assert result1 == ["оригинал 1", "вариант 1"]
            assert result2 == ["оригинал 2", "вариант 2"]
            assert mock_adaptive.call_count == 2


class TestStatistics:
    """Тесты для статистики."""
    
    def test_get_stats_structure(self):
        """Тест структуры статистики."""
        stats = get_rewriter_stats()
        
        # Проверяем что все ключи присутствуют
        assert 'total_requests' in stats
        assert 'cache_hits' in stats
        assert 'cache_size' in stats
        assert 'ollama_success' in stats
        assert 'ollama_failed' in stats
        assert 'openrouter_success' in stats
        assert 'openrouter_failed' in stats
        assert 'no_rewriting' in stats
        assert 'cache_hit_rate' in stats
        
        # Проверяем типы
        assert isinstance(stats['total_requests'], int)
        assert isinstance(stats['cache_hits'], int)
        assert isinstance(stats['cache_size'], int)
    
    @patch('query_rewriter.rewrite_query_with_ollama')
    def test_stats_after_requests(self, mock_ollama):
        """Тест статистики после запросов."""
        clear_rewriter_cache()
        
        mock_ollama.return_value = ["оригинал", "вариант"]
        
        # Получаем начальную статистику
        initial_stats = get_rewriter_stats()
        initial_total = initial_stats['total_requests']
        
        # Делаем несколько запросов
        result1 = rewrite_query_adaptive("тестовый запрос для статистики 1")
        result2 = rewrite_query_adaptive("тестовый запрос для статистики 2")
        
        # Проверяем что функция работает
        assert result1 is not None
        assert result2 is not None
        assert len(result1) >= 1
        assert len(result2) >= 1
        
        # Проверяем что функция была вызвана
        assert mock_ollama.call_count == 2
        
        # Проверяем что статистика увеличилась
        stats = get_rewriter_stats()
        assert stats['total_requests'] > initial_total
    
    def test_clear_cache(self):
        """Тест очистки кэша."""
        with patch('query_rewriter.rewrite_query_adaptive') as mock_adaptive:
            mock_adaptive.return_value = ["оригинал", "вариант"]
            
            # Заполняем кэш
            cached_rewrite_query("запрос 1")
            cached_rewrite_query("запрос 2")
            
            stats_before = get_rewriter_stats()
            assert stats_before['cache_size'] == 2
            
            # Очищаем кэш
            clear_rewriter_cache()
            
            stats_after = get_rewriter_stats()
            assert stats_after['cache_size'] == 0


class TestIntegration:
    """Интеграционные тесты."""
    
    def test_full_flow_with_mocks(self, monkeypatch):
        """Тест полного потока с моками."""
        clear_rewriter_cache()
        monkeypatch.setenv('USE_OLLAMA_FOR_QUERY_EXPANSION', 'true')
        monkeypatch.setenv('REWRITE_CACHE_TTL', '3600')
        
        # Получаем начальную статистику
        initial_stats = get_rewriter_stats()
        initial_cache_hits = initial_stats['cache_hits']
        initial_total = initial_stats['total_requests']
        
        with patch('query_rewriter.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'response': 'вариант 1\nвариант 2'
            }
            mock_post.return_value = mock_response
            
            # Первый запрос - должен вызвать Ollama
            result1 = cached_rewrite_query("тестовый запрос")
            assert len(result1) >= 1
            assert mock_post.call_count == 1
            
            # Второй запрос - должен использовать кэш
            result2 = cached_rewrite_query("тестовый запрос")
            assert result2 == result1
            assert mock_post.call_count == 1  # Не должен вызваться снова
            
            # Проверяем статистику (учитываем начальные значения)
            stats = get_rewriter_stats()
            # Проверяем что кэш действительно использован
            assert stats['cache_hits'] > initial_cache_hits
            # Проверяем что общее количество запросов увеличилось
            assert stats['total_requests'] > initial_total
            # Проверяем что кэш содержит запись
            assert stats['cache_size'] > 0

