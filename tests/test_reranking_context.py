#!/usr/bin/env python3
"""Тест обогащения контекста для reranking"""
from sentence_transformers import CrossEncoder

ranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
query = "технологический стек проекта RAUII"

# Chunk #8 - самый релевантный, но низкий score
text_original = """Направления > Стек технологий > MCP- Протокол от Anthropic для безопасного подключения внешних инструментов и данных кLLM.

- Syntaxcheck - Анализирует BSL-код на наличие синтаксических ошибок с помощью bsl-language-server."""

# Тест 1: Оригинальный текст (как есть в Qdrant)
score1 = ranker.predict([[query, text_original]])[0]
print("=" * 60)
print("Тест 1: Оригинальный текст (как в Qdrant)")
print("=" * 60)
print(f"Text length: {len(text_original)} chars")
print(f"Text: {text_original[:200]}...")
print(f"Score: {score1:.6f}")
print()

# Тест 2: С обогащением метаданными (page_title, heading, space)
text_enriched = """Space: RAUII
Page: Общая информация о проекте
Heading: Стек технологий

""" + text_original

score2 = ranker.predict([[query, text_enriched]])[0]
print("=" * 60)
print("Тест 2: С обогащением (space, title, heading)")
print("=" * 60)
print(f"Text length: {len(text_enriched)} chars")
print(f"Text: {text_enriched[:200]}...")
print(f"Score: {score2:.6f}")
print(f"Улучшение: {((score2/score1 - 1) * 100):.1f}%")
print()

# Тест 3: Только ключевая часть
text_short = """Стек технологий проекта RAUII:

Направления:
- MCP - Протокол от Anthropic для безопасного подключения внешних инструментов
- Syntaxcheck - Анализирует BSL-код
- Docsearch - Поиск документации
- Ollama - Локальный сервер для LLM
- OpenRouter - Онлайн агрегатор LLM"""

score3 = ranker.predict([[query, text_short]])[0]
print("=" * 60)
print("Тест 3: Сокращённый текст с ключевыми словами")
print("=" * 60)
print(f"Text length: {len(text_short)} chars")
print(f"Text: {text_short}")
print(f"Score: {score3:.6f}")
print(f"Улучшение от original: {((score3/score1 - 1) * 100):.1f}%")
print()

# Тест 4: Прямой ответ на вопрос
text_direct = """Технологический стек проекта RAUII включает:

Работа с ИИ моделями:
- Ollama - Локальный сервер для запуска open-source LLM моделей
- OpenRouter - Онлайн агрегатор для доступа к множеству LLM-моделей
- LiteLLM Proxy - Прокси-сервер, унифицирующий API разных LLM-провайдеров

Интерфейс:
- Open WebUI - Веб-интерфейс для взаимодействия с LLM"""

score4 = ranker.predict([[query, text_direct]])[0]
print("=" * 60)
print("Тест 4: Прямой ответ на вопрос")
print("=" * 60)
print(f"Text length: {len(text_direct)} chars")
print(f"Text: {text_direct}")
print(f"Score: {score4:.6f}")
print(f"Улучшение от original: {((score4/score1 - 1) * 100):.1f}%")
print()

print("=" * 60)
print("ИТОГО:")
print("=" * 60)
print(f"1. Original (1840 chars): {score1:.6f}")
print(f"2. Enriched (metadata):   {score2:.6f}  ({'+' if score2 > score1 else ''}{((score2/score1 - 1) * 100):.1f}%)")
print(f"3. Short (key parts):     {score3:.6f}  ({'+' if score3 > score1 else ''}{((score3/score1 - 1) * 100):.1f}%)")
print(f"4. Direct answer:         {score4:.6f}  ({'+' if score4 > score1 else ''}{((score4/score1 - 1) * 100):.1f}%)")
print()
print("ВЫВОД:")
if score4 > 0.1:
    print("  ✅ Reranker МОЖЕТ давать высокие scores с правильным текстом!")
    print("  ❌ Проблема: текст в Qdrant НЕ ОПТИМАЛЕН для reranking")
else:
    print("  ❌ Reranker НЕ МОЖЕТ правильно оценить релевантность для этого query")
    print("  🔧 Возможно, проблема с embedding model или самим reranker")

