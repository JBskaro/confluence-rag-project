"""
Тест reranker с реальными данными из логов
Запуск: docker-compose exec confluence-rag python test_reranker_real.py
"""

import numpy as np
from sentence_transformers import CrossEncoder

print("=" * 60)
print("Загрузка модели BAAI/bge-reranker-v2-m3...")
print("=" * 60)
ranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
print("✅ Модель загружена\n")

# Реальные данные из логов
query = "технологический стек проекта RAUII"

# Текст из первого документа (299 chars из логов)
text_short = "направление стек технология интерфейс и аутентификация open webui веб интерфейс для взаимодействие с llm аналог chatgpt для самохостинг аутендификатор выполняться oauth 2 0 authorization code flow через microsoft entra id azure ad по callback сопоставлять telegram id entra user применять роль квота"

# Расширенный текст (добавляем больше контекста)
text_long = text_short + " " + """
Open WebUI - это веб-интерфейс для взаимодействия с языковыми моделями, аналог ChatGPT для самохостинга. 
Аутентификатор выполняет OAuth 2.0 Authorization Code Flow через Microsoft Entra ID (Azure AD). 
По callback сопоставляется Telegram ID с Entra User и применяется роль и квота.

Ollama - локальный сервер для запуска open source LLM моделей на собственном железе.
OpenRouter - онлайн агрегатор для доступа к множеству LLM моделей через единый интерфейс.
LiteLLM Proxy - прокси-сервер, унифицирующий API разных LLM провайдеров под OpenAI-совместимый интерфейс.

Технологический стек проекта включает:
- Open WebUI для веб-интерфейса
- Ollama для локальных моделей
- OpenRouter для облачных моделей
- LiteLLM для унификации API
- Microsoft Entra ID для аутентификации
"""

# Нерелевантный текст
text_bad = "Префикс проекта: NIIRAUII. Данные о Заказчике и проекте."

pairs = [
    [query, text_short],  # Короткий текст (299 chars)
    [query, text_long],   # Длинный текст (расширенный)
    [query, text_bad],    # Нерелевантный
]

print("=" * 60)
print("ТЕСТ: Разные длины текста")
print("=" * 60)
print(f"Query: '{query}'\n")
print(f"Pair 0 (короткий, {len(text_short)} chars): {text_short[:100]}...")
print(f"Pair 1 (длинный, {len(text_long)} chars): {text_long[:100]}...")
print(f"Pair 2 (нерелевантный, {len(text_bad)} chars): {text_bad}\n")

scores = ranker.predict(pairs, apply_softmax=False)

print("SCORES от модели:")
for i, s in enumerate(scores):
    print(f"  Pair {i}: {s:.6f}")

print("\n" + "=" * 60)
print("ВЫВОДЫ:")
print("=" * 60)
print(f"Короткий текст ({len(text_short)} chars): {scores[0]:.6f}")
print(f"Длинный текст ({len(text_long)} chars): {scores[1]:.6f}")
print(f"Нерелевантный: {scores[2]:.6f}")
print(f"\nРазница длинный - короткий: {scores[1] - scores[0]:.6f}")
print(f"Разница короткий - нерелевантный: {scores[0] - scores[2]:.6f}")

if scores[1] > scores[0]:
    print("\n✅ Длинный текст даёт ЛУЧШИЙ score!")
else:
    print("\n⚠️ Длинный текст НЕ даёт лучший score - возможно проблема в содержании")

if scores[0] > scores[2]:
    print("✅ Короткий текст релевантнее нерелевантного")
else:
    print("⚠️ Короткий текст НЕ релевантнее нерелевантного - проблема!")

