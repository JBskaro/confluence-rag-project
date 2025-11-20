#!/usr/bin/env python3
"""
Тестовый поиск по запросу о номенклатуре на складе.
"""
import sys
sys.path.insert(0, '/app')

from qdrant_storage import init_qdrant_client, QDRANT_COLLECTION, extract_text_from_payload
from qdrant_client.models import Filter, FieldCondition, MatchValue
from embeddings import get_embed_model

print("=== Тестовый поиск по Surveys ===\n")

# Инициализация
client = init_qdrant_client()
embed_model = get_embed_model()

# Запрос
query = "уточняющие вопросы для обследования по учету номенклатуры на складе"
print(f"Запрос: {query}")

# Создание embedding
print("\n1. Создание embedding...")
query_embedding = embed_model.get_text_embedding(query)
print(f"   Embedding size: {len(query_embedding)}D")

# Поиск в Qdrant
print("\n2. Поиск в Qdrant...")
results = client.search(
    collection_name=QDRANT_COLLECTION,
    query_vector=query_embedding,
    query_filter=Filter(must=[FieldCondition(key='space', match=MatchValue(value='Surveys'))]),
    limit=5,
    with_payload=True
)

print(f"   Найдено: {len(results)} результатов\n")

# Вывод результатов
for i, result in enumerate(results, 1):
    print(f"--- Результат {i} ---")
    print(f"Score: {result.score:.4f}")
    print(f"Page ID: {result.payload.get('page_id')}")
    print(f"Title: {result.payload.get('title')}")
    
    # Извлечение текста
    text = extract_text_from_payload(result.payload)
    print(f"Text length: {len(text)}")
    if text:
        preview = text[:300].replace('\n', ' ')
        print(f"Text preview: {preview}...")
    else:
        print("❌ Текст не извлечен!")
    print()

print("\n=== Поиск завершен ===")

