#!/usr/bin/env python3
"""
Тестовый поиск по запросу о технологическом стеке RAUII.
"""
import sys
sys.path.insert(0, '/app')

from qdrant_storage import init_qdrant_client, QDRANT_COLLECTION, extract_text_from_payload
from qdrant_client.models import Filter, FieldCondition, MatchValue
from embeddings import get_embed_model
import os

print("=== Тестовый поиск по RAUII ===\n")

# Инициализация
client = init_qdrant_client()
embed_model = get_embed_model()

# Запрос
query = "Какой технологический стек используется в проекте RAUII?"
print(f"Запрос: {query}")
print(f"Space filter: RAUII")

# Проверяем ENV
print(f"\nENV пороги:")
print(f"  RERANK_THRESHOLD_TECHNICAL: {os.getenv('RERANK_THRESHOLD_TECHNICAL')}")
print(f"  RERANK_THRESHOLD_GENERAL: {os.getenv('RERANK_THRESHOLD_GENERAL')}")
print(f"  RE_RANKER_MODEL: {os.getenv('RE_RANKER_MODEL')}")

# Создание embedding
print("\n1. Создание embedding...")
query_embedding = embed_model.get_text_embedding(query)
print(f"   Embedding size: {len(query_embedding)}D")

# Поиск в Qdrant (без reranking, только vector search)
print("\n2. Vector search в Qdrant...")
results = client.search(
    collection_name=QDRANT_COLLECTION,
    query_vector=query_embedding,
    query_filter=Filter(must=[FieldCondition(key='space', match=MatchValue(value='RAUII'))]),
    limit=10,
    with_payload=True
)

print(f"   Найдено: {len(results)} результатов\n")

# Вывод результатов
for i, result in enumerate(results, 1):
    print(f"--- Результат {i} ---")
    print(f"Vector Score: {result.score:.4f}")
    print(f"Page ID: {result.payload.get('page_id')}")
    print(f"Title: {result.payload.get('title')}")
    
    # Извлечение текста
    text = extract_text_from_payload(result.payload)
    print(f"Text length: {len(text)}")
    if text:
        preview = text[:200].replace('\n', ' ')
        print(f"Text preview: {preview}...")
    
    # Проверка на нужную страницу
    if result.payload.get('page_id') == '18153591':
        print("✅ НАЙДЕНА НУЖНАЯ СТРАНИЦА!")
    print()

print("\n=== Vector search завершен ===")
print("\nТеперь результаты будут переранжированы через CrossEncoder с порогом 0.1")

