#!/usr/bin/env python3
"""
Тест reranking для результатов vector search.
"""
import sys
sys.path.insert(0, '/app')
import os

from qdrant_storage import init_qdrant_client, QDRANT_COLLECTION, extract_text_from_payload
from qdrant_client.models import Filter, FieldCondition, MatchValue
from embeddings import get_embed_model

# Загрузка reranker
from sentence_transformers import CrossEncoder

print("=== Тест Reranking ===\n")

# Инициализация
client = init_qdrant_client()
embed_model = get_embed_model()

# Загрузка reranker
reranker_model = os.getenv('RE_RANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
print(f"1. Загрузка reranker: {reranker_model}")
reranker = CrossEncoder(reranker_model, max_length=512)
print("   ✓ Загружен\n")

# Запрос
query = "Какой технологический стек используется в проекте RAUII?"
print(f"2. Запрос: {query}\n")

# Vector search (топ-20 для reranking)
print("3. Vector search (top-20)...")
query_embedding = embed_model.get_text_embedding(query)
results = client.search(
    collection_name=QDRANT_COLLECTION,
    query_vector=query_embedding,
    query_filter=Filter(must=[FieldCondition(key='space', match=MatchValue(value='RAUII'))]),
    limit=20,
    with_payload=True
)
print(f"   Найдено: {len(results)} результатов\n")

# Подготовка пар для reranking
print("4. Reranking...")
pairs = []
for r in results:
    text = extract_text_from_payload(r.payload)
    pairs.append([query, text])

# Reranking
rerank_scores = reranker.predict(pairs)
print(f"   ✓ Переранжировано {len(rerank_scores)} результатов\n")

# Объединение scores (создаем список кортежей)
results_with_rerank = []
for i, score in enumerate(rerank_scores):
    results_with_rerank.append((results[i], float(score)))

# Сортировка по rerank_score
results_sorted = sorted(results_with_rerank, key=lambda x: x[1], reverse=True)

# Проверка порога
threshold = float(os.getenv('RERANK_THRESHOLD_GENERAL', '0.1'))
print(f"5. Фильтрация по порогу: {threshold}\n")

print("="*80)
print("РЕЗУЛЬТАТЫ (топ-10 после reranking):")
print("="*80)

for i, (r, rerank_score) in enumerate(results_sorted[:10], 1):
    print(f"\n--- Результат {i} ---")
    print(f"Rerank Score: {rerank_score:.4f} {'✓' if rerank_score >= threshold else '❌ FILTERED'}")
    print(f"Vector Score: {r.score:.4f}")
    print(f"Page ID: {r.payload.get('page_id')}")
    print(f"Title: {r.payload.get('title')}")
    
    if r.payload.get('page_id') == '18153591':
        print("🎯 ЭТО НУЖНАЯ СТРАНИЦА!")
    
    text = extract_text_from_payload(r.payload)
    if text:
        preview = text[:150].replace('\n', ' ')
        print(f"Preview: {preview}...")

print(f"\n{'='*80}")
print(f"Результатов с score >= {threshold}: {len([x for x in results_sorted if x[1] >= threshold])}")
print(f"{'='*80}")

