#!/usr/bin/env python3
"""Тест Hybrid Search с новыми весами."""
import sys
sys.path.insert(0, '/app')

from embeddings import generate_query_embedding
from qdrant_storage import search_in_qdrant, init_qdrant_client
from hybrid_search import hybrid_search

query = 'технологический стек проекта RAUII'
space = 'RAUII'
TARGET = '18153591'

print("="*80)
print("ТЕСТ HYBRID SEARCH С НОВЫМИ ВЕСАМИ (0.4/0.6)")
print("="*80)

# Vector
print("\n1. Vector Search...")
emb = generate_query_embedding(query)
vector_raw = search_in_qdrant(emb, limit=50, space=space)
vector = []
for r in vector_raw:
    vector.append({
        'id': r['id'],
        'score': r['score'],
        'metadata': r.get('payload', {}),
        'text': r.get('payload', {}).get('text', '')
    })

print(f"   Найдено: {len(vector)}")
print(f"   Keys: {list(vector[0].keys())}")
print(f"   Sample score: {vector[0]['score']:.6f}")

# Hybrid
print("\n2. Hybrid Search...")
qdrant = init_qdrant_client()
hybrid = hybrid_search(
    query=query, 
    qdrant_client=qdrant, 
    vector_results=vector, 
    space_filter=space, 
    limit=100
)

print(f"   Результатов: {len(hybrid)}")
if hybrid:
    print(f"   Keys: {list(hybrid[0].keys())}")
    print(f"   Sample score: {hybrid[0].get('score', 0)}")
    print(f"   Sample rrf_score: {hybrid[0].get('rrf_score', 0)}")

# Поиск target
print(f"\n3. Поиск target page_id={TARGET}...")
target_found = False
for i, r in enumerate(hybrid, 1):
    meta = r.get('metadata', {})
    if meta.get('page_id') == TARGET:
        score = r.get('score', 0)
        rrf = r.get('rrf_score', 0)
        print(f"   ⭐ НАЙДЕН на позиции #{i}")
        print(f"      score: {score}")
        print(f"      rrf_score: {rrf}")
        target_found = True
        break

if not target_found:
    print(f"   ❌ Target НЕ найден в {len(hybrid)} результатах")

# Топ-10
print(f"\n4. Топ-10 Hybrid:")
for i, r in enumerate(hybrid[:10], 1):
    meta = r.get('metadata', {})
    page_id = meta.get('page_id', 'N/A')
    score = r.get('score', 0)
    marker = " ⭐" if page_id == TARGET else ""
    print(f"   #{i:2d}: page_id={page_id}, score={score:.6f}{marker}")

print("\n" + "="*80)

