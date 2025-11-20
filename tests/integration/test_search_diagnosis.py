#!/usr/bin/env python3
"""
Test search diagnosis - Step 3
"""
import sys
sys.path.insert(0, '/app')

from embeddings import generate_query_embedding
from qdrant_storage import search_in_qdrant

query = 'технологический стек'
print(f'Query: {query}')

# Get embedding
embedding = generate_query_embedding(query)
print(f'Embedding dimension: {len(embedding) if embedding else None}')

if not embedding:
    print('ERROR: Embedding is None!')
    sys.exit(1)

# Search without filter
results_no_filter = search_in_qdrant(
    query_embedding=embedding,
    limit=5,
    space=None
)
print(f'\nResults without filter: {len(results_no_filter)}')
for i, r in enumerate(results_no_filter[:3], 1):
    title = r.get('metadata', {}).get('title', 'N/A')
    space = r.get('metadata', {}).get('space', 'N/A')
    score = r.get('score', 0)
    print(f"  {i}. {title}")
    print(f"     space={space}, score={score:.4f}")

# Search with RAUII filter
results_with_filter = search_in_qdrant(
    query_embedding=embedding,
    limit=5,
    space='RAUII'
)
print(f'\nResults with space=RAUII: {len(results_with_filter)}')
for i, r in enumerate(results_with_filter[:3], 1):
    title = r.get('metadata', {}).get('title', 'N/A')
    page_id = r.get('metadata', {}).get('page_id', 'N/A')
    score = r.get('score', 0)
    print(f"  {i}. {title}")
    print(f"     page_id={page_id}, score={score:.4f}")

