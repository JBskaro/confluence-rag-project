#!/usr/bin/env python3
"""
Test search_in_qdrant() function directly
"""
import sys
sys.path.insert(0, '/app')

from embeddings import generate_query_embedding
from qdrant_storage import search_in_qdrant

query = 'технологический стек'
print(f'Query: {query}\n')

# Get embedding
embedding = generate_query_embedding(query)
print(f'Embedding dimension: {len(embedding)}\n')

# Search with RAUII filter
results = search_in_qdrant(
    query_embedding=embedding,
    limit=5,
    space='RAUII'
)

print(f'search_in_qdrant() returned {len(results)} results\n')

for i, result in enumerate(results[:3], 1):
    print(f'Result #{i}:')
    print(f'  Keys: {list(result.keys())}')
    print(f'  ID: {result.get("id", "N/A")}')
    print(f'  Score: {result.get("score", 0):.4f}')
    
    # Check if payload exists
    payload = result.get('payload', {})
    print(f'  Payload keys: {list(payload.keys()) if payload else "NO PAYLOAD"}')
    
    if payload:
        print(f'  Title: {payload.get("title", "N/A")}')
        print(f'  Space: {payload.get("space", "N/A")}')
        print(f'  Page ID: {payload.get("page_id", "N/A")}')
        print(f'  Text in payload: {"_node_content" in payload or "text" in payload}')
    print()

