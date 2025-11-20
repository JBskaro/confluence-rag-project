#!/usr/bin/env python3
"""
Check if metadata exists in Qdrant
"""
import sys
sys.path.insert(0, '/app')

from qdrant_storage import init_qdrant_client
import random

client = init_qdrant_client()

# Get one random point to check metadata
dummy_vector = [random.random() for _ in range(4096)]
results = client.search(
    collection_name='confluence',
    query_vector=dummy_vector,
    limit=3,
    with_payload=True,  # ВАЖНО: запрашиваем payload
    with_vectors=False
)

print(f'Found {len(results)} results')
print()

for i, result in enumerate(results, 1):
    print(f'Result #{i}:')
    print(f'  ID: {result.id}')
    print(f'  Score: {result.score:.4f}')
    print(f'  Payload keys: {list(result.payload.keys()) if result.payload else "NO PAYLOAD"}')
    
    if result.payload:
        print(f'  Title: {result.payload.get("title", "N/A")}')
        print(f'  Space: {result.payload.get("space", "N/A")}')
        print(f'  Page ID: {result.payload.get("page_id", "N/A")}')
    print()

