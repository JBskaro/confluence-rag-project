#!/usr/bin/env python3
"""
Test execute_single_query_search() function
"""
import sys
sys.path.insert(0, '/app')

# Import from server
import server

from embeddings import generate_query_embedding
from qdrant_storage import init_qdrant_client

query = 'технологический стек'
print(f'Query: {query}\n')

# Get embedding
embedding = generate_query_embedding(query)
client = init_qdrant_client()

# Call execute_single_query_search
results = server.execute_single_query_search(
    query_embedding=embedding,
    query_text=query,
    search_limit=5,
    where_filter=None,
    qdrant_client=client,
    space='RAUII'
)

print(f'execute_single_query_search() returned {len(results)} results\n')

for i, result in enumerate(results[:3], 1):
    print(f'Result #{i}:')
    print(f'  Keys: {list(result.keys())}')
    print(f'  ID: {result.get("id", "N/A")}')
    
    # Check metadata
    metadata = result.get('metadata', {})
    print(f'  Metadata keys: {list(metadata.keys()) if metadata else "NO METADATA"}')
    
    if metadata:
        print(f'  Title: {metadata.get("title", "N/A")}')
        print(f'  Space: {metadata.get("space", "N/A")}')
        print(f'  Page ID: {metadata.get("page_id", "N/A")}')
    
    # Check text
    text = result.get('text', '')
    print(f'  Text length: {len(text) if text else 0} chars')
    if text:
        print(f'  Text preview: {text[:80]}...')
    else:
        print(f'  ❌ NO TEXT!')
    
    print()

