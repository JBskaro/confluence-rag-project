#!/usr/bin/env python3
"""
Check if text exists in Qdrant payloads
"""
import sys
sys.path.insert(0, '/app')

from qdrant_storage import init_qdrant_client, extract_text_from_payload
from embeddings import generate_query_embedding

client = init_qdrant_client()

# Get embedding for search
query = 'технологический стек'
embedding = generate_query_embedding(query)

# Search with payload
results = client.search(
    collection_name='confluence',
    query_vector=embedding,
    limit=3,
    with_payload=True,
    with_vectors=False
)

print(f'Found {len(results)} results\n')

for i, result in enumerate(results, 1):
    print(f'Result #{i}:')
    print(f'  ID: {result.id}')
    print(f'  Score: {result.score:.4f}')
    
    if result.payload:
        # Try to extract text
        extracted_text = extract_text_from_payload(result.payload)
        
        print(f'  Payload keys: {list(result.payload.keys())}')
        print(f'  Title: {result.payload.get("title", "N/A")}')
        print(f'  Space: {result.payload.get("space", "N/A")}')
        print(f'  Page ID: {result.payload.get("page_id", "N/A")}')
        print(f'  Has "text" field: {"text" in result.payload}')
        print(f'  Has "_node_content" field: {"_node_content" in result.payload}')
        
        if extracted_text:
            print(f'  Extracted text length: {len(extracted_text)} chars')
            print(f'  Text preview: {extracted_text[:100]}...')
        else:
            print(f'  ❌ NO TEXT EXTRACTED!')
            # Debug _node_content
            if '_node_content' in result.payload:
                node_content = result.payload['_node_content']
                print(f'  _node_content type: {type(node_content)}')
                print(f'  _node_content preview: {str(node_content)[:200]}...')
    else:
        print('  ❌ NO PAYLOAD!')
    
    print()

