#!/usr/bin/env python3
"""Debug RRF scores для страницы 18153591."""
import sys
sys.path.insert(0, '/app')

from hybrid_search import hybrid_search, init_bm25_retriever
from qdrant_storage import init_qdrant_client, search_in_qdrant
from embeddings import generate_query_embedding

query = "технологический стек проекта RAUII"
space_filter = "RAUII"
TARGET_PAGE_ID = "18153591"

print(f"🔍 Query: '{query}'")
print(f"🎯 Target: {TARGET_PAGE_ID}\n")

# === 1. Vector Search ===
emb = generate_query_embedding(query)
vector_results_raw = search_in_qdrant(emb, limit=50, space=space_filter)

# Конвертируем формат
vector_results = []
for r in vector_results_raw:
    vector_results.append({
        'id': r['id'],
        'score': r['score'],
        'metadata': r.get('payload', {}),
        'text': r.get('payload', {}).get('text', '')
    })

print(f"Vector results: {len(vector_results)}")
# Проверяем есть ли target в vector
target_in_vector = None
for i, r in enumerate(vector_results, 1):
    if r['metadata'].get('page_id') == TARGET_PAGE_ID:
        target_in_vector = i
        print(f"  ⭐ Target в Vector на позиции #{i}")
        break
if not target_in_vector:
    print(f"  ❌ Target НЕ в топ-{len(vector_results)} vector")

# === 2. BM25 Search ===
from utils.lemmatizer import lemmatize_text
qdrant_client = init_qdrant_client()
bm25 = init_bm25_retriever(qdrant_client)
query_lemmatized = lemmatize_text(query)
bm25_nodes = bm25.retrieve(query_lemmatized)

bm25_results = []
target_in_bm25 = None
for i, node in enumerate(bm25_nodes, 1):
    metadata = node.metadata if hasattr(node, 'metadata') else {}
    if metadata.get('space') == space_filter:
        bm25_results.append({
            'id': node.node_id if hasattr(node, 'node_id') else node.id_,
            'score': node.score if hasattr(node, 'score') else 0.0,
            'metadata': metadata,
            'text': node.text if hasattr(node, 'text') else ''
        })
        
        if metadata.get('page_id') == TARGET_PAGE_ID:
            if not target_in_bm25:
                target_in_bm25 = len(bm25_results)

print(f"\nBM25 results (после space фильтра): {len(bm25_results)}")
if target_in_bm25:
    print(f"  ⭐ Target в BM25 на позиции #{target_in_bm25}")
else:
    print(f"  ❌ Target НЕ в BM25")

# === 3. RRF Simulation ===
print(f"\n{'='*80}")
print("📊 RRF SCORE CALCULATION")
print(f"{'='*80}\n")

k = 60
vector_weight = 0.60
bm25_weight = 0.40

print(f"Parameters: k={k}, vector_weight={vector_weight}, bm25_weight={bm25_weight}\n")

# Считаем RRF score для target
if target_in_vector:
    vector_rrf = vector_weight * (1.0 / (k + target_in_vector))
    print(f"Target Vector RRF: {vector_weight} * (1 / ({k} + {target_in_vector})) = {vector_rrf:.6f}")
else:
    vector_rrf = 0.0
    print(f"Target Vector RRF: 0 (not in vector results)")

if target_in_bm25:
    bm25_rrf = bm25_weight * (1.0 / (k + target_in_bm25))
    print(f"Target BM25 RRF:   {bm25_weight} * (1 / ({k} + {target_in_bm25})) = {bm25_rrf:.6f}")
else:
    bm25_rrf = 0.0
    print(f"Target BM25 RRF:   0 (not in BM25 results)")

target_total_rrf = vector_rrf + bm25_rrf
print(f"\n🎯 Target TOTAL RRF: {target_total_rrf:.6f}")

# Считаем RRF для топ-10 vector
print(f"\n📈 Топ-10 Vector results RRF scores:")
for i in range(1, min(11, len(vector_results) + 1)):
    rrf = vector_weight * (1.0 / (k + i))
    page_id = vector_results[i-1]['metadata'].get('page_id', 'N/A')
    marker = " ⭐" if page_id == TARGET_PAGE_ID else ""
    print(f"  Vector #{i:2d} (page_id={page_id}): RRF = {rrf:.6f}{marker}")

print(f"\n🔍 ВЫВОД:")
if target_total_rrf > 0:
    # Оцениваем позицию
    better_count = 0
    for i in range(1, len(vector_results) + 1):
        rrf = vector_weight * (1.0 / (k + i))
        if rrf > target_total_rrf:
            better_count += 1
    
    estimated_position = better_count + 1
    print(f"  Target RRF score: {target_total_rrf:.6f}")
    print(f"  Ориентировочная позиция в RRF: #{estimated_position}")
    print(f"  Rerank limit: 9")
    
    if estimated_position <= 9:
        print(f"  ✅ Target ДОЛЖЕН попасть в reranking!")
    else:
        print(f"  ❌ Target ОТСЕЧЁН ДО reranking (позиция #{estimated_position} > limit 9)")
else:
    print(f"  ❌ Target не найден ни в Vector, ни в BM25!")

