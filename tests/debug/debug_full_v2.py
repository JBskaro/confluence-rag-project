#!/usr/bin/env python3
"""Debug полного pipeline v2 с корректным импортом."""
import sys
sys.path.insert(0, '/app')

TARGET_PAGE_ID = "18153591"

def check_target(results, stage_name):
    """Проверить есть ли target в результатах."""
    found = []
    for i, r in enumerate(results, 1):
        metadata = r.get('metadata', {}) or r.get('payload', {})
        page_id = metadata.get('page_id')
        if page_id == TARGET_PAGE_ID:
            score = r.get('score', 0.0)
            found.append((i, score))
    
    if found:
        print(f"  ✅ {stage_name}: Target на позициях {found}")
        return True, found[0][0]
    else:
        print(f"  ❌ {stage_name}: Target ПОТЕРЯН!")
        return False, None

print("="*80)
print("🔍 FULL PIPELINE DEBUG")
print("="*80)

query = "технологический стек проекта RAUII"
space = "RAUII"

print(f"\nQuery: '{query}'")
print(f"Target: {TARGET_PAGE_ID}\n")

# === 1. Vector Search ===
print("ЭТАП 1: Vector Search (50 результатов)")
from embeddings import generate_query_embedding
from qdrant_storage import search_in_qdrant

emb = generate_query_embedding(query)
vector_raw = search_in_qdrant(emb, limit=50, space=space)
print(f"  Найдено: {len(vector_raw)}")
found_vector, pos_vector = check_target(vector_raw, "Vector")

# === 2. Hybrid Search ===
print("\nЭТАП 2: Hybrid Search (RRF)")
from hybrid_search import hybrid_search
from qdrant_storage import init_qdrant_client

vector_formatted = []
for r in vector_raw:
    vector_formatted.append({
        'id': r['id'],
        'score': r['score'],
        'metadata': r.get('payload', {}),
        'text': r.get('payload', {}).get('text', '')
    })

qdrant_client = init_qdrant_client()
hybrid = hybrid_search(
    query=query,
    qdrant_client=qdrant_client,
    vector_results=vector_formatted,
    space_filter=space,
    limit=100
)
print(f"  Найдено: {len(hybrid)}")
found_hybrid, pos_hybrid = check_target(hybrid, "Hybrid")

if found_hybrid and pos_hybrid:
    print(f"\n  📊 Топ-10 после Hybrid Search:")
    for i, r in enumerate(hybrid[:10], 1):
        meta = r.get('metadata', {})
        page_id = meta.get('page_id', 'N/A')
        score = r.get('score', 0.0)
        rrf = r.get('rrf_score', 0.0)
        marker = " ⭐" if page_id == TARGET_PAGE_ID else ""
        print(f"    #{i}: page_id={page_id}, score={score:.6f}, rrf={rrf:.6f}{marker}")

# === 3. Дедупликация ===
print("\nЭТАП 3: Дедупликация")
from mcp_rag_secure import deduplicate_results

dedup = deduplicate_results(hybrid)
print(f"  После дедупликации: {len(dedup)}")
found_dedup, pos_dedup = check_target(dedup, "Dedup")

if found_dedup and pos_dedup:
    print(f"\n  📊 Топ-10 после дедупликации:")
    for i, r in enumerate(dedup[:10], 1):
        meta = r.get('metadata', {})
        page_id = meta.get('page_id', 'N/A')
        score = r.get('score', 0.0)
        marker = " ⭐" if page_id == TARGET_PAGE_ID else ""
        print(f"    #{i}: page_id={page_id}, score={score:.6f}{marker}")

# === 4. Adaptive Rerank Limit ===
print("\nЭТАП 4: Adaptive Rerank Limit")
from mcp_rag_secure import get_adaptive_rerank_limit

# Копирую логику из mcp_rag_secure.py
def calc_rerank_limit(query: str, candidate_count: int, has_space_filter: bool) -> int:
    query_words = len(query.split())
    if query_words <= 2:
        base_limit = 3
    elif query_words <= 4:
        base_limit = min(9, candidate_count)
    elif query_words <= 6:
        base_limit = min(15, candidate_count)
    else:
        base_limit = min(20, candidate_count)
    
    if has_space_filter and candidate_count > 5:
        base_limit = max(base_limit, min(12, candidate_count))
    
    return min(base_limit, candidate_count)

query_words = len(query.split())
rerank_limit = calc_rerank_limit(query, len(dedup), True)
print(f"  Query words: {query_words}")
print(f"  Candidates: {len(dedup)}")
print(f"  Rerank limit: {rerank_limit}")

rerank_input = dedup[:rerank_limit]
print(f"  Ограничение: {len(dedup)} → {len(rerank_input)}")
found_rerank, pos_rerank = check_target(rerank_input, "Rerank Input")

# === ВЫВОД ===
print("\n" + "="*80)
print("📊 ИТОГ")
print("="*80)

if found_vector:
    print(f"✅ Vector Search:     позиция #{pos_vector}")
else:
    print(f"❌ Vector Search:     НЕ НАЙДЕН")

if found_hybrid:
    print(f"✅ Hybrid Search:     позиция #{pos_hybrid}")
else:
    print(f"❌ Hybrid Search:     НЕ НАЙДЕН")

if found_dedup:
    print(f"✅ Дедупликация:      позиция #{pos_dedup}")
else:
    print(f"❌ Дедупликация:      НЕ НАЙДЕН")

if found_rerank:
    print(f"✅ Rerank Input:      позиция #{pos_rerank}")
else:
    print(f"❌ Rerank Input:      ОТСЕЧЁН (был на #{pos_dedup if pos_dedup else '?'}, limit={rerank_limit})")

print("\n" + "="*80)
if not found_rerank and pos_dedup:
    print(f"🔴 ПРОБЛЕМА: Target отсечён на этапе Adaptive Rerank Limit!")
    print(f"   Позиция после дедупликации: #{pos_dedup}")
    print(f"   Rerank limit: {rerank_limit}")
    print(f"   РЕШЕНИЕ: Нужно увеличить rerank_limit или улучшить RRF веса")

