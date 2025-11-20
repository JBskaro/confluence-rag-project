#!/usr/bin/env python3
"""Debug полного MCP pipeline для страницы 18153591."""
import sys
sys.path.insert(0, '/app')

TARGET_PAGE_ID = "18153591"

def check_target_in_results(results, stage_name):
    """Проверить есть ли target в результатах."""
    found_positions = []
    for i, r in enumerate(results, 1):
        metadata = r.get('metadata', {}) or r.get('payload', {})
        page_id = metadata.get('page_id')
        if page_id == TARGET_PAGE_ID:
            score = r.get('score', 0.0)
            found_positions.append((i, score))
    
    if found_positions:
        print(f"  ✅ {stage_name}: Target найден на позициях {found_positions}")
        return True
    else:
        print(f"  ❌ {stage_name}: Target ПОТЕРЯН!")
        return False

print("="*80)
print("🔍 DEBUG FULL PIPELINE")
print("="*80)

query = "технологический стек проекта RAUII"
space = "RAUII"
limit = 10

print(f"\nQuery: '{query}'")
print(f"Space: {space}")
print(f"Limit: {limit}\n")

# === ЭТАП 1: Vector Search ===
print("ЭТАП 1: Vector Search")
from embeddings import generate_query_embedding
from qdrant_storage import search_in_qdrant

emb = generate_query_embedding(query)
vector_results_raw = search_in_qdrant(emb, limit=50, space=space)
print(f"  Найдено: {len(vector_results_raw)} результатов")
check_target_in_results(vector_results_raw, "Vector")

# === ЭТАП 2: Hybrid Search (RRF) ===
print("\nЭТАП 2: Hybrid Search (Vector + BM25 с RRF)")
from hybrid_search import hybrid_search, init_bm25_retriever
from qdrant_storage import init_qdrant_client

# Конвертируем формат
vector_results = []
for r in vector_results_raw:
    vector_results.append({
        'id': r['id'],
        'score': r['score'],
        'metadata': r.get('payload', {}),
        'text': r.get('payload', {}).get('text', '')
    })

qdrant_client = init_qdrant_client()
hybrid_results = hybrid_search(
    query=query,
    qdrant_client=qdrant_client,
    vector_results=vector_results,
    space_filter=space,
    limit=100  # Берём больше чтобы не отсекать
)
print(f"  Найдено: {len(hybrid_results)} результатов")
check_target_in_results(hybrid_results, "Hybrid")

# === ЭТАП 3: Дедупликация ===
print("\nЭТАП 3: Дедупликация")
from deduplication import deduplicate_results
dedup_results = deduplicate_results(hybrid_results)
print(f"  После дедупликации: {len(dedup_results)} результатов")
check_target_in_results(dedup_results, "Dedup")

# === ЭТАП 4: Reranking (с ограничением) ===
print("\nЭТАП 4: Reranking")
# Ограничиваем до 9 (как в логах)
rerank_input = dedup_results[:9]
print(f"  Ограничение для reranking: {len(dedup_results)} → {len(rerank_input)}")
target_before_rerank = check_target_in_results(rerank_input, "Rerank Input")

if not target_before_rerank:
    print(f"\n  🔴 ПРОБЛЕМА: Target отсечён ДО reranking!")
    print(f"  Target был на позиции > 9 после дедупликации")
    print(f"\n  Проверяю позицию target в dedup_results:")
    for i, r in enumerate(dedup_results, 1):
        metadata = r.get('metadata', {})
        if metadata.get('page_id') == TARGET_PAGE_ID:
            score = r.get('score', 0.0)
            print(f"    ⭐ Target на позиции #{i}, score={score:.6f}")
            
            print(f"\n  Топ-10 после дедупликации:")
            for j, r2 in enumerate(dedup_results[:10], 1):
                meta2 = r2.get('metadata', {})
                page_id2 = meta2.get('page_id', 'N/A')
                score2 = r2.get('score', 0.0)
                marker = " ⭐" if page_id2 == TARGET_PAGE_ID else ""
                print(f"    #{j}: page_id={page_id2}, score={score2:.6f}{marker}")
            break

# === ВЫВОД ===
print("\n" + "="*80)
print("📊 ИТОГИ")
print("="*80)
print(f"""
Vector Search:   {'✅ Target найден' if check_target_in_results(vector_results_raw, '') else '❌ Target потерян'}
Hybrid Search:   {'✅ Target найден' if check_target_in_results(hybrid_results, '') else '❌ Target потерян'}
Дедупликация:    {'✅ Target найден' if check_target_in_results(dedup_results, '') else '❌ Target потерян'}
Rerank Input:    {'✅ Target найден' if target_before_rerank else '❌ Target ОТСЕЧЁН'}
""")

if not target_before_rerank:
    print("🔴 ПРОБЛЕМА: Target отсечён при ограничении rerank limit!")
    print("   РЕШЕНИЕ: Увеличить rerank_limit или изменить adaptive rerank logic")

