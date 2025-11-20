#!/usr/bin/env python3
"""Тест space filtering в Qdrant."""
import sys
sys.path.insert(0, '/app')

from qdrant_storage import search_in_qdrant
from embeddings import generate_query_embedding

print("Генерирую тестовый embedding...")
emb = generate_query_embedding('технологический стек')

print("\nТест 1: Поиск БЕЗ space фильтра")
r1 = search_in_qdrant(emb, limit=5)
print(f"  Найдено: {len(r1)} результатов")
if r1:
    spaces = [r.get('payload', {}).get('space', 'N/A') for r in r1]
    print(f"  Spaces: {set(spaces)}")

print("\nТест 2: Поиск С space='RAUII'")
r2 = search_in_qdrant(emb, limit=5, space='RAUII')
print(f"  Найдено: {len(r2)} результатов")
if r2:
    spaces = [r.get('payload', {}).get('space', 'N/A') for r in r2]
    page_ids = [r.get('payload', {}).get('page_id', 'N/A') for r in r2]
    print(f"  Все spaces: {set(spaces)}")
    print(f"  Page IDs: {page_ids}")
    
    # Проверяем есть ли 18153591
    if '18153591' in page_ids:
        print(f"  ⭐ НАЙДЕН page_id=18153591!")

print("\n✅ Space filtering работает!" if r2 else "❌ Space filtering НЕ работает!")

