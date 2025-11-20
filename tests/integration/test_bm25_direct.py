#!/usr/bin/env python3
"""Прямой тест BM25 retriever с лемматизацией."""
import sys
sys.path.insert(0, '/app')

from hybrid_search import init_bm25_retriever
from qdrant_storage import init_qdrant_client
from utils.lemmatizer import lemmatize_text

# Инициализация
print("Инициализация BM25 retriever...")
qdrant_client = init_qdrant_client()
bm25 = init_bm25_retriever(qdrant_client)

if not bm25:
    print("❌ BM25 retriever НЕ инициализирован!")
    sys.exit(1)

print("✅ BM25 retriever готов\n")

# Тест 1: Оригинальный query
query = "технологический стек проекта RAUII"
query_lemmatized = lemmatize_text(query)

print(f"Query: '{query}'")
print(f"Lemmatized: '{query_lemmatized}'\n")

# Выполнить поиск
print("Выполняю BM25 поиск...")
try:
    nodes = bm25.retrieve(query_lemmatized)
    print(f"Найдено результатов: {len(nodes)}\n")
    
    if nodes:
        print("Топ-5 результатов:")
        for i, node in enumerate(nodes[:5], 1):
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            space = metadata.get('space', 'N/A')
            page_id = metadata.get('page_id', 'N/A')
            heading = metadata.get('heading', 'N/A')
            score = node.score if hasattr(node, 'score') else 0.0
            
            print(f"\n  #{i}:")
            print(f"    Page ID: {page_id}")
            print(f"    Space: {space}")
            print(f"    Heading: {heading[:60]}")
            print(f"    Score: {score:.4f}")
            
            # Проверка на страницу 18153591
            if page_id == '18153591':
                print(f"    ⭐ НАШЛИ СТРАНИЦУ 18153591!")
    else:
        print("❌ BM25 НЕ НАШЁЛ НИ ОДНОГО РЕЗУЛЬТАТА!")
        print("\nВозможные причины:")
        print("1. Лемматизированный query не совпадает с лемматизированными документами")
        print("2. BM25 индекс пуст или повреждён")
        print("3. Токенизация query/documents различается")
        
except Exception as e:
    print(f"❌ Ошибка BM25 поиска: {e}")
    import traceback
    traceback.print_exc()

