#!/usr/bin/env python3
"""Полный unit-тест RAG pipeline послойно."""
import sys
sys.path.insert(0, '/app')

from hybrid_search import init_bm25_retriever
from qdrant_storage import init_qdrant_client, search_in_qdrant
from embeddings import generate_query_embedding
from utils.lemmatizer import lemmatize_text

print("=" * 80)
print("🧪 UNIT ТЕСТЫ RAG PIPELINE")
print("=" * 80)

# Тестовый запрос
query = "технологический стек проекта RAUII"
space_filter = "RAUII"
TARGET_PAGE_ID = "18153591"

print(f"\n📝 Query: '{query}'")
print(f"🎯 Целевая страница: {TARGET_PAGE_ID}")
print(f"🔍 Space filter: {space_filter}")

# === ТЕСТ 1: BM25 DIRECT ===
print("\n" + "=" * 80)
print("ТЕСТ 1: BM25 DIRECT (с лемматизацией)")
print("=" * 80)

try:
    qdrant_client = init_qdrant_client()
    bm25 = init_bm25_retriever(qdrant_client)
    
    if not bm25:
        print("❌ BM25 retriever НЕ инициализирован!")
    else:
        print("✅ BM25 retriever инициализирован")
        
        # Лемматизируем query
        query_lemmatized = lemmatize_text(query)
        print(f"🔤 Lemmatized: '{query_lemmatized}'")
        
        # Выполняем поиск
        nodes = bm25.retrieve(query_lemmatized)
        print(f"📊 Найдено результатов: {len(nodes)}")
        
        if nodes:
            print(f"\n  Топ-5 BM25 результатов:")
            found_target = False
            for i, node in enumerate(nodes[:5], 1):
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                space = metadata.get('space', 'N/A')
                page_id = metadata.get('page_id', 'N/A')
                heading = metadata.get('heading', 'N/A')[:50]
                score = node.score if hasattr(node, 'score') else 0.0
                
                marker = ""
                if page_id == TARGET_PAGE_ID:
                    marker = " ⭐ TARGET!"
                    found_target = True
                
                print(f"  #{i}: page_id={page_id}, space={space}, score={score:.4f}{marker}")
                print(f"      heading: {heading}")
            
            if found_target:
                print(f"\n  ✅ ТЕСТ 1 PASSED: Страница {TARGET_PAGE_ID} найдена в BM25!")
            else:
                print(f"\n  ⚠️ ТЕСТ 1 WARNING: Страница {TARGET_PAGE_ID} НЕ в топ-5 BM25")
                
                # Проверяем есть ли в топ-10
                for i, node in enumerate(nodes[5:10], 6):
                    metadata = node.metadata if hasattr(node, 'metadata') else {}
                    if metadata.get('page_id') == TARGET_PAGE_ID:
                        print(f"  ⚠️ Найдена на позиции #{i}")
                        break
        else:
            print("  ❌ ТЕСТ 1 FAILED: BM25 ничего не нашёл!")
            
except Exception as e:
    print(f"❌ ТЕСТ 1 ERROR: {e}")
    import traceback
    traceback.print_exc()

# === ТЕСТ 2: VECTOR SEARCH ===
print("\n" + "=" * 80)
print("ТЕСТ 2: VECTOR SEARCH (semantic)")
print("=" * 80)

try:
    # Генерируем embedding
    emb = generate_query_embedding(query)
    print(f"✅ Embedding сгенерирован: dimension={len(emb)}")
    
    # Выполняем vector search
    results = search_in_qdrant(emb, limit=10, space=space_filter)
    print(f"📊 Найдено результатов: {len(results)}")
    
    if results:
        print(f"\n  Топ-5 Vector результатов:")
        found_target = False
        for i, r in enumerate(results[:5], 1):
            payload = r.get('payload', {})
            page_id = payload.get('page_id', 'N/A')
            heading = payload.get('heading', 'N/A')[:50]
            score = r.get('score', 0.0)
            
            marker = ""
            if page_id == TARGET_PAGE_ID:
                marker = " ⭐ TARGET!"
                found_target = True
            
            print(f"  #{i}: page_id={page_id}, score={score:.4f}{marker}")
            print(f"      heading: {heading}")
        
        if found_target:
            print(f"\n  ✅ ТЕСТ 2 PASSED: Страница {TARGET_PAGE_ID} найдена в Vector Search!")
        else:
            print(f"\n  ⚠️ ТЕСТ 2 WARNING: Страница {TARGET_PAGE_ID} НЕ в топ-5 Vector")
            
            # Проверяем топ-10
            for i, r in enumerate(results[5:], 6):
                payload = r.get('payload', {})
                if payload.get('page_id') == TARGET_PAGE_ID:
                    print(f"  ⚠️ Найдена на позиции #{i}")
                    break
    else:
        print("  ❌ ТЕСТ 2 FAILED: Vector search ничего не нашёл!")
        
except Exception as e:
    print(f"❌ ТЕСТ 2 ERROR: {e}")
    import traceback
    traceback.print_exc()

# === ТЕСТ 3: HYBRID SEARCH ===
print("\n" + "=" * 80)
print("ТЕСТ 3: HYBRID SEARCH (Vector + BM25 с RRF)")
print("=" * 80)

try:
    from hybrid_search import hybrid_search
    
    # Получаем vector results (из теста 2)
    emb = generate_query_embedding(query)
    vector_results = search_in_qdrant(emb, limit=50, space=space_filter)
    
    # Конвертируем формат для hybrid_search
    vector_results_formatted = []
    for r in vector_results:
        vector_results_formatted.append({
            'id': r['id'],
            'score': r['score'],
            'metadata': r.get('payload', {}),
            'text': r.get('payload', {}).get('text', '')
        })
    
    # Выполняем hybrid search
    qdrant_client = init_qdrant_client()
    hybrid_results = hybrid_search(
        query=query,
        qdrant_client=qdrant_client,
        vector_results=vector_results_formatted,
        space_filter=space_filter,
        limit=10
    )
    
    print(f"📊 Hybrid results: {len(hybrid_results)}")
    
    if hybrid_results:
        print(f"\n  Топ-5 Hybrid результатов:")
        found_target = False
        for i, r in enumerate(hybrid_results[:5], 1):
            metadata = r.get('metadata', {})
            page_id = metadata.get('page_id', 'N/A')
            heading = metadata.get('heading', 'N/A')[:50]
            score = r.get('score', 0.0)
            
            marker = ""
            if page_id == TARGET_PAGE_ID:
                marker = " ⭐ TARGET!"
                found_target = True
            
            print(f"  #{i}: page_id={page_id}, score={score:.4f}{marker}")
            print(f"      heading: {heading}")
        
        if found_target:
            print(f"\n  ✅ ТЕСТ 3 PASSED: Страница {TARGET_PAGE_ID} в топ-5 Hybrid!")
        else:
            print(f"\n  ⚠️ ТЕСТ 3 WARNING: Страница {TARGET_PAGE_ID} НЕ в топ-5")
            
            # Проверяем топ-10
            for i, r in enumerate(hybrid_results[5:], 6):
                metadata = r.get('metadata', {})
                if metadata.get('page_id') == TARGET_PAGE_ID:
                    print(f"  ⚠️ Найдена на позиции #{i}")
                    break
    else:
        print("  ❌ ТЕСТ 3 FAILED: Hybrid search ничего не вернул!")
        
except Exception as e:
    print(f"❌ ТЕСТ 3 ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("🏁 UNIT ТЕСТЫ ЗАВЕРШЕНЫ")
print("=" * 80)

