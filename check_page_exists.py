#!/usr/bin/env python3
"""Проверка наличия страницы pageId=18153754 в базе"""
import sys

try:
    from server import collection
    
    # Ищем страницу по page_id
    print("Поиск страницы pageId=18153754...")
    all_data = collection.get(limit=10000)
    
    page_ids = all_data.get('ids', [])
    metadatas = all_data.get('metadatas', [])
    
    found = False
    for idx, metadata in enumerate(metadatas):
        if metadata and metadata.get('page_id') == '18153754':
            found = True
            print(f"\n✅ СТРАНИЦА НАЙДЕНА!")
            print(f"   ID в ChromaDB: {page_ids[idx]}")
            print(f"   Title: {metadata.get('title', 'N/A')}")
            print(f"   Space: {metadata.get('space', 'N/A')}")
            print(f"   URL: {metadata.get('url', 'N/A')}")
            print(f"   Chunk: {metadata.get('chunk', 'N/A')}")
            break
    
    if not found:
        print("\n❌ Страница pageId=18153754 НЕ НАЙДЕНА в базе!")
        print(f"Всего документов в базе: {len(page_ids)}")
        
        # Ищем похожие страницы
        print("\nПоиск похожих страниц с 'номенклатур' в названии...")
        nom_pages = []
        for idx, metadata in enumerate(metadatas):
            if metadata:
                title = metadata.get('title', '').lower()
                if 'номенклатур' in title:
                    nom_pages.append({
                        'id': page_ids[idx],
                        'title': metadata.get('title', 'N/A'),
                        'page_id': metadata.get('page_id', 'N/A'),
                        'space': metadata.get('space', 'N/A')
                    })
        
        if nom_pages:
            print(f"\nНайдено {len(nom_pages)} страниц:")
            for page in nom_pages[:10]:
                print(f"  - {page['title']} (pageId={page['page_id']}, space={page['space']})")
        else:
            print("Страницы с 'номенклатур' не найдены")
    
except Exception as e:
    print(f"❌ ОШИБКА: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

