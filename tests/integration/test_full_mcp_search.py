#!/usr/bin/env python3
"""
Тестовый MCP запрос с полным пайплайном (vector search + reranking).
"""
import sys
sys.path.insert(0, '/app')
import json

# Имитация MCP запроса
query = "Какой технологический стек используется в проекте RAUII?"
space = "RAUII"
limit = 10

print(f"=== MCP Search Test ===")
print(f"Query: {query}")
print(f"Space: {space}")
print(f"Limit: {limit}\n")

# Импортируем функцию поиска
# Импортируем напрямую, минуя декоратор
import importlib.util
spec = importlib.util.spec_from_file_location("mcp_rag_secure", "/app/server.py")
mcp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_module)

# Получаем функцию
confluence_semantic_search = mcp_module.confluence_semantic_search

try:
    result = confluence_semantic_search(
        query=query,
        limit=limit,
        space=space
    )
    
    print(f"\n{'='*80}")
    print("РЕЗУЛЬТАТ MCP ПОИСКА:")
    print(f"{'='*80}\n")
    
    # Парсим JSON результат
    if isinstance(result, str):
        result_data = json.loads(result)
    else:
        result_data = result
    
    # Выводим результаты
    if isinstance(result_data, dict):
        print(json.dumps(result_data, indent=2, ensure_ascii=False))
    else:
        print(result_data)
        
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

