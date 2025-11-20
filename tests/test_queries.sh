#!/bin/bash
# test_queries.sh - Автоматическое тестирование запросов

echo "======================================================================"
echo "ТЕСТ 1: Учет номенклатуры"
echo "======================================================================"

curl -X POST http://localhost:8012/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "test-1",
    "method": "tools/call",
    "params": {
      "name": "confluence_semantic_search",
      "arguments": {
        "query": "Провожу обследование компании по блоку Склад, а точнее Учет номенклатуры. Подготовь список вопросов.",
        "limit": 20
      }
    }
  }' | tee test_result_1.json

echo ""
echo "======================================================================"
echo "ТЕСТ 2: Технологический стек"
echo "======================================================================"

curl -X POST http://localhost:8012/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "test-2",
    "method": "tools/call",
    "params": {
      "name": "confluence_semantic_search",
      "arguments": {
        "query": "Ккой тхнологиеский стек исользуется в проект рау ии.",
        "limit": 10,
        "space": "RAUII"
      }
    }
  }' | tee test_result_2.json

echo ""
echo "======================================================================"
echo "ПРОВЕРКА РЕЗУЛЬТАТОВ"
echo "======================================================================"

if grep -q "18153754" test_result_1.json; then
  echo "[PASSED] Тест 1: Найдена страница 18153754"
else
  echo "[FAILED] Тест 1: Страница 18153754 не найдена"
fi

if grep -q "18153591" test_result_2.json; then
  echo "[PASSED] Тест 2: Найдена страница 18153591"
else
  echo "[FAILED] Тест 2: Страница 18153591 не найдена"
fi

