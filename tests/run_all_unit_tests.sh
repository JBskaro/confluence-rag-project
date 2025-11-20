#!/bin/bash
# Скрипт для запуска всех unit тестов (без контейнера)

echo "======================================================================"
echo "ЗАПУСК ВСЕХ UNIT ТЕСТОВ"
echo "======================================================================"

FAILED=0
PASSED=0

# Новые тесты для новых модулей
echo ""
echo "--- Тесты keyword_extraction ---"
python tests/test_keyword_extraction.py
if [ $? -eq 0 ]; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

echo ""
echo "--- Тесты intent_config ---"
python tests/test_intent_config.py
if [ $? -eq 0 ]; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

echo ""
echo "--- Тесты response_formatter ---"
python tests/test_response_formatter.py
if [ $? -eq 0 ]; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Существующие unit тесты
echo ""
echo "--- Тесты embeddings ---"
python tests/test_embeddings.py
if [ $? -eq 0 ]; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

echo ""
echo "--- Тесты hybrid_search ---"
python tests/test_hybrid_search.py
if [ $? -eq 0 ]; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

echo ""
echo "--- Тесты query_rewriter ---"
python tests/test_query_rewriter.py
if [ $? -eq 0 ]; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

echo ""
echo "======================================================================"
echo "ИТОГО: $PASSED passed, $FAILED failed"
echo "======================================================================"

exit $FAILED

