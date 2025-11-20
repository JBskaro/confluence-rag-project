# PowerShell скрипт для запуска всех unit тестов (без контейнера)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "ЗАПУСК ВСЕХ UNIT ТЕСТОВ" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$FAILED = 0
$PASSED = 0

# Новые тесты для новых модулей
Write-Host ""
Write-Host "--- Тесты keyword_extraction ---" -ForegroundColor Yellow
python tests/test_keyword_extraction.py
if ($LASTEXITCODE -eq 0) {
    $PASSED++
} else {
    $FAILED++
}

Write-Host ""
Write-Host "--- Тесты intent_config ---" -ForegroundColor Yellow
python tests/test_intent_config.py
if ($LASTEXITCODE -eq 0) {
    $PASSED++
} else {
    $FAILED++
}

Write-Host ""
Write-Host "--- Тесты response_formatter ---" -ForegroundColor Yellow
python tests/test_response_formatter.py
if ($LASTEXITCODE -eq 0) {
    $PASSED++
} else {
    $FAILED++
}

# Существующие unit тесты
Write-Host ""
Write-Host "--- Тесты embeddings ---" -ForegroundColor Yellow
python tests/test_embeddings.py
if ($LASTEXITCODE -eq 0) {
    $PASSED++
} else {
    $FAILED++
}

Write-Host ""
Write-Host "--- Тесты hybrid_search ---" -ForegroundColor Yellow
python tests/test_hybrid_search.py
if ($LASTEXITCODE -eq 0) {
    $PASSED++
} else {
    $FAILED++
}

Write-Host ""
Write-Host "--- Тесты query_rewriter ---" -ForegroundColor Yellow
python tests/test_query_rewriter.py
if ($LASTEXITCODE -eq 0) {
    $PASSED++
} else {
    $FAILED++
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "ИТОГО: $PASSED passed, $FAILED failed" -ForegroundColor $(if ($FAILED -eq 0) { "Green" } else { "Red" })
Write-Host "======================================================================" -ForegroundColor Cyan

exit $FAILED

