# reset_index.ps1 - Полная очистка индекса и состояния синхронизации
# Использование: .\reset_index.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ПОЛНАЯ ОЧИСТКА ИНДЕКСА" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Остановка контейнера
Write-Host "[1/5] Остановка контейнера..." -ForegroundColor Yellow
docker-compose down
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Предупреждение: docker-compose down завершился с ошибкой" -ForegroundColor Yellow
}

# Удаление chroma_data
Write-Host "[2/5] Удаление chroma_data..." -ForegroundColor Yellow
if (Test-Path chroma_data) {
    Remove-Item -Recurse -Force chroma_data -ErrorAction SilentlyContinue
    Write-Host "✅ chroma_data удален" -ForegroundColor Green
} else {
    Write-Host "ℹ️  chroma_data не найден" -ForegroundColor Gray
}

# Удаление sync_state.json
Write-Host "[3/5] Удаление sync_state.json..." -ForegroundColor Yellow
if (Test-Path data/sync_state.json) {
    Remove-Item data/sync_state.json -Force -ErrorAction SilentlyContinue
    Write-Host "✅ sync_state.json удален" -ForegroundColor Green
} else {
    Write-Host "ℹ️  sync_state.json не найден" -ForegroundColor Gray
}

# Перезапуск контейнера
Write-Host "[4/5] Перезапуск контейнера..." -ForegroundColor Yellow
docker-compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Ошибка при запуске контейнера" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Контейнер запущен" -ForegroundColor Green

# Ожидание и проверка
Write-Host "[5/5] Ожидание индексации (30 секунд)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ПРОВЕРКА СТАТУСА" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Проверка количества документов
Write-Host "Проверка количества документов..." -ForegroundColor Yellow
$docCount = docker-compose exec -T confluence-rag python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_data'); coll=c.get_collection('confluence'); print(coll.count())" 2>&1 | Select-Object -Last 1

if ($docCount -match '^\d+$') {
    Write-Host "✅ Документов в индексе: $docCount" -ForegroundColor Green
    if ([int]$docCount -gt 0) {
        Write-Host "✅ Индексация прошла успешно!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Индексация еще не завершена. Проверьте логи:" -ForegroundColor Yellow
        Write-Host "   docker-compose logs confluence-rag --follow" -ForegroundColor Gray
    }
} else {
    Write-Host "⚠️  Не удалось проверить количество документов" -ForegroundColor Yellow
    Write-Host "   Проверьте логи: docker-compose logs confluence-rag --tail=50" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Для просмотра логов используйте:" -ForegroundColor Cyan
Write-Host "   docker-compose logs confluence-rag --follow" -ForegroundColor Gray
Write-Host ""

