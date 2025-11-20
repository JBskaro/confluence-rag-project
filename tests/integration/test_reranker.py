"""
Test для bge-reranker-v2-m3
Запуск: docker-compose exec confluence-rag python test_reranker.py
"""

import numpy as np
from sentence_transformers import CrossEncoder

print("=" * 60)
print("Загрузка модели BAAI/bge-reranker-v2-m3...")
print("=" * 60)
ranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
print("✅ Модель загружена\n")

query = "технологический стек проекта RAUII"
pairs = [
    [query, "направление стек технология open webui ollama"],  # Highly relevant
    [query, "проект использует python fastapi docker"],  # Medium
    [query, "Префикс проекта NIIRAUII"],  # Low
    [query, "large page excerpt"]  # Not relevant
]

print("=" * 60)
print("ТЕСТ 1: По умолчанию (apply_softmax=False)")
print("=" * 60)
raw_scores = ranker.predict(pairs, apply_softmax=False)  # По умолчанию RAW logits
sigmoid_scores = 1 / (1 + np.exp(-raw_scores))

print("\nRAW LOGITS (от модели):")
for i, s in enumerate(raw_scores):
    print(f"  Pair {i}: {s:.6f}")

print("\nAFTER SIGMOID (наш код):")
for i, s in enumerate(sigmoid_scores):
    print(f"  Pair {i}: {s:.6f}")

print("\n" + "=" * 60)
print("ТЕСТ 2: apply_softmax=True (если нужны probabilities)")
print("=" * 60)
try:
    softmax_scores = ranker.predict(pairs, apply_softmax=True)
    print("\nSOFTMAX SCORES (от модели):")
    for i, s in enumerate(softmax_scores):
        print(f"  Pair {i}: {s:.6f}")
except Exception as e:
    print(f"❌ apply_softmax=True ошибка: {e}")

print("\n" + "=" * 60)
print("ОЖИДАЕМО (normalize=False):")
print("  RAW: [2.5-5.0, 0.5-2.0, -1.0-0.5, -5.0--2.0]")
print("  SIGMOID: [0.92-0.99, 0.62-0.88, 0.27-0.62, 0.01-0.12]")
print("\nЕСЛИ ВСЕ ≈ 0.0 или ≈ 0.50 → ПРОБЛЕМА!")
print("=" * 60)

