#!/usr/bin/env python3
"""
Прямой тест reranker для проверки scores.
"""
import sys
import os
sys.path.insert(0, '/app/rag_server')

from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reranker():
    """Тестирует reranker напрямую."""
    print("=" * 80)
    print("ПРЯМОЙ ТЕСТ RERANKER")
    print("=" * 80)
    
    # Проверка переменных окружения
    model_name = os.getenv('RE_RANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
    print(f"\nМодель: {model_name}")
    print(f"RERANK_THRESHOLD_TECHNICAL: {os.getenv('RERANK_THRESHOLD_TECHNICAL', 'НЕ УСТАНОВЛЕНО')}")
    print(f"RERANK_THRESHOLD_GENERAL: {os.getenv('RERANK_THRESHOLD_GENERAL', 'НЕ УСТАНОВЛЕНО')}")
    
    # Инициализация reranker
    print(f"\nИнициализация reranker...")
    try:
        ranker = CrossEncoder(model_name)
        print("✅ Reranker инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return 1
    
    # Тестовые пары
    test_pairs = [
        ("уточняющие вопросы для обследования по учету номенклатуры на складе", 
         "Для проведения обследования по учету номенклатуры на складе необходимо подготовить уточняющие вопросы. Эти вопросы помогут выявить особенности учета различных позиций."),
        ("как настроить API", 
         "Настройка API включает в себя создание endpoints, настройку аутентификации и определение формата данных."),
        ("процессы складского учета", 
         "Процессы складского учета включают приемку товаров, инвентаризацию, списание и перемещение между складами."),
    ]
    
    print(f"\nТестирование {len(test_pairs)} пар...")
    print("-" * 80)
    
    # Получаем scores
    pairs = [(query, doc) for query, doc in test_pairs]
    scores = ranker.predict(pairs)
    
    # Диагностика
    print(f"\nДИАГНОСТИКА SCORES:")
    print(f"  Тип: {type(scores)}")
    print(f"  Длина: {len(scores) if hasattr(scores, '__len__') else 'N/A'}")
    
    scores_list = list(scores) if hasattr(scores, '__len__') and not isinstance(scores, list) else scores
    
    if len(scores_list) > 0:
        print(f"  Min: {min(scores_list):.6f}")
        print(f"  Max: {max(scores_list):.6f}")
        print(f"  Среднее: {sum(scores_list)/len(scores_list):.6f}")
        print(f"\nДетальные scores:")
        for i, (query, doc), score in zip(range(len(test_pairs)), test_pairs, scores_list):
            print(f"  [{i}] Score: {score:.6f} ({type(score)})")
            print(f"      Query: {query[:50]}...")
            print(f"      Doc: {doc[:50]}...")
    
    # Проверка порогов
    print(f"\nПРОВЕРКА ПОРОГОВ:")
    threshold_technical = float(os.getenv('RERANK_THRESHOLD_TECHNICAL', '0.0001'))
    threshold_general = float(os.getenv('RERANK_THRESHOLD_GENERAL', '0.00001'))
    
    print(f"  RERANK_THRESHOLD_TECHNICAL: {threshold_technical}")
    print(f"  RERANK_THRESHOLD_GENERAL: {threshold_general}")
    
    passed_technical = sum(1 for s in scores_list if s >= threshold_technical)
    passed_general = sum(1 for s in scores_list if s >= threshold_general)
    
    print(f"\nРезультаты фильтрации:")
    print(f"  Прошло технический порог ({threshold_technical}): {passed_technical}/{len(scores_list)}")
    print(f"  Прошло общий порог ({threshold_general}): {passed_general}/{len(scores_list)}")
    
    if all(s < threshold_general for s in scores_list):
        print(f"\n⚠️  ВНИМАНИЕ: Все scores ниже общего порога!")
        print(f"   Это объясняет, почему все результаты отфильтровываются.")
    
    print("\n" + "=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(test_reranker())

