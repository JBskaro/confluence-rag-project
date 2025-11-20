#!/usr/bin/env python3
"""
Тест порогов reranking после обновления.
Проверяет, что пороги применяются правильно.
"""
import sys
import os
sys.path.insert(0, '/app/rag_server')

from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_thresholds():
    """Тестирует пороги с разными типами запросов."""
    print("=" * 80)
    print("ТЕСТ ПОРОГОВ RERANKING")
    print("=" * 80)
    
    # Проверка переменных окружения
    threshold_technical = float(os.getenv('RERANK_THRESHOLD_TECHNICAL', '0.01'))
    threshold_general = float(os.getenv('RERANK_THRESHOLD_GENERAL', '0.001'))
    model_name = os.getenv('RE_RANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
    
    print(f"\nКонфигурация:")
    print(f"  Модель: {model_name}")
    print(f"  RERANK_THRESHOLD_TECHNICAL: {threshold_technical}")
    print(f"  RERANK_THRESHOLD_GENERAL: {threshold_general}")
    
    # Инициализация reranker
    print(f"\nИнициализация reranker...")
    try:
        ranker = CrossEncoder(model_name)
        print("✅ Reranker инициализирован")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return 1
    
    # Тестовые пары с разной релевантностью
    test_cases = [
        {
            "name": "Высокая релевантность",
            "pairs": [
                ("уточняющие вопросы для обследования по учету номенклатуры на складе", 
                 "Для проведения обследования по учету номенклатуры на складе необходимо подготовить уточняющие вопросы. Эти вопросы помогут выявить особенности учета различных позиций."),
            ]
        },
        {
            "name": "Средняя релевантность",
            "pairs": [
                ("как настроить API", 
                 "Настройка API включает в себя создание endpoints, настройку аутентификации и определение формата данных."),
            ]
        },
        {
            "name": "Низкая релевантность",
            "pairs": [
                ("уточняющие вопросы для обследования", 
                 "Сегодня хорошая погода. Солнце светит ярко. Люди гуляют в парке."),
            ]
        },
    ]
    
    print(f"\nТестирование {len(test_cases)} сценариев...")
    print("-" * 80)
    
    all_results = []
    
    for case in test_cases:
        print(f"\n📋 {case['name']}:")
        pairs = [(q, d) for q, d in case['pairs']]
        scores = ranker.predict(pairs)
        scores_list = list(scores) if hasattr(scores, '__len__') and not isinstance(scores, list) else scores
        
        for i, (query, doc), score in zip(range(len(pairs)), case['pairs'], scores_list):
            passed_technical = score >= threshold_technical
            passed_general = score >= threshold_general
            
            status_tech = "✅" if passed_technical else "❌"
            status_gen = "✅" if passed_general else "❌"
            
            print(f"  Score: {score:.6f} | Technical ({threshold_technical}): {status_tech} | General ({threshold_general}): {status_gen}")
            print(f"    Query: {query[:50]}...")
            
            all_results.append({
                'case': case['name'],
                'score': float(score),
                'passed_technical': passed_technical,
                'passed_general': passed_general
            })
    
    # Итоговая статистика
    print(f"\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print("-" * 80)
    
    total = len(all_results)
    passed_tech = sum(1 for r in all_results if r['passed_technical'])
    passed_gen = sum(1 for r in all_results if r['passed_general'])
    
    print(f"Всего тестов: {total}")
    print(f"Прошло технический порог ({threshold_technical}): {passed_tech}/{total} ({passed_tech*100//total}%)")
    print(f"Прошло общий порог ({threshold_general}): {passed_gen}/{total} ({passed_gen*100//total}%)")
    
    if all_results:
        scores_only = [r['score'] for r in all_results]
        print(f"\nДиапазон scores: {min(scores_only):.6f} - {max(scores_only):.6f}")
        print(f"Средний score: {sum(scores_only)/len(scores_only):.6f}")
    
    print("\n" + "=" * 80)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(test_thresholds())

