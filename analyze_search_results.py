#!/usr/bin/env python3
"""
Анализ результатов поиска и сравнение с ожидаемыми
"""
import json
import re
from typing import Dict, List, Tuple

EXPECTED_RESULTS = {
    "test_1": {
        "query": "Провожу обследование компании по блоку Склад, а точнее Учет номенклатуры. Подготовь список вопросов.",
        "expected_page_id": "18153754",
        "expected_keywords": [
            "номенклатур", "классификац", "серии", "характеристик", 
            "габарит", "штрихкод", "Классификация номенклатуры",
            "Учет движения и остатков", "Логистические и финансовые параметры"
        ],
        "expected_questions_count": 30,
        "expected_sections": [
            "1. Классификация номенклатуры",
            "2. Учет движения и остатков",
            "3. Логистические и финансовые параметры",
            "4. Классификация и хранение товаров",
            "5. Дополнительные вопросы"
        ]
    },
    "test_2": {
        "query": "Ккой тхнологиеский стек исользуется в проект рау ии.",
        "expected_page_id": "18153591",
        "expected_keywords": [
            "Ollama", "OpenRouter", "LiteLLM", "MCP", 
            "технологи", "стек", "Syntaxcheck", "Docsearch",
            "Codesearch", "Templatesearch", "Open WebUI"
        ],
        "expected_sections": [
            "Работа с ИИ моделями",
            "Интерфейс и аутентификация"
        ]
    }
}

def analyze_result(test_name: str, actual_text: str) -> Dict:
    """Анализ фактического результата"""
    expected = EXPECTED_RESULTS[test_name]
    
    print(f"\n{'='*80}")
    print(f"АНАЛИЗ: {test_name.upper()}")
    print(f"{'='*80}")
    print(f"\nЗапрос: {expected['query'][:80]}...")
    print(f"Длина ответа: {len(actual_text)} символов")
    
    # Проверка page_id
    page_found = expected['expected_page_id'] in actual_text
    print(f"\n[{'✅' if page_found else '❌'}] Страница {expected['expected_page_id']}: {'НАЙДЕНА' if page_found else 'НЕ НАЙДЕНА'}")
    
    # Поиск всех page_id в тексте
    page_ids = re.findall(r'\d{8,}', actual_text)
    if page_ids:
        print(f"  Найденные page_id в ответе: {', '.join(set(page_ids))}")
    
    # Проверка ключевых слов
    print(f"\nПроверка ключевых слов:")
    found_keywords = []
    missing_keywords = []
    
    for keyword in expected['expected_keywords']:
        if keyword.lower() in actual_text.lower():
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    print(f"  ✅ Найдено: {len(found_keywords)}/{len(expected['expected_keywords'])}")
    if found_keywords:
        print(f"     {', '.join(found_keywords[:10])}")
    if missing_keywords:
        print(f"  ❌ Отсутствуют: {', '.join(missing_keywords[:10])}")
    
    # Проверка разделов (для теста 1)
    if 'expected_sections' in expected:
        print(f"\nПроверка разделов:")
        found_sections = []
        for section in expected['expected_sections']:
            if section.lower() in actual_text.lower():
                found_sections.append(section)
        
        print(f"  ✅ Найдено разделов: {len(found_sections)}/{len(expected['expected_sections'])}")
        if found_sections:
            for section in found_sections:
                print(f"     - {section}")
    
    # Подсчет вопросов (для теста 1)
    if 'expected_questions_count' in expected:
        question_pattern = r'\d+\.\s+[А-ЯЁ]'
        questions = re.findall(question_pattern, actual_text)
        print(f"\n  📊 Найдено вопросов (по паттерну): {len(questions)}")
        print(f"  Ожидалось минимум: {expected['expected_questions_count']}")
    
    # Поиск page_id в тексте для контекста
    if expected['expected_page_id'] in actual_text:
        idx = actual_text.find(expected['expected_page_id'])
        context = actual_text[max(0, idx-100):idx+200]
        print(f"\n  Контекст вокруг page_id:")
        print(f"     ...{context}...")
    else:
        # Показываем начало ответа
        print(f"\n  Начало ответа:")
        print(f"     {actual_text[:300]}...")
    
    # Общая оценка
    score = 0
    max_score = 100
    
    if page_found:
        score += 40
    else:
        print(f"\n  ⚠️  КРИТИЧНО: Ожидаемая страница не найдена!")
    
    keyword_score = (len(found_keywords) / len(expected['expected_keywords'])) * 40
    score += keyword_score
    
    if 'expected_sections' in expected:
        section_score = (len(found_sections) / len(expected['expected_sections'])) * 20
        score += section_score
    
    print(f"\n{'='*80}")
    print(f"ОЦЕНКА: {score:.1f}/100")
    print(f"{'='*80}")
    
    if score >= 80:
        print("✅ РЕЗУЛЬТАТ: ОТЛИЧНО")
    elif score >= 60:
        print("⚠️  РЕЗУЛЬТАТ: ХОРОШО (есть улучшения)")
    elif score >= 40:
        print("❌ РЕЗУЛЬТАТ: ПЛОХО (нужны улучшения)")
    else:
        print("❌ РЕЗУЛЬТАТ: ОТВРАТИТЕЛЬНО (критические проблемы)")
    
    return {
        "page_found": page_found,
        "keywords_found": len(found_keywords),
        "keywords_total": len(expected['expected_keywords']),
        "score": score,
        "found_page_ids": list(set(page_ids)) if page_ids else []
    }

def main():
    print("\n" + "="*80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ПОИСКА")
    print("="*80)
    print("\nВставьте фактический результат поиска из Open WebUI.")
    print("Для выхода введите 'exit'")
    
    results = {}
    
    for test_name, expected in EXPECTED_RESULTS.items():
        print(f"\n{'='*80}")
        print(f"ТЕСТ: {test_name}")
        print(f"{'='*80}")
        print(f"Запрос: {expected['query']}")
        print(f"Ожидаемая страница: {expected['expected_page_id']}")
        
        user_input = input("\nВставьте фактический результат поиска (или 'skip' для пропуска):\n")
        
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input.lower() == 'skip':
            continue
        
        # Пробуем распарсить как JSON (если это ответ от MCP)
        try:
            json_data = json.loads(user_input)
            if 'result' in json_data and 'content' in json_data['result']:
                actual_text = json_data['result']['content'][0].get('text', '')
            else:
                actual_text = user_input
        except:
            actual_text = user_input
        
        results[test_name] = analyze_result(test_name, actual_text)
    
    # Итоги
    if results:
        print(f"\n{'='*80}")
        print("ИТОГОВАЯ СВОДКА")
        print(f"{'='*80}")
        
        for test_name, result in results.items():
            status = "✅" if result['page_found'] else "❌"
            print(f"{status} {test_name}: {result['score']:.1f}/100 "
                  f"(keywords: {result['keywords_found']}/{result['keywords_total']}, "
                  f"pages: {', '.join(result['found_page_ids']) if result['found_page_ids'] else 'нет'})")
        
        avg_score = sum(r['score'] for r in results.values()) / len(results)
        print(f"\nСредняя оценка: {avg_score:.1f}/100")
        
        # Рекомендации
        print(f"\n{'='*80}")
        print("РЕКОМЕНДАЦИИ")
        print(f"{'='*80}")
        
        if avg_score < 40:
            print("❌ КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")
            print("   1. Проверьте, что индекс переиндексирован (Documents > 0)")
            print("   2. Проверьте, что указан правильный space (RAUII, а не Surveys)")
            print("   3. Проверьте пороги reranking (RERANK_THRESHOLD_*)")
            print("   4. Проверьте, что Query Rewriting работает (Ollama fallback)")
        elif avg_score < 60:
            print("⚠️  НУЖНЫ УЛУЧШЕНИЯ:")
            print("   1. Проверьте пороги reranking - возможно, они слишком высокие")
            print("   2. Проверьте, что Query Rewriting работает")
            print("   3. Проверьте, что указан правильный space")

if __name__ == "__main__":
    main()

