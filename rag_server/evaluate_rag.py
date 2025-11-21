import os
import json
import asyncio
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from rag_server.config import settings
from rag_server.mcp_rag_secure import confluence_semantic_search

# Настройка OpenAI API Key для Ragas
if settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

async def generate_rag_answers(questions: List[str]) -> List[Dict[str, Any]]:
    """
    Генерирует ответы используя текущий RAG pipeline.
    """
    results = []
    for q in questions:
        # Выполняем поиск через наш RAG
        # Используем mcp tool как интерфейс
        search_result_text = await confluence_semantic_search(q, limit=5)
        
        # В реальном сценарии здесь должен быть вызов LLM для генерации ответа на основе search_result_text.
        # Для оценки retrieval metrics (context_recall, context_precision) нам достаточно контекста.
        # Для оценки generation metrics (faithfulness, answer_relevancy) нужен ответ LLM.
        
        # Эмуляция ответа LLM (или реальный вызов если есть integration)
        # Пока используем простой placeholder или можно интегрировать вызов LLM если есть.
        # Для целей данного скрипта, предположим мы оцениваем retrieval качество в основном.
        
        # Парсим результаты поиска обратно (это текст) чтобы достать контексты
        contexts = []
        if "✅ Найдено" in search_result_text:
            # Простой парсинг текста результатов
            lines = search_result_text.split('\n')
            current_context = ""
            for line in lines:
                if "💬" in line:
                    current_context = line.replace("💬", "").strip()
                    contexts.append(current_context)
        
        # Заглушка для ответа, так как у нас только retrieval часть сейчас exposed через mcp tool явно.
        # В полноценном пайплайне тут был бы вызов generate_response(query, contexts)
        answer = "Generated answer based on retrieved contexts." 

        results.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            # Ground truth добавляется из датасета
        })
    
    return results

def run_evaluation(golden_dataset_path: str = "data/golden_dataset.json"):
    """
    Запуск оценки RAG пайплайна.
    """
    if not os.path.exists(golden_dataset_path):
        print(f"⚠️ Dataset not found at {golden_dataset_path}. Please create one first.")
        return

    with open(golden_dataset_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)

    questions = [item['question'] for item in golden_data]
    ground_truths = [[item['ground_truth']] for item in golden_data] # Ragas ожидает list of lists

    print(f"🚀 Starting evaluation for {len(questions)} questions...")

    # Генерация ответов системой
    rag_outputs = asyncio.run(generate_rag_answers(questions))
    
    # Подготовка данных для Ragas
    data = {
        'question': questions,
        'answer': [item['answer'] for item in rag_outputs],
        'contexts': [item['contexts'] for item in rag_outputs],
        'ground_truth': ground_truths
    }
    
    dataset = Dataset.from_dict(data)

    # Оценка
    # Используем только retrieval метрики если нет реальной генерации ответов
    metrics = [
        context_recall, 
        context_precision
    ]
    
    # Если есть OpenAI ключ, можно оценить и генерацию (faithfulness, answer_relevancy)
    if os.environ.get("OPENAI_API_KEY"):
        metrics.extend([faithfulness, answer_relevancy])
    else:
        print("⚠️ OpenAI API Key not found. Skipping generation metrics (faithfulness, answer_relevancy).")

    results = evaluate(
        dataset=dataset,
        metrics=metrics
    )

    print("\n📊 Evaluation Results:")
    print(results)
    
    # Сохранение результатов
    df = results.to_pandas()
    output_path = "docs/analysis/rag_evaluation_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Detailed results saved to {output_path}")

if __name__ == "__main__":
    # Пример создания dummy датасета если нет
    if not os.path.exists("data/golden_dataset.json"):
        os.makedirs("data", exist_ok=True)
        dummy_data = [
            {
                "question": "Как настроить Qdrant?",
                "ground_truth": "Для настройки Qdrant необходимо задать host и port в конфигурации."
            },
            {
                "question": "Какие метрики используются для оценки?",
                "ground_truth": "Используются faithfulness, answer_relevancy, context_recall и context_precision."
            }
        ]
        with open("data/golden_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, ensure_ascii=False, indent=2)
        print("Created dummy golden_dataset.json for testing.")

    run_evaluation()

