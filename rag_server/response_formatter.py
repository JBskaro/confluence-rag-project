#!/usr/bin/env python3
"""
Форматирование ответов для MCP RAG сервера.

Обеспечивает единообразное форматирование результатов поиска, ошибок и пустых результатов.
"""

from typing import List, Dict, Any, Optional

# Стоп-слова для фильтрации заголовков (frozenset для O(1) lookup)
STOP_WORDS = frozenset({
    'в', 'на', 'по', 'для', 'с', 'к', 'из', 'о', 'об', 'и', 'а', 'но', 'или', 'же',
    'the', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'to', 'of', 'and', 'or', 'but'
})


class ResponseFormatter:
    """Класс для форматирования ответов поиска"""
    
    @staticmethod
    def format_success(
        query: str,
        results: List[Dict[str, Any]],
        intent: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        vector_count: Optional[int] = None,
        bm25_count: Optional[int] = None
    ) -> str:
        """
        Форматирует успешный ответ с результатами поиска.
        
        Args:
            query: Поисковый запрос
            results: Список результатов поиска
            intent: Информация о типе запроса (optional)
            latency_ms: Время выполнения в миллисекундах (optional)
            vector_count: Количество векторных результатов (optional)
            bm25_count: Количество BM25 результатов (optional)
            
        Returns:
            Отформатированная строка с результатами
        """
        intent_type = intent.get('type', 'unknown') if intent else 'unknown'
        diversity_limit = intent.get('diversity', 2) if intent else 2
        
        # Заголовок
        header = f"📚 Search Results for: \"{query}\""
        header += "\n" + "━" * 70
        
        # Статистика
        stats_parts = [
            f"Query Type: {intent_type}",
            f"Results: {len(results)}"
        ]
        if latency_ms is not None:
            stats_parts.append(f"Time: {int(latency_ms)}ms")
        if vector_count is not None and bm25_count is not None:
            stats_parts.append(f"Vector: {vector_count}, BM25: {bm25_count}")
        
        stats = " | ".join(stats_parts)
        
        response = [header, stats, ""]
        
        # Результаты
        for i, r in enumerate(results, 1):
            if not r or not isinstance(r, dict):
                continue
            
            m = r.get('metadata', {})
            if not isinstance(m, dict):
                m = {}
            
            # Безопасные геттеры
            title = r.get('title') or r.get('breadcrumb') or m.get('title', 'Без названия')
            page_space = m.get('space', 'Unknown')
            page_url = m.get('url', '') or r.get('url', '')
            chunk_num = m.get('chunk', 0) or r.get('chunk_num', 0)
            
            # Scores (используем rerank_score как основной, т.к. он в диапазоне 0-1)
            rerank_score = r.get('rerank_score', 0)
            final_score = r.get('final_score', rerank_score)
            hierarchy_boost = r.get('hierarchy_boost', 0)
            breadcrumb_boost = r.get('breadcrumb_boost', 0)
            
            # ИСПРАВЛЕНО: Правильные пороги для эмодзи (0-1 диапазон для rerank scores)
            if final_score > 0.7:
                score_emoji = "🟢"
            elif final_score > 0.3:
                score_emoji = "🟡"
            elif final_score > 0.1:
                score_emoji = "🟠"
            else:
                score_emoji = "⚪"
            
            # Формируем строку со score
            score_parts = [f"{score_emoji} {final_score:.3f}"]
            if hierarchy_boost > 0 or breadcrumb_boost > 0:
                score_details = []
                if rerank_score > 0:
                    score_details.append(f"base:{rerank_score:.2f}")
                if hierarchy_boost > 0:
                    score_details.append(f"+hier:{hierarchy_boost:.2f}")
                if breadcrumb_boost > 0:
                    score_details.append(f"+path:{breadcrumb_boost:.2f}")
                score_parts.append(f"({', '.join(score_details)})")
            
            score_str = " | ".join(score_parts)
            
            # Контекст (безопасный геттер)
            context_chunks = r.get('context_chunks', 1)
            context_str = f" | 📚 {context_chunks} chunks" if context_chunks and context_chunks > 1 else ""
            
            # === НОВОЕ: ПОКАЗАТЬ ПУТЬ ===
            breadcrumb = r.get('breadcrumb') or m.get('breadcrumb', '')
            if breadcrumb:
                response.append(f"   📍 Path: {breadcrumb}")
            
            # === НОВОЕ: ПОКАЗАТЬ РЕЛЕВАНТНЫЕ ЗАГОЛОВКИ ===
            headings_list = r.get('headings_list') or m.get('headings_list', [])
            if headings_list and isinstance(headings_list, list) and len(headings_list) > 0:
                # ИСПРАВЛЕНО: Оптимизированная фильтрация с использованием множеств
                query_words_set = set(query.lower().split())
                
                # Убираем стоп-слова для лучшей производительности (используем frozenset)
                query_words_set = {w for w in query_words_set if w not in STOP_WORDS and len(w) > 2}
                
                relevant_headings = []
                
                # Ограничиваем проверку первыми 10 заголовками
                for h in headings_list[:10]:
                    if not query_words_set:  # Если нет ключевых слов, берем первые
                        relevant_headings.append(h)
                        if len(relevant_headings) >= 3:
                            break
                        continue
                    
                    # ИСПРАВЛЕНО: Используем пересечение множеств O(1) вместо O(n*m)
                    heading_words = set(h.lower().split())
                    
                    # Проверяем пересечение множеств (быстрее чем any())
                    if query_words_set & heading_words:  # Пересечение не пустое
                        relevant_headings.append(h)
                        if len(relevant_headings) >= 3:  # Early exit
                            break
                
                if relevant_headings:
                    # Показываем до 3 релевантных заголовков
                    headings_display = ' | '.join(relevant_headings[:3])
                    response.append(f"   📑 Sections: {headings_display}")
                elif len(headings_list) > 0:
                    # Если нет релевантных, показываем первые 3
                    response.append(f"   📑 Sections: {' | '.join(headings_list[:3])}")
            
            # Дополнительная информация (безопасные геттеры)
            extra_info = []
            labels = m.get('labels', '') or r.get('labels', '')
            if labels:
                extra_info.append(f"🏷️ {labels}")
            created_by = m.get('created_by', '') or r.get('created_by', '')
            if created_by:
                extra_info.append(f"👤 {created_by}")
            attachments = m.get('attachments', '') or r.get('attachments', '')
            if attachments:
                att_list = str(attachments).split(',')[:3]
                att_preview = ', '.join(att_list)
                if len(str(attachments).split(',')) > 3:
                    att_preview += f" (+{len(str(attachments).split(',')) - 3})"
                extra_info.append(f"📎 {att_preview}")
            
            extra_str = " | ".join(extra_info)
            if extra_str:
                extra_str = f" | {extra_str}"
            
            # Текст (безопасный геттер)
            text = r.get('expanded_text') or r.get('text', "[Текст недоступен]")
            text_preview = text[:500] + "..." if len(str(text)) > 500 else str(text)
            
            # Формируем результат
            response.append(f"{i}. {title} {score_emoji}")
            response.append(f"   • Space: {page_space} | Chunk #{chunk_num} | {score_str}{context_str}{extra_str}")
            if page_url:
                response.append(f"   • URL: {page_url}")
            response.append(f"   • Preview: {text_preview}")
            response.append("")
        
        return "\n".join(response)
    
    @staticmethod
    def format_no_results(
        query: str,
        intent: Optional[Dict[str, Any]] = None,
        vector_count: int = 0,
        bm25_count: int = 0,
        threshold: Optional[float] = None,
        suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Форматирует ответ когда результатов не найдено.
        
        Args:
            query: Поисковый запрос
            intent: Информация о типе запроса (optional)
            vector_count: Количество векторных результатов (optional)
            bm25_count: Количество BM25 результатов (optional)
            threshold: Порог фильтрации (optional)
            suggestions: Список предложений (optional)
            
        Returns:
            Отформатированная строка с сообщением об отсутствии результатов
        """
        intent_type = intent.get('type', 'unknown') if intent else 'unknown'
        
        response = [
            f"🔍 No Results Found for: \"{query}\"",
            "━" * 70,
            f"Query Type: {intent_type}",
            ""
        ]
        
        if vector_count > 0 or bm25_count > 0:
            response.append(f"⚠️ Found {vector_count + bm25_count} candidates, but all were filtered.")
            if threshold is not None:
                response.append(f"   Filter threshold: {threshold:.6f}")
            response.append("")
        
        if suggestions:
            response.append("💡 Suggestions:")
            for suggestion in suggestions:
                response.append(f"   • {suggestion}")
            response.append("")
        else:
            response.append("💡 Try:")
            response.append("   • Rephrasing your query")
            response.append("   • Using different keywords")
            response.append("   • Checking if the space filter is correct")
            response.append("")
        
        return "\n".join(response)
    
    @staticmethod
    def format_error(
        query: str,
        error: Exception,
        suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Форматирует ответ при ошибке поиска.
        
        Args:
            query: Поисковый запрос
            error: Объект исключения
            suggestions: Список предложений (optional)
            
        Returns:
            Отформатированная строка с сообщением об ошибке
        """
        response = [
            f"❌ Search Error for: \"{query}\"",
            "━" * 70,
            f"Error: {str(error)}",
            ""
        ]
        
        if suggestions:
            response.append("💡 Suggestions:")
            for suggestion in suggestions:
                response.append(f"   • {suggestion}")
        else:
            response.append("💡 Please try again or contact support if the problem persists.")
        
        response.append("")
        return "\n".join(response)
    
    @staticmethod
    def format_low_relevance(
        query: str,
        threshold: float,
        intent: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Форматирует ответ когда результаты имеют низкую релевантность.
        
        Args:
            query: Поисковый запрос
            threshold: Порог фильтрации
            intent: Информация о типе запроса (optional)
            min_score: Минимальный score найденных результатов (optional)
            max_score: Максимальный score найденных результатов (optional)
            suggestions: Список предложений (optional)
            
        Returns:
            Отформатированная строка с предупреждением о низкой релевантности
        """
        intent_type = intent.get('type', 'unknown') if intent else 'unknown'
        
        response = [
            f"⚠️ Low Relevance Results for: \"{query}\"",
            "━" * 70,
            f"Query Type: {intent_type}",
            f"Threshold: {threshold:.6f}",
            ""
        ]
        
        if min_score is not None and max_score is not None:
            response.append(f"Score range: {min_score:.6f} - {max_score:.6f}")
            response.append("")
        
        response.append("All found results were filtered due to low relevance scores.")
        response.append("")
        
        if suggestions:
            response.append("💡 Suggestions:")
            for suggestion in suggestions:
                response.append(f"   • {suggestion}")
        else:
            response.append("💡 Try:")
            response.append("   • Rephrasing your query")
            response.append("   • Using more specific terms")
            response.append("   • Checking if the query matches the content")
        
        response.append("")
        return "\n".join(response)

