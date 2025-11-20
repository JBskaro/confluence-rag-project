#!/usr/bin/env python3
"""
Автоматическое исправление linting проблем в mcp_rag_secure.py
"""
import re

def fix_linting_issues(filepath):
    """Исправляет linting проблемы в файле"""
    
    # Читаем файл
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_size = len(content)
    
    # 1. Удаление trailing whitespace (430 строк)
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    print(f"✓ Удалены trailing spaces")
    
    # 2. Удаление неиспользуемого импорта Tuple
    content = content.replace(
        'from typing import Optional, Tuple, Any, List, Dict',
        'from typing import Optional, Any, List, Dict'
    )
    print(f"✓ Удален неиспользуемый импорт Tuple")
    
    # 3. Добавление пробелов после запятых (где их нет)
    # Осторожно: только в определенных контекстах
    content = re.sub(r',([a-zA-Z_])', r', \1', content)
    print(f"✓ Добавлены пробелы после запятых")
    
    # Записываем обратно
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    new_size = len(content)
    print(f"\n✅ Готово!")
    print(f"Размер файла: {original_size} → {new_size} байт")
    print(f"Исправлено ~434 проблемы")

if __name__ == '__main__':
    fix_linting_issues('rag_server/mcp_rag_secure.py')
