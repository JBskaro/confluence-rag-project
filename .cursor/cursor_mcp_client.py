#!/usr/bin/env python3
"""
MCP client для Cursor IDE
Проксирует запросы между Cursor и локальным MCP сервером
"""
import sys
import json
import requests
import os

MCP_URL = os.getenv('MCP_SERVER_URL', 'http://localhost:8012/mcp')

def main():
    """Читает запросы из stdin, отправляет к MCP серверу, пишет ответы в stdout"""
    
    # Читаем построчно из stdin
    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:
                continue
            
            # Парсим JSON запрос от Cursor
            request = json.loads(line)
            
            # Отправляем к MCP серверу
            response = requests.post(
                MCP_URL,
                json=request,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            
            # Пишем ответ в stdout
            sys.stdout.write(response.text + "\n")
            sys.stdout.flush()
            
        except json.JSONDecodeError as e:
            # Ошибка парсинга JSON
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
            
        except requests.RequestException as e:
            # Ошибка HTTP запроса
            error_response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if 'request' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": f"HTTP error: {str(e)}"
                }
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            # Другие ошибки
            error_response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if 'request' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()

