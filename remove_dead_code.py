#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to remove dead code from mcp_rag_secure.py (lines 1521-2020)"""

file_path = r'c:\MCP\confluence-rag-project — копия (4)\rag_server\mcp_rag_secure.py'

print("Reading file...")
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines before: {len(lines)}")
print(f"Removing lines 1521-2020 (indices 1520-2019)")

# Keep lines 1-1520 and 2021-end
new_lines = lines[:1520] + lines[2020:]

print(f"Total lines after: {len(new_lines)}")
print(f"Removed {len(lines) - len(new_lines)} lines")

print("Writing file...")
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Dead code removed successfully!")
print("\nNext steps:")
print("1. Check the file for syntax errors")
print("2. Remove unused imports")
print("3. Fix remaining linting issues")
