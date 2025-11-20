#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to auto-fix whitespace and blank line issues in mcp_rag_secure.py"""

import re

file_path = r'c:\MCP\confluence-rag-project — копия (4)\rag_server\mcp_rag_secure.py'

print("Reading file...")
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"Original size: {len(content)} bytes")

# Fix 1: Remove trailing whitespace from all lines
lines = content.split('\n')
lines = [line.rstrip() for line in lines]
content = '\n'.join(lines)

# Fix 2: Remove excessive blank lines (more than 2 consecutive)
content = re.sub(r'\n\n\n+', '\n\n', content)

# Fix 3: Ensure 2 blank lines before top-level function/class definitions
# (but not at the start of file or after decorators)
content = re.sub(r'(\n[^\n\s#@].*\n)\n(def [a-z_])', r'\1\n\n\2', content)
content = re.sub(r'(\n[^\n\s#@].*\n)\n(class [A-Z])', r'\1\n\n\2', content)

print(f"Fixed size: {len(content)} bytes")

print("Writing file...")
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Whitespace issues fixed!")
print("\nRemaining issues to fix manually:")
print("1. Unused imports")
print("2. Undefined variables")
print("3. Complexity warnings")
