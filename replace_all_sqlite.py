#!/usr/bin/env python3
"""
Remplacer TOUTES les références sqlite3 par des retours vides ou des TODO
"""

import re

def replace_all_sqlite():
    file_path = '/home/mohammed/Bureau/ai scan/app_web.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Sauvegarder
    with open(file_path + '.before_global_replace', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Compter avant
    sqlite_count = content.count('sqlite3')
    
    # Remplacer toutes les connexions sqlite3 par des blocs try-except qui retournent des valeurs vides
    # Pattern: conn = sqlite3.connect(DATABASE_PATH)
    content = re.sub(
        r'conn\s*=\s*sqlite3\.connect\([^)]+\)',
        '# conn = sqlite3.connect() # DISABLED - MongoDB used instead',
        content
    )
    
    # Pattern: cursor = conn.cursor()
    content = re.sub(
        r'cursor\s*=\s*conn\.cursor\(\)',
        '# cursor = conn.cursor() # DISABLED',
        content
    )
    
    # Pattern: cursor.execute(...)
    content = re.sub(
        r'cursor\.execute\(',
        '# cursor.execute( # DISABLED',
        content
    )
    
    # Pattern: conn.commit()
    content = re.sub(
        r'conn\.commit\(\)',
        '# conn.commit() # DISABLED',
        content
    )
    
    # Pattern: conn.close()
    content = re.sub(
        r'conn\.close\(\)',
        '# conn.close() # DISABLED',
        content
    )
    
    # Pattern: cursor.fetchone()
    content = re.sub(
        r'(\w+)\s*=\s*cursor\.fetchone\(\)',
        r'\1 = None  # cursor.fetchone() # DISABLED',
        content
    )
    
    # Pattern: cursor.fetchall()
    content = re.sub(
        r'(\w+)\s*=\s*cursor\.fetchall\(\)',
        r'\1 = []  # cursor.fetchall() # DISABLED',
        content
    )
    
    # Sauvegarder
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    sqlite_count_after = content.count('sqlite3')
    
    print(f"✅ Références sqlite3:")
    print(f"   Avant: {sqlite_count}")
    print(f"   Après: {sqlite_count_after}")
    print(f"   💾 Sauvegarde: {file_path}.before_global_replace")

if __name__ == '__main__':
    replace_all_sqlite()
