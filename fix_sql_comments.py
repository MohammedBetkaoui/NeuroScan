#!/usr/bin/env python3
"""
Script pour corriger les commentaires SQL mal formatés dans app_web.py
"""

import re

def fix_sql_comments(filepath):
    """Lire le fichier et corriger les blocs SQL mal commentés"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern pour trouver les blocs SQL mal commentés
    # Chercher: # cursor.execute( # DISABLED''' suivi de SQL puis ''', (...)
    pattern = r"# cursor\.execute\( # DISABLED(?:f)?'''(.*?)'''(?:, \([^)]*\))?"
    
    def replace_sql_block(match):
        """Remplacer le bloc SQL par un commentaire propre"""
        sql_content = match.group(1)
        # Mettre tout le SQL en commentaire ligne par ligne
        lines = sql_content.strip().split('\n')
        commented = '\n        '.join(f'# {line}' if line.strip() else '#' for line in lines)
        return f'# MongoDB query needed here\n        {commented}'
    
    # Remplacer tous les blocs SQL mal formatés
    content_fixed = re.sub(pattern, replace_sql_block, content, flags=re.DOTALL)
    
    # Sauvegarder
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content_fixed)
    
    print(f"✅ Fichier corrigé: {filepath}")

if __name__ == '__main__':
    fix_sql_comments('/home/mohammed/Bureau/ai scan/app_web.py')
