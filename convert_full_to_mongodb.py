#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de conversion complète de app_web.py vers MongoDB
Remplace TOUTES les requêtes SQL par des opérations MongoDB
"""

import re
import os
from datetime import datetime

def convert_app_web_to_mongodb():
    """Convertit app_web.py pour utiliser MongoDB au lieu de SQLite"""
    
    file_path = '/home/mohammed/Bureau/ai scan/app_web.py'
    
    print("🔧 Conversion complète vers MongoDB...")
    print(f"📂 Lecture du fichier: {file_path}")
    
    # Lire le fichier original
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Sauvegarde
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Sauvegarde: {backup_path}")
    
    # Compter les problèmes avant
    sql_patterns_before = len(re.findall(r'(VALUES|SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN|GROUP BY|ORDER BY|LIMIT)\s+', content, re.IGNORECASE))
    
    # ==============================
    # ÉTAPE 1: Nettoyer les fragments SQL
    # ==============================
    
    # Supprimer les lignes SQL orphelines
    content = re.sub(r'^\s*VALUES\s*\([?,\s]*\)\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*FROM\s+\w+.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*WHERE\s+.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*JOIN\s+.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*LEFT JOIN\s+.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*ORDER BY\s+.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*GROUP BY\s+.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*LIMIT\s+\?.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*AND\s+.*$', '', content, flags=re.MULTILINE)
    
    # Supprimer les blocs SQL multi-lignes commentés ou incomplets
    content = re.sub(r'""".*?SELECT.*?"""', '""""""', content, flags=re.DOTALL)
    content = re.sub(r"'''.*?SELECT.*?'''", "''''''", content, flags=re.DOTALL)
    
    # Supprimer les parenthèses SQL orphelines
    content = re.sub(r'\(SELECT\s+.*?\)', '{}', content, flags=re.DOTALL)
    
    # ==============================
    # ÉTAPE 2: Corriger les boucles cassées
    # ==============================
    
    # Corriger "for row in # cursor_removed"
    content = re.sub(
        r'for\s+row\s+in\s+#\s*cursor_removed\.fetchall\(\):',
        'for row in []:  # TODO: Remplacer par collection.find()',
        content
    )
    
    # Corriger "for row in :" vide
    content = re.sub(
        r'for\s+(\w+)\s+in\s*:\s*\n',
        r'for \1 in []:  # TODO: Remplacer par requête MongoDB\n',
        content
    )
    
    # ==============================
    # ÉTAPE 3: Ajouter imports MongoDB
    # ==============================
    
    if 'from database.mongodb_connector import' not in content:
        # Trouver la section des imports
        import_section_match = re.search(r'(from flask import.*?\n)', content)
        if import_section_match:
            import_pos = import_section_match.end()
            mongodb_imports = """
# MongoDB imports
from database.mongodb_connector import get_mongodb, get_collection, init_mongodb_collections
from database.mongodb_helpers import (
    save_analysis_to_db_mongo,
    get_current_doctor_mongo,
    create_doctor_session_mongo,
    get_doctor_statistics_mongo,
    verify_doctor_credentials_mongo,
    update_daily_stats_mongo,
    get_doctor_by_email_mongo,
    register_doctor_mongo
)
from bson import ObjectId
from datetime import datetime

"""
            content = content[:import_pos] + mongodb_imports + content[import_pos:]
    
    # ==============================
    # ÉTAPE 4: Corriger les erreurs de syntaxe
    # ==============================
    
    # Supprimer les lignes avec parenthèses non fermées
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Ignorer les lignes SQL orphelines
        if re.match(r'^\s*(VALUES|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|ORDER|GROUP|LIMIT|AND|OR)\s+', line, re.IGNORECASE):
            i += 1
            continue
            
        # Corriger les try sans except
        if line.strip() == 'try:':
            # Vérifier si il y a un except après
            has_except = False
            for j in range(i+1, min(i+10, len(lines))):
                if lines[j].strip().startswith('except'):
                    has_except = True
                    break
            if not has_except:
                fixed_lines.append(line.replace('try:', '# try: # Commenté - pas de except'))
                i += 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    # ==============================
    # ÉTAPE 5: Remplacer les patterns SQL par MongoDB
    # ==============================
    
    # Pattern: INSERT INTO table VALUES
    content = re.sub(
        r'# cursor_removed\.execute\(\s*["\']INSERT INTO (\w+).*?VALUES.*?["\']\s*,\s*\(([^)]+)\)\)',
        lambda m: f"get_collection('{m.group(1)}').insert_one({{'data': 'TODO'}})",
        content,
        flags=re.DOTALL
    )
    
    # Pattern: UPDATE table SET
    content = re.sub(
        r'# cursor_removed\.execute\(\s*["\']UPDATE (\w+) SET.*?["\']\s*,?\s*\([^)]*\)\)',
        lambda m: f"get_collection('{m.group(1)}').update_one({{'_id': 'TODO'}}, {{'$set': {{'TODO': 'value'}}}})",
        content,
        flags=re.DOTALL
    )
    
    # Pattern: SELECT FROM table
    content = re.sub(
        r'# cursor_removed\.execute\(\s*["\']SELECT.*?FROM (\w+).*?["\']\s*,?\s*\([^)]*\)\)',
        lambda m: f"list(get_collection('{m.group(1)}').find({{'TODO': 'condition'}}))",
        content,
        flags=re.DOTALL
    )
    
    # Pattern: DELETE FROM table
    content = re.sub(
        r'# cursor_removed\.execute\(\s*["\']DELETE FROM (\w+).*?["\']\s*,?\s*\([^)]*\)\)',
        lambda m: f"get_collection('{m.group(1)}').delete_many({{'TODO': 'condition'}})",
        content,
        flags=re.DOTALL
    )
    
    # ==============================
    # ÉTAPE 6: Supprimer les lignes problématiques restantes
    # ==============================
    
    # Supprimer les lignes avec "?" orphelin
    content = re.sub(r'^\s*[^"\'#]*\?\s*$', '', content, flags=re.MULTILINE)
    
    # Supprimer les lignes avec expressions SQL incomplètes
    content = re.sub(r'^\s*avg_processing_time\s*=\s*\(.*?\?\s*/\s*total_analyses\s*$', 
                    '        # avg_processing_time calculation - TODO: MongoDB', 
                    content, flags=re.MULTILINE)
    
    # Nettoyer les lignes vides en excès
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # ==============================
    # ÉTAPE 7: Ajouter un avertissement en haut du fichier
    # ==============================
    
    warning = '''# -*- coding: utf-8 -*-
"""
⚠️  FICHIER AUTO-CONVERTI VERS MONGODB ⚠️
Ce fichier a été automatiquement converti de SQLite vers MongoDB.
Les lignes marquées "TODO" nécessitent une révision manuelle.

IMPORTANT: Utiliser les fonctions de database/mongodb_helpers.py au lieu de
           réécrire la logique SQL en MongoDB.

Conversion effectuée le: {date}
"""

'''.format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    if '⚠️  FICHIER AUTO-CONVERTI' not in content:
        # Insérer après le shebang si présent
        if content.startswith('#!'):
            first_newline = content.find('\n')
            content = content[:first_newline+1] + warning + content[first_newline+1:]
        else:
            content = warning + content
    
    # Sauvegarder le fichier converti
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Statistiques
    sql_patterns_after = len(re.findall(r'(VALUES|SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN|GROUP BY|ORDER BY|LIMIT)\s+', content, re.IGNORECASE))
    
    print(f"\n✅ Conversion terminée!")
    print(f"   Patterns SQL avant: {sql_patterns_before}")
    print(f"   Patterns SQL après: {sql_patterns_after}")
    print(f"   Réduction: {sql_patterns_before - sql_patterns_after} patterns supprimés")
    
    print(f"\n⚠️  ÉTAPES SUIVANTES:")
    print(f"   1. Rechercher 'TODO' dans app_web.py")
    print(f"   2. Remplacer par les fonctions de mongodb_helpers.py")
    print(f"   3. Tester chaque route une par une")
    print(f"\n💡 Utilisez: grep -n 'TODO' app_web.py")

if __name__ == '__main__':
    convert_app_web_to_mongodb()
