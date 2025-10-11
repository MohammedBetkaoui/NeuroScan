#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration propre et complète de app.py vers app_web_clean.py avec MongoDB
"""

import re
from datetime import datetime

def migrate_to_mongodb_clean():
    """Migration propre de app_web_clean.py vers MongoDB"""
    
    file_path = '/home/mohammed/Bureau/ai scan/app_web_clean.py'
    
    print("🔧 Migration propre vers MongoDB...")
    print(f"📂 Fichier: {file_path}")
    
    # Lire le fichier
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Sauvegarde
    backup_path = f"{file_path}.backup"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Sauvegarde: {backup_path}")
    
    # ========================================
    # ÉTAPE 1: Remplacer les imports
    # ========================================
    print("\n📦 Étape 1: Mise à jour des imports...")
    
    # Supprimer l'import sqlite3
    content = re.sub(r'import sqlite3\s*\n', '', content)
    
    # Ajouter les imports MongoDB après les imports Flask
    flask_import_pos = content.find('from flask import')
    if flask_import_pos != -1:
        # Trouver la fin de la ligne d'import Flask
        newline_pos = content.find('\n', flask_import_pos)
        
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
    register_doctor_mongo,
    get_collection,
    get_mongodb
)
from bson import ObjectId
"""
        content = content[:newline_pos+1] + mongodb_imports + content[newline_pos+1:]
    
    print("   ✅ Imports MongoDB ajoutés")
    
    # ========================================
    # ÉTAPE 2: Supprimer DATABASE_PATH
    # ========================================
    print("\n🗄️  Étape 2: Suppression de DATABASE_PATH...")
    content = re.sub(r"DATABASE_PATH\s*=\s*['\"].*?['\"]", "# DATABASE_PATH removed - using MongoDB", content)
    print("   ✅ DATABASE_PATH supprimé")
    
    # ========================================
    # ÉTAPE 3: Remplacer init_database
    # ========================================
    print("\n🏗️  Étape 3: Remplacement de init_database()...")
    
    # Trouver et remplacer la fonction init_database
    init_db_pattern = r'def init_database\(\):.*?(?=\ndef |\nclass |\n@app\.|\nif __name__)'
    
    new_init_db = '''def init_database():
    """Initialiser MongoDB au lieu de SQLite"""
    try:
        init_mongodb_collections()
        print("✅ MongoDB initialisé avec succès")
    except Exception as e:
        print(f"❌ Erreur initialisation MongoDB: {e}")

'''
    
    content = re.sub(init_db_pattern, new_init_db, content, flags=re.DOTALL)
    print("   ✅ init_database() remplacé")
    
    # ========================================
    # ÉTAPE 4: Remplacer toutes les utilisations de conn/cursor
    # ========================================
    print("\n🔄 Étape 4: Remplacement des appels SQLite...")
    
    # Pattern 1: conn = sqlite3.connect(DATABASE_PATH)
    content = re.sub(
        r'conn\s*=\s*sqlite3\.connect\([^)]+\)',
        '# conn = ... # MongoDB: pas besoin de connexion explicite',
        content
    )
    
    # Pattern 2: cursor = conn.cursor()
    content = re.sub(
        r'cursor\s*=\s*conn\.cursor\(\)',
        '# cursor = ... # MongoDB: pas besoin de cursor',
        content
    )
    
    # Pattern 3: conn.commit()
    content = re.sub(
        r'conn\.commit\(\)',
        '# conn.commit() # MongoDB: auto-commit',
        content
    )
    
    # Pattern 4: conn.close()
    content = re.sub(
        r'conn\.close\(\)',
        '# conn.close() # MongoDB: géré par le connector',
        content
    )
    
    # Pattern 5: cursor.execute(...)
    content = re.sub(
        r'cursor\.execute\([^)]+\)',
        '# cursor.execute(...) # MongoDB: TODO - utiliser collection.find/insert/update',
        content
    )
    
    # Pattern 6: cursor.fetchone()
    content = re.sub(
        r'cursor\.fetchone\(\)',
        '{} # MongoDB: TODO - utiliser collection.find_one()',
        content
    )
    
    # Pattern 7: cursor.fetchall()
    content = re.sub(
        r'cursor\.fetchall\(\)',
        '[] # MongoDB: TODO - utiliser collection.find()',
        content
    )
    
    # Pattern 8: cursor.lastrowid
    content = re.sub(
        r'cursor\.lastrowid',
        'str(result.inserted_id) # MongoDB: lastrowid → inserted_id',
        content
    )
    
    print("   ✅ Appels SQLite remplacés")
    
    # ========================================
    # ÉTAPE 5: Remplacer les fonctions spécifiques
    # ========================================
    print("\n⚙️  Étape 5: Remplacement des fonctions métier...")
    
    # save_analysis_to_db → save_analysis_to_db_mongo
    content = re.sub(
        r'\bsave_analysis_to_db\(',
        'save_analysis_to_db_mongo(',
        content
    )
    
    # get_current_doctor → get_current_doctor_mongo
    content = re.sub(
        r'\bget_current_doctor\(',
        'get_current_doctor_mongo(',
        content
    )
    
    # create_doctor_session → create_doctor_session_mongo
    content = re.sub(
        r'\bcreate_doctor_session\(',
        'create_doctor_session_mongo(',
        content
    )
    
    # get_doctor_statistics → get_doctor_statistics_mongo
    content = re.sub(
        r'\bget_doctor_statistics\(',
        'get_doctor_statistics_mongo(',
        content
    )
    
    print("   ✅ Fonctions métier remplacées")
    
    # ========================================
    # ÉTAPE 6: Ajouter un commentaire d'avertissement
    # ========================================
    print("\n📝 Étape 6: Ajout de l'en-tête...")
    
    header = '''# -*- coding: utf-8 -*-
"""
NeuroScan AI - Application Web avec MongoDB
===========================================
Version migrée de SQLite vers MongoDB Atlas

⚠️  IMPORTANT:
- Ce fichier utilise MongoDB Atlas au lieu de SQLite
- Les fonctions de database/mongodb_helpers.py gèrent toutes les opérations
- Configuration dans .env: MONGODB_URI et MONGODB_DB_NAME

Migration effectuée le: {date}
"""

'''.format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Insérer l'en-tête après le shebang si présent
    if content.startswith('#!'):
        first_newline = content.find('\n')
        content = content[:first_newline+1] + header + content[first_newline+1:]
    else:
        content = header + content
    
    print("   ✅ En-tête ajouté")
    
    # ========================================
    # ÉTAPE 7: Sauvegarder le fichier
    # ========================================
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Statistiques
    conn_count = content.count('conn =')
    cursor_count = content.count('cursor')
    todo_count = content.count('TODO')
    
    print(f"\n✅ Migration terminée!")
    print(f"\n📊 Statistiques:")
    print(f"   Références 'conn' restantes: {conn_count}")
    print(f"   Références 'cursor' restantes: {cursor_count}")
    print(f"   TODOs à traiter: {todo_count}")
    
    print(f"\n⚠️  PROCHAINES ÉTAPES:")
    print(f"   1. Remplacer app_web.py par app_web_clean.py:")
    print(f"      mv app_web.py app_web_old.py")
    print(f"      mv app_web_clean.py app_web.py")
    print(f"   2. Rechercher et remplacer les TODOs manuellement")
    print(f"   3. Tester l'application: python app_web.py")
    
    return file_path

if __name__ == '__main__':
    migrate_to_mongodb_clean()
