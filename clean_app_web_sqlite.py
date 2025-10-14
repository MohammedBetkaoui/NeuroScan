#!/usr/bin/env python3
"""
Script pour nettoyer app_web.py de tout le code SQLite
"""

import re

def clean_sqlite_code():
    file_path = '/home/mohammed/Bureau/ai scan/app_web.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Trouver la position de la fin de init_database
    # Chercher "def save_analysis_to_db"
    match = re.search(r'def init_database\(\):.*?(?=\ndef [a-z_]+\()', content, re.DOTALL)
    if match:
        # Remplacer tout le contenu de init_database
        new_init = '''def init_database():
    """Initialiser MongoDB au lieu de SQLite"""
    try:
        init_mongodb_collections()
        print("✅ MongoDB initialisé avec succès")
    except Exception as e:
        print(f"❌ Erreur initialisation MongoDB: {e}")

'''
        content = content[:match.start()] + new_init + content[match.end():]
    
    # Remplacer save_analysis_to_db par un wrapper
    match = re.search(r'def save_analysis_to_db\(.*?\):.*?(?=\ndef [a-z_]+\()', content, re.DOTALL)
    if match:
        # Extraire la signature
        sig_match = re.search(r'def save_analysis_to_db\((.*?)\):', content[match.start():match.end()])
        if sig_match:
            params = sig_match.group(1)
            new_save = f'''def save_analysis_to_db({params}):
    """Wrapper pour save_analysis_to_db_mongo"""
    return save_analysis_to_db_mongo(results, filename, processing_time, session_id, ip_address, 
                                     patient_id, patient_name, exam_date, doctor_id)

'''
            content = content[:match.start()] + new_save + content[match.end():]
    
    # Sauvegarder
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Code SQLite nettoyé")

if __name__ == '__main__':
    clean_sqlite_code()
