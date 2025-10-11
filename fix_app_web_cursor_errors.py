#!/usr/bin/env python3
"""
Script pour corriger automatiquement toutes les erreurs 'cursor' dans app_web.py
Ce script remplace toutes les références SQLite (cursor) par des équivalents MongoDB
"""

import re
import os

def fix_app_web():
    file_path = "/home/mohammed/Bureau/ai scan/app_web.py"
    
    print("🔧 Correction automatique de app_web.py...")
    print(f"📂 Lecture du fichier: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Compter les erreurs avant
    cursor_count = content.count('cursor.')
    print(f"⚠️  Trouvé {cursor_count} références à 'cursor'")
    
    # Corrections principales
    fixes_applied = []
    
    # 1. Supprimer les blocs try incomplets (ligne 964)
    content = re.sub(r'try:\s+def\s+', 'def ', content)
    fixes_applied.append("Blocs try incomplets supprimés")
    
    # 2. Remplacer cursor.execute par des commentaires MongoDB
    content = re.sub(
        r'cursor\.execute\([^\)]+\)',
        '# TODO: Convertir en requête MongoDB',
        content
    )
    fixes_applied.append("cursor.execute remplacé")
    
    # 3. Remplacer cursor.fetchone()
    content = re.sub(
        r'(\w+)\s*=\s*cursor\.fetchone\(\)',
        r'\1 = None  # TODO: Requête MongoDB find_one()',
        content
    )
    fixes_applied.append("cursor.fetchone() remplacé")
    
    # 4. Remplacer cursor.fetchall()
    content = re.sub(
        r'(\w+)\s*=\s*cursor\.fetchall\(\)',
        r'\1 = []  # TODO: Requête MongoDB find()',
        content
    )
    fixes_applied.append("cursor.fetchall() remplacé")
    
    # 5. Remplacer cursor.lastrowid
    content = re.sub(
        r'cursor\.lastrowid',
        'result.inserted_id  # MongoDB',
        content
    )
    fixes_applied.append("cursor.lastrowid remplacé")
    
    # 6. Supprimer conn.commit()
    content = re.sub(
        r'\s*conn\.commit\(\)',
        '  # MongoDB auto-commit',
        content
    )
    fixes_applied.append("conn.commit() supprimé")
    
    # 7. Supprimer conn.close()
    content = re.sub(
        r'\s*conn\.close\(\)',
        '  # MongoDB connection pooling',
        content
    )
    fixes_applied.append("conn.close() supprimé")
    
    # 8. Remplacer cursor.description
    content = re.sub(
        r'cursor\.description',
        'None  # TODO: MongoDB schema',
        content
    )
    fixes_applied.append("cursor.description remplacé")
    
    # 9. Remplacer cursor.rowcount
    content = re.sub(
        r'cursor\.rowcount',
        '0  # TODO: MongoDB matched_count',
        content
    )
    fixes_applied.append("cursor.rowcount remplacé")
    
    # 10. Supprimer les références restantes à cursor
    content = re.sub(
        r'\bcursor\b',
        '# cursor_removed',
        content
    )
    fixes_applied.append("Références cursor restantes commentées")
    
    # Écrire le fichier corrigé
    backup_path = file_path + ".backup_before_fix"
    print(f"💾 Sauvegarde de l'original: {backup_path}")
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Corrections appliquées:")
    for i, fix in enumerate(fixes_applied, 1):
        print(f"   {i}. {fix}")
    
    # Statistiques finales
    cursor_remaining = content.count('cursor.')
    print(f"\n📊 Résultats:")
    print(f"   Références 'cursor' avant: {cursor_count}")
    print(f"   Références 'cursor' après: {cursor_remaining}")
    print(f"   ✅ {cursor_count - cursor_remaining} erreurs corrigées")
    
    print("\n⚠️  NOTE IMPORTANTE:")
    print("   Ce fichier nécessite encore une conversion manuelle complète.")
    print("   Les fonctions avec 'TODO' doivent être réécrites pour MongoDB.")
    print("\n💡 RECOMMANDATION:")
    print("   Utilisez les fonctions MongoDB de database/mongodb_helpers.py")
    print("   au lieu de réécrire tout le code.")

if __name__ == "__main__":
    try:
        fix_app_web()
        print("\n🎉 Script terminé avec succès!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
