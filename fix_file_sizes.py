"""
Script pour ajouter file_size_formatted aux fichiers existants dans MongoDB
"""
import os
import sys
from database.mongodb_connector import get_collection

def format_file_size(size_bytes):
    """Formater la taille du fichier"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def fix_file_sizes():
    """Ajouter file_size_formatted aux fichiers qui n'en ont pas"""
    try:
        files_collection = get_collection('message_files')
        
        if files_collection is None:
            print("❌ Erreur: Impossible de se connecter à la collection message_files")
            return
        
        # Trouver tous les fichiers sans file_size_formatted
        files_without_formatted = files_collection.find({
            '$or': [
                {'file_size_formatted': {'$exists': False}},
                {'file_size_formatted': '0 B'},
                {'file_size_formatted': ''}
            ]
        })
        
        count = 0
        updated = 0
        
        for file_doc in files_without_formatted:
            count += 1
            file_size = file_doc.get('file_size', 0)
            
            if file_size > 0:
                formatted_size = format_file_size(file_size)
                
                # Mettre à jour le document
                files_collection.update_one(
                    {'_id': file_doc['_id']},
                    {'$set': {'file_size_formatted': formatted_size}}
                )
                
                updated += 1
                print(f"✅ Fichier {file_doc.get('original_filename')} : {formatted_size}")
            else:
                print(f"⚠️  Fichier {file_doc.get('original_filename')} : Taille = 0")
        
        print(f"\n📊 Résumé:")
        print(f"   - Fichiers trouvés: {count}")
        print(f"   - Fichiers mis à jour: {updated}")
        
        if updated > 0:
            print(f"\n✅ Correction terminée avec succès!")
        else:
            print(f"\n ℹ️  Aucun fichier à corriger")
        
    except Exception as e:
        print(f"❌ Erreur lors de la correction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("🔧 Correction des tailles de fichiers dans MongoDB\n")
    fix_file_sizes()
