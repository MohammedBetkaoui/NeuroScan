"""
NeuroScan AI - Gestion des uploads de fichiers
Gestion du stockage et récupération des fichiers partagés
"""

import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from database.mongodb_connector import get_database

# Configuration
UPLOAD_FOLDER = 'uploads/messages'
ALLOWED_EXTENSIONS = {
    'images': {'png', 'jpg', 'jpeg', 'gif', 'webp'},
    'documents': {'pdf', 'doc', 'docx', 'txt', 'rtf'},
    'medical': {'dcm', 'nii', 'nii.gz'},
    'archives': {'zip', 'rar', '7z'}
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

def init_upload_folder():
    """Créer le dossier d'upload s'il n'existe pas"""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        print(f"✅ Dossier d'upload créé: {UPLOAD_FOLDER}")

def get_file_category(filename):
    """Déterminer la catégorie du fichier"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    for category, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return category
    
    return 'other'

def is_allowed_file(filename):
    """Vérifier si le fichier est autorisé"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    # Vérifier dans toutes les catégories
    for extensions in ALLOWED_EXTENSIONS.values():
        if ext in extensions:
            return True
    
    return False

def get_file_size_mb(size_bytes):
    """Convertir la taille en MB"""
    return round(size_bytes / (1024 * 1024), 2)

def save_uploaded_file(file, doctor_id, conversation_id=None):
    """
    Sauvegarder un fichier uploadé
    
    Args:
        file: Fichier Flask (werkzeug.datastructures.FileStorage)
        doctor_id: ID du médecin qui upload
        conversation_id: ID de la conversation (optionnel)
    
    Returns:
        dict: Informations du fichier sauvegardé ou None si erreur
    """
    try:
        # Vérifier que le fichier existe
        if not file or file.filename == '':
            return {'error': 'Aucun fichier sélectionné'}
        
        # Vérifier l'extension
        if not is_allowed_file(file.filename):
            return {'error': 'Type de fichier non autorisé'}
        
        # Vérifier la taille (si possible)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return {'error': f'Fichier trop volumineux (max {get_file_size_mb(MAX_FILE_SIZE)} MB)'}
        
        # Créer le dossier si nécessaire
        init_upload_folder()
        
        # Générer un nom de fichier unique
        original_filename = secure_filename(file.filename)
        file_ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Sauvegarder le fichier
        file.save(file_path)
        
        # Préparer les métadonnées
        file_info = {
            'file_id': str(uuid.uuid4()),
            'original_name': original_filename,
            'stored_name': unique_filename,
            'file_path': file_path,
            'file_size': file_size,
            'file_size_mb': get_file_size_mb(file_size),
            'file_type': file_ext,
            'category': get_file_category(original_filename),
            'uploaded_by': doctor_id,
            'conversation_id': conversation_id,
            'uploaded_at': datetime.now(),
            'download_count': 0
        }
        
        # Enregistrer dans MongoDB
        db = get_database()
        if db is not None:
            files_collection = db['message_files']
            result = files_collection.insert_one(file_info)
            file_info['_id'] = str(result.inserted_id)
        
        print(f"✅ Fichier sauvegardé: {original_filename} ({get_file_size_mb(file_size)} MB)")
        
        return {
            'success': True,
            'file_id': file_info['file_id'],
            'original_name': original_filename,
            'file_size': file_size,
            'file_size_mb': get_file_size_mb(file_size),
            'file_type': file_ext,
            'category': file_info['category'],
            'file_path': f"/uploads/messages/{unique_filename}",
            'uploaded_at': file_info['uploaded_at'].isoformat()
        }
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde fichier: {e}")
        return {'error': str(e)}

def get_file_info(file_id):
    """Récupérer les informations d'un fichier"""
    try:
        db = get_database()
        if db is not None:
            files_collection = db['message_files']
            file_info = files_collection.find_one({'file_id': file_id})
            
            if file_info:
                file_info['_id'] = str(file_info['_id'])
                return file_info
        
        return None
        
    except Exception as e:
        print(f"❌ Erreur récupération fichier: {e}")
        return None

def increment_download_count(file_id):
    """Incrémenter le compteur de téléchargements"""
    try:
        db = get_database()
        if db is not None:
            files_collection = db['message_files']
            files_collection.update_one(
                {'file_id': file_id},
                {'$inc': {'download_count': 1}}
            )
        
    except Exception as e:
        print(f"❌ Erreur incrémentation téléchargements: {e}")

def delete_file(file_id, doctor_id):
    """
    Supprimer un fichier
    
    Args:
        file_id: ID du fichier
        doctor_id: ID du médecin (doit être l'uploader)
    
    Returns:
        bool: True si succès, False sinon
    """
    try:
        # Récupérer les infos du fichier
        file_info = get_file_info(file_id)
        
        if not file_info:
            return False
        
        # Vérifier que c'est bien l'uploader
        if file_info['uploaded_by'] != doctor_id:
            print(f"❌ Tentative de suppression non autorisée")
            return False
        
        # Supprimer le fichier physique
        if os.path.exists(file_info['file_path']):
            os.remove(file_info['file_path'])
        
        # Supprimer de la DB
        db = get_database()
        if db is not None:
            files_collection = db['message_files']
            files_collection.delete_one({'file_id': file_id})
        
        print(f"✅ Fichier supprimé: {file_info['original_name']}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur suppression fichier: {e}")
        return False

def get_conversation_files(conversation_id):
    """Récupérer tous les fichiers d'une conversation"""
    try:
        db = get_database()
        if db is not None:
            files_collection = db['message_files']
            files = list(files_collection.find({'conversation_id': conversation_id}))
            
            for file in files:
                file['_id'] = str(file['_id'])
            
            return files
        
        return []
        
    except Exception as e:
        print(f"❌ Erreur récupération fichiers conversation: {e}")
        return []

def get_doctor_files(doctor_id, limit=50):
    """Récupérer les fichiers uploadés par un médecin"""
    try:
        db = get_database()
        if db is not None:
            files_collection = db['message_files']
            files = list(files_collection.find({'uploaded_by': doctor_id})
                        .sort('uploaded_at', -1)
                        .limit(limit))
            
            for file in files:
                file['_id'] = str(file['_id'])
            
            return files
        
        return []
        
    except Exception as e:
        print(f"❌ Erreur récupération fichiers médecin: {e}")
        return []

def get_storage_stats(doctor_id=None):
    """Obtenir les statistiques de stockage"""
    try:
        db = get_database()
        if db is not None:
            files_collection = db['message_files']
            
            # Filtre optionnel par médecin
            query = {'uploaded_by': doctor_id} if doctor_id else {}
            
            # Compter les fichiers
            total_files = files_collection.count_documents(query)
            
            # Calculer la taille totale
            pipeline = [
                {'$match': query},
                {'$group': {
                    '_id': None,
                    'total_size': {'$sum': '$file_size'},
                    'total_files': {'$sum': 1}
                }}
            ]
            
            result = list(files_collection.aggregate(pipeline))
            
            if result:
                total_size = result[0]['total_size']
                total_size_mb = get_file_size_mb(total_size)
            else:
                total_size = 0
                total_size_mb = 0
            
            return {
                'total_files': total_files,
                'total_size': total_size,
                'total_size_mb': total_size_mb,
                'max_size_mb': get_file_size_mb(MAX_FILE_SIZE),
                'allowed_extensions': ALLOWED_EXTENSIONS
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Erreur statistiques stockage: {e}")
        return None

# Initialiser le dossier au chargement du module
init_upload_folder()

print("✅ Module file_upload chargé")
