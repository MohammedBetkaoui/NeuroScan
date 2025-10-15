"""
NeuroScan AI - Gestionnaire de Fichiers
Gestion sécurisée de l'upload et du stockage des fichiers pour la messagerie
"""

import os
import uuid
import mimetypes
from datetime import datetime
from werkzeug.utils import secure_filename
from pathlib import Path

# Configuration
UPLOAD_FOLDER = 'uploads/messages'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {
    'images': {'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'},
    'documents': {'pdf', 'doc', 'docx', 'txt', 'rtf', 'odt'},
    'spreadsheets': {'xls', 'xlsx', 'csv', 'ods'},
    'medical': {'dcm', 'nii', 'nii.gz'},  # DICOM, NIfTI
    'archives': {'zip', 'rar', '7z', 'tar', 'gz'}
}

# Types MIME autorisés
ALLOWED_MIME_TYPES = {
    # Images
    'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp', 'image/svg+xml',
    # Documents
    'application/pdf', 
    'application/msword', 
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain', 'application/rtf',
    # Spreadsheets
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'text/csv',
    # Medical
    'application/dicom',
    # Archives
    'application/zip', 'application/x-rar-compressed', 'application/x-7z-compressed'
}

class FileManager:
    """Gestionnaire de fichiers pour la messagerie"""
    
    def __init__(self, upload_folder=UPLOAD_FOLDER):
        self.upload_folder = upload_folder
        self.ensure_upload_folder()
    
    def ensure_upload_folder(self):
        """Créer le dossier d'upload s'il n'existe pas"""
        Path(self.upload_folder).mkdir(parents=True, exist_ok=True)
    
    def get_file_extension(self, filename):
        """Obtenir l'extension du fichier"""
        return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    def get_file_category(self, extension):
        """Déterminer la catégorie du fichier"""
        for category, extensions in ALLOWED_EXTENSIONS.items():
            if extension in extensions:
                return category
        return 'other'
    
    def is_allowed_file(self, filename):
        """Vérifier si le fichier est autorisé"""
        ext = self.get_file_extension(filename)
        
        # Vérifier l'extension
        for extensions in ALLOWED_EXTENSIONS.values():
            if ext in extensions:
                return True
        return False
    
    def is_allowed_mime(self, mime_type):
        """Vérifier si le type MIME est autorisé"""
        return mime_type in ALLOWED_MIME_TYPES
    
    def generate_unique_filename(self, original_filename):
        """Générer un nom de fichier unique et sécurisé"""
        # Sécuriser le nom original
        safe_filename = secure_filename(original_filename)
        
        # Obtenir l'extension
        ext = self.get_file_extension(safe_filename)
        
        # Générer un UUID
        unique_id = str(uuid.uuid4())
        
        # Créer le nouveau nom
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"{timestamp}_{unique_id}.{ext}"
        
        return new_filename
    
    def get_file_size_formatted(self, size_bytes):
        """Formater la taille du fichier"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def validate_file(self, file):
        """Valider un fichier uploadé"""
        errors = []
        
        # Vérifier si un fichier a été fourni
        if not file or file.filename == '':
            errors.append("Aucun fichier sélectionné")
            return False, errors
        
        # Vérifier le nom du fichier
        if not self.is_allowed_file(file.filename):
            errors.append(f"Type de fichier non autorisé: {self.get_file_extension(file.filename)}")
        
        # Vérifier le type MIME
        mime_type = file.content_type
        if mime_type and not self.is_allowed_mime(mime_type):
            errors.append(f"Type MIME non autorisé: {mime_type}")
        
        # Vérifier la taille (si possible)
        if hasattr(file, 'content_length') and file.content_length:
            if file.content_length > MAX_FILE_SIZE:
                max_size_formatted = self.get_file_size_formatted(MAX_FILE_SIZE)
                errors.append(f"Fichier trop volumineux (max: {max_size_formatted})")
        
        if errors:
            return False, errors
        
        return True, []
    
    def save_file(self, file, subfolder=None):
        """
        Sauvegarder un fichier uploadé
        
        Args:
            file: Fichier Flask (FileStorage)
            subfolder: Sous-dossier optionnel (ex: conversation_id)
        
        Returns:
            dict: Informations sur le fichier sauvegardé ou None en cas d'erreur
        """
        # Valider le fichier
        is_valid, errors = self.validate_file(file)
        if not is_valid:
            return {
                'success': False,
                'errors': errors
            }
        
        try:
            # Générer un nom unique
            original_filename = secure_filename(file.filename)
            unique_filename = self.generate_unique_filename(original_filename)
            
            # Déterminer le chemin de sauvegarde
            if subfolder:
                save_folder = os.path.join(self.upload_folder, subfolder)
                Path(save_folder).mkdir(parents=True, exist_ok=True)
            else:
                save_folder = self.upload_folder
            
            # Chemin complet
            file_path = os.path.join(save_folder, unique_filename)
            
            # Sauvegarder le fichier
            file.save(file_path)
            
            # Obtenir les informations du fichier
            file_size = os.path.getsize(file_path)
            file_extension = self.get_file_extension(original_filename)
            file_category = self.get_file_category(file_extension)
            
            # Chemin relatif pour la base de données
            relative_path = os.path.join('messages', subfolder, unique_filename) if subfolder else os.path.join('messages', unique_filename)
            
            return {
                'success': True,
                'original_filename': original_filename,
                'stored_filename': unique_filename,
                'file_path': file_path,
                'relative_path': relative_path,
                'file_size': file_size,
                'file_size_formatted': self.get_file_size_formatted(file_size),
                'file_extension': file_extension,
                'file_category': file_category,
                'mime_type': file.content_type,
                'uploaded_at': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Erreur lors de la sauvegarde: {str(e)}"]
            }
    
    def delete_file(self, file_path):
        """Supprimer un fichier"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"Erreur lors de la suppression du fichier: {e}")
            return False
    
    def get_file_info(self, file_path):
        """Obtenir les informations d'un fichier"""
        if not os.path.exists(file_path):
            return None
        
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_extension = self.get_file_extension(file_name)
        file_category = self.get_file_category(file_extension)
        
        # Type MIME
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Dates
        created_at = datetime.fromtimestamp(os.path.getctime(file_path))
        modified_at = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        return {
            'filename': file_name,
            'file_path': file_path,
            'file_size': file_size,
            'file_size_formatted': self.get_file_size_formatted(file_size),
            'file_extension': file_extension,
            'file_category': file_category,
            'mime_type': mime_type,
            'created_at': created_at.isoformat(),
            'modified_at': modified_at.isoformat()
        }
    
    def list_files(self, subfolder=None):
        """Lister les fichiers dans un dossier"""
        folder = os.path.join(self.upload_folder, subfolder) if subfolder else self.upload_folder
        
        if not os.path.exists(folder):
            return []
        
        files = []
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                info = self.get_file_info(file_path)
                if info:
                    files.append(info)
        
        return files
    
    def get_storage_stats(self):
        """Obtenir les statistiques de stockage"""
        total_size = 0
        total_files = 0
        categories = {}
        
        for root, dirs, files in os.walk(self.upload_folder):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                total_files += 1
                
                # Catégorie
                ext = self.get_file_extension(filename)
                category = self.get_file_category(ext)
                if category not in categories:
                    categories[category] = {'count': 0, 'size': 0}
                categories[category]['count'] += 1
                categories[category]['size'] += file_size
        
        return {
            'total_files': total_files,
            'total_size': total_size,
            'total_size_formatted': self.get_file_size_formatted(total_size),
            'categories': {
                cat: {
                    'count': stats['count'],
                    'size': stats['size'],
                    'size_formatted': self.get_file_size_formatted(stats['size'])
                }
                for cat, stats in categories.items()
            }
        }

# Instance globale
file_manager = FileManager()

def get_file_manager():
    """Obtenir l'instance du gestionnaire de fichiers"""
    return file_manager
