"""
Connecteur MongoDB pour NeuroScan AI
Gestion de la connexion et des opérations de base de données MongoDB
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

# Charger les variables d'environnement
load_dotenv()

# Configuration MongoDB
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://betkaoui_mohammed:betkaoui@2002@cluster0.xdhjsc1.mongodb.net/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'NeuroScan')

class MongoDBConnector:
    """Classe pour gérer la connexion et les opérations MongoDB"""
    
    _instance = None
    _client = None
    _db = None
    
    def __new__(cls):
        """Singleton pour éviter les connexions multiples"""
        if cls._instance is None:
            cls._instance = super(MongoDBConnector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialiser la connexion MongoDB"""
        if self._client is None:
            self.connect()
    
    def connect(self):
        """Établir la connexion à MongoDB"""
        try:
            self._client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                retryWrites=True,
                w='majority'
            )
            
            # Tester la connexion
            self._client.admin.command('ping')
            
            # Sélectionner la base de données
            self._db = self._client[MONGODB_DB_NAME]
            
            # Initialiser les collections
            self.init_collections()
            
            print(f"✅ Connexion MongoDB établie avec succès - Base: {MONGODB_DB_NAME}")
            return True
            
        except ConnectionFailure as e:
            print(f"❌ Erreur de connexion MongoDB: {e}")
            return False
        except Exception as e:
            print(f"❌ Erreur lors de la connexion MongoDB: {e}")
            return False
    
    def init_collections(self):
        """Initialiser les collections et créer les index"""
        try:
            # Collection analyses
            if 'analyses' not in self._db.list_collection_names():
                self._db.create_collection('analyses')
            
            analyses = self._db.analyses
            analyses.create_index([('timestamp', DESCENDING)])
            analyses.create_index([('patient_id', ASCENDING)])
            analyses.create_index([('doctor_id', ASCENDING)])
            analyses.create_index([('exam_date', DESCENDING)])
            
            # Collection patients
            if 'patients' not in self._db.list_collection_names():
                self._db.create_collection('patients')
            
            patients = self._db.patients
            patients.create_index([('patient_id', ASCENDING), ('doctor_id', ASCENDING)], unique=True)
            patients.create_index([('doctor_id', ASCENDING)])
            
            # Collection tumor_evolution
            if 'tumor_evolution' not in self._db.list_collection_names():
                self._db.create_collection('tumor_evolution')
            
            tumor_evolution = self._db.tumor_evolution
            tumor_evolution.create_index([('patient_id', ASCENDING)])
            tumor_evolution.create_index([('exam_date', DESCENDING)])
            
            # Collection medical_alerts
            if 'medical_alerts' not in self._db.list_collection_names():
                self._db.create_collection('medical_alerts')
            
            medical_alerts = self._db.medical_alerts
            medical_alerts.create_index([('doctor_id', ASCENDING)])
            medical_alerts.create_index([('patient_id', ASCENDING)])
            medical_alerts.create_index([('is_read', ASCENDING)])
            medical_alerts.create_index([('created_at', DESCENDING)])
            
            # Collection notifications
            if 'notifications' not in self._db.list_collection_names():
                self._db.create_collection('notifications')
            
            notifications = self._db.notifications
            notifications.create_index([('doctor_id', ASCENDING)])
            notifications.create_index([('is_read', ASCENDING)])
            notifications.create_index([('created_at', DESCENDING)])
            
            # Collection daily_stats
            if 'daily_stats' not in self._db.list_collection_names():
                self._db.create_collection('daily_stats')
            
            daily_stats = self._db.daily_stats
            daily_stats.create_index([('date', DESCENDING)], unique=True)
            
            # Collection doctors
            if 'doctors' not in self._db.list_collection_names():
                self._db.create_collection('doctors')
            
            doctors = self._db.doctors
            doctors.create_index([('email', ASCENDING)], unique=True)
            
            # Collection doctor_sessions
            if 'doctor_sessions' not in self._db.list_collection_names():
                self._db.create_collection('doctor_sessions')
            
            doctor_sessions = self._db.doctor_sessions
            doctor_sessions.create_index([('session_token', ASCENDING)], unique=True)
            doctor_sessions.create_index([('doctor_id', ASCENDING)])
            doctor_sessions.create_index([('expires_at', ASCENDING)])
            
            # Collection chat_conversations
            if 'chat_conversations' not in self._db.list_collection_names():
                self._db.create_collection('chat_conversations')
            
            chat_conversations = self._db.chat_conversations
            chat_conversations.create_index([('doctor_id', ASCENDING)])
            chat_conversations.create_index([('updated_at', DESCENDING)])
            
            # Collection chat_messages
            if 'chat_messages' not in self._db.list_collection_names():
                self._db.create_collection('chat_messages')
            
            chat_messages = self._db.chat_messages
            chat_messages.create_index([('conversation_id', ASCENDING)])
            chat_messages.create_index([('timestamp', ASCENDING)])
            
            # Collection chat_attachments
            if 'chat_attachments' not in self._db.list_collection_names():
                self._db.create_collection('chat_attachments')
            
            chat_attachments = self._db.chat_attachments
            chat_attachments.create_index([('message_id', ASCENDING)])
            
            print("✅ Collections MongoDB initialisées avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation des collections: {e}")
    
    def get_db(self):
        """Obtenir l'instance de la base de données"""
        if self._db is None:
            self.connect()
        return self._db
    
    def get_collection(self, collection_name):
        """Obtenir une collection spécifique"""
        db = self.get_db()
        return db[collection_name] if db else None
    
    def close(self):
        """Fermer la connexion MongoDB"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            print("✅ Connexion MongoDB fermée")
    
    def test_connection(self):
        """Tester la connexion MongoDB"""
        try:
            self._client.admin.command('ping')
            return True
        except Exception as e:
            print(f"❌ Test de connexion échoué: {e}")
            return False


# Instance globale du connecteur
mongodb_connector = MongoDBConnector()

def get_mongodb():
    """Fonction helper pour obtenir la base de données"""
    return mongodb_connector.get_db()

def get_collection(collection_name):
    """Fonction helper pour obtenir une collection"""
    return mongodb_connector.get_collection(collection_name)

def init_mongodb_collections():
    """Fonction helper pour initialiser les collections MongoDB"""
    return mongodb_connector.init_collections()
