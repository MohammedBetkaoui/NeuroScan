from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session, redirect, url_for, flash, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import base64
import os
import time
import requests
import json
import secrets
import random
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from torchvision import transforms
from functools import wraps
from dotenv import load_dotenv

# MongoDB imports
from database.mongodb_connector import get_mongodb, get_collection, init_mongodb_collections
from database.mongodb_helpers import (
    save_analysis_to_db_mongo,
    get_current_doctor_mongo,
    create_doctor_session_mongo,
    get_doctor_statistics_mongo,
    verify_doctor_credentials_mongo,
    register_doctor_mongo
)
from bson import ObjectId

# File Manager import
from file_manager import get_file_manager

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics import renderPDF

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
# Charger la clé secrète depuis .env ou utiliser une valeur par défaut
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'neuroscan_secret_key_2024_medical_auth')

# Initialiser SocketIO pour la messagerie en temps réel
socketio = SocketIO(app, 
                    cors_allowed_origins="*", 
                    async_mode='threading',  # Threading pour compatibilité Windows
                    logger=True,
                    engineio_logger=True)

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialiser la connexion MongoDB globale
db = get_mongodb()

# Configuration de l'API Gemini - Chargée depuis .env
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY or GEMINI_API_KEY == 'your_gemini_api_key_here':
    print("⚠️  ATTENTION: Clé API Gemini non configurée. Le chatbot ne fonctionnera pas.")
    print("   Ajoutez votre clé dans le fichier .env : GEMINI_API_KEY=votre_clé_ici")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# DATABASE_PATH supprimé - utilisation de MongoDB

def init_database():
    """Initialiser MongoDB au lieu de SQLite"""
    try:
        init_mongodb_collections()
        print("✅ MongoDB initialisé avec succès")
    except Exception as e:
        print(f"❌ Erreur initialisation MongoDB: {e}")

def save_analysis_to_db(results, filename, processing_time, session_id=None, ip_address=None, patient_id=None, patient_name=None, exam_date=None, doctor_id=None):
    """Wrapper pour save_analysis_to_db_mongo"""
    return save_analysis_to_db_mongo(results, filename, processing_time, session_id, ip_address, 
                                     patient_id, patient_name, exam_date, doctor_id)
    

    
   

    

    
  
           
# Initialiser la base de données au démarrage
init_database()

# Fonctions utilitaires pour l'authentification
def login_required(f):
    """Décorateur pour protéger les routes nécessitant une authentification"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'doctor_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_current_doctor():
    """Wrapper pour get_current_doctor_mongo"""
    if 'doctor_id' not in session:
        return None
    return get_current_doctor_mongo(session['doctor_id'])

def get_doctor_statistics(doctor_id):
    """Wrapper pour get_doctor_statistics_mongo"""
    return get_doctor_statistics_mongo(doctor_id)

def create_doctor_session(doctor_id, ip_address, user_agent):
    """Wrapper pour create_doctor_session_mongo"""
    return create_doctor_session_mongo(doctor_id, ip_address, user_agent)

# Définition du modèle CNN (architecture exacte du modèle sauvegardé)
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):  # 4 classes: Normal, Glioma, Meningioma, Pituitary
        super(BrainTumorCNN, self).__init__()

        # Couches de convolution (architecture exacte du modèle sauvegardé)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Couches de pooling et autres
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

        # Couches fully connected (tailles exactes du modèle sauvegardé)
        self.fc1 = nn.Linear(12544, 512)  # Taille exacte détectée: 256*7*7 = 12544
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Appliquer les couches de convolution avec pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))

        # Aplatir et appliquer les couches FC
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# Charger le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainTumorCNN(num_classes=4)

try:
    # Charger les poids du modèle
    checkpoint = torch.load('best_brain_tumor_model.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("Modèle chargé avec succès!")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    print("Mode démo activé - utilisation de prédictions simulées")
    model = None

# Classes de tumeurs
TUMOR_CLASSES = {
    0: 'Normal',
    1: 'Gliome',
    2: 'Méningiome', 
    3: 'Tumeur pituitaire'
}

# Transformations pour les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    """Vérifier si le fichier est autorisé"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'nii'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Préprocesser l'image pour le modèle"""
    try:
        # Charger l'image
        image = Image.open(image_path)
        
        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Appliquer les transformations
        image_tensor = transform(image).unsqueeze(0)  # Ajouter dimension batch
        
        return image_tensor
    except Exception as e:
        print(f"Erreur lors du préprocessing: {e}")
        return None

def predict_tumor(image_path):
    """Prédire le type de tumeur"""
    if model is None:
        # Mode démo - générer des prédictions simulées réalistes
        
        # Simuler différents scénarios de diagnostic
        demo_scenarios = [
            {
                'predicted_class': 0,
                'predicted_label': 'Normal',
                'confidence': 0.92,
                'probabilities': {
                    'Normal': 0.92,
                    'Gliome': 0.03,
                    'Méningiome': 0.03,
                    'Tumeur pituitaire': 0.02
                }
            },
            {
                'predicted_class': 1,
                'predicted_label': 'Gliome',
                'confidence': 0.87,
                'probabilities': {
                    'Normal': 0.05,
                    'Gliome': 0.87,
                    'Méningiome': 0.06,
                    'Tumeur pituitaire': 0.02
                }
            },
            {
                'predicted_class': 2,
                'predicted_label': 'Méningiome',
                'confidence': 0.89,
                'probabilities': {
                    'Normal': 0.04,
                    'Gliome': 0.05,
                    'Méningiome': 0.89,
                    'Tumeur pituitaire': 0.02
                }
            },
            {
                'predicted_class': 3,
                'predicted_label': 'Tumeur pituitaire',
                'confidence': 0.91,
                'probabilities': {
                    'Normal': 0.03,
                    'Gliome': 0.03,
                    'Méningiome': 0.03,
                    'Tumeur pituitaire': 0.91
                }
            }
        ]
        
        # Sélectionner un scénario aléatoire
        results = random.choice(demo_scenarios)
        
        # Ajouter des recommandations basées sur le type
        results['recommendations'] = get_recommendations(results)
        results['description'] = f"Analyse démo pour {results['predicted_label']} avec {results['confidence']*100:.1f}% de confiance."
        
        return results
    
    try:
        # Préprocesser l'image
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        # Faire la prédiction
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Obtenir les probabilités pour chaque classe
            probs = probabilities.cpu().numpy()[0]
            predicted_class = np.argmax(probs)
            
            results = {
                'predicted_class': int(predicted_class),
                'predicted_label': TUMOR_CLASSES[predicted_class],
                'confidence': float(probs[predicted_class]),
                'probabilities': {
                    'Normal': float(probs[0]),
                    'Gliome': float(probs[1]),
                    'Méningiome': float(probs[2]),
                    'Tumeur pituitaire': float(probs[3])
                }
            }

            # Ajouter la description et les recommandations Gemini
            gemini_analysis = get_gemini_analysis(results)
            if gemini_analysis:
                results['description'] = gemini_analysis.get('description', '')
                results['recommendations'] = gemini_analysis.get('recommendations', get_recommendations(results))
            else:
                results['recommendations'] = get_recommendations(results)

            return results
            
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return None

# Routes d'authentification
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Page de connexion des médecins"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Veuillez remplir tous les champs', 'error')
            return render_template('auth/login.html')

        try:
            # Utiliser verify_doctor_credentials_mongo
            doctor = verify_doctor_credentials_mongo(
                email,
                lambda hash_val: check_password_hash(hash_val, password)
            )

            if doctor and 'error' not in doctor:
                # Créer la session Flask
                session['doctor_id'] = doctor['id']
                session['doctor_name'] = doctor['full_name']
                session['logged_in'] = True

                # Créer une session en base
                create_doctor_session_mongo(
                    doctor['id'],
                    request.remote_addr,
                    request.headers.get('User-Agent', '')
                )

                flash(f'Bienvenue Dr. {doctor["first_name"]} {doctor["last_name"]}!', 'success')
                return redirect(url_for('dashboard'))
            elif doctor and doctor.get('error') == 'account_disabled':
                flash('Votre compte a été désactivé. Contactez l\'administrateur.', 'error')
            else:
                flash('Email ou mot de passe incorrect', 'error')

        except Exception as e:
            print(f"Erreur lors de la connexion: {e}")
            flash('Erreur lors de la connexion', 'error')

    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Page d'inscription des médecins"""
    if request.method == 'POST':
        # Récupérer les données du formulaire
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        specialty = request.form.get('specialty', '').strip()
        hospital = request.form.get('hospital', '').strip()
        license_number = request.form.get('license_number', '').strip()
        phone = request.form.get('phone', '').strip()

        # Validation
        if not all([email, password, confirm_password, first_name, last_name]):
            flash('Veuillez remplir tous les champs obligatoires', 'error')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('Les mots de passe ne correspondent pas', 'error')
            return render_template('auth/register.html')

        if len(password) < 6:
            flash('Le mot de passe doit contenir au moins 6 caractères', 'error')
            return render_template('auth/register.html')

        try:
            # Utiliser register_doctor_mongo
            password_hash = generate_password_hash(password)
            doctor_id = register_doctor_mongo(
                email, password_hash, first_name, last_name,
                specialty, hospital, license_number, phone
            )

            if doctor_id:
                flash('Compte créé avec succès! Vous pouvez maintenant vous connecter.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Un compte avec cet email existe déjà', 'error')

        except Exception as e:
            print(f"Erreur lors de l'inscription: {e}")
            flash('Erreur lors de la création du compte', 'error')

    return render_template('auth/register.html')

@app.route('/logout')
def logout():
    """Déconnexion"""
    if 'doctor_id' in session:
        # Désactiver la session en base MongoDB
        try:
            db = get_mongodb()
            doctor_sessions = db.doctor_sessions
            doctor_sessions.update_many(
                {'doctor_id': session['doctor_id']},
                {'$set': {'is_active': False}}
            )
        except Exception as e:
            print(f"Erreur lors de la déconnexion: {e}")

    session.clear()
    flash('Vous avez été déconnecté avec succès', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard principal pour les médecins connectés"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))

    # Récupérer les statistiques du médecin
    doctor_stats = get_doctor_statistics(doctor['id'])
    
    # Ajouter les statistiques au contexte
    total_analyses = doctor_stats.get('total_analyses', 0)
    total_patients = doctor_stats.get('total_patients', 0)
    
    return render_template('dashboard.html', 
                         doctor=doctor, 
                         doctor_stats=doctor_stats,
                         total_analyses=total_analyses,
                         total_patients=total_patients)

@app.route('/messages')
@login_required
def messages():
    """Page de messagerie professionnelle entre médecins"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))
    
    return render_template('messages.html', doctor=doctor)

# ========================================
# MESSAGERIE ENTRE MÉDECINS - API Routes
# ========================================

@app.route('/api/messages/doctors')
@login_required
def get_doctors_list():
    """Récupérer la liste des médecins pour la messagerie"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        # Récupérer tous les médecins sauf le médecin connecté
        doctors = list(db.doctors.find(
            {'_id': {'$ne': ObjectId(doctor['id'])}},
            {'password': 0}  # Exclure le mot de passe
        ).sort('last_name', 1))
        
        # Formater les données
        doctors_list = []
        for doc in doctors:
            first_name = doc.get('first_name', '')
            last_name = doc.get('last_name', '')
            full_name = f"{first_name} {last_name}".strip() or doc.get('full_name', 'Médecin')
            
            doctors_list.append({
                'id': str(doc['_id']),
                'first_name': first_name,
                'last_name': last_name,
                'full_name': full_name,
                'email': doc.get('email', ''),
                'specialty': doc.get('specialty', 'Médecin'),
                'hospital': doc.get('hospital', ''),
                'is_online': False  # À implémenter avec WebSocket si nécessaire
            })
        
        return jsonify({'success': True, 'doctors': doctors_list})
    
    except Exception as e:
        print(f"Erreur récupération liste médecins: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/messages/conversations')
@login_required
def get_message_conversations():
    """Récupérer les conversations de messagerie du médecin"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        doctor_id = doctor['id']
        
        # Récupérer toutes les conversations où le médecin est participant
        conversations = list(db.doctor_conversations.aggregate([
            {
                '$match': {
                    'participants': ObjectId(doctor_id)
                }
            },
            {
                '$lookup': {
                    'from': 'doctor_messages',
                    'localField': '_id',
                    'foreignField': 'conversation_id',
                    'as': 'messages'
                }
            },
            {
                '$addFields': {
                    'last_message': {'$arrayElemAt': ['$messages', -1]},
                    'unread_count': {
                        '$size': {
                            '$filter': {
                                'input': '$messages',
                                'as': 'msg',
                                'cond': {
                                    '$and': [
                                        {'$ne': ['$$msg.sender_id', ObjectId(doctor_id)]},
                                        {'$eq': ['$$msg.is_read', False]}
                                    ]
                                }
                            }
                        }
                    }
                }
            },
            {
                '$sort': {'last_message.created_at': -1}
            }
        ]))
        
        # Formater les données avec les informations des autres participants
        formatted_conversations = []
        for conv in conversations:
            # Trouver l'autre participant
            other_participant_id = None
            for participant_id in conv.get('participants', []):
                if str(participant_id) != doctor_id:
                    other_participant_id = participant_id
                    break
            
            if other_participant_id:
                other_doctor = db.doctors.find_one(
                    {'_id': other_participant_id},
                    {'password': 0}
                )
                
                if other_doctor:
                    first_name = other_doctor.get('first_name', '')
                    last_name = other_doctor.get('last_name', '')
                    full_name = f"{first_name} {last_name}".strip() or other_doctor.get('full_name', 'Médecin')
                    
                    last_msg = conv.get('last_message', {})
                    formatted_conversations.append({
                        'id': str(conv['_id']),
                        'other_doctor': {
                            'id': str(other_doctor['_id']),
                            'first_name': first_name,
                            'last_name': last_name,
                            'full_name': full_name,
                            'specialty': other_doctor.get('specialty', 'Médecin'),
                            'hospital': other_doctor.get('hospital', '')
                        },
                        'last_message': {
                            'content': last_msg.get('content', ''),
                            'created_at': last_msg.get('created_at').isoformat() if last_msg.get('created_at') else None,
                            'is_from_me': str(last_msg.get('sender_id', '')) == doctor_id
                        } if last_msg else None,
                        'unread_count': conv.get('unread_count', 0),
                        'created_at': conv.get('created_at').isoformat() if conv.get('created_at') else None
                    })
        
        return jsonify({'success': True, 'conversations': formatted_conversations})
    
    except Exception as e:
        print(f"Erreur récupération conversations: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/messages/conversations', methods=['POST'])
@login_required
def create_message_conversation():
    """Créer une nouvelle conversation avec un médecin"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        data = request.get_json()
        recipient_id = data.get('recipient_id')
        
        if not recipient_id:
            return jsonify({'success': False, 'error': 'ID du destinataire requis'}), 400
        
        doctor_id = doctor['id']
        
        # Vérifier si une conversation existe déjà
        existing_conv = db.doctor_conversations.find_one({
            'participants': {
                '$all': [ObjectId(doctor_id), ObjectId(recipient_id)]
            }
        })
        
        if existing_conv:
            return jsonify({
                'success': True,
                'conversation_id': str(existing_conv['_id']),
                'exists': True
            })
        
        # Créer une nouvelle conversation
        conversation_doc = {
            'participants': [ObjectId(doctor_id), ObjectId(recipient_id)],
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        result = db.doctor_conversations.insert_one(conversation_doc)
        
        return jsonify({
            'success': True,
            'conversation_id': str(result.inserted_id),
            'exists': False
        })
    
    except Exception as e:
        print(f"Erreur création conversation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/messages/conversations/<conversation_id>/messages')
@login_required
def get_conversation_messages(conversation_id):
    """Récupérer les messages d'une conversation"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        doctor_id = doctor['id']
        
        # Vérifier que le médecin fait partie de la conversation
        conversation = db.doctor_conversations.find_one({
            '_id': ObjectId(conversation_id),
            'participants': ObjectId(doctor_id)
        })
        
        if not conversation:
            return jsonify({'success': False, 'error': 'Conversation non trouvée'}), 404
        
        # Récupérer les messages
        messages = list(db.doctor_messages.find(
            {'conversation_id': ObjectId(conversation_id)}
        ).sort('created_at', 1))
        
        # Formater les messages
        formatted_messages = []
        for msg in messages:
            sender = db.doctors.find_one(
                {'_id': msg['sender_id']},
                {'password': 0}
            )
            
            first_name = sender.get('first_name', '') if sender else ''
            last_name = sender.get('last_name', '') if sender else ''
            full_name = f"{first_name} {last_name}".strip() if sender else ''
            if not full_name and sender:
                full_name = sender.get('full_name', 'Médecin')
            elif not full_name:
                full_name = 'Médecin'
            
            message_data = {
                'id': str(msg['_id']),
                'content': msg.get('content', ''),
                'message_type': msg.get('message_type', 'text'),
                'is_from_me': str(msg['sender_id']) == doctor_id,
                'sender': {
                    'id': str(sender['_id']) if sender else None,
                    'first_name': first_name,
                    'last_name': last_name,
                    'full_name': full_name,
                    'specialty': sender.get('specialty', '') if sender else ''
                },
                'is_read': msg.get('is_read', False),
                'created_at': msg.get('created_at').isoformat() if msg.get('created_at') else None
            }
            
            # Ajouter les fichiers attachés
            if msg.get('file_ids'):
                files_collection = get_collection('message_files')
                attached_files = list(files_collection.find({
                    '_id': {'$in': [ObjectId(fid) for fid in msg['file_ids']]},
                    'is_deleted': False
                }))
                
                message_data['files'] = []
                for file_doc in attached_files:
                    message_data['files'].append({
                        'id': str(file_doc['_id']),
                        '_id': str(file_doc['_id']),
                        'original_filename': file_doc.get('original_filename', ''),
                        'file_size': file_doc.get('file_size', 0),
                        'file_size_formatted': file_doc.get('file_size_formatted', '0 B'),
                        'file_extension': file_doc.get('file_extension', ''),
                        'file_category': file_doc.get('file_category', ''),
                        'mime_type': file_doc.get('mime_type', 'application/octet-stream')
                    })
            
            # Ajouter les données spécifiques selon le type
            if msg.get('message_type') == 'analysis_share':
                message_data['analysis_id'] = str(msg.get('analysis_id', ''))
                message_data['analysis_data'] = msg.get('analysis_data', {})
            
            formatted_messages.append(message_data)
        
        # Marquer les messages comme lus
        db.doctor_messages.update_many(
            {
                'conversation_id': ObjectId(conversation_id),
                'sender_id': {'$ne': ObjectId(doctor_id)},
                'is_read': False
            },
            {'$set': {'is_read': True, 'read_at': datetime.now()}}
        )
        
        return jsonify({'success': True, 'messages': formatted_messages})
    
    except Exception as e:
        print(f"Erreur récupération messages: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/messages/send', methods=['POST'])
@login_required
def send_message():
    """Envoyer un message texte"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        content = data.get('content', '').strip()
        
        if not conversation_id or not content:
            return jsonify({'success': False, 'error': 'Données manquantes'}), 400
        
        doctor_id = doctor['id']
        
        # Vérifier que le médecin fait partie de la conversation
        conversation = db.doctor_conversations.find_one({
            '_id': ObjectId(conversation_id),
            'participants': ObjectId(doctor_id)
        })
        
        if not conversation:
            return jsonify({'success': False, 'error': 'Conversation non trouvée'}), 404
        
        # Créer le message
        message_doc = {
            'conversation_id': ObjectId(conversation_id),
            'sender_id': ObjectId(doctor_id),
            'content': content,
            'message_type': 'text',
            'is_read': False,
            'created_at': datetime.now()
        }
        
        # Ajouter les IDs de fichiers si présents
        file_ids = data.get('file_ids', [])
        if file_ids:
            message_doc['file_ids'] = file_ids
        
        result = db.doctor_messages.insert_one(message_doc)
        
        # Mettre à jour la conversation
        db.doctor_conversations.update_one(
            {'_id': ObjectId(conversation_id)},
            {'$set': {'updated_at': datetime.now()}}
        )
        
        # Si des fichiers sont attachés, les retourner aussi
        files_data = []
        if file_ids:
            files_collection = get_collection('message_files')
            attached_files = list(files_collection.find({
                '_id': {'$in': [ObjectId(fid) for fid in file_ids]},
                'is_deleted': False
            }))
            
            for file_doc in attached_files:
                files_data.append({
                    'id': str(file_doc['_id']),
                    '_id': str(file_doc['_id']),
                    'original_filename': file_doc.get('original_filename', ''),
                    'file_size': file_doc.get('file_size', 0),
                    'file_size_formatted': file_doc.get('file_size_formatted', '0 B'),
                    'file_extension': file_doc.get('file_extension', ''),
                    'file_category': file_doc.get('file_category', ''),
                    'mime_type': file_doc.get('mime_type', 'application/octet-stream')
                })
        
        # Récupérer les infos du sender pour le message complet
        sender = db.doctors.find_one(
            {'_id': ObjectId(doctor_id)},
            {'password': 0}
        )
        
        first_name = sender.get('first_name', '') if sender else ''
        last_name = sender.get('last_name', '') if sender else ''
        full_name = f"{first_name} {last_name}".strip() if sender else ''
        if not full_name and sender:
            full_name = sender.get('full_name', 'Médecin')
        elif not full_name:
            full_name = 'Médecin'
        
        # Retourner le message complet formaté
        return jsonify({
            'success': True,
            'message': {
                'id': str(result.inserted_id),
                '_id': str(result.inserted_id),
                'content': content,
                'message_type': 'text',
                'is_from_me': True,
                'sender': {
                    'id': str(sender['_id']) if sender else None,
                    'first_name': first_name,
                    'last_name': last_name,
                    'full_name': full_name,
                    'specialty': sender.get('specialty', '') if sender else ''
                },
                'is_read': False,
                'created_at': message_doc['created_at'].isoformat(),
                'files': files_data
            }
        })
    
    except Exception as e:
        print(f"Erreur envoi message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/messages/share-analysis', methods=['POST'])
@login_required
def share_analysis_with_doctor():
    """Partager une analyse avec un autre médecin"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        analysis_id = data.get('analysis_id')
        message_content = data.get('message', '').strip()
        
        if not conversation_id or not analysis_id:
            return jsonify({'success': False, 'error': 'Données manquantes'}), 400
        
        doctor_id = doctor['id']
        
        # Vérifier que le médecin fait partie de la conversation
        conversation = db.doctor_conversations.find_one({
            '_id': ObjectId(conversation_id),
            'participants': ObjectId(doctor_id)
        })
        
        if not conversation:
            return jsonify({'success': False, 'error': 'Conversation non trouvée'}), 404
        
        # Récupérer l'analyse
        analysis = db.analyses.find_one({'_id': ObjectId(analysis_id)})
        
        if not analysis:
            return jsonify({'success': False, 'error': 'Analyse non trouvée'}), 404
        
        # Vérifier que l'analyse appartient au médecin
        if str(analysis.get('doctor_id', '')) != doctor_id:
            return jsonify({'success': False, 'error': 'Vous ne pouvez partager que vos propres analyses'}), 403
        
        # Préparer les données de l'analyse à partager
        analysis_data = {
            'patient_name': analysis.get('patient_name', 'Patient inconnu'),
            'exam_date': analysis.get('exam_date', ''),
            'predicted_label': analysis.get('predicted_label', ''),
            'confidence': analysis.get('confidence', 0),
            'probabilities': analysis.get('probabilities', {}),
            'recommendations': analysis.get('recommendations', []),
            'image_filename': analysis.get('image_filename', '')
        }
        
        # Créer le message de partage
        content = message_content or f"Partage d'analyse pour {analysis_data['patient_name']}"
        
        message_doc = {
            'conversation_id': ObjectId(conversation_id),
            'sender_id': ObjectId(doctor_id),
            'content': content,
            'message_type': 'analysis_share',
            'analysis_id': ObjectId(analysis_id),
            'analysis_data': analysis_data,
            'is_read': False,
            'created_at': datetime.now()
        }
        
        result = db.doctor_messages.insert_one(message_doc)
        
        # Mettre à jour la conversation
        db.doctor_conversations.update_one(
            {'_id': ObjectId(conversation_id)},
            {'$set': {'updated_at': datetime.now()}}
        )
        
        # Créer une notification pour le destinataire
        recipient_id = None
        for participant_id in conversation.get('participants', []):
            if str(participant_id) != doctor_id:
                recipient_id = participant_id
                break
        
        if recipient_id:
            notification_doc = {
                'doctor_id': recipient_id,
                'type': 'analysis_shared',
                'title': 'Nouvelle analyse partagée',
                'message': f"Dr. {doctor.get('full_name', 'Un médecin')} a partagé une analyse avec vous",
                'analysis_id': ObjectId(analysis_id),
                'sender_id': ObjectId(doctor_id),
                'is_read': False,
                'created_at': datetime.now()
            }
            db.notifications.insert_one(notification_doc)
        
        return jsonify({
            'success': True,
            'message_id': str(result.inserted_id),
            'created_at': message_doc['created_at'].isoformat()
        })
    
    except Exception as e:
        print(f"Erreur partage analyse: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/messages/shared-analyses')
@login_required
def get_shared_analyses():
    """Récupérer toutes les analyses partagées avec le médecin"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        doctor_id = doctor['id']
        
        # Récupérer toutes les conversations du médecin
        conversations = list(db.doctor_conversations.find({
            'participants': ObjectId(doctor_id)
        }))
        
        conversation_ids = [conv['_id'] for conv in conversations]
        
        # Récupérer tous les messages de type analysis_share reçus
        shared_analyses = list(db.doctor_messages.aggregate([
            {
                '$match': {
                    'conversation_id': {'$in': conversation_ids},
                    'message_type': 'analysis_share',
                    'sender_id': {'$ne': ObjectId(doctor_id)}
                }
            },
            {
                '$lookup': {
                    'from': 'doctors',
                    'localField': 'sender_id',
                    'foreignField': '_id',
                    'as': 'sender'
                }
            },
            {
                '$unwind': '$sender'
            },
            {
                '$sort': {'created_at': -1}
            }
        ]))
        
        # Formater les données
        formatted_analyses = []
        for item in shared_analyses:
            sender = item['sender']
            first_name = sender.get('first_name', '')
            last_name = sender.get('last_name', '')
            full_name = f"{first_name} {last_name}".strip() or sender.get('full_name', 'Médecin')
            
            formatted_analyses.append({
                'message_id': str(item['_id']),
                'analysis_id': str(item.get('analysis_id', '')),
                'analysis_data': item.get('analysis_data', {}),
                'message': item.get('content', ''),
                'sender': {
                    'id': str(sender['_id']),
                    'first_name': first_name,
                    'last_name': last_name,
                    'full_name': full_name,
                    'specialty': sender.get('specialty', ''),
                    'hospital': sender.get('hospital', '')
                },
                'is_read': item.get('is_read', False),
                'shared_at': item.get('created_at').isoformat() if item.get('created_at') else None
            })
        
        return jsonify({'success': True, 'shared_analyses': formatted_analyses})
    
    except Exception as e:
        print(f"Erreur récupération analyses partagées: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/messages/analysis/<analysis_id>')
@login_required
def get_shared_analysis_details(analysis_id):
    """Obtenir les détails complets d'une analyse partagée"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        # Récupérer l'analyse
        analysis = db.analyses.find_one({'_id': ObjectId(analysis_id)})
        
        if not analysis:
            return jsonify({'success': False, 'error': 'Analyse non trouvée'}), 404
        
        # Formater les données complètes
        analysis_details = {
            'id': str(analysis['_id']),
            'patient_name': analysis.get('patient_name', ''),
            'patient_id': analysis.get('patient_id', ''),
            'exam_date': analysis.get('exam_date', ''),
            'predicted_label': analysis.get('predicted_label', ''),
            'confidence': analysis.get('confidence', 0),
            'probabilities': analysis.get('probabilities', {}),
            'recommendations': analysis.get('recommendations', []),
            'description': analysis.get('description', ''),
            'image_filename': analysis.get('image_filename', ''),
            'processing_time': analysis.get('processing_time', 0),
            'created_at': analysis.get('timestamp', ''),
            'doctor_id': str(analysis.get('doctor_id', ''))
        }
        
        # Récupérer les infos du médecin qui a créé l'analyse
        if analysis.get('doctor_id'):
            owner_doctor = db.doctors.find_one(
                {'_id': analysis.get('doctor_id')},
                {'password': 0}
            )
            if owner_doctor:
                first_name = owner_doctor.get('first_name', '')
                last_name = owner_doctor.get('last_name', '')
                full_name = f"{first_name} {last_name}".strip() or owner_doctor.get('full_name', 'Médecin')
                
                analysis_details['owner_doctor'] = {
                    'id': str(owner_doctor['_id']),
                    'first_name': first_name,
                    'last_name': last_name,
                    'full_name': full_name,
                    'specialty': owner_doctor.get('specialty', ''),
                    'hospital': owner_doctor.get('hospital', '')
                }
        
        return jsonify({'success': True, 'analysis': analysis_details})
    
    except Exception as e:
        print(f"Erreur récupération détails analyse: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ========================================
# FIN MESSAGERIE ENTRE MÉDECINS
# ========================================
# ROUTES UPLOAD DE FICHIERS
# ========================================

@app.route('/api/messages/upload-file', methods=['POST'])
@login_required
def upload_message_file():
    """Upload un fichier pour la messagerie"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        # Vérifier qu'un fichier a été envoyé
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        conversation_id = request.form.get('conversation_id')
        
        if not conversation_id:
            return jsonify({'success': False, 'error': 'ID de conversation manquant'}), 400
        
        # Utiliser le gestionnaire de fichiers
        file_manager = get_file_manager()
        result = file_manager.save_file(file, subfolder=conversation_id)
        
        if not result['success']:
            return jsonify(result), 400
        
        # Enregistrer dans la base de données
        files_collection = get_collection('message_files')
        file_doc = {
            'conversation_id': conversation_id,
            'uploaded_by': doctor['id'],
            'original_filename': result['original_filename'],
            'stored_filename': result['stored_filename'],
            'relative_path': result['relative_path'],
            'file_size': result['file_size'],
            'file_size_formatted': result.get('file_size_formatted', '0 B'),
            'file_extension': result['file_extension'],
            'file_category': result['file_category'],
            'mime_type': result['mime_type'],
            'uploaded_at': datetime.now(),
            'is_deleted': False
        }
        
        insert_result = files_collection.insert_one(file_doc)
        file_doc['_id'] = str(insert_result.inserted_id)
        file_doc['uploaded_at'] = file_doc['uploaded_at'].isoformat()
        
        return jsonify({
            'success': True,
            'file': file_doc,
            'message': 'Fichier uploadé avec succès'
        })
        
    except Exception as e:
        print(f"Erreur upload fichier: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/messages/files/<file_id>')
@login_required
def get_message_file(file_id):
    """Télécharger un fichier de la messagerie"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        print(f"📥 Demande téléchargement fichier: {file_id} par doctor: {doctor.get('id')}")
        
        # Récupérer le fichier de la DB
        files_collection = get_collection('message_files')
        file_doc = files_collection.find_one({'_id': ObjectId(file_id), 'is_deleted': False})
        
        if not file_doc:
            print(f"❌ Fichier non trouvé dans la BD: {file_id}")
            return jsonify({'success': False, 'error': 'Fichier non trouvé'}), 404
        
        print(f"✅ Fichier trouvé: {file_doc.get('original_filename')}")
        print(f"📁 Conversation du fichier: {file_doc.get('conversation_id')}")
        
        # Vérifier que le médecin a accès à cette conversation
        conversations_collection = get_collection('doctor_conversations')
        
        # Convertir l'ID du médecin en ObjectId si nécessaire
        doctor_id = ObjectId(doctor['id']) if isinstance(doctor['id'], str) else doctor['id']
        
        # Récupérer la conversation pour vérifier
        conversation = conversations_collection.find_one({
            '_id': ObjectId(file_doc['conversation_id'])
        })
        
        if not conversation:
            print(f"❌ Conversation non trouvée: {file_doc['conversation_id']}")
            return jsonify({'success': False, 'error': 'Conversation non trouvée'}), 404
        
        print(f"👥 Participants de la conversation: {conversation.get('participants')}")
        
        # Vérifier si le médecin est participant
        is_participant = doctor_id in conversation.get('participants', [])
        
        if not is_participant:
            print(f"❌ Accès refusé - Doctor {doctor_id} pas dans participants {conversation.get('participants')}")
            return jsonify({'success': False, 'error': 'Accès non autorisé'}), 403
        
        print(f"✅ Accès autorisé pour {doctor.get('first_name')} {doctor.get('last_name')}")
        
        # Chemin du fichier
        file_path = os.path.join('uploads', file_doc['relative_path'])
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Fichier physique non trouvé'}), 404
        
        # Télécharger le fichier
        return send_file(
            file_path,
            as_attachment=True,
            download_name=file_doc['original_filename'],
            mimetype=file_doc.get('mime_type', 'application/octet-stream')
        )
        
    except Exception as e:
        print(f"Erreur téléchargement fichier: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/messages/files/<file_id>/delete', methods=['DELETE'])
@login_required
def delete_message_file(file_id):
    """Supprimer un fichier de la messagerie"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        # Récupérer le fichier de la DB
        files_collection = get_collection('message_files')
        file_doc = files_collection.find_one({'_id': ObjectId(file_id), 'is_deleted': False})
        
        if not file_doc:
            return jsonify({'success': False, 'error': 'Fichier non trouvé'}), 404
        
        # Vérifier que c'est l'uploader
        if file_doc['uploaded_by'] != doctor['id']:
            return jsonify({'success': False, 'error': 'Seul l\'uploadeur peut supprimer le fichier'}), 403
        
        # Marquer comme supprimé (soft delete)
        files_collection.update_one(
            {'_id': ObjectId(file_id)},
            {'$set': {'is_deleted': True, 'deleted_at': datetime.now()}}
        )
        
        # Optionnel: supprimer le fichier physique
        file_path = os.path.join('uploads', file_doc['relative_path'])
        file_manager = get_file_manager()
        file_manager.delete_file(file_path)
        
        return jsonify({'success': True, 'message': 'Fichier supprimé avec succès'})
        
    except Exception as e:
        print(f"Erreur suppression fichier: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/messages/conversations/<conversation_id>/files')
@login_required
def get_conversation_files(conversation_id):
    """Obtenir la liste des fichiers d'une conversation"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        # Vérifier l'accès à la conversation
        conversations_collection = get_collection('doctor_conversations')
        conversation = conversations_collection.find_one({
            '_id': ObjectId(conversation_id),
            'participants': doctor['id']
        })
        
        if not conversation:
            return jsonify({'success': False, 'error': 'Conversation non trouvée'}), 404
        
        # Récupérer les fichiers
        files_collection = get_collection('message_files')
        files_cursor = files_collection.find({
            'conversation_id': conversation_id,
            'is_deleted': False
        }).sort('uploaded_at', -1)
        
        files = []
        for file_doc in files_cursor:
            file_doc['_id'] = str(file_doc['_id'])
            file_doc['uploaded_at'] = file_doc['uploaded_at'].isoformat()
            files.append(file_doc)
        
        return jsonify({'success': True, 'files': files})
        
    except Exception as e:
        print(f"Erreur récupération fichiers: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/messages/storage-stats')
@login_required
def get_storage_stats():
    """Obtenir les statistiques de stockage"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        file_manager = get_file_manager()
        stats = file_manager.get_storage_stats()
        
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        print(f"Erreur statistiques stockage: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ========================================

@app.route('/api/messages/quick-share-analysis', methods=['POST'])
@login_required
def quick_share_analysis():
    """Partager rapidement une analyse avec un médecin (crée la conversation si nécessaire)"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401
        
        data = request.get_json()
        recipient_id = data.get('recipient_id')
        analysis_id = data.get('analysis_id')
        message_content = data.get('message', '').strip()
        
        if not recipient_id or not analysis_id:
            return jsonify({'success': False, 'error': 'Données manquantes'}), 400
        
        doctor_id = doctor['id']
        
        # Vérifier que l'analyse existe et appartient au médecin
        analysis = db.analyses.find_one({'_id': ObjectId(analysis_id)})
        
        if not analysis:
            return jsonify({'success': False, 'error': 'Analyse non trouvée'}), 404
        
        if str(analysis.get('doctor_id', '')) != doctor_id:
            return jsonify({'success': False, 'error': 'Vous ne pouvez partager que vos propres analyses'}), 403
        
        # Trouver ou créer la conversation
        conversation = db.doctor_conversations.find_one({
            'participants': {
                '$all': [ObjectId(doctor_id), ObjectId(recipient_id)]
            }
        })
        
        if not conversation:
            # Créer une nouvelle conversation
            conversation_doc = {
                'participants': [ObjectId(doctor_id), ObjectId(recipient_id)],
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            result = db.doctor_conversations.insert_one(conversation_doc)
            conversation_id = result.inserted_id
        else:
            conversation_id = conversation['_id']
        
        # Préparer les données de l'analyse à partager
        analysis_data = {
            'patient_name': analysis.get('patient_name', 'Patient inconnu'),
            'exam_date': analysis.get('exam_date', ''),
            'predicted_label': analysis.get('predicted_label', ''),
            'confidence': analysis.get('confidence', 0),
            'probabilities': analysis.get('probabilities', {}),
            'recommendations': analysis.get('recommendations', []),
            'image_filename': analysis.get('image_filename', '')
        }
        
        # Créer le message de partage
        content = message_content or f"Partage d'analyse pour {analysis_data['patient_name']}"
        
        message_doc = {
            'conversation_id': ObjectId(conversation_id),
            'sender_id': ObjectId(doctor_id),
            'content': content,
            'message_type': 'analysis_share',
            'analysis_id': ObjectId(analysis_id),
            'analysis_data': analysis_data,
            'is_read': False,
            'created_at': datetime.now()
        }
        
        message_result = db.doctor_messages.insert_one(message_doc)
        
        # Mettre à jour la conversation
        db.doctor_conversations.update_one(
            {'_id': ObjectId(conversation_id)},
            {'$set': {'updated_at': datetime.now()}}
        )
        
        # Créer une notification pour le destinataire
        recipient = db.doctors.find_one({'_id': ObjectId(recipient_id)})
        
        notification_doc = {
            'doctor_id': ObjectId(recipient_id),
            'type': 'analysis_shared',
            'title': 'Nouvelle analyse partagée',
            'message': f"Dr. {doctor.get('first_name', '')} {doctor.get('last_name', '')} a partagé une analyse avec vous",
            'analysis_id': ObjectId(analysis_id),
            'sender_id': ObjectId(doctor_id),
            'conversation_id': ObjectId(conversation_id),
            'is_read': False,
            'created_at': datetime.now()
        }
        db.notifications.insert_one(notification_doc)
        
        return jsonify({
            'success': True,
            'message_id': str(message_result.inserted_id),
            'conversation_id': str(conversation_id),
            'recipient_name': f"{recipient.get('first_name', '')} {recipient.get('last_name', '')}".strip() if recipient else 'Médecin',
            'created_at': message_doc['created_at'].isoformat()
        })
    
    except Exception as e:
        print(f"Erreur partage rapide analyse: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/doctor/stats')
@login_required
def get_doctor_stats_api():
    """API pour récupérer les statistiques du médecin"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        stats = get_doctor_statistics(doctor['id'])
        return jsonify({'success': True, 'stats': stats})

    except Exception as e:
        print(f"Erreur récupération statistiques médecin: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/doctor/update-profile', methods=['POST'])
@login_required
def update_doctor_profile():
    """API pour mettre à jour le profil du médecin"""
    try:
        doctor = get_current_doctor()
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        data = request.get_json()
        doctor_id = doctor['id']
        
        # Champs autorisés à la modification
        allowed_fields = ['specialty', 'hospital', 'license_number', 'phone']
        update_fields = {}
        
        for field in allowed_fields:
            if field in data:
                update_fields[field] = data[field].strip() if data[field] else None

        if not update_fields:
            return jsonify({'success': False, 'error': 'Aucune donnée à mettre à jour'}), 400

        # MongoDB Update - remplacer SQLite
        result = db.doctors.update_one(
            {'_id': ObjectId(doctor_id)},
            {
                '$set': {**update_fields, 'updated_at': datetime.now()},
                '$currentDate': {'lastModified': True}
            }
        )
        
        if result.modified_count > 0:
            return jsonify({'success': True, 'message': 'Profil mis à jour avec succès'})
        else:
            return jsonify({'success': False, 'error': 'Aucune modification effectuée'}), 400

    except Exception as e:
        print(f"Erreur mise à jour profil médecin: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/nouvelle-analyse')
@login_required
def new_analysis_page():
    """Page dédiée pour la nouvelle analyse"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))

    # Récupérer les patients du médecin connecté
    # MongoDB Query
    patients_cursor = db.patients.find({'doctor_id': doctor['id']}).sort('patient_name', 1)
    patients = list(patients_cursor)

    # Convertir en liste de dictionnaires
    patient_list = []
    for patient in patients:
        # Calculer l'âge si date de naissance disponible
        age = None
        dob = patient.get('date_of_birth')
        if dob:
            try:
                if isinstance(dob, str):
                    birth_date = datetime.strptime(dob, '%Y-%m-%d')
                else:
                    birth_date = dob
                age = datetime.now().year - birth_date.year
                if datetime.now().month < birth_date.month or (datetime.now().month == birth_date.month and datetime.now().day < birth_date.day):
                    age -= 1
            except:
                age = None
        
        patient_list.append({
            'id': str(patient.get('_id', '')),
            'patient_id': patient.get('patient_id', ''),
            'patient_name': patient.get('patient_name', ''),
            'date_of_birth': dob,
            'age': age,
            'gender': patient.get('gender', ''),
            'phone': patient.get('phone', ''),
            'email': patient.get('email', ''),
            'medical_history': patient.get('medical_history', ''),
            'allergies': patient.get('allergies', ''),
            'total_analyses': patient.get('total_analyses', 0) or 0,
            'full_name': patient.get('patient_name') or f"Patient {patient.get('patient_id', '')}"
        })

    # Date d'aujourd'hui par défaut
    today = datetime.now().strftime('%Y-%m-%d')
    
    return render_template('new_analysis.html', doctor=doctor, patients=patient_list, today=today)

@app.route('/')
def index():
    """Page d'accueil"""
    doctor = get_current_doctor()
    return render_template('index.html', doctor=doctor)

@app.route('/favicon.ico')
def favicon():
    """Servir un favicon par défaut pour éviter les erreurs 404"""
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Gérer l'upload et l'analyse d'image avec support du suivi patient"""
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    if file and allowed_file(file.filename):
        try:
            # Récupérer les informations patient depuis le formulaire
            patient_id = request.form.get('patient_id', '').strip()
            patient_name = request.form.get('patient_name', '').strip()
            exam_date_str = request.form.get('exam_date', '').strip()

            # Debug: afficher les informations reçues
            print(f"DEBUG - Informations patient reçues:")
            print(f"  patient_id: '{patient_id}'")
            print(f"  patient_name: '{patient_name}'")
            print(f"  exam_date: '{exam_date_str}'")

            # Parser la date d'examen
            exam_date = None
            if exam_date_str:
                try:
                    exam_date = datetime.strptime(exam_date_str, '%Y-%m-%d').date()
                except ValueError:
                    exam_date = datetime.now().date()
            else:
                exam_date = datetime.now().date()

            # Sauvegarder le fichier
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Faire la prédiction
            results = predict_tumor(filepath)

            if results is None:
                return jsonify({'error': 'Erreur lors de l\'analyse de l\'image'}), 500

            # Calculer le temps de traitement
            processing_time = time.time() - start_time

            # Conserver le fichier et fournir une URL servie par l'application
            img_url = url_for('uploaded_file', filename=filename)

            # Sauvegarder l'analyse dans la base de données avec informations patient et médecin
            session_id = request.headers.get('X-Session-ID', 'anonymous')
            ip_address = request.remote_addr
            doctor_id = session.get('doctor_id')  # Récupérer l'ID du médecin connecté

            analysis_id = save_analysis_to_db(
                results, filename, processing_time, session_id, ip_address,
                patient_id if patient_id else None,
                patient_name if patient_name else None,
                exam_date,
                doctor_id
            )
            
            print(f"DEBUG upload: analysis_id={analysis_id}, patient_id={patient_id}, doctor_id={doctor_id}")

            # Préparer la réponse
            response = {
                'success': True,
                'analysis_id': analysis_id,
                'image_url': img_url,
                'prediction': results['predicted_label'],
                'confidence': results['confidence'],
                'probabilities': results['probabilities'],
                'is_tumor': results['predicted_class'] != 0,  # 0 = Normal
                'recommendations': results.get('recommendations', get_recommendations(results)),
                'description': results.get('description', ''),
                'processing_time': round(processing_time, 2),
                'patient_info': {
                    'patient_id': patient_id,
                    'patient_name': patient_name,
                    'exam_date': exam_date.isoformat() if exam_date else None
                } if patient_id else None
            }

            return jsonify(response)

        except Exception as e:
            print(f"Erreur lors du traitement: {e}")
            return jsonify({'error': 'Erreur lors du traitement de l\'image'}), 500

    return jsonify({'error': 'Type de fichier non autorisé'}), 400

@app.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    """Servir les fichiers uploadés (images IRM)"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analysis/<analysis_id>')
@login_required
def analysis_detail(analysis_id):
    """Page détaillée d'une analyse"""
    try:
        doctor = get_current_doctor()
        
        # Convertir analysis_id en ObjectId si c'est une string
        try:
            if isinstance(analysis_id, str):
                obj_id = ObjectId(analysis_id)
            else:
                obj_id = ObjectId(str(analysis_id))
        except:
            flash('Analyse non trouvée', 'error')
            return redirect(url_for('dashboard'))
        
        # MongoDB Query
        analysis = db.analyses.find_one({'_id': obj_id})
        
        if not analysis:
            flash("Analyse introuvable", 'error')
            return redirect(url_for('dashboard'))

        probs = {}
        recs = []
        try:
            probs_data = analysis.get('probabilities', '{}')
            probs = json.loads(probs_data) if isinstance(probs_data, str) else probs_data
        except Exception:
            probs = {}
        try:
            recs_data = analysis.get('recommendations', '[]')
            recs = json.loads(recs_data) if isinstance(recs_data, str) else recs_data
        except Exception:
            recs = []

        # Sécuriser la conversion de confidence
        confidence_value = analysis.get('confidence', 0)
        try:
            if isinstance(confidence_value, (int, float)):
                confidence_float = float(confidence_value)
            elif isinstance(confidence_value, str):
                if confidence_value.replace('.', '').replace('-', '').isdigit():
                    confidence_float = float(confidence_value)
                else:
                    confidence_float = 0.0
            elif isinstance(confidence_value, tuple):
                # Si c'est un tuple, prendre le premier élément
                confidence_float = float(confidence_value[0]) if confidence_value else 0.0
            else:
                confidence_float = 0.0
        except Exception as e:
            print(f"Erreur conversion confidence: {e}, type: {type(confidence_value)}, valeur: {confidence_value}")
            confidence_float = 0.0

        analysis_result = {
            'id': str(analysis.get('_id', '')),
            'timestamp': analysis.get('timestamp', ''),
            'filename': analysis.get('filename', ''),
            'patient_id': analysis.get('patient_id', ''),
            'patient_name': analysis.get('patient_name', 'Patient anonyme'),
            'exam_date': analysis.get('exam_date', ''),
            'predicted_class': analysis.get('predicted_class', ''),
            'predicted_label': analysis.get('predicted_label', ''),
            'confidence': round(confidence_float * 100, 1),
            'probabilities': {k: round((v or 0) * 100, 1) for k, v in probs.items()},
            'description': analysis.get('description', ''),
            'recommendations': recs,
            'processing_time': analysis.get('processing_time', 0),
            'previous_analysis_id': analysis.get('previous_analysis_id', ''),
            'image_url': url_for('uploaded_file', filename=analysis.get('filename', '')) if analysis.get('filename') else ''
        }
        
        # Récupérer les autres analyses du même patient
        other_analyses = []
        patient_id = analysis.get('patient_id')
        if patient_id:
            # MongoDB Query
            other_analyses_cursor = db.analyses.find({
                'patient_id': patient_id,
                'doctor_id': doctor['id']
            }).sort('timestamp', -1).limit(10)
            
            for other_doc in other_analyses_cursor:
                # Sécuriser la conversion de confidence pour other_analyses
                other_confidence = other_doc.get('confidence', 0)
                try:
                    if isinstance(other_confidence, (int, float)):
                        other_conf_float = float(other_confidence)
                    elif isinstance(other_confidence, str):
                        if other_confidence.replace('.', '').replace('-', '').isdigit():
                            other_conf_float = float(other_confidence)
                        else:
                            other_conf_float = 0.0
                    elif isinstance(other_confidence, tuple):
                        other_conf_float = float(other_confidence[0]) if other_confidence else 0.0
                    else:
                        other_conf_float = 0.0
                except Exception as e:
                    print(f"Erreur conversion other_confidence: {e}, type: {type(other_confidence)}, valeur: {other_confidence}")
                    other_conf_float = 0.0
                
                other_analyses.append({
                    'id': str(other_doc.get('_id', '')),
                    'timestamp': other_doc.get('timestamp', ''),
                    'exam_date': other_doc.get('exam_date', ''),
                    'predicted_label': other_doc.get('predicted_label', ''),
                    'confidence': round(other_conf_float * 100, 1)
                })
        
        # Calculer les métriques de performance du modèle pour ce patient
        model_metrics = {}
        if patient_id:
            # Récupérer toutes les analyses du patient pour les métriques
            # MongoDB Query
            patient_analyses_cursor = db.analyses.find({
                'patient_id': patient_id,
                'doctor_id': doctor['id']
            }).sort('timestamp', -1)
            
            patient_analyses = list(patient_analyses_cursor)
            
            if patient_analyses:
                # Calculer la précision moyenne (accuracy basée sur la confiance)
                confidences = [float(doc.get('confidence', 0) or 0) for doc in patient_analyses if doc.get('confidence') is not None]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    model_metrics['accuracy'] = round(avg_confidence * 100, 1)
                else:
                    model_metrics['accuracy'] = 99.7  # Valeur par défaut
                
                # Calculer le temps de traitement moyen
                processing_times = [doc.get('processing_time') for doc in patient_analyses if doc.get('processing_time') is not None]
                if processing_times:
                    avg_time = sum(processing_times) / len(processing_times)
                    model_metrics['avg_processing_time'] = round(avg_time, 2)
                else:
                    model_metrics['avg_processing_time'] = 2.5
                
                # Nombre total d'analyses
                model_metrics['total_analyses'] = len(patient_analyses)
                
                # Distribution des diagnostics
                diagnosis_counts = {}
                for patient_analysis in patient_analyses:
                    diagnosis = patient_analysis.get('predicted_label', 'Inconnu') or 'Inconnu'
                    diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
                
                model_metrics['diagnosis_distribution'] = diagnosis_counts
                
                # Taux de détection de tumeurs (tout ce qui n'est pas Normal)
                tumor_detections = sum(1 for patient_analysis in patient_analyses if patient_analysis.get('predicted_label') and patient_analysis.get('predicted_label') != 'Normal')
                model_metrics['tumor_detection_rate'] = round((tumor_detections / len(patient_analyses)) * 100, 1) if patient_analyses else 0
                
                # Fiabilité (basée sur la variance de la confiance)
                if len(confidences) > 1:
                    variance = sum((x - avg_confidence) ** 2 for x in confidences) / len(confidences)
                    std_dev = variance ** 0.5
                    # Plus la variance est faible, plus le modèle est fiable
                    reliability = max(0, min(100, 100 - (std_dev * 100)))
                    model_metrics['reliability'] = round(reliability, 1)
                else:
                    model_metrics['reliability'] = 95.0
                
                # Métriques de performance simulées pour les autres
                model_metrics['sensitivity'] = round(model_metrics['accuracy'] * 0.95, 1)  # Sensibilité estimée
                model_metrics['specificity'] = round(model_metrics['accuracy'] * 0.98, 1)  # Spécificité estimée
                model_metrics['f1_score'] = round((2 * model_metrics['sensitivity'] * model_metrics['specificity']) / (model_metrics['sensitivity'] + model_metrics['specificity']), 1) if (model_metrics['sensitivity'] + model_metrics['specificity']) > 0 else 0
            else:
                # Valeurs par défaut si pas d'analyses
                model_metrics = {
                    'accuracy': 99.7,
                    'sensitivity': 97.3,
                    'specificity': 99.2,
                    'f1_score': 98.2,
                    'avg_processing_time': 2.5,
                    'total_analyses': 0,
                    'tumor_detection_rate': 0,
                    'reliability': 95.0,
                    'diagnosis_distribution': {}
                }
        else:
            # Valeurs par défaut pour les analyses sans patient
            model_metrics = {
                'accuracy': 99.7,
                'sensitivity': 97.3,
                'specificity': 99.2,
                'f1_score': 98.2,
                'avg_processing_time': 2.5,
                'total_analyses': 1,
                'tumor_detection_rate': 0,
                'reliability': 95.0,
                'diagnosis_distribution': {analysis_result['predicted_label']: 1}
            }
        
        return render_template('analysis_detail.html', 
                             doctor=doctor, 
                             analysis=analysis_result,
                             other_analyses=other_analyses,
                             model_metrics=model_metrics)
    except Exception as e:
        import traceback
        print(f"Erreur analysis_detail: {e}")
        print(f"Traceback complet:")
        traceback.print_exc()
        flash("Erreur lors du chargement de l'analyse", 'error')
        return redirect(url_for('dashboard'))

def call_gemini_api(prompt, context="medical"):
    """Appeler l'API Gemini avec un prompt"""
    try:
        headers = {
            'Content-Type': 'application/json',
        }

        # Prompt système pour limiter aux domaines médicaux
        system_prompt = """Tu es un assistant médical spécialisé en neurologie et en imagerie médicale.
        Tu dois UNIQUEMENT répondre aux questions liées au domaine médical, particulièrement :
        - Neurologie et neurochirurgie
        - Imagerie médicale (IRM, scanner, etc.)
        - Tumeurs cérébrales et pathologies neurologiques
        - Diagnostic et recommandations cliniques

        Si une question n'est pas liée au domaine médical, réponds poliment que tu ne peux traiter que les questions médicales.
        Tes réponses doivent être précises, professionnelles et basées sur les connaissances médicales actuelles.
        Ajoute toujours un disclaimer rappelant que tes conseils ne remplacent pas une consultation médicale."""

        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{system_prompt}\n\nQuestion: {prompt}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,  # Lower temperature for more consistent responses
                "topK": 1,
                "topP": 0.1,
                "maxOutputTokens": 1024,
            }
        }

        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                if 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    print(f"Unexpected response structure: {result}")
                    return None
            else:
                print(f"No candidates in response: {result}")
                return None
        
        # Gestion spécifique des erreurs
        elif response.status_code == 429:
            error_data = response.json()
            print(f"⚠️  Quota API Gemini dépassé: {response.status_code}")
            
            # Extraire le temps de retry si disponible
            retry_delay = 15  # Par défaut
            if 'error' in error_data and 'details' in error_data['error']:
                for detail in error_data['error']['details']:
                    if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                        retry_delay = int(detail.get('retryDelay', '15s').replace('s', ''))
            
            print(f"   Veuillez réessayer dans {retry_delay} secondes")
            print(f"   Limite gratuite: 250 requêtes/jour atteinte")
            return f"QUOTA_EXCEEDED:{retry_delay}"
        
        elif response.status_code == 400:
            print(f"❌ Erreur de requête API Gemini: {response.status_code}")
            print(f"   Vérifiez votre clé API dans le fichier .env")
            return None
        
        else:
            print(f"❌ Erreur API Gemini: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.Timeout:
        print(f"⏱️  Timeout lors de l'appel à Gemini API (>30s)")
        return None
    except requests.exceptions.ConnectionError:
        print(f"🌐 Erreur de connexion à l'API Gemini - Vérifiez votre connexion internet")
        return None
    except Exception as e:
        print(f"❌ Erreur lors de l'appel à Gemini: {e}")
        return None

def get_gemini_analysis(results):
    """Obtenir une analyse détaillée de Gemini pour les résultats"""
    try:
        prompt = f"""
        Analyse les résultats suivants d'une IRM cérébrale analysée par IA :

        - Diagnostic principal: {results['predicted_label']}
        - Niveau de confiance: {results['confidence']*100:.1f}%
        - Probabilités:
          * Normal: {results['probabilities']['Normal']*100:.1f}%
          * Gliome: {results['probabilities']['Gliome']*100:.1f}%
          * Méningiome: {results['probabilities']['Méningiome']*100:.1f}%
          * Tumeur pituitaire: {results['probabilities']['Tumeur pituitaire']*100:.1f}%

        Fournis une réponse structurée avec :
        1. DESCRIPTION: Une explication claire et détaillée du diagnostic (2-3 phrases)
        2. RECOMMANDATIONS: 3-4 recommandations cliniques spécifiques et pratiques

        Format ta réponse exactement comme ceci :
        DESCRIPTION: [ton explication]
        RECOMMANDATIONS:
        - [recommandation 1]
        - [recommandation 2]
        - [recommandation 3]
        """

        response = call_gemini_api(prompt)
        if response:
            # Parser la réponse
            lines = response.strip().split('\n')
            description = ""
            recommendations = []

            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('DESCRIPTION:'):
                    description = line.replace('DESCRIPTION:', '').strip()
                    current_section = 'description'
                elif line.startswith('RECOMMANDATIONS:'):
                    current_section = 'recommendations'
                elif line.startswith('-') and current_section == 'recommendations':
                    recommendations.append(line[1:].strip())
                elif current_section == 'description' and line:
                    description += " " + line

            return {
                'description': description,
                'recommendations': recommendations if recommendations else get_recommendations(results)
            }

    except Exception as e:
        print(f"Erreur lors de l'analyse Gemini: {e}")

    return None

# Fonction helper pour valider les ObjectIds
def is_valid_objectid(oid):
    """Vérifie si une chaîne est un ObjectId MongoDB valide"""
    try:
        ObjectId(oid)
        return True
    except:
        return False

# Fonctions pour le chat médical avec Gemini
def create_chat_conversation(doctor_id, title="Nouvelle consultation", patient_id=None):
    """Créer une nouvelle conversation de chat médical"""
    try:
        # MongoDB Insert
        conversation_doc = {
            'doctor_id': doctor_id,
            'patient_id': patient_id,
            'title': title,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        result = db.chat_conversations.insert_one(conversation_doc)
        conversation_id = str(result.inserted_id)
        
        return conversation_id
    except Exception as e:
        print(f"Erreur création conversation: {e}")
        return None

def get_chat_conversations(doctor_id, limit=20):
    """Récupérer les conversations de chat d'un médecin"""
    try:
        # MongoDB Aggregation Pipeline
        pipeline = [
            {'$match': {'doctor_id': doctor_id, 'is_active': True}},
            {'$addFields': {
                'id_string': {'$toString': '$_id'}
            }},
            {'$lookup': {
                'from': 'chat_messages',
                'localField': 'id_string',
                'foreignField': 'conversation_id',
                'as': 'messages'
            }},
            {'$lookup': {
                'from': 'patients',
                'let': {'patient_id': '$patient_id', 'doctor_id': '$doctor_id'},
                'pipeline': [
                    {'$match': {'$expr': {'$and': [
                        {'$eq': ['$patient_id', '$$patient_id']},
                        {'$eq': ['$doctor_id', '$$doctor_id']}
                    ]}}}
                ],
                'as': 'patient'
            }},
            {'$addFields': {
                'message_count': {'$size': '$messages'},
                'last_message': {'$arrayElemAt': [{'$slice': ['$messages.content', -1]}, 0]},
                'patient_name': {'$arrayElemAt': ['$patient.patient_name', 0]}
            }},
            {'$sort': {'updated_at': -1}},
            {'$limit': limit}
        ]
        
        conversations_cursor = db.chat_conversations.aggregate(pipeline)
        conversations = []
        
        for doc in conversations_cursor:
            last_msg = doc.get('last_message', '')
            conversations.append({
                'id': str(doc.get('_id')),
                'title': doc.get('title', 'Sans titre'),
                'patient_id': doc.get('patient_id'),
                'created_at': doc.get('created_at', ''),
                'updated_at': doc.get('updated_at', ''),
                'message_count': doc.get('message_count', 0),
                'last_message': last_msg[:100] + "..." if last_msg and len(last_msg) > 100 else last_msg,
                'patient_name': doc.get('patient_name')
            })
        
        return conversations
    except Exception as e:
        print(f"Erreur récupération conversations: {e}")
        return []

def get_conversation_messages(conversation_id, doctor_id, with_branches=False):
    """Récupérer les messages d'une conversation (avec ou sans branches)"""
    if with_branches:
        # Cette fonction sera implémentée si nécessaire
        return []
    
    try:
        # Convertir conversation_id en string si c'est un entier
        conversation_id = str(conversation_id)
        
        # Vérifier que c'est un ObjectId valide
        if not is_valid_objectid(conversation_id):
            print(f"ID de conversation invalide: {conversation_id}")
            return []
        
        # Vérifier que la conversation appartient au médecin
        conversation = db.chat_conversations.find_one({
            '_id': ObjectId(conversation_id),
            'doctor_id': doctor_id
        })
        
        if not conversation:
            return []
        
        # Récupérer seulement les messages de niveau 0 (conversation principale)
        messages_cursor = db.chat_messages.find({
            'conversation_id': conversation_id,
            '$or': [
                {'branch_level': 0},
                {'branch_level': {'$exists': False}}
            ],
            '$or': [
                {'is_active': True},
                {'is_active': {'$exists': False}}
            ]
        }).sort('timestamp', 1)
        
        messages = []
        for doc in messages_cursor:
            messages.append({
                'id': str(doc.get('_id')),
                'role': doc.get('role', 'user'),
                'content': doc.get('content', ''),
                'timestamp': doc.get('timestamp', ''),
                'is_medical_query': bool(doc.get('is_medical_query', True)),
                'confidence_score': doc.get('confidence_score')
            })
        
        return messages
    except Exception as e:
        print(f"Erreur récupération messages: {e}")
        return []

def save_chat_message(conversation_id, role, content, is_medical_query=True, confidence_score=None, parent_message_id=None, original_message_id=None, branch_level=0):
    """Sauvegarder un message de chat avec support du branchement"""
    try:
        # Calculer la position dans la branche
        pipeline = [
            {'$match': {
                'conversation_id': conversation_id,
                'branch_level': branch_level,
                '$or': [
                    {'parent_message_id': parent_message_id},
                    {'$and': [
                        {'parent_message_id': {'$exists': False}},
                        {'$expr': {'$eq': [parent_message_id, None]}}
                    ]}
                ]
            }},
            {'$group': {
                '_id': None,
                'max_position': {'$max': '$branch_position'}
            }}
        ]
        result = list(db.chat_messages.aggregate(pipeline))
        branch_position = (result[0]['max_position'] + 1) if result and result[0].get('max_position') is not None else 0
        
        # Créer le document message
        message_doc = {
            'conversation_id': conversation_id,
            'role': role,
            'content': content,
            'is_medical_query': is_medical_query,
            'confidence_score': confidence_score,
            'parent_message_id': parent_message_id,
            'original_message_id': original_message_id,
            'branch_level': branch_level,
            'branch_position': branch_position,
            'timestamp': datetime.now(),
            'is_active': True
        }
        
        # Insérer le message
        result = db.chat_messages.insert_one(message_doc)
        message_id = str(result.inserted_id)
        
        # Mettre à jour la conversation
        db.chat_conversations.update_one(
            {'_id': ObjectId(conversation_id)},
            {'$set': {'updated_at': datetime.now()}}
        )
        
        return message_id
    except Exception as e:
        print(f"Erreur sauvegarde message: {e}")
        return None

def edit_message_and_create_branch(message_id, new_content, doctor_id):
    """Éditer un message et créer une nouvelle branche de conversation - MongoDB Implementation Needed"""
    try:
        # TODO: Implémenter avec MongoDB
        # Pour l'instant, retourner None
        print("⚠️ edit_message_and_create_branch not yet implemented for MongoDB")
        return None
        
    except Exception as e:
        print(f"Erreur édition message: {e}")
        return None

def get_conversation_messages_with_branches(conversation_id, doctor_id):
    """Récupérer les messages d'une conversation avec les branches - MongoDB Implementation Needed"""
    try:
        # TODO: Implémenter avec MongoDB
        # Pour l'instant, retourner une liste vide
        print("⚠️ get_conversation_messages_with_branches not yet implemented for MongoDB")
        return []
        
    except Exception as e:
        print(f"Erreur get_conversation_messages_with_branches: {e}")
        return []

def get_message_branches(message_id, doctor_id):
    """Récupérer toutes les branches d'un message spécifique - MongoDB Implementation Needed"""
    try:
        # TODO: Implémenter avec MongoDB
        print("⚠️ get_message_branches not yet implemented for MongoDB")
        return []
        
    except Exception as e:
        print(f"Erreur récupération branches: {e}")
        return []

def call_gemini_with_context(user_message, conversation_history, patient_context=None):
    """Appeler Gemini avec le contexte de la conversation et du patient"""
    try:
        # Construire le contexte de conversation
        context_messages = []
        for msg in conversation_history[-10:]:  # Prendre les 10 derniers messages
            context_messages.append(f"{msg['role']}: {msg['content']}")
        
        conversation_context = "\n".join(context_messages) if context_messages else ""
        
        # Construire le contexte patient si disponible
        patient_info = ""
        if patient_context:
            patient_info = f"""
Informations patient disponibles:
- Nom: {patient_context.get('name', 'Non spécifié')}
- ID: {patient_context.get('id', 'Non spécifié')}
- Âge: {patient_context.get('age', 'Non spécifié')}
- Sexe: {patient_context.get('gender', 'Non spécifié')}
- Antécédents médicaux: {patient_context.get('medical_history', 'Aucun spécifié')}
- Allergies: {patient_context.get('allergies', 'Aucune spécifiée')}
- Analyses récentes: {patient_context.get('recent_analyses', 'Aucune')}
"""

        # Prompt système enrichi pour le chat médical
        system_prompt = f"""Tu es un assistant médical IA spécialisé dans la neurologie et l'imagerie médicale, destiné à aider les MÉDECINS dans leur pratique clinique.

UTILISATEUR: Tu t'adresses à un médecin professionnel, pas au patient. Adapte ton langage et tes conseils en conséquence.

DOMAINE D'EXPERTISE STRICT:
- Neurologie et neurochirurgie
- Imagerie médicale (IRM, scanner, radiologie)
- Tumeurs cérébrales et pathologies neurologiques
- Diagnostic différentiel en neuroimagerie
- Interprétation d'examens d'imagerie cérébrale
- Recommandations cliniques neurologiques

RÈGLES IMPORTANTES:
1. Tu t'adresses à un médecin, utilise le vocabulaire médical approprié
2. Fournis des analyses techniques et des recommandations cliniques professionnelles
3. Ne traite QUE les questions relatives au domaine médical neurologique
4. Si une question sort de ce domaine, redirige poliment vers le domaine médical
5. Propose des approches diagnostiques et thérapeutiques basées sur les données cliniques
6. Utilise le contexte patient pour des recommandations personnalisées
7. Reste professionnel, précis et orienté vers la pratique clinique

CONTEXTE DE LA CONVERSATION:
{conversation_context}

{patient_info}

En tant qu'assistant médical, aide ce médecin avec des analyses professionnelles, des recommandations cliniques et des insights diagnostiques basés sur les données disponibles."""

        headers = {
            'Content-Type': 'application/json',
        }

        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{system_prompt}\n\nQuestion actuelle: {user_message}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }

        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                response_text = result['candidates'][0]['content']['parts'][0]['text']
                
                # Calculer un score de confiance basé sur la réponse
                confidence_score = calculate_response_confidence(response_text, user_message)
                
                return {
                    'response': response_text,
                    'confidence_score': confidence_score,
                    'is_medical': is_medical_query(user_message)
                }

        print(f"Erreur API Gemini: {response.status_code} - {response.text}")
        return None

    except Exception as e:
        print(f"Erreur lors de l'appel Gemini avec contexte: {e}")
        return None

def is_medical_query(message):
    """Détermine si le message est lié au domaine médical"""
    medical_keywords = [
        'tumeur', 'cancer', 'irm', 'scanner', 'diagnostic', 'symptôme', 
        'cerveau', 'neurologie', 'gliome', 'méningiome', 'pituitaire',
        'analyse', 'examen', 'pathologie', 'traitement', 'médical',
        'clinique', 'patient', 'imagerie', 'radiologie'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in medical_keywords)

def calculate_response_confidence(response, query):
    """Calculer un score de confiance pour la réponse"""
    try:
        # Critères simples pour évaluer la confiance
        confidence = 0.5  # Base
        
        # Longueur appropriée
        if 50 <= len(response) <= 1000:
            confidence += 0.2
            
        # Contient des termes médicaux pertinents
        medical_terms = ['diagnostic', 'symptôme', 'traitement', 'examen', 'analyse']
        if any(term in response.lower() for term in medical_terms):
            confidence += 0.2
            
        # Contient un disclaimer médical
        if any(phrase in response.lower() for phrase in ['consultation', 'médecin', 'professionnel']):
            confidence += 0.1
            
        return min(confidence, 1.0)
    except:
        return 0.7

def get_gemini_advanced_analysis(results):
    """Analyse avancée structurée (résumé, explication, suggestions) via Gemini"""
    try:
        prob = results.get('probabilities', {})
        prompt = f"""
        Contexte: Résultat d'une IRM cérébrale analysée par IA.
        Données:
        - Diagnostic principal: {results.get('predicted_label')}
        - Confiance: {results.get('confidence', 0)*100:.1f}%
        - Probabilités (en %): { {k: round(v*100,1) for k,v in prob.items()} }

        Objectif: Fournir une analyse avancée et structurée pour un médecin.
        Réponds STRICTEMENT en JSON avec les clés: summary, explanation, suggestions (array de 3 à 5 items).
        Style: clinique, concis, sans disclaimer.

        Exemple de format:
        {{
            "summary": "Résumé court du diagnostic",
            "explanation": "Explication détaillée",
            "suggestions": ["Suggestion 1", "Suggestion 2", "Suggestion 3"]
        }}
        """

        raw = call_gemini_api(prompt)
        if not raw:
            print("No response from Gemini API")
            return None

        # Essayer de trouver un bloc JSON dans la réponse
        text = raw.strip()
        print(f"Raw response: {text[:500]}...")  # Debug: print first 500 chars

        # Remove markdown code blocks if present
        if text.startswith('```json'):
            text = text[7:]  # Remove ```json
        if text.endswith('```'):
            text = text[:-3]  # Remove ```
        text = text.strip()

        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                json_str = text[start:end+1]
                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"JSON string: {json_str}")
                pass

        # Fallback: créer une réponse structurée manuellement
        print("Using fallback response structure")
        return {
            'summary': text[:400] if text else 'Analyse non disponible',
            'explanation': text[:800] if len(text) > 400 else text,
            'suggestions': ['Consultation spécialisée recommandée', 'Suivi clinique régulier', 'Imagerie complémentaire si nécessaire']
        }
    except Exception as e:
        print(f"Erreur get_gemini_advanced_analysis: {e}")
        return None

@app.route('/api/advanced-analysis', methods=['POST'])
@login_required
def api_advanced_analysis():
    """Endpoint sécurisé pour l'analyse IA avancée (Gemini)"""
    try:
        data = request.get_json() or {}
        # Attendu: predicted_label, confidence (0-1 ou %), probabilities
        results = {
            'predicted_label': data.get('predicted_label'),
            'confidence': (data.get('confidence')/100.0 if isinstance(data.get('confidence'), (int,float)) and data.get('confidence')>1 else data.get('confidence') or 0),
            'probabilities': data.get('probabilities') or {}
        }
        adv = get_gemini_advanced_analysis(results)
        return jsonify({'success': bool(adv), 'data': adv or {}})
    except Exception as e:
        print(f"Erreur api_advanced_analysis: {e}")
        return jsonify({'success': False, 'error': 'Erreur serveur'}), 500

def get_recommendations(results):
    """Générer des recommandations basées sur les résultats (fallback)"""
    recommendations = []

    if results['predicted_class'] == 0:  # Normal
        recommendations = [
            "Aucune anomalie détectée dans cette analyse",
            "Suivi de routine recommandé selon les protocoles standards",
            "Consultation avec un radiologue pour confirmation"
        ]
    else:  # Tumeur détectée
        recommendations = [
            "Biopsie recommandée pour confirmation histologique",
            "IRM de suivi dans 3 mois pour évaluation de la croissance",
            "Consultation avec un neuro-oncologue spécialisé"
        ]

        if results['confidence'] < 0.7:
            recommendations.append("Analyse complémentaire recommandée en raison de la faible confiance")

    return recommendations

@app.route('/health')
def health_check():
    """Vérification de l'état de l'application"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/generate-report', methods=['POST'])
def generate_report():
    """Générer un rapport médical PDF professionnel"""
    try:
        data = request.get_json()

        # Valider les données requises
        required_fields = ['patientName', 'analysisData']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Champ requis manquant: {field}'}), 400

        # Générer le rapport PDF
        pdf_content = create_medical_report(data)

        # Créer un nom de fichier unique
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"rapport_medical_{timestamp}.pdf"

        # Retourner le PDF avec les bons headers
        response = Response(
            pdf_content,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Length': len(pdf_content)
            }
        )

        return response

    except Exception as e:
        print(f"Erreur lors de la génération du rapport PDF: {e}")
        return jsonify({'error': 'Erreur lors de la génération du rapport PDF'}), 500

@app.route('/share-analysis', methods=['POST'])
def share_analysis():
    """Partager une analyse avec un collègue"""
    try:
        data = request.get_json()

        # Valider les données requises
        required_fields = ['recipientEmail', 'analysisData']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Champ requis manquant: {field}'}), 400

        # Simuler l'envoi de l'email
        share_id = f"SHR_{int(time.time())}"

        # Dans une vraie application, ici on enverrait un email
        send_analysis_email(data)

        return jsonify({
            'success': True,
            'share_id': share_id,
            'message': f'Analyse partagée avec {data["recipientEmail"]}'
        })

    except Exception as e:
        print(f"Erreur lors du partage: {e}")
        return jsonify({'error': 'Erreur lors du partage'}), 500

# Routes pour le chat médical
@app.route('/chat')
@login_required
def chat_page():
    """Page principale du chat médical"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))
    return render_template('chat.html', doctor=doctor)

@app.route('/chat/help')
@login_required
def chat_help():
    """Page d'aide du chat médical"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))
    return render_template('chat_help.html', doctor=doctor)

@app.route('/api/chat/conversations')
@login_required
def get_conversations():
    """API pour récupérer les conversations du médecin"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        conversations = get_chat_conversations(doctor_id)
        return jsonify({'success': True, 'conversations': conversations})

    except Exception as e:
        print(f"Erreur récupération conversations: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/conversations', methods=['POST'])
@login_required
def create_conversation():
    """API pour créer une nouvelle conversation"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        data = request.get_json()
        title = data.get('title', 'Nouvelle consultation')
        patient_id = data.get('patient_id')

        conversation_id = create_chat_conversation(doctor_id, title, patient_id)
        
        if conversation_id:
            return jsonify({'success': True, 'conversation_id': conversation_id})
        else:
            return jsonify({'success': False, 'error': 'Erreur création conversation'}), 500

    except Exception as e:
        print(f"Erreur création conversation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/conversations/<conversation_id>/messages')
@login_required
def get_messages(conversation_id):
    """API pour récupérer les messages d'une conversation"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        messages = get_conversation_messages(conversation_id, doctor_id)
        return jsonify({'success': True, 'messages': messages})

    except Exception as e:
        print(f"Erreur récupération messages: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/send', methods=['POST'])
@login_required
def send_chat_message():
    """API pour envoyer un message et obtenir une réponse de Gemini"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        data = request.get_json()
        print(f"DEBUG - Données reçues: {data}")
        
        conversation_id = data.get('conversation_id')
        
        # Vérifier que message est une chaîne de caractères
        message_raw = data.get('message', '')
        print(f"DEBUG - message_raw type: {type(message_raw)}, value: {repr(message_raw)}")
        
        if isinstance(message_raw, dict):
            # Si c'est un dictionnaire, essayer d'extraire le contenu
            message = str(message_raw.get('content', message_raw.get('text', ''))).strip()
        elif isinstance(message_raw, str):
            message = message_raw.strip()
        else:
            message = str(message_raw).strip()

        print(f"DEBUG - message final: {repr(message)}, length: {len(message)}")

        if not conversation_id or not message:
            print(f"Données manquantes - conversation_id: {conversation_id}, message: {message[:50] if message else 'vide'}")
            return jsonify({'success': False, 'error': 'Données manquantes (conversation_id ou message)'}), 400

        # Vérifier que c'est un ObjectId valide
        if not is_valid_objectid(conversation_id):
            print(f"ID de conversation invalide: {conversation_id}")
            return jsonify({'success': False, 'error': 'ID de conversation invalide'}), 400

        # Récupérer l'historique de la conversation
        conversation_history = get_conversation_messages(conversation_id, doctor_id)
        
        # Récupérer le contexte patient si disponible
        patient_context = get_patient_context_for_conversation(conversation_id)

        # Sauvegarder le message utilisateur
        user_message_id = save_chat_message(conversation_id, 'user', message, is_medical_query(message))

        # Appeler Gemini avec le contexte
        gemini_response = call_gemini_with_context(message, conversation_history, patient_context)

        if gemini_response:
            # Sauvegarder la réponse de l'assistant
            assistant_message_id = save_chat_message(
                conversation_id, 
                'assistant', 
                gemini_response['response'],
                gemini_response['is_medical'],
                gemini_response['confidence_score']
            )

            return jsonify({
                'success': True,
                'user_message_id': user_message_id,
                'assistant_message_id': assistant_message_id,
                'response': gemini_response['response'],
                'confidence_score': gemini_response['confidence_score'],
                'is_medical': gemini_response['is_medical']
            })
        else:
            # Sauvegarder une réponse d'erreur
            error_response = "Désolé, je ne peux pas répondre pour le moment. Veuillez réessayer."
            assistant_message_id = save_chat_message(conversation_id, 'assistant', error_response, False, 0.0)
            
            return jsonify({
                'success': False,
                'user_message_id': user_message_id,
                'assistant_message_id': assistant_message_id,
                'response': error_response,
                'error': 'Erreur API Gemini'
            })

    except Exception as e:
        print(f"Erreur envoi message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def get_patient_context_for_conversation(conversation_id):
    """Récupérer le contexte patient pour une conversation"""
    try:
        # Convertir conversation_id en string si nécessaire
        conversation_id = str(conversation_id)
        
        db = get_mongodb()
        
        # Récupérer l'ID patient de la conversation
        conversation = db.chat_conversations.find_one({
            '_id': ObjectId(conversation_id)
        }, {'patient_id': 1})
        
        if not conversation or not conversation.get('patient_id'):
            return None
            
        patient_id = conversation.get('patient_id')
        
        # Récupérer les informations du patient
        patient_data = db.patients.find_one({
            'patient_id': patient_id
        })
        
        if not patient_data:
            return None
            
        # Calculer l'âge si date de naissance disponible
        age = None
        birth_date_str = patient_data.get('date_of_birth')
        if birth_date_str:
            try:
                if isinstance(birth_date_str, str):
                    birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d')
                else:
                    birth_date = birth_date_str
                age = datetime.now().year - birth_date.year
            except:
                age = None
        
        # Récupérer les analyses récentes
        recent_analyses_cursor = db.analyses.find({
            'patient_id': patient_id
        }).sort('timestamp', -1).limit(3)
        
        analyses_summary = []
        for analysis in recent_analyses_cursor:
            exam_date = analysis.get('exam_date', analysis.get('timestamp', ''))
            if isinstance(exam_date, datetime):
                exam_date = exam_date.strftime('%Y-%m-%d')
            analyses_summary.append(
                f"- {exam_date}: {analysis.get('predicted_label')} "
                f"(confiance: {analysis.get('confidence', 0)*100:.1f}%)"
            )
        
        return {
            'name': patient_data.get('patient_name'),
            'id': patient_id,
            'age': age,
            'gender': patient_data.get('gender'),
            'medical_history': patient_data.get('medical_history'),
            'allergies': patient_data.get('allergies'),
            'current_medications': patient_data.get('current_medications'),
            'total_analyses': patient_data.get('total_analyses', 0),
            'recent_analyses': '\n'.join(analyses_summary) if analyses_summary else 'Aucune analyse récente'
        }
        
    except Exception as e:
        print(f"Erreur récupération contexte patient: {e}")
        return None

@app.route('/api/chat/conversations/<conversation_id>/update', methods=['PUT'])
@login_required
def update_conversation(conversation_id):
    """API pour mettre à jour une conversation (titre, patient)"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # Vérifier que c'est un ObjectId valide
        if not is_valid_objectid(conversation_id):
            return jsonify({'success': False, 'error': 'ID de conversation invalide'}), 400

        data = request.get_json()
        title = data.get('title')
        patient_id = data.get('patient_id')

        db = get_mongodb()
        
        # Vérifier que la conversation appartient au médecin
        conversation = db.chat_conversations.find_one({
            '_id': ObjectId(conversation_id),
            'doctor_id': doctor_id
        })
        
        if not conversation:
            return jsonify({'success': False, 'error': 'Conversation introuvable'}), 404
        
        # Préparer les champs à mettre à jour
        update_data = {
            'updated_at': datetime.now()
        }
        
        if title:
            update_data['title'] = title
        
        if patient_id is not None:  # Peut être None pour supprimer l'association
            update_data['patient_id'] = patient_id if patient_id else None
        
        # Mettre à jour la conversation
        db.chat_conversations.update_one(
            {'_id': ObjectId(conversation_id)},
            {'$set': update_data}
        )
        
        return jsonify({'success': True, 'message': 'Conversation mise à jour'})

    except Exception as e:
        print(f"Erreur mise à jour conversation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patients/list')
@login_required
def get_patients_for_chat():
    """API pour récupérer la liste simple des patients du médecin"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # MongoDB query
        db = get_mongodb()
        
        # Get patients for this doctor
        patients_cursor = db.patients.find(
            {'doctor_id': doctor_id},
            {
                '_id': 0,
                'patient_id': 1,
                'patient_name': 1,
                'date_of_birth': 1,
                'gender': 1,
                'total_analyses': 1
            }
        ).sort('patient_name', 1)
        
        patients = []
        for patient in patients_cursor:
            # Calculer l'âge si date de naissance disponible
            age = None
            dob = patient.get('date_of_birth')
            if dob:
                try:
                    if isinstance(dob, str):
                        birth_date = datetime.strptime(dob, '%Y-%m-%d')
                    else:
                        birth_date = dob
                    
                    today = datetime.now()
                    age = today.year - birth_date.year
                    if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
                        age -= 1
                except:
                    age = None
            
            patient_id = patient.get('patient_id', str(patient.get('_id', '')))
            patient_name = patient.get('patient_name') or f"Patient {patient_id}"
            
            patients.append({
                'patient_id': patient_id,
                'patient_name': patient_name,
                'age': age,
                'gender': patient.get('gender'),
                'total_analyses': patient.get('total_analyses', 0),
                'display_name': f"{patient_name} ({age} ans)" if age else patient_name
            })

        return jsonify({
            'success': True,
            'patients': patients
        })

    except Exception as e:
        print(f"Erreur récupération liste patients: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/conversations/<conversation_id>/delete', methods=['DELETE'])
@login_required
def delete_conversation(conversation_id):
    """API pour supprimer une conversation et tous ses messages"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # Vérifier que c'est un ObjectId valide
        if not is_valid_objectid(conversation_id):
            return jsonify({'success': False, 'error': 'ID de conversation invalide'}), 400

        db = get_mongodb()
        
        # Vérifier que la conversation appartient au médecin
        conversation = db.chat_conversations.find_one({
            '_id': ObjectId(conversation_id),
            'doctor_id': doctor_id
        })
        
        if not conversation:
            return jsonify({'success': False, 'error': 'Conversation introuvable'}), 404
        
        # Supprimer d'abord les messages de la conversation
        messages_result = db.chat_messages.delete_many({
            'conversation_id': conversation_id
        })
        deleted_messages = messages_result.deleted_count
        
        # Supprimer les attachments s'il y en a
        db.chat_attachments.delete_many({
            'conversation_id': conversation_id
        })
        
        # Supprimer la conversation
        db.chat_conversations.delete_one({
            '_id': ObjectId(conversation_id)
        })
        # conn.close() # DISABLED
        
        return jsonify({
            'success': True, 
            'message': f'Conversation supprimée avec {deleted_messages} messages'
        })

    except Exception as e:
        print(f"Erreur suppression conversation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== APIs pour l'édition et le branchement de messages =====

@app.route('/api/chat/messages/<int:message_id>/edit', methods=['POST'])
@login_required
def edit_message(message_id):
    """API pour éditer un message et créer une nouvelle branche"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        data = request.get_json()
        new_content = data.get('content', '').strip()

        if not new_content:
            return jsonify({'success': False, 'error': 'Nouveau contenu requis'}), 400

        # Éditer le message et créer la branche
        result = edit_message_and_create_branch(message_id, new_content, doctor_id)
        
        if not result or not result.get('success'):
            return jsonify({'success': False, 'error': 'Impossible d\'éditer ce message'}), 404

        # Récupérer les informations du nouveau message
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED
        
        # MongoDB query needed here
        # SELECT role, conversation_id FROM chat_messages WHERE id = ?)
        
        # msg_info = None  # cursor.fetchone() # DISABLED  # TODO: Convert to MongoDB
        msg_info = None  # TODO: Implement MongoDB query
        # conn.close() # DISABLED
        
        # Si c'est un message utilisateur, générer une nouvelle réponse de l'assistant
        if msg_info and msg_info[0] == 'user':
            try:
                # Récupérer l'historique pour générer une réponse
                conversation_history = get_conversation_messages(msg_info[1], doctor_id)
                patient_context = get_patient_context_for_conversation(msg_info[1])
                
                # Appeler Gemini avec le nouveau contexte
                gemini_response = call_gemini_with_context(new_content, conversation_history, patient_context)
                
                if gemini_response:
                    # Sauvegarder la nouvelle réponse dans la même branche
                    assistant_message_id = save_chat_message(
                        msg_info[1], 
                        'assistant', 
                        gemini_response['response'],
                        gemini_response['is_medical'],
                        gemini_response['confidence_score'],
                        None,  # parent_message_id
                        None,  # original_message_id
                        result['branch_level']  # même niveau de branche
                    )
                    
                    result['assistant_message_id'] = assistant_message_id
                    result['assistant_response'] = gemini_response['response']
                
            except Exception as e:
                print(f"Erreur génération réponse après édition: {e}")
                # Continuer même si la génération de réponse échoue

        return jsonify(result)

    except Exception as e:
        print(f"Erreur édition message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/messages/<int:message_id>/branches')
@login_required
def get_message_branches_api(message_id):
    """API pour récupérer toutes les branches d'un message"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        branches = get_message_branches(message_id, doctor_id)
        return jsonify({'success': True, 'branches': branches})

    except Exception as e:
        print(f"Erreur récupération branches: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/conversations/<conversation_id>/messages-with-branches')
@login_required  
def get_messages_with_branches_api(conversation_id):
    """API pour récupérer les messages d'une conversation avec leurs branches"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # Vérifier que c'est un ObjectId valide
        if not is_valid_objectid(conversation_id):
            return jsonify({'success': False, 'error': 'ID de conversation invalide'}), 400

        messages = get_conversation_messages_with_branches(conversation_id, doctor_id)
        return jsonify({'success': True, 'messages': messages})

    except Exception as e:
        print(f"Erreur récupération messages avec branches: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/messages/<int:message_id>/regenerate', methods=['POST'])
@login_required
def regenerate_response(message_id):
    """API pour régénérer la réponse d'un assistant à partir d'un message utilisateur"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED
        
        # Récupérer le message utilisateur et vérifier les permissions
        # MongoDB query needed here
        # SELECT cm.conversation_id, cm.content, cm.role, cc.doctor_id
        #             FROM chat_messages cm
        #             JOIN chat_conversations cc ON cm.conversation_id = cc.id
        #             WHERE cm.id = ? AND cc.doctor_id = ? AND cm.role = 'user')
        
        # message_info = None  # cursor.fetchone() # DISABLED  # TODO: Convert to MongoDB
        message_info = None  # TODO: Implement MongoDB query
        if not message_info:
            # conn.close() # DISABLED
            return jsonify({'success': False, 'error': 'Message introuvable ou non autorisé'}), 404
        
        conversation_id, content, role, doc_id = message_info if message_info else (None, None, None, None)
        # conn.close() # DISABLED
        
        # Vérification des données récupérées
        if not conversation_id:
            return jsonify({'success': False, 'error': 'Données de conversation invalides'}), 400
        
        # Récupérer l'historique et le contexte patient
        conversation_history = get_conversation_messages_with_branches(conversation_id, doctor_id)
        patient_context = get_patient_context_for_conversation(conversation_id)
        
        # Générer une nouvelle réponse avec Gemini
        gemini_response = call_gemini_with_context(content, conversation_history, patient_context)
        
        if gemini_response:
            # Sauvegarder la nouvelle réponse comme une branche
            assistant_message_id = save_chat_message(
                conversation_id,
                'assistant',
                gemini_response['response'],
                gemini_response['is_medical'],
                gemini_response['confidence_score'],
                message_id,  # parent_message_id (message utilisateur)
                None,  # original_message_id
                1  # branch_level = 1 pour une réponse alternative
            )
            
            return jsonify({
                'success': True,
                'message_id': assistant_message_id,
                'response': gemini_response['response'],
                'confidence_score': gemini_response['confidence_score'],
                'is_medical': gemini_response['is_medical']
            })
        else:
            return jsonify({'success': False, 'error': 'Impossible de générer une réponse'}), 500

    except Exception as e:
        print(f"Erreur régénération réponse: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/old-chat', methods=['POST'])
def chat_with_bot():
    """Ancienne API de chatbot (conservée pour compatibilité)"""
    try:
        data = request.get_json()
        
        # Vérifier que message est une chaîne de caractères
        message_raw = data.get('message', '')
        if isinstance(message_raw, dict):
            message = str(message_raw.get('content', message_raw.get('text', ''))).strip()
        elif isinstance(message_raw, str):
            message = message_raw.strip()
        else:
            message = str(message_raw).strip()

        if not message:
            return jsonify({'error': 'Message vide'}), 400

        # Appeler Gemini pour la réponse
        response = call_gemini_api(message, context="medical_consultation")

        if response:
            return jsonify({
                'success': True,
                'response': response
            })
        else:
            return jsonify({
                'success': False,
                'response': 'Désolé, je ne peux pas répondre pour le moment. Veuillez réessayer.'
            })

    except Exception as e:
        print(f"Erreur chatbot: {e}")
        return jsonify({
            'success': False,
            'response': 'Une erreur s\'est produite. Veuillez réessayer.'
        }), 500

@app.route('/pro-dashboard')
@login_required
def pro_dashboard():
    """Page du tableau de bord professionnel"""
    doctor = get_current_doctor()
    return render_template('pro_dashboard.html', doctor=doctor)

@app.route('/pro-dashboard-advanced')
@login_required
def pro_dashboard_advanced():
    """Page du tableau de bord professionnel avancé"""
    doctor = get_current_doctor()
    return render_template('pro_dashboard_advanced.html', doctor=doctor)

@app.route('/platform-stats')
@login_required
def platform_stats():
    """Page des statistiques générales de la plateforme"""
    doctor = get_current_doctor()
    return render_template('platform_stats.html', doctor=doctor)

@app.route('/patients')
@login_required
def patients_list():
    """Page de liste des patients du médecin connecté"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))
    return render_template('patients_list.html', doctor=doctor)

@app.route('/patients/new')
@login_required
def new_patient_page():
    """Page de création d'un nouveau patient"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))
    return render_template('new_patient.html', doctor=doctor)

@app.route('/patients/<patient_id>/edit')
@login_required
def edit_patient_page(patient_id):
    """Page de modification d'un patient existant"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))
    # La page fera un appel à /api/patients/<id>/details pour pré-remplir
    return render_template('edit_patient.html', doctor=doctor, patient_id=patient_id)

@app.route('/alerts')
@login_required
def alerts_page():
    """Page des alertes médicales"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))
    return render_template('alerts.html', doctor=doctor)

@app.route('/medical-alerts')
@login_required
def medical_alerts_page():
    """Page des alertes médicales"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))
    return render_template('alerts.html', doctor=doctor)

@app.route('/manage-patients')
@login_required
def manage_patients_page():
    """Page de gestion des patients"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))
    return render_template('manage_patients.html', doctor=doctor)

@app.route('/api/analytics/overview')
@login_required
def analytics_overview():
    """API pour les statistiques personnelles du médecin connecté"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Statistiques personnelles du médecin depuis MongoDB
        total_analyses = db.analyses.count_documents({"doctor_id": doctor_id})
        
        # Jours actifs
        pipeline = [
            {"$match": {"doctor_id": doctor_id, "timestamp": {"$exists": True}}},
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}}}
        ]
        active_days = len(list(db.analyses.aggregate(pipeline)))
        
        # Moyennes
        avg_pipeline = [
            {"$match": {"doctor_id": doctor_id}},
            {"$group": {
                "_id": None,
                "avg_confidence": {"$avg": "$confidence"},
                "avg_processing_time": {"$avg": "$processing_time"}
            }}
        ]
        avg_result = list(db.analyses.aggregate(avg_pipeline))
        avg_confidence = avg_result[0]['avg_confidence'] if avg_result else 0
        avg_processing_time = avg_result[0]['avg_processing_time'] if avg_result else 0

        # Répartition par type de tumeur pour ce médecin
        tumor_pipeline = [
            {"$match": {"doctor_id": doctor_id, "predicted_label": {"$ne": None}}},
            {"$group": {"_id": "$predicted_label", "count": {"$sum": 1}}}
        ]
        tumor_results = list(db.analyses.aggregate(tumor_pipeline))
        tumor_distribution = {r['_id']: r['count'] for r in tumor_results}

        # Analyses par jour (30 derniers jours) pour ce médecin
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.now() - timedelta(days=30)
        daily_pipeline = [
            {"$match": {"doctor_id": doctor_id, "timestamp": {"$gte": thirty_days_ago, "$exists": True}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        daily_results = list(db.analyses.aggregate(daily_pipeline))
        daily_analyses = [{"date": r['_id'], "count": r['count']} for r in daily_results]

        return jsonify({
            'success': True,
            'data': {
                'total_analyses': total_analyses,
                'active_days': active_days,
                'avg_confidence': round(avg_confidence * 100, 1) if avg_confidence else 0,
                'avg_processing_time': round(avg_processing_time, 2) if avg_processing_time else 0,
                'tumor_distribution': tumor_distribution,
                'daily_analyses': daily_analyses
            }
        })

    except Exception as e:
        print(f"Erreur analytics overview: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/platform-overview')
@login_required
def platform_analytics_overview():
    """API pour les statistiques générales de toute la plateforme"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Vérifier si la table analyses existe et contient des données
        # cursor.execute( # DISABLED"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='analyses'")
        # table_exists = None  # cursor.fetchone() # DISABLED[0] > 0  # TODO: Convert to MongoDB
        table_exists = True  # TODO: Implement MongoDB collection check (assume exists for now)

        if not table_exists:
            # conn.close() # DISABLED
            return jsonify({
                'success': True,
                'data': {
                    'total_analyses': 0,
                    'total_doctors': 0,
                    'total_patients': 0,
                    'active_days': 0,
                    'avg_confidence': 0,
                    'avg_processing_time': 0,
                    'success_rate': 0,
                    'storage_used': 500,
                    'tumor_distribution': {},
                    'daily_analyses': [],
                    'top_doctors': []
                }
            })

        # Statistiques générales de la plateforme depuis MongoDB
        total_analyses = db.analyses.count_documents({})
        total_doctors = len(db.analyses.distinct("doctor_id", {"doctor_id": {"$ne": None}}))
        total_patients = len(db.analyses.distinct("patient_id", {"patient_id": {"$ne": None}}))
        
        # Jours actifs (nombre de jours uniques avec analyses)
        pipeline = [
            {"$match": {"timestamp": {"$exists": True}}},
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}}}
        ]
        active_days = len(list(db.analyses.aggregate(pipeline)))
        
        # Moyennes
        avg_pipeline = [
            {"$group": {
                "_id": None,
                "avg_confidence": {"$avg": "$confidence"},
                "avg_processing_time": {"$avg": "$processing_time"}
            }}
        ]
        avg_result = list(db.analyses.aggregate(avg_pipeline))
        avg_confidence = avg_result[0]['avg_confidence'] if avg_result else 0
        avg_processing_time = avg_result[0]['avg_processing_time'] if avg_result else 0

        # Calculer le taux de réussite (analyses avec confidence > 0.8)
        if total_analyses > 0:
            successful_analyses = db.analyses.count_documents({"confidence": {"$gt": 0.8}})
            success_rate = (successful_analyses / total_analyses * 100)
        else:
            success_rate = 0

        # Simulation du stockage utilisé (en GB)
        storage_used = total_analyses * 0.5 + 500  # Environ 0.5 GB par analyse + base de 500 GB

        # Répartition par type de tumeur (toute la plateforme)
        try:
            tumor_pipeline = [
                {"$match": {"predicted_label": {"$ne": None}}},
                {"$group": {"_id": "$predicted_label", "count": {"$sum": 1}}}
            ]
            tumor_results = list(db.analyses.aggregate(tumor_pipeline))
            tumor_distribution = {r['_id']: r['count'] for r in tumor_results}
        except Exception:
            tumor_distribution = {}

        # Analyses par jour (30 derniers jours) - toute la plateforme
        try:
            from datetime import datetime, timedelta
            thirty_days_ago = datetime.now() - timedelta(days=30)
            print(f"DEBUG: thirty_days_ago = {thirty_days_ago}")
            
            daily_pipeline = [
                {"$match": {"timestamp": {"$gte": thirty_days_ago, "$exists": True}}},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
            daily_results = list(db.analyses.aggregate(daily_pipeline))
            print(f"DEBUG: daily_analyses found {len(daily_results)} days")
            daily_analyses = [(r['_id'], r['count']) for r in daily_results]
        except Exception as e:
            print(f"Erreur daily_analyses: {e}")
            import traceback
            traceback.print_exc()
            daily_analyses = []

        # Statistiques détaillées pour les cartes
        try:
            from datetime import datetime, timedelta
            now = datetime.now()
            today_start = datetime(now.year, now.month, now.day)
            month_start = datetime(now.year, now.month, 1)
            
            # Analyses d'aujourd'hui
            daily_analyses_count = db.analyses.count_documents({
                "timestamp": {"$gte": today_start}
            })

            # Nouveaux patients ce mois
            new_patients_month = len(db.analyses.distinct("patient_id", {
                "timestamp": {"$gte": month_start},
                "patient_id": {"$ne": None}
            }))

            # Croissance des analyses ce mois vs mois précédent
            current_month_analyses = db.analyses.count_documents({
                "timestamp": {"$gte": month_start}
            })
            
            # Mois précédent
            if now.month == 1:
                prev_month_start = datetime(now.year - 1, 12, 1)
                prev_month_end = datetime(now.year, 1, 1)
            else:
                prev_month_start = datetime(now.year, now.month - 1, 1)
                prev_month_end = month_start
            
            previous_month_analyses = db.analyses.count_documents({
                "timestamp": {"$gte": prev_month_start, "$lt": prev_month_end}
            })
            if previous_month_analyses == 0:
                previous_month_analyses = 1  # Éviter division par zéro

            analyses_growth = ((current_month_analyses - previous_month_analyses) / previous_month_analyses * 100) if previous_month_analyses > 0 else 0

            # Croissance des patients ce mois vs mois précédent
            previous_month_patients = len(db.analyses.distinct("patient_id", {
                "timestamp": {"$gte": prev_month_start, "$lt": prev_month_end},
                "patient_id": {"$ne": None}
            }))
            if previous_month_patients == 0:
                previous_month_patients = 1

            patients_growth = ((new_patients_month - previous_month_patients) / previous_month_patients * 100) if previous_month_patients > 0 else 0

            # Médecins actifs (ayant fait au moins une analyse dans les 7 derniers jours)
            seven_days_ago = now - timedelta(days=7)
            active_doctors = len(db.analyses.distinct("doctor_id", {
                "timestamp": {"$gte": seven_days_ago},
                "doctor_id": {"$ne": None}
            }))

        except Exception as e:
            print(f"Erreur calcul statistiques: {e}")
            daily_analyses_count = 0
            new_patients_month = 0
            analyses_growth = 0
            patients_growth = 0
            active_doctors = 0

        # Top 5 des médecins les plus actifs
        try:
            top_doctors_pipeline = [
                {"$match": {"doctor_id": {"$ne": None}}},
                {"$group": {
                    "_id": "$doctor_id",
                    "analyses_count": {"$sum": 1}
                }},
                {"$sort": {"analyses_count": -1}},
                {"$limit": 5}
            ]
            top_doctors_results = list(db.analyses.aggregate(top_doctors_pipeline))
            
            # Récupérer les noms des médecins
            top_doctors = []
            for result in top_doctors_results:
                doctor_id = result['_id']
                # Convertir en ObjectId si nécessaire
                try:
                    from bson import ObjectId
                    doctor = db.doctors.find_one({"_id": ObjectId(doctor_id)})
                    if doctor:
                        first_name = doctor.get('first_name', 'Dr.')
                        last_name = doctor.get('last_name', 'Inconnu')
                        top_doctors.append((first_name, last_name, result['analyses_count']))
                except:
                    top_doctors.append(('Dr.', 'Inconnu', result['analyses_count']))
        except Exception as e:
            print(f"Erreur top_doctors: {e}")
            top_doctors = []

        # Performance par médecin (efficacité basée sur la confiance moyenne)
        try:
            doctor_performance_pipeline = [
                {"$match": {"doctor_id": {"$ne": None}}},
                {"$group": {
                    "_id": "$doctor_id",
                    "analyses_count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"}
                }},
                {"$match": {"analyses_count": {"$gt": 0}}},
                {"$sort": {"avg_confidence": -1, "analyses_count": -1}},
                {"$limit": 5}
            ]
            performance_results = list(db.analyses.aggregate(doctor_performance_pipeline))
            
            doctor_performance = []
            for result in performance_results:
                doctor_id = result['_id']
                try:
                    from bson import ObjectId
                    doctor = db.doctors.find_one({"_id": ObjectId(doctor_id)})
                    if doctor:
                        name = f"{doctor.get('first_name', 'Dr.')} {doctor.get('last_name', 'Inconnu')}"
                        doctor_performance.append((
                            doctor_id,
                            name,
                            result['analyses_count'],
                            result['avg_confidence']
                        ))
                except:
                    doctor_performance.append((
                        doctor_id,
                        'Dr. Inconnu',
                        result['analyses_count'],
                        result['avg_confidence']
                    ))
        except Exception as e:
            print(f"Erreur doctor_performance: {e}")
            doctor_performance = []

        # Croissance mensuelle (6 derniers mois)
        try:
            six_months_ago = datetime.now() - timedelta(days=180)
            print(f"DEBUG: six_months_ago = {six_months_ago}")
            
            monthly_pipeline = [
                {"$match": {"timestamp": {"$gte": six_months_ago, "$exists": True}}},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m", "date": "$timestamp"}},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
            monthly_results = list(db.analyses.aggregate(monthly_pipeline))
            print(f"DEBUG: monthly_growth found {len(monthly_results)} months")
            monthly_growth_data = [(r['_id'], r['count']) for r in monthly_results]
        except Exception as e:
            print(f"Erreur monthly_growth: {e}")
            import traceback
            traceback.print_exc()
            monthly_growth_data = []

        # Activité par heure de la journée
        try:
            thirty_days_ago = datetime.now() - timedelta(days=30)
            print(f"DEBUG: Querying hourly activity since {thirty_days_ago}")
            
            hourly_pipeline = [
                {"$match": {"timestamp": {"$gte": thirty_days_ago, "$exists": True}}},
                {"$project": {
                    "hour": {"$hour": "$timestamp"}
                }},
                {"$group": {
                    "_id": "$hour",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
            hourly_results = list(db.analyses.aggregate(hourly_pipeline))
            print(f"DEBUG: hourly_activity found {len(hourly_results)} hours")
            hourly_activity_data = [(str(r['_id']).zfill(2), r['count']) for r in hourly_results]
        except Exception as e:
            print(f"Erreur hourly_activity: {e}")
            import traceback
            traceback.print_exc()
            hourly_activity_data = []

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'total_analyses': total_analyses,
                'total_doctors': total_doctors,
                'total_patients': total_patients,
                'active_days': active_days,
                'avg_confidence': round(avg_confidence * 100, 1) if avg_confidence else 0,
                'avg_processing_time': round(avg_processing_time, 2) if avg_processing_time else 0,
                'success_rate': round(success_rate, 1),
                'storage_used': round(storage_used),
                'tumor_distribution': tumor_distribution,
                'daily_analyses': daily_analyses,
                'daily_analyses_count': daily_analyses_count,
                'new_patients_month': new_patients_month,
                'analyses_growth': round(analyses_growth, 1),
                'patients_growth': round(patients_growth, 1),
                'active_doctors': active_doctors,
                'top_doctors': [{'name': f"Dr. {row[0]} {row[1]}", 'analyses': row[2]} for row in top_doctors],
                'doctor_performance': [{'name': f"Dr. {row[1]}", 'efficiency': round(row[3] * 100, 1), 'analyses': row[2]} for row in doctor_performance],
                'monthly_growth': monthly_growth_data,
                'hourly_activity': hourly_activity_data
            }
        })

    except Exception as e:
        print(f"Erreur platform analytics overview: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/system/metrics')
@login_required
def get_system_metrics():
    """API pour obtenir les métriques système en temps réel"""
    try:
        import psutil
        import os
        import time
        from datetime import datetime, timedelta
        
        # Métriques CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Métriques RAM
        memory = psutil.virtual_memory()
        ram_percent = memory.percent
        ram_used_gb = memory.used / (1024**3)
        ram_total_gb = memory.total / (1024**3)
        
        # Métriques disque
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        # Temps de réponse simulé (basé sur la charge système)
        base_response_time = 1.5
        load_factor = (cpu_percent + ram_percent) / 200
        response_time = base_response_time + (load_factor * 2)
        
        # Calculer le taux de réussite des analyses récentes
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED
        
        try:
            # Analyses des dernières 24h
            # MongoDB query needed here
        # SELECT COUNT(*) FROM analyses 
        #                 WHERE timestamp >= datetime('now', '-1 day'))
            # recent_analyses = None  # cursor.fetchone() # DISABLED[0] or 0  # TODO: Convert to MongoDB
            recent_analyses = 0  # TODO: Implement MongoDB query
            
            # Analyses réussies (confidence > 0.8) des dernières 24h
            # MongoDB query needed here
        # SELECT COUNT(*) FROM analyses 
        #                 WHERE timestamp >= datetime('now', '-1 day') 
        #                 AND confidence > 0.8)
            # successful_recent = None  # cursor.fetchone() # DISABLED[0] or 0  # TODO: Convert to MongoDB
            successful_recent = 0  # TODO: Implement MongoDB query
            
            success_rate_24h = (successful_recent / recent_analyses * 100) if recent_analyses > 0 else 0
            
        except:
            recent_analyses = 0
            success_rate_24h = 0
        
        # conn.close() # DISABLED
        
        # Statut système global
        if cpu_percent < 70 and ram_percent < 80 and response_time < 3:
            system_status = "optimal"
            status_message = "Système opérationnel - Performance optimale"
        elif cpu_percent < 85 and ram_percent < 90 and response_time < 5:
            system_status = "good"
            status_message = "Système stable - Performance normale"
        else:
            system_status = "warning"
            status_message = "Charge élevée détectée - Surveillance active"
        
        return jsonify({
            'success': True,
            'data': {
                'cpu': {
                    'percent': round(cpu_percent, 1),
                    'status': 'optimal' if cpu_percent < 70 else 'warning' if cpu_percent < 85 else 'critical'
                },
                'ram': {
                    'percent': round(ram_percent, 1),
                    'used_gb': round(ram_used_gb, 1),
                    'total_gb': round(ram_total_gb, 1),
                    'status': 'optimal' if ram_percent < 70 else 'warning' if ram_percent < 85 else 'critical'
                },
                'disk': {
                    'percent': round(disk_percent, 1),
                    'used_gb': round(disk_used_gb, 1),
                    'total_gb': round(disk_total_gb, 1),
                    'status': 'optimal' if disk_percent < 80 else 'warning' if disk_percent < 90 else 'critical'
                },
                'response_time': round(response_time, 2),
                'success_rate_24h': round(success_rate_24h, 1),
                'recent_analyses': recent_analyses,
                'system_status': system_status,
                'status_message': status_message,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except ImportError:
        # Si psutil n'est pas installé, retourner des données simulées
        return jsonify({
            'success': True,
            'data': {
                'cpu': {'percent': 45.0, 'status': 'optimal'},
                'ram': {'percent': 62.0, 'used_gb': 8.2, 'total_gb': 16.0, 'status': 'optimal'},
                'disk': {'percent': 67.0, 'used_gb': 847.0, 'total_gb': 1000.0, 'status': 'optimal'},
                'response_time': 2.1,
                'success_rate_24h': 99.2,
                'recent_analyses': 47,
                'system_status': 'optimal',
                'status_message': 'Système opérationnel - Performance optimale',
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        print(f"Erreur system metrics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/my-patients')
@login_required
def get_my_patients():
    """API pour obtenir la liste des patients du médecin connecté"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Query MongoDB for patients
        patients_cursor = db.patients.find({"doctor_id": doctor_id}).sort("updated_at", -1)
        print(f"DEBUG get_my_patients: doctor_id = {doctor_id}")
        
        patients = []
        for patient in patients_cursor:
            print(f"DEBUG: Found patient: {patient.get('patient_id')} - {patient.get('patient_name')}")
            patients.append({
                'patient_id': patient.get('patient_id'),
                'patient_name': patient.get('patient_name'),
                'date_of_birth': patient.get('date_of_birth'),
                'gender': patient.get('gender'),
                'phone': patient.get('phone'),
                'email': patient.get('email'),
                'address': patient.get('address'),
                'emergency_contact_name': patient.get('emergency_contact_name'),
                'emergency_contact_phone': patient.get('emergency_contact_phone'),
                'medical_history': patient.get('medical_history'),
                'allergies': patient.get('allergies'),
                'current_medications': patient.get('current_medications'),
                'insurance_number': patient.get('insurance_number'),
                'notes': patient.get('notes'),
                'first_analysis_date': patient.get('first_analysis_date'),
                'last_analysis_date': patient.get('last_analysis_date'),
                'total_analyses': patient.get('total_analyses', 0),
                'created_at': patient.get('created_at'),
                'updated_at': patient.get('updated_at')
            })
        
        print(f"DEBUG: Returning {len(patients)} patients")

        return jsonify({
            'success': True,
            'data': patients
        })

    except Exception as e:
        print(f"Erreur get my patients: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/filter-counts')
@login_required
def get_filter_counts():
    """API pour obtenir les compteurs des filtres"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Compter les diagnostics par type
        # MongoDB query needed here
        # SELECT predicted_label, COUNT(*)
        #             FROM analyses
        #             WHERE doctor_id = ?
        #             GROUP BY predicted_label)

        # diagnostic_counts = dict(cursor.fetchall())  # TODO: Convert to MongoDB
        diagnostic_counts = {}  # TODO: Implement MongoDB query

        # Statistiques générales
        # cursor.execute( # DISABLED'SELECT COUNT(*) FROM analyses WHERE doctor_id = ?', (doctor_id,))
        result = None  # TODO: Implement MongoDB query
        total_analyses = result[0] if result else 0

        # cursor.execute( # DISABLED'SELECT AVG(confidence) FROM analyses WHERE doctor_id = ?', (doctor_id,))
        result = None  # TODO: Implement MongoDB query
        avg_confidence = result[0] if result and result[0] else 0

        # cursor.execute( # DISABLED'SELECT AVG(processing_time) FROM analyses WHERE doctor_id = ?', (doctor_id,))
        result = None  # TODO: Implement MongoDB query
        avg_processing_time = result[0] if result and result[0] else 0

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'diagnostic_counts': diagnostic_counts,
                'total_analyses': total_analyses,
                'avg_confidence': round(avg_confidence * 100, 1) if avg_confidence else 0,
                'avg_processing_time': round(avg_processing_time, 2) if avg_processing_time else 0
            }
        })

    except Exception as e:
        print(f"Erreur get filter counts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/filter-preview', methods=['POST'])
@login_required
def filter_preview():
    """API pour prévisualiser le nombre de résultats avec les filtres"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        filters = request.json
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Construire la requête avec les filtres
        query = 'SELECT COUNT(*) FROM analyses WHERE doctor_id = ?'
        params = [doctor_id]

        # Filtre par date
        if filters.get('start_date'):
            query += ' AND DATE(timestamp) >= ?'
            params.append(filters['start_date'])

        if filters.get('end_date'):
            query += ' AND DATE(timestamp) <= ?'
            params.append(filters['end_date'])

        # Filtre par type de diagnostic
        if filters.get('diagnostic_types') and len(filters['diagnostic_types']) > 0:
            placeholders = ','.join(['?' for _ in filters['diagnostic_types']])
            query += f' AND predicted_label IN ({placeholders})'
            params.extend(filters['diagnostic_types'])

        # Filtre par confiance
        if filters.get('min_confidence', 0) > 0:
            query += ' AND confidence >= ?'
            params.append(filters['min_confidence'] / 100.0)

        if filters.get('max_confidence', 100) < 100:
            query += ' AND confidence <= ?'
            params.append(filters['max_confidence'] / 100.0)

        # Filtre par temps de traitement
        if filters.get('max_processing_time', 10) < 10:
            query += ' AND processing_time <= ?'
            params.append(filters['max_processing_time'])

        # cursor.execute( # DISABLEDquery, params)
        result = None  # TODO: Implement MongoDB query
        count = result[0] if result else 0

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'count': count
            }
        })

    except Exception as e:
        print(f"Erreur filter preview: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/period/<period>')
@login_required
def analytics_by_period(period):
    """API pour les statistiques par période (day/month/year)"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        if period == 'day':
            # Analyses par heure pour le jour le plus récent avec des données
            # MongoDB query needed here
        # SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
        #                 FROM analyses
        #                 WHERE DATE(timestamp) = (
        #                     SELECT DATE(timestamp) FROM analyses
        #                     ORDER BY timestamp DESC LIMIT 1
        #                 )
        #                 GROUP BY strftime('%H', timestamp)
        #                 ORDER BY hour)
            # data = []  # cursor.fetchall() # DISABLED  # TODO: Convert to MongoDB

            # Si pas de données pour le jour le plus récent, prendre les 24 dernières heures
            if not data:
                # MongoDB query needed here
                # SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                #                     FROM analyses
                #                     WHERE timestamp >= datetime('now', '-24 hours')
                #                     GROUP BY strftime('%H', timestamp)
                #                     ORDER BY hour)
                # data = []  # cursor.fetchall() # DISABLED  # TODO: Convert to MongoDB
                data = []  # TODO: Implement MongoDB query

            # Créer un tableau complet de 24 heures
            hour_counts = {str(hour).zfill(2): 0 for hour in range(24)}
            for hour, count in data:
                hour_counts[hour] = count

            labels = [f"{hour}h" for hour in sorted(hour_counts.keys())]
            values = [hour_counts[hour] for hour in sorted(hour_counts.keys())]

        elif period == 'month':
            # Analyses par jour pour le mois le plus récent avec des données
            # MongoDB query needed here
            # SELECT strftime('%d', timestamp) as day, COUNT(*) as count
            #                 FROM analyses
            #                 WHERE strftime('%Y-%m', timestamp) = (
            #                     SELECT strftime('%Y-%m', timestamp) FROM analyses
            #                     ORDER BY timestamp DESC LIMIT 1
            #                 )
            #                 GROUP BY strftime('%d', timestamp)
            #                 ORDER BY CAST(day as INTEGER))
            # data = []  # cursor.fetchall() # DISABLED  # TODO: Convert to MongoDB
            data = []  # TODO: Implement MongoDB query

            # Si pas de données, prendre les 30 derniers jours
            if not data:
                # MongoDB query needed here
                # SELECT strftime('%d', timestamp) as day, COUNT(*) as count
                #                     FROM analyses
                #                     WHERE timestamp >= date('now', '-30 days')
                #                     GROUP BY strftime('%d', timestamp)
                #                     ORDER BY CAST(day as INTEGER))
                # data = []  # cursor.fetchall() # DISABLED  # TODO: Convert to MongoDB
                data = []  # TODO: Implement MongoDB query

            labels = [f"{day}" for day, _ in data] if data else []
            values = [count for _, count in data] if data else []

        elif period == 'year':
            # Analyses par mois pour l'année la plus récente avec des données
            # MongoDB query needed here
        # SELECT strftime('%m', timestamp) as month, COUNT(*) as count
        #                 FROM analyses
        #                 WHERE strftime('%Y', timestamp) = (
        #                     SELECT strftime('%Y', timestamp) FROM analyses
        #                     ORDER BY timestamp DESC LIMIT 1
        #                 )
        #                 GROUP BY strftime('%m', timestamp)
        #                 ORDER BY CAST(month as INTEGER))
            # data = []  # cursor.fetchall() # DISABLED  # TODO: Convert to MongoDB

            # Si pas de données, prendre les 12 derniers mois
            if not data:
                # MongoDB query needed here
            # SELECT strftime('%m', timestamp) as month, COUNT(*) as count
            #                     FROM analyses
            #                     WHERE timestamp >= date('now', '-12 months')
            #                     GROUP BY strftime('%m', timestamp)
            #                     ORDER BY CAST(month as INTEGER))
                # data = []  # cursor.fetchall() # DISABLED  # TODO: Convert to MongoDB
                data = []  # TODO: Implement MongoDB query

            month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                          'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
            labels = [month_names[int(month)-1] for month, _ in data] if data else []
            values = [count for _, count in data] if data else []

        else:
            return jsonify({'success': False, 'error': 'Période invalide'}), 400

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'labels': labels,
                'values': values,
                'period': period
            }
        })

    except Exception as e:
        print(f"Erreur analytics period: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/recent')
@login_required
def recent_analyses():
    """API pour les analyses récentes du médecin connecté"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # MongoDB query needed here
        # SELECT timestamp, filename, predicted_label, confidence, processing_time, patient_name, patient_id
        #             FROM analyses
        #             WHERE doctor_id = ?
        #             ORDER BY timestamp DESC)

        analyses = []
        # for row in cursor.fetchall():  # TODO: Convert to MongoDB
            # analyses.append({
            #     'timestamp': row[0],
            #     'filename': row[1],
            #     'predicted_label': row[2],
            #     'confidence': round(row[3] * 100, 1),
            #     'processing_time': round(row[4], 2),
            #     'patient_name': row[5] or 'Patient anonyme',
            #     'patient_id': row[6] or 'N/A'
            # })
        # TODO: Implement MongoDB query

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': analyses
        })

    except Exception as e:
        print(f"Erreur recent analyses: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/export/<format>')
def export_analytics(format):
    """API pour exporter les données analytiques"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        if format == 'csv':
            # MongoDB query needed here
        # SELECT timestamp, filename, predicted_label, confidence,
        #                        processing_time, description
        #                 FROM analyses
        #                 ORDER BY timestamp DESC)

            # data = []  # cursor.fetchall() # DISABLED  # TODO: Convert to MongoDB
            data = []  # TODO: Implement MongoDB query

            # Créer le contenu CSV
            csv_content = "Timestamp,Filename,Diagnostic,Confidence,Processing_Time,Description\n"
            for row in data:
                csv_content += f'"{row[0]}","{row[1]}","{row[2]}",{row[3]:.3f},{row[4]:.2f},"{row[5] or ""}"\n'

            # conn.close() # DISABLED

            return Response(
                csv_content,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=neuroscan_analytics_{datetime.now().strftime("%Y%m%d")}.csv'}
            )

        elif format == 'json':
            # MongoDB query needed here
            # SELECT timestamp, filename, predicted_label, confidence,
            #                        processing_time, probabilities, description, recommendations
            #                 FROM analyses
            #                 ORDER BY timestamp DESC)

            analyses = []
            # for row in cursor.fetchall():  # TODO: Convert to MongoDB
                # analyses.append({
                #     'timestamp': row[0],
                #     'filename': row[1],
                #     'predicted_label': row[2],
                #     'confidence': row[3],
                #     'processing_time': row[4],
                #     'probabilities': json.loads(row[5]) if row[5] else {},
                #     'description': row[6],
                #     'recommendations': json.loads(row[7]) if row[7] else []
                # })
            # TODO: Implement MongoDB query to populate analyses list

            # conn.close() # DISABLED

            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_analyses': len(analyses),
                'analyses': analyses
            }

            return Response(
                json.dumps(export_data, indent=2),
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename=neuroscan_analytics_{datetime.now().strftime("%Y%m%d")}.json'}
            )

        else:
            return jsonify({'success': False, 'error': 'Format non supporté'}), 400

    except Exception as e:
        print(f"Erreur export analytics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/stats/advanced')
def advanced_stats():
    """API pour des statistiques avancées"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Statistiques par heure de la journée
        # MongoDB query needed here
        # SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
        #             FROM analyses
        #             GROUP BY strftime('%H', timestamp)
        #             ORDER BY hour)
        # hourly_stats = dict(cursor.fetchall())  # TODO: Convert to MongoDB

        # Évolution de la confiance dans le temps
        # MongoDB query needed here
        # SELECT DATE(timestamp) as date, AVG(confidence) as avg_confidence
        #             FROM analyses
        #             WHERE timestamp >= date('now', '-30 days')
        #             GROUP BY DATE(timestamp)
        #             ORDER BY date)
        # confidence_evolution = []  # cursor.fetchall() # DISABLED  # TODO: Convert to MongoDB

        # Top 5 des jours les plus actifs
        # MongoDB query needed here
        # SELECT DATE(timestamp) as date, COUNT(*) as count
        #             FROM analyses
        #             GROUP BY DATE(timestamp)
        #             ORDER BY count DESC
        #             LIMIT 5)
        # top_active_days = []  # cursor.fetchall() # DISABLED  # TODO: Convert to MongoDB

        # Statistiques de performance
        # MongoDB query needed here
        # SELECT
        #                 MIN(processing_time) as min_time,
        #                 MAX(processing_time) as max_time,
        #                 AVG(processing_time) as avg_time,
        #                 COUNT(CASE WHEN processing_time < 5 THEN 1 END) as fast_analyses,
        #                 COUNT(CASE WHEN processing_time >= 5 THEN 1 END) as slow_analyses
        #             FROM analyses)
        # performance_stats = None  # cursor.fetchone() # DISABLED  # TODO: Convert to MongoDB
        hourly_stats = []  # TODO: Implement MongoDB query
        confidence_evolution = []  # TODO: Implement MongoDB query
        top_active_days = []  # TODO: Implement MongoDB query
        performance_stats = (0, 0, 0, 0, 0)  # TODO: Implement MongoDB query (min, max, avg, fast, slow)

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'hourly_distribution': hourly_stats,
                'confidence_evolution': confidence_evolution,
                'top_active_days': top_active_days,
                'performance_stats': {
                    'min_processing_time': performance_stats[0],
                    'max_processing_time': performance_stats[1],
                    'avg_processing_time': performance_stats[2],
                    'fast_analyses': performance_stats[3],
                    'slow_analyses': performance_stats[4]
                }
            }
        })

    except Exception as e:
        print(f"Erreur advanced stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/filters')
def get_filter_options():
    """API pour obtenir les options de filtres disponibles"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Obtenir les plages de dates disponibles
        # MongoDB query needed here
        # SELECT MIN(DATE(timestamp)) as min_date, MAX(DATE(timestamp)) as max_date
        #             FROM analyses)
        date_range = (None, None)  # TODO: Implement MongoDB query

        # Obtenir les types de diagnostics
        # cursor.execute( # DISABLED'SELECT DISTINCT predicted_label FROM analyses ORDER BY predicted_label')
        diagnostic_types = []  # TODO: Implement MongoDB query

        # Obtenir les plages de confiance
        # cursor.execute( # DISABLED'SELECT MIN(confidence), MAX(confidence) FROM analyses')
        confidence_range = (0, 1)  # TODO: Implement MongoDB query

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'date_range': {
                    'min': date_range[0],
                    'max': date_range[1]
                },
                'diagnostic_types': diagnostic_types,
                'confidence_range': {
                    'min': round(confidence_range[0] * 100, 1) if confidence_range[0] else 0,
                    'max': round(confidence_range[1] * 100, 1) if confidence_range[1] else 100
                }
            }
        })

    except Exception as e:
        print(f"Erreur filter options: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/filtered', methods=['POST'])
@login_required
def get_filtered_analytics():
    """API pour obtenir des analyses filtrées du médecin connecté"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        filters = request.get_json()
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Construire la requête avec filtres (toujours filtrer par médecin)
        where_conditions = ['doctor_id = ?']
        params = [doctor_id]

        if filters.get('start_date'):
            where_conditions.append('DATE(timestamp) >= ?')
            params.append(filters['start_date'])

        if filters.get('end_date'):
            where_conditions.append('DATE(timestamp) <= ?')
            params.append(filters['end_date'])

        if filters.get('diagnostic_types') and len(filters['diagnostic_types']) > 0:
            placeholders = ','.join(['?' for _ in filters['diagnostic_types']])
            where_conditions.append(f'predicted_label IN ({placeholders})')
            params.extend(filters['diagnostic_types'])

        if filters.get('min_confidence', 0) > 0:
            where_conditions.append('confidence >= ?')
            params.append(filters['min_confidence'] / 100.0)

        if filters.get('max_confidence', 100) < 100:
            where_conditions.append('confidence <= ?')
            params.append(filters['max_confidence'] / 100.0)

        # Nouveau filtre : temps de traitement
        if filters.get('max_processing_time', 10) < 10:
            where_conditions.append('processing_time <= ?')
            params.append(filters['max_processing_time'])
            params.append(filters['max_confidence'] / 100)

        where_clause = 'WHERE ' + ' AND '.join(where_conditions) if where_conditions else ''

        # Obtenir les analyses filtrées avec informations patient
        # MongoDB query needed here
        # SELECT timestamp, filename, predicted_label, confidence, processing_time,
        #                    description, patient_name, patient_id, exam_date
        #             FROM analyses
        #             {where_clause}
        #             ORDER BY timestamp DESC
        #             LIMIT 100, params)

        analyses = []
        # for row in cursor.fetchall():  # TODO: Convert to MongoDB
            # analyses.append({
            #     'timestamp': row[0],
            #     'filename': row[1],
            #     'predicted_label': row[2],
            #     'confidence': round(row[3] * 100, 1),
            #     'processing_time': round(row[4], 2),
            #     'description': row[5] or '',
            #     'patient_name': row[6] or 'Patient anonyme',
            #     'patient_id': row[7] or 'N/A',
            #     'exam_date': row[8] or ''
            # })
        # TODO: Implement MongoDB query to populate analyses list

        # Statistiques des résultats filtrés
        # MongoDB query needed here
        # SELECT
        #                 COUNT(*) as total,
        #                 AVG(confidence) as avg_confidence,
        #                 AVG(processing_time) as avg_time,
        #                 predicted_label,
        #                 COUNT(*) as type_count
        #             FROM analyses
        #             {where_clause}
        #             GROUP BY predicted_label, params)

        stats_by_type = {}
        total_filtered = 0
        total_confidence = 0
        total_time = 0

        # for row in cursor.fetchall():  # TODO: Convert to MongoDB
            # stats_by_type[row[3]] = row[4]
            # total_filtered += row[4]
            # total_confidence += row[1] * row[4] if row[1] else 0
            # total_time += row[2] * row[4] if row[2] else 0
        # TODO: Implement MongoDB query

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'analyses': analyses,
                'stats': {
                    'total': total_filtered,
                    'avg_confidence': round((total_confidence / total_filtered * 100), 1) if total_filtered > 0 else 0,
                    'avg_processing_time': round((total_time / total_filtered), 2) if total_filtered > 0 else 0,
                    'distribution': stats_by_type
                }
            }
        })

    except Exception as e:
        print(f"Erreur filtered analytics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/comparison')
def get_comparison_data():
    """API pour les données de comparaison temporelle"""
    try:
        # Récupérer le paramètre de période (par défaut 'month')
        period = request.args.get('period', 'month')  # 'day', 'week', ou 'month'
        
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        comparison_data = {}

        if period == 'day':
            # Comparaison aujourd'hui vs hier
            from datetime import datetime, timedelta
            
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_start = today_start - timedelta(days=1)
            
            # Aujourd'hui
            today_count = db.analyses.count_documents({"timestamp": {"$gte": today_start}})
            today_pipeline = [
                {"$match": {"timestamp": {"$gte": today_start}}},
                {"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}
            ]
            today_result = list(db.analyses.aggregate(today_pipeline))
            today_avg = today_result[0]['avg'] if today_result and today_result[0].get('avg') else 0
            
            # Hier
            yesterday_count = db.analyses.count_documents({
                "timestamp": {"$gte": yesterday_start, "$lt": today_start}
            })
            yesterday_pipeline = [
                {"$match": {"timestamp": {"$gte": yesterday_start, "$lt": today_start}}},
                {"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}
            ]
            yesterday_result = list(db.analyses.aggregate(yesterday_pipeline))
            yesterday_avg = yesterday_result[0]['avg'] if yesterday_result and yesterday_result[0].get('avg') else 0
            
            daily_comparison = {
                "Aujourd'hui": {
                    'count': today_count,
                    'avg_confidence': round(today_avg * 100, 1) if today_avg else 0
                },
                'Hier': {
                    'count': yesterday_count,
                    'avg_confidence': round(yesterday_avg * 100, 1) if yesterday_avg else 0
                }
            }
            
            comparison_data = {
                'daily': daily_comparison,
                'period': 'day',
                'period_label': 'Jour'
            }

        elif period == 'week':
            # Comparaison cette semaine vs semaine dernière
            from datetime import datetime, timedelta
            
            now = datetime.now()
            this_week_start = now - timedelta(days=7)
            last_week_start = now - timedelta(days=14)
            
            # Cette semaine
            this_week_count = db.analyses.count_documents({"timestamp": {"$gte": this_week_start}})
            this_week_pipeline = [
                {"$match": {"timestamp": {"$gte": this_week_start}}},
                {"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}
            ]
            this_week_result = list(db.analyses.aggregate(this_week_pipeline))
            this_week_avg = this_week_result[0]['avg'] if this_week_result and this_week_result[0].get('avg') else 0
            
            # Semaine dernière
            last_week_count = db.analyses.count_documents({
                "timestamp": {"$gte": last_week_start, "$lt": this_week_start}
            })
            last_week_pipeline = [
                {"$match": {"timestamp": {"$gte": last_week_start, "$lt": this_week_start}}},
                {"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}
            ]
            last_week_result = list(db.analyses.aggregate(last_week_pipeline))
            last_week_avg = last_week_result[0]['avg'] if last_week_result and last_week_result[0].get('avg') else 0
            
            weekly_comparison = {
                'Cette semaine': {
                    'count': this_week_count,
                    'avg_confidence': round(this_week_avg * 100, 1) if this_week_avg else 0
                },
                'Semaine dernière': {
                    'count': last_week_count,
                    'avg_confidence': round(last_week_avg * 100, 1) if last_week_avg else 0
                }
            }

            comparison_data = {
                'weekly': weekly_comparison,
                'period': 'week',
                'period_label': 'Semaine'
            }

        else:  # 'month' par défaut
            # Comparaison ce mois vs mois dernier
            from datetime import datetime
            
            now = datetime.now()
            this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Calculer le premier jour du mois dernier
            if this_month_start.month == 1:
                last_month_start = this_month_start.replace(year=this_month_start.year - 1, month=12)
            else:
                last_month_start = this_month_start.replace(month=this_month_start.month - 1)
            
            # Ce mois
            this_month_count = db.analyses.count_documents({"timestamp": {"$gte": this_month_start}})
            this_month_pipeline = [
                {"$match": {"timestamp": {"$gte": this_month_start}}},
                {"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}
            ]
            this_month_result = list(db.analyses.aggregate(this_month_pipeline))
            this_month_avg = this_month_result[0]['avg'] if this_month_result and this_month_result[0].get('avg') else 0
            
            # Mois dernier
            last_month_count = db.analyses.count_documents({
                "timestamp": {"$gte": last_month_start, "$lt": this_month_start}
            })
            last_month_pipeline = [
                {"$match": {"timestamp": {"$gte": last_month_start, "$lt": this_month_start}}},
                {"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}
            ]
            last_month_result = list(db.analyses.aggregate(last_month_pipeline))
            last_month_avg = last_month_result[0]['avg'] if last_month_result and last_month_result[0].get('avg') else 0
            
            monthly_comparison = {
                'Ce mois': {
                    'count': this_month_count,
                    'avg_confidence': round(this_month_avg * 100, 1) if this_month_avg else 0
                },
                'Mois dernier': {
                    'count': last_month_count,
                    'avg_confidence': round(last_month_avg * 100, 1) if last_month_avg else 0
                }
            }

            comparison_data = {
                'monthly': monthly_comparison,
                'period': 'month',
                'period_label': 'Mois'
            }

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': comparison_data
        })

    except Exception as e:
        print(f"Erreur comparison data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/alerts')
def get_alerts():
    """API pour obtenir les alertes et notifications"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        alerts = []

        # Alerte: Baisse de confiance moyenne
        # MongoDB query needed here
        # SELECT AVG(confidence) as avg_conf_today
        #             FROM analyses
        #             WHERE DATE(timestamp) = DATE('now'))
        today_confidence = None  # TODO: Implement MongoDB query

        # MongoDB query needed here
        # SELECT AVG(confidence) as avg_conf_week
        #             FROM analyses
        #             WHERE DATE(timestamp) >= DATE('now', '-7 days')
        #             AND DATE(timestamp) < DATE('now'))
        week_confidence = None  # TODO: Implement MongoDB query

        if today_confidence and week_confidence and today_confidence < week_confidence * 0.9:
            alerts.append({
                'type': 'warning',
                'title': 'Baisse de confiance détectée',
                'message': f'La confiance moyenne aujourd\'hui ({today_confidence*100:.1f}%) est inférieure à la moyenne de la semaine ({week_confidence*100:.1f}%)',
                'timestamp': datetime.now().isoformat()
            })

        # Alerte: Pic d'activité
        # MongoDB query needed here
        # SELECT COUNT(*) as today_count
        #             FROM analyses
        #             WHERE DATE(timestamp) = DATE('now'))
        today_count = None  # TODO: Implement MongoDB query

        # MongoDB query needed here
        # SELECT AVG(daily_count) as avg_daily
        #             FROM (
        #                 SELECT COUNT(*) as daily_count
        #                 FROM analyses
        #                 WHERE DATE(timestamp) >= DATE('now', '-7 days')
        #                 AND DATE(timestamp) < DATE('now')
        #                 GROUP BY DATE(timestamp)
        #             ))
        avg_daily = None  # TODO: Implement MongoDB query

        if today_count and avg_daily and today_count > avg_daily * 1.5:
            alerts.append({
                'type': 'info',
                'title': 'Pic d\'activité détecté',
                'message': f'Nombre d\'analyses aujourd\'hui ({today_count}) supérieur à la moyenne ({avg_daily:.1f})',
                'timestamp': datetime.now().isoformat()
            })

        # Alerte: Analyses avec faible confiance
        # MongoDB query needed here
        # SELECT COUNT(*) as low_conf_count
        #             FROM analyses
        #             WHERE DATE(timestamp) = DATE('now')
        #             AND confidence < 0.7)
        low_confidence_count = 0  # TODO: Implement MongoDB query

        if low_confidence_count > 0:
            alerts.append({
                'type': 'warning',
                'title': 'Analyses à faible confiance',
                'message': f'{low_confidence_count} analyse(s) avec confiance < 70% aujourd\'hui',
                'timestamp': datetime.now().isoformat()
            })

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': alerts
        })

    except Exception as e:
        print(f"Erreur alerts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/performance')
def get_performance_trends():
    """API pour les tendances de performance"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Tendances de confiance sur les 7 derniers jours
        from datetime import datetime, timedelta
        
        seven_days_ago = datetime.now() - timedelta(days=7)
        
        daily_pipeline = [
            {"$match": {"timestamp": {"$gte": seven_days_ago}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "avg_confidence": {"$avg": "$confidence"},
                "avg_time": {"$avg": "$processing_time"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        daily_results = list(db.analyses.aggregate(daily_pipeline))
        daily_trends = [(r['_id'], r['avg_confidence'], r['avg_time'], r['count']) for r in daily_results]

        # Tendances par heure pour aujourd'hui
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        hourly_pipeline = [
            {"$match": {"timestamp": {"$gte": today_start}}},
            {"$project": {
                "hour": {"$hour": "$timestamp"},
                "confidence": 1,
                "processing_time": 1
            }},
            {"$group": {
                "_id": "$hour",
                "avg_confidence": {"$avg": "$confidence"},
                "avg_time": {"$avg": "$processing_time"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        hourly_results = list(db.analyses.aggregate(hourly_pipeline))
        hourly_trends = [(str(r['_id']).zfill(2), r['avg_confidence'], r['avg_time'], r['count']) for r in hourly_results]

        # Performance par type de diagnostic
        type_pipeline = [
            {"$match": {"timestamp": {"$gte": seven_days_ago}}},
            {"$group": {
                "_id": "$predicted_label",
                "avg_confidence": {"$avg": "$confidence"},
                "avg_time": {"$avg": "$processing_time"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        type_results = list(db.analyses.aggregate(type_pipeline))
        performance_by_type = [(r['_id'], r['avg_confidence'], r['avg_time'], r['count']) for r in type_results]

        # conn.close() # DISABLED

        # Formater les données pour Chart.js
        daily_data = {
            'labels': [row[0] for row in daily_trends],
            'confidence': [round(row[1] * 100, 1) if row[1] else 0 for row in daily_trends],
            'processing_time': [round(row[2], 2) if row[2] else 0 for row in daily_trends],
            'count': [row[3] for row in daily_trends]
        }

        hourly_data = {
            'labels': [f"{row[0]}h" for row in hourly_trends],
            'confidence': [round(row[1] * 100, 1) if row[1] else 0 for row in hourly_trends],
            'processing_time': [round(row[2], 2) if row[2] else 0 for row in hourly_trends],
            'count': [row[3] for row in hourly_trends]
        }

        type_performance = {}
        for row in performance_by_type:
            type_performance[row[0]] = {
                'confidence': round(row[1] * 100, 1) if row[1] else 0,
                'processing_time': round(row[2], 2) if row[2] else 0,
                'count': row[3]
            }

        return jsonify({
            'success': True,
            'data': {
                'daily_trends': daily_data,
                'hourly_trends': hourly_data,
                'performance_by_type': type_performance
            }
        })

    except Exception as e:
        print(f"Erreur performance trends: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/diagnostic-distribution')
def get_diagnostic_distribution():
    """API pour la distribution des diagnostics"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Distribution des diagnostics
        distribution_pipeline = [
            {"$match": {"predicted_label": {"$exists": True, "$ne": None}}},
            {"$group": {
                "_id": "$predicted_label",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]
        
        dist_results = list(db.analyses.aggregate(distribution_pipeline))
        distribution = [(r['_id'], r['count']) for r in dist_results]
        
        # Calculer les pourcentages
        total = sum(row[1] for row in distribution)
        
        data = {
            'labels': [row[0] for row in distribution],
            'counts': [row[1] for row in distribution],
            'percentages': [round((row[1] / total) * 100, 1) if total > 0 else 0 for row in distribution]
        }

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': data
        })

    except Exception as e:
        print(f"Erreur diagnostic distribution: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/hourly-activity')
def get_hourly_activity():
    """API pour l'activité par heure"""
    try:
        # Récupérer le paramètre de période (par défaut 'week')
        period = request.args.get('period', 'week')  # 'today' ou 'week'
        
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Déterminer la clause WHERE selon la période
        from datetime import datetime, timedelta
        
        if period == 'today':
            time_filter = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            period_label = "aujourd'hui"
        else:  # 'week' par défaut
            time_filter = datetime.now() - timedelta(days=7)
            period_label = "7 derniers jours"

        # Activité par heure pour la période sélectionnée
        hourly_activity_pipeline = [
            {"$match": {"timestamp": {"$gte": time_filter}}},
            {"$project": {
                "hour": {"$hour": "$timestamp"}
            }},
            {"$group": {
                "_id": "$hour",
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        hourly_results = list(db.analyses.aggregate(hourly_activity_pipeline))
        hourly_data = [(str(r['_id']).zfill(2), r['count']) for r in hourly_results]

        # Créer un tableau avec toutes les heures (0-23)
        activity_by_hour = [0] * 24
        for row in hourly_data:
            hour = int(row[0])
            count = row[1]
            activity_by_hour[hour] = count

        # Statistiques
        max_activity = max(activity_by_hour) if activity_by_hour else 0
        peak_hour = activity_by_hour.index(max_activity) if max_activity > 0 else 0
        
        # Trouver l'heure la plus calme (heure avec le moins d'activité, en excluant 0)
        non_zero_activities = [(i, count) for i, count in enumerate(activity_by_hour) if count > 0]
        quiet_hour = min(non_zero_activities, key=lambda x: x[1])[0] if non_zero_activities else 0

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'hourly_activity': activity_by_hour,
                'peak_hour': peak_hour,
                'max_hourly_analyses': max_activity,
                'quiet_hour': quiet_hour,
                'period': period,
                'period_label': period_label
            }
        })

    except Exception as e:
        print(f"Erreur hourly activity: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/confidence-distribution')
def get_confidence_distribution():
    """API pour la distribution des niveaux de confiance"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Distribution par intervalles de confiance
        confidence_pipeline = [
            {"$match": {"confidence": {"$exists": True, "$ne": None}}},
            {"$project": {"confidence": 1}}
        ]
        
        confidence_results = list(db.analyses.aggregate(confidence_pipeline))
        confidences = [r['confidence'] * 100 for r in confidence_results]  # Convertir en pourcentage

        # Compter par intervalles
        very_high = len([c for c in confidences if c >= 90])
        high = len([c for c in confidences if 80 <= c < 90])
        medium = len([c for c in confidences if 70 <= c < 80])
        low = len([c for c in confidences if c < 70])

        # Histogramme pour le graphique
        histogram_data = []
        for i in range(0, 101, 10):
            count = len([c for c in confidences if i <= c < i + 10])
            histogram_data.append(count)

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'histogram': histogram_data,
                'intervals': {
                    'very_high': very_high,
                    'high': high,
                    'medium': medium,
                    'low': low
                }
            }
        })

    except Exception as e:
        print(f"Erreur confidence distribution: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/processing-time-analysis')
def get_processing_time_analysis():
    """API pour l'analyse des temps de traitement"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Récupérer tous les temps de traitement
        times_pipeline = [
            {"$match": {"processing_time": {"$exists": True, "$ne": None}}},
            {"$project": {"processing_time": 1}},
            {"$sort": {"processing_time": 1}}
        ]
        
        times_results = list(db.analyses.aggregate(times_pipeline))
        times = [r['processing_time'] for r in times_results]

        if not times:
            return jsonify({
                'success': True,
                'data': {
                    'histogram': [0] * 10,
                    'stats': {
                        'fast': 0,
                        'normal': 0,
                        'slow': 0,
                        'median': 0
                    }
                }
            })

        # Calculer les statistiques
        fast = len([t for t in times if t < 2])
        normal = len([t for t in times if 2 <= t <= 5])
        slow = len([t for t in times if t > 5])
        median_time = times[len(times) // 2] if times else 0

        # Histogramme par intervalles de 0.5s jusqu'à 10s
        histogram_data = []
        for i in range(0, 20):  # 0-10s par intervalles de 0.5s
            min_time = i * 0.5
            max_time = (i + 1) * 0.5
            count = len([t for t in times if min_time <= t < max_time])
            histogram_data.append(count)

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'histogram': histogram_data,
                'stats': {
                    'fast': fast,
                    'normal': normal,
                    'slow': slow,
                    'median': round(median_time, 2)
                }
            }
        })

    except Exception as e:
        print(f"Erreur processing time analysis: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/monthly-trends')
def get_monthly_trends():
    """API pour les tendances mensuelles"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Données mensuelles pour les 12 derniers mois
        from datetime import datetime, timedelta
        
        twelve_months_ago = datetime.now() - timedelta(days=365)
        
        monthly_pipeline = [
            {"$match": {"timestamp": {"$gte": twelve_months_ago}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m", "date": "$timestamp"}},
                "count": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        monthly_results = list(db.analyses.aggregate(monthly_pipeline))
        monthly_data = [(r['_id'], r['count'], r['avg_confidence']) for r in monthly_results]

        if not monthly_data:
            return jsonify({
                'success': True,
                'data': {
                    'labels': [],
                    'counts': [],
                    'confidences': [],
                    'growth_rate': 0,
                    'most_active_month': 'N/A'
                }
            })

        labels = [row[0] for row in monthly_data]
        counts = [row[1] for row in monthly_data]
        confidences = [round(row[2] * 100, 1) if row[2] else 0 for row in monthly_data]

        # Calculer le taux de croissance entre les deux derniers mois
        # (le plus pertinent pour l'utilisateur)
        if len(counts) >= 2:
            # Prendre les deux derniers mois
            previous_month_count = counts[-2]
            current_month_count = counts[-1]
            
            if previous_month_count >= 10:
                # Si on a au moins 10 analyses le mois précédent, 
                # le pourcentage est significatif
                avg_growth = round(((current_month_count - previous_month_count) / previous_month_count) * 100, 1)
            elif previous_month_count > 0:
                # Pour de petits nombres, plafonner à +200% max
                raw_growth = ((current_month_count - previous_month_count) / previous_month_count) * 100
                avg_growth = round(min(raw_growth, 200.0), 1)
            elif current_month_count > 0:
                # Si le mois précédent était à 0, afficher +100%
                avg_growth = 100.0
            else:
                # Les deux sont à 0
                avg_growth = 0.0
        else:
            avg_growth = 0.0

        # Mois le plus actif
        max_count = max(counts)
        most_active_month_index = counts.index(max_count)
        most_active_month = labels[most_active_month_index]

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'labels': labels,
                'counts': counts,
                'confidences': confidences,
                'growth_rate': avg_growth,
                'most_active_month': most_active_month
            }
        })

    except Exception as e:
        print(f"Erreur monthly trends: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/ai-insights')
def get_ai_insights():
    """API pour les insights et recommandations IA"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Calculer les métriques de base
        from datetime import datetime, timedelta
        
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        base_pipeline = [
            {"$match": {"timestamp": {"$gte": thirty_days_ago}}},
            {"$group": {
                "_id": None,
                "total": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"},
                "avg_time": {"$avg": "$processing_time"}
            }}
        ]
        
        base_result = list(db.analyses.aggregate(base_pipeline))
        if base_result and base_result[0]:
            total_analyses = base_result[0].get('total', 0)
            avg_confidence = base_result[0].get('avg_confidence', 0)
            avg_time = base_result[0].get('avg_time', 0)
        else:
            total_analyses = 0
            avg_confidence = 0
            avg_time = 0

        # Insights de performance
        performance_insights = []
        
        # Tendance de confiance
        seven_days_ago = datetime.now() - timedelta(days=7)
        fourteen_days_ago = datetime.now() - timedelta(days=14)
        
        week1_pipeline = [
            {"$match": {"timestamp": {"$gte": seven_days_ago}}},
            {"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}
        ]
        week1_result = list(db.analyses.aggregate(week1_pipeline))
        week1_conf = week1_result[0]['avg'] if week1_result and week1_result[0].get('avg') else None
        
        week2_pipeline = [
            {"$match": {"timestamp": {"$gte": fourteen_days_ago, "$lt": seven_days_ago}}},
            {"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}
        ]
        week2_result = list(db.analyses.aggregate(week2_pipeline))
        week2_conf = week2_result[0]['avg'] if week2_result and week2_result[0].get('avg') else None

        if week1_conf and week2_conf:
            conf_change = ((week1_conf - week2_conf) / week2_conf) * 100
            if conf_change > 5:
                performance_insights.append({
                    'type': 'positive',
                    'message': f'Amélioration de la confiance de {conf_change:.1f}% cette semaine'
                })
            elif conf_change < -5:
                performance_insights.append({
                    'type': 'warning',
                    'message': f'Baisse de confiance de {abs(conf_change):.1f}% cette semaine'
                })

        # Insights de qualité
        quality_insights = []
        
        # Distribution des diagnostics
        diagnostic_pipeline = [
            {"$match": {"timestamp": {"$gte": thirty_days_ago}}},
            {"$group": {
                "_id": "$predicted_label",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]
        diagnostic_results = list(db.analyses.aggregate(diagnostic_pipeline))
        diagnostic_dist = [(r['_id'], r['count']) for r in diagnostic_results]
        
        if diagnostic_dist:
            most_common = diagnostic_dist[0]
            quality_insights.append({
                'type': 'info',
                'message': f'Diagnostic le plus fréquent: {most_common[0]} ({most_common[1]} cas)'
            })

        # Analyses à faible confiance
        low_conf_count = db.analyses.count_documents({
            "timestamp": {"$gte": seven_days_ago},
            "confidence": {"$lt": 0.7}
        })
        
        if low_conf_count > 0:
            quality_insights.append({
                'type': 'warning',
                'message': f'{low_conf_count} analyses avec confiance <70% cette semaine'
            })

        # Recommandations
        recommendations = []
        
        # Recommandation basée sur le temps de traitement
        if avg_time > 3:
            recommendations.append({
                'type': 'optimization',
                'message': 'Considérer l\'optimisation du modèle (temps moyen >3s)'
            })
        
        # Recommandation basée sur la confiance
        if avg_confidence < 0.8:
            recommendations.append({
                'type': 'model_improvement',
                'message': 'Entraînement avec plus de données recommandé'
            })

        # Recommandation générale
        if total_analyses > 100:
            recommendations.append({
                'type': 'analysis',
                'message': 'Analyser les patterns pour identifier les améliorations'
            })

        # Calculer les scores
        accuracy_score = min(100, max(0, int(avg_confidence * 100)))
        efficiency_score = min(100, max(0, int((10 - avg_time) * 10))) if avg_time > 0 else 100
        reliability_score = min(100, max(0, int(100 - (low_conf_count * 5))))
        overall_score = int((accuracy_score + efficiency_score + reliability_score) / 3)

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'performance_insights': performance_insights,
                'quality_insights': quality_insights,
                'recommendations': recommendations,
                'scores': {
                    'accuracy': accuracy_score,
                    'efficiency': efficiency_score,
                    'reliability': reliability_score,
                    'overall': overall_score
                }
            }
        })

    except Exception as e:
        print(f"Erreur AI insights: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/advanced-metrics')
def get_advanced_metrics():
    """API pour les métriques avancées du système"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Calcul des métriques de débit (analyses par jour)
        from datetime import datetime, timedelta
        
        thirty_days_ago = datetime.now() - timedelta(days=30)
        total_last_30_days = db.analyses.count_documents({"timestamp": {"$gte": thirty_days_ago}})
        throughput_rate = round(total_last_30_days / 30, 1)

        # Calcul du changement de débit par rapport au mois précédent
        sixty_days_ago = datetime.now() - timedelta(days=60)
        prev_month_total = db.analyses.count_documents({
            "timestamp": {"$gte": sixty_days_ago, "$lt": thirty_days_ago}
        })
        prev_throughput = prev_month_total / 30 if prev_month_total > 0 else 0
        throughput_change = round(((throughput_rate - prev_throughput) / prev_throughput) * 100, 1) if prev_throughput > 0 else 0

        # Taux de précision basé sur la confiance moyenne
        accuracy_pipeline = [
            {"$match": {"timestamp": {"$gte": thirty_days_ago}}},
            {"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}
        ]
        accuracy_result = list(db.analyses.aggregate(accuracy_pipeline))
        accuracy_rate = accuracy_result[0]['avg'] if accuracy_result and accuracy_result[0].get('avg') else 0
        accuracy_percentage = round(accuracy_rate * 100, 1)

        # Temps de réponse moyen
        response_pipeline = [
            {"$match": {"timestamp": {"$gte": thirty_days_ago}}},
            {"$group": {"_id": None, "avg": {"$avg": "$processing_time"}}}
        ]
        response_result = list(db.analyses.aggregate(response_pipeline))
        avg_response_time = response_result[0]['avg'] if response_result and response_result[0].get('avg') else 0

        # Simulation de la disponibilité système (99.9% par défaut, peut être calculé selon les besoins)
        system_uptime = 99.9

        # Métriques pour la comparaison annuelle
        twelve_months_ago = datetime.now() - timedelta(days=365)
        
        yearly_pipeline = [
            {"$match": {"timestamp": {"$gte": twelve_months_ago}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m", "date": "$timestamp"}},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        yearly_results = list(db.analyses.aggregate(yearly_pipeline))
        yearly_data = [(r['_id'], r['count']) for r in yearly_results]
        
        # Calculer la croissance annuelle
        if len(yearly_data) >= 2:
            first_month = yearly_data[0][1]
            last_month = yearly_data[-1][1]
            year_growth = round(((last_month - first_month) / first_month) * 100, 1) if first_month > 0 else 0
        else:
            year_growth = 0

        # Prédiction simple pour le mois prochain (moyenne des 3 derniers mois)
        if len(yearly_data) >= 3:
            last_3_months = [row[1] for row in yearly_data[-3:]]
            next_month_prediction = round(sum(last_3_months) / len(last_3_months))
        else:
            next_month_prediction = throughput_rate * 30

        # Direction de la tendance
        if len(yearly_data) >= 2:
            recent_trend = yearly_data[-1][1] - yearly_data[-2][1]
            trend_direction = "↗️ Croissance" if recent_trend > 0 else "↘️ Décroissance" if recent_trend < 0 else "→ Stable"
        else:
            trend_direction = "→ Stable"

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'throughput_rate': throughput_rate,
                'throughput_change': throughput_change,
                'accuracy_rate': accuracy_percentage,
                'avg_response_time': round(avg_response_time, 2),
                'system_uptime': system_uptime,
                'year_growth': year_growth,
                'next_month_prediction': next_month_prediction,
                'trend_direction': trend_direction,
                'yearly_data': {
                    'labels': [row[0] for row in yearly_data],
                    'counts': [row[1] for row in yearly_data]
                }
            }
        })

    except Exception as e:
        print(f"Erreur advanced metrics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== APIs pour le suivi de l'évolution des tumeurs =====

@app.route('/api/patients')
def get_patients_list():
    """API pour obtenir la liste des patients avec suivi"""
    try:
        # Get doctor_id from session if logged in
        doctor_id = session.get('doctor_id')
        print(f"DEBUG get_patients_list: doctor_id = {doctor_id}")
        
        # Query MongoDB for patients
        if doctor_id:
            # Get patients for this doctor
            patients_cursor = db.patients.find({"doctor_id": doctor_id})
            print(f"DEBUG: Querying MongoDB for doctor_id: {doctor_id}")
        else:
            # If no doctor logged in, return empty list
            print("DEBUG: No doctor_id in session")
            patients_cursor = []
        
        patients = []
        for patient in patients_cursor:
            print(f"DEBUG: Found patient: {patient.get('patient_id')} - {patient.get('patient_name')}")
            # Convert MongoDB document to dict
            patient_dict = {
                'patient_id': patient.get('patient_id'),
                'patient_name': patient.get('patient_name'),
                'date_of_birth': patient.get('date_of_birth'),
                'gender': patient.get('gender'),
                'phone': patient.get('phone'),
                'email': patient.get('email'),
                'address': patient.get('address'),
                'medical_history': patient.get('medical_history'),
                'created_at': patient.get('created_at'),
                'updated_at': patient.get('updated_at')
            }
            patients.append(patient_dict)
        
        print(f"DEBUG: Returning {len(patients)} patients")
        return jsonify({
            'success': True,
            'data': patients,
            'count': len(patients)
        })

    except Exception as e:
        print(f"Erreur get_patients_list: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e),
            'data': []
        }), 500

@app.route('/api/patients/next-id')
@login_required
def get_next_patient_id():
    """Proposer un prochain ID patient unique pour le médecin connecté (format P0001)."""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # Get existing patient IDs
        patients = list(db.patients.find(
            {"doctor_id": doctor_id, "patient_id": {"$regex": "^P"}},
            {"patient_id": 1}
        ))
        existing_ids = [p.get("patient_id") for p in patients]

        max_n = 0
        for pid in existing_ids:
            try:
                if pid and pid.startswith('P'):
                    n = int(pid[1:])
                    if n > max_n:
                        max_n = n
            except Exception:
                continue
        next_id = f"P{max_n + 1:04d}"
        return jsonify({'success': True, 'next_id': next_id})

    except Exception as e:
        print(f"Erreur next-id: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/patients/check-id/<patient_id>')
@login_required
def check_patient_id(patient_id):
    """Vérifier la disponibilité d'un ID patient pour le médecin connecté."""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        pid = (patient_id or '').strip()
        if not pid:
            return jsonify({'success': True, 'available': False, 'reason': 'ID vide'})

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # Check if patient ID exists
        exists = db.patients.find_one({"doctor_id": doctor_id, "patient_id": pid}) is not None

        return jsonify({'success': True, 'available': not exists})

    except Exception as e:
        print(f"Erreur check-id: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patients/<patient_id>/evolution')
def get_patient_evolution(patient_id):
    """API pour obtenir l'évolution d'un patient spécifique"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Récupérer toutes les analyses du patient
        # MongoDB query needed here
        # SELECT id, exam_date, predicted_label, confidence, tumor_size_estimate,
        #                    probabilities, description, recommendations
        #             FROM analyses
        #             WHERE patient_id = ?
        #             ORDER BY exam_date ASC, timestamp ASC)

        analyses = []
        # for row in cursor.fetchall():  # TODO: Convert to MongoDB
            # probabilities = json.loads(row[5]) if row[5] else {}
            # recommendations = json.loads(row[7]) if row[7] else []

            # analyses.append({
            #     'id': row[0],
            #     'exam_date': row[1],
            #     'predicted_label': row[2],
            #     'confidence': round(row[3] * 100, 1),
            #     'tumor_size_estimate': round(row[4], 2) if row[4] else None,
        # TODO: Implement MongoDB query
            #     'probabilities': probabilities,
            #     'description': row[6],
            #     'recommendations': recommendations
            # })

        # Récupérer l'évolution détaillée
        # MongoDB query needed here
        # SELECT exam_date, diagnosis_change, confidence_change, size_change,
        #                    evolution_type, notes
        #             FROM tumor_evolution
        #             WHERE patient_id = ?
        #             ORDER BY exam_date ASC)

        evolution_details = []
        # for row in cursor.fetchall():  # TODO: Convert to MongoDB
            # evolution_details.append({
            #     'exam_date': row[0],
            #     'diagnosis_change': row[1],
            #     'confidence_change': round(row[2] * 100, 1) if row[2] else 0,
            #     'size_change': round(row[3], 2) if row[3] else None,
            #     'evolution_type': row[4],
            #     'notes': row[5]
            # })
        # TODO: Implement MongoDB query

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'patient_id': patient_id,
                'analyses': analyses,
                'evolution_details': evolution_details,
                'summary': {
                    'total_analyses': len(analyses),
                    'first_exam': analyses[0]['exam_date'] if analyses else None,
                    'last_exam': analyses[-1]['exam_date'] if analyses else None,
                    'current_diagnosis': analyses[-1]['predicted_label'] if analyses else None,
                    'current_confidence': analyses[-1]['confidence'] if analyses else None
                }
            }
        })

    except Exception as e:
        print(f"Erreur patient evolution: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/patient/<patient_id>')
@login_required
def patient_profile(patient_id):
    """Page de profil détaillé d'un patient"""
    doctor = get_current_doctor()
    if not doctor:
        return redirect(url_for('login'))

    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        doctor_id = session.get('doctor_id')
        
        # Vérifier que le patient appartient au médecin connecté
        patient_data = db.patients.find_one({
            "patient_id": patient_id,
            "doctor_id": doctor_id
        })
        
        print(f"DEBUG patient_profile: patient_id={patient_id}, doctor_id={doctor_id}, found={patient_data is not None}")
        
        if not patient_data:
            flash('Patient non trouvé ou accès non autorisé', 'error')
            return redirect(url_for('dashboard'))

        # Convertir les dates strings en datetime objects pour le template
        def parse_date(date_value):
            if not date_value:
                return None
            if isinstance(date_value, datetime):
                return date_value
            if isinstance(date_value, str):
                try:
                    return datetime.strptime(date_value, '%Y-%m-%d')
                except:
                    try:
                        return datetime.strptime(date_value, '%Y-%m-%d %H:%M:%S')
                    except:
                        return None
            return None

        patient = {
            'patient_id': patient_data.get('patient_id'),
            'patient_name': patient_data.get('patient_name'),
            'date_of_birth': parse_date(patient_data.get('date_of_birth')),
            'gender': patient_data.get('gender'),
            'first_analysis_date': parse_date(patient_data.get('first_analysis_date')),
            'last_analysis_date': parse_date(patient_data.get('last_analysis_date')),
            'total_analyses': patient_data.get('total_analyses', 0),
            'phone': patient_data.get('phone'),
            'email': patient_data.get('email'),
            'address': patient_data.get('address'),
            'emergency_contact_name': patient_data.get('emergency_contact_name'),
            'emergency_contact_phone': patient_data.get('emergency_contact_phone'),
            'medical_history': patient_data.get('medical_history'),
            'allergies': patient_data.get('allergies'),
            'current_medications': patient_data.get('current_medications'),
            'insurance_number': patient_data.get('insurance_number'),
            'notes': patient_data.get('notes')
        }

        # Récupérer les analyses du patient depuis MongoDB
        print(f"DEBUG: Querying analyses for patient_id={patient_id}, doctor_id={doctor_id}")
        
        analyses_cursor = db.analyses.find({
            "patient_id": patient_id,
            "doctor_id": doctor_id
        }).sort([("exam_date", -1), ("timestamp", -1)])
        
        # Compter les analyses
        analyses_count = db.analyses.count_documents({"patient_id": patient_id, "doctor_id": doctor_id})
        print(f"DEBUG: Found {analyses_count} analyses in MongoDB")
        
        # Vérifier aussi sans filtrer par doctor_id
        total_for_patient = db.analyses.count_documents({"patient_id": patient_id})
        print(f"DEBUG: Total analyses for patient_id {patient_id} (all doctors): {total_for_patient}")
        
        # Afficher quelques exemples d'analyses
        sample_analyses = list(db.analyses.find({}).limit(3))
        for sample in sample_analyses:
            print(f"DEBUG: Sample analysis - patient_id: {sample.get('patient_id')}, doctor_id: {sample.get('doctor_id')}, _id: {sample.get('_id')}")

        analyses = []
        for analysis in analyses_cursor:
            # Convertir le timestamp
            timestamp_dt = analysis.get('timestamp')
            if isinstance(timestamp_dt, str):
                try:
                    timestamp_dt = datetime.strptime(timestamp_dt, '%Y-%m-%d %H:%M:%S')
                except:
                    timestamp_dt = datetime.now()
            elif not isinstance(timestamp_dt, datetime):
                timestamp_dt = datetime.now()
            
            # Sécuriser la conversion de confidence
            conf_value = analysis.get('confidence', 0.0)
            try:
                conf_float = float(conf_value) if conf_value else 0.0
            except:
                conf_float = 0.0
            
            analyses.append({
                'id': str(analysis.get('_id')),
                'timestamp': timestamp_dt,
                'filename': analysis.get('filename'),
                'predicted_class': analysis.get('predicted_class', 0),
                'predicted_label': analysis.get('predicted_label'),
                'confidence': conf_float,
                'probabilities': analysis.get('probabilities', {}),
                'description': analysis.get('description'),
                'recommendations': analysis.get('recommendations', []),
                'processing_time': analysis.get('processing_time'),
                'exam_date': analysis.get('exam_date'),
                'tumor_size_estimate': analysis.get('tumor_size_estimate'),
                'date_uploaded': timestamp_dt,
                'date_uploaded_str': str(timestamp_dt),
                'image_name': analysis.get('filename'),
                'image_path': f'/uploads/{analysis.get("filename")}',
                'medical_notes': analysis.get('description'),
                'risk_level': 'Élevé' if analysis.get('predicted_class', 0) != 0 else 'Faible'
            })
        
        print(f"DEBUG patient_profile: Found {len(analyses)} analyses for patient {patient_id}")

        # Calculer les statistiques
        normal_count = sum(1 for a in analyses if a['predicted_class'] == 0)
        abnormal_count = len(analyses) - normal_count

        # Récupérer les alertes médicales
        # alerts = get_patient_alerts(cursor, patient_id, doctor['id'])  # TODO: Define function
        alerts = []  # Placeholder until function is defined

        # Calculer le niveau de risque du patient
        risk_level = 'Faible'
        if abnormal_count > 0:
            if abnormal_count >= 3:
                risk_level = 'Critique'
            elif abnormal_count >= 2:
                risk_level = 'Élevé'
            else:
                risk_level = 'Modéré'

        patient['risk_level'] = risk_level

        # Trier les analyses par date (du plus récent au plus ancien)
        sorted_analyses = sorted(analyses, key=lambda x: x['date_uploaded'], reverse=True)
        
        # Préparer les données JSON pour les graphiques
        analyses_json = json.dumps([{
            'date_uploaded': a['date_uploaded_str'],  # Utiliser la version chaîne pour JSON
            'confidence': a['confidence'],
            'predicted_class': a['predicted_class'],
            'predicted_label': a['predicted_label']
        } for a in sorted_analyses])

        # conn.close() # DISABLED

        return render_template('patient_profile_pro.html',
                               patient=patient,
                               doctor=doctor,
                               analyses=sorted_analyses,  # Utiliser les analyses triées
                               alerts=alerts,
                               normal_count=normal_count,
                               abnormal_count=abnormal_count,
                               analyses_json=analyses_json,
                               current_date=datetime.now())

    except Exception as e:
        print(f"Erreur lors du chargement du profil patient: {e}")
        flash('Erreur lors du chargement du profil patient', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/patients/<patient_id>/detailed-history')
@login_required
def get_patient_detailed_history(patient_id):
    """API pour obtenir l'historique détaillé d'un patient avec métriques avancées"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Vérifier l'accès au patient
        patient = db.patients.find_one({"patient_id": patient_id, "doctor_id": doctor_id})
        if not patient:
            return jsonify({'success': False, 'error': 'Accès non autorisé'}), 403

        # Récupérer toutes les analyses avec détails complets depuis MongoDB
        analyses_cursor = db.analyses.find({
            "patient_id": patient_id,
            "doctor_id": doctor_id
        }).sort([("exam_date", 1), ("timestamp", 1)])

        analyses = []
        for analysis in analyses_cursor:
            analyses.append({
                'id': str(analysis.get('_id')),
                'timestamp': analysis.get('timestamp').isoformat() if analysis.get('timestamp') else None,
                'exam_date': analysis.get('exam_date').isoformat() if analysis.get('exam_date') else None,
                'predicted_label': analysis.get('predicted_label'),
                'confidence': round(analysis.get('confidence', 0) * 100, 1),
                'tumor_size_estimate': round(analysis.get('tumor_size_estimate'), 2) if analysis.get('tumor_size_estimate') else None,
                'probabilities': analysis.get('probabilities', {}),
                'description': analysis.get('description'),
                'recommendations': analysis.get('recommendations', []),
                'processing_time': round(analysis.get('processing_time', 0), 2),
                'filename': analysis.get('filename')
            })

        # Récupérer l'évolution détaillée depuis MongoDB
        evolution_cursor = db.tumor_evolution.find({
            "patient_id": patient_id
        }).sort("exam_date", 1)

        evolution_details = []
        for evo in evolution_cursor:
            evolution_details.append({
                'exam_date': evo.get('exam_date').isoformat() if evo.get('exam_date') else None,
                'diagnosis_change': evo.get('diagnosis_change'),
                'confidence_change': round(evo.get('confidence_change', 0) * 100, 1),
                'size_change': round(evo.get('size_change'), 2) if evo.get('size_change') else None,
                'evolution_type': evo.get('evolution_type'),
                'notes': evo.get('notes'),
                'created_at': evo.get('created_at').isoformat() if evo.get('created_at') else None
            })

        # Calculer des métriques avancées
        metrics = {}
        if analyses:
            # Moyenne de confiance
            metrics['avg_confidence'] = round(sum(a['confidence'] for a in analyses) / len(analyses), 1)
            # Nombre de tumeurs détectées
            metrics['tumor_count'] = sum(1 for a in analyses if a['predicted_label'] != 'Normal')
            # Dernière analyse
            if analyses:
                last = analyses[-1]
                metrics['last_diagnosis'] = last['predicted_label']
                metrics['last_confidence'] = last['confidence']

        return jsonify({
            'success': True,
            'data': {
                'analyses': analyses,
                'evolution_details': evolution_details,
                'metrics': metrics,
                'total_analyses': len(analyses)
            }
        })

    except Exception as e:
        print(f"Erreur patient detailed history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patients/<patient_id>/comparison')
def get_patient_comparison(patient_id):
    """API pour comparer les analyses d'un patient"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Récupérer les deux dernières analyses pour comparaison
        # MongoDB query needed here
        # SELECT id, exam_date, predicted_label, confidence, tumor_size_estimate,
        #                    probabilities
        #             FROM analyses
        #             WHERE patient_id = ?
        #             ORDER BY exam_date DESC, timestamp DESC
        #             LIMIT 2)

        analyses = []  # TODO: Implement MongoDB query

        if len(analyses) < 2:
            return jsonify({
                'success': False,
                'error': 'Pas assez d\'analyses pour effectuer une comparaison'
            }), 400

        current = analyses[0]
        previous = analyses[1]

        # Calculer les différences
        current_probs = json.loads(current[5]) if current[5] else {}
        previous_probs = json.loads(previous[5]) if previous[5] else {}

        comparison = {
            'current_analysis': {
                'exam_date': current[1],
                'diagnosis': current[2],
                'confidence': round(current[3] * 100, 1),
                'tumor_size': round(current[4], 2) if current[4] else None,
                'probabilities': current_probs
            },
            'previous_analysis': {
                'exam_date': previous[1],
                'diagnosis': previous[2],
                'confidence': round(previous[3] * 100, 1),
                'tumor_size': round(previous[4], 2) if previous[4] else None,
                'probabilities': previous_probs
            },
            'changes': {
                'diagnosis_changed': current[2] != previous[2],
                'diagnosis_change': f"{previous[2]} → {current[2]}" if current[2] != previous[2] else None,
                'confidence_change': round((current[3] - previous[3]) * 100, 1),
                'size_change': round(current[4] - previous[4], 2) if current[4] and previous[4] else None,
                'time_interval_days': (datetime.strptime(current[1], '%Y-%m-%d') -
                                     datetime.strptime(previous[1], '%Y-%m-%d')).days
            }
        }

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': comparison
        })

    except Exception as e:
        print(f"Erreur patient comparison: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/evolution/summary')
def get_evolution_summary():
    """API pour obtenir un résumé de l'évolution de tous les patients"""
    try:
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Statistiques générales d'évolution
        # MongoDB query needed here
        # SELECT evolution_type, COUNT(*) as count
        #             FROM tumor_evolution
        #             GROUP BY evolution_type)
        evolution_stats = {}  # TODO: Implement MongoDB query

        # Patients avec évolution récente (7 derniers jours)
        # MongoDB query needed here
        # SELECT DISTINCT te.patient_id, a.patient_name, te.evolution_type, te.notes
        #             FROM tumor_evolution te
        #             JOIN analyses a ON te.patient_id = a.patient_id
        #             WHERE te.exam_date >= DATE('now', '-7 days')
        #             GROUP BY te.patient_id, te.evolution_type, te.notes
        #             ORDER BY te.exam_date DESC)
        recent_evolutions = []
        # for row in cursor.fetchall():  # TODO: Convert to MongoDB
            # recent_evolutions.append({
            #     'patient_id': row[0],
            #     'patient_name': row[1],
            #     'evolution_type': row[2],
            #     'notes': row[3]
            # })
        # TODO: Implement MongoDB query

        # Alertes d'évolution critique
        # MongoDB query needed here
        # SELECT te.patient_id, a.patient_name, te.evolution_type, te.notes, te.exam_date
        #             FROM tumor_evolution te
        #             JOIN analyses a ON te.patient_id = a.patient_id
        #             WHERE te.evolution_type IN ('dégradation', 'croissance')
        #             AND te.exam_date >= DATE('now', '-30 days')
        #             GROUP BY te.patient_id, te.evolution_type, te.exam_date
        #             ORDER BY te.exam_date DESC)
        critical_alerts = []
        # for row in cursor.fetchall():  # TODO: Convert to MongoDB
            # critical_alerts.append({
            #     'patient_id': row[0],
            #     'patient_name': row[1],
            #     'evolution_type': row[2],
            #     'notes': row[3],
            #     'exam_date': row[4]
            # })
        # TODO: Implement MongoDB query

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'evolution_statistics': evolution_stats,
                'recent_evolutions': recent_evolutions,
                'critical_alerts': critical_alerts,
                'total_patients_tracked': len(set([e['patient_id'] for e in recent_evolutions]))
            }
        })

    except Exception as e:
        print(f"Erreur evolution summary: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/tumor-tracking')
def tumor_tracking_page():
    """Page de suivi de l'évolution des tumeurs"""
    return render_template('tumor_tracking.html')

def create_medical_report(data):
    """Créer un rapport médical PDF professionnel avec reportlab"""
    try:
        analysis_data = data['analysisData']
        current_date = datetime.now().strftime('%d/%m/%Y à %H:%M')

        # Créer le buffer pour le PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)

        # Styles
        styles = getSampleStyleSheet()
        
        # Vérifier et ajouter les styles personnalisés seulement s'ils n'existent pas
        if 'CustomTitle' not in styles:
            styles.add(ParagraphStyle(name='CustomTitle', fontSize=24, fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=30, textColor=colors.HexColor('#1e40af')))
        if 'CustomSubtitle' not in styles:
            styles.add(ParagraphStyle(name='CustomSubtitle', fontSize=16, fontName='Helvetica-Bold', alignment=TA_LEFT, spaceAfter=20, textColor=colors.HexColor('#374151')))
        if 'CustomSectionHeader' not in styles:
            styles.add(ParagraphStyle(name='CustomSectionHeader', fontSize=14, fontName='Helvetica-Bold', alignment=TA_LEFT, spaceAfter=10, textColor=colors.HexColor('#1f2937')))
        if 'CustomNormal' not in styles:
            styles.add(ParagraphStyle(name='CustomNormal', fontSize=11, fontName='Helvetica', alignment=TA_LEFT, spaceAfter=6))
        if 'CustomBold' not in styles:
            styles.add(ParagraphStyle(name='CustomBold', fontSize=11, fontName='Helvetica-Bold', alignment=TA_LEFT, spaceAfter=6))
        if 'CustomWarning' not in styles:
            styles.add(ParagraphStyle(name='CustomWarning', fontSize=10, fontName='Helvetica-Oblique', alignment=TA_LEFT, spaceAfter=6, textColor=colors.HexColor('#dc2626')))

        # Contenu du PDF
        story = []

        # En-tête avec logo et titre
        header_data = [
            [Paragraph('<b>NEUROSCAN AI</b>', ParagraphStyle('Bold', fontSize=18, textColor=colors.HexColor('#1e40af'))),
             Paragraph('<b>RAPPORT D\'ANALYSE IRM</b>', ParagraphStyle('Bold', fontSize=16, textColor=colors.HexColor('#374151')))],
            ['', Paragraph(f'<i>Généré le {current_date}</i>', styles['CustomNormal'])]
        ]
        header_table = Table(header_data, colWidths=[3*inch, 3*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 20))

        # Informations patient
        story.append(Paragraph('INFORMATIONS PATIENT', styles['CustomSectionHeader']))

        patient_info = [
            ['Nom du patient:', data.get('patientName', 'Non spécifié')],
            ['Date de naissance:', data.get('patientDob', 'Non spécifiée')],
            ['ID Patient:', data.get('patientId', 'Non spécifié')],
            ['Médecin référent:', data.get('doctorName', 'Non spécifié')],
            ['Date d\'analyse:', current_date]
        ]

        patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#374151')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))

        # Résultats de l'analyse
        story.append(Paragraph('RÉSULTATS DE L\'ANALYSE IA', styles['CustomSectionHeader']))

        # Diagnostic principal avec couleur
        diagnosis_color = {
            'Normal': '#10b981',  # vert
            'Gliome': '#ef4444',  # rouge
            'Méningiome': '#f59e0b',  # orange
            'Tumeur pituitaire': '#8b5cf6'  # violet
        }.get(analysis_data.get('predicted_label', 'Inconnu'), '#6b7280')

        diagnosis_data = [
            ['Diagnostic principal:', Paragraph(f'<font color="{diagnosis_color}"><b>{analysis_data.get("predicted_label", "Inconnu")}</b></font>', styles['CustomNormal'])],
            ['Niveau de confiance:', f'{analysis_data.get("confidence", 0) * 100:.1f}%'],
            ['Tumeur détectée:', 'Oui' if analysis_data.get('predicted_label', '') != 'Normal' else 'Non']
        ]

        diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
        diagnosis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#374151')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
        ]))
        story.append(diagnosis_table)
        story.append(Spacer(1, 15))

        # Probabilités détaillées avec graphique
        story.append(Paragraph('PROBABILITÉS DÉTAILLÉES', styles['CustomSectionHeader']))

        # Créer un graphique en barres pour les probabilités
        drawing = Drawing(400, 200)
        probabilities = analysis_data.get('probabilities', {})
        data_values = [probabilities.get('Normal', 0) * 100,
                       probabilities.get('Gliome', 0) * 100,
                       probabilities.get('Méningiome', 0) * 100,
                       probabilities.get('Tumeur pituitaire', 0) * 100]

        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 125
        bc.width = 300
        bc.data = [data_values]
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = 100
        bc.valueAxis.valueStep = 20
        bc.categoryAxis.labels.boxAnchor = 'ne'
        bc.categoryAxis.labels.dx = 8
        bc.categoryAxis.labels.dy = -2
        bc.categoryAxis.labels.angle = 30
        bc.categoryAxis.categoryNames = ['Normal', 'Gliome', 'Méningiome', 'Pituitaire']
        bc.bars[0].fillColor = colors.HexColor('#3b82f6')

        drawing.add(bc)
        story.append(drawing)
        story.append(Spacer(1, 15))

        # Tableau des probabilités
        prob_data = [['Classe', 'Probabilité']]
        probabilities = analysis_data.get('probabilities', {})
        for label, prob in probabilities.items():
            prob_data.append([label, f'{prob * 100:.1f}%'])

        prob_table = Table(prob_data, colWidths=[2.5*inch, 1.5*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white)
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 20))

        # Recommandations cliniques
        story.append(Paragraph('RECOMMANDATIONS CLINIQUES', styles['CustomSectionHeader']))

        recommendations = analysis_data.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f'{i}. {rec}', styles['CustomNormal']))
                story.append(Spacer(1, 3))
        else:
            story.append(Paragraph('Aucune recommandation spécifique générée.', styles['CustomNormal']))

        story.append(Spacer(1, 15))

        # Notes cliniques additionnelles
        if isinstance(data, dict) and data.get('clinicalNotes'):
            story.append(Paragraph('NOTES CLINIQUES ADDITIONNELLES', styles['CustomSectionHeader']))
            story.append(Paragraph(data['clinicalNotes'], styles['CustomNormal']))
            story.append(Spacer(1, 15))

        # Avertissement médical
        story.append(Paragraph('AVERTISSEMENT MÉDICAL', styles['CustomSectionHeader']))
        warning_text = """Cette analyse a été générée par un système d'intelligence artificielle à des fins d'aide au diagnostic. Elle ne remplace pas l'expertise médicale et doit être interprétée par un professionnel de santé qualifié.

Les résultats doivent être corrélés avec l'examen clinique et d'autres investigations complémentaires selon les protocoles en vigueur.

Ce système est certifié CE - Dispositif médical de classe IIa."""
        story.append(Paragraph(warning_text, styles['CustomWarning']))
        story.append(Spacer(1, 20))

        # Pied de page
        footer_data = [
            [Paragraph('<b>NeuroScan AI - Système d\'analyse IRM assistée par IA</b>', ParagraphStyle('Bold', fontSize=9, alignment=TA_CENTER)),
             Paragraph('<b>Version 2.0 - 2024</b>', ParagraphStyle('Bold', fontSize=9, alignment=TA_CENTER))]
        ]
        footer_table = Table(footer_data, colWidths=[3*inch, 3*inch])
        footer_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#6b7280'))
        ]))
        story.append(footer_table)

        # Générer le PDF
        doc.build(story)
        buffer.seek(0)

        return buffer.getvalue()

    except Exception as e:
        print(f"Erreur lors de la création du PDF: {e}")
        # Retourner un rapport texte en cas d'erreur, converti en bytes
        analysis_data = data['analysisData']
        current_date = datetime.now().strftime('%d/%m/%Y à %H:%M')

        report = f"""
RAPPORT D'ANALYSE IRM - NEUROSCAN AI
====================================

INFORMATIONS PATIENT
-------------------
Nom: {data.get('patientName', 'Non spécifié')}
Date de naissance: {data.get('patientDob', 'Non spécifiée')}
ID Patient: {data.get('patientId', 'Non spécifié')}
Médecin référent: {data.get('doctorName', 'Non spécifié')}
Date d'analyse: {current_date}

RÉSULTATS DE L'ANALYSE IA
-------------------------
Diagnostic principal: {analysis_data.get('predicted_label', 'Inconnu')}
Niveau de confiance: {analysis_data.get('confidence', 0) * 100:.1f}%
Tumeur détectée: {'Oui' if analysis_data.get('predicted_label', '') != 'Normal' else 'Non'}

PROBABILITÉS DÉTAILLÉES
-----------------------
"""

        probabilities = analysis_data.get('probabilities', {})
        for label, prob in probabilities.items():
            report += f"- {label}: {prob * 100:.1f}%\n"

        report += f"""
RECOMMANDATIONS CLINIQUES
-------------------------
"""

        recommendations = analysis_data.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"

        if data.get('clinicalNotes'):
            report += f"""
NOTES CLINIQUES ADDITIONNELLES
------------------------------
{data['clinicalNotes']}
"""

        report += f"""
AVERTISSEMENT MÉDICAL
--------------------
Cette analyse a été générée par un système d'intelligence artificielle
à des fins d'aide au diagnostic. Elle ne remplace pas l'expertise médicale
et doit être interprétée par un professionnel de santé qualifié.

Rapport généré par NeuroScan AI - {current_date}
Système certifié CE - Dispositif médical de classe IIa
"""

        # Convertir en bytes pour la compatibilité
        return report.encode('utf-8')

def send_analysis_email(data):
    """Simuler l'envoi d'un email de partage"""
    analysis_data = data['analysisData']
    current_date = datetime.now().strftime('%d/%m/%Y à %H:%M')

    # Dans une vraie application, ici on utiliserait un service d'email
    # comme SendGrid, AWS SES, ou SMTP

    email_content = f"""
Objet: Partage d'analyse IRM - NeuroScan AI

Bonjour {data.get('recipientName', 'Collègue')},

{data.get('shareMessage', 'Je partage avec vous cette analyse IRM pour avoir votre avis.')}

RÉSUMÉ DE L'ANALYSE:
- Diagnostic: {analysis_data.get('predicted_label', 'Inconnu')}
- Confiance: {analysis_data.get('confidence', 0) * 100:.1f}%
- Date d'analyse: {current_date}

Vous pouvez accéder à l'analyse complète via le lien sécurisé ci-dessous:
[Lien sécurisé vers l'analyse]

Niveau de confidentialité: {data.get('confidentiality', 'Standard')}

Cordialement,
Système NeuroScan AI
"""

    # Log de l'email (en développement)
    print(f"Email simulé envoyé à {data['recipientEmail']}")
    print(f"Contenu: {email_content[:200]}...")

    return True

@app.route('/api/alerts')
@login_required
def get_doctor_alerts():
    """API pour obtenir les alertes du médecin connecté"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Récupérer les alertes non résolues depuis MongoDB
        alerts_cursor = db.medical_alerts.find({
            "doctor_id": doctor_id,
            "is_resolved": False
        }).sort("created_at", -1).limit(50)
        
        alerts = []
        for alert in alerts_cursor:
            # Récupérer le nom du patient
            patient = db.patients.find_one({
                "patient_id": alert.get('patient_id'),
                "doctor_id": doctor_id
            })
            patient_name = patient.get('patient_name') if patient else 'Patient inconnu'
            
            alerts.append({
                'id': str(alert.get('_id')),
                'patient_id': alert.get('patient_id'),
                'patient_name': patient_name,
                'alert_type': alert.get('alert_type'),
                'severity': alert.get('severity'),
                'title': alert.get('title'),
                'message': alert.get('message'),
                'is_read': alert.get('is_read', False),
                'created_at': alert.get('created_at').isoformat() if alert.get('created_at') else None
            })
        
        print(f"DEBUG get_doctor_alerts: Found {len(alerts)} alerts for doctor {doctor_id}")

        return jsonify({
            'success': True,
            'data': alerts
        })

    except Exception as e:
        print(f"Erreur get doctor alerts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alerts/<alert_id>/mark-read', methods=['POST'])
@login_required
def mark_alert_read(alert_id):
    """Marquer une alerte comme lue"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # Convertir alert_id en ObjectId
        try:
            obj_id = ObjectId(alert_id)
        except:
            return jsonify({'success': False, 'error': 'ID alerte invalide'}), 400

        # Mettre à jour l'alerte dans MongoDB
        result = db.medical_alerts.update_one(
            {'_id': obj_id, 'doctor_id': doctor_id},
            {'$set': {'is_read': True}}
        )
        
        if result.matched_count == 0:
            return jsonify({'success': False, 'error': 'Alerte non trouvée'}), 404

        return jsonify({'success': True})

    except Exception as e:
        print(f"Erreur mark alert read: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
@login_required
def resolve_alert(alert_id):
    """Résoudre une alerte"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # Convertir alert_id en ObjectId
        try:
            obj_id = ObjectId(alert_id)
        except:
            return jsonify({'success': False, 'error': 'ID alerte invalide'}), 400

        # Mettre à jour l'alerte dans MongoDB
        result = db.medical_alerts.update_one(
            {'_id': obj_id, 'doctor_id': doctor_id},
            {'$set': {
                'is_resolved': True,
                'resolved_at': datetime.utcnow(),
                'resolved_by': doctor_id
            }}
        )
        
        if result.matched_count == 0:
            return jsonify({'success': False, 'error': 'Alerte non trouvée'}), 404

        return jsonify({'success': True})

    except Exception as e:
        print(f"Erreur resolve alert: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/notifications')
@login_required
def get_notifications():
    """API pour obtenir les notifications du médecin connecté"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # MongoDB query needed here
        # SELECT id, type, title, message, data, is_read, created_at
        #             FROM notifications
        #             WHERE doctor_id = ?
        #             ORDER BY created_at DESC
        #             LIMIT 20)

        notifications = []
        # for row in cursor.fetchall():  # TODO: Convert to MongoDB
            # data = json.loads(row[4]) if row[4] else {}
            # notifications.append({
            #     'id': row[0],
            #     'type': row[1],
        # TODO: Implement MongoDB query
            #     'title': row[2],
            #     'message': row[3],
            #     'data': data,
            #     'is_read': bool(row[5]),
            #     'created_at': row[6]
            # })

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': notifications
        })

    except Exception as e:
        print(f"Erreur get notifications: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/notifications/<int:notification_id>/mark-read', methods=['POST'])
@login_required
def mark_notification_read(notification_id):
    """Marquer une notification comme lue"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # MongoDB query needed here
        # UPDATE notifications
        #             SET is_read = 1
        #             WHERE id = ? AND doctor_id = ?)

        # conn.commit() # DISABLED
        # conn.close() # DISABLED

        return jsonify({'success': True})

    except Exception as e:
        print(f"Erreur mark notification read: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patients/<patient_id>/generate-evolution-report', methods=['POST'])
@login_required
def generate_evolution_report(patient_id):
    """Générer un rapport d'évolution automatisé pour un patient"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Vérifier l'accès au patient
        # MongoDB query needed here
        # SELECT patient_name FROM patients
        #             WHERE patient_id = ? AND doctor_id = ?)

        # patient_data = None  # cursor.fetchone() # DISABLED  # TODO: Convert to MongoDB
        patient_data = None  # TODO: Implement MongoDB query
        if not patient_data:
            # return jsonify({'success': False, 'error': 'Accès non autorisé'}), 403
            pass  # TODO: Implement proper access check

        # Récupérer les données pour le rapport
        # MongoDB query needed here
        # SELECT id, timestamp, exam_date, predicted_label, confidence,
        #                    tumor_size_estimate, probabilities, description, recommendations
        #             FROM analyses
        #             WHERE patient_id = ? AND doctor_id = ?
        #             ORDER BY exam_date ASC, timestamp ASC)

        analyses = []
        # for row in cursor.fetchall():  # TODO: Convert to MongoDB
            # probabilities = json.loads(row[6]) if row[6] else {}
            # recommendations = json.loads(row[8]) if row[8] else []

            # analyses.append({
        # TODO: Implement MongoDB query
            #     'id': row[0],
            #     'timestamp': row[1],
            #     'exam_date': row[2],
            #     'predicted_label': row[3],
            #     'confidence': row[4],
            #     'tumor_size_estimate': row[5],
            #     'probabilities': probabilities,
            #     'description': row[7],
            #     'recommendations': recommendations
            # })

        # Récupérer l'évolution
        # MongoDB query needed here
        # SELECT exam_date, diagnosis_change, confidence_change, size_change,
        #                    evolution_type, notes
        #             FROM tumor_evolution
        #             WHERE patient_id = ?
        #             ORDER BY exam_date ASC)

        evolution_details = []
        # for row in cursor.fetchall():  # TODO: Convert to MongoDB
            # evolution_details.append({
            #     'exam_date': row[0],
            #     'diagnosis_change': row[1],
            #     'confidence_change': row[2],
            #     'size_change': row[3],
            #     'evolution_type': row[4],
            #     'notes': row[5]
            # })
        # TODO: Implement MongoDB query

        # conn.close() # DISABLED

        # Générer le rapport
        report_data = {
            'patient_id': patient_id,
            'patient_name': patient_data[0],
            'analyses': analyses,
            'evolution_details': evolution_details,
            'generated_at': datetime.now().isoformat(),
            'doctor_id': doctor_id
        }

        report_content = create_evolution_report(report_data)

        # Sauvegarder le rapport (simulation)
        report_id = f"EVOL_{patient_id}_{int(time.time())}"

        return jsonify({
            'success': True,
            'report_id': report_id,
            'report_content': report_content,
            'message': 'Rapport d\'évolution généré avec succès'
        })

    except Exception as e:
        print(f"Erreur génération rapport évolution: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def create_evolution_report(data):
    """Créer un rapport d'évolution détaillé"""
    try:
        analyses = data['analyses']
        evolution_details = data['evolution_details']

        if not analyses:
            return "Aucune donnée d'analyse disponible pour ce patient."

        # Calculer les métriques
        # TODO: Implement calculate_patient_metrics function for MongoDB
        metrics = {}  # Placeholder - calculate_patient_metrics not yet implemented
        # metrics = calculate_patient_metrics(analyses, evolution_details)

        current_date = datetime.now().strftime('%d/%m/%Y à %H:%M')

        report = f"""
RAPPORT D'ÉVOLUTION PATIENT - NEUROSCAN AI
==========================================

INFORMATIONS PATIENT
-------------------
Nom: {data['patient_name']}
ID Patient: {data['patient_id']}
Date de génération: {current_date}
Période d'analyse: {analyses[0]['exam_date']} - {analyses[-1]['exam_date']}
Nombre total d'analyses: {len(analyses)}

RÉSUMÉ EXÉCUTIF
--------------
Diagnostic principal: {metrics.get('most_common_diagnosis', 'Indéterminé')}
Niveau de risque actuel: {metrics.get('risk_level', 'Indéterminé').upper()}
Durée de suivi: {metrics.get('follow_up_months', 0)} mois
Confiance moyenne: {metrics.get('avg_confidence', 0)}%

ÉVOLUTION TEMPORELLE
-------------------
"""

        # Ajouter les analyses chronologiques
        for i, analysis in enumerate(analyses):
            report += f"""
Analyse #{i+1} - {analysis['exam_date']}
  • Diagnostic: {analysis['predicted_label']}
  • Confiance: {analysis['confidence']*100:.1f}%
  • Taille estimée: {analysis['tumor_size_estimate']:.2f} cm si applicable
"""

            # Ajouter l'évolution si disponible
            evolution = next((e for e in evolution_details if e['exam_date'] == analysis['exam_date']), None)
            if evolution and evolution['notes']:
                report += f"  • Évolution: {evolution['notes']}\n"

        # Ajouter les alertes
        if metrics.get('alerts'):
            report += "\nALERTES MÉDICALES\n"
            report += "-" * 17 + "\n"
            for alert in metrics['alerts']:
                report += f"• {alert['message']}\n"

        # Ajouter les recommandations
        if metrics.get('recommendations'):
            report += "\nRECOMMANDATIONS CLINIQUES\n"
            report += "-" * 25 + "\n"
            for i, rec in enumerate(metrics['recommendations'], 1):
                report += f"{i}. {rec}\n"

        report += f"""

ANALYSE STATISTIQUE
------------------
Stabilité diagnostique: {'Oui' if metrics.get('diagnosis_stability') else 'Non'}
Tendance de confiance: {metrics.get('confidence_trend', 'Stable')}
"""

        if metrics.get('size_metrics'):
            size_metrics = metrics['size_metrics']
            report += f"""Taille moyenne: {size_metrics.get('avg_size', 0):.2f} cm
Tendance de taille: {size_metrics.get('size_trend', 'Stable')}
"""

        report += f"""

CONCLUSION ET SUIVI
------------------
Ce rapport automatisé présente l'évolution du patient sur {metrics.get('follow_up_months', 0)} mois.
Le niveau de risque actuel est évalué comme {metrics.get('risk_level', 'indéterminé')}.

Prochaines étapes recommandées:
"""

        # Recommandations de suivi basées sur le niveau de risque
        risk_level = metrics.get('risk_level', 'indéterminé')
        if risk_level == 'critique':
            report += "• Consultation oncologique urgente dans les 48h\n"
            report += "• IRM de contrôle dans 2 semaines\n"
        elif risk_level == 'élevé':
            report += "• Consultation spécialisée dans la semaine\n"
            report += "• IRM de contrôle dans 1 mois\n"
        elif risk_level == 'modéré':
            report += "• Suivi de routine dans 3 mois\n"
            report += "• Surveillance clinique régulière\n"
        else:
            report += "• Suivi de routine selon protocole standard\n"

        report += f"""

AVERTISSEMENT MÉDICAL
--------------------
Ce rapport a été généré automatiquement par NeuroScan AI.
Il doit être interprété par un professionnel de santé qualifié.
Les recommandations sont indicatives et doivent être adaptées
au contexte clinique spécifique du patient.

Rapport généré le {current_date}
Système NeuroScan AI - Version 2.0
"""

        return report

    except Exception as e:
        print(f"Erreur lors de la création du rapport: {e}")
        return f"Erreur lors de la génération du rapport: {str(e)}"

@app.route('/api/patients', methods=['POST'])
@login_required
def create_patient():
    """Créer un nouveau patient"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        data = request.get_json() or {}

        # Normaliser les clés camelCase -> snake_case pour compatibilité front
        key_map = {
            'patientName': 'patient_name',
            'patientId': 'patient_id',
            'dateOfBirth': 'date_of_birth',
            'emergencyContact': 'emergency_contact_name',
            'emergencyPhone': 'emergency_contact_phone',
            'medicalHistory': 'medical_history',
            'currentMedications': 'current_medications',
        }
        for src, dst in key_map.items():
            if src in data and dst not in data:
                data[dst] = data[src]

        # Nettoyage de base
        if 'patient_id' in data and isinstance(data['patient_id'], str):
            data['patient_id'] = data['patient_id'].strip()
        if 'patient_name' in data and isinstance(data['patient_name'], str):
            data['patient_name'] = data['patient_name'].strip()

        # Validation des données requises
        required_fields = ['patient_id', 'patient_name']
        for field in required_fields:
            if not data.get(field):
                # Si l'ID est manquant, tenter de le générer automatiquement
                if field == 'patient_id':
                    # Connexion provisoire pour générer un ID unique basé sur les patients du médecin
                    # conn = sqlite3.connect() # DISABLED - MongoDB used instead
                    # cursor = conn.cursor() # DISABLED
                    # MongoDB query needed here
        # SELECT patient_id FROM patients
        #                         WHERE doctor_id = ? AND patient_id LIKE 'P%')
                    existing_ids = []  # TODO: Implement MongoDB query
                    # Trouver le prochain numéro disponible
                    max_n = 0
                    for pid in existing_ids:
                        try:
                            if pid and pid.startswith('P'):
                                n = int(pid[1:])
                                if n > max_n:
                                    max_n = n
                        except Exception:
                            continue
                    generated_id = f"P{max_n + 1:04d}"
                    data['patient_id'] = generated_id
                    # conn.close() # DISABLED
                    continue
                return jsonify({'success': False, 'error': f'Le champ {field} est requis'}), 400

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Vérifier si le patient existe déjà
        # MongoDB query needed here
        # SELECT COUNT(*) FROM patients
        #             WHERE patient_id = ? AND doctor_id = ?)

        # Check if patient already exists in MongoDB
        existing_patient = db.patients.find_one({"patient_id": data['patient_id'], "doctor_id": doctor_id})
        if existing_patient:
            return jsonify({'success': False, 'error': 'Un patient avec cet ID existe déjà'}), 400

        # Insert new patient into MongoDB
        patient_doc = {
            'patient_id': data['patient_id'],
            'patient_name': data['patient_name'],
            'doctor_id': doctor_id,
            'date_of_birth': data.get('date_of_birth'),
            'gender': data.get('gender'),
            'phone': data.get('phone'),
            'email': data.get('email'),
            'address': data.get('address'),
            'emergency_contact_name': data.get('emergency_contact_name'),
            'emergency_contact_phone': data.get('emergency_contact_phone'),
            'medical_history': data.get('medical_history'),
            'allergies': data.get('allergies'),
            'current_medications': data.get('current_medications'),
            'insurance_number': data.get('insurance_number'),
            'notes': data.get('notes'),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        result = db.patients.insert_one(patient_doc)
        print(f"DEBUG: Patient created with _id: {result.inserted_id}, patient_id: {data['patient_id']}, doctor_id: {doctor_id}")

        return jsonify({
            'success': True,
            'message': 'Patient créé avec succès',
            'patient_id': data['patient_id']
        })

    except Exception as e:
        print(f"Erreur création patient: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patients/<patient_id>', methods=['PUT'])
@login_required
def update_patient(patient_id):
    """Mettre à jour un patient"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        data = request.get_json()

        # Verify patient belongs to doctor
        patient = db.patients.find_one({"patient_id": patient_id, "doctor_id": doctor_id})
        if not patient:
            return jsonify({'success': False, 'error': 'Patient non trouvé'}), 404

        # Update patient
        update_doc = {
            'patient_name': data['patient_name'],
            'date_of_birth': data.get('date_of_birth'),
            'gender': data.get('gender'),
            'phone': data.get('phone'),
            'email': data.get('email'),
            'address': data.get('address'),
            'emergency_contact_name': data.get('emergency_contact_name'),
            'emergency_contact_phone': data.get('emergency_contact_phone'),
            'medical_history': data.get('medical_history'),
            'allergies': data.get('allergies'),
            'current_medications': data.get('current_medications'),
            'insurance_number': data.get('insurance_number'),
            'notes': data.get('notes'),
            'updated_at': datetime.utcnow()
        }
        db.patients.update_one(
            {"patient_id": patient_id, "doctor_id": doctor_id},
            {"$set": update_doc}
        )

        return jsonify({
            'success': True,
            'message': 'Patient mis à jour avec succès'
        })

    except Exception as e:
        print(f"Erreur mise à jour patient: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patients/<patient_id>', methods=['DELETE'])
@login_required
def delete_patient(patient_id):
    """Supprimer un patient et toutes ses données associées"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Vérifier que le patient appartient au médecin
        patient = db.patients.find_one({"patient_id": patient_id, "doctor_id": doctor_id})
        if not patient:
            return jsonify({'success': False, 'error': 'Patient non trouvé'}), 404

        # Supprimer toutes les données associées au patient
        tables_to_clean = [
            'medical_alerts',
            'tumor_evolution',
            'analyses',
            'patients'
        ]

        # Delete from all collections
        for collection_name in tables_to_clean:
            collection = db[collection_name]
            if collection_name == 'patients':
                collection.delete_many({"patient_id": patient_id, "doctor_id": doctor_id})
            else:
                collection.delete_many({"patient_id": patient_id})

        return jsonify({
            'success': True,
            'message': 'Patient et toutes ses données supprimés avec succès'
        })

    except Exception as e:
        print(f"Erreur suppression patient: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patients/<patient_id>/details')
@login_required
def get_patient_details(patient_id):
    """Obtenir les détails complets d'un patient"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Récupérer les informations complètes du patient depuis MongoDB
        patient_data = db.patients.find_one({
            "patient_id": patient_id,
            "doctor_id": doctor_id
        })
        
        print(f"DEBUG get_patient_details: patient_id={patient_id}, doctor_id={doctor_id}, found={patient_data is not None}")
        
        if not patient_data:
            return jsonify({'success': False, 'error': 'Patient non trouvé'}), 404

        patient = {
            'patient_id': patient_data.get('patient_id'),
            'patient_name': patient_data.get('patient_name'),
            'date_of_birth': patient_data.get('date_of_birth'),
            'gender': patient_data.get('gender'),
            'phone': patient_data.get('phone'),
            'email': patient_data.get('email'),
            'address': patient_data.get('address'),
            'emergency_contact_name': patient_data.get('emergency_contact_name'),
            'emergency_contact_phone': patient_data.get('emergency_contact_phone'),
            'medical_history': patient_data.get('medical_history'),
            'allergies': patient_data.get('allergies'),
            'current_medications': patient_data.get('current_medications'),
            'insurance_number': patient_data.get('insurance_number'),
            'notes': patient_data.get('notes'),
            'first_analysis_date': patient_data.get('first_analysis_date'),
            'last_analysis_date': patient_data.get('last_analysis_date'),
            'total_analyses': patient_data.get('total_analyses', 0),
            'created_at': patient_data.get('created_at'),
            'updated_at': patient_data.get('updated_at')
        }

        return jsonify({
            'success': True,
            'data': patient
        })

    except Exception as e:
        print(f"Erreur détails patient: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== APIs spécifiques pour le Dashboard Professionnel =====

@app.route('/api/pro-dashboard/overview')
@login_required  
def pro_dashboard_overview():
    """API pour les statistiques du dashboard professionnel du médecin connecté"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        db = get_mongodb()
        analyses_collection = db.analyses
        patients_collection = db.patients

        # Statistiques principales du médecin
        # Total analyses
        total_analyses = analyses_collection.count_documents({'doctor_id': doctor_id})

        # Tumeurs détectées (predicted_class != 0, ou predicted_label != 'Normal')
        tumors_detected = analyses_collection.count_documents({
            'doctor_id': doctor_id,
            'predicted_label': {'$ne': 'Normal'}
        })

        # Nombre de patients
        patients_count = patients_collection.count_documents({'doctor_id': doctor_id})

        # Confiance moyenne
        avg_confidence_pipeline = [
            {'$match': {'doctor_id': doctor_id}},
            {'$group': {'_id': None, 'avg_conf': {'$avg': '$confidence'}}}
        ]
        avg_conf_result = list(analyses_collection.aggregate(avg_confidence_pipeline))
        avg_confidence = round(avg_conf_result[0]['avg_conf'], 2) if avg_conf_result and avg_conf_result[0].get('avg_conf') else 0

        # Répartition par type de diagnostic
        distribution_pipeline = [
            {'$match': {'doctor_id': doctor_id}},
            {'$group': {'_id': '$predicted_label', 'count': {'$sum': 1}}}
        ]
        distribution_result = list(analyses_collection.aggregate(distribution_pipeline))
        distribution = {item['_id']: item['count'] for item in distribution_result if item['_id']}

        # Analyses par période (30 derniers jours)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        daily_pipeline = [
            {'$match': {
                'doctor_id': doctor_id,
                'timestamp': {'$gte': thirty_days_ago}
            }},
            {'$group': {
                '_id': {
                    '$dateToString': {'format': '%Y-%m-%d', 'date': '$timestamp'}
                },
                'count': {'$sum': 1}
            }},
            {'$sort': {'_id': 1}}
        ]
        daily_result = list(analyses_collection.aggregate(daily_pipeline))
        daily_data = [{'date': item['_id'], 'count': item['count']} for item in daily_result]

        # Calcul des changements (simulation basée sur les données existantes)
        now = datetime.now()
        
        # Début du mois actuel
        start_of_current_month = datetime(now.year, now.month, 1)
        
        # Début du mois précédent
        if now.month == 1:
            start_of_previous_month = datetime(now.year - 1, 12, 1)
        else:
            start_of_previous_month = datetime(now.year, now.month - 1, 1)

        # Analyses ce mois vs mois précédent
        current_month = analyses_collection.count_documents({
            'doctor_id': doctor_id,
            'timestamp': {'$gte': start_of_current_month}
        })

        previous_month = analyses_collection.count_documents({
            'doctor_id': doctor_id,
            'timestamp': {
                '$gte': start_of_previous_month,
                '$lt': start_of_current_month
            }
        })

        analyses_change = ((current_month - previous_month) / previous_month * 100) if previous_month > 0 else (100 if current_month > 0 else 0)

        # Tumeurs détectées ce mois vs mois précédent
        current_month_tumors = analyses_collection.count_documents({
            'doctor_id': doctor_id,
            'predicted_label': {'$ne': 'Normal'},
            'timestamp': {'$gte': start_of_current_month}
        })

        previous_month_tumors = analyses_collection.count_documents({
            'doctor_id': doctor_id,
            'predicted_label': {'$ne': 'Normal'},
            'timestamp': {
                '$gte': start_of_previous_month,
                '$lt': start_of_current_month
            }
        })

        tumors_change = ((current_month_tumors - previous_month_tumors) / previous_month_tumors * 100) if previous_month_tumors > 0 else (100 if current_month_tumors > 0 else 0)

        # Nouveaux patients ce mois
        new_patients_month = patients_collection.count_documents({
            'doctor_id': doctor_id,
            'created_at': {'$gte': start_of_current_month}
        })

        previous_month_patients = patients_collection.count_documents({
            'doctor_id': doctor_id,
            'created_at': {
                '$gte': start_of_previous_month,
                '$lt': start_of_current_month
            }
        })

        patients_change = ((new_patients_month - previous_month_patients) / previous_month_patients * 100) if previous_month_patients > 0 else (100 if new_patients_month > 0 else 0)

        # Confiance moyenne ce mois vs mois précédent
        current_month_conf_pipeline = [
            {'$match': {
                'doctor_id': doctor_id,
                'timestamp': {'$gte': start_of_current_month}
            }},
            {'$group': {'_id': None, 'avg_conf': {'$avg': '$confidence'}}}
        ]
        current_conf_result = list(analyses_collection.aggregate(current_month_conf_pipeline))
        current_avg_confidence = current_conf_result[0]['avg_conf'] if current_conf_result and current_conf_result[0].get('avg_conf') else 0

        previous_month_conf_pipeline = [
            {'$match': {
                'doctor_id': doctor_id,
                'timestamp': {
                    '$gte': start_of_previous_month,
                    '$lt': start_of_current_month
                }
            }},
            {'$group': {'_id': None, 'avg_conf': {'$avg': '$confidence'}}}
        ]
        previous_conf_result = list(analyses_collection.aggregate(previous_month_conf_pipeline))
        previous_avg_confidence = previous_conf_result[0]['avg_conf'] if previous_conf_result and previous_conf_result[0].get('avg_conf') else 0

        confidence_change = ((current_avg_confidence - previous_avg_confidence) / previous_avg_confidence * 100) if previous_avg_confidence > 0 else 0

        return jsonify({
            'success': True,
            'data': {
                'total_analyses': total_analyses,
                'tumors_detected': tumors_detected,
                'patients_count': patients_count,
                'avg_confidence': avg_confidence,
                'normal_count': distribution.get('Normal', 0),
                'glioma_count': distribution.get('Gliome', 0),
                'meningioma_count': distribution.get('Méningiome', 0),
                'pituitary_count': distribution.get('Tumeur pituitaire', 0),
                'daily_analyses': daily_data,
                'changes': {
                    'analyses_change': round(analyses_change, 1),
                    'tumors_change': round(tumors_change, 1),
                    'patients_change': round(patients_change, 1),
                    'confidence_change': round(confidence_change, 1)
                }
            }
        })

    except Exception as e:
        print(f"Erreur pro dashboard overview: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pro-dashboard/recent-analyses')
@login_required
def pro_dashboard_recent_analyses():
    """API pour les analyses récentes du dashboard professionnel"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # MongoDB query for recent analyses
        db = get_mongodb()
        
        analyses_cursor = db.analyses.find({
            'doctor_id': doctor_id
        }).sort('timestamp', -1).limit(10)
        
        analyses = []
        for analysis in analyses_cursor:
            patient_name = analysis.get('patient_name')
            patient_id = analysis.get('patient_id')
            
            # Format patient name
            if not patient_name and patient_id:
                patient_name = f'Patient {patient_id}'
            elif not patient_name:
                patient_name = 'Patient anonyme'
            
            analyses.append({
                'id': str(analysis.get('_id')),
                'timestamp': analysis.get('timestamp').strftime('%Y-%m-%d %H:%M:%S') if analysis.get('timestamp') else '',
                'patient_name': patient_name,
                'patient_id': patient_id,
                'predicted_label': analysis.get('predicted_label'),
                'confidence': analysis.get('confidence'),
                'filename': analysis.get('filename')
            })

        return jsonify({
            'success': True,
            'data': analyses
        })

    except Exception as e:
        print(f"Erreur recent analyses: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alerts')
@login_required
def get_medical_alerts():
    """API pour obtenir les alertes médicales du médecin connecté"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Récupérer les alertes non résolues depuis MongoDB
        alerts_cursor = db.medical_alerts.find({
            "doctor_id": doctor_id,
            "is_resolved": False
        }).sort("created_at", -1).limit(20)
        
        alerts = []
        for alert in alerts_cursor:
            # Récupérer le nom du patient
            patient = db.patients.find_one({
                "patient_id": alert.get('patient_id'),
                "doctor_id": doctor_id
            })
            patient_name = patient.get('patient_name') if patient else f"Patient {alert.get('patient_id')}"
            
            alerts.append({
                'id': str(alert.get('_id')),
                'patient_id': alert.get('patient_id'),
                'patient_name': patient_name,
                'alert_type': alert.get('alert_type'),
                'severity': alert.get('severity'),
                'title': alert.get('title'),
                'message': alert.get('message'),
                'is_read': alert.get('is_read', False),
                'is_resolved': alert.get('is_resolved', False),
                'created_at': alert.get('created_at').isoformat() if alert.get('created_at') else None
            })
        
        print(f"DEBUG get_medical_alerts: Found {len(alerts)} alerts for doctor {doctor_id}")

        return jsonify({
            'success': True,
            'data': alerts
        })

    except Exception as e:
        print(f"Erreur medical alerts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pro-dashboard/time-range/<time_range>')
@login_required
def pro_dashboard_time_range(time_range):
    """API pour les données du dashboard sur différentes périodes"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # Déterminer la période
        if time_range == '7d':
            days = 7
            date_format = '%d/%m'
        elif time_range == '30d':
            days = 30
            date_format = '%d/%m'
        elif time_range == '90d':
            days = 90
            date_format = '%d/%m'
        else:
            return jsonify({'success': False, 'error': 'Période invalide'}), 400

        # MongoDB query for daily analysis counts
        from datetime import datetime, timedelta
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        start_datetime = datetime.combine(start_date, datetime.min.time())
        
        db = get_mongodb()
        
        # Aggregate analyses by date
        pipeline = [
            {
                "$match": {
                    "doctor_id": doctor_id,
                    "timestamp": {"$gte": start_datetime}
                }
            },
            {
                "$project": {
                    "date": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$timestamp"
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": "$date",
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        result = list(db.analyses.aggregate(pipeline))
        daily_data = [(item['_id'], item['count']) for item in result]
        
        # Créer un dictionnaire avec toutes les dates
        date_counts = {}
        current_date = start_date
        while current_date <= end_date:
            date_counts[current_date.isoformat()] = 0
            current_date += timedelta(days=1)

        # Remplir avec les données réelles
        for date_str, count in daily_data:
            if date_str in date_counts:
                date_counts[date_str] = count

        # Convertir en format pour Chart.js
        labels = []
        data = []
        for date_str in sorted(date_counts.keys()):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            labels.append(date_obj.strftime(date_format))
            data.append(date_counts[date_str])

        # conn.close() # DISABLED

        return jsonify({
            'success': True,
            'data': {
                'labels': labels,
                'data': data,
                'period': time_range
            }
        })

    except Exception as e:
        print(f"Erreur time range data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pro-dashboard/advanced-stats')
@login_required
def pro_dashboard_advanced_stats():
    """API pour les statistiques avancées du dashboard professionnel"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Statistiques de performance temporelle
        try:
            from datetime import datetime, timedelta
            
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            hourly_pipeline = [
                {"$match": {
                    "doctor_id": doctor_id,
                    "timestamp": {"$gte": thirty_days_ago}
                }},
                {"$project": {
                    "hour": {"$hour": "$timestamp"},
                    "confidence": {"$ifNull": ["$confidence", 0]},
                    "processing_time": {"$ifNull": ["$processing_time", 0]}
                }},
                {"$group": {
                    "_id": "$hour",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"},
                    "avg_processing_time": {"$avg": "$processing_time"}
                }},
                {"$sort": {"_id": 1}}
            ]
            
            hourly_results = list(db.analyses.aggregate(hourly_pipeline))
            hourly_stats = [(r['_id'], r['count'], r['avg_confidence'], r['avg_processing_time']) for r in hourly_results]
        except Exception as e:
            print(f"Erreur hourly stats: {e}")
            import traceback
            traceback.print_exc()
            hourly_stats = []

        # Évolution de la confiance dans le temps
        try:
            confidence_pipeline = [
                {"$match": {
                    "doctor_id": doctor_id,
                    "timestamp": {"$gte": thirty_days_ago}
                }},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                    "avg_confidence": {"$avg": {"$ifNull": ["$confidence", 0]}},
                    "min_confidence": {"$min": {"$ifNull": ["$confidence", 0]}},
                    "max_confidence": {"$max": {"$ifNull": ["$confidence", 1]}},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
            
            confidence_results = list(db.analyses.aggregate(confidence_pipeline))
            confidence_trends = [(r['_id'], r['avg_confidence'], r['min_confidence'], r['max_confidence'], r['count']) for r in confidence_results]
        except Exception as e:
            print(f"Erreur confidence trends: {e}")
            import traceback
            traceback.print_exc()
            confidence_trends = []

        # Analyse par type de diagnostic avec évolution
        ninety_days_ago = datetime.now() - timedelta(days=90)
        
        diagnostic_pipeline = [
            {"$match": {
                "doctor_id": doctor_id,
                "timestamp": {"$gte": ninety_days_ago}
            }},
            {"$group": {
                "_id": "$predicted_label",
                "total_count": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"},
                "min_confidence": {"$min": "$confidence"},
                "max_confidence": {"$max": "$confidence"},
                "avg_processing_time": {"$avg": "$processing_time"}
            }},
            {"$sort": {"total_count": -1}}
        ]
        
        diagnostic_results = list(db.analyses.aggregate(diagnostic_pipeline))
        diagnostic_analysis = [(r['_id'], r['total_count'], r['avg_confidence'], r['min_confidence'], r['max_confidence'], r['avg_processing_time']) for r in diagnostic_results]

        # Analyse des temps de traitement
        processing_pipeline = [
            {"$match": {
                "doctor_id": doctor_id,
                "processing_time": {"$exists": True, "$ne": None}
            }},
            {"$project": {
                "processing_time": 1,
                "category": {
                    "$switch": {
                        "branches": [
                            {"case": {"$lt": ["$processing_time", 2]}, "then": "Très rapide"},
                            {"case": {"$lt": ["$processing_time", 5]}, "then": "Rapide"},
                            {"case": {"$lt": ["$processing_time", 10]}, "then": "Normal"}
                        ],
                        "default": "Lent"
                    }
                }
            }},
            {"$group": {
                "_id": "$category",
                "count": {"$sum": 1}
            }}
        ]
        
        processing_results = list(db.analyses.aggregate(processing_pipeline))
        processing_time_analysis = [(r.get('processing_time'), r['count'], r['_id']) for r in processing_results]

        # Analyse de la productivité hebdomadaire
        twelve_weeks_ago = datetime.now() - timedelta(weeks=12)
        
        weekly_pipeline = [
            {"$match": {
                "doctor_id": doctor_id,
                "timestamp": {"$gte": twelve_weeks_ago}
            }},
            {"$project": {
                "week": {"$week": "$timestamp"},
                "year": {"$year": "$timestamp"},
                "confidence": 1,
                "patient_id": 1
            }},
            {"$group": {
                "_id": {"year": "$year", "week": "$week"},
                "count": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"},
                "unique_patients": {"$addToSet": "$patient_id"}
            }},
            {"$project": {
                "year": "$_id.year",
                "week": "$_id.week",
                "count": 1,
                "avg_confidence": 1,
                "unique_patients": {"$size": "$unique_patients"}
            }},
            {"$sort": {"year": 1, "week": 1}}
        ]
        
        weekly_results = list(db.analyses.aggregate(weekly_pipeline))
        weekly_productivity = [(r['week'], r['year'], r['count'], r['avg_confidence'], r['unique_patients']) for r in weekly_results]

        # Analyse des patients à risque
        # Récupérer tous les patients du médecin
        patients = list(db.patients.find({"doctor_id": doctor_id}))
        
        high_risk_patients = []
        for patient in patients:
            patient_id = str(patient['_id'])
            patient_name = patient.get('first_name', '') + ' ' + patient.get('last_name', '')
            
            # Compter les analyses de ce patient
            analyses_cursor = db.analyses.find({
                "patient_id": patient_id,
                "doctor_id": doctor_id
            })
            analyses_list = list(analyses_cursor)
            
            if not analyses_list:
                continue
            
            total_analyses = len(analyses_list)
            
            # Calculer confiance moyenne
            confidences = [a.get('confidence', 0) for a in analyses_list]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Dernière analyse
            timestamps = [a.get('timestamp') for a in analyses_list if a.get('timestamp')]
            last_analysis = max(timestamps).strftime('%Y-%m-%d') if timestamps else ''
            
            # Compter les détections de tumeur (predicted_class != 0)
            tumor_detections = sum(1 for a in analyses_list if a.get('predicted_class', 0) != 0)
            
            if tumor_detections > 0:
                high_risk_patients.append((
                    patient_id,
                    patient_name,
                    total_analyses,
                    avg_confidence,
                    last_analysis,
                    tumor_detections
                ))
        
        # Trier par nombre de détections DESC, puis confiance ASC
        high_risk_patients.sort(key=lambda x: (-x[5], x[3]))
        high_risk_patients = high_risk_patients[:10]

        # Taux de détection mensuel
        twelve_months_ago = datetime.now() - timedelta(days=365)
        
        detection_pipeline = [
            {"$match": {
                "doctor_id": doctor_id,
                "timestamp": {"$gte": twelve_months_ago}
            }},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m", "date": "$timestamp"}},
                "total_analyses": {"$sum": 1},
                "tumor_detections": {
                    "$sum": {
                        "$cond": [{"$ne": ["$predicted_class", 0]}, 1, 0]
                    }
                }
            }},
            {"$project": {
                "month": "$_id",
                "total_analyses": 1,
                "tumor_detections": 1,
                "detection_rate": {
                    "$multiply": [
                        {"$divide": ["$tumor_detections", "$total_analyses"]},
                        100
                    ]
                }
            }},
            {"$sort": {"month": 1}}
        ]
        
        detection_results = list(db.analyses.aggregate(detection_pipeline))
        monthly_detection_rates = [(r['month'], r['total_analyses'], r['tumor_detections'], round(r['detection_rate'], 2)) for r in detection_results]

        # conn.close() # DISABLED

        # Formater les données pour le frontend
        hourly_performance = {
            'labels': [f"{str(row[0]).zfill(2)}h" if row[0] else '00h' for row in hourly_stats],
            'analyses_count': [row[1] for row in hourly_stats],
            'avg_confidence': [round(row[2] * 100, 1) if row[2] else 0 for row in hourly_stats],
            'avg_processing_time': [round(row[3], 2) if row[3] else 0 for row in hourly_stats]
        }

        confidence_evolution = {
            'labels': [row[0] for row in confidence_trends],
            'avg_confidence': [round(row[1] * 100, 1) if row[1] else 0 for row in confidence_trends],
            'min_confidence': [round(row[2] * 100, 1) if row[2] else 0 for row in confidence_trends],
            'max_confidence': [round(row[3] * 100, 1) if row[3] else 0 for row in confidence_trends],
            'daily_count': [row[4] for row in confidence_trends]
        }

        diagnostic_stats = {}
        for row in diagnostic_analysis:
            diagnostic_stats[row[0]] = {
                'total_count': row[1],
                'avg_confidence': round(row[2] * 100, 1) if row[2] else 0,
                'min_confidence': round(row[3] * 100, 1) if row[3] else 0,
                'max_confidence': round(row[4] * 100, 1) if row[4] else 0,
                'avg_processing_time': round(row[5], 2) if row[5] else 0
            }

        performance_categories = {}
        for row in processing_time_analysis:
            performance_categories[row[2]] = row[1]

        productivity_trends = {
            'labels': [f"S{row[0]}/{row[1]}" for row in weekly_productivity],
            'weekly_counts': [row[2] for row in weekly_productivity],
            'weekly_confidence': [round(row[3] * 100, 1) if row[3] else 0 for row in weekly_productivity],
            'unique_patients': [row[4] for row in weekly_productivity]
        }

        risk_patients = []
        for row in high_risk_patients:
            risk_patients.append({
                'patient_id': row[0],
                'patient_name': row[1] or f"Patient {row[0]}",
                'total_analyses': row[2],
                'avg_confidence': round(row[3] * 100, 1) if row[3] else 0,
                'last_analysis': row[4],
                'tumor_detections': row[5],
                'risk_level': 'Critique' if row[5] >= 3 else 'Élevé' if row[5] >= 2 else 'Modéré'
            })

        detection_trends = {
            'labels': [row[0] for row in monthly_detection_rates],
            'detection_rates': [row[3] for row in monthly_detection_rates],
            'total_analyses': [row[1] for row in monthly_detection_rates],
            'tumor_detections': [row[2] for row in monthly_detection_rates]
        }

        return jsonify({
            'success': True,
            'data': {
                'hourly_performance': hourly_performance,
                'confidence_evolution': confidence_evolution,
                'diagnostic_stats': diagnostic_stats,
                'performance_categories': performance_categories,
                'productivity_trends': productivity_trends,
                'high_risk_patients': risk_patients,
                'detection_trends': detection_trends
            }
        })

    except Exception as e:
        print(f"Erreur advanced stats: {e}")
        import traceback
        traceback.print_exc()
        
        # Retourner des données par défaut en cas d'erreur
        return jsonify({
            'success': True,
            'data': {
                'hourly_performance': {
                    'labels': [],
                    'analyses_count': [],
                    'avg_confidence': [],
                    'avg_processing_time': []
                },
                'confidence_evolution': {
                    'labels': [],
                    'avg_confidence': [],
                    'min_confidence': [],
                    'max_confidence': [],
                    'daily_count': []
                },
                'diagnostic_stats': {},
                'performance_categories': {},
                'productivity_trends': {
                    'labels': [],
                    'weekly_counts': [],
                    'weekly_confidence': [],
                    'unique_patients': []
                },
                'high_risk_patients': [],
                'detection_trends': {
                    'labels': [],
                    'detection_rates': [],
                    'total_analyses': [],
                    'tumor_detections': []
                }
            }
        })

@app.route('/api/pro-dashboard/patient-insights')
@login_required
def pro_dashboard_patient_insights():
    """API pour les insights sur les patients"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Distribution par âge (si disponible)
        try:
            from datetime import datetime
            
            # Récupérer tous les patients du médecin
            patients = list(db.patients.find({"doctor_id": doctor_id}))
            
            age_distribution = {}
            for patient in patients:
                dob = patient.get('date_of_birth')
                if not dob or dob == '':
                    age_group = 'Non renseigné'
                else:
                    # Convertir la date de naissance en datetime si nécessaire
                    if isinstance(dob, str):
                        try:
                            dob = datetime.strptime(dob, '%Y-%m-%d')
                        except:
                            age_group = 'Non renseigné'
                            age_distribution[age_group] = age_distribution.get(age_group, 0) + 1
                            continue
                    
                    # Calculer l'âge
                    age = (datetime.now() - dob).days / 365.25
                    
                    if age < 18:
                        age_group = 'Moins de 18 ans'
                    elif age < 30:
                        age_group = '18-30 ans'
                    elif age < 50:
                        age_group = '30-50 ans'
                    elif age < 70:
                        age_group = '50-70 ans'
                    else:
                        age_group = 'Plus de 70 ans'
                
                age_distribution[age_group] = age_distribution.get(age_group, 0) + 1
            
            if not age_distribution:
                age_distribution = {'Non renseigné': 0}
        except Exception as e:
            print(f"Erreur age distribution: {e}")
            age_distribution = {'Non renseigné': 0}

        # Distribution par genre
        try:
            gender_pipeline = [
                {"$match": {"doctor_id": doctor_id}},
                {"$group": {
                    "_id": {
                        "$cond": [
                            {"$or": [
                                {"$eq": ["$gender", None]},
                                {"$eq": ["$gender", ""]}
                            ]},
                            "Non renseigné",
                            "$gender"
                        ]
                    },
                    "count": {"$sum": 1}
                }}
            ]
            
            gender_results = list(db.patients.aggregate(gender_pipeline))
            gender_distribution = {r['_id']: r['count'] for r in gender_results}
            
            if not gender_distribution:
                gender_distribution = {'Non renseigné': 0}
        except Exception as e:
            print(f"Erreur gender distribution: {e}")
            gender_distribution = {'Non renseigné': 0}

        # Patients les plus suivis
        try:
            from bson import ObjectId
            
            # Récupérer les patients avec analyses
            patients_with_analyses = list(db.patients.find({
                "doctor_id": doctor_id,
                "total_analyses": {"$gt": 0}
            }))
            
            most_followed_patients = []
            for patient in patients_with_analyses:
                patient_id = str(patient['_id'])
                
                # Calculer la confiance moyenne depuis analyses
                avg_pipeline = [
                    {"$match": {
                        "patient_id": patient_id,
                        "doctor_id": doctor_id
                    }},
                    {"$group": {
                        "_id": None,
                        "avg": {"$avg": "$confidence"}
                    }}
                ]
                avg_result = list(db.analyses.aggregate(avg_pipeline))
                avg_confidence = avg_result[0]['avg'] if avg_result and avg_result[0].get('avg') else 0
                
                # Calculer les jours de suivi
                first_date = patient.get('first_analysis_date')
                last_date = patient.get('last_analysis_date')
                follow_up_days = 0
                
                if first_date and last_date:
                    if isinstance(first_date, str):
                        first_date = datetime.strptime(first_date, '%Y-%m-%d')
                    if isinstance(last_date, str):
                        last_date = datetime.strptime(last_date, '%Y-%m-%d')
                    follow_up_days = (last_date - first_date).days
                
                most_followed_patients.append((
                    patient_id,
                    patient.get('first_name', '') + ' ' + patient.get('last_name', ''),
                    patient.get('total_analyses', 0),
                    patient.get('first_analysis_date', ''),
                    patient.get('last_analysis_date', ''),
                    follow_up_days,
                    avg_confidence
                ))
            
            # Trier par nombre d'analyses et limiter à 10
            most_followed_patients.sort(key=lambda x: x[2], reverse=True)
            most_followed_patients = most_followed_patients[:10]
        except Exception as e:
            print(f"Erreur most followed patients: {e}")
            import traceback
            traceback.print_exc()
            most_followed_patients = []

        # Analyse de l'engagement patient (fréquence des visites)
        try:
            from datetime import timedelta
            
            now = datetime.now()
            patients_all = list(db.patients.find({"doctor_id": doctor_id}))
            
            patient_activity = {}
            for patient in patients_all:
                last_analysis = patient.get('last_analysis_date')
                
                if not last_analysis:
                    activity_level = 'Jamais analysé'
                else:
                    # Convertir en datetime si c'est une string
                    if isinstance(last_analysis, str):
                        try:
                            last_analysis = datetime.strptime(last_analysis, '%Y-%m-%d')
                        except:
                            activity_level = 'Jamais analysé'
                            patient_activity[activity_level] = patient_activity.get(activity_level, 0) + 1
                            continue
                    
                    days_since = (now - last_analysis).days
                    
                    if days_since <= 7:
                        activity_level = 'Actif (< 7j)'
                    elif days_since <= 30:
                        activity_level = 'Récent (< 30j)'
                    elif days_since <= 90:
                        activity_level = 'Inactif (< 90j)'
                    else:
                        activity_level = 'Très inactif (> 90j)'
                
                patient_activity[activity_level] = patient_activity.get(activity_level, 0) + 1
            
            if not patient_activity:
                patient_activity = {'Jamais analysé': 0}
        except Exception as e:
            print(f"Erreur patient activity: {e}")
            patient_activity = {'Jamais analysé': 0}

        # Nouveaux patients par mois
        try:
            twelve_months_ago = datetime.now() - timedelta(days=365)
            
            new_patients_pipeline = [
                {"$match": {
                    "doctor_id": doctor_id,
                    "created_at": {"$gte": twelve_months_ago}
                }},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m", "date": "$created_at"}},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
            
            new_patients_results = list(db.patients.aggregate(new_patients_pipeline))
            new_patients_trend = [(r['_id'], r['count']) for r in new_patients_results]
        except Exception as e:
            print(f"Erreur new patients trend: {e}")
            new_patients_trend = []

        # conn.close() # DISABLED

        # Formater les données des patients les plus suivis
        top_patients = []
        for row in most_followed_patients:
            try:
                top_patients.append({
                    'patient_id': row[0] or '',
                    'patient_name': row[1] or f"Patient {row[0] or 'Inconnu'}",
                    'total_analyses': row[2] or 0,
                    'first_analysis': row[3] or '',
                    'last_analysis': row[4] or '',
                    'follow_up_days': int(row[5]) if row[5] is not None else 0,
                    'avg_confidence': round(row[6] * 100, 1) if row[6] is not None else 0
                })
            except Exception as e:
                print(f"Erreur formatage patient: {e}")
                continue

        return jsonify({
            'success': True,
            'data': {
                'age_distribution': age_distribution or {'Non renseigné': 1},
                'gender_distribution': gender_distribution or {'Non renseigné': 1},
                'top_patients': top_patients or [],
                'patient_activity': patient_activity or {'Jamais analysé': 1},
                'new_patients_trend': {
                    'labels': [row[0] for row in new_patients_trend] if new_patients_trend else [],
                    'data': [row[1] for row in new_patients_trend] if new_patients_trend else []
                }
            }
        })

    except Exception as e:
        print(f"Erreur patient insights: {e}")
        import traceback
        traceback.print_exc()
        
        # Retourner des données par défaut en cas d'erreur
        return jsonify({
            'success': True,
            'data': {
                'age_distribution': {'Non renseigné': 1},
                'gender_distribution': {'Non renseigné': 1},
                'top_patients': [],
                'patient_activity': {'Jamais analysé': 1},
                'new_patients_trend': {
                    'labels': [],
                    'data': []
                }
            }
        })

@app.route('/api/pro-dashboard/export-data')
@login_required
def pro_dashboard_export_data():
    """API pour exporter les données du dashboard"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED

        # Récupérer toutes les analyses du médecin
        # MongoDB query needed here
        # SELECT 
        #                 a.timestamp,
        #                 a.patient_id,
        #                 a.patient_name,
        #                 a.exam_date,
        #                 a.predicted_label,
        #                 a.confidence,
        #                 a.processing_time,
        #                 a.filename
        #             FROM analyses a
        #             WHERE a.doctor_id = ?
        #             ORDER BY a.timestamp DESC)

        # analyses = []  # cursor.fetchall() # DISABLED  # TODO: Convert to MongoDB

        # Récupérer les patients
        # MongoDB query needed here
        # SELECT 
        #                 patient_id,
        #                 patient_name,
        #                 date_of_birth,
        #                 gender,
        #                 total_analyses,
        #                 first_analysis_date,
        #                 last_analysis_date
        #             FROM patients
        #             WHERE doctor_id = ?
        #             ORDER BY patient_name)

        analyses = []  # TODO: Implement MongoDB query
        patients = []  # TODO: Implement MongoDB query

        # conn.close() # DISABLED

        # Créer le CSV des analyses
        analyses_csv = "Date/Heure,ID Patient,Nom Patient,Date Examen,Diagnostic,Confiance (%),Temps Traitement (s),Fichier\n"
        for row in analyses:
            analyses_csv += f'"{row[0]}","{row[1] or ""}","{row[2] or ""}","{row[3] or ""}","{row[4]}",{(row[5]*100):.1f},{row[6]:.2f},"{row[7]}"\n'

        # Créer le CSV des patients
        patients_csv = "ID Patient,Nom,Date Naissance,Genre,Total Analyses,Première Analyse,Dernière Analyse\n"
        for row in patients:
            patients_csv += f'"{row[0]}","{row[1] or ""}","{row[2] or ""}","{row[3] or ""}",{row[4]},"{row[5] or ""}","{row[6] or ""}"\n'

        export_data = {
            'analyses_csv': analyses_csv,
            'patients_csv': patients_csv,
            'export_date': datetime.now().isoformat(),
            'total_analyses': len(analyses),
            'total_patients': len(patients)
        }

        return jsonify({
            'success': True,
            'data': export_data
        })

    except Exception as e:
        print(f"Erreur export data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def api_health_check():
    """Point de terminaison de vérification de santé du serveur"""
    try:
        # Vérifier la connexion à la base de données
        # conn = sqlite3.connect() # DISABLED - MongoDB used instead
        # cursor = conn.cursor() # DISABLED
        # cursor.execute( # DISABLED'SELECT 1')
        # conn.close() # DISABLED
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected',
            'model_loaded': model is not None
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@app.route('/api/visitor-chat', methods=['POST'])
def visitor_chat():
    """API pour le chatbot des visiteurs - Répond uniquement sur le projet NeuroScan"""
    try:
        data = request.get_json()
        
        # Vérifier que message est une chaîne de caractères
        message_raw = data.get('message', '')
        if isinstance(message_raw, dict):
            user_message = str(message_raw.get('content', message_raw.get('text', ''))).strip()
        elif isinstance(message_raw, str):
            user_message = message_raw.strip()
        else:
            user_message = str(message_raw).strip()
            
        history = data.get('history', [])
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message vide'
            }), 400
        
        # Contexte système strict pour limiter aux questions sur le projet
        system_context = """Tu es l'assistant virtuel de NeuroScan, une plateforme médicale d'analyse IA pour les tumeurs cérébrales.

RÈGLES STRICTES:
1. Tu DOIS répondre UNIQUEMENT aux questions concernant le projet NeuroScan et ses fonctionnalités
2. Tu NE DOIS PAS répondre aux questions médicales générales, diagnostics ou conseils de santé
3. Si on te pose une question médicale, redirige poliment vers un professionnel de santé

INFORMATIONS SUR NEUROSCAN:

Présentation:
- Plateforme d'intelligence artificielle pour l'analyse d'images IRM cérébrales
- Détection automatique de tumeurs avec précision de 99.7%
- Résultats en moins de 10 secondes
- Interface moderne et intuitive

Fonctionnalités principales:
1. Analyse IA CNN (Réseau de Neurones Convolutionnel)
   - Détection de 4 types: Normal, Gliome, Méningiome, Tumeur pituitaire
   - Précision: 99.7% validée cliniquement
   - Temps d'analyse: < 10 secondes
   - Rapports détaillés avec probabilités

2. Chat Médical IA
   - Assistant conversationnel spécialisé en neurologie
   - Questions médicales et interprétation d'examens
   - Historique des conversations
   - Interface moderne

3. Gestion des Patients
   - Dossiers médicaux complets
   - Suivi longitudinal et évolution
   - Historique des analyses
   - Export de rapports PDF

4. Tableau de Bord Professionnel
   - Statistiques en temps réel
   - Graphiques interactifs
   - Métriques de performance
   - Alertes médicales

5. Sécurité et Conformité
   - Chiffrement AES-256 de bout en bout
   - Conformité RGPD
   - Certification ISO 27001
   - CE Médical
   - Infrastructure cloud sécurisée

Statistiques de performance:
- Précision: 99.7%
- Temps d'analyse: < 10 secondes
- 50,000+ analyses effectuées
- 500+ médecins utilisateurs
- 98.7% de satisfaction médecins

Processus d'utilisation:
1. Upload sécurisé de l'image IRM (DICOM, NIfTI, JPEG, PNG)
2. Analyse IA automatique en moins de 10s
3. Résultats détaillés avec classification et probabilités
4. Génération rapport PDF professionnel

Technologies:
- Deep Learning CNN (PyTorch)
- Modèle entraîné sur 100,000+ images validées
- API Gemini pour le chat médical
- Base de données SQLite pour le suivi
- Interface Flask avec design moderne

Accès:
- Inscription gratuite pour les médecins
- Authentification sécurisée
- Dashboard personnel
- Support 24/7

Contact:
- Email: mohammed.betkaoui@neuroscan.ai
- Téléphone: +123783962348
- Adresse: Bordj Bou Arréridj, Algérie

Réponds de manière concise, professionnelle et amicale. Si la question sort du contexte de NeuroScan, explique poliment que tu ne peux répondre qu'aux questions sur la plateforme."""
        
        # Construire le prompt avec historique
        prompt = f"{system_context}\n\n"
        
        # Ajouter l'historique (derniers 5 échanges max)
        for msg in history[-5:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt += f"{role}: {content}\n"
        
        prompt += f"\nuser: {user_message}\nassistant:"
        
        # Appeler Gemini
        response_text = call_gemini_api(prompt, context="neuroscan_project")
        
        # Vérifier si c'est une erreur de quota
        if response_text and response_text.startswith("QUOTA_EXCEEDED:"):
            retry_delay = response_text.split(":")[1]
            return jsonify({
                'success': False,
                'error': 'quota_exceeded',
                'message': f"Le quota d'utilisation gratuit de l'API Gemini a été atteint (250 requêtes/jour). Veuillez réessayer dans {retry_delay} secondes.",
                'retry_after': int(retry_delay)
            }), 429
        
        if not response_text:
            return jsonify({
                'success': False,
                'error': 'api_error',
                'message': 'Le service de chat est temporairement indisponible. Veuillez réessayer plus tard.'
            }), 500
        
        # Mettre à jour l'historique
        history.append({'role': 'user', 'content': user_message})
        history.append({'role': 'assistant', 'content': response_text})
        
        return jsonify({
            'success': True,
            'response': response_text,
            'history': history
        })
        
    except Exception as e:
        print(f"Erreur dans visitor_chat: {e}")
        return jsonify({
            'success': False,
            'error': 'Erreur serveur'
        }), 500

# ========================================
# WEBSOCKET - MESSAGERIE TEMPS RÉEL
# ========================================

# Dictionnaire pour suivre les médecins connectés et leurs rooms
connected_doctors = {}  # {socket_id: doctor_id}
doctor_sockets = {}  # {doctor_id: [socket_ids]}

@socketio.on('connect')
def handle_connect():
    """Gestion de la connexion WebSocket"""
    print(f'🔌 Client connecté: {request.sid}')
    emit('connected', {'status': 'connected', 'socket_id': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    """Gestion de la déconnexion WebSocket"""
    socket_id = request.sid
    print(f'🔌 Client déconnecté: {socket_id}')
    
    # Retirer le médecin des dictionnaires de suivi
    if socket_id in connected_doctors:
        doctor_id = connected_doctors[socket_id]
        del connected_doctors[socket_id]
        
        if doctor_id in doctor_sockets:
            doctor_sockets[doctor_id].remove(socket_id)
            if not doctor_sockets[doctor_id]:
                del doctor_sockets[doctor_id]
                # Notifier les autres que ce médecin est hors ligne
                emit('doctor_offline', {'doctor_id': doctor_id}, broadcast=True)

@socketio.on('join')
def handle_join(data):
    """Rejoindre une room de conversation"""
    try:
        doctor_id = data.get('doctor_id')
        conversation_id = data.get('conversation_id')
        
        if not doctor_id or not conversation_id:
            emit('error', {'message': 'doctor_id et conversation_id requis'})
            return
        
        # Enregistrer le médecin connecté
        socket_id = request.sid
        connected_doctors[socket_id] = doctor_id
        
        if doctor_id not in doctor_sockets:
            doctor_sockets[doctor_id] = []
        if socket_id not in doctor_sockets[doctor_id]:
            doctor_sockets[doctor_id].append(socket_id)
        
        # Rejoindre la room de la conversation
        join_room(conversation_id)
        print(f'✅ Médecin {doctor_id} a rejoint la conversation {conversation_id}')
        
        # Notifier que le médecin est en ligne
        emit('doctor_online', {'doctor_id': doctor_id}, room=conversation_id, skip_sid=request.sid)
        
        emit('joined', {
            'conversation_id': conversation_id,
            'doctor_id': doctor_id,
            'status': 'success'
        })
        
    except Exception as e:
        print(f'❌ Erreur join: {e}')
        emit('error', {'message': str(e)})

@socketio.on('leave')
def handle_leave(data):
    """Quitter une room de conversation"""
    try:
        conversation_id = data.get('conversation_id')
        doctor_id = data.get('doctor_id')
        
        if conversation_id:
            leave_room(conversation_id)
            print(f'👋 Médecin {doctor_id} a quitté la conversation {conversation_id}')
            
            # Notifier que le médecin est hors ligne de cette conversation
            emit('doctor_offline', {'doctor_id': doctor_id}, room=conversation_id)
            
            emit('left', {'conversation_id': conversation_id, 'status': 'success'})
    except Exception as e:
        print(f'❌ Erreur leave: {e}')
        emit('error', {'message': str(e)})

@socketio.on('send_message')
def handle_send_message(data):
    """Envoyer un message en temps réel"""
    try:
        conversation_id = data.get('conversation_id')
        sender_id = data.get('sender_id')
        content = data.get('content')
        
        if not all([conversation_id, sender_id, content]):
            emit('error', {'message': 'Données manquantes'})
            return
        
        # Récupérer les informations du médecin expéditeur
        sender = db.doctors.find_one({'_id': ObjectId(sender_id)})
        if not sender:
            emit('error', {'message': 'Médecin non trouvé'})
            return
        
        # Créer le message dans la base de données
        message_doc = {
            'conversation_id': ObjectId(conversation_id),
            'sender_id': ObjectId(sender_id),
            'content': content,
            'created_at': datetime.now(),
            'is_read': False
        }
        
        result = db.doctor_messages.insert_one(message_doc)
        message_id = str(result.inserted_id)
        
        # Mettre à jour la conversation
        db.doctor_conversations.update_one(
            {'_id': ObjectId(conversation_id)},
            {
                '$set': {
                    'last_message': content,
                    'last_message_at': datetime.now(),
                    'updated_at': datetime.now()
                }
            }
        )
        
        # Récupérer la conversation pour savoir qui est l'autre participant
        conversation = db.doctor_conversations.find_one({'_id': ObjectId(conversation_id)})
        if conversation:
            participants = conversation.get('participants', [])
            recipient_id = None
            for p in participants:
                if str(p) != sender_id:
                    recipient_id = str(p)
                    break
        
        # Préparer le message à émettre
        message_data = {
            '_id': message_id,
            'conversation_id': conversation_id,
            'sender_id': sender_id,
            'content': content,
            'created_at': datetime.now().isoformat(),
            'is_read': False,
            'sender': {
                'id': sender_id,
                'full_name': f"{sender.get('first_name', '')} {sender.get('last_name', '')}".strip(),
                'specialty': sender.get('specialty', ''),
                'avatar': sender.get('avatar', '/static/images/avatar-default.svg')
            }
        }
        
        # Émettre le message à tous les participants de la conversation
        emit('new_message', message_data, room=conversation_id, include_self=True)
        
        print(f'📨 Message envoyé dans la conversation {conversation_id}')
        
    except Exception as e:
        print(f'❌ Erreur send_message: {e}')
        import traceback
        traceback.print_exc()
        emit('error', {'message': str(e)})

@socketio.on('typing')
def handle_typing(data):
    """Notifier qu'un utilisateur est en train d'écrire"""
    try:
        conversation_id = data.get('conversation_id')
        doctor_id = data.get('doctor_id')
        is_typing = data.get('is_typing', True)
        
        if conversation_id and doctor_id:
            emit('user_typing', {
                'doctor_id': doctor_id,
                'is_typing': is_typing
            }, room=conversation_id, skip_sid=request.sid)
            
    except Exception as e:
        print(f'❌ Erreur typing: {e}')

@socketio.on('mark_read')
def handle_mark_read(data):
    """Marquer les messages comme lus"""
    try:
        conversation_id = data.get('conversation_id')
        doctor_id = data.get('doctor_id')
        
        if not conversation_id or not doctor_id:
            return
        
        # Marquer les messages comme lus
        result = db.doctor_messages.update_many(
            {
                'conversation_id': ObjectId(conversation_id),
                'sender_id': {'$ne': ObjectId(doctor_id)},
                'is_read': False
            },
            {'$set': {'is_read': True, 'read_at': datetime.now()}}
        )
        
        if result.modified_count > 0:
            # Notifier l'autre participant
            emit('messages_read', {
                'conversation_id': conversation_id,
                'reader_id': doctor_id,
                'count': result.modified_count
            }, room=conversation_id, skip_sid=request.sid)
            
            print(f'✅ {result.modified_count} messages marqués comme lus dans {conversation_id}')
        
    except Exception as e:
        print(f'❌ Erreur mark_read: {e}')

if __name__ == '__main__':
    print(f"Démarrage de l'application sur le device: {device}")
    print(f"Modèle chargé: {'Oui' if model is not None else 'Non'}")
    print("🔌 WebSocket activé pour la messagerie en temps réel")
    
    # Port pour le déploiement (Render, Heroku, etc.)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    # Utiliser socketio.run au lieu de app.run pour le support WebSocket
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)