# -*- coding: utf-8 -*-
"""
NeuroScan AI - Application Web avec MongoDB (Version Minimale)
==============================================================
Version de base avec authentification MongoDB fonctionnelle
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session, redirect, url_for, flash
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

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best_brain_tumor_model.pth'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================================
# MODÈLE DE CLASSIFICATION (copié de app.py)
# ============================================================================

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        
        # Encoder path
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.pool4(x)
        
        x = self.adaptive_pool(x)
        x = x.view(-1, 512 * 7 * 7)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# Charger le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainTumorCNN(num_classes=4).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")

# Classes
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ============================================================================
# FONCTIONS D'INITIALISATION
# ============================================================================

def init_database():
    """Initialiser MongoDB au lieu de SQLite"""
    try:
        init_mongodb_collections()
        print("✅ MongoDB initialisé avec succès")
    except Exception as e:
        print(f"❌ Erreur initialisation MongoDB: {e}")

# ============================================================================
# DÉCORATEURS
# ============================================================================

def login_required(f):
    """Décorateur pour protéger les routes nécessitant une authentification"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'doctor_id' not in session or not session.get('logged_in'):
            flash('Vous devez être connecté pour accéder à cette page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# ROUTES D'AUTHENTIFICATION
# ============================================================================

@app.route('/')
def index():
    """Page d'accueil"""
    if 'doctor_id' in session and session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return render_template('index.html')

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
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))

    # Récupérer les statistiques du médecin
    doctor_stats = get_doctor_statistics_mongo(doctor['id'])
    
    # Ajouter les statistiques au contexte
    total_analyses = doctor_stats.get('total_analyses', 0)
    total_patients = doctor_stats.get('total_patients', 0)
    
    return render_template('dashboard.html', 
                         doctor=doctor, 
                         doctor_stats=doctor_stats,
                         total_analyses=total_analyses,
                         total_patients=total_patients)

@app.route('/pro-dashboard')
@login_required
def pro_dashboard():
    """Dashboard professionnel avancé"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    
    doctor_stats = get_doctor_statistics_mongo(doctor['id'])
    
    return render_template('pro_dashboard.html', 
                         doctor=doctor, 
                         doctor_stats=doctor_stats)

@app.route('/patients')
@login_required
def patients_list():
    """Liste des patients"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    
    # Récupérer la liste des patients depuis MongoDB
    try:
        db = get_mongodb()
        patients = db.patients
        patient_list = list(patients.find({'doctor_id': doctor['id']}).sort('last_exam_date', -1))
        
        # Convertir ObjectId en string pour le template
        for patient in patient_list:
            patient['_id'] = str(patient['_id'])
        
        return render_template('patients_list.html', 
                             doctor=doctor, 
                             patients=patient_list)
    except Exception as e:
        print(f"Erreur récupération patients: {e}")
        flash('Erreur lors de la récupération des patients', 'error')
        return render_template('patients_list.html', 
                             doctor=doctor, 
                             patients=[])

@app.route('/alerts')
@login_required
def medical_alerts_page():
    """Page des alertes médicales"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    
    # Récupérer les alertes depuis MongoDB
    try:
        db = get_mongodb()
        alerts = db.medical_alerts
        alert_list = list(alerts.find({'doctor_id': doctor['id']}).sort('created_at', -1).limit(50))
        
        # Convertir ObjectId en string
        for alert in alert_list:
            alert['_id'] = str(alert['_id'])
            if 'patient_id' in alert:
                alert['patient_id'] = str(alert['patient_id'])
        
        return render_template('alerts.html', 
                             doctor=doctor, 
                             alerts=alert_list)
    except Exception as e:
        print(f"Erreur récupération alertes: {e}")
        flash('Erreur lors de la récupération des alertes', 'error')
        return render_template('alerts.html', 
                             doctor=doctor, 
                             alerts=[])

@app.route('/manage-patients')
@login_required
def manage_patients_page():
    """Page de gestion des patients"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    return render_template('manage_patients.html', doctor=doctor)

@app.route('/medical-alerts')
@login_required
def medical_alerts():
    """Alias pour la page des alertes médicales"""
    return redirect(url_for('medical_alerts_page'))

@app.route('/nouvelle-analyse')
@login_required
def nouvelle_analyse():
    """Page de nouvelle analyse"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    return render_template('new_analysis.html', doctor=doctor)

@app.route('/platform-stats')
@login_required
def platform_stats():
    """Statistiques de la plateforme"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    
    doctor_stats = get_doctor_statistics_mongo(doctor['id'])
    return render_template('platform_stats.html', 
                         doctor=doctor, 
                         doctor_stats=doctor_stats)

@app.route('/tumor-tracking')
@login_required
def tumor_tracking():
    """Page de suivi des tumeurs"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    return render_template('tumor_tracking.html', doctor=doctor)

@app.route('/pro-dashboard-advanced')
@login_required
def pro_dashboard_advanced():
    """Dashboard professionnel avancé"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    
    doctor_stats = get_doctor_statistics_mongo(doctor['id'])
    return render_template('pro_dashboard_advanced.html', 
                         doctor=doctor, 
                         doctor_stats=doctor_stats)

@app.route('/patients/new')
@login_required
def new_patient():
    """Page d'ajout d'un nouveau patient"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    return render_template('new_patient.html', doctor=doctor)

@app.route('/patients/<patient_id>/edit')
@login_required
def edit_patient(patient_id):
    """Page d'édition d'un patient"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    
    # Récupérer les infos du patient
    try:
        db = get_mongodb()
        patients = db.patients
        patient = patients.find_one({
            'patient_id': patient_id,
            'doctor_id': doctor['id']
        })
        
        if patient:
            patient['_id'] = str(patient['_id'])
            return render_template('edit_patient.html', doctor=doctor, patient=patient)
        else:
            flash('Patient non trouvé', 'error')
            return redirect(url_for('patients_list'))
    except Exception as e:
        print(f"Erreur récupération patient: {e}")
        flash('Erreur lors de la récupération du patient', 'error')
        return redirect(url_for('patients_list'))

@app.route('/patient/<patient_id>')
@login_required
def patient_profile(patient_id):
    """Profil détaillé d'un patient"""
    doctor = get_current_doctor_mongo(session.get('doctor_id'))
    if not doctor:
        return redirect(url_for('login'))
    
    try:
        db = get_mongodb()
        patients = db.patients
        analyses = db.analyses
        
        # Récupérer le patient
        patient = patients.find_one({
            'patient_id': patient_id,
            'doctor_id': doctor['id']
        })
        
        if not patient:
            flash('Patient non trouvé', 'error')
            return redirect(url_for('patients_list'))
        
        # Récupérer les analyses du patient
        patient_analyses = list(analyses.find({
            'patient_id': patient_id,
            'doctor_id': doctor['id']
        }).sort('exam_date', -1))
        
        # Convertir les ObjectId
        patient['_id'] = str(patient['_id'])
        for analysis in patient_analyses:
            analysis['_id'] = str(analysis['_id'])
        
        return render_template('patient_profile_pro.html', 
                             doctor=doctor, 
                             patient=patient,
                             analyses=patient_analyses)
    except Exception as e:
        print(f"Erreur profil patient: {e}")
        flash('Erreur lors du chargement du profil', 'error')
        return redirect(url_for('patients_list'))

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/my-patients')
@login_required
def api_my_patients():
    """API pour récupérer les patients du médecin"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Non connecté'}), 401
        
        db = get_mongodb()
        patients = db.patients
        patient_list = list(patients.find({'doctor_id': doctor_id}).sort('last_exam_date', -1))
        
        # Convertir pour JSON
        for patient in patient_list:
            patient['_id'] = str(patient['_id'])
            if 'last_exam_date' in patient and hasattr(patient['last_exam_date'], 'isoformat'):
                patient['last_exam_date'] = patient['last_exam_date'].isoformat()
        
        return jsonify({'success': True, 'patients': patient_list})
    except Exception as e:
        print(f"Erreur API my patients: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/overview')
@login_required
def analytics_overview():
    """API pour les statistiques du médecin"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Non connecté'}), 401
        
        stats = get_doctor_statistics_mongo(doctor_id)
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        print(f"Erreur analytics overview: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alerts')
@login_required
def api_alerts():
    """API pour récupérer les alertes"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Non connecté'}), 401
        
        db = get_mongodb()
        alerts = db.medical_alerts
        alert_list = list(alerts.find({'doctor_id': doctor_id}).sort('created_at', -1).limit(20))
        
        # Convertir pour JSON
        for alert in alert_list:
            alert['_id'] = str(alert['_id'])
            if 'created_at' in alert and hasattr(alert['created_at'], 'isoformat'):
                alert['created_at'] = alert['created_at'].isoformat()
        
        return jsonify({'success': True, 'alerts': alert_list})
    except Exception as e:
        print(f"Erreur API alerts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/notifications')
@login_required
def api_notifications():
    """API pour récupérer les notifications"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Non connecté'}), 401
        
        db = get_mongodb()
        notifications = db.notifications
        notif_list = list(notifications.find({'doctor_id': doctor_id}).sort('created_at', -1).limit(20))
        
        # Convertir pour JSON
        for notif in notif_list:
            notif['_id'] = str(notif['_id'])
            if 'created_at' in notif and hasattr(notif['created_at'], 'isoformat'):
                notif['created_at'] = notif['created_at'].isoformat()
        
        return jsonify({'success': True, 'notifications': notif_list})
    except Exception as e:
        print(f"Erreur API notifications: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        db = get_mongodb()
        # Test de connexion
        db.command('ping')
        return jsonify({'status': 'healthy', 'database': 'connected'})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/health')
def api_health():
    """API Health check"""
    return health()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    init_database()
    
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    
    print(f"\n🚀 NeuroScan AI - Version MongoDB")
    print(f"📍 http://localhost:{port}")
    print(f"🐛 Debug mode: {debug}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
