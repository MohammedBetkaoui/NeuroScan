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
import sqlite3
import secrets
import random
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from torchvision import transforms
from functools import wraps

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'neuroscan_secret_key_2024_medical_auth'  # Clé secrète pour les sessions

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration de l'API Gemini
GEMINI_API_KEY = "AIzaSyBC3sAJjh9_32jTgKXJxcdOTM7HzyNJPng"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Configuration de la base de données
DATABASE_PATH = 'neuroscan_analytics.db'

def init_database():
    """Initialiser la base de données avec les tables nécessaires"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Table des analyses (modifiée pour le suivi temporel et relation médecin)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            patient_id TEXT,
            patient_name TEXT,
            exam_date DATE,
            predicted_class INTEGER,
            predicted_label TEXT,
            confidence REAL,
            probabilities TEXT,
            description TEXT,
            recommendations TEXT,
            processing_time REAL,
            user_session TEXT,
            ip_address TEXT,
            tumor_size_estimate REAL,
            previous_analysis_id INTEGER,
            doctor_id INTEGER NOT NULL,
            FOREIGN KEY (previous_analysis_id) REFERENCES analyses(id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(id)
        )
    ''')

    # Table des patients pour le suivi (avec relation médecin)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            patient_name TEXT,
            date_of_birth DATE,
            gender TEXT,
            first_analysis_date DATE,
            last_analysis_date DATE,
            total_analyses INTEGER DEFAULT 0,
            doctor_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(patient_id, doctor_id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(id)
        )
    ''')

    # Table pour l'évolution des tumeurs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tumor_evolution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            analysis_id INTEGER,
            exam_date DATE,
            diagnosis_change TEXT,
            confidence_change REAL,
            size_change REAL,
            evolution_type TEXT,
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id),
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')

    # Table des alertes médicales
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            doctor_id INTEGER NOT NULL,
            analysis_id INTEGER,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            is_read BOOLEAN DEFAULT 0,
            is_resolved BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at DATETIME,
            resolved_by INTEGER,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(id),
            FOREIGN KEY (analysis_id) REFERENCES analyses(id),
            FOREIGN KEY (resolved_by) REFERENCES doctors(id)
        )
    ''')

    # Table des notifications push
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            data TEXT,
            is_read BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doctor_id) REFERENCES doctors(id)
        )
    ''')

    # Table des statistiques quotidiennes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE,
            total_analyses INTEGER DEFAULT 0,
            normal_count INTEGER DEFAULT 0,
            gliome_count INTEGER DEFAULT 0,
            meningiome_count INTEGER DEFAULT 0,
            pituitary_count INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            avg_processing_time REAL DEFAULT 0
        )
    ''')

    # Table des sessions utilisateur
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            analyses_count INTEGER DEFAULT 0,
            ip_address TEXT,
            user_agent TEXT
        )
    ''')

    # Table des médecins (authentification)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            specialty TEXT,
            hospital TEXT,
            license_number TEXT,
            phone TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME,
            login_count INTEGER DEFAULT 0
        )
    ''')

    # Table des sessions de médecins
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctor_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (doctor_id) REFERENCES doctors(id)
        )
    ''')

    # Ajouter les colonnes manquantes aux tables existantes si nécessaire
    try:
        cursor.execute('ALTER TABLE analyses ADD COLUMN patient_id TEXT')
    except sqlite3.OperationalError:
        pass  # La colonne existe déjà

    try:
        cursor.execute('ALTER TABLE analyses ADD COLUMN patient_name TEXT')
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute('ALTER TABLE analyses ADD COLUMN exam_date DATE')
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute('ALTER TABLE analyses ADD COLUMN tumor_size_estimate REAL')
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute('ALTER TABLE analyses ADD COLUMN previous_analysis_id INTEGER')
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute('ALTER TABLE analyses ADD COLUMN doctor_id INTEGER')
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute('ALTER TABLE patients ADD COLUMN doctor_id INTEGER')
    except sqlite3.OperationalError:
        pass

    # Ajouter les colonnes manquantes pour les informations détaillées des patients
    additional_columns = [
        'phone TEXT',
        'email TEXT',
        'address TEXT',
        'emergency_contact_name TEXT',
        'emergency_contact_phone TEXT',
        'medical_history TEXT',
        'allergies TEXT',
        'current_medications TEXT',
        'insurance_number TEXT',
        'notes TEXT',
        'updated_at DATETIME'
    ]

    for column in additional_columns:
        try:
            cursor.execute(f'ALTER TABLE patients ADD COLUMN {column}')
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

def save_analysis_to_db(results, filename, processing_time, session_id=None, ip_address=None, patient_id=None, patient_name=None, exam_date=None, doctor_id=None):
    """Sauvegarder une analyse dans la base de données avec support du suivi temporel et relation médecin"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Convertir les probabilités en JSON
        probabilities_json = json.dumps(results['probabilities'])
        recommendations_json = json.dumps(results.get('recommendations', []))

        # Utiliser la date actuelle si exam_date n'est pas fournie
        if exam_date is None:
            exam_date = datetime.now().date()

        # Estimer la taille de la tumeur basée sur la confiance (simulation)
        tumor_size_estimate = None
        if results['predicted_class'] != 0:  # Si ce n'est pas normal
            # Simulation d'estimation de taille basée sur la confiance et le type
            base_size = {
                1: 2.5,  # Gliome
                2: 1.8,  # Méningiome
                3: 1.2   # Tumeur pituitaire
            }.get(results['predicted_class'], 2.0)
            tumor_size_estimate = base_size * results['confidence'] * (0.8 + 0.4 * np.random.random())

        # Vérifier que doctor_id est fourni
        if not doctor_id:
            print("Erreur: doctor_id requis pour sauvegarder l'analyse")
            return False

        # Trouver l'analyse précédente pour ce patient et ce médecin
        previous_analysis_id = None
        if patient_id:
            cursor.execute('''
                SELECT id FROM analyses
                WHERE patient_id = ? AND doctor_id = ?
                ORDER BY exam_date DESC, timestamp DESC
                LIMIT 1
            ''', (patient_id, doctor_id))
            prev_result = cursor.fetchone()
            if prev_result:
                previous_analysis_id = prev_result[0]

        cursor.execute('''
            INSERT INTO analyses
            (filename, patient_id, patient_name, exam_date, predicted_class, predicted_label,
             confidence, probabilities, description, recommendations, processing_time,
             user_session, ip_address, tumor_size_estimate, previous_analysis_id, doctor_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            patient_id,
            patient_name,
            exam_date,
            results['predicted_class'],
            results['predicted_label'],
            results['confidence'],
            probabilities_json,
            results.get('description', ''),
            recommendations_json,
            processing_time,
            session_id,
            ip_address,
            tumor_size_estimate,
            previous_analysis_id,
            doctor_id
        ))

        analysis_id = cursor.lastrowid

        # Gérer le patient (créer ou mettre à jour)
        if patient_id and doctor_id:
            manage_patient_record(cursor, patient_id, patient_name, exam_date, doctor_id)

        # Analyser l'évolution si il y a une analyse précédente et un patient_id
        if patient_id and previous_analysis_id:
            analyze_tumor_evolution(cursor, patient_id, analysis_id, previous_analysis_id, results, exam_date)
            # Créer des alertes si nécessaire
            create_medical_alerts(cursor, patient_id, analysis_id, results, doctor_id, previous_analysis_id)

        # Mettre à jour les statistiques quotidiennes
        today = datetime.now().date()
        cursor.execute('''
            INSERT OR IGNORE INTO daily_stats (date) VALUES (?)
        ''', (today,))

        # Incrémenter les compteurs
        label_column = {
            'Normal': 'normal_count',
            'Gliome': 'gliome_count',
            'Méningiome': 'meningiome_count',
            'Tumeur pituitaire': 'pituitary_count'
        }.get(results['predicted_label'], 'normal_count')

        cursor.execute(f'''
            UPDATE daily_stats
            SET total_analyses = total_analyses + 1,
                {label_column} = {label_column} + 1,
                avg_confidence = (avg_confidence * (total_analyses - 1) + ?) / total_analyses,
                avg_processing_time = (avg_processing_time * (total_analyses - 1) + ?) / total_analyses
            WHERE date = ?
        ''', (results['confidence'], processing_time, today))

        conn.commit()
        conn.close()
        return analysis_id
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        return False

def manage_patient_record(cursor, patient_id, patient_name, exam_date, doctor_id):
    """Gérer l'enregistrement du patient (créer ou mettre à jour)"""
    try:
        # Vérifier si le patient existe déjà pour ce médecin
        cursor.execute('''
            SELECT id, first_analysis_date, total_analyses
            FROM patients
            WHERE patient_id = ? AND doctor_id = ?
        ''', (patient_id, doctor_id))

        existing_patient = cursor.fetchone()

        if existing_patient:
            # Mettre à jour le patient existant
            cursor.execute('''
                UPDATE patients
                SET patient_name = ?,
                    last_analysis_date = ?,
                    total_analyses = total_analyses + 1
                WHERE patient_id = ? AND doctor_id = ?
            ''', (patient_name, exam_date, patient_id, doctor_id))
        else:
            # Créer un nouveau patient
            cursor.execute('''
                INSERT INTO patients
                (patient_id, patient_name, first_analysis_date, last_analysis_date, total_analyses, doctor_id)
                VALUES (?, ?, ?, ?, 1, ?)
            ''', (patient_id, patient_name, exam_date, exam_date, doctor_id))

    except Exception as e:
        print(f"Erreur lors de la gestion du patient: {e}")

def analyze_tumor_evolution(cursor, patient_id, current_analysis_id, previous_analysis_id, current_results, exam_date):
    """Analyser l'évolution d'une tumeur entre deux analyses"""
    try:
        # Récupérer l'analyse précédente
        cursor.execute('''
            SELECT predicted_label, confidence, tumor_size_estimate, exam_date
            FROM analyses WHERE id = ?
        ''', (previous_analysis_id,))
        prev_data = cursor.fetchone()

        if not prev_data:
            return

        prev_label, prev_confidence, prev_size, prev_date = prev_data

        # Analyser les changements
        diagnosis_change = None
        if prev_label != current_results['predicted_label']:
            diagnosis_change = f"{prev_label} → {current_results['predicted_label']}"

        confidence_change = current_results['confidence'] - prev_confidence

        size_change = None
        current_size = None
        if current_results['predicted_class'] != 0:  # Si tumeur détectée
            # Récupérer la taille estimée actuelle
            cursor.execute('SELECT tumor_size_estimate FROM analyses WHERE id = ?', (current_analysis_id,))
            size_result = cursor.fetchone()
            if size_result and size_result[0]:
                current_size = size_result[0]
                if prev_size:
                    size_change = current_size - prev_size

        # Déterminer le type d'évolution
        evolution_type = "stable"
        if diagnosis_change:
            if "Normal" in diagnosis_change:
                evolution_type = "amélioration" if current_results['predicted_label'] == "Normal" else "dégradation"
            else:
                evolution_type = "changement_type"
        elif size_change:
            if abs(size_change) > 0.2:  # Seuil de changement significatif
                evolution_type = "croissance" if size_change > 0 else "réduction"
        elif abs(confidence_change) > 0.1:  # Changement de confiance significatif
            evolution_type = "confiance_modifiée"

        # Générer des notes automatiques
        notes = []
        if diagnosis_change:
            notes.append(f"Changement de diagnostic: {diagnosis_change}")
        if abs(confidence_change) > 0.05:
            direction = "augmentation" if confidence_change > 0 else "diminution"
            notes.append(f"{direction.capitalize()} de confiance: {confidence_change*100:+.1f}%")
        if size_change and abs(size_change) > 0.1:
            direction = "augmentation" if size_change > 0 else "diminution"
            notes.append(f"{direction.capitalize()} de taille estimée: {size_change:+.1f}cm")

        notes_text = "; ".join(notes) if notes else "Évolution stable"

        # Enregistrer l'évolution
        cursor.execute('''
            INSERT INTO tumor_evolution
            (patient_id, analysis_id, exam_date, diagnosis_change, confidence_change,
             size_change, evolution_type, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (patient_id, current_analysis_id, exam_date, diagnosis_change,
              confidence_change, size_change, evolution_type, notes_text))

    except Exception as e:
        print(f"Erreur lors de l'analyse d'évolution: {e}")

def calculate_patient_metrics(analyses, evolution_details):
    """Calculer des métriques avancées pour un patient"""
    if not analyses:
        return {}

    try:
        # Métriques de base
        total_analyses = len(analyses)

        # Période de suivi
        if total_analyses > 1:
            first_date_str = analyses[0]['exam_date']
            last_date_str = analyses[-1]['exam_date']
            
            # S'assurer que les dates sont des chaînes de caractères avant de les parser
            if isinstance(first_date_str, str) and isinstance(last_date_str, str):
                first_date = datetime.strptime(first_date_str, '%Y-%m-%d')
                last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                follow_up_days = (last_date - first_date).days
                follow_up_months = round(follow_up_days / 30.44, 1)
            else:
                # Gérer le cas où les dates ne sont pas des chaînes (improbable mais sécuritaire)
                follow_up_days = 0
                follow_up_months = 0
        else:
            follow_up_days = 0
            follow_up_months = 0

        # Analyse des diagnostics
        diagnoses = [a['predicted_label'] for a in analyses]
        diagnosis_counts = Counter(diagnoses)
        most_common_diagnosis = diagnosis_counts.most_common(1)[0] if diagnosis_counts else ('Inconnu', 0)

        # Évolution de la confiance
        confidences = [a['confidence'] for a in analyses]
        avg_confidence = round(sum(confidences) / len(confidences), 1)
        confidence_trend = "stable"
        if len(confidences) > 1:
            confidence_change = confidences[-1] - confidences[0]
            if confidence_change > 5:
                confidence_trend = "amélioration"
            elif confidence_change < -5:
                confidence_trend = "dégradation"

        # Analyse des tailles de tumeur
        sizes = [a['tumor_size_estimate'] for a in analyses if a['tumor_size_estimate']]
        size_metrics = {}
        if sizes:
            size_metrics = {
                'avg_size': round(sum(sizes) / len(sizes), 2),
                'min_size': round(min(sizes), 2),
                'max_size': round(max(sizes), 2),
                'size_trend': "stable"
            }
            if len(sizes) > 1:
                size_change = sizes[-1] - sizes[0]
                if size_change > 0.3:
                    size_metrics['size_trend'] = "croissance"
                elif size_change < -0.3:
                    size_metrics['size_trend'] = "réduction"

        # Analyse des évolutions
        evolution_types = [e['evolution_type'] for e in evolution_details]
        evolution_counts = Counter(evolution_types)

        # Alertes et recommandations
        alerts = []
        recommendations = []

        # Vérifier les changements récents
        if len(analyses) >= 2:
            recent_analysis = analyses[-1]
            previous_analysis = analyses[-2]

            # Alerte changement de diagnostic
            if recent_analysis['predicted_label'] != previous_analysis['predicted_label']:
                alerts.append({
                    'type': 'diagnostic_change',
                    'severity': 'high',
                    'message': f"Changement de diagnostic: {previous_analysis['predicted_label']} → {recent_analysis['predicted_label']}"
                })
                recommendations.append("Consultation urgente recommandée suite au changement de diagnostic")

            # Alerte baisse de confiance
            confidence_drop = recent_analysis['confidence'] - previous_analysis['confidence']
            if confidence_drop < -15:
                alerts.append({
                    'type': 'confidence_drop',
                    'severity': 'medium',
                    'message': f"Baisse significative de confiance: {confidence_drop:+.1f}%"
                })
                recommendations.append("Analyse complémentaire recommandée")

            # Alerte croissance de tumeur
            if (recent_analysis['tumor_size_estimate'] and previous_analysis['tumor_size_estimate']):
                size_growth = recent_analysis['tumor_size_estimate'] - previous_analysis['tumor_size_estimate']
                if size_growth > 0.5:
                    alerts.append({
                        'type': 'tumor_growth',
                        'severity': 'high',
                        'message': f"Croissance significative de la tumeur: +{size_growth:.1f}cm"
                    })
                    recommendations.append("Évaluation oncologique urgente recommandée")

        # Recommandations basées sur le suivi
        if follow_up_months > 6 and total_analyses < 3:
            recommendations.append("Augmenter la fréquence des examens de suivi")

        if most_common_diagnosis[0] != 'Normal' and follow_up_months > 3:
            recommendations.append("Suivi oncologique spécialisé recommandé")

        return {
            'total_analyses': total_analyses,
            'follow_up_days': follow_up_days,
            'follow_up_months': follow_up_months,
            'most_common_diagnosis': most_common_diagnosis[0],
            'diagnosis_stability': len(set(diagnoses)) == 1,
            'avg_confidence': avg_confidence,
            'confidence_trend': confidence_trend,
            'size_metrics': size_metrics,
            'evolution_counts': dict(evolution_counts),
            'alerts': alerts,
            'recommendations': recommendations,
            'risk_level': calculate_risk_level(analyses, evolution_details, alerts)
        }

    except Exception as e:
        print(f"Erreur lors du calcul des métriques: {e}")
        return {}

def calculate_risk_level(analyses, evolution_details, alerts):
    """Calculer le niveau de risque d'un patient"""
    try:
        risk_score = 0

        # Facteurs de risque basés sur les alertes
        for alert in alerts:
            if alert['severity'] == 'high':
                risk_score += 3
            elif alert['severity'] == 'medium':
                risk_score += 2
            else:
                risk_score += 1

        # Facteurs de risque basés sur les diagnostics
        recent_diagnoses = [a['predicted_label'] for a in analyses[-3:]]  # 3 dernières analyses
        tumor_count = sum(1 for d in recent_diagnoses if d != 'Normal')
        risk_score += tumor_count

        # Facteurs de risque basés sur l'évolution
        negative_evolutions = ['dégradation', 'croissance', 'changement_type']
        negative_count = sum(1 for e in evolution_details[-3:] if e['evolution_type'] in negative_evolutions)
        risk_score += negative_count * 2

        # Déterminer le niveau de risque
        if risk_score >= 8:
            return 'critique'
        elif risk_score >= 5:
            return 'élevé'
        elif risk_score >= 2:
            return 'modéré'
        else:
            return 'faible'

    except Exception as e:
        print(f"Erreur lors du calcul du niveau de risque: {e}")
        return 'indéterminé'

def create_medical_alerts(cursor, patient_id, analysis_id, current_results, doctor_id, previous_analysis_id):
    """Créer des alertes médicales automatiques basées sur l'analyse"""
    try:
        # Récupérer l'analyse précédente
        cursor.execute('''
            SELECT predicted_label, confidence, tumor_size_estimate
            FROM analyses WHERE id = ?
        ''', (previous_analysis_id,))
        prev_data = cursor.fetchone()

        if not prev_data:
            return

        prev_label, prev_confidence, prev_size = prev_data
        alerts_to_create = []

        # Alerte changement de diagnostic critique
        if prev_label != current_results['predicted_label']:
            if prev_label == 'Normal' and current_results['predicted_label'] != 'Normal':
                alerts_to_create.append({
                    'type': 'new_tumor_detected',
                    'severity': 'high',
                    'title': 'Nouvelle tumeur détectée',
                    'message': f'Changement de diagnostic: {prev_label} → {current_results["predicted_label"]}. Consultation urgente recommandée.'
                })
            elif prev_label != 'Normal' and current_results['predicted_label'] == 'Normal':
                alerts_to_create.append({
                    'type': 'tumor_resolved',
                    'severity': 'medium',
                    'title': 'Amélioration significative',
                    'message': f'Changement positif: {prev_label} → {current_results["predicted_label"]}. Suivi recommandé.'
                })
            else:
                alerts_to_create.append({
                    'type': 'diagnosis_change',
                    'severity': 'high',
                    'title': 'Changement de type de tumeur',
                    'message': f'Changement de diagnostic: {prev_label} → {current_results["predicted_label"]}. Réévaluation nécessaire.'
                })

        # Alerte baisse significative de confiance
        confidence_change = current_results['confidence'] - prev_confidence
        if confidence_change < -0.2:  # Baisse de plus de 20%
            alerts_to_create.append({
                'type': 'confidence_drop',
                'severity': 'medium',
                'title': 'Baisse de confiance diagnostique',
                'message': f'Baisse significative de confiance: {confidence_change*100:+.1f}%. Analyse complémentaire recommandée.'
            })

        # Alerte croissance rapide de tumeur
        if current_results['predicted_class'] != 0 and prev_size:  # Si tumeur détectée
            # Estimer la taille actuelle (même logique que dans save_analysis_to_db)
            base_size = {1: 2.5, 2: 1.8, 3: 1.2}.get(current_results['predicted_class'], 2.0)
            current_size = base_size * current_results['confidence'] * (0.8 + 0.4 * np.random.random())

            size_change = current_size - prev_size
            if size_change > 0.5:  # Croissance de plus de 0.5cm
                alerts_to_create.append({
                    'type': 'rapid_growth',
                    'severity': 'high',
                    'title': 'Croissance rapide de tumeur',
                    'message': f'Augmentation significative de taille: +{size_change:.1f}cm. Évaluation oncologique urgente.'
                })

        # Alerte tumeur de haut grade
        if current_results['predicted_label'] == 'Gliome' and current_results['confidence'] > 0.9:
            alerts_to_create.append({
                'type': 'high_grade_tumor',
                'severity': 'high',
                'title': 'Tumeur de haut grade suspectée',
                'message': f'Gliome détecté avec haute confiance ({current_results["confidence"]*100:.1f}%). Prise en charge oncologique urgente.'
            })

        # Créer les alertes en base de données
        for alert in alerts_to_create:
            cursor.execute('''
                INSERT INTO medical_alerts
                (patient_id, doctor_id, analysis_id, alert_type, severity, title, message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (patient_id, doctor_id, analysis_id, alert['type'], alert['severity'],
                  alert['title'], alert['message']))

            # Créer aussi une notification
            cursor.execute('''
                INSERT INTO notifications
                (doctor_id, type, title, message, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (doctor_id, 'medical_alert', alert['title'], alert['message'],
                  json.dumps({'patient_id': patient_id, 'analysis_id': analysis_id})))

    except Exception as e:
        print(f"Erreur lors de la création des alertes: {e}")

def get_patient_alerts(cursor, patient_id, doctor_id, limit=None):
    """Récupérer les alertes d'un patient"""
    try:
        query = '''
            SELECT id, alert_type, severity, title, message, is_read, is_resolved, created_at
            FROM medical_alerts
            WHERE patient_id = ? AND doctor_id = ?
            ORDER BY created_at DESC
        '''
        params = [patient_id, doctor_id]

        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        cursor.execute(query, params)

        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'id': row[0],
                'alert_type': row[1],
                'severity': row[2],
                'title': row[3],
                'message': row[4],
                'is_read': bool(row[5]),
                'is_resolved': bool(row[6]),
                'created_at': row[7]
            })

        return alerts

    except Exception as e:
        print(f"Erreur lors de la récupération des alertes: {e}")
        return []

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
    """Obtenir les informations du médecin connecté"""
    if 'doctor_id' not in session:
        return None

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, email, first_name, last_name, specialty, hospital, license_number
            FROM doctors WHERE id = ? AND is_active = 1
        ''', (session['doctor_id'],))

        doctor_data = cursor.fetchone()
        conn.close()

        if doctor_data:
            return {
                'id': doctor_data[0],
                'email': doctor_data[1],
                'first_name': doctor_data[2],
                'last_name': doctor_data[3],
                'specialty': doctor_data[4],
                'hospital': doctor_data[5],
                'license_number': doctor_data[6],
                'full_name': f"{doctor_data[2]} {doctor_data[3]}"
            }
    except Exception as e:
        print(f"Erreur lors de la récupération du médecin: {e}")

    return None

def create_doctor_session(doctor_id, ip_address, user_agent):
    """Créer une session pour un médecin"""
    try:
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=7)  # Session valide 7 jours

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Désactiver les anciennes sessions
        cursor.execute('''
            UPDATE doctor_sessions
            SET is_active = 0
            WHERE doctor_id = ?
        ''', (doctor_id,))

        # Créer la nouvelle session
        cursor.execute('''
            INSERT INTO doctor_sessions
            (doctor_id, session_token, expires_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        ''', (doctor_id, session_token, expires_at, ip_address, user_agent))

        conn.commit()
        conn.close()

        return session_token
    except Exception as e:
        print(f"Erreur lors de la création de session: {e}")
        return None

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
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, password_hash, first_name, last_name, is_active
                FROM doctors WHERE email = ?
            ''', (email,))

            doctor = cursor.fetchone()

            if doctor and check_password_hash(doctor[1], password):
                if not doctor[4]:  # is_active
                    flash('Votre compte a été désactivé. Contactez l\'administrateur.', 'error')
                    conn.close()
                    return render_template('auth/login.html')

                # Mettre à jour les statistiques de connexion
                cursor.execute('''
                    UPDATE doctors
                    SET last_login = CURRENT_TIMESTAMP, login_count = login_count + 1
                    WHERE id = ?
                ''', (doctor[0],))

                conn.commit()
                conn.close()

                # Créer la session
                session['doctor_id'] = doctor[0]
                session['doctor_name'] = f"{doctor[2]} {doctor[3]}"
                session['logged_in'] = True

                # Créer une session en base
                create_doctor_session(
                    doctor[0],
                    request.remote_addr,
                    request.headers.get('User-Agent', '')
                )

                flash(f'Bienvenue Dr. {doctor[2]} {doctor[3]}!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Email ou mot de passe incorrect', 'error')
                conn.close()

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
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()

            # Vérifier si l'email existe déjà
            cursor.execute('SELECT id FROM doctors WHERE email = ?', (email,))
            if cursor.fetchone():
                flash('Un compte avec cet email existe déjà', 'error')
                conn.close()
                return render_template('auth/register.html')

            # Créer le compte
            password_hash = generate_password_hash(password)
            cursor.execute('''
                INSERT INTO doctors
                (email, password_hash, first_name, last_name, specialty, hospital, license_number, phone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (email, password_hash, first_name, last_name, specialty, hospital, license_number, phone))

            conn.commit()
            conn.close()

            flash('Compte créé avec succès! Vous pouvez maintenant vous connecter.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            print(f"Erreur lors de l'inscription: {e}")
            flash('Erreur lors de la création du compte', 'error')

    return render_template('auth/register.html')

@app.route('/logout')
def logout():
    """Déconnexion"""
    if 'doctor_id' in session:
        # Désactiver la session en base
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE doctor_sessions
                SET is_active = 0
                WHERE doctor_id = ?
            ''', (session['doctor_id'],))
            conn.commit()
            conn.close()
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

    return render_template('dashboard.html', doctor=doctor)

@app.route('/')
def index():
    """Page d'accueil"""
    doctor = get_current_doctor()
    return render_template('index.html', doctor=doctor)

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

            # Convertir l'image en base64 pour l'affichage
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
                img_url = f"data:image/jpeg;base64,{img_data}"

            # Nettoyer le fichier temporaire
            os.remove(filepath)

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
                return result['candidates'][0]['content']['parts'][0]['text']

        print(f"Erreur API Gemini: {response.status_code} - {response.text}")
        return None

    except Exception as e:
        print(f"Erreur lors de l'appel à Gemini: {e}")
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
    """Générer un rapport médical"""
    try:
        data = request.get_json()

        # Valider les données requises
        required_fields = ['patientName', 'analysisData']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Champ requis manquant: {field}'}), 400

        # Générer le rapport
        report_content = create_medical_report(data)

        # Simuler la sauvegarde du rapport
        report_id = f"RPT_{int(time.time())}"

        return jsonify({
            'success': True,
            'report_id': report_id,
            'message': 'Rapport généré avec succès',
            'download_url': f'/download-report/{report_id}'
        })

    except Exception as e:
        print(f"Erreur lors de la génération du rapport: {e}")
        return jsonify({'error': 'Erreur lors de la génération du rapport'}), 500

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

@app.route('/chat', methods=['POST'])
def chat_with_bot():
    """Chatbot médical avec Gemini"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Statistiques personnelles du médecin
        cursor.execute('SELECT COUNT(*) FROM analyses WHERE doctor_id = ?', (doctor_id,))
        result = cursor.fetchone()
        total_analyses = result[0] if result else 0

        cursor.execute('SELECT COUNT(DISTINCT DATE(timestamp)) FROM analyses WHERE doctor_id = ?', (doctor_id,))
        result = cursor.fetchone()
        active_days = result[0] if result else 0

        cursor.execute('SELECT AVG(confidence) FROM analyses WHERE doctor_id = ?', (doctor_id,))
        result = cursor.fetchone()
        avg_confidence = result[0] if result and result[0] else 0

        cursor.execute('SELECT AVG(processing_time) FROM analyses WHERE doctor_id = ?', (doctor_id,))
        result = cursor.fetchone()
        avg_processing_time = result[0] if result and result[0] else 0

        # Répartition par type de tumeur pour ce médecin
        cursor.execute('''
            SELECT predicted_label, COUNT(*)
            FROM analyses
            WHERE doctor_id = ?
            GROUP BY predicted_label
        ''', (doctor_id,))
        tumor_distribution = dict(cursor.fetchall())

        # Analyses par jour (30 derniers jours) pour ce médecin
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM analyses
            WHERE doctor_id = ? AND timestamp >= date('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (doctor_id,))
        daily_analyses = cursor.fetchall()

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Statistiques générales de la plateforme
        cursor.execute('SELECT COUNT(*) FROM analyses')
        result = cursor.fetchone()
        total_analyses = result[0] if result else 0

        cursor.execute('SELECT COUNT(DISTINCT doctor_id) FROM analyses')
        result = cursor.fetchone()
        total_doctors = result[0] if result else 0

        cursor.execute('SELECT COUNT(DISTINCT patient_id) FROM analyses')
        result = cursor.fetchone()
        total_patients = result[0] if result else 0

        cursor.execute('SELECT COUNT(DISTINCT DATE(timestamp)) FROM analyses')
        result = cursor.fetchone()
        active_days = result[0] if result else 0

        cursor.execute('SELECT AVG(confidence) FROM analyses')
        result = cursor.fetchone()
        avg_confidence = result[0] if result and result[0] else 0

        cursor.execute('SELECT AVG(processing_time) FROM analyses')
        result = cursor.fetchone()
        avg_processing_time = result[0] if result and result[0] else 0

        # Répartition par type de tumeur (toute la plateforme)
        cursor.execute('''
            SELECT predicted_label, COUNT(*)
            FROM analyses
            GROUP BY predicted_label
        ''')
        tumor_distribution = dict(cursor.fetchall())

        # Analyses par jour (30 derniers jours) - toute la plateforme
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM analyses
            WHERE timestamp >= date('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''')
        daily_analyses = cursor.fetchall()

        # Top 5 des médecins les plus actifs
        cursor.execute('''
            SELECT d.first_name, d.last_name, COUNT(a.id) as analyses_count
            FROM doctors d
            LEFT JOIN analyses a ON d.id = a.doctor_id
            GROUP BY d.id, d.first_name, d.last_name
            ORDER BY analyses_count DESC
            LIMIT 5
        ''')
        top_doctors = cursor.fetchall()

        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'total_analyses': total_analyses,
                'total_doctors': total_doctors,
                'total_patients': total_patients,
                'active_days': active_days,
                'avg_confidence': round(avg_confidence * 100, 1) if avg_confidence else 0,
                'avg_processing_time': round(avg_processing_time, 2) if avg_processing_time else 0,
                'tumor_distribution': tumor_distribution,
                'daily_analyses': daily_analyses,
                'top_doctors': [{'name': f"Dr. {row[0]} {row[1]}", 'analyses': row[2]} for row in top_doctors]
            }
        })

    except Exception as e:
        print(f"Erreur platform analytics overview: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/my-patients')
@login_required
def get_my_patients():
    """API pour obtenir la liste des patients du médecin connecté"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT patient_id, patient_name, date_of_birth, gender, phone, email,
                   address, emergency_contact_name, emergency_contact_phone,
                   medical_history, allergies, current_medications, insurance_number,
                   notes, first_analysis_date, last_analysis_date, total_analyses,
                   created_at, updated_at
            FROM patients
            WHERE doctor_id = ?
            ORDER BY COALESCE(updated_at, created_at) DESC
        ''', (doctor_id,))

        patients = []
        for row in cursor.fetchall():
            patients.append({
                'patient_id': row[0],
                'patient_name': row[1],
                'date_of_birth': datetime.strptime(row[2], '%Y-%m-%d') if row[2] else None,
                'gender': row[3],
                'phone': row[4],
                'email': row[5],
                'address': row[6],
                'emergency_contact_name': row[7],
                'emergency_contact_phone': row[8],
                'medical_history': row[9],
                'allergies': row[10],
                'current_medications': row[11],
                'insurance_number': row[12],
                'notes': row[13],
                'first_analysis_date': row[14],
                'last_analysis_date': row[15],
                'total_analyses': row[16],
                'created_at': row[17],
                'updated_at': row[18]
            })

        conn.close()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Compter les diagnostics par type
        cursor.execute('''
            SELECT predicted_label, COUNT(*)
            FROM analyses
            WHERE doctor_id = ?
            GROUP BY predicted_label
        ''', (doctor_id,))

        diagnostic_counts = dict(cursor.fetchall())

        # Statistiques générales
        cursor.execute('SELECT COUNT(*) FROM analyses WHERE doctor_id = ?', (doctor_id,))
        result = cursor.fetchone()
        total_analyses = result[0] if result else 0

        cursor.execute('SELECT AVG(confidence) FROM analyses WHERE doctor_id = ?', (doctor_id,))
        result = cursor.fetchone()
        avg_confidence = result[0] if result and result[0] else 0

        cursor.execute('SELECT AVG(processing_time) FROM analyses WHERE doctor_id = ?', (doctor_id,))
        result = cursor.fetchone()
        avg_processing_time = result[0] if result and result[0] else 0

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

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

        cursor.execute(query, params)
        result = cursor.fetchone()
        count = result[0] if result else 0

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        if period == 'day':
            # Analyses par heure pour le jour le plus récent avec des données
            cursor.execute('''
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                FROM analyses
                WHERE DATE(timestamp) = (
                    SELECT DATE(timestamp) FROM analyses
                    ORDER BY timestamp DESC LIMIT 1
                )
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''')
            data = cursor.fetchall()

            # Si pas de données pour le jour le plus récent, prendre les 24 dernières heures
            if not data:
                cursor.execute('''
                    SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                    FROM analyses
                    WHERE timestamp >= datetime('now', '-24 hours')
                    GROUP BY strftime('%H', timestamp)
                    ORDER BY hour
                ''')
                data = cursor.fetchall()

            # Créer un tableau complet de 24 heures
            hour_counts = {str(hour).zfill(2): 0 for hour in range(24)}
            for hour, count in data:
                hour_counts[hour] = count

            labels = [f"{hour}h" for hour in sorted(hour_counts.keys())]
            values = [hour_counts[hour] for hour in sorted(hour_counts.keys())]

        elif period == 'month':
            # Analyses par jour pour le mois le plus récent avec des données
            cursor.execute('''
                SELECT strftime('%d', timestamp) as day, COUNT(*) as count
                FROM analyses
                WHERE strftime('%Y-%m', timestamp) = (
                    SELECT strftime('%Y-%m', timestamp) FROM analyses
                    ORDER BY timestamp DESC LIMIT 1
                )
                GROUP BY strftime('%d', timestamp)
                ORDER BY CAST(day as INTEGER)
            ''')
            data = cursor.fetchall()

            # Si pas de données, prendre les 30 derniers jours
            if not data:
                cursor.execute('''
                    SELECT strftime('%d', timestamp) as day, COUNT(*) as count
                    FROM analyses
                    WHERE timestamp >= date('now', '-30 days')
                    GROUP BY strftime('%d', timestamp)
                    ORDER BY CAST(day as INTEGER)
                ''')
                data = cursor.fetchall()

            labels = [f"{day}" for day, _ in data]
            values = [count for _, count in data]

        elif period == 'year':
            # Analyses par mois pour l'année la plus récente avec des données
            cursor.execute('''
                SELECT strftime('%m', timestamp) as month, COUNT(*) as count
                FROM analyses
                WHERE strftime('%Y', timestamp) = (
                    SELECT strftime('%Y', timestamp) FROM analyses
                    ORDER BY timestamp DESC LIMIT 1
                )
                GROUP BY strftime('%m', timestamp)
                ORDER BY CAST(month as INTEGER)
            ''')
            data = cursor.fetchall()

            # Si pas de données, prendre les 12 derniers mois
            if not data:
                cursor.execute('''
                    SELECT strftime('%m', timestamp) as month, COUNT(*) as count
                    FROM analyses
                    WHERE timestamp >= date('now', '-12 months')
                    GROUP BY strftime('%m', timestamp)
                    ORDER BY CAST(month as INTEGER)
                ''')
                data = cursor.fetchall()

            month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                          'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
            labels = [month_names[int(month)-1] for month, _ in data]
            values = [count for _, count in data]

        else:
            return jsonify({'success': False, 'error': 'Période invalide'}), 400

        conn.close()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT timestamp, filename, predicted_label, confidence, processing_time, patient_name, patient_id
            FROM analyses
            WHERE doctor_id = ?
            ORDER BY timestamp DESC
        ''', (doctor_id,))

        analyses = []
        for row in cursor.fetchall():
            analyses.append({
                'timestamp': row[0],
                'filename': row[1],
                'predicted_label': row[2],
                'confidence': round(row[3] * 100, 1),
                'processing_time': round(row[4], 2),
                'patient_name': row[5] or 'Patient anonyme',
                'patient_id': row[6] or 'N/A'
            })

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        if format == 'csv':
            cursor.execute('''
                SELECT timestamp, filename, predicted_label, confidence,
                       processing_time, description
                FROM analyses
                ORDER BY timestamp DESC
            ''')

            data = cursor.fetchall()

            # Créer le contenu CSV
            csv_content = "Timestamp,Filename,Diagnostic,Confidence,Processing_Time,Description\n"
            for row in data:
                csv_content += f'"{row[0]}","{row[1]}","{row[2]}",{row[3]:.3f},{row[4]:.2f},"{row[5] or ""}"\n'

            conn.close()

            return Response(
                csv_content,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=neuroscan_analytics_{datetime.now().strftime("%Y%m%d")}.csv'}
            )

        elif format == 'json':
            cursor.execute('''
                SELECT timestamp, filename, predicted_label, confidence,
                       processing_time, probabilities, description, recommendations
                FROM analyses
                ORDER BY timestamp DESC
            ''')

            analyses = []
            for row in cursor.fetchall():
                analyses.append({
                    'timestamp': row[0],
                    'filename': row[1],
                    'predicted_label': row[2],
                    'confidence': row[3],
                    'processing_time': row[4],
                    'probabilities': json.loads(row[5]) if row[5] else {},
                    'description': row[6],
                    'recommendations': json.loads(row[7]) if row[7] else []
                })

            conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Statistiques par heure de la journée
        cursor.execute('''
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM analyses
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        ''')
        hourly_stats = dict(cursor.fetchall())

        # Évolution de la confiance dans le temps
        cursor.execute('''
            SELECT DATE(timestamp) as date, AVG(confidence) as avg_confidence
            FROM analyses
            WHERE timestamp >= date('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''')
        confidence_evolution = cursor.fetchall()

        # Top 5 des jours les plus actifs
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM analyses
            GROUP BY DATE(timestamp)
            ORDER BY count DESC
            LIMIT 5
        ''')
        top_active_days = cursor.fetchall()

        # Statistiques de performance
        cursor.execute('''
            SELECT
                MIN(processing_time) as min_time,
                MAX(processing_time) as max_time,
                AVG(processing_time) as avg_time,
                COUNT(CASE WHEN processing_time < 5 THEN 1 END) as fast_analyses,
                COUNT(CASE WHEN processing_time >= 5 THEN 1 END) as slow_analyses
            FROM analyses
        ''')
        performance_stats = cursor.fetchone()

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Obtenir les plages de dates disponibles
        cursor.execute('''
            SELECT MIN(DATE(timestamp)) as min_date, MAX(DATE(timestamp)) as max_date
            FROM analyses
        ''')
        date_range = cursor.fetchone()

        # Obtenir les types de diagnostics
        cursor.execute('SELECT DISTINCT predicted_label FROM analyses ORDER BY predicted_label')
        diagnostic_types = [row[0] for row in cursor.fetchall()]

        # Obtenir les plages de confiance
        cursor.execute('SELECT MIN(confidence), MAX(confidence) FROM analyses')
        confidence_range = cursor.fetchone()

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

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
        cursor.execute(f'''
            SELECT timestamp, filename, predicted_label, confidence, processing_time,
                   description, patient_name, patient_id, exam_date
            FROM analyses
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT 100
        ''', params)

        analyses = []
        for row in cursor.fetchall():
            analyses.append({
                'timestamp': row[0],
                'filename': row[1],
                'predicted_label': row[2],
                'confidence': round(row[3] * 100, 1),
                'processing_time': round(row[4], 2),
                'description': row[5] or '',
                'patient_name': row[6] or 'Patient anonyme',
                'patient_id': row[7] or 'N/A',
                'exam_date': row[8] or ''
            })

        # Statistiques des résultats filtrés
        cursor.execute(f'''
            SELECT
                COUNT(*) as total,
                AVG(confidence) as avg_confidence,
                AVG(processing_time) as avg_time,
                predicted_label,
                COUNT(*) as type_count
            FROM analyses
            {where_clause}
            GROUP BY predicted_label
        ''', params)

        stats_by_type = {}
        total_filtered = 0
        total_confidence = 0
        total_time = 0

        for row in cursor.fetchall():
            stats_by_type[row[3]] = row[4]
            total_filtered += row[4]
            total_confidence += row[1] * row[4] if row[1] else 0
            total_time += row[2] * row[4] if row[2] else 0

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Comparaison cette semaine vs semaine dernière
        cursor.execute('''
            SELECT
                CASE
                    WHEN DATE(timestamp) >= DATE('now', '-7 days') THEN 'Cette semaine'
                    WHEN DATE(timestamp) >= DATE('now', '-14 days') THEN 'Semaine dernière'
                END as period,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                predicted_label
            FROM analyses
            WHERE DATE(timestamp) >= DATE('now', '-14 days')
            GROUP BY period, predicted_label
            ORDER BY period, predicted_label
        ''')

        weekly_comparison = {}
        for row in cursor.fetchall():
            period = row[0]
            if period not in weekly_comparison:
                weekly_comparison[period] = {}
            weekly_comparison[period][row[3]] = {
                'count': row[1],
                'avg_confidence': round(row[2] * 100, 1) if row[2] else 0
            }

        # Comparaison ce mois vs mois dernier
        cursor.execute('''
            SELECT
                CASE
                    WHEN strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now') THEN 'Ce mois'
                    WHEN strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now', '-1 month') THEN 'Mois dernier'
                END as period,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM analyses
            WHERE strftime('%Y-%m', timestamp) >= strftime('%Y-%m', 'now', '-1 month')
            GROUP BY period
        ''')

        monthly_comparison = {}
        for row in cursor.fetchall():
            if row[0]:  # Vérifier que period n'est pas None
                monthly_comparison[row[0]] = {
                    'count': row[1],
                    'avg_confidence': round(row[2] * 100, 1) if row[2] else 0
                }

        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'weekly': weekly_comparison,
                'monthly': monthly_comparison
            }
        })

    except Exception as e:
        print(f"Erreur comparison data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/alerts')
def get_alerts():
    """API pour obtenir les alertes et notifications"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        alerts = []

        # Alerte: Baisse de confiance moyenne
        cursor.execute('''
            SELECT AVG(confidence) as avg_conf_today
            FROM analyses
            WHERE DATE(timestamp) = DATE('now')
        ''')
        today_confidence = cursor.fetchone()[0]

        cursor.execute('''
            SELECT AVG(confidence) as avg_conf_week
            FROM analyses
            WHERE DATE(timestamp) >= DATE('now', '-7 days')
            AND DATE(timestamp) < DATE('now')
        ''')
        week_confidence = cursor.fetchone()[0]

        if today_confidence and week_confidence and today_confidence < week_confidence * 0.9:
            alerts.append({
                'type': 'warning',
                'title': 'Baisse de confiance détectée',
                'message': f'La confiance moyenne aujourd\'hui ({today_confidence*100:.1f}%) est inférieure à la moyenne de la semaine ({week_confidence*100:.1f}%)',
                'timestamp': datetime.now().isoformat()
            })

        # Alerte: Pic d'activité
        cursor.execute('''
            SELECT COUNT(*) as today_count
            FROM analyses
            WHERE DATE(timestamp) = DATE('now')
        ''')
        today_count = cursor.fetchone()[0]

        cursor.execute('''
            SELECT AVG(daily_count) as avg_daily
            FROM (
                SELECT COUNT(*) as daily_count
                FROM analyses
                WHERE DATE(timestamp) >= DATE('now', '-7 days')
                AND DATE(timestamp) < DATE('now')
                GROUP BY DATE(timestamp)
            )
        ''')
        avg_daily = cursor.fetchone()[0]

        if today_count and avg_daily and today_count > avg_daily * 1.5:
            alerts.append({
                'type': 'info',
                'title': 'Pic d\'activité détecté',
                'message': f'Nombre d\'analyses aujourd\'hui ({today_count}) supérieur à la moyenne ({avg_daily:.1f})',
                'timestamp': datetime.now().isoformat()
            })

        # Alerte: Analyses avec faible confiance
        cursor.execute('''
            SELECT COUNT(*) as low_conf_count
            FROM analyses
            WHERE DATE(timestamp) = DATE('now')
            AND confidence < 0.7
        ''')
        low_confidence_count = cursor.fetchone()[0]

        if low_confidence_count > 0:
            alerts.append({
                'type': 'warning',
                'title': 'Analyses à faible confiance',
                'message': f'{low_confidence_count} analyse(s) avec confiance < 70% aujourd\'hui',
                'timestamp': datetime.now().isoformat()
            })

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Tendances de confiance sur les 7 derniers jours
        cursor.execute('''
            SELECT
                DATE(timestamp) as date,
                AVG(confidence) as avg_confidence,
                AVG(processing_time) as avg_time,
                COUNT(*) as daily_count
            FROM analyses
            WHERE DATE(timestamp) >= DATE('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''')

        daily_trends = cursor.fetchall()

        # Tendances par heure pour aujourd'hui
        cursor.execute('''
            SELECT
                strftime('%H', timestamp) as hour,
                AVG(confidence) as avg_confidence,
                AVG(processing_time) as avg_time,
                COUNT(*) as hourly_count
            FROM analyses
            WHERE DATE(timestamp) = DATE('now')
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        ''')

        hourly_trends = cursor.fetchall()

        # Performance par type de diagnostic
        cursor.execute('''
            SELECT
                predicted_label,
                AVG(confidence) as avg_confidence,
                AVG(processing_time) as avg_time,
                COUNT(*) as count
            FROM analyses
            WHERE DATE(timestamp) >= DATE('now', '-7 days')
            GROUP BY predicted_label
            ORDER BY predicted_label
        ''')

        performance_by_type = cursor.fetchall()

        conn.close()

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

# ===== APIs pour le suivi de l'évolution des tumeurs =====

@app.route('/api/patients')
def get_patients_list():
    """API pour obtenir la liste des patients avec suivi"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT a.patient_id, a.patient_name,
                   MIN(a.exam_date) as first_analysis_date,
                   MAX(a.exam_date) as last_analysis_date,
                   COUNT(*) as total_analyses
            FROM analyses a
            WHERE a.patient_id IS NOT NULL
            GROUP BY a.patient_id, a.patient_name
            ORDER BY MAX(a.exam_date) DESC
        ''')

        patients = []
        for row in cursor.fetchall():
            patient_id = row[0]

            # Récupérer le dernier diagnostic pour ce patient
            cursor.execute('''
                SELECT predicted_label, confidence
                FROM analyses
                WHERE patient_id = ?
                ORDER BY exam_date DESC, timestamp DESC
                LIMIT 1
            ''', (patient_id,))

            last_analysis = cursor.fetchone()
            last_diagnosis = last_analysis[0] if last_analysis else None
            last_confidence = round(last_analysis[1] * 100, 1) if last_analysis and last_analysis[1] else 0

            patients.append({
                'patient_id': row[0],
                'patient_name': row[1],
                'first_analysis_date': row[2],
                'last_analysis_date': row[3],
                'total_analyses': row[4],
                'last_diagnosis': last_diagnosis,
                'last_confidence': last_confidence
            })

        conn.close()

        return jsonify({
            'success': True,
            'data': patients
        })

    except Exception as e:
        print(f"Erreur patients list: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patients/<patient_id>/evolution')
def get_patient_evolution(patient_id):
    """API pour obtenir l'évolution d'un patient spécifique"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Récupérer toutes les analyses du patient
        cursor.execute('''
            SELECT id, exam_date, predicted_label, confidence, tumor_size_estimate,
                   probabilities, description, recommendations
            FROM analyses
            WHERE patient_id = ?
            ORDER BY exam_date ASC, timestamp ASC
        ''', (patient_id,))

        analyses = []
        for row in cursor.fetchall():
            probabilities = json.loads(row[5]) if row[5] else {}
            recommendations = json.loads(row[7]) if row[7] else []

            analyses.append({
                'id': row[0],
                'exam_date': row[1],
                'predicted_label': row[2],
                'confidence': round(row[3] * 100, 1),
                'tumor_size_estimate': round(row[4], 2) if row[4] else None,
                'probabilities': probabilities,
                'description': row[6],
                'recommendations': recommendations
            })

        # Récupérer l'évolution détaillée
        cursor.execute('''
            SELECT exam_date, diagnosis_change, confidence_change, size_change,
                   evolution_type, notes
            FROM tumor_evolution
            WHERE patient_id = ?
            ORDER BY exam_date ASC
        ''', (patient_id,))

        evolution_details = []
        for row in cursor.fetchall():
            evolution_details.append({
                'exam_date': row[0],
                'diagnosis_change': row[1],
                'confidence_change': round(row[2] * 100, 1) if row[2] else 0,
                'size_change': round(row[3], 2) if row[3] else None,
                'evolution_type': row[4],
                'notes': row[5]
            })

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Vérifier que le patient appartient au médecin connecté
        cursor.execute('''
            SELECT patient_id, patient_name, date_of_birth, gender,
                   first_analysis_date, last_analysis_date, total_analyses,
                   phone, email, address, emergency_contact_name,
                   emergency_contact_phone, medical_history, allergies,
                   current_medications, insurance_number, notes
            FROM patients
            WHERE patient_id = ? AND doctor_id = ?
        ''', (patient_id, doctor['id']))

        patient_data = cursor.fetchone()
        if not patient_data:
            flash('Patient non trouvé ou accès non autorisé', 'error')
            return redirect(url_for('dashboard'))

        patient = {
            'patient_id': patient_data[0],
            'patient_name': patient_data[1],
            'date_of_birth': datetime.strptime(patient_data[2], '%Y-%m-%d') if patient_data[2] else None,
            'gender': patient_data[3],
            'first_analysis_date': datetime.strptime(patient_data[4], '%Y-%m-%d') if patient_data[4] else None,
            'last_analysis_date': datetime.strptime(patient_data[5], '%Y-%m-%d') if patient_data[5] else None,
            'total_analyses': patient_data[6],
            'phone': patient_data[7],
            'email': patient_data[8],
            'address': patient_data[9],
            'emergency_contact_name': patient_data[10],
            'emergency_contact_phone': patient_data[11],
            'medical_history': patient_data[12],
            'allergies': patient_data[13],
            'current_medications': patient_data[14],
            'insurance_number': patient_data[15],
            'notes': patient_data[16]
        }

        # Récupérer les analyses du patient
        cursor.execute('''
            SELECT id, timestamp, filename, predicted_class, predicted_label,
                   confidence, probabilities, description, recommendations,
                   processing_time, exam_date, tumor_size_estimate
            FROM analyses
            WHERE patient_id = ? AND doctor_id = ?
            ORDER BY exam_date DESC, timestamp DESC
        ''', (patient_id, doctor['id']))

        analyses = []
        for row in cursor.fetchall():
            probabilities = json.loads(row[6]) if row[6] else {}
            recommendations = json.loads(row[8]) if row[8] else []

            # Convertir le timestamp en datetime si c'est une chaîne
            timestamp_dt = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S') if isinstance(row[1], str) and row[1] else datetime.now()
            
            analyses.append({
                'id': row[0],
                'timestamp': timestamp_dt,  # Stocker comme datetime pour les calculs
                'filename': row[2],
                'predicted_class': row[3],
                'predicted_label': row[4],
                'confidence': row[5],
                'probabilities': probabilities,
                'description': row[7],
                'recommendations': recommendations,
                'processing_time': row[9],
                'exam_date': row[10],
                'tumor_size_estimate': row[11],
                'date_uploaded': timestamp_dt,  # Garder comme objet datetime pour le template
                'date_uploaded_str': str(row[1]) if row[1] else str(datetime.now()),  # Version chaîne pour JSON
                'image_name': row[2],
                'image_path': f'/uploads/{row[2]}',
                'medical_notes': row[7],
                'risk_level': 'Élevé' if row[3] != 0 else 'Faible'
            })

        # Calculer les statistiques
        normal_count = sum(1 for a in analyses if a['predicted_class'] == 0)
        abnormal_count = len(analyses) - normal_count

        # Récupérer les alertes médicales
        alerts = get_patient_alerts(cursor, patient_id, doctor['id'])

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

        conn.close()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Vérifier l'accès au patient
        cursor.execute('''
            SELECT COUNT(*) FROM patients
            WHERE patient_id = ? AND doctor_id = ?
        ''', (patient_id, doctor_id))

        if cursor.fetchone()[0] == 0:
            return jsonify({'success': False, 'error': 'Accès non autorisé'}), 403

        # Récupérer toutes les analyses avec détails complets
        cursor.execute('''
            SELECT id, timestamp, exam_date, predicted_label, confidence,
                   tumor_size_estimate, probabilities, description,
                   recommendations, processing_time, filename
            FROM analyses
            WHERE patient_id = ? AND doctor_id = ?
            ORDER BY exam_date ASC, timestamp ASC
        ''', (patient_id, doctor_id))

        analyses = []
        for row in cursor.fetchall():
            probabilities = json.loads(row[6]) if row[6] else {}
            recommendations = json.loads(row[8]) if row[8] else []

            analyses.append({
                'id': row[0],
                'timestamp': row[1],
                'exam_date': row[2],
                'predicted_label': row[3],
                'confidence': round(row[4] * 100, 1),
                'tumor_size_estimate': round(row[5], 2) if row[5] else None,
                'probabilities': probabilities,
                'description': row[7],
                'recommendations': recommendations,
                'processing_time': round(row[9], 2) if row[9] else 0,
                'filename': row[10]
            })

        # Récupérer l'évolution détaillée
        cursor.execute('''
            SELECT exam_date, diagnosis_change, confidence_change, size_change,
                   evolution_type, notes, created_at
            FROM tumor_evolution
            WHERE patient_id = ?
            ORDER BY exam_date ASC
        ''', (patient_id,))

        evolution_details = []
        for row in cursor.fetchall():
            evolution_details.append({
                'exam_date': row[0],
                'diagnosis_change': row[1],
                'confidence_change': round(row[2] * 100, 1) if row[2] else 0,
                'size_change': round(row[3], 2) if row[3] else None,
                'evolution_type': row[4],
                'notes': row[5],
                'created_at': row[6]
            })

        # Calculer des métriques avancées
        metrics = calculate_patient_metrics(analyses, evolution_details)

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Récupérer les deux dernières analyses pour comparaison
        cursor.execute('''
            SELECT id, exam_date, predicted_label, confidence, tumor_size_estimate,
                   probabilities
            FROM analyses
            WHERE patient_id = ?
            ORDER BY exam_date DESC, timestamp DESC
            LIMIT 2
        ''', (patient_id,))

        analyses = cursor.fetchall()

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

        conn.close()

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
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Statistiques générales d'évolution
        cursor.execute('''
            SELECT evolution_type, COUNT(*) as count
            FROM tumor_evolution
            GROUP BY evolution_type
        ''')
        evolution_stats = dict(cursor.fetchall())

        # Patients avec évolution récente (7 derniers jours)
        cursor.execute('''
            SELECT DISTINCT te.patient_id, a.patient_name, te.evolution_type, te.notes
            FROM tumor_evolution te
            JOIN analyses a ON te.patient_id = a.patient_id
            WHERE te.exam_date >= DATE('now', '-7 days')
            GROUP BY te.patient_id, te.evolution_type, te.notes
            ORDER BY te.exam_date DESC
        ''')
        recent_evolutions = []
        for row in cursor.fetchall():
            recent_evolutions.append({
                'patient_id': row[0],
                'patient_name': row[1],
                'evolution_type': row[2],
                'notes': row[3]
            })

        # Alertes d'évolution critique
        cursor.execute('''
            SELECT te.patient_id, a.patient_name, te.evolution_type, te.notes, te.exam_date
            FROM tumor_evolution te
            JOIN analyses a ON te.patient_id = a.patient_id
            WHERE te.evolution_type IN ('dégradation', 'croissance')
            AND te.exam_date >= DATE('now', '-30 days')
            GROUP BY te.patient_id, te.evolution_type, te.exam_date
            ORDER BY te.exam_date DESC
        ''')
        critical_alerts = []
        for row in cursor.fetchall():
            critical_alerts.append({
                'patient_id': row[0],
                'patient_name': row[1],
                'evolution_type': row[2],
                'notes': row[3],
                'exam_date': row[4]
            })

        conn.close()

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
    """Créer un rapport médical formaté"""
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
Diagnostic principal: {analysis_data['prediction']}
Niveau de confiance: {analysis_data['confidence'] * 100:.1f}%
Tumeur détectée: {'Oui' if analysis_data['is_tumor'] else 'Non'}

PROBABILITÉS DÉTAILLÉES
-----------------------
- Normal: {analysis_data['probabilities']['Normal'] * 100:.1f}%
- Gliome: {analysis_data['probabilities']['Gliome'] * 100:.1f}%
- Méningiome: {analysis_data['probabilities']['Méningiome'] * 100:.1f}%
- Tumeur pituitaire: {analysis_data['probabilities']['Tumeur pituitaire'] * 100:.1f}%

RECOMMANDATIONS CLINIQUES
-------------------------
"""

    for i, rec in enumerate(analysis_data['recommendations'], 1):
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

Les résultats doivent être corrélés avec l'examen clinique et d'autres
investigations complémentaires selon les protocoles en vigueur.

Rapport généré par NeuroScan AI - {current_date}
Système certifié CE - Dispositif médical de classe IIa
"""

    return report

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
- Diagnostic: {analysis_data['prediction']}
- Confiance: {analysis_data['confidence'] * 100:.1f}%
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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Récupérer les alertes non résolues
        cursor.execute('''
            SELECT ma.id, ma.patient_id, p.patient_name, ma.alert_type, ma.severity,
                   ma.title, ma.message, ma.is_read, ma.created_at
            FROM medical_alerts ma
            LEFT JOIN patients p ON ma.patient_id = p.patient_id AND ma.doctor_id = p.doctor_id
            WHERE ma.doctor_id = ? AND ma.is_resolved = 0
            ORDER BY ma.created_at DESC
            LIMIT 50
        ''', (doctor_id,))

        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'id': row[0],
                'patient_id': row[1],
                'patient_name': row[2],
                'alert_type': row[3],
                'severity': row[4],
                'title': row[5],
                'message': row[6],
                'is_read': bool(row[7]),
                'created_at': row[8]
            })

        conn.close()

        return jsonify({
            'success': True,
            'data': alerts
        })

    except Exception as e:
        print(f"Erreur get doctor alerts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alerts/<int:alert_id>/mark-read', methods=['POST'])
@login_required
def mark_alert_read(alert_id):
    """Marquer une alerte comme lue"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE medical_alerts
            SET is_read = 1
            WHERE id = ? AND doctor_id = ?
        ''', (alert_id, doctor_id))

        conn.commit()
        conn.close()

        return jsonify({'success': True})

    except Exception as e:
        print(f"Erreur mark alert read: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alerts/<int:alert_id>/resolve', methods=['POST'])
@login_required
def resolve_alert(alert_id):
    """Résoudre une alerte"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({'success': False, 'error': 'Médecin non connecté'}), 401

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE medical_alerts
            SET is_resolved = 1, resolved_at = CURRENT_TIMESTAMP, resolved_by = ?
            WHERE id = ? AND doctor_id = ?
        ''', (doctor_id, alert_id, doctor_id))

        conn.commit()
        conn.close()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, type, title, message, data, is_read, created_at
            FROM notifications
            WHERE doctor_id = ?
            ORDER BY created_at DESC
            LIMIT 20
        ''', (doctor_id,))

        notifications = []
        for row in cursor.fetchall():
            data = json.loads(row[4]) if row[4] else {}
            notifications.append({
                'id': row[0],
                'type': row[1],
                'title': row[2],
                'message': row[3],
                'data': data,
                'is_read': bool(row[5]),
                'created_at': row[6]
            })

        conn.close()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE notifications
            SET is_read = 1
            WHERE id = ? AND doctor_id = ?
        ''', (notification_id, doctor_id))

        conn.commit()
        conn.close()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Vérifier l'accès au patient
        cursor.execute('''
            SELECT patient_name FROM patients
            WHERE patient_id = ? AND doctor_id = ?
        ''', (patient_id, doctor_id))

        patient_data = cursor.fetchone()
        if not patient_data:
            return jsonify({'success': False, 'error': 'Accès non autorisé'}), 403

        # Récupérer les données pour le rapport
        cursor.execute('''
            SELECT id, timestamp, exam_date, predicted_label, confidence,
                   tumor_size_estimate, probabilities, description, recommendations
            FROM analyses
            WHERE patient_id = ? AND doctor_id = ?
            ORDER BY exam_date ASC, timestamp ASC
        ''', (patient_id, doctor_id))

        analyses = []
        for row in cursor.fetchall():
            probabilities = json.loads(row[6]) if row[6] else {}
            recommendations = json.loads(row[8]) if row[8] else []

            analyses.append({
                'id': row[0],
                'timestamp': row[1],
                'exam_date': row[2],
                'predicted_label': row[3],
                'confidence': row[4],
                'tumor_size_estimate': row[5],
                'probabilities': probabilities,
                'description': row[7],
                'recommendations': recommendations
            })

        # Récupérer l'évolution
        cursor.execute('''
            SELECT exam_date, diagnosis_change, confidence_change, size_change,
                   evolution_type, notes
            FROM tumor_evolution
            WHERE patient_id = ?
            ORDER BY exam_date ASC
        ''', (patient_id,))

        evolution_details = []
        for row in cursor.fetchall():
            evolution_details.append({
                'exam_date': row[0],
                'diagnosis_change': row[1],
                'confidence_change': row[2],
                'size_change': row[3],
                'evolution_type': row[4],
                'notes': row[5]
            })

        conn.close()

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
        metrics = calculate_patient_metrics(analyses, evolution_details)

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

        data = request.get_json()

        # Validation des données requises
        required_fields = ['patient_id', 'patient_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'Le champ {field} est requis'}), 400

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Vérifier si le patient existe déjà
        cursor.execute('''
            SELECT COUNT(*) FROM patients
            WHERE patient_id = ? AND doctor_id = ?
        ''', (data['patient_id'], doctor_id))

        if cursor.fetchone()[0] > 0:
            return jsonify({'success': False, 'error': 'Un patient avec cet ID existe déjà'}), 400

        # Insérer le nouveau patient
        cursor.execute('''
            INSERT INTO patients
            (patient_id, patient_name, doctor_id, date_of_birth, gender, phone, email,
             address, emergency_contact_name, emergency_contact_phone, medical_history,
             allergies, current_medications, insurance_number, notes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            data['patient_id'],
            data['patient_name'],
            doctor_id,
            data.get('date_of_birth'),
            data.get('gender'),
            data.get('phone'),
            data.get('email'),
            data.get('address'),
            data.get('emergency_contact_name'),
            data.get('emergency_contact_phone'),
            data.get('medical_history'),
            data.get('allergies'),
            data.get('current_medications'),
            data.get('insurance_number'),
            data.get('notes')
        ))

        conn.commit()
        conn.close()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Vérifier que le patient appartient au médecin
        cursor.execute('''
            SELECT COUNT(*) FROM patients
            WHERE patient_id = ? AND doctor_id = ?
        ''', (patient_id, doctor_id))

        if cursor.fetchone()[0] == 0:
            return jsonify({'success': False, 'error': 'Patient non trouvé'}), 404

        # Mettre à jour le patient
        cursor.execute('''
            UPDATE patients SET
                patient_name = ?, date_of_birth = ?, gender = ?, phone = ?, email = ?,
                address = ?, emergency_contact_name = ?, emergency_contact_phone = ?,
                medical_history = ?, allergies = ?, current_medications = ?,
                insurance_number = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
            WHERE patient_id = ? AND doctor_id = ?
        ''', (
            data.get('patient_name'),
            data.get('date_of_birth'),
            data.get('gender'),
            data.get('phone'),
            data.get('email'),
            data.get('address'),
            data.get('emergency_contact_name'),
            data.get('emergency_contact_phone'),
            data.get('medical_history'),
            data.get('allergies'),
            data.get('current_medications'),
            data.get('insurance_number'),
            data.get('notes'),
            patient_id,
            doctor_id
        ))

        conn.commit()
        conn.close()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Vérifier que le patient appartient au médecin
        cursor.execute('''
            SELECT COUNT(*) FROM patients
            WHERE patient_id = ? AND doctor_id = ?
        ''', (patient_id, doctor_id))

        if cursor.fetchone()[0] == 0:
            return jsonify({'success': False, 'error': 'Patient non trouvé'}), 404

        # Supprimer toutes les données associées au patient
        tables_to_clean = [
            'medical_alerts',
            'tumor_evolution',
            'analyses',
            'patients'
        ]

        for table in tables_to_clean:
            if table == 'patients':
                cursor.execute(f'DELETE FROM {table} WHERE patient_id = ? AND doctor_id = ?',
                             (patient_id, doctor_id))
            else:
                cursor.execute(f'DELETE FROM {table} WHERE patient_id = ?', (patient_id,))

        conn.commit()
        conn.close()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Récupérer les informations complètes du patient
        cursor.execute('''
            SELECT patient_id, patient_name, date_of_birth, gender, phone, email,
                   address, emergency_contact_name, emergency_contact_phone,
                   medical_history, allergies, current_medications, insurance_number,
                   notes, first_analysis_date, last_analysis_date, total_analyses,
                   created_at, updated_at
            FROM patients
            WHERE patient_id = ? AND doctor_id = ?
        ''', (patient_id, doctor_id))

        patient_data = cursor.fetchone()
        if not patient_data:
            return jsonify({'success': False, 'error': 'Patient non trouvé'}), 404

        patient = {
            'patient_id': patient_data[0],
            'patient_name': patient_data[1],
            'date_of_birth': datetime.strptime(patient_data[2], '%Y-%m-%d') if patient_data[2] else None,
            'gender': patient_data[3],
            'phone': patient_data[4],
            'email': patient_data[5],
            'address': patient_data[6],
            'emergency_contact_name': patient_data[7],
            'emergency_contact_phone': patient_data[8],
            'medical_history': patient_data[9],
            'allergies': patient_data[10],
            'current_medications': patient_data[11],
            'insurance_number': patient_data[12],
            'notes': patient_data[13],
            'first_analysis_date': patient_data[14],
            'last_analysis_date': patient_data[15],
            'total_analyses': patient_data[16],
            'created_at': patient_data[17],
            'updated_at': patient_data[18]
        }

        conn.close()

        return jsonify({
            'success': True,
            'data': patient
        })

    except Exception as e:
        print(f"Erreur détails patient: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Démarrage de l'application sur le device: {device}")
    print(f"Modèle chargé: {'Oui' if model is not None else 'Non'}")
    
    # Port pour le déploiement (Render, Heroku, etc.)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
