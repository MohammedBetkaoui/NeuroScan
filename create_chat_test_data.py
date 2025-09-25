#!/usr/bin/env python3
"""
Script de test pour créer des données de test pour le chat médical
"""

import sqlite3
import sys
import os
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = 'neuroscan_analytics.db'

def create_test_doctor():
    """Créer un médecin de test"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Vérifier si le médecin test existe déjà
    cursor.execute('SELECT id FROM doctors WHERE email = ?', ('test@neuroscan.com',))
    existing = cursor.fetchone()
    
    if existing:
        print(f"✓ Médecin de test existe déjà (ID: {existing[0]})")
        conn.close()
        return existing[0]
    
    # Créer le médecin
    password_hash = generate_password_hash('test123')
    cursor.execute('''
        INSERT INTO doctors 
        (email, password_hash, first_name, last_name, specialty, hospital, license_number, phone)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        'test@neuroscan.com',
        password_hash,
        'Jean',
        'Dupont',
        'Neurologie',
        'CHU Test',
        '12345678',
        '+33123456789'
    ))
    
    doctor_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"✓ Médecin de test créé (ID: {doctor_id})")
    print("  Email: test@neuroscan.com")
    print("  Mot de passe: test123")
    return doctor_id

def create_test_patients(doctor_id):
    """Créer des patients de test"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    test_patients = [
        {
            'patient_id': 'PAT001',
            'patient_name': 'Marie Martin',
            'date_of_birth': '1980-05-15',
            'gender': 'F',
            'phone': '+33123456789',
            'email': 'marie.martin@email.com',
            'medical_history': 'Antécédents de migraines chroniques',
            'allergies': 'Aucune allergie connue',
            'current_medications': 'Sumatriptan en cas de crise'
        },
        {
            'patient_id': 'PAT002',
            'patient_name': 'Pierre Durand',
            'date_of_birth': '1965-09-23',
            'gender': 'M',
            'phone': '+33987654321',
            'email': 'pierre.durand@email.com',
            'medical_history': 'Hypertension artérielle, diabète type 2',
            'allergies': 'Allergie à la pénicilline',
            'current_medications': 'Amlodipine, Metformine'
        },
        {
            'patient_id': 'PAT003',
            'patient_name': 'Sophie Lefebvre',
            'date_of_birth': '1975-12-08',
            'gender': 'F',
            'phone': '+33555666777',
            'email': 'sophie.lefebvre@email.com',
            'medical_history': 'Epilepsie contrôlée',
            'allergies': 'Aucune',
            'current_medications': 'Lévétiracétam 500mg x2/j'
        }
    ]
    
    for patient in test_patients:
        # Vérifier si le patient existe déjà
        cursor.execute('''
            SELECT id FROM patients 
            WHERE patient_id = ? AND doctor_id = ?
        ''', (patient['patient_id'], doctor_id))
        
        if cursor.fetchone():
            continue
            
        cursor.execute('''
            INSERT INTO patients
            (patient_id, patient_name, date_of_birth, gender, phone, email,
             medical_history, allergies, current_medications, doctor_id,
             first_analysis_date, last_analysis_date, total_analyses)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient['patient_id'],
            patient['patient_name'],
            patient['date_of_birth'],
            patient['gender'],
            patient['phone'],
            patient['email'],
            patient['medical_history'],
            patient['allergies'],
            patient['current_medications'],
            doctor_id,
            datetime.now().date(),
            datetime.now().date(),
            0
        ))
        
        print(f"✓ Patient créé: {patient['patient_name']} ({patient['patient_id']})")
    
    conn.commit()
    conn.close()

def create_test_conversations(doctor_id):
    """Créer des conversations de test"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Vérifier si des conversations existent déjà
    cursor.execute('SELECT COUNT(*) FROM chat_conversations WHERE doctor_id = ?', (doctor_id,))
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"✓ {count} conversations de test existent déjà")
        conn.close()
        return
    
    test_conversations = [
        {
            'title': 'Consultation céphalées - Marie Martin',
            'patient_id': 'PAT001'
        },
        {
            'title': 'Suivi post-IRM - Pierre Durand',
            'patient_id': 'PAT002'
        },
        {
            'title': 'Questions générales neurologie',
            'patient_id': None
        }
    ]
    
    for conv in test_conversations:
        cursor.execute('''
            INSERT INTO chat_conversations (doctor_id, patient_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            doctor_id,
            conv['patient_id'],
            conv['title'],
            datetime.now() - timedelta(days=2),
            datetime.now() - timedelta(hours=1)
        ))
        
        conv_id = cursor.lastrowid
        
        # Ajouter quelques messages de test
        test_messages = [
            {
                'role': 'user',
                'content': 'Bonjour, j\'ai des questions concernant l\'interprétation d\'une IRM cérébrale.',
                'timestamp': datetime.now() - timedelta(hours=2)
            },
            {
                'role': 'assistant',
                'content': 'Bonjour ! Je serais ravi de vous aider avec l\'interprétation d\'imagerie cérébrale. Pouvez-vous me décrire les findings principaux que vous observez sur cette IRM ?',
                'timestamp': datetime.now() - timedelta(hours=2, minutes=-2),
                'confidence_score': 0.95
            },
            {
                'role': 'user',
                'content': 'On observe une lésion hyperintense en T2 dans le lobe frontal droit, avec un léger effet de masse.',
                'timestamp': datetime.now() - timedelta(hours=1, minutes=30)
            },
            {
                'role': 'assistant',
                'content': 'Cette description suggère plusieurs possibilités diagnostiques. Une lésion hyperintense en T2 avec effet de masse dans le lobe frontal peut évoquer :\n\n1. **Gliome de bas grade** - particulièrement si la lésion est bien délimitée\n2. **Méningiome** - si la lésion est extra-axiale avec prise de contraste homogène\n3. **Métastase** - surtout s\'il y a un contexte oncologique\n\nPour affiner le diagnostic, il serait important de connaître :\n- La prise de contraste après injection de gadolinium\n- L\'âge du patient et le contexte clinique\n- La présence d\'œdème périlésionnel\n\n**Disclaimer médical :** Ces informations sont fournies à titre éducatif et ne remplacent pas une consultation médicale professionnelle.',
                'timestamp': datetime.now() - timedelta(hours=1, minutes=28),
                'confidence_score': 0.92
            }
        ]
        
        for msg in test_messages:
            cursor.execute('''
                INSERT INTO chat_messages
                (conversation_id, role, content, timestamp, is_medical_query, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                conv_id,
                msg['role'],
                msg['content'],
                msg['timestamp'],
                True,
                msg.get('confidence_score')
            ))
        
        print(f"✓ Conversation créée: {conv['title']} avec {len(test_messages)} messages")
    
    conn.commit()
    conn.close()

def create_test_analyses(doctor_id):
    """Créer quelques analyses de test pour enrichir le contexte patient"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Vérifier si des analyses existent déjà pour ce médecin
    cursor.execute('SELECT COUNT(*) FROM analyses WHERE doctor_id = ?', (doctor_id,))
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"✓ {count} analyses de test existent déjà")
        conn.close()
        return
    
    test_analyses = [
        {
            'patient_id': 'PAT001',
            'patient_name': 'Marie Martin',
            'exam_date': datetime.now().date() - timedelta(days=30),
            'predicted_class': 0,
            'predicted_label': 'Normal',
            'confidence': 0.94,
            'description': 'IRM cérébrale normale, pas d\'anomalie détectée'
        },
        {
            'patient_id': 'PAT002',
            'patient_name': 'Pierre Durand',
            'exam_date': datetime.now().date() - timedelta(days=15),
            'predicted_class': 2,
            'predicted_label': 'Méningiome',
            'confidence': 0.87,
            'description': 'Suspicion de méningiome frontal droit'
        },
        {
            'patient_id': 'PAT003',
            'patient_name': 'Sophie Lefebvre',
            'exam_date': datetime.now().date() - timedelta(days=7),
            'predicted_class': 0,
            'predicted_label': 'Normal',
            'confidence': 0.91,
            'description': 'Contrôle post-thérapeutique, pas de récidive épileptique'
        }
    ]
    
    for analysis in test_analyses:
        probabilities = {
            'Normal': 0.94 if analysis['predicted_class'] == 0 else 0.06,
            'Gliome': 0.02 if analysis['predicted_class'] != 1 else 0.88,
            'Méningiome': 0.87 if analysis['predicted_class'] == 2 else 0.03,
            'Tumeur pituitaire': 0.02
        }
        
        cursor.execute('''
            INSERT INTO analyses
            (filename, patient_id, patient_name, exam_date, predicted_class,
             predicted_label, confidence, probabilities, description,
             processing_time, doctor_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"test_{analysis['patient_id'].lower()}.dcm",
            analysis['patient_id'],
            analysis['patient_name'],
            analysis['exam_date'],
            analysis['predicted_class'],
            analysis['predicted_label'],
            analysis['confidence'],
            str(probabilities).replace("'", '"'),
            analysis['description'],
            2.5,
            doctor_id,
            datetime.now() - timedelta(days=(datetime.now().date() - analysis['exam_date']).days)
        ))
        
        print(f"✓ Analyse créée: {analysis['patient_name']} - {analysis['predicted_label']}")
    
    # Mettre à jour le compteur d'analyses des patients
    cursor.execute('''
        UPDATE patients 
        SET total_analyses = 1
        WHERE doctor_id = ? AND patient_id IN ('PAT001', 'PAT002', 'PAT003')
    ''', (doctor_id,))
    
    conn.commit()
    conn.close()

def main():
    """Fonction principale"""
    print("🧠 Création des données de test pour le chat médical NeuroScan")
    print("=" * 60)
    
    try:
        # Créer le médecin de test
        doctor_id = create_test_doctor()
        
        # Créer les patients de test
        create_test_patients(doctor_id)
        
        # Créer les analyses de test
        create_test_analyses(doctor_id)
        
        # Créer les conversations de test
        create_test_conversations(doctor_id)
        
        print("\n" + "=" * 60)
        print("✅ Données de test créées avec succès !")
        print("\nPour tester le chat médical :")
        print("1. Démarrez l'application Flask")
        print("2. Connectez-vous avec:")
        print("   Email: test@neuroscan.com")
        print("   Mot de passe: test123")
        print("3. Accédez au Chat Médical depuis le dashboard")
        print("4. Explorez les conversations de test ou créez-en de nouvelles")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des données de test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
