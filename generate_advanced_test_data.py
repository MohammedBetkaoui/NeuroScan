#!/usr/bin/env python3
"""
Script pour générer des données de test avancées pour l'espace professionnel
"""

import sqlite3
import json
import random
from datetime import datetime, timedelta

DATABASE_PATH = 'neuroscan_analytics.db'

def generate_advanced_test_data():
    """Générer des données de test avec des patterns intéressants pour les alertes"""
    
    tumor_types = [
        {'class': 0, 'label': 'Normal'},
        {'class': 1, 'label': 'Gliome'},
        {'class': 2, 'label': 'Méningiome'},
        {'class': 3, 'label': 'Tumeur pituitaire'}
    ]
    
    filenames = [
        'brain_scan_001.jpg', 'mri_patient_002.png', 'irm_cerveau_003.jpg',
        'brain_tumor_004.png', 'scan_medical_005.jpg', 'patient_006_irm.png'
    ]
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    now = datetime.now()
    print("🧪 Génération de données de test avancées...")
    
    # 1. Générer des données pour les 30 derniers jours avec des patterns
    for days_ago in range(30, 0, -1):
        date = now - timedelta(days=days_ago)
        
        # Simuler une baisse de confiance il y a 5 jours
        if days_ago == 5:
            confidence_factor = 0.7  # Confiance plus faible
            daily_count = random.randint(15, 25)  # Plus d'analyses
        elif days_ago <= 7:
            confidence_factor = 0.85
            daily_count = random.randint(8, 15)
        else:
            confidence_factor = 1.0
            daily_count = random.randint(5, 12)
        
        for _ in range(daily_count):
            hour = random.randint(8, 18)
            minute = random.randint(0, 59)
            timestamp = date.replace(hour=hour, minute=minute)
            
            # Distribution des tumeurs
            tumor_weights = [0.6, 0.15, 0.15, 0.1]
            tumor_type = random.choices(tumor_types, weights=tumor_weights)[0]
            
            # Générer probabilités avec facteur de confiance
            base_confidence = random.uniform(0.7, 0.95) * confidence_factor
            probabilities = generate_probabilities(tumor_type, tumor_types, base_confidence)
            
            insert_analysis(cursor, timestamp, tumor_type, probabilities, 
                          base_confidence, filenames)
    
    # 2. Générer des données pour aujourd'hui avec pic d'activité
    print("📈 Génération du pic d'activité d'aujourd'hui...")
    today_analyses = random.randint(20, 30)  # Pic d'activité
    
    for _ in range(today_analyses):
        hour = random.randint(8, 18)
        minute = random.randint(0, 59)
        timestamp = now.replace(hour=hour, minute=minute)
        
        # Quelques analyses à faible confiance pour déclencher l'alerte
        if random.random() < 0.15:  # 15% de chance d'avoir une faible confiance
            confidence_factor = 0.6  # Confiance < 70%
        else:
            confidence_factor = random.uniform(0.8, 0.95)
        
        tumor_weights = [0.6, 0.15, 0.15, 0.1]
        tumor_type = random.choices(tumor_types, weights=tumor_weights)[0]
        
        probabilities = generate_probabilities(tumor_type, tumor_types, confidence_factor)
        
        insert_analysis(cursor, timestamp, tumor_type, probabilities, 
                      confidence_factor, filenames)
    
    # 3. Mettre à jour les statistiques quotidiennes
    print("📊 Mise à jour des statistiques...")
    update_daily_stats(cursor)
    
    conn.commit()
    
    # Afficher les statistiques
    cursor.execute('SELECT COUNT(*) FROM analyses')
    total_analyses = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM analyses WHERE DATE(timestamp) = DATE("now")')
    today_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM analyses WHERE DATE(timestamp) = DATE("now") AND confidence < 0.7')
    low_confidence_today = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(confidence) FROM analyses WHERE DATE(timestamp) = DATE("now")')
    today_avg_confidence = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(confidence) FROM analyses WHERE DATE(timestamp) >= DATE("now", "-7 days") AND DATE(timestamp) < DATE("now")')
    week_avg_confidence = cursor.fetchone()[0]
    
    print(f"\n✅ Données de test avancées générées!")
    print(f"📊 Total analyses: {total_analyses}")
    print(f"📅 Analyses aujourd'hui: {today_count}")
    print(f"⚠️  Analyses faible confiance aujourd'hui: {low_confidence_today}")
    print(f"📈 Confiance moyenne aujourd'hui: {today_avg_confidence*100:.1f}%")
    print(f"📈 Confiance moyenne semaine: {week_avg_confidence*100:.1f}%" if week_avg_confidence else "N/A")
    
    # Prédire les alertes qui seront générées
    print(f"\n🚨 Alertes prévues:")
    if low_confidence_today > 0:
        print(f"   - Analyses à faible confiance: {low_confidence_today} analyse(s)")
    
    if today_count > 15:
        print(f"   - Pic d'activité détecté: {today_count} analyses (> moyenne)")
    
    if today_avg_confidence and week_avg_confidence and today_avg_confidence < week_avg_confidence * 0.9:
        print(f"   - Baisse de confiance: {today_avg_confidence*100:.1f}% vs {week_avg_confidence*100:.1f}%")
    
    conn.close()

def generate_probabilities(tumor_type, tumor_types, base_confidence):
    """Générer des probabilités réalistes"""
    probabilities = {}
    
    for t in tumor_types:
        if t['class'] == tumor_type['class']:
            probabilities[t['label']] = base_confidence
        else:
            probabilities[t['label']] = random.uniform(0.01, 0.3)
    
    # Normaliser
    total = sum(probabilities.values())
    probabilities = {k: v/total for k, v in probabilities.items()}
    return probabilities

def insert_analysis(cursor, timestamp, tumor_type, probabilities, confidence, filenames):
    """Insérer une analyse dans la base de données"""
    
    descriptions = {
        'Normal': "Structures cérébrales normales sans anomalie détectable.",
        'Gliome': "Lésion compatible avec un gliome nécessitant une évaluation approfondie.",
        'Méningiome': "Formation tumorale évocatrice d'un méningiome bien délimité.",
        'Tumeur pituitaire': "Anomalie hypophysaire nécessitant un bilan endocrinologique."
    }
    
    recommendations = {
        'Normal': ["Suivi de routine", "Consultation radiologique"],
        'Gliome': ["Biopsie recommandée", "Consultation neuro-oncologique"],
        'Méningiome': ["Surveillance radiologique", "Évaluation neurochirurgicale"],
        'Tumeur pituitaire': ["Bilan hormonal", "Consultation endocrinologique"]
    }
    
    cursor.execute('''
        INSERT INTO analyses 
        (timestamp, filename, predicted_class, predicted_label, confidence, 
         probabilities, description, recommendations, processing_time, 
         user_session, ip_address)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp.isoformat(),
        random.choice(filenames),
        tumor_type['class'],
        tumor_type['label'],
        confidence,
        json.dumps(probabilities),
        descriptions[tumor_type['label']],
        json.dumps(recommendations[tumor_type['label']]),
        random.uniform(2.5, 8.5),
        f"session_{random.randint(1000, 9999)}",
        f"192.168.1.{random.randint(1, 254)}"
    ))

def update_daily_stats(cursor):
    """Mettre à jour les statistiques quotidiennes"""
    cursor.execute('''
        INSERT OR REPLACE INTO daily_stats 
        (date, total_analyses, normal_count, gliome_count, meningiome_count, 
         pituitary_count, avg_confidence, avg_processing_time)
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as total_analyses,
            SUM(CASE WHEN predicted_label = 'Normal' THEN 1 ELSE 0 END) as normal_count,
            SUM(CASE WHEN predicted_label = 'Gliome' THEN 1 ELSE 0 END) as gliome_count,
            SUM(CASE WHEN predicted_label = 'Méningiome' THEN 1 ELSE 0 END) as meningiome_count,
            SUM(CASE WHEN predicted_label = 'Tumeur pituitaire' THEN 1 ELSE 0 END) as pituitary_count,
            AVG(confidence) as avg_confidence,
            AVG(processing_time) as avg_processing_time
        FROM analyses
        GROUP BY DATE(timestamp)
    ''')

if __name__ == "__main__":
    generate_advanced_test_data()
