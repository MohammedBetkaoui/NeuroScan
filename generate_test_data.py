#!/usr/bin/env python3
"""
Script pour générer des données de test pour le tableau de bord NeuroScan Pro
"""

import sqlite3
import json
import random
from datetime import datetime, timedelta

DATABASE_PATH = 'neuroscan_analytics.db'

def generate_test_data():
    """Générer des données de test pour la base de données"""
    
    # Données de test
    tumor_types = [
        {'class': 0, 'label': 'Normal'},
        {'class': 1, 'label': 'Gliome'},
        {'class': 2, 'label': 'Méningiome'},
        {'class': 3, 'label': 'Tumeur pituitaire'}
    ]
    
    filenames = [
        'brain_scan_001.jpg', 'mri_patient_002.png', 'irm_cerveau_003.jpg',
        'brain_tumor_004.png', 'scan_medical_005.jpg', 'patient_006_irm.png',
        'neurologie_007.jpg', 'brain_analysis_008.png', 'medical_scan_009.jpg',
        'irm_diagnostic_010.png', 'brain_study_011.jpg', 'neuro_scan_012.png'
    ]
    
    recommendations_templates = {
        'Normal': [
            "Aucune anomalie détectée dans cette analyse",
            "Suivi de routine recommandé selon les protocoles standards",
            "Consultation avec un radiologue pour confirmation"
        ],
        'Gliome': [
            "Biopsie recommandée pour confirmation histologique",
            "IRM de suivi dans 3 mois pour évaluation de la croissance",
            "Consultation avec un neuro-oncologue spécialisé",
            "Évaluation neuropsychologique recommandée"
        ],
        'Méningiome': [
            "Surveillance radiologique tous les 6 mois",
            "Consultation neurochirurgicale pour évaluation",
            "IRM avec contraste pour meilleure caractérisation",
            "Évaluation des symptômes neurologiques"
        ],
        'Tumeur pituitaire': [
            "Bilan hormonal complet recommandé",
            "Consultation endocrinologique urgente",
            "IRM hypophysaire avec coupes fines",
            "Évaluation ophtalmologique (champ visuel)"
        ]
    }
    
    descriptions_templates = {
        'Normal': [
            "L'analyse révèle des structures cérébrales normales sans anomalie détectable.",
            "Aucune lésion suspecte identifiée sur cette imagerie cérébrale.",
            "Les tissus cérébraux présentent un aspect normal et homogène."
        ],
        'Gliome': [
            "Présence d'une lésion compatible avec un gliome de grade indéterminé.",
            "Masse tissulaire suspecte avec caractéristiques évocatrices d'un gliome.",
            "Lésion infiltrante suggérant un processus glial néoplasique."
        ],
        'Méningiome': [
            "Lésion extra-axiale évocatrice d'un méningiome bien délimité.",
            "Masse arrondie avec prise de contraste homogène, typique d'un méningiome.",
            "Formation tumorale bénigne compatible avec un méningiome."
        ],
        'Tumeur pituitaire': [
            "Anomalie de la région sellaire évocatrice d'un adénome hypophysaire.",
            "Lésion de l'hypophyse avec extension suprasellaire possible.",
            "Masse hypophysaire nécessitant une évaluation endocrinologique."
        ]
    }
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Générer des analyses sur les 30 derniers jours
    base_date = datetime.now() - timedelta(days=30)
    
    print("Génération de données de test...")
    
    for day in range(30):
        current_date = base_date + timedelta(days=day)
        
        # Nombre d'analyses par jour (entre 1 et 15)
        daily_analyses = random.randint(1, 15)
        
        for _ in range(daily_analyses):
            # Choisir un type de tumeur avec des probabilités réalistes
            tumor_weights = [0.6, 0.15, 0.15, 0.1]  # Normal plus fréquent
            tumor_type = random.choices(tumor_types, weights=tumor_weights)[0]
            
            # Générer des probabilités réalistes
            probabilities = {}
            for t in tumor_types:
                if t['class'] == tumor_type['class']:
                    # Probabilité élevée pour la classe prédite
                    probabilities[t['label']] = random.uniform(0.7, 0.95)
                else:
                    # Probabilités faibles pour les autres classes
                    probabilities[t['label']] = random.uniform(0.01, 0.3)
            
            # Normaliser les probabilités
            total = sum(probabilities.values())
            probabilities = {k: v/total for k, v in probabilities.items()}
            
            # Confidence = probabilité de la classe prédite
            confidence = probabilities[tumor_type['label']]
            
            # Choisir des recommandations et description
            available_recs = recommendations_templates[tumor_type['label']]
            num_recs = min(random.randint(2, 4), len(available_recs))
            recommendations = random.sample(available_recs, num_recs)
            description = random.choice(descriptions_templates[tumor_type['label']])
            
            # Générer un timestamp aléatoire dans la journée
            hour = random.randint(8, 18)  # Heures de travail
            minute = random.randint(0, 59)
            timestamp = current_date.replace(hour=hour, minute=minute)
            
            # Insérer l'analyse
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
                description,
                json.dumps(recommendations),
                random.uniform(2.5, 8.5),  # Temps de traitement
                f"session_{random.randint(1000, 9999)}",
                f"192.168.1.{random.randint(1, 254)}"
            ))
    
    # Mettre à jour les statistiques quotidiennes
    print("Mise à jour des statistiques quotidiennes...")
    
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
    
    conn.commit()
    
    # Afficher les statistiques
    cursor.execute('SELECT COUNT(*) FROM analyses')
    total_analyses = cursor.fetchone()[0]
    
    cursor.execute('SELECT predicted_label, COUNT(*) FROM analyses GROUP BY predicted_label')
    distribution = cursor.fetchall()
    
    print(f"\n✅ Données de test générées avec succès!")
    print(f"📊 Total des analyses: {total_analyses}")
    print(f"📈 Répartition par type:")
    for label, count in distribution:
        percentage = (count / total_analyses) * 100
        print(f"   - {label}: {count} ({percentage:.1f}%)")
    
    conn.close()

if __name__ == "__main__":
    generate_test_data()
