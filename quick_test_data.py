#!/usr/bin/env python3
"""
Script rapide pour générer quelques données de test pour aujourd'hui
"""

import sqlite3
import json
import random
from datetime import datetime

DATABASE_PATH = 'neuroscan_analytics.db'

def add_test_data():
    """Ajouter quelques données de test pour aujourd'hui"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    now = datetime.now()
    tumor_types = ['Normal', 'Gliome', 'Méningiome', 'Tumeur pituitaire']
    filenames = ['test_brain_001.jpg', 'test_mri_002.png', 'test_scan_003.jpg']
    
    print("🧪 Génération de données de test pour aujourd'hui...")
    
    # Générer des analyses pour différentes heures d'aujourd'hui
    test_hours = [9, 10, 11, 14, 15, 16, 17]
    total_added = 0
    
    for hour in test_hours:
        # 1-4 analyses par heure
        num_analyses = random.randint(1, 4)
        
        for _ in range(num_analyses):
            minute = random.randint(0, 59)
            timestamp = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # Choisir un type de tumeur avec distribution réaliste
            tumor_weights = [0.6, 0.15, 0.15, 0.1]  # Normal plus fréquent
            tumor_type = random.choices(tumor_types, weights=tumor_weights)[0]
            tumor_class = tumor_types.index(tumor_type)
            
            # Générer des probabilités réalistes
            probabilities = {}
            for t in tumor_types:
                if t == tumor_type:
                    probabilities[t] = random.uniform(0.7, 0.95)
                else:
                    probabilities[t] = random.uniform(0.01, 0.3)
            
            # Normaliser les probabilités
            total_prob = sum(probabilities.values())
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
            
            confidence = probabilities[tumor_type]
            
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
                tumor_class,
                tumor_type,
                confidence,
                json.dumps(probabilities),
                f"Analyse de test pour {tumor_type}",
                json.dumps([f"Recommandation test pour {tumor_type}"]),
                random.uniform(2.5, 8.5),
                f"test_session_{random.randint(1000, 9999)}",
                '127.0.0.1'
            ))
            
            total_added += 1
    
    conn.commit()
    
    # Vérifier les données ajoutées
    cursor.execute('SELECT COUNT(*) FROM analyses WHERE DATE(timestamp) = DATE("now")')
    today_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM analyses')
    total_count = cursor.fetchone()[0]
    
    print(f"✅ {total_added} nouvelles analyses ajoutées")
    print(f"📊 Total analyses aujourd'hui: {today_count}")
    print(f"📊 Total analyses dans la base: {total_count}")
    
    # Afficher la répartition d'aujourd'hui
    cursor.execute('''
        SELECT predicted_label, COUNT(*) 
        FROM analyses 
        WHERE DATE(timestamp) = DATE("now")
        GROUP BY predicted_label
    ''')
    
    today_distribution = cursor.fetchall()
    if today_distribution:
        print("📈 Répartition aujourd'hui:")
        for label, count in today_distribution:
            print(f"   - {label}: {count}")
    
    conn.close()

if __name__ == "__main__":
    add_test_data()
