#!/usr/bin/env python3
"""
Script pour créer des données de test pour le système de suivi des tumeurs
"""

import sqlite3
import json
from datetime import datetime, timedelta
import random

DATABASE_PATH = 'neuroscan_analytics.db'

def create_test_data():
    """Créer des données de test pour démontrer le système de suivi"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Patients de test
    test_patients = [
        {
            'patient_id': 'P001',
            'patient_name': 'Jean Dupont',
            'analyses': [
                {'days_ago': 90, 'diagnosis': 'Normal', 'confidence': 0.95},
                {'days_ago': 60, 'diagnosis': 'Gliome', 'confidence': 0.78},
                {'days_ago': 30, 'diagnosis': 'Gliome', 'confidence': 0.85},
                {'days_ago': 0, 'diagnosis': 'Gliome', 'confidence': 0.92}
            ]
        },
        {
            'patient_id': 'P002',
            'patient_name': 'Marie Martin',
            'analyses': [
                {'days_ago': 120, 'diagnosis': 'Méningiome', 'confidence': 0.82},
                {'days_ago': 90, 'diagnosis': 'Méningiome', 'confidence': 0.88},
                {'days_ago': 60, 'diagnosis': 'Méningiome', 'confidence': 0.75},
                {'days_ago': 30, 'diagnosis': 'Normal', 'confidence': 0.91}
            ]
        },
        {
            'patient_id': 'P003',
            'patient_name': 'Pierre Durand',
            'analyses': [
                {'days_ago': 45, 'diagnosis': 'Tumeur pituitaire', 'confidence': 0.89},
                {'days_ago': 15, 'diagnosis': 'Tumeur pituitaire', 'confidence': 0.93}
            ]
        }
    ]
    
    # Mapping des diagnostics vers les classes
    diagnosis_mapping = {
        'Normal': 0,
        'Gliome': 1,
        'Méningiome': 2,
        'Tumeur pituitaire': 3
    }
    
    # Insérer les données de test
    for patient in test_patients:
        patient_id = patient['patient_id']
        patient_name = patient['patient_name']
        
        previous_analysis_id = None
        
        for i, analysis in enumerate(patient['analyses']):
            # Calculer la date d'examen
            exam_date = (datetime.now() - timedelta(days=analysis['days_ago'])).date()
            
            # Créer les probabilités
            predicted_class = diagnosis_mapping[analysis['diagnosis']]
            probabilities = {
                'Normal': 0.1,
                'Gliome': 0.1,
                'Méningiome': 0.1,
                'Tumeur pituitaire': 0.1
            }
            probabilities[analysis['diagnosis']] = analysis['confidence']
            
            # Normaliser les probabilités
            total = sum(probabilities.values())
            for key in probabilities:
                probabilities[key] = probabilities[key] / total
            
            # Estimer la taille de la tumeur
            tumor_size_estimate = None
            if predicted_class != 0:  # Si ce n'est pas normal
                base_size = {1: 2.5, 2: 1.8, 3: 1.2}.get(predicted_class, 2.0)
                tumor_size_estimate = base_size * analysis['confidence'] * (0.8 + 0.4 * random.random())
            
            # Insérer l'analyse
            cursor.execute('''
                INSERT INTO analyses
                (timestamp, filename, patient_id, patient_name, exam_date, predicted_class, 
                 predicted_label, confidence, probabilities, description, recommendations, 
                 processing_time, user_session, ip_address, tumor_size_estimate, previous_analysis_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now() - timedelta(days=analysis['days_ago']),
                f"test_image_{patient_id}_{i}.jpg",
                patient_id,
                patient_name,
                exam_date,
                predicted_class,
                analysis['diagnosis'],
                analysis['confidence'],
                json.dumps(probabilities),
                f"Analyse de test pour {patient_name}",
                json.dumps([f"Recommandation pour {analysis['diagnosis']}"]),
                2.5,
                'test_session',
                '127.0.0.1',
                tumor_size_estimate,
                previous_analysis_id
            ))
            
            analysis_id = cursor.lastrowid
            
            # Analyser l'évolution si il y a une analyse précédente
            if previous_analysis_id:
                # Récupérer l'analyse précédente
                cursor.execute('''
                    SELECT predicted_label, confidence, tumor_size_estimate
                    FROM analyses WHERE id = ?
                ''', (previous_analysis_id,))
                prev_data = cursor.fetchone()
                
                if prev_data:
                    prev_label, prev_confidence, prev_size = prev_data
                    
                    # Analyser les changements
                    diagnosis_change = None
                    if prev_label != analysis['diagnosis']:
                        diagnosis_change = f"{prev_label} → {analysis['diagnosis']}"
                    
                    confidence_change = analysis['confidence'] - prev_confidence
                    
                    size_change = None
                    if tumor_size_estimate and prev_size:
                        size_change = tumor_size_estimate - prev_size
                    
                    # Déterminer le type d'évolution
                    evolution_type = "stable"
                    if diagnosis_change:
                        if "Normal" in diagnosis_change:
                            evolution_type = "amélioration" if analysis['diagnosis'] == "Normal" else "dégradation"
                        else:
                            evolution_type = "changement_type"
                    elif size_change and abs(size_change) > 0.2:
                        evolution_type = "croissance" if size_change > 0 else "réduction"
                    elif abs(confidence_change) > 0.1:
                        evolution_type = "confiance_modifiée"
                    
                    # Générer des notes
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
                    ''', (patient_id, analysis_id, exam_date, diagnosis_change, 
                          confidence_change, size_change, evolution_type, notes_text))
            
            previous_analysis_id = analysis_id
    
    conn.commit()
    conn.close()
    print("Données de test créées avec succès!")

if __name__ == "__main__":
    create_test_data()
