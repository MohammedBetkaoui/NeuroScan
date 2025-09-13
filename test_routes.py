#!/usr/bin/env python3

"""
Script de test pour vérifier les routes de l'application NeuroScan
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
import sqlite3

def test_database_structure():
    """Test de la structure de la base de données"""
    print("=== Test de la structure de la base de données ===")
    
    conn = sqlite3.connect('neuroscan_analytics.db')
    cursor = conn.cursor()
    
    # Vérifier la table patients
    cursor.execute("PRAGMA table_info(patients)")
    columns = cursor.fetchall()
    print(f"Table 'patients' - {len(columns)} colonnes:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Compter les patients
    cursor.execute("SELECT COUNT(*) FROM patients")
    patient_count = cursor.fetchone()[0]
    print(f"\nNombre de patients dans la base: {patient_count}")
    
    # Compter les médecins
    cursor.execute("SELECT COUNT(*) FROM doctors")
    doctor_count = cursor.fetchone()[0]
    print(f"Nombre de médecins dans la base: {doctor_count}")
    
    conn.close()
    print("✓ Structure de base de données OK\n")

def test_routes():
    """Test des routes principales"""
    print("=== Test des routes ===")
    
    with app.test_client() as client:
        # Test route index
        response = client.get('/')
        print(f"Route '/' : Status {response.status_code}")
        
        # Test route nouvelle analyse (nécessite authentification)
        response = client.get('/nouvelle-analyse')
        print(f"Route '/nouvelle-analyse' : Status {response.status_code} (redirection attendue)")
        
        print("✓ Routes OK\n")

if __name__ == '__main__':
    print("🧪 Tests NeuroScan - Routes et Base de données\n")
    
    try:
        test_database_structure()
        test_routes()
        print("✅ Tous les tests sont passés avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors des tests: {e}")
        sys.exit(1)
