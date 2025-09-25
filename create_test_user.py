#!/usr/bin/env python3
"""
Créer un médecin de test pour les tests automatisés
"""

import sqlite3
import hashlib
from datetime import datetime

def create_test_doctor():
    """Créer un médecin de test"""
    
    # Connexion à la base de données
    conn = sqlite3.connect('neuroscan_analytics.db')
    cursor = conn.cursor()
    
    # Vérifier si le médecin de test existe déjà
    cursor.execute('SELECT id FROM doctors WHERE email = ?', ('test@neuroscan.com',))
    existing = cursor.fetchone()
    
    if existing:
        print(f"✅ Médecin de test existe déjà avec ID: {existing[0]}")
        conn.close()
        return existing[0]
    
    # Créer le hash du mot de passe
    password = "test123"
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Insérer le médecin de test
    cursor.execute('''
        INSERT INTO doctors (name, email, password_hash, speciality, registration_date)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        'Dr. Test User',
        'test@neuroscan.com',
        password_hash,
        'Neurologie',
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    
    doctor_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"✅ Médecin de test créé avec ID: {doctor_id}")
    print(f"   Email: test@neuroscan.com")
    print(f"   Mot de passe: {password}")
    
    return doctor_id

if __name__ == "__main__":
    create_test_doctor()
