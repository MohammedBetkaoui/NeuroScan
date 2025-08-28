#!/usr/bin/env python3
"""
Script pour créer un compte médecin de test
"""

import sqlite3
from werkzeug.security import generate_password_hash
from datetime import datetime

def create_test_doctor():
    """Créer un compte médecin de test"""
    try:
        conn = sqlite3.connect('neuroscan_analytics.db')
        cursor = conn.cursor()
        
        # Vérifier si le médecin existe déjà
        cursor.execute('SELECT id FROM doctors WHERE email = ?', ('test@neuroscan.com',))
        if cursor.fetchone():
            print("✅ Le médecin de test existe déjà")
            print("📧 Email: test@neuroscan.com")
            print("🔑 Mot de passe: test123")
            conn.close()
            return
        
        # Créer le médecin de test
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
            'Hôpital Central',
            'MD123456',
            '+33 1 23 45 67 89'
        ))
        
        conn.commit()
        doctor_id = cursor.lastrowid
        
        print("✅ Médecin de test créé avec succès!")
        print(f"🆔 ID: {doctor_id}")
        print("📧 Email: test@neuroscan.com")
        print("🔑 Mot de passe: test123")
        print("👨‍⚕️ Nom: Dr. Jean Dupont")
        print("🏥 Spécialité: Neurologie")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Erreur lors de la création du médecin: {e}")

if __name__ == '__main__':
    create_test_doctor()
