#!/usr/bin/env python3
"""
Script pour ajouter des patients de test avec informations complètes
"""

import sqlite3
from datetime import datetime, timedelta
import random

def add_test_patients():
    """Ajouter des patients de test avec informations complètes"""
    try:
        conn = sqlite3.connect('neuroscan_analytics.db')
        cursor = conn.cursor()
        
        # Récupérer l'ID du médecin de test
        cursor.execute('SELECT id FROM doctors WHERE email = ?', ('test@neuroscan.com',))
        doctor_result = cursor.fetchone()
        if not doctor_result:
            print("❌ Médecin de test non trouvé. Exécutez d'abord create_test_doctor.py")
            return False
        
        doctor_id = doctor_result[0]
        
        # Patients de test avec informations complètes
        test_patients = [
            {
                'patient_id': 'PAT001',
                'patient_name': 'Marie Dubois',
                'date_of_birth': '1985-03-15',
                'gender': 'F',
                'phone': '+33 1 23 45 67 89',
                'email': 'marie.dubois@email.com',
                'address': '123 Rue de la Paix, 75001 Paris',
                'emergency_contact_name': 'Pierre Dubois',
                'emergency_contact_phone': '+33 6 12 34 56 78',
                'medical_history': 'Hypertension artérielle depuis 2018. Antécédents familiaux de cancer.',
                'allergies': 'Allergie à la pénicilline et aux fruits de mer.',
                'current_medications': 'Lisinopril 10mg/jour, Aspirine 75mg/jour',
                'insurance_number': '1850315123456',
                'notes': 'Patiente très coopérative. Anxiété légère avant les examens.'
            },
            {
                'patient_id': 'PAT002',
                'patient_name': 'Jean Martin',
                'date_of_birth': '1972-11-08',
                'gender': 'M',
                'phone': '+33 1 98 76 54 32',
                'email': 'jean.martin@email.com',
                'address': '456 Avenue des Champs, 69000 Lyon',
                'emergency_contact_name': 'Sophie Martin',
                'emergency_contact_phone': '+33 6 87 65 43 21',
                'medical_history': 'Diabète type 2 diagnostiqué en 2015. Chirurgie de la cataracte en 2020.',
                'allergies': 'Aucune allergie connue.',
                'current_medications': 'Metformine 1000mg 2x/jour, Atorvastatine 20mg/jour',
                'insurance_number': '1721108987654',
                'notes': 'Patient diabétique bien équilibré. Suivi régulier recommandé.'
            },
            {
                'patient_id': 'PAT003',
                'patient_name': 'Claire Rousseau',
                'date_of_birth': '1990-07-22',
                'gender': 'F',
                'phone': '+33 1 11 22 33 44',
                'email': 'claire.rousseau@email.com',
                'address': '789 Boulevard Saint-Michel, 33000 Bordeaux',
                'emergency_contact_name': 'Marc Rousseau',
                'emergency_contact_phone': '+33 6 44 33 22 11',
                'medical_history': 'Migraines chroniques depuis l\'adolescence. Aucun autre antécédent notable.',
                'allergies': 'Allergie au pollen et aux acariens.',
                'current_medications': 'Sumatriptan en cas de crise, Antihistaminiques saisonniers',
                'insurance_number': '2900722345678',
                'notes': 'Jeune patiente, migraines bien contrôlées avec traitement actuel.'
            },
            {
                'patient_id': 'PAT004',
                'patient_name': 'Robert Leroy',
                'date_of_birth': '1955-12-03',
                'gender': 'M',
                'phone': '+33 1 55 66 77 88',
                'email': 'robert.leroy@email.com',
                'address': '321 Rue Victor Hugo, 59000 Lille',
                'emergency_contact_name': 'Françoise Leroy',
                'emergency_contact_phone': '+33 6 88 77 66 55',
                'medical_history': 'Infarctus du myocarde en 2019. Hypertension. Hypercholestérolémie.',
                'allergies': 'Allergie à l\'iode (produits de contraste).',
                'current_medications': 'Clopidogrel 75mg/jour, Ramipril 5mg/jour, Simvastatine 40mg/jour',
                'insurance_number': '1551203456789',
                'notes': 'Patient cardiaque stable. Attention aux produits de contraste iodés.'
            },
            {
                'patient_id': 'PAT005',
                'patient_name': 'Isabelle Moreau',
                'date_of_birth': '1978-09-14',
                'gender': 'F',
                'phone': '+33 1 99 88 77 66',
                'email': 'isabelle.moreau@email.com',
                'address': '654 Place de la République, 13000 Marseille',
                'emergency_contact_name': 'Thomas Moreau',
                'emergency_contact_phone': '+33 6 66 77 88 99',
                'medical_history': 'Thyroïdectomie partielle en 2020. Hypothyroïdie substituée.',
                'allergies': 'Allergie au latex.',
                'current_medications': 'Lévothyroxine 100µg/jour',
                'insurance_number': '2780914567890',
                'notes': 'Patiente sous substitution thyroïdienne. Contrôles biologiques réguliers.'
            }
        ]
        
        added_count = 0
        updated_count = 0
        
        for patient_data in test_patients:
            # Vérifier si le patient existe déjà
            cursor.execute('''
                SELECT COUNT(*) FROM patients 
                WHERE patient_id = ? AND doctor_id = ?
            ''', (patient_data['patient_id'], doctor_id))
            
            if cursor.fetchone()[0] > 0:
                # Mettre à jour le patient existant
                cursor.execute('''
                    UPDATE patients SET
                        patient_name = ?, date_of_birth = ?, gender = ?, phone = ?, email = ?,
                        address = ?, emergency_contact_name = ?, emergency_contact_phone = ?,
                        medical_history = ?, allergies = ?, current_medications = ?,
                        insurance_number = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE patient_id = ? AND doctor_id = ?
                ''', (
                    patient_data['patient_name'],
                    patient_data['date_of_birth'],
                    patient_data['gender'],
                    patient_data['phone'],
                    patient_data['email'],
                    patient_data['address'],
                    patient_data['emergency_contact_name'],
                    patient_data['emergency_contact_phone'],
                    patient_data['medical_history'],
                    patient_data['allergies'],
                    patient_data['current_medications'],
                    patient_data['insurance_number'],
                    patient_data['notes'],
                    patient_data['patient_id'],
                    doctor_id
                ))
                updated_count += 1
                print(f"✅ Mis à jour: {patient_data['patient_name']} ({patient_data['patient_id']})")
            else:
                # Ajouter le nouveau patient
                cursor.execute('''
                    INSERT INTO patients 
                    (patient_id, patient_name, doctor_id, date_of_birth, gender, phone, email, 
                     address, emergency_contact_name, emergency_contact_phone, medical_history, 
                     allergies, current_medications, insurance_number, notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    patient_data['patient_id'],
                    patient_data['patient_name'],
                    doctor_id,
                    patient_data['date_of_birth'],
                    patient_data['gender'],
                    patient_data['phone'],
                    patient_data['email'],
                    patient_data['address'],
                    patient_data['emergency_contact_name'],
                    patient_data['emergency_contact_phone'],
                    patient_data['medical_history'],
                    patient_data['allergies'],
                    patient_data['current_medications'],
                    patient_data['insurance_number'],
                    patient_data['notes']
                ))
                added_count += 1
                print(f"✅ Ajouté: {patient_data['patient_name']} ({patient_data['patient_id']})")
        
        conn.commit()
        conn.close()
        
        print(f"\n🎉 Opération terminée!")
        print(f"📊 Patients ajoutés: {added_count}")
        print(f"🔄 Patients mis à jour: {updated_count}")
        print(f"📋 Total traité: {len(test_patients)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'ajout des patients: {e}")
        return False

if __name__ == '__main__':
    print("👥 Ajout de patients de test avec informations complètes...")
    print("=" * 60)
    
    success = add_test_patients()
    
    if success:
        print("\n🚀 Vous pouvez maintenant tester la page de gestion des patients!")
        print("🌐 Accédez à: http://localhost:5000/manage-patients")
    else:
        print("\n⚠️ Erreur lors de l'ajout des patients de test.")
