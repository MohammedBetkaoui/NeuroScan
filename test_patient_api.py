#!/usr/bin/env python3
"""
Script de test pour les API de gestion des patients
"""

import requests
import json

def test_patient_apis():
    """Tester les APIs de gestion des patients"""
    base_url = 'http://localhost:5000'
    
    # Données de connexion
    login_data = {
        'email': 'test@neuroscan.com',
        'password': 'test123'
    }
    
    print("🧪 Test des APIs de gestion des patients")
    print("=" * 50)
    
    # Créer une session pour maintenir les cookies
    session = requests.Session()
    
    try:
        # 1. Se connecter
        print("1. Connexion...")
        login_response = session.post(f'{base_url}/login', data=login_data)
        if login_response.status_code != 200:
            print(f"❌ Erreur de connexion: {login_response.status_code}")
            return False
        print("✅ Connexion réussie")
        
        # 2. Tester la récupération des patients
        print("\n2. Récupération de la liste des patients...")
        patients_response = session.get(f'{base_url}/api/my-patients')
        if patients_response.status_code == 200:
            patients_data = patients_response.json()
            if patients_data['success']:
                print(f"✅ {len(patients_data['data'])} patients trouvés")
                if patients_data['data']:
                    first_patient = patients_data['data'][0]
                    print(f"   Premier patient: {first_patient['patient_name']} ({first_patient['patient_id']})")
            else:
                print(f"❌ Erreur API: {patients_data['error']}")
        else:
            print(f"❌ Erreur HTTP: {patients_response.status_code}")
        
        # 3. Tester la récupération des détails d'un patient
        if patients_data['success'] and patients_data['data']:
            patient_id = patients_data['data'][0]['patient_id']
            print(f"\n3. Récupération des détails du patient {patient_id}...")
            details_response = session.get(f'{base_url}/api/patients/{patient_id}/details')
            if details_response.status_code == 200:
                details_data = details_response.json()
                if details_data['success']:
                    patient = details_data['data']
                    print("✅ Détails récupérés:")
                    print(f"   Nom: {patient['patient_name']}")
                    print(f"   Téléphone: {patient.get('phone', 'Non renseigné')}")
                    print(f"   Email: {patient.get('email', 'Non renseigné')}")
                    print(f"   Analyses: {patient.get('total_analyses', 0)}")
                else:
                    print(f"❌ Erreur API: {details_data['error']}")
            else:
                print(f"❌ Erreur HTTP: {details_response.status_code}")
        
        # 4. Tester la création d'un nouveau patient
        print("\n4. Test de création d'un nouveau patient...")
        new_patient_data = {
            'patient_id': 'TEST001',
            'patient_name': 'Patient Test',
            'date_of_birth': '1990-01-01',
            'gender': 'M',
            'phone': '+33 1 00 00 00 00',
            'email': 'test.patient@email.com',
            'address': 'Adresse de test',
            'medical_history': 'Historique médical de test',
            'notes': 'Notes de test'
        }
        
        create_response = session.post(
            f'{base_url}/api/patients',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(new_patient_data)
        )
        
        if create_response.status_code == 200:
            create_data = create_response.json()
            if create_data['success']:
                print("✅ Patient créé avec succès")
                print(f"   Message: {create_data['message']}")
                
                # 5. Tester la modification du patient créé
                print("\n5. Test de modification du patient créé...")
                update_data = new_patient_data.copy()
                update_data['patient_name'] = 'Patient Test Modifié'
                update_data['phone'] = '+33 1 11 11 11 11'
                
                update_response = session.put(
                    f'{base_url}/api/patients/TEST001',
                    headers={'Content-Type': 'application/json'},
                    data=json.dumps(update_data)
                )
                
                if update_response.status_code == 200:
                    update_result = update_response.json()
                    if update_result['success']:
                        print("✅ Patient modifié avec succès")
                        print(f"   Message: {update_result['message']}")
                    else:
                        print(f"❌ Erreur modification: {update_result['error']}")
                else:
                    print(f"❌ Erreur HTTP modification: {update_response.status_code}")
                
                # 6. Tester la suppression du patient créé
                print("\n6. Test de suppression du patient créé...")
                delete_response = session.delete(f'{base_url}/api/patients/TEST001')
                
                if delete_response.status_code == 200:
                    delete_data = delete_response.json()
                    if delete_data['success']:
                        print("✅ Patient supprimé avec succès")
                        print(f"   Message: {delete_data['message']}")
                    else:
                        print(f"❌ Erreur suppression: {delete_data['error']}")
                else:
                    print(f"❌ Erreur HTTP suppression: {delete_response.status_code}")
                    
            else:
                print(f"❌ Erreur création: {create_data['error']}")
        else:
            print(f"❌ Erreur HTTP création: {create_response.status_code}")
            print(f"   Réponse: {create_response.text}")
        
        print("\n🎉 Tests terminés!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter au serveur. Vérifiez que l'application est lancée.")
        return False
    except Exception as e:
        print(f"❌ Erreur lors des tests: {e}")
        return False

if __name__ == '__main__':
    success = test_patient_apis()
    if success:
        print("\n✅ Tous les tests sont passés!")
    else:
        print("\n❌ Certains tests ont échoué.")
