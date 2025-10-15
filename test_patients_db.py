"""
Script de test pour vérifier les patients dans MongoDB
"""
import sys
from database.mongodb_connector import get_mongodb

def test_patients():
    """Tester la récupération des patients"""
    try:
        db = get_mongodb()
        
        print("=" * 60)
        print("TEST: Vérification des patients dans MongoDB")
        print("=" * 60)
        
        # Compter le nombre total de patients
        total_patients = db.patients.count_documents({})
        print(f"\n✅ Nombre total de patients: {total_patients}")
        
        if total_patients == 0:
            print("❌ Aucun patient trouvé dans la base de données!")
            return
        
        # Récupérer tous les patients
        print("\n📋 Liste des patients:")
        print("-" * 60)
        
        patients = db.patients.find({})
        for i, patient in enumerate(patients, 1):
            print(f"\nPatient #{i}:")
            print(f"  - ID: {patient.get('patient_id')}")
            print(f"  - Nom: {patient.get('patient_name')}")
            print(f"  - Doctor ID: {patient.get('doctor_id')}")
            print(f"  - Date de naissance: {patient.get('date_of_birth')}")
            print(f"  - Genre: {patient.get('gender')}")
            print(f"  - Total analyses: {patient.get('total_analyses', 0)}")
        
        # Grouper par doctor_id
        print("\n" + "=" * 60)
        print("Patients groupés par médecin:")
        print("=" * 60)
        
        pipeline = [
            {'$group': {
                '_id': '$doctor_id',
                'count': {'$sum': 1},
                'patients': {'$push': '$patient_name'}
            }},
            {'$sort': {'count': -1}}
        ]
        
        doctors_patients = list(db.patients.aggregate(pipeline))
        for doc in doctors_patients:
            doctor_id = doc['_id']
            count = doc['count']
            patients_list = doc['patients']
            print(f"\nDoctor ID: {doctor_id}")
            print(f"  Nombre de patients: {count}")
            print(f"  Patients: {', '.join(patients_list[:5])}")
            if count > 5:
                print(f"  ... et {count - 5} autres")
        
        print("\n" + "=" * 60)
        print("✅ Test terminé avec succès")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_patients()
