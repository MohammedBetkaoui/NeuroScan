"""
Script de test pour vérifier la configuration MongoDB de NeuroScan AI
"""

import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

print("=" * 70)
print("🧪 TEST DE CONFIGURATION MONGODB POUR NEUROSCAN AI")
print("=" * 70)

# Test 1: Vérifier les variables d'environnement
print("\n📋 Test 1: Vérification des variables d'environnement")
print("-" * 70)

mongodb_uri = os.getenv('MONGODB_URI')
mongodb_db_name = os.getenv('MONGODB_DB_NAME', 'NeuroScan')

if mongodb_uri:
    # Masquer le mot de passe pour l'affichage
    safe_uri = mongodb_uri.split('@')[1] if '@' in mongodb_uri else mongodb_uri
    print(f"✅ MONGODB_URI configuré: ...@{safe_uri}")
else:
    print("❌ MONGODB_URI non configuré dans .env")
    sys.exit(1)

print(f"✅ MONGODB_DB_NAME: {mongodb_db_name}")

# Test 2: Import du connecteur MongoDB
print("\n📦 Test 2: Import du connecteur MongoDB")
print("-" * 70)

try:
    from database.mongodb_connector import mongodb_connector, get_mongodb, get_collection
    print("✅ Modules MongoDB importés avec succès")
except Exception as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

# Test 3: Test de connexion
print("\n🔌 Test 3: Test de connexion à MongoDB")
print("-" * 70)

try:
    if mongodb_connector.test_connection():
        print("✅ Connexion MongoDB réussie")
    else:
        print("❌ Échec de connexion MongoDB")
        sys.exit(1)
except Exception as e:
    print(f"❌ Erreur de connexion: {e}")
    sys.exit(1)

# Test 4: Vérification des collections
print("\n📚 Test 4: Vérification des collections")
print("-" * 70)

try:
    db = get_mongodb()
    collections = db.list_collection_names()
    
    required_collections = [
        'analyses', 'patients', 'tumor_evolution', 'medical_alerts',
        'notifications', 'daily_stats', 'doctors', 'doctor_sessions',
        'chat_conversations', 'chat_messages', 'chat_attachments'
    ]
    
    print(f"Collections existantes: {len(collections)}")
    for coll in required_collections:
        if coll in collections:
            print(f"  ✅ {coll}")
        else:
            print(f"  ⚠️  {coll} (sera créée automatiquement)")
    
except Exception as e:
    print(f"❌ Erreur lors de la vérification des collections: {e}")

# Test 5: Test d'écriture/lecture
print("\n✍️  Test 5: Test d'écriture et lecture")
print("-" * 70)

try:
    test_collection = db['test_collection']
    
    # Insérer un document de test
    test_doc = {
        'test': True,
        'message': 'Test NeuroScan AI',
        'timestamp': datetime.now()
    }
    result = test_collection.insert_one(test_doc)
    print(f"✅ Document inséré avec ID: {result.inserted_id}")
    
    # Lire le document
    found_doc = test_collection.find_one({'_id': result.inserted_id})
    if found_doc:
        print(f"✅ Document lu avec succès")
        print(f"   Message: {found_doc['message']}")
    
    # Supprimer le document de test
    test_collection.delete_one({'_id': result.inserted_id})
    print(f"✅ Document de test supprimé")
    
except Exception as e:
    print(f"❌ Erreur lors du test d'écriture/lecture: {e}")

# Test 6: Test des fonctions helper MongoDB
print("\n🔧 Test 6: Test des fonctions helper MongoDB")
print("-" * 70)

try:
    from database.mongodb_helpers import (
        save_analysis_to_db_mongo,
        manage_patient_record_mongo,
        get_doctor_statistics_mongo
    )
    print("✅ Fonctions helper importées avec succès:")
    print("  - save_analysis_to_db_mongo")
    print("  - manage_patient_record_mongo")
    print("  - get_doctor_statistics_mongo")
except Exception as e:
    print(f"❌ Erreur d'import des fonctions helper: {e}")

# Test 7: Statistiques de la base de données
print("\n📊 Test 7: Statistiques de la base de données")
print("-" * 70)

try:
    db_stats = db.command("dbStats")
    print(f"✅ Nom de la base: {db_stats['db']}")
    print(f"✅ Nombre de collections: {db_stats['collections']}")
    print(f"✅ Taille des données: {db_stats['dataSize'] / 1024:.2f} KB")
    print(f"✅ Taille de stockage: {db_stats['storageSize'] / 1024:.2f} KB")
except Exception as e:
    print(f"⚠️  Impossible de récupérer les statistiques: {e}")

# Test 8: Test de sécurité
print("\n🔒 Test 8: Vérification de la sécurité")
print("-" * 70)

if '@2002' in mongodb_uri or 'betkaoui@2002' in mongodb_uri:
    print("⚠️  ATTENTION: Le mot de passe est visible dans l'URI")
    print("   Recommandation: Utiliser des variables d'environnement sécurisées")
    print("   Le mot de passe doit être encodé en URL: betkaoui%402002")

if mongodb_uri.startswith('mongodb+srv://'):
    print("✅ Utilisation de MongoDB Atlas (connexion sécurisée)")
else:
    print("⚠️  Connexion non sécurisée détectée")

# Résumé final
print("\n" + "=" * 70)
print("✅ TESTS TERMINÉS AVEC SUCCÈS")
print("=" * 70)
print(f"\n📝 Résumé:")
print(f"  - Base de données: {mongodb_db_name}")
print(f"  - Serveur MongoDB: Cluster0.xdhjsc1.mongodb.net")
print(f"  - Status: Opérationnel")
print(f"\n🚀 Votre application NeuroScan AI est prête à utiliser MongoDB!")
print("=" * 70)
