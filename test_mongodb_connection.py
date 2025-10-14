"""
Script de test pour vérifier la connexion MongoDB
"""

import sys
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

print("=" * 60)
print("🧪 Test de connexion MongoDB pour NeuroScan AI")
print("=" * 60)

# Afficher les informations de configuration
MONGODB_URI = os.getenv('MONGODB_URI', '')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'NeuroScan')

# Masquer le mot de passe dans l'affichage
masked_uri = MONGODB_URI
if '@' in MONGODB_URI:
    parts = MONGODB_URI.split('@')
    if '//' in parts[0]:
        protocol_and_creds = parts[0].split('//')
        if ':' in protocol_and_creds[1]:
            username = protocol_and_creds[1].split(':')[0]
            masked_uri = f"{protocol_and_creds[0]}//{username}:****@{parts[1]}"

print(f"\n📋 Configuration:")
print(f"   URI MongoDB: {masked_uri}")
print(f"   Base de données: {MONGODB_DB_NAME}")

try:
    print("\n🔌 Importation du connecteur MongoDB...")
    from database.mongodb_connector import mongodb_connector, get_mongodb
    
    print("✅ Connecteur importé avec succès")
    
    print("\n🔗 Test de connexion au cluster MongoDB Atlas...")
    if mongodb_connector.test_connection():
        print("✅ Connexion établie avec succès!")
        
        # Récupérer la base de données
        db = get_mongodb()
        
        if db is not None:
            print(f"\n📊 Base de données active: {db.name}")
            
            # Lister les collections
            print("\n📁 Collections existantes:")
            collections = db.list_collection_names()
            if collections:
                for col in collections:
                    count = db[col].count_documents({})
                    print(f"   - {col}: {count} documents")
            else:
                print("   Aucune collection (base de données vide - normal pour une nouvelle installation)")
            
            # Test d'écriture
            print("\n✍️  Test d'écriture...")
            test_collection = db.test_collection
            result = test_collection.insert_one({
                'test': True,
                'message': 'Test de connexion NeuroScan AI',
                'timestamp': '2025-10-10'
            })
            print(f"✅ Document de test inséré avec ID: {result.inserted_id}")
            
            # Test de lecture
            print("\n📖 Test de lecture...")
            doc = test_collection.find_one({'test': True})
            if doc:
                print(f"✅ Document lu: {doc.get('message')}")
            
            # Nettoyage
            print("\n🧹 Nettoyage...")
            test_collection.delete_one({'test': True})
            print("✅ Document de test supprimé")
            
            print("\n" + "=" * 60)
            print("✅ TOUS LES TESTS RÉUSSIS!")
            print("=" * 60)
            print("\n💡 Votre configuration MongoDB est prête pour NeuroScan AI")
            print("   Vous pouvez maintenant lancer: python3 app_web.py")
            
        else:
            print("❌ Impossible d'accéder à la base de données")
            sys.exit(1)
    else:
        print("❌ Échec de la connexion au cluster MongoDB")
        print("\n🔍 Vérifications à effectuer:")
        print("   1. Vérifiez votre connexion Internet")
        print("   2. Vérifiez que l'IP de votre machine est autorisée dans MongoDB Atlas")
        print("   3. Vérifiez le mot de passe dans le fichier .env")
        print("   4. Vérifiez que le cluster existe et est actif")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    print("\n💡 Solution: Installez les dépendances manquantes")
    print("   pip install pymongo python-dotenv")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Erreur lors du test: {e}")
    print(f"\n🔍 Type d'erreur: {type(e).__name__}")
    
    # Suggestions selon le type d'erreur
    if "DNS" in str(e) or "resolve" in str(e).lower():
        print("\n💡 Problème de résolution DNS détecté")
        print("   Solutions possibles:")
        print("   1. Vérifiez votre connexion Internet")
        print("   2. Essayez de changer de DNS (ex: 8.8.8.8)")
        print("   3. Vérifiez que le nom du cluster est correct")
    elif "authentication" in str(e).lower() or "auth" in str(e).lower():
        print("\n💡 Problème d'authentification détecté")
        print("   Solutions possibles:")
        print("   1. Vérifiez le nom d'utilisateur et le mot de passe dans .env")
        print("   2. Vérifiez que l'utilisateur existe dans MongoDB Atlas")
        print("   3. Caractères spéciaux dans le mot de passe? Encodez-les en URL")
    elif "timeout" in str(e).lower():
        print("\n💡 Problème de timeout détecté")
        print("   Solutions possibles:")
        print("   1. Vérifiez que votre IP est autorisée dans MongoDB Atlas")
        print("   2. Vérifiez votre pare-feu")
        print("   3. Essayez d'augmenter le timeout dans mongodb_connector.py")
    
    sys.exit(1)
