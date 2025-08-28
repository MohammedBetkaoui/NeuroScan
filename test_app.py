#!/usr/bin/env python3
"""
Script de test pour l'application NeuroScan
"""

import sys
import sqlite3
from datetime import datetime

def test_database():
    """Tester la connexion à la base de données"""
    try:
        conn = sqlite3.connect('neuroscan_analytics.db')
        cursor = conn.cursor()
        
        # Tester les tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print("✅ Connexion à la base de données réussie")
        print(f"📊 Tables trouvées: {len(tables)}")
        for table in tables:
            print(f"   - {table}")
        
        # Tester les médecins
        cursor.execute("SELECT COUNT(*) FROM doctors")
        doctor_count = cursor.fetchone()[0]
        print(f"👨‍⚕️ Nombre de médecins: {doctor_count}")
        
        # Tester les patients
        cursor.execute("SELECT COUNT(*) FROM patients")
        patient_count = cursor.fetchone()[0]
        print(f"🏥 Nombre de patients: {patient_count}")
        
        # Tester les analyses
        cursor.execute("SELECT COUNT(*) FROM analyses")
        analysis_count = cursor.fetchone()[0]
        print(f"🧠 Nombre d'analyses: {analysis_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur de base de données: {e}")
        return False

def test_imports():
    """Tester les imports Python"""
    try:
        import flask
        print(f"✅ Flask version: {flask.__version__}")
        
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        import PIL
        print(f"✅ Pillow version: {PIL.__version__}")
        
        import numpy
        print(f"✅ NumPy version: {numpy.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def test_model():
    """Tester le modèle de deep learning"""
    try:
        import os
        model_path = 'best_brain_tumor_model.pth'
        
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"✅ Modèle trouvé: {model_path} ({size:.1f} MB)")
            return True
        else:
            print(f"❌ Modèle non trouvé: {model_path}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return False

def test_flask_app():
    """Tester l'importation de l'application Flask"""
    try:
        import app
        print("✅ Application Flask importée avec succès")
        
        # Tester quelques routes
        with app.app.test_client() as client:
            # Test de la page d'accueil
            response = client.get('/')
            print(f"✅ Route '/' : Status {response.status_code}")
            
            # Test de la page de login
            response = client.get('/login')
            print(f"✅ Route '/login' : Status {response.status_code}")
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur Flask: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 TESTS DE L'APPLICATION NEUROSCAN")
    print("=" * 40)
    
    tests = [
        ("Base de données", test_database),
        ("Imports Python", test_imports),
        ("Modèle IA", test_model),
        ("Application Flask", test_flask_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Test: {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 40)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Tous les tests sont passés ! L'application est prête.")
        print("\n🚀 Pour lancer l'application:")
        print("   python3 app.py")
        print("\n🌐 Puis ouvrez: http://localhost:5000")
        print("📧 Login: test@neuroscan.com")
        print("🔑 Mot de passe: test123")
    else:
        print("\n⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
