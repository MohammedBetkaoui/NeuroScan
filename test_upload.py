#!/usr/bin/env python3
"""
Script de test pour vérifier le fonctionnement de l'upload d'images
"""

import requests
import os
from pathlib import Path

def test_upload():
    """Tester l'upload d'une image"""
    
    # URL de l'application
    url = "http://localhost:5000/upload"
    
    # Vérifier si les images de test existent
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print("❌ Dossier test_images non trouvé. Créez d'abord les images de test avec:")
        print("   python3 create_test_image.py")
        return False
    
    # Tester avec l'image normale
    normal_image = test_images_dir / "brain_normal.jpg"
    tumor_image = test_images_dir / "brain_with_tumor.jpg"
    
    if not normal_image.exists() or not tumor_image.exists():
        print("❌ Images de test non trouvées. Créez-les avec:")
        print("   python3 create_test_image.py")
        return False
    
    print("🧪 Test de l'upload d'images...")
    print("=" * 50)
    
    # Test 1: Image normale
    print("\n📸 Test 1: Upload d'une image normale")
    try:
        with open(normal_image, 'rb') as f:
            files = {'file': ('brain_normal.jpg', f, 'image/jpeg')}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Upload réussi!")
                print(f"   Prédiction: {data.get('prediction')}")
                print(f"   Confiance: {data.get('confidence', 0)*100:.1f}%")
                print(f"   Tumeur détectée: {'Oui' if data.get('is_tumor') else 'Non'}")
            else:
                print(f"❌ Erreur dans la réponse: {data.get('error')}")
                return False
        else:
            print(f"❌ Erreur HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de connexion: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
    
    # Test 2: Image avec tumeur
    print("\n📸 Test 2: Upload d'une image avec tumeur")
    try:
        with open(tumor_image, 'rb') as f:
            files = {'file': ('brain_with_tumor.jpg', f, 'image/jpeg')}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Upload réussi!")
                print(f"   Prédiction: {data.get('prediction')}")
                print(f"   Confiance: {data.get('confidence', 0)*100:.1f}%")
                print(f"   Tumeur détectée: {'Oui' if data.get('is_tumor') else 'Non'}")
                
                # Afficher les probabilités
                probs = data.get('probabilities', {})
                print("   Probabilités détaillées:")
                for tumor_type, prob in probs.items():
                    print(f"     - {tumor_type}: {prob*100:.1f}%")
                    
            else:
                print(f"❌ Erreur dans la réponse: {data.get('error')}")
                return False
        else:
            print(f"❌ Erreur HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de connexion: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
    
    print("\n🎉 Tous les tests sont passés avec succès!")
    return True

def test_health():
    """Tester l'endpoint de santé"""
    
    print("\n🏥 Test de l'endpoint de santé...")
    try:
        response = requests.get("http://localhost:5000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Application en bonne santé")
            print(f"   Status: {data.get('status')}")
            print(f"   Modèle chargé: {data.get('model_loaded')}")
            print(f"   Device: {data.get('device')}")
            return True
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Fonction principale"""
    
    print("🧠 NeuroScan - Tests de fonctionnement")
    print("=" * 50)
    
    # Test de santé
    if not test_health():
        print("\n❌ L'application ne répond pas. Vérifiez qu'elle est démarrée:")
        print("   source venv/bin/activate && python3 app.py")
        return
    
    # Test d'upload
    if test_upload():
        print("\n✅ Tous les tests sont réussis!")
        print("\n🌐 L'application est accessible sur: http://localhost:5000")
        print("📱 Vous pouvez maintenant tester l'interface web")
    else:
        print("\n❌ Certains tests ont échoué")

if __name__ == "__main__":
    main()
