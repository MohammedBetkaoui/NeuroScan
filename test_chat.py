#!/usr/bin/env python3
"""
Script de test pour diagnostiquer les problèmes du chat
"""

import sys
sys.path.insert(0, '/home/mohammed/Bureau/ai scan/venv/lib/python3.12/site-packages')

from app import app
import requests
import json

# Test 1: Vérifier que l'API Gemini est configurée
print("=" * 60)
print("🔍 Test de diagnostic du Chat NeuroScan")
print("=" * 60)

print("\n1️⃣ Test de la configuration API Gemini...")
import os
from dotenv import load_dotenv
load_dotenv()

gemini_key = os.getenv('GEMINI_API_KEY')
if gemini_key:
    print(f"   ✅ Clé API Gemini configurée: {gemini_key[:20]}...")
else:
    print("   ❌ Clé API Gemini manquante!")
    print("   💡 Vérifiez votre fichier .env")

# Test 2: Vérifier les routes API
print("\n2️⃣ Test des routes API du chat...")
with app.app_context():
    with app.test_client() as client:
        # Test création de conversation (nécessite authentification)
        print("   📝 Routes API disponibles:")
        for rule in app.url_map.iter_rules():
            if 'chat' in rule.rule:
                print(f"      - {rule.rule} [{', '.join(rule.methods - {'HEAD', 'OPTIONS'})}]")

# Test 3: Test d'appel API (simulé)
print("\n3️⃣ Test de la connectivité API...")
try:
    response = requests.get('http://127.0.0.1:5000/api/alerts', timeout=2)
    if response.status_code == 200:
        print("   ✅ Serveur Flask accessible")
    else:
        print(f"   ⚠️  Serveur répond avec status {response.status_code}")
except requests.exceptions.ConnectionRefused:
    print("   ❌ Serveur Flask non accessible")
    print("   💡 Lancez d'abord l'application avec: ./launch_neuroscan.sh")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

print("\n" + "=" * 60)
print("🎯 Diagnostic terminé")
print("=" * 60)
