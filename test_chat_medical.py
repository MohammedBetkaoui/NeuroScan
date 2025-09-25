#!/usr/bin/env python3
"""
Script de test complet pour le Chat Médical NeuroScan
Ce script teste toutes les fonctionnalités du chat médical avec Gemini
"""

import requests
import json
import sys
import sqlite3
from datetime import datetime

# Configuration
BASE_URL = "http://127.0.0.1:5000"
DATABASE_PATH = 'neuroscan_analytics.db'

class ChatMedicalTester:
    def __init__(self):
        self.session = requests.Session()
        self.doctor_id = None
        self.conversation_id = None
        
    def test_database_tables(self):
        """Tester la présence des tables de chat dans la base de données"""
        print("🔍 Test des tables de base de données...")
        
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # Vérifier les tables chat
            tables_to_check = [
                'chat_conversations',
                'chat_messages', 
                'chat_attachments'
            ]
            
            for table in tables_to_check:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if cursor.fetchone():
                    print(f"✅ Table {table} existe")
                else:
                    print(f"❌ Table {table} manquante")
                    
            conn.close()
            print("✅ Test des tables terminé\n")
            
        except Exception as e:
            print(f"❌ Erreur test database: {e}\n")
    
    def login_as_test_doctor(self):
        """Se connecter avec le médecin de test"""
        print("🔐 Connexion en tant que médecin de test...")
        
        try:
            # D'abord obtenir la page de login
            response = self.session.get(f"{BASE_URL}/login")
            if response.status_code != 200:
                print(f"❌ Impossible d'accéder à la page de login: {response.status_code}")
                return False
            
            # Se connecter avec le médecin de test
            login_data = {
                'email': 'dr.test@neuroscan.fr',
                'password': 'test123'
            }
            
            response = self.session.post(f"{BASE_URL}/login", data=login_data)
            
            if response.status_code == 302 or "dashboard" in response.url:
                print("✅ Connexion réussie")
                return True
            else:
                print(f"❌ Échec de connexion: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de la connexion: {e}")
            return False
    
    def test_chat_page_access(self):
        """Tester l'accès à la page de chat"""
        print("📱 Test d'accès à la page de chat...")
        
        try:
            response = self.session.get(f"{BASE_URL}/chat")
            
            if response.status_code == 200:
                print("✅ Page de chat accessible")
                
                # Vérifier que la page contient les éléments clés
                content = response.text
                key_elements = [
                    "Chat Médical",
                    "Nouvelle consultation",
                    "Assistant IA spécialisé",
                    "messagesContainer",
                    "messageForm"
                ]
                
                for element in key_elements:
                    if element in content:
                        print(f"✅ Élément '{element}' présent")
                    else:
                        print(f"❌ Élément '{element}' manquant")
                        
                return True
            else:
                print(f"❌ Impossible d'accéder à la page de chat: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur accès page chat: {e}")
            return False
    
    def test_create_conversation(self):
        """Tester la création d'une nouvelle conversation"""
        print("💬 Test de création de conversation...")
        
        try:
            conversation_data = {
                'title': f'Test Conversation {datetime.now().strftime("%H:%M:%S")}'
            }
            
            response = self.session.post(
                f"{BASE_URL}/api/chat/conversations",
                json=conversation_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('conversation_id'):
                    self.conversation_id = data['conversation_id']
                    print(f"✅ Conversation créée avec ID: {self.conversation_id}")
                    return True
                else:
                    print(f"❌ Échec création conversation: {data}")
                    return False
            else:
                print(f"❌ Erreur création conversation: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur création conversation: {e}")
            return False
    
    def test_send_message(self):
        """Tester l'envoi d'un message et la réponse Gemini"""
        print("📤 Test d'envoi de message...")
        
        if not self.conversation_id:
            print("❌ Pas de conversation active")
            return False
            
        try:
            message_data = {
                'conversation_id': self.conversation_id,
                'message': 'Quels sont les signes radiologiques d\'un gliome de bas grade à l\'IRM ?'
            }
            
            response = self.session.post(
                f"{BASE_URL}/api/chat/send",
                json=message_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') or data.get('response'):
                    print("✅ Message envoyé avec succès")
                    print(f"🤖 Réponse IA (extrait): {data.get('response', '')[:150]}...")
                    
                    if data.get('confidence_score'):
                        print(f"📊 Score de confiance: {data['confidence_score']:.2f}")
                        
                    if data.get('is_medical'):
                        print("🏥 Question identifiée comme médicale")
                        
                    return True
                else:
                    print(f"❌ Réponse invalide: {data}")
                    return False
            else:
                print(f"❌ Erreur envoi message: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur envoi message: {e}")
            return False
    
    def test_get_conversations(self):
        """Tester la récupération des conversations"""
        print("📋 Test de récupération des conversations...")
        
        try:
            response = self.session.get(f"{BASE_URL}/api/chat/conversations")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and 'conversations' in data:
                    conversations = data['conversations']
                    print(f"✅ {len(conversations)} conversation(s) récupérée(s)")
                    
                    if conversations:
                        conv = conversations[0]
                        print(f"📝 Première conversation: '{conv.get('title', 'Sans titre')}'")
                        print(f"📅 Créée le: {conv.get('created_at', 'Date inconnue')}")
                        
                    return True
                else:
                    print(f"❌ Format de réponse invalide: {data}")
                    return False
            else:
                print(f"❌ Erreur récupération conversations: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur récupération conversations: {e}")
            return False
    
    def test_get_messages(self):
        """Tester la récupération des messages d'une conversation"""
        print("📨 Test de récupération des messages...")
        
        if not self.conversation_id:
            print("❌ Pas de conversation active")
            return False
            
        try:
            response = self.session.get(
                f"{BASE_URL}/api/chat/conversations/{self.conversation_id}/messages"
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and 'messages' in data:
                    messages = data['messages']
                    print(f"✅ {len(messages)} message(s) récupéré(s)")
                    
                    for msg in messages:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        print(f"💬 {role}: {content[:50]}...")
                        
                    return True
                else:
                    print(f"❌ Format de réponse invalide: {data}")
                    return False
            else:
                print(f"❌ Erreur récupération messages: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur récupération messages: {e}")
            return False
    
    def test_patients_list(self):
        """Tester la récupération de la liste des patients"""
        print("👥 Test de récupération des patients...")
        
        try:
            response = self.session.get(f"{BASE_URL}/api/patients/list")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and 'patients' in data:
                    patients = data['patients']
                    print(f"✅ {len(patients)} patient(s) trouvé(s)")
                    
                    if patients:
                        patient = patients[0]
                        print(f"👤 Premier patient: {patient.get('display_name', 'Nom inconnu')}")
                        
                    return True
                else:
                    print(f"❌ Format de réponse invalide: {data}")
                    return False
            else:
                print(f"❌ Erreur récupération patients: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur récupération patients: {e}")
            return False
    
    def test_non_medical_query(self):
        """Tester le filtrage des questions non-médicales"""
        print("🚫 Test de filtrage non-médical...")
        
        if not self.conversation_id:
            print("❌ Pas de conversation active")
            return False
            
        try:
            message_data = {
                'conversation_id': self.conversation_id,
                'message': 'Quelle est la capitale de la France ?'
            }
            
            response = self.session.post(
                f"{BASE_URL}/api/chat/send",
                json=message_data
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get('response', '').lower()
                
                # Vérifier que l'IA redirige vers le domaine médical
                medical_redirect_keywords = [
                    'médical', 'médecin', 'santé', 'domaine médical', 
                    'questions médicales', 'spécialisé en', 'consultation'
                ]
                
                if any(keyword in response_text for keyword in medical_redirect_keywords):
                    print("✅ Question non-médicale correctement filtrée")
                    print(f"🔄 Réponse de redirection: {data.get('response', '')[:100]}...")
                    return True
                else:
                    print("⚠️ La question non-médicale n'a pas été filtrée comme attendu")
                    print(f"📝 Réponse: {response_text[:100]}...")
                    return False
                    
            else:
                print(f"❌ Erreur test filtrage: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur test filtrage: {e}")
            return False
    
    def run_all_tests(self):
        """Exécuter tous les tests"""
        print("🚀 Démarrage des tests du Chat Médical NeuroScan")
        print("=" * 60)
        
        tests = [
            ("Base de données", self.test_database_tables),
            ("Connexion médecin", self.login_as_test_doctor),
            ("Accès page chat", self.test_chat_page_access),
            ("Création conversation", self.test_create_conversation),
            ("Envoi message", self.test_send_message),
            ("Récupération conversations", self.test_get_conversations),
            ("Récupération messages", self.test_get_messages),
            ("Liste patients", self.test_patients_list),
            ("Filtrage non-médical", self.test_non_medical_query),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🧪 Test: {test_name}")
            print("-" * 40)
            
            try:
                if test_func():
                    passed += 1
                    print(f"✅ {test_name}: RÉUSSI")
                else:
                    print(f"❌ {test_name}: ÉCHOUÉ")
            except Exception as e:
                print(f"💥 {test_name}: ERREUR - {e}")
        
        print("\n" + "=" * 60)
        print(f"📊 RÉSULTATS FINAUX: {passed}/{total} tests réussis")
        
        if passed == total:
            print("🎉 Tous les tests sont passés! Le Chat Médical fonctionne parfaitement.")
            return True
        else:
            print(f"⚠️ {total - passed} test(s) ont échoué. Vérifiez la configuration.")
            return False

def main():
    """Fonction principale"""
    print("🏥 Chat Médical NeuroScan - Suite de Tests")
    print("Développé pour valider toutes les fonctionnalités du chat médical")
    print()
    
    # Vérifier que le serveur est accessible
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"✅ Serveur accessible sur {BASE_URL}")
    except Exception as e:
        print(f"❌ Serveur inaccessible sur {BASE_URL}: {e}")
        print("Assurez-vous que l'application Flask est démarrée")
        sys.exit(1)
    
    # Exécuter les tests
    tester = ChatMedicalTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
