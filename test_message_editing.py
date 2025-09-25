#!/usr/bin/env python3
"""
Script de test pour la fonctionnalité d'édition de messages et branchement
"""

import requests
import json
import sys
import sqlite3
from datetime import datetime

# Configuration
BASE_URL = "http://127.0.0.1:5001"
DATABASE_PATH = 'neuroscan_analytics.db'

class MessageEditingTester:
    def __init__(self):
        self.session = requests.Session()
        self.doctor_id = None
        self.conversation_id = None
        self.user_message_id = None
        
    def test_login(self):
        """Test de connexion avec le médecin de test"""
        print("🔐 Test de connexion...")
        
        try:
            response = self.session.post(f"{BASE_URL}/login", data={
                'email': 'test@neuroscan.com',
                'password': 'test123'
            }, allow_redirects=False)
            
            if response.status_code == 302:  # Redirection après connexion
                print("✅ Connexion réussie")
                return True
            else:
                print(f"❌ Échec connexion: {response.status_code}")
                if response.text:
                    print(f"   Réponse: {response.text[:100]}...")
                return False
                
        except Exception as e:
            print(f"❌ Erreur connexion: {e}")
            return False
    
    def test_create_conversation(self):
        """Créer une conversation de test"""
        print("💬 Test de création de conversation...")
        
        try:
            response = self.session.post(f"{BASE_URL}/api/chat/conversations", json={
                'title': f'Test Édition Messages {datetime.now().strftime("%H:%M:%S")}'
            })
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get('success'):
                        self.conversation_id = data['conversation_id']
                        print(f"✅ Conversation créée avec ID: {self.conversation_id}")
                        return True
                except json.JSONDecodeError:
                    print(f"❌ Réponse non-JSON: {response.text[:100]}...")
                    return False
            
            print(f"❌ Échec création conversation: {response.status_code}")
            if response.text:
                print(f"   Réponse: {response.text[:100]}...")
            return False
            
        except Exception as e:
            print(f"❌ Erreur création conversation: {e}")
            return False
    
    def test_send_initial_message(self):
        """Envoyer un message initial"""
        print("📤 Test d'envoi de message initial...")
        
        try:
            message_content = "Pouvez-vous m'expliquer les différents types de tumeurs cérébrales ?"
            
            response = self.session.post(f"{BASE_URL}/api/chat/send", json={
                'conversation_id': self.conversation_id,
                'message': message_content
            })
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get('success'):
                        self.user_message_id = data['user_message_id']
                        print(f"✅ Message envoyé. ID utilisateur: {self.user_message_id}")
                        print(f"🤖 Réponse IA reçue: {data['response'][:100]}...")
                        return True
                except json.JSONDecodeError:
                    print(f"❌ Réponse non-JSON: {response.text[:100]}...")
                    return False
            
            print(f"❌ Échec envoi message: {response.status_code}")
            if response.text:
                print(f"   Réponse: {response.text[:100]}...")
            return False
            
        except Exception as e:
            print(f"❌ Erreur envoi message: {e}")
            return False
    
    def test_edit_message(self):
        """Tester l'édition du message"""
        print("✏️  Test d'édition de message...")
        
        try:
            new_content = "Pouvez-vous m'expliquer les différents types de tumeurs cérébrales et leurs traitements ?"
            
            response = self.session.post(f"{BASE_URL}/api/chat/messages/{self.user_message_id}/edit", json={
                'content': new_content
            })
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ Message édité avec succès!")
                    print(f"   - Message original ID: {data['original_message_id']}")
                    print(f"   - Nouveau message ID: {data['new_message_id']}")
                    print(f"   - Niveau de branche: {data.get('branch_level', 'N/A')}")
                    
                    if data.get('assistant_response'):
                        print(f"🤖 Nouvelle réponse générée: {data['assistant_response'][:100]}...")
                    
                    return True
            
            print(f"❌ Échec édition: {response.status_code} - {response.text}")
            return False
            
        except Exception as e:
            print(f"❌ Erreur édition: {e}")
            return False
    
    def test_get_messages_with_branches(self):
        """Tester la récupération des messages avec branches"""
        print("🌳 Test de récupération des messages avec branches...")
        
        try:
            response = self.session.get(f"{BASE_URL}/api/chat/conversations/{self.conversation_id}/messages-with-branches")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    messages = data['messages']
                    print(f"✅ {len(messages)} message(s) récupéré(s)")
                    
                    for msg in messages:
                        print(f"   📝 {msg['role']}: {msg['content'][:50]}...")
                        if msg.get('branches'):
                            print(f"      🌿 {len(msg['branches'])} branche(s)")
                        if msg.get('is_edited'):
                            print(f"      ✏️  Édité {msg.get('edit_count', 1)} fois")
                    
                    return True
            
            print(f"❌ Échec récupération: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"❌ Erreur récupération: {e}")
            return False
    
    def test_get_message_branches(self):
        """Tester la récupération des branches d'un message"""
        print("🌲 Test de récupération des branches du message...")
        
        try:
            response = self.session.get(f"{BASE_URL}/api/chat/messages/{self.user_message_id}/branches")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    branches = data['branches']
                    print(f"✅ {len(branches)} branche(s) trouvée(s)")
                    
                    for i, branch in enumerate(branches):
                        print(f"   {i+1}. {'[Original]' if branch.get('is_original') else '[Branche]'} Niveau {branch['branch_level']}")
                        print(f"      📝 {branch['content'][:80]}...")
                    
                    return True
            
            print(f"❌ Échec récupération branches: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"❌ Erreur récupération branches: {e}")
            return False
    
    def test_regenerate_response(self):
        """Tester la régénération de réponse"""
        print("🔄 Test de régénération de réponse...")
        
        try:
            response = self.session.post(f"{BASE_URL}/api/chat/messages/{self.user_message_id}/regenerate")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ Nouvelle réponse générée!")
                    print(f"   - Nouvelle réponse ID: {data['message_id']}")
                    print(f"🤖 Réponse: {data['response'][:100]}...")
                    return True
            
            print(f"❌ Échec régénération: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"❌ Erreur régénération: {e}")
            return False
    
    def inspect_database(self):
        """Inspecter la base de données pour voir les structures créées"""
        print("🔍 Inspection de la base de données...")
        
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # Vérifier les messages de la conversation de test
            cursor.execute('''
                SELECT id, role, content, branch_level, branch_position, is_edited, 
                       parent_message_id, original_message_id
                FROM chat_messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
            ''', (self.conversation_id,))
            
            messages = cursor.fetchall()
            
            print(f"📊 Messages dans la base de données:")
            for msg in messages:
                msg_id, role, content, branch_level, branch_pos, is_edited, parent_id, original_id = msg
                edited_mark = "✏️ " if is_edited else ""
                print(f"   {edited_mark}ID:{msg_id} [{role}] Niveau:{branch_level or 0} Pos:{branch_pos or 0}")
                print(f"      {content[:60]}...")
                if parent_id:
                    print(f"      Parent: {parent_id}")
                if original_id:
                    print(f"      Original: {original_id}")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Erreur inspection DB: {e}")
            return False
    
    def run_all_tests(self):
        """Exécuter tous les tests"""
        print("🚀 Démarrage des tests d'édition de messages")
        print("=" * 50)
        
        tests = [
            self.test_login,
            self.test_create_conversation,
            self.test_send_initial_message,
            self.test_edit_message,
            self.test_get_messages_with_branches,
            self.test_get_message_branches,
            self.test_regenerate_response,
            self.inspect_database
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                print()
            except Exception as e:
                print(f"❌ Erreur dans {test.__name__}: {e}")
                results.append(False)
                print()
        
        # Résumé
        success_count = sum(results)
        total_count = len(results)
        
        print("=" * 50)
        print(f"📈 Résultats: {success_count}/{total_count} tests réussis")
        
        if success_count == total_count:
            print("🎉 Tous les tests sont passés!")
            print("✨ La fonctionnalité d'édition de messages fonctionne correctement")
        else:
            print("⚠️  Certains tests ont échoué")
        
        return success_count == total_count

def main():
    """Fonction principale"""
    tester = MessageEditingTester()
    
    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrompus par l'utilisateur")
        sys.exit(1)

if __name__ == "__main__":
    main()
