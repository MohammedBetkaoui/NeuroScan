#!/usr/bin/env python3
"""
Script pour mettre à jour la table chat_messages pour supporter l'édition et le branchement
"""

import sqlite3
import sys
from datetime import datetime

DATABASE_PATH = 'neuroscan_analytics.db'

def update_chat_messages_table():
    """Ajouter les champs nécessaires pour l'édition et le branchement"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        print("🔄 Mise à jour de la table chat_messages pour supporter l'édition...")
        
        # Vérifier les colonnes existantes
        cursor.execute("PRAGMA table_info(chat_messages)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        print("📋 Colonnes existantes:")
        for col in existing_columns:
            print(f"  - {col}")
        
        # Liste des nouvelles colonnes à ajouter
        new_fields = [
            ('parent_message_id', 'INTEGER'), # ID du message parent (pour les branches)
            ('original_message_id', 'INTEGER'), # ID du message original (référence)
            ('is_edited', 'BOOLEAN DEFAULT 0'), # Indique si le message a été édité
            ('edit_count', 'INTEGER DEFAULT 0'), # Nombre d'éditions
            ('last_edited_at', 'DATETIME'), # Dernière date d'édition
            ('branch_level', 'INTEGER DEFAULT 0'), # Niveau de branche (0 = principal)
            ('branch_position', 'INTEGER DEFAULT 0'), # Position dans la branche
        ]
        
        # Ajouter les nouveaux champs s'ils n'existent pas
        added_fields = []
        for field_name, field_type in new_fields:
            if field_name not in existing_columns:
                try:
                    cursor.execute(f'ALTER TABLE chat_messages ADD COLUMN {field_name} {field_type}')
                    added_fields.append(field_name)
                    print(f"  ✅ Ajouté: {field_name}")
                except sqlite3.Error as e:
                    print(f"  ❌ Erreur ajout {field_name}: {e}")
            else:
                print(f"  ⚠️  {field_name} existe déjà")
        
        if added_fields:
            # Créer les index pour améliorer les performances
            indexes = [
                ('idx_chat_messages_parent', 'CREATE INDEX IF NOT EXISTS idx_chat_messages_parent ON chat_messages(parent_message_id)'),
                ('idx_chat_messages_original', 'CREATE INDEX IF NOT EXISTS idx_chat_messages_original ON chat_messages(original_message_id)'),
                ('idx_chat_messages_conversation_branch', 'CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_branch ON chat_messages(conversation_id, branch_level, branch_position)'),
            ]
            
            print("\n📊 Création des index de performance...")
            for index_name, index_query in indexes:
                try:
                    cursor.execute(index_query)
                    print(f"  ✅ Index {index_name} créé")
                except sqlite3.Error as e:
                    print(f"  ❌ Erreur index {index_name}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ Mise à jour terminée! {len(added_fields)} champ(s) ajouté(s)")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour: {e}")
        return False

def verify_update():
    """Vérifier que la mise à jour s'est bien passée"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        print("\n🔍 Vérification de la structure mise à jour:")
        cursor.execute("PRAGMA table_info(chat_messages)")
        columns = cursor.fetchall()
        
        expected_fields = [
            'parent_message_id', 'original_message_id', 'is_edited', 
            'edit_count', 'last_edited_at', 'branch_level', 'branch_position'
        ]
        
        for field in expected_fields:
            found = any(col[1] == field for col in columns)
            status = "✅" if found else "❌"
            print(f"  {status} {field}")
        
        # Compter les messages existants
        cursor.execute("SELECT COUNT(*) FROM chat_messages")
        message_count = cursor.fetchone()[0]
        print(f"\n📊 Nombre de messages existants: {message_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur vérification: {e}")
        return False

def main():
    """Fonction principale"""
    print("🚀 Mise à jour de la table chat_messages pour l'édition de messages")
    print("=" * 65)
    
    if update_chat_messages_table():
        verify_update()
        print("\n🎉 Mise à jour réussie!")
        print("ℹ️  La table chat_messages supporte maintenant l'édition et le branchement")
    else:
        print("\n💥 Échec de la mise à jour")
        sys.exit(1)

if __name__ == "__main__":
    main()
