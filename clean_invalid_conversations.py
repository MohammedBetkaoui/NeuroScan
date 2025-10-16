#!/usr/bin/env python3
"""
Script pour nettoyer les conversations avec des IDs invalides (anciennes données SQLite)
"""

from database.mongodb_connector import get_mongodb
from bson import ObjectId
from datetime import datetime

def clean_invalid_conversations():
    """Supprimer ou corriger les conversations avec des IDs invalides"""
    try:
        db = get_mongodb()
        
        print("🔍 Vérification des conversations dans la base de données...")
        
        # Récupérer toutes les conversations
        all_conversations = list(db.chat_conversations.find())
        print(f"   Nombre total de conversations: {len(all_conversations)}")
        
        # Compter les conversations valides
        valid_count = 0
        for conv in all_conversations:
            conv_id = conv.get('_id')
            if isinstance(conv_id, ObjectId):
                valid_count += 1
        
        print(f"   Conversations avec ObjectId valide: {valid_count}")
        print(f"   Conversations avec ID invalide: {len(all_conversations) - valid_count}")
        
        # Vérifier les messages orphelins (avec conversation_id qui n'est pas un ObjectId valide)
        print("\n🔍 Vérification des messages...")
        all_messages = list(db.chat_messages.find())
        print(f"   Nombre total de messages: {len(all_messages)}")
        
        # Grouper les messages par conversation_id
        conversation_ids_in_messages = set()
        for msg in all_messages:
            conv_id = msg.get('conversation_id')
            if conv_id:
                conversation_ids_in_messages.add(str(conv_id))
        
        print(f"   IDs de conversation uniques dans les messages: {len(conversation_ids_in_messages)}")
        
        # Vérifier les IDs invalides
        invalid_conv_ids = []
        for conv_id in conversation_ids_in_messages:
            try:
                ObjectId(conv_id)
            except:
                invalid_conv_ids.append(conv_id)
        
        if invalid_conv_ids:
            print(f"\n⚠️  {len(invalid_conv_ids)} conversation(s) avec ID invalide trouvée(s):")
            for inv_id in invalid_conv_ids:
                # Compter les messages pour cet ID
                msg_count = db.chat_messages.count_documents({'conversation_id': inv_id})
                print(f"      - ID: {inv_id} ({msg_count} message(s))")
            
            # Demander confirmation pour supprimer
            print("\n❓ Voulez-vous supprimer ces conversations invalides et leurs messages ?")
            response = input("   Tapez 'oui' pour confirmer: ").strip().lower()
            
            if response == 'oui':
                print("\n🗑️  Suppression en cours...")
                total_deleted_messages = 0
                
                for inv_id in invalid_conv_ids:
                    # Supprimer les messages
                    result = db.chat_messages.delete_many({'conversation_id': inv_id})
                    deleted = result.deleted_count
                    total_deleted_messages += deleted
                    print(f"      - Supprimé {deleted} message(s) pour la conversation {inv_id}")
                    
                    # Supprimer les attachments
                    db.chat_attachments.delete_many({'conversation_id': inv_id})
                
                print(f"\n✅ Nettoyage terminé:")
                print(f"   - {total_deleted_messages} message(s) supprimé(s)")
                print(f"   - {len(invalid_conv_ids)} conversation(s) invalide(s) nettoyée(s)")
            else:
                print("\n❌ Nettoyage annulé.")
        else:
            print("\n✅ Aucune conversation invalide trouvée!")
        
        # Afficher un résumé final
        print("\n📊 Résumé de la base de données:")
        print(f"   - Conversations: {db.chat_conversations.count_documents({})}")
        print(f"   - Messages: {db.chat_messages.count_documents({})}")
        print(f"   - Attachments: {db.chat_attachments.count_documents({})}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 70)
    print("  NETTOYAGE DES CONVERSATIONS INVALIDES")
    print("=" * 70)
    print()
    clean_invalid_conversations()
    print()
    print("=" * 70)
