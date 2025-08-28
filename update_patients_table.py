#!/usr/bin/env python3
"""
Script pour mettre à jour la table patients avec des champs supplémentaires
"""

import sqlite3
from datetime import datetime

def update_patients_table():
    """Ajouter les nouveaux champs à la table patients"""
    try:
        conn = sqlite3.connect('neuroscan_analytics.db')
        cursor = conn.cursor()
        
        # Liste des nouveaux champs à ajouter
        new_fields = [
            ('phone', 'TEXT'),
            ('email', 'TEXT'),
            ('address', 'TEXT'),
            ('emergency_contact_name', 'TEXT'),
            ('emergency_contact_phone', 'TEXT'),
            ('medical_history', 'TEXT'),
            ('allergies', 'TEXT'),
            ('current_medications', 'TEXT'),
            ('insurance_number', 'TEXT'),
            ('notes', 'TEXT'),
            ('updated_at', 'DATETIME')
        ]
        
        # Vérifier quels champs existent déjà
        cursor.execute("PRAGMA table_info(patients)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        print("Colonnes existantes dans la table patients:")
        for col in existing_columns:
            print(f"  - {col}")
        
        # Ajouter les nouveaux champs
        added_fields = []
        for field_name, field_type in new_fields:
            if field_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE patients ADD COLUMN {field_name} {field_type}")
                    added_fields.append(field_name)
                    print(f"✅ Ajouté: {field_name} ({field_type})")
                except sqlite3.Error as e:
                    print(f"❌ Erreur pour {field_name}: {e}")
        
        if added_fields:
            conn.commit()
            print(f"\n✅ {len(added_fields)} nouveaux champs ajoutés avec succès!")
        else:
            print("\n✅ Tous les champs existent déjà.")
        
        # Vérifier la structure finale
        cursor.execute("PRAGMA table_info(patients)")
        final_columns = [row[1] for row in cursor.fetchall()]
        
        print(f"\nStructure finale de la table patients ({len(final_columns)} colonnes):")
        for col in final_columns:
            print(f"  - {col}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour: {e}")
        return False

if __name__ == '__main__':
    print("🔄 Mise à jour de la table patients...")
    print("=" * 50)
    
    success = update_patients_table()
    
    if success:
        print("\n🎉 Mise à jour terminée avec succès!")
    else:
        print("\n⚠️ Erreur lors de la mise à jour.")
