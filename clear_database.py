#!/usr/bin/env python3
"""
Script pour vider et réinitialiser la base de données NeuroScan
"""

import sqlite3
import os

DATABASE_PATH = 'neuroscan_analytics.db'

def clear_database():
    """Vider complètement la base de données"""
    try:
        if os.path.exists(DATABASE_PATH):
            print(f"🗑️  Suppression de la base de données existante: {DATABASE_PATH}")
            os.remove(DATABASE_PATH)
            print("✅ Base de données supprimée avec succès")
        else:
            print("ℹ️  Aucune base de données existante trouvée")
        
        # Réinitialiser la base de données
        print("🔄 Réinitialisation de la base de données...")
        init_database()
        print("✅ Base de données réinitialisée avec succès")
        
    except Exception as e:
        print(f"❌ Erreur lors de la suppression: {e}")

def init_database():
    """Initialiser la base de données avec les tables nécessaires"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    print("📋 Création des tables...")
    
    # Table des médecins (authentification)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            specialty TEXT,
            hospital TEXT,
            license_number TEXT,
            phone TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME,
            login_count INTEGER DEFAULT 0
        )
    ''')
    print("  ✅ Table 'doctors' créée")
    
    # Table des sessions de médecins
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctor_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER,
            session_token TEXT UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME,
            is_active BOOLEAN DEFAULT 1,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY (doctor_id) REFERENCES doctors (id)
        )
    ''')
    print("  ✅ Table 'doctor_sessions' créée")
    
    # Table des patients
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            patient_name TEXT,
            date_of_birth DATE,
            gender TEXT,
            first_analysis_date DATE,
            last_analysis_date DATE,
            total_analyses INTEGER DEFAULT 0,
            doctor_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(patient_id, doctor_id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(id)
        )
    ''')
    print("  ✅ Table 'patients' créée")
    
    # Table des analyses
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            patient_id TEXT,
            patient_name TEXT,
            exam_date DATE,
            predicted_class INTEGER,
            predicted_label TEXT,
            confidence REAL,
            probabilities TEXT,
            description TEXT,
            recommendations TEXT,
            processing_time REAL,
            user_session TEXT,
            ip_address TEXT,
            tumor_size_estimate REAL,
            previous_analysis_id INTEGER,
            doctor_id INTEGER NOT NULL,
            FOREIGN KEY (previous_analysis_id) REFERENCES analyses(id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(id)
        )
    ''')
    print("  ✅ Table 'analyses' créée")
    
    # Table des statistiques quotidiennes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE,
            total_analyses INTEGER DEFAULT 0,
            normal_count INTEGER DEFAULT 0,
            gliome_count INTEGER DEFAULT 0,
            meningiome_count INTEGER DEFAULT 0,
            pituitary_count INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            avg_processing_time REAL DEFAULT 0
        )
    ''')
    print("  ✅ Table 'daily_stats' créée")
    
    # Table des sessions utilisateur
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            analyses_count INTEGER DEFAULT 0,
            ip_address TEXT,
            user_agent TEXT
        )
    ''')
    print("  ✅ Table 'user_sessions' créée")
    
    # Table des évolutions tumorales
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tumor_evolution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            analysis_id INTEGER,
            exam_date DATE,
            diagnosis_change TEXT,
            confidence_change REAL,
            size_change REAL,
            evolution_type TEXT,
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id),
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')
    print("  ✅ Table 'tumor_evolution' créée")
    
    # Table des alertes médicales
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            doctor_id INTEGER NOT NULL,
            analysis_id INTEGER,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            is_read BOOLEAN DEFAULT 0,
            is_resolved BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at DATETIME,
            resolved_by INTEGER,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(id),
            FOREIGN KEY (analysis_id) REFERENCES analyses(id),
            FOREIGN KEY (resolved_by) REFERENCES doctors(id)
        )
    ''')
    print("  ✅ Table 'medical_alerts' créée")
    
    # Table des notifications push
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            is_read BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doctor_id) REFERENCES doctors(id)
        )
    ''')
    print("  ✅ Table 'notifications' créée")
    
    conn.commit()
    conn.close()

def show_database_stats():
    """Afficher les statistiques de la base de données"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Compter les analyses
        cursor.execute('SELECT COUNT(*) FROM analyses')
        analyses_count = cursor.fetchone()[0]
        
        # Compter les stats quotidiennes
        cursor.execute('SELECT COUNT(*) FROM daily_stats')
        daily_stats_count = cursor.fetchone()[0]
        
        # Compter les sessions
        cursor.execute('SELECT COUNT(*) FROM user_sessions')
        sessions_count = cursor.fetchone()[0]
        
        # Compter les médecins
        cursor.execute('SELECT COUNT(*) FROM doctors')
        doctors_count = cursor.fetchone()[0]
        
        # Compter les patients
        cursor.execute('SELECT COUNT(*) FROM patients')
        patients_count = cursor.fetchone()[0]
        
        # Compter les alertes médicales
        cursor.execute('SELECT COUNT(*) FROM medical_alerts')
        alerts_count = cursor.fetchone()[0]
        
        print("\n📊 Statistiques de la base de données:")
        print(f"  📈 Analyses: {analyses_count}")
        print(f"  👥 Patients: {patients_count}")
        print(f"  👨‍⚕️ Médecins: {doctors_count}")
        print(f"  🚨 Alertes médicales: {alerts_count}")
        print(f"  📅 Statistiques quotidiennes: {daily_stats_count}")
        print(f"  👥 Sessions utilisateur: {sessions_count}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Erreur lors de la lecture des statistiques: {e}")

def clear_specific_table(table_name):
    """Vider une table spécifique"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Vérifier si la table existe
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        ''', (table_name,))
        
        if cursor.fetchone():
            cursor.execute(f'DELETE FROM {table_name}')
            cursor.execute(f'DELETE FROM sqlite_sequence WHERE name=?', (table_name,))
            conn.commit()
            print(f"✅ Table '{table_name}' vidée avec succès")
        else:
            print(f"❌ Table '{table_name}' non trouvée")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Erreur lors du vidage de la table '{table_name}': {e}")

if __name__ == "__main__":
    import sys
    
    print("🧠 NeuroScan - Gestion de la base de données")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "clear":
            print("🗑️  Vidage complet de la base de données...")
            clear_database()
            
        elif command == "stats":
            print("📊 Affichage des statistiques...")
            show_database_stats()
            
        elif command == "clear-analyses":
            print("🗑️  Vidage de la table 'analyses'...")
            clear_specific_table('analyses')
            
        elif command == "clear-stats":
            print("🗑️  Vidage de la table 'daily_stats'...")
            clear_specific_table('daily_stats')
            
        elif command == "clear-sessions":
            print("🗑️  Vidage de la table 'user_sessions'...")
            clear_specific_table('user_sessions')
            
        else:
            print(f"❌ Commande inconnue: {command}")
            print("\nCommandes disponibles:")
            print("  clear           - Vider complètement la base de données")
            print("  stats           - Afficher les statistiques")
            print("  clear-analyses  - Vider seulement la table des analyses")
            print("  clear-stats     - Vider seulement les statistiques quotidiennes")
            print("  clear-sessions  - Vider seulement les sessions utilisateur")
    else:
        # Mode interactif
        print("Mode interactif - Choisissez une option:")
        print("1. Vider complètement la base de données")
        print("2. Afficher les statistiques")
        print("3. Vider seulement les analyses")
        print("4. Vider seulement les statistiques quotidiennes")
        print("5. Vider seulement les sessions utilisateur")
        print("6. Quitter")
        
        while True:
            try:
                choice = input("\nVotre choix (1-6): ").strip()
                
                if choice == "1":
                    confirm = input("⚠️  Êtes-vous sûr de vouloir vider complètement la base de données? (oui/non): ")
                    if confirm.lower() in ['oui', 'o', 'yes', 'y']:
                        clear_database()
                    else:
                        print("❌ Opération annulée")
                        
                elif choice == "2":
                    show_database_stats()
                    
                elif choice == "3":
                    confirm = input("⚠️  Vider toutes les analyses? (oui/non): ")
                    if confirm.lower() in ['oui', 'o', 'yes', 'y']:
                        clear_specific_table('analyses')
                    else:
                        print("❌ Opération annulée")
                        
                elif choice == "4":
                    clear_specific_table('daily_stats')
                    
                elif choice == "5":
                    clear_specific_table('user_sessions')
                    
                elif choice == "6":
                    print("👋 Au revoir!")
                    break
                    
                else:
                    print("❌ Choix invalide. Veuillez choisir entre 1 et 6.")
                    
            except KeyboardInterrupt:
                print("\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
    
    print("\n🎉 Opération terminée!")
