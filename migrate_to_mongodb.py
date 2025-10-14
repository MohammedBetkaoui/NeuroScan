"""
Script de migration de app.py vers app_web.py avec MongoDB
"""

import re

# Lire app.py
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remplacements pour MongoDB
replacements = {
    # Import SQLite -> MongoDB
    "import sqlite3": "# import sqlite3 - Remplacé par MongoDB\nfrom database.mongodb_connector import get_mongodb, get_collection",
    
    # Configuration de la base de données
    "DATABASE_PATH = 'neuroscan_analytics.db'": "# MongoDB utilisé au lieu de SQLite\n# Configuration dans database/mongodb_connector.py",
    
    # Fonction init_database
    "def init_database():\n    \"\"\"Initialiser la base de données avec les tables nécessaires\"\"\"\n    conn = sqlite3.connect(DATABASE_PATH)\n    cursor = conn.cursor()": 
    "def init_database():\n    \"\"\"Initialiser la base de données MongoDB avec les collections nécessaires\"\"\"\n    # MongoDB s'initialise automatiquement via mongodb_connector\n    from database.mongodb_connector import mongodb_connector\n    return mongodb_connector.init_collections()",
}

# Appliquer les remplacements de base
for old, new in replacements.items():
    content = content.replace(old, new)

# Sauvegarder dans app_web.py
with open('app_web.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fichier app_web.py créé avec succès")
print("⚠️  Note: Des ajustements manuels supplémentaires sont nécessaires pour adapter toutes les fonctions à MongoDB")
