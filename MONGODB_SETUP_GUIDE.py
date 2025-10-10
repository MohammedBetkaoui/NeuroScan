"""
Guide de configuration MongoDB Atlas pour NeuroScan AI
"""

print("=" * 70)
print("📚 GUIDE DE CONFIGURATION MONGODB ATLAS")
print("=" * 70)

print("""
🔍 VÉRIFICATION DU CLUSTER MONGODB

Vous avez fourni les informations suivantes :
- Nom du cluster : cluster0.xdhjsc1.mongodb.net
- Nom d'utilisateur : betkaoui_mohammed
- Mot de passe : betkaoui@2002
- IPs autorisées : 105.235.136.57/32 et 0.0.0.0/0

⚠️  PROBLÈME DÉTECTÉ :
Le DNS ne peut pas résoudre 'cluster0.xdhjsc1.mongodb.net'

Cela signifie généralement que :
1. Le cluster n'existe pas ou a été supprimé
2. Le nom du cluster est incorrect
3. Le cluster est en cours de création

📋 ÉTAPES À SUIVRE :

1️⃣  Connectez-vous à MongoDB Atlas
   👉 https://cloud.mongodb.com/

2️⃣  Vérifiez votre cluster
   - Cliquez sur "Database" dans le menu de gauche
   - Vérifiez que vous avez un cluster actif
   - Notez le nom exact du cluster

3️⃣  Obtenez la chaîne de connexion correcte
   - Cliquez sur "Connect" sur votre cluster
   - Choisissez "Connect your application"
   - Sélectionnez "Python" et version "3.12 or later"
   - Copiez la chaîne de connexion (elle ressemble à ceci) :
   
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority

4️⃣  Si vous n'avez pas de cluster :
   - Cliquez sur "+ Create" ou "Build a Database"
   - Choisissez "FREE" (M0 Sandbox)
   - Sélectionnez une région proche (Europe si vous êtes en Afrique du Nord)
   - Nommez votre cluster (ex: Cluster0)
   - Cliquez sur "Create Cluster"
   - Attendez quelques minutes que le cluster soit créé

5️⃣  Créer un utilisateur de base de données :
   - Allez dans "Database Access"
   - Cliquez sur "+ ADD NEW DATABASE USER"
   - Username : betkaoui_mohammed
   - Password : betkaoui@2002 (ou un autre mot de passe sécurisé)
   - Database User Privileges : "Read and write to any database"
   - Cliquez sur "Add User"

6️⃣  Autoriser l'accès réseau :
   - Allez dans "Network Access"
   - Cliquez sur "+ ADD IP ADDRESS"
   - Choisissez "ALLOW ACCESS FROM ANYWHERE" (0.0.0.0/0)
     ⚠️  Pour la production, limitez aux IPs spécifiques
   - Cliquez sur "Confirm"

7️⃣  Mettre à jour le fichier .env
   Une fois que vous avez la bonne chaîne de connexion :
   
   a) Remplacez <password> par votre mot de passe
   b) Si le mot de passe contient des caractères spéciaux, encodez-les :
      @ → %40
      # → %23
      $ → %24
      % → %25
      & → %26
   
   c) Ajoutez le nom de la base de données après mongodb.net/ :
      mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net/NeuroScan?...

8️⃣  Tester la connexion
   Une fois le .env mis à jour, exécutez :
   python3 test_mongodb_connection.py

═════════════════════════════════════════════════════════════════════

🆘 BESOIN D'AIDE ?

Option 1: Créer un nouveau cluster (RECOMMANDÉ)
- Suivez les étapes ci-dessus
- C'est gratuit (M0 tier)
- Prend 3-5 minutes

Option 2: Utiliser SQLite (temporaire)
- Si vous voulez tester l'application immédiatement
- Exécutez : python3 app.py (version SQLite)
- Note : SQLite est local, MongoDB est pour le web

Option 3: Vérifier le cluster existant
- Si vous pensez que le cluster existe déjà
- Connectez-vous à MongoDB Atlas
- Copiez la chaîne de connexion exacte
- Mettez à jour le fichier .env

═════════════════════════════════════════════════════════════════════

💡 CONSEIL :
Le nom 'cluster0.xdhjsc1.mongodb.net' semble être un ancien cluster
ou un cluster de test. Créez un nouveau cluster pour être sûr.

""")

# Proposer de créer un template .env
print("\n❓ Voulez-vous que je crée un template .env pour vous ?")
print("   Après avoir obtenu votre chaîne de connexion MongoDB Atlas,")
print("   vous pourrez facilement la mettre à jour.")

template = """
# NeuroScan Environment Variables Template
# Après avoir créé votre cluster MongoDB Atlas, mettez à jour cette configuration

# Flask Configuration
SECRET_KEY=neuroscan-secret-key-change-in-production
FLASK_ENV=development
FLASK_DEBUG=1

# Google Gemini API
GEMINI_API_KEY=AIzaSyD9H1Odcbk1Zo8KuvzBvqhvkAx0wJhqBS8

# Database - MongoDB Atlas (Cloud)
# FORMAT : mongodb+srv://<username>:<password_encoded>@<cluster>.mongodb.net/<database>?retryWrites=true&w=majority
# IMPORTANT : Encodez les caractères spéciaux du mot de passe (@ → %40, # → %23, etc.)
MONGODB_URI=mongodb+srv://betkaoui_mohammed:VOTRE_MOT_DE_PASSE_ENCODE@VOTRE_CLUSTER.mongodb.net/NeuroScan?retryWrites=true&w=majority&appName=Cluster0
MONGODB_DB_NAME=NeuroScan

# Upload Settings
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=uploads

# EXEMPLE avec mot de passe "betkaoui@2002" :
# @ devient %40, donc : betkaoui%402002
# MONGODB_URI=mongodb+srv://betkaoui_mohammed:betkaoui%402002@cluster0.xxxxx.mongodb.net/NeuroScan?retryWrites=true&w=majority
"""

print("\n📄 Template .env :")
print(template)

print("\n" + "=" * 70)
print("✅ Suivez les étapes ci-dessus et revenez quand c'est prêt !")
print("=" * 70)
