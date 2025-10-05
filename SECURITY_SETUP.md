# 🔐 Configuration de sécurité - NeuroScan

## ✅ Ce qui a été fait

### 1. Sécurisation de la clé API Gemini
- ✅ Clé API déplacée du code source vers un fichier `.env` séparé
- ✅ Fichier `.env` ajouté au `.gitignore` (ne sera jamais commité sur Git)
- ✅ Chargement automatique avec `python-dotenv`

### 2. Fichiers créés
- ✅ `.env` - Contient toutes les variables d'environnement sensibles
- ✅ `.env.example` - Template pour les nouveaux utilisateurs
- ✅ `setup_env.sh` - Script automatique de configuration
- ✅ `uploads/.gitkeep` - Pour garder le dossier uploads dans Git

### 3. Modifications du code
- ✅ `app.py` - Chargement des variables depuis `.env` avec dotenv
- ✅ Protection contre clé manquante (affiche un warning)
- ✅ README.md mis à jour avec documentation complète

## 🔒 Sécurité

### Fichiers sensibles protégés
```
.env                    # Variables d'environnement (dans .gitignore)
*.db                    # Base de données (dans .gitignore)
uploads/*               # Images uploadées (dans .gitignore)
```

### Bonnes pratiques appliquées
✅ Clés API dans `.env` et non dans le code
✅ `.env` dans `.gitignore` pour ne jamais être commité
✅ `.env.example` pour documenter les variables nécessaires
✅ Script `setup_env.sh` pour faciliter la configuration
✅ Vérification de présence de la clé au démarrage

## 📝 Utilisation

### Configuration initiale (nouvelle installation)
```bash
# Méthode 1 : Script automatique
./setup_env.sh

# Méthode 2 : Manuel
cp .env.example .env
nano .env  # Éditer et ajouter votre clé API
```

### Vérification
```bash
# Vérifier que .env est bien chargé
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); \
print('✅ OK' if os.getenv('GEMINI_API_KEY') else '❌ Clé manquante')"
```

### Démarrage de l'application
```bash
# L'application charge automatiquement .env au démarrage
source venv/bin/activate
python3 app.py
```

## ⚠️ Important

### À NE JAMAIS FAIRE
❌ Commiter le fichier `.env` sur Git
❌ Partager votre clé API publiquement
❌ Mettre des clés API directement dans le code
❌ Supprimer `.env` du `.gitignore`

### À FAIRE
✅ Garder `.env` local sur votre machine
✅ Utiliser `.env.example` comme documentation
✅ Régénérer la clé API si compromise
✅ Utiliser des clés différentes pour dev/prod

## 🧪 Tests effectués

### Test 1 : Chargement de .env
```
✅ Clé API Gemini chargée avec succès
Clé: AIzaSyBC3sAJjh9_32jT...M7HzyNJPng
```

### Test 2 : Appel API Gemini
```
✅ Test API Gemini réussi!
Réponse du chatbot: Bonjour, je suis le chatbot NeuroScan !
```

## 📚 Documentation

- README.md : Section "Configuration de l'API Gemini" mise à jour
- .env.example : Template documenté
- setup_env.sh : Script de configuration automatique

## 🎯 Résultat

Votre clé API Gemini est maintenant :
- 🔒 Sécurisée (hors du code source)
- 📝 Documentée (.env.example)
- 🚫 Protégée (dans .gitignore)
- ✅ Fonctionnelle (tests réussis)
- 🔄 Facilement configurable (script setup_env.sh)

Le chatbot NeuroScan est prêt à fonctionner ! 🚀
