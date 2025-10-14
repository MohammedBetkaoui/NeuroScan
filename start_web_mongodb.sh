#!/bin/bash

# Script de démarrage pour NeuroScan AI avec MongoDB
# Usage: ./start_web_mongodb.sh

echo "🚀 Démarrage de NeuroScan AI avec MongoDB..."
echo "=============================================="
echo ""

# Vérifier que le venv existe
if [ ! -d "venv" ]; then
    echo "❌ Environnement virtuel non trouvé!"
    echo "   Créez-le avec: python3 -m venv venv"
    exit 1
fi

# Activer l'environnement virtuel
source venv/bin/activate

# Vérifier que les dépendances MongoDB sont installées
echo "📦 Vérification des dépendances..."
python -c "import pymongo" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installation de pymongo..."
    pip install pymongo dnspython -q
fi

python -c "import dnspython" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installation de dnspython..."
    pip install dnspython -q
fi

echo "✅ Dépendances OK"
echo ""

# Vérifier le fichier .env
if [ ! -f ".env" ]; then
    echo "❌ Fichier .env non trouvé!"
    echo "   Copiez .env.example vers .env et configurez vos variables"
    exit 1
fi

echo "✅ Fichier .env trouvé"
echo ""

# Tester la connexion MongoDB (optionnel)
echo "🔌 Test de connexion MongoDB..."
python test_mongodb_configuration.py 2>&1 | head -n 20
MONGO_STATUS=$?

if [ $MONGO_STATUS -ne 0 ]; then
    echo ""
    echo "⚠️  ATTENTION: La connexion MongoDB a échoué!"
    echo "   L'application démarrera mais certaines fonctionnalités ne fonctionneront pas."
    echo "   Consultez MONGODB_MIGRATION_COMPLETE.md pour la configuration."
    echo ""
    read -p "Continuer quand même ? (o/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[OoYy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🌐 Démarrage de l'application web..."
echo "   URL: http://localhost:5000"
echo "   Pour arrêter: Ctrl+C"
echo ""
echo "=============================================="
echo ""

# Démarrer l'application
python app_web.py
