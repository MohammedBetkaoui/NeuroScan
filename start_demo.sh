#!/bin/bash

echo "=== NeuroScan - Démarrage de l'application démo ==="
echo ""

# Vérifier si l'environnement virtuel existe
if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo "Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances si nécessaire
echo "Vérification des dépendances..."
pip install Flask Pillow numpy Werkzeug > /dev/null 2>&1

echo ""
echo "=== Démarrage de l'application ==="
echo "L'application sera accessible à l'adresse: http://localhost:5000"
echo "Appuyez sur Ctrl+C pour arrêter l'application"
echo ""

# Démarrer l'application
python3 app_demo.py
