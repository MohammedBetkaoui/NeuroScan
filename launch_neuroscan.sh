#!/bin/bash
# Script de lancement de NeuroScan AI Desktop

echo "============================================="
echo "🧠 Lancement de NeuroScan AI"
echo "============================================="
echo ""

# Se placer dans le répertoire du script
cd "$(dirname "$0")"

# Vérifier si l'environnement virtuel existe
if [ ! -d "venv" ]; then
    echo "❌ Erreur: Environnement virtuel introuvable!"
    echo "💡 Créez d'abord l'environnement avec:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

echo "🚀 Démarrage de l'application avec Python..."
echo ""

# Activer l'environnement virtuel et lancer
source venv/bin/activate
python3 run_app.py

echo ""
echo "============================================="
echo "👋 Application fermée"
echo "============================================="
