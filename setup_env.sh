#!/bin/bash

# Script de configuration de l'environnement NeuroScan
# Ce script aide à créer et configurer le fichier .env

echo "================================================"
echo "  Configuration de l'environnement NeuroScan"
echo "================================================"
echo ""

# Vérifier si .env existe déjà
if [ -f ".env" ]; then
    echo "⚠️  Le fichier .env existe déjà."
    read -p "Voulez-vous le recréer ? (o/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Oo]$ ]]; then
        echo "Configuration annulée."
        exit 0
    fi
fi

# Copier le template
cp .env.example .env
echo "✅ Fichier .env créé depuis .env.example"
echo ""

# Générer une clé secrète aléatoire pour Flask
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/changez-cette-cle-secrete-en-production/$SECRET_KEY/" .env
else
    # Linux
    sed -i "s/changez-cette-cle-secrete-en-production/$SECRET_KEY/" .env
fi
echo "✅ Clé secrète Flask générée automatiquement"
echo ""

# Demander la clé API Gemini
echo "📝 Configuration de l'API Gemini"
echo "   Pour obtenir une clé API gratuite:"
echo "   👉 https://makersuite.google.com/app/apikey"
echo ""
read -p "Entrez votre clé API Gemini (ou appuyez sur Entrée pour ignorer): " GEMINI_KEY

if [ ! -z "$GEMINI_KEY" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/your_gemini_api_key_here/$GEMINI_KEY/" .env
    else
        sed -i "s/your_gemini_api_key_here/$GEMINI_KEY/" .env
    fi
    echo "✅ Clé API Gemini configurée"
else
    echo "⚠️  Clé API Gemini non configurée. Le chatbot ne fonctionnera pas."
    echo "   Vous pourrez l'ajouter plus tard dans le fichier .env"
fi

echo ""
echo "================================================"
echo "✅ Configuration terminée !"
echo "================================================"
echo ""
echo "Le fichier .env a été créé avec:"
echo "  - SECRET_KEY (générée automatiquement)"
echo "  - GEMINI_API_KEY (${GEMINI_KEY:+configurée}${GEMINI_KEY:-non configurée})"
echo ""
echo "Vous pouvez modifier .env pour ajuster d'autres paramètres."
echo ""
echo "Pour démarrer l'application:"
echo "  ./start_demo.sh    # Mode démo (sans PyTorch)"
echo "  python3 app.py     # Mode complet (avec IA)"
echo ""
