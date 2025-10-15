#!/bin/bash

echo "=== Installation de PyTorch pour NeuroScan ==="
echo ""

# Activer l'environnement virtuel
if [ -d "venv" ]; then
    echo "Activation de l'environnement virtuel..."
    source venv/bin/activate
else
    echo "Erreur: Environnement virtuel non trouvé. Exécutez d'abord start_demo.sh"
    exit 1
fi

echo "Installation de PyTorch (version CPU)..."
echo "Cela peut prendre plusieurs minutes..."
echo ""

# Installer PyTorch CPU uniquement (plus léger)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Installer opencv-python
pip install opencv-python

echo ""
echo "=== Installation terminée ==="
echo "Vous pouvez maintenant utiliser app.py avec le vrai modèle:"
echo "  source venv/bin/activate"
echo "  python3 app.py"
echo ""
