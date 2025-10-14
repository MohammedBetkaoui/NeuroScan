#!/bin/bash
# Script de compilation de NeuroScan AI en application desktop Linux

echo "============================================="
echo "🧠 Compilation de NeuroScan AI"
echo "============================================="

# Activer l'environnement virtuel
source venv/bin/activate

echo "📦 Création de l'exécutable avec PyInstaller..."
echo ""

# Compiler avec PyInstaller
pyinstaller --clean --noconfirm \
  --name="NeuroScan_AI" \
  --onefile \
  --windowed \
  --add-data "templates:templates" \
  --add-data "static:static" \
  --add-data "best_brain_tumor_model.pth:." \
  --add-data "neuroscan_analytics.db:." \
  --add-data ".env:." \
  --hidden-import="webview" \
  --hidden-import="PIL._tkinter_finder" \
  --hidden-import="pkg_resources.py2_warn" \
  --collect-all="torch" \
  --collect-all="torchvision" \
  --collect-all="PIL" \
  --collect-all="reportlab" \
  --collect-all="flask" \
  --collect-all="webview" \
  --collect-all="google.generativeai" \
  run_app.py

echo ""
echo "============================================="
echo "✅ Compilation terminée!"
echo "============================================="
echo ""
echo "📂 Fichier exécutable créé dans: dist/NeuroScan_AI"
echo "🚀 Pour lancer l'application:"
echo "   ./dist/NeuroScan_AI"
echo ""
echo "============================================="
