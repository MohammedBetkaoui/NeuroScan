#!/bin/bash
# Script de vérification de l'installation NeuroScan AI

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  🔍 Vérification de l'installation NeuroScan AI             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Compteur de succès
SUCCESS=0
TOTAL=0

# Fonction de vérification
check() {
    TOTAL=$((TOTAL + 1))
    if [ $1 -eq 0 ]; then
        echo "✅ $2"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "❌ $2"
    fi
}

# Vérifier les fichiers principaux
echo "📁 Vérification des fichiers..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ -f "run_app.py" ]; check $? "run_app.py existe"
[ -f "build_app.sh" ] && [ -x "build_app.sh" ]; check $? "build_app.sh existe et est exécutable"
[ -f "launch_neuroscan.sh" ] && [ -x "launch_neuroscan.sh" ]; check $? "launch_neuroscan.sh existe et est exécutable"
[ -f "install_system.sh" ] && [ -x "install_system.sh" ]; check $? "install_system.sh existe et est exécutable"
[ -f "uninstall_system.sh" ] && [ -x "uninstall_system.sh" ]; check $? "uninstall_system.sh existe et est exécutable"
[ -f "neuroscan-ai.desktop" ]; check $? "neuroscan-ai.desktop existe"
[ -f "DISTRIBUTION.md" ]; check $? "DISTRIBUTION.md existe"
[ -f "TRANSFORMATION_COMPLETE.md" ]; check $? "TRANSFORMATION_COMPLETE.md existe"

echo ""
echo "📦 Vérification de l'exécutable..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ -f "dist/NeuroScan_AI" ]; check $? "Exécutable dist/NeuroScan_AI existe"
[ -x "dist/NeuroScan_AI" ]; check $? "Exécutable est exécutable"

if [ -f "dist/NeuroScan_AI" ]; then
    SIZE=$(du -h dist/NeuroScan_AI | cut -f1)
    echo "   📊 Taille: $SIZE"
fi

echo ""
echo "🔧 Vérification des dépendances système..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

dpkg -l | grep -q "python3-gi"; check $? "python3-gi installé"
dpkg -l | grep -q "gir1.2-gtk-3.0"; check $? "gir1.2-gtk-3.0 installé"
dpkg -l | grep -q "gir1.2-webkit2-4"; check $? "gir1.2-webkit2-4.x installé"

echo ""
echo "🐍 Vérification de l'environnement virtuel..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ -d "venv" ]; check $? "Environnement virtuel existe"
[ -f "venv/bin/activate" ]; check $? "Script d'activation existe"

if [ -f "venv/bin/python" ]; then
    PYTHON_VERSION=$(venv/bin/python --version 2>&1 | cut -d' ' -f2)
    echo "   🐍 Version Python: $PYTHON_VERSION"
fi

echo ""
echo "📚 Vérification des fichiers du projet..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ -f "app.py" ]; check $? "app.py existe"
[ -f "best_brain_tumor_model.pth" ]; check $? "Modèle PyTorch existe"
[ -f "neuroscan_analytics.db" ]; check $? "Base de données existe"
[ -d "templates" ]; check $? "Dossier templates/ existe"
[ -d "static" ]; check $? "Dossier static/ existe"
[ -f ".env" ]; check $? "Fichier .env existe"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📊 RÉSULTAT: $SUCCESS/$TOTAL vérifications réussies"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ $SUCCESS -eq $TOTAL ]; then
    echo "✨ Tout est en ordre ! Vous pouvez lancer l'application avec:"
    echo "   ./launch_neuroscan.sh"
    echo ""
    exit 0
else
    FAILED=$((TOTAL - SUCCESS))
    echo "⚠️  $FAILED vérification(s) échouée(s)"
    echo ""
    echo "💡 Actions recommandées:"
    
    if ! dpkg -l | grep -q "gir1.2-webkit2-4"; then
        echo "   • Installer les dépendances système:"
        echo "     sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1"
    fi
    
    if [ ! -f "dist/NeuroScan_AI" ]; then
        echo "   • Compiler l'application:"
        echo "     ./build_app.sh"
    fi
    
    echo ""
    exit 1
fi
