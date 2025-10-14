#!/bin/bash
# Script d'installation de NeuroScan AI dans le système

echo "============================================="
echo "🧠 Installation de NeuroScan AI"
echo "============================================="
echo ""

# Vérifier si on est root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Ce script nécessite les privilèges root"
    echo "💡 Relancez avec: sudo ./install_system.sh"
    exit 1
fi

# Vérifier les dépendances système
echo "📦 Vérification des dépendances système..."
apt-get update -qq
apt-get install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1

# Créer le répertoire d'installation
INSTALL_DIR="/opt/neuroscan-ai"
echo "📂 Création du répertoire d'installation: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Copier les fichiers
echo "📋 Copie des fichiers..."
cp -r dist/NeuroScan_AI "$INSTALL_DIR/"
cp -r static "$INSTALL_DIR/"
cp neuroscan_analytics.db "$INSTALL_DIR/" 2>/dev/null || true

# Définir les permissions
echo "🔐 Configuration des permissions..."
chmod +x "$INSTALL_DIR/NeuroScan_AI"
chown -R $SUDO_USER:$SUDO_USER "$INSTALL_DIR"

# Créer un lien symbolique
echo "🔗 Création du lien symbolique..."
ln -sf "$INSTALL_DIR/NeuroScan_AI" /usr/local/bin/neuroscan-ai

# Installer le fichier .desktop
echo "🖥️  Installation du raccourci dans le menu..."
DESKTOP_FILE="/usr/share/applications/neuroscan-ai.desktop"

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=NeuroScan AI
Comment=Application de diagnostic médical par IA pour tumeurs cérébrales
Exec=$INSTALL_DIR/NeuroScan_AI
Icon=$INSTALL_DIR/static/images/logo.png
Terminal=false
Categories=Medical;Science;Education;
Keywords=medical;AI;brain;tumor;diagnosis;
StartupNotify=true
EOF

chmod 644 "$DESKTOP_FILE"

# Mise à jour du cache des applications
echo "🔄 Mise à jour du cache des applications..."
update-desktop-database /usr/share/applications 2>/dev/null || true

echo ""
echo "============================================="
echo "✅ Installation terminée avec succès!"
echo "============================================="
echo ""
echo "🚀 Vous pouvez maintenant lancer NeuroScan AI de 3 façons:"
echo ""
echo "   1. Depuis le menu des applications"
echo "   2. En tapant: neuroscan-ai"
echo "   3. Directement: $INSTALL_DIR/NeuroScan_AI"
echo ""
echo "============================================="
