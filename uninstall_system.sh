#!/bin/bash
# Script de désinstallation de NeuroScan AI du système

echo "============================================="
echo "🧠 Désinstallation de NeuroScan AI"
echo "============================================="
echo ""

# Vérifier si on est root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Ce script nécessite les privilèges root"
    echo "💡 Relancez avec: sudo ./uninstall_system.sh"
    exit 1
fi

INSTALL_DIR="/opt/neuroscan-ai"
DESKTOP_FILE="/usr/share/applications/neuroscan-ai.desktop"

# Supprimer le lien symbolique
echo "🗑️  Suppression du lien symbolique..."
rm -f /usr/local/bin/neuroscan-ai

# Supprimer le fichier .desktop
echo "🗑️  Suppression du raccourci menu..."
rm -f "$DESKTOP_FILE"

# Supprimer le répertoire d'installation
echo "🗑️  Suppression des fichiers d'installation..."
rm -rf "$INSTALL_DIR"

# Mise à jour du cache des applications
echo "🔄 Mise à jour du cache des applications..."
update-desktop-database /usr/share/applications 2>/dev/null || true

echo ""
echo "============================================="
echo "✅ Désinstallation terminée!"
echo "============================================="
echo ""
echo "ℹ️  NeuroScan AI a été supprimé du système"
echo ""
